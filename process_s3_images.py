import face_recognition
import numpy as np
import os
import sqlite3
import cv2
import uuid
import hashlib
import sys
from bs4 import BeautifulSoup as bs
sys.path.append('./final')
import requests
from const import DB_FILE, CUTOUTS_DIR
from face_recognizer_3 import FaceRecognizer
from datetime import datetime
import pandas as pd
# import pdb
import time
from db import setup_database
import boto3
import pytz
from final.s3_client import get_s3_client

india_tz = pytz.timezone('Asia/Kolkata')

def sort_all_s3_objects_datewise(all_s3_filtered):
    # returns: Dict<date,List<objs>>
    datewise_objs = {}
    for obj in all_s3_filtered:
        obj_datetime = obj['LastModified'].astimezone(india_tz)

        obj_date = obj_datetime.date()
        
        if(obj_date in datewise_objs.keys()):
            ############ WARNING: NEED TO REMOVE FOR FINAL CORRECT REPORT - THIS IS ONLY FOR TESTING (process first 20 images per day only)
            if(len(datewise_objs[obj_date])<20):
                datewise_objs[obj_date].append(obj)
        else:
            datewise_objs[obj_date] = [obj]
    
    return datewise_objs

def process_s3_objects_date(s3client , edate, all_s3_for_date, req_bucket_name):
    # we have a day of s3 objects here
    # import pdb; pdb.set_trace()
    s3 = s3client
    year, month, day = edate.year, edate.month, edate.day
    eventname = edate.strftime('%Y%m%d')

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    print(f"Processing event: {eventname}   numimages:{len(all_s3_for_date)}")
    face_recog_ = FaceRecognizer()

    for s3obj in all_s3_for_date:
        print(f"Loading image {s3obj['Key']}...")
        
        response = s3.get_object(Bucket=req_bucket_name, Key=s3obj['Key'])
        image_content = response['Body'].read()  # like response.content from requests

        # Need to give it to face recognizer directly
        faces = face_recog_.identify_faces(image_content)

        print(f"Found {len(faces)} faces in {eventname}:{s3obj['Key']}")
        
        for face in faces:
            clust_id = face["clust_id"]
            face_id = face["face_id"]
            location = face["location"]
            c.execute("INSERT INTO faces (event_date, event, image_path, location, face_id, cluster_id) VALUES (?, ?, ?, ?, ?, ?)",
                (edate.strftime('%Y-%m-%d'), eventname, s3obj['Key'], str(location), face_id, clust_id))
            
            # here we also need to replace the cluster id of all the face_id in faces.db with the return value
            c.execute(f"UPDATE faces SET cluster_id = {clust_id} WHERE face_id = {face_id};")

        print(f"🟢  Processed {len(faces)} faces in {eventname}:{s3obj['Key']}")

    conn.commit()
    conn.close()


def get_already_processed_images():
    # we have a day of s3 objects here
    # import pdb; pdb.set_trace()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute(f"""
        SELECT DISTINCT image_path FROM faces;
        """)
   
    data = c.fetchall()
    data = pd.DataFrame(data)

    # import pdb; pdb.set_trace()
    
    conn.close()
    if(len(data.index)==0):
        return []
        
    return data[0].tolist();


def get_all_images_in_s3(s3client, req_bucket_name, req_prefix=''):
    s3 = s3client

    response = s3.list_objects_v2(Bucket=req_bucket_name)
    paginator = s3.get_paginator('list_objects_v2')
    all_objs= []
    for page in paginator.paginate(Bucket=req_bucket_name, Prefix=req_prefix):
        if 'Contents' in page:
            # for all the objs in page['Content'] filter the full sized ones only

            for obj in page['Contents']:
                # we do not want the "_xyz" files (they are smaller in size)
                if( len( obj['Key'].split('/')[-1].split('_')) ==1 ):
                    all_objs.append(obj)
            # for obj in page['Contents']:
                # print(obj['Key'])
    # import pdb; pdb.set_trace()
    return all_objs


def main():
    tday = datetime.today()
    
    enter_year_of_events = input(f"Enter year of the events: (default: {tday.strftime('%Y')})")
    if(not enter_year_of_events):
        enter_year_of_events = tday.strftime('%Y')

    enter_month_of_events = input(f"Enter month of the events: (default: {tday.strftime('%m')})")
    if(not enter_month_of_events):
        enter_month_of_events = tday.strftime('%m')

    enter_day_of_events = input(f"Enter day of events since when you want events to be processed: (default: {tday.strftime('%d')})")
    if(not enter_day_of_events):
        enter_day_of_events = tday.strftime('%d')
    
    since_date = pd.to_datetime(enter_year_of_events+'-'+enter_month_of_events+'-'+enter_day_of_events, format='%Y-%m-%d')
    print(f"Will consider images since {since_date.strftime('%Y-%m-%d')} of the month/year {enter_month_of_events}/{enter_year_of_events}")
    prefix = f'file/pic/photo/{enter_year_of_events}/{enter_month_of_events}/'
    bucket_name = 'soulbook-replica'
    s3 = get_s3_client()
    all_s3_objects_filtered = get_all_images_in_s3(s3, bucket_name, req_prefix=prefix)
    datewise_s3_objs = sort_all_s3_objects_datewise(all_s3_objects_filtered)
    
    to_be_deleted_keys = []
    for key in datewise_s3_objs.keys():
        if(key < since_date.date()):
            print(f"Need to ignore images for the date:{key}")
            # del datewise_s3_objs[key]
            to_be_deleted_keys.append(key)
    
    for key_ in to_be_deleted_keys:
        del datewise_s3_objs[key_]

    print(f"Received {len(all_s3_objects_filtered)} all_s3_objects_filtered and numdates:{len(datewise_s3_objs.keys())}")

    # We don't want to repeat process images, so can filter based on dates and images processed
    # remove images that are already processed
    setup_database()

    already_processed_s3keys = get_already_processed_images()

    for key in datewise_s3_objs.keys():
        print(f"Orig length of images in date:{key}    : {len(datewise_s3_objs[key])}")
        to_be_removed_imgk = []
        for imgk in datewise_s3_objs[key]:
            # import pdb; pdb.set_trace()
            if(imgk['Key'] in already_processed_s3keys):
                to_be_removed_imgk.append(imgk)

        for tbrimgk in to_be_removed_imgk:
            datewise_s3_objs[key].remove(tbrimgk)
        print(f"Final length of images in date:{key}    : {len(datewise_s3_objs[key])}")
    
    # import pdb; pdb.set_trace();
    t1 = time.time()
    for edate in datewise_s3_objs.keys():
        print(f"processing edate: {edate}")
        t1a = time.time()

        process_s3_objects_date(s3, edate, datewise_s3_objs[edate], bucket_name)
        t1b = time.time()
        
        print(f"✅ Faces added to database for {edate.strftime('%Y-%m-%d')} time taken:{t1b-t1a}")

        print(f"✅ Daily counts computed for {edate.strftime('%Y-%m-%d')} and stored!")

        end_time = time.time()
        print(f"\n⏱  Elapsed time: {end_time - t1a:.2f} seconds")

    
    t2 = time.time()
    s3.close()
    print(f"COMPLETED: Events processed:{len(datewise_s3_objs.keys())}   Total time taken:{t2-t1}")


if __name__ == "__main__":
    main()


# Accuracy - Mismatches, and spot checking of individual face matches, => the report should be right.
# 1. run for 6 months and generate a report, 
# 2. note down timings of the code (file down + categorisation) and configuration of the machine  -> remove gurudevs face, sort the faces by count, 
# 3. ask someone else to run it. - everyone runs on their machine - 
# 4. Remove known faces - 
# 5. customize dates - 

