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
import pdb
import time
from db import setup_database



def get_images_existing_on_page(html_page, event_name):
    soup = bs(html_page, "html.parser")

    # INFO: Get the div tag with the id - js_album_content
    all_photos_div = soup.find("div", {"id": "js_album_content"})

    if all_photos_div is None:
        raise Exception("Could not find photos, maybe the cookie is invalid")

    # INFO: Get the div inside the all_photos_div with class "col-12 col-sm-12 col-md-12 col-lg-12 col-xl-12"
    all_photos_div = all_photos_div.find(
        "div", {"class": "col-12 col-sm-12 col-md-12 col-lg-12 col-xl-12"}
    )

    if all_photos_div is None:
        print("Could not find photos, maybe the previous page was the last.")
        return []

    return get_image_urls(all_photos_div)



def get_image_urls(all_photos_div):
    imge_urls_store = []

    # INFO: Get all the article tags with the class within all_photos_div with id starting with "js_photo_id_*"
    all_articles = all_photos_div.find_all(
        "article", {"id": lambda x: x and x.startswith("js_photo_id_")}
    )
    if not all_articles:
        raise Exception(
            "Could not find photos, maybe the cookie is invalid or the layout of the webpage is different."
        )

    for article in all_articles:
        # INFO: Get the a tag in the article which has an img tag as child
        a_tag = article.find_all("a")

        if not a_tag:
            raise Exception(
                "Could not find photos, maybe the cookie is invalid or the layout of the webpage is different."
            )
        img_tag = None
        img_name = ""
        for a in a_tag:
            img_tag = a.find("img")
            if img_tag:
                break
                # INFO: Get the name of the photo if the href contains "photo"
            elif "photo" in a["href"]:
                img_name = a.text

        # INFO: Get the img tag in the a tag

        if img_tag is None:
            raise Exception(
                "Could not find photos, maybe the cookie is invalid or the layout of the webpage is different."
            )

        # INFO: Get the data-src attribute of the img tag
        image_url = img_tag["data-src"]

        imge_urls_store.append(
            {"name": img_name.split(" ")[0], "url": image_url[:-8]+"."+image_url.split('.')[-1]}
        )
        # import pdb; pdb.set_trace()
    return imge_urls_store


def date_from_url(url):
    if url[-1] == "/":
        event_name = url.split("/")[-2]
    else:
        event_name = url.split("/")[-1]

    day=None
    month=None
    year=None
    for dn in event_name.split('-'):
        if(dn.isnumeric() and month is None):
            day=dn
            continue
        if(not dn.isnumeric() and month is None):
            month = dn
            continue
        if(month is not None):
            year = dn
            break
    if(day is not None and month is not None and year is not None):
        pdt = pd.to_datetime(year+"-"+month+"-"+day)
        return pdt.year, pdt.month, pdt.day
    else:
        return None, None, None


def eventname_from_url(url):
    if url[-1] == "/":
        event_name = url.split("/")[-2]
    else:
        event_name = url.split("/")[-1]

    return event_name



def get_all_image_url(event_url, cookie):
    print(f"Analysing {event_url}...")
    all_img_urls_evnt = []
    html_page = get_html_page(event_url, cookie)
    # INFO: Get the name from the end of the URL
    event_name = eventname_from_url(event_url)
    queryparams = {"page": 1}

    image_exists = get_images_existing_on_page(html_page, event_name)
    while True:
        queryparams["page"] += 1
        print(f"Fetching page {queryparams['page']}...")

        try:
            html_page = get_html_page(event_url, cookie, queryparams)
            images_page = get_images_existing_on_page(html_page, event_name)
            if(len(images_page)==0):
                break
            else:
                all_img_urls_evnt += images_page
        except Exception as e:
            print(f"Skipping {event_name} page {queryparams['page']}: {e}")
            continue
    
    print(f"Images total found for the event:{event_name}  - {len(all_img_urls_evnt)}")
    return all_img_urls_evnt


def process_event_url(url, cookie): # year, month, day, event_path):
    """Process an event folder, extract faces, and store embeddings and cutouts."""
    year, month, day = date_from_url(url)

    #import pdb; pdb.set_trace();

    if(year is None or month is None or day is None):
        print(f"ERROR: date not recognized in url:{url}. Skipping......... ")
        return
    eventname = eventname_from_url(url)
    all_img_urls_evnt = get_all_image_url(url,cookie)
    
    print(f"eventname: {eventname}   num_images:{len(all_img_urls_evnt)}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # event = os.path.basename(event_path)
    print(f"Processing event: {eventname}   numimages:{len(all_img_urls_evnt)}")
    face_recog_ = FaceRecognizer()
    for image_file_url in all_img_urls_evnt:
        print(f"Loading image {image_file_url['name']}...")
        response = requests.get(image_file_url['url'])

        # Need to give it to face recognizer directly
        faces = face_recog_.identify_faces(response.content)

        print(f"Found {len(faces)} faces in {image_file_url['name']}:{image_file_url['url']}")
        
        for face in faces:
            face_id = face["face_id"]
            location = face["location"]
            c.execute("INSERT INTO faces (event_date, event, image_path, location, face_id) VALUES (?, ?, ?, ?, ?)",
                (pd.to_datetime(f"{year}-{month}-{day}").strftime('%Y-%m-%d'), eventname, image_file_url['url'], str(location), face_id))
        
        print(f"🟢  Processed {len(faces)} faces in {image_file_url['name']}:{image_file_url['url']}")

    conn.commit()

    conn.close()


def get_html_page(url, cookie, queryparams=None):
    headers = {
        "cookie": cookie,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    }
    response = requests.get(url, headers=headers, params=queryparams)
    # The HTML page is not fully loaded here yet. remaining photos load when we scroll down. Get all the photos by scrolling down.
    return response.text
    



def main():
    t1 = time.time()
    # step 0 => get months in s3 bucket
    # foreach month
        # step 1 => get dates in s3 bucket
        # step 2 => for each date get all s3 image urls
        
    event_url_fname = input("Enter URLs filename: (default: small_urls_2.txt)")
    if(event_url_fname==""):
        event_url_fname="small_urls_2.txt"
    event_urls = pd.read_csv(event_url_fname, header=None)
    event_urls = list(event_urls[0].unique())
    
    print(f"Received {len(event_urls)} unique urls")

    cookie = input("Enter your cookie: (default: cookie.txt)")
    if (cookie==""):
        cookie = pd.read_csv("cookie.txt", header=None).iloc[0][0]

    all_image_urls = {}
    all_dates_events = {}


    # Ensure `faces` table exists
    setup_database()


    for eurl in event_urls:
        t1a = time.time()
        year,month,day = date_from_url(eurl)
        if(year is None or month is None or day is None):
            print(f"ERROR: Cannot process {eurl}  because date not found in the url. Continuing...")
            continue
        date = f"{year}-{month}-{day}"
        ename = eventname_from_url(eurl)
        print(f"Processing event:{ename}  date:{year}-{month}-{day}")
        # all_dates.append((year,month,day))
        all_dates_events[(year,month,day)] = eurl
        #all_image_urls[ename] = get_all_image_url(eurl, cookie) # dictionary of 'name' and 'url'
        
        process_event_url(eurl, cookie)
        t1b = time.time()
        print(f"✅ Faces added to database for {date} time taken:{t1b-t1a}")

        
        # conn.commit()
        # conn.close()
        print(f"✅ Daily counts computed for {date} and stored!")

        end_time = time.time()
        print(f"\n⏱  Elapsed time: {end_time - t1a:.2f} seconds")

    
    t2 = time.time()
    print(f"COMPLETED: Events processed:{len(event_urls)}   Total time taken:{t2-t1}")
    # for each event now we have unique face ids that we have recogzied
    # Now across all events across all URLs i need the count of each face_id
    # get a daily count for each year,month,day
    # then sum up the count for all entries - done
    # as new urls keep getting added the DB will keep on becoming bigger
    # if a repeat entry of event is processed - how do we ensure that no new entries are created?
    





if __name__ == "__main__":
    main()


# Accuracy - Mismatches, and spot checking of individual face matches, => the report should be right.
# 1. run for 6 months and generate a report, 
# 2. note down timings of the code (file down + categorisation) and configuration of the machine  -> remove gurudevs face, sort the faces by count, 
# 3. ask someone else to run it. - everyone runs on their machine - 
# 4. Remove known faces - 
# 5. customize dates - 

