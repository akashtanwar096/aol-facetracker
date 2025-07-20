# import face_recognition
import numpy as np
import os
import sqlite3
# import cv2
# import uuid
# import hashlib
import sys
# from bs4 import BeautifulSoup as bs
sys.path.append('./final')
# import requests
from const import DB_FILE, CUTOUTS_DIR
# from face_recognizer_3 import FaceRecognizer
from final.face_embedding_db_3 import FaceEmbeddingDB  # Adjust path accordingly
from datetime import datetime
import pandas as pd
# import pdb
import time
from db import setup_database
import boto3
import pytz
# from final.s3_client import get_s3_client

india_tz = pytz.timezone('Asia/Kolkata')



def insert_new_face_instance_in_db(faces, edate, eventname, s3obj):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for face in faces:
        clust_id = face["clust_id"]
        face_id = face["face_id"]
        location = face["location"]
        c.execute("INSERT INTO faces (event_date, event, image_path, location, face_id, cluster_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (edate.strftime('%Y-%m-%d'), eventname, s3obj['Key'], str(location), face_id, clust_id))
                
        # here we also need to replace the cluster id of all the face_id in faces.db with the return value
        c.execute(f"UPDATE faces SET cluster_id = {clust_id} WHERE face_id = {face_id};")

    conn.commit()
    conn.close()



def merge_2_clusters(clust_id_1, clust_id_2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    faces_db = FaceEmbeddingDB(device=device)
    fids_2,clust_id_1 = faces_db.merge_clusters(clust_id_1, clust_id_2)
    
    # for all these faceids we need to change the cluster id
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for fid in fids_2:
        # Here we also need to replace the cluster id of all the face_id in faces.db with the return value
        c.execute(f"UPDATE faces SET cluster_id = {clust_id_1} WHERE face_id = {fid};")

    conn.commit()
    conn.close()



def unmerge_2_faces(face_id_1, face_id_2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    faces_db = FaceEmbeddingDB(device=device)

    new_clust_id = unmerge_faces(face_id_1, face_id_2)
    
    # for all these faceids we need to change the cluster id
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute(f"UPDATE faces SET cluster_id = {new_clust_id} WHERE face_id = {face_id_2};")

    conn.commit()
    conn.close()

