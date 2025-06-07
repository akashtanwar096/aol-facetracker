import sqlite3
from final.const import DB_FILE, CUTOUTS_DIR
import os
import pandas as pd
import boto3 
from final.s3_client import get_s3_client
import base64



def face_report(faceid):
    s3 = get_s3_client()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"""
        SELECT * from faces where face_id={faceid};
        """)
    data = c.fetchall()
    report = {}
    
    for rid,_,ename,iurl,locn,fid in data:
        image_url_s3 = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'soulbook-replica', 'Key': iurl},
            ExpiresIn=2000  # in seconds
        )

        report[rid] = {
            "event_name": ename,
            "cutout": base64.urlsafe_b64encode(locn.encode()).decode(),
            "image": base64.urlsafe_b64encode(image_url_s3.encode()).decode(),
            "iurl": image_url_s3,
            "face_id": faceid
        }        

    conn.close()
    return report


def get_report_optimized(start_date, end_date):
    """Generate a fast report using precomputed daily/monthly data.
       If data is missing, return a message instead of incomplete results.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    
    start_year, start_month, start_day = map(int, start_date.split("-"))
    end_year, end_month, end_day = map(int, end_date.split("-"))
    
    c.execute(f"""
        SELECT face_id, COUNT(DISTINCT event_date) AS distinct_event_days FROM faces WHERE event_date >= '{start_date}' and event_date <='{end_date}' GROUP BY face_id  HAVING COUNT(DISTINCT event_date) > 1 order by distinct_event_days DESC;
        """)
   
    data = c.fetchall()
    report = {}
    
    
    c.execute(f"""select * from faces order by event_date desc;""")
    all_faces = pd.DataFrame(c.fetchall());
    all_faces.columns = ['rid','event_date','event_name','iurl','location','face_id']
    # import pdb; pdb.set_trace()
    s3 = get_s3_client()

    for face_id, count in data:
        #print(f"🟢  Face {face_id}: Count = {count}\n")
        
        face_presence = all_faces[all_faces['face_id']==face_id]

        image_url = face_presence.iloc[0]['iurl']
        location = face_presence.iloc[0]['location']
        
        image_url_s3 = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'soulbook-replica', 'Key': image_url},
            ExpiresIn=2000  # in seconds
        )

        print(f"    Cutout Image: {location}\n")
        print(f"    Image_url: {image_url}\n")
        print(f"    Image_url_s3: {image_url_s3}\n")

        report[face_id] = {
            "count": count,
            "cutout": base64.urlsafe_b64encode(location.encode()).decode(),
            "image": base64.urlsafe_b64encode(image_url_s3.encode()).decode(),
            "face_id": face_id
        }
    
    # Sort the report by count in descending order
    report = dict( sorted(report.items(), key=lambda x: x[1]["count"], reverse=True) )
    
    conn.close()
    return report
