import sqlite3
from final.const import DB_FILE, CUTOUTS_DIR
import os
import pandas as pd
import boto3 
from final.s3_client import get_s3_client
import base64


def face_report(clustid):
    s3 = get_s3_client()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # now every face is a list of face_ids that are stored in a pickle object

    c.execute(f"""
        SELECT * from faces where cluster_id={clustid};
        """)
    data = c.fetchall()
    report = {}
    
    for rid,_,ename,iurl,locn,fid,cid in data:
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
            "face_id": fid,
            "clust_id": clustid,
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
        SELECT cluster_id, COUNT(DISTINCT event_date) AS distinct_event_days FROM faces WHERE event_date >= '{start_date}' and event_date <='{end_date}' GROUP BY cluster_id HAVING COUNT(DISTINCT event_date) > 1  order by distinct_event_days DESC;
        """)#
   
    data = c.fetchall()
    report = {}
    
    c.execute(f"""select * from faces order by event_date desc;""")
    all_faces = pd.DataFrame(c.fetchall());
    all_faces.columns = ['rid','event_date','event_name','iurl','location','face_id','cluster_id']
    
    # import pdb; pdb.set_trace()

    s3 = get_s3_client()

    for clust_id, count in data:
        #print(f"🟢  Face {face_id}: Count = {count}\n")
        
        face_presence = all_faces[all_faces['cluster_id']==clust_id]

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

        report[clust_id] = {
            "count": count,
            "cutout": base64.urlsafe_b64encode(location.encode()).decode(),
            "image": base64.urlsafe_b64encode(image_url_s3.encode()).decode(),
            # "face_id": face_id,
            "clust_id": clust_id
        }
    
    # Sort the report by count in descending order
    report = dict( sorted(report.items(), key=lambda x: x[1]["count"], reverse=True) )
    
    conn.close()
    return report
