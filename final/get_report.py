import sqlite3
from final.const import DB_FILE, CUTOUTS_DIR
import os
import pandas as pd

def get_report(start_date, end_date):
    """Retrieve unique face counts, cutout paths, and original image paths for a date range."""
    start_year, start_month, start_day = map(int, start_date.split("-"))
    end_year, end_month, end_day = map(int, end_date.split("-"))
    print(start_year, start_month, start_day)
    print(end_year, end_month, end_day)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Fetch counts and image paths
    c.execute("""
        SELECT face_id, COUNT(*)
        FROM faces
        WHERE (year || '-' || month || '-' || day) BETWEEN ? AND ?
        GROUP BY face_id
    """, (start_date, end_date))
    data = c.fetchall()

    report = {}

    for face_id, count in data:
        cutout_path = os.path.join(CUTOUTS_DIR, f"face_{face_id}.jpg")

        # Fetch one original image path for this face
        c.execute("SELECT image_path FROM faces WHERE face_id = ? LIMIT 1", (face_id,))
        image_path = c.fetchone()[0] if c.fetchone() else None

        report[face_id] = {
            "count": count,
            "cutout": cutout_path,
            "image": image_path  # Original image path
        }

    conn.close()
    return report


def face_report(faceid):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"""
        SELECT * from faces where face_id={faceid};
        """)
    data = c.fetchall()
    report = {}
    for rid,_,ename,iurl,locn,fid in data:
        report[rid] = {
            "event_name": ename,
            "cutout": locn,
            "image": iurl,
            "iurl": iurl,
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
    for face_id, count in data:
        
        
        face_presence = all_faces[all_faces['face_id']==face_id]

        
        image_url = face_presence.iloc[0]['iurl']
        location = face_presence.iloc[0]['location']
        

        #print(f"    Cutout Image: {location}\n")
        #print(f"    Example Image: {image_url}\n")
        report[face_id] = {
            "count": count,
            "cutout": location,
            "image": image_url,
            "face_id": face_id
        }
    
    # Sort the report by count in descending order
    report = dict( sorted(report.items(), key=lambda x: x[1]["count"], reverse=True) )
    
    conn.close()
    return report
