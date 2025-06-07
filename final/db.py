import sqlite3
from const import DB_FILE

def setup_database():
    """Create necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Table for storing extracted face embeddings
    c.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date DATE,
                event TEXT,
                image_path TEXT,  -- Store the original image path
                location TEXT,  -- The location of the face in the image
                face_id TEXT  -- Unique ID for each face
            )
        """)

    # Table for storing precomputed daily counts
    # c.execute("""
    #     CREATE TABLE IF NOT EXISTS daily_counts (
    #         event_datetime DATETIME,
    #         face_id INTEGER,
    #         image_path TEXT,
    #         location TEXT,
    #         count INTEGER,
    #         PRIMARY KEY (year, month, day, face_id)
    #     )
    # """)

    # # Table for storing precomputed monthly counts
    # c.execute("""
    #     CREATE TABLE IF NOT EXISTS monthly_counts (
    #         year INTEGER,
    #         month INTEGER,
    #         face_id INTEGER,
    #         count INTEGER,
    #         PRIMARY KEY (year, month, face_id)
    #     )
    # """)

    conn.commit()
    conn.close()
