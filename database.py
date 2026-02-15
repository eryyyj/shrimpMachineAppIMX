import sqlite3, os, datetime, bcrypt, uuid
from pymongo import MongoClient
from bson import ObjectId

# Configuration
DB_PATH = "local.db"
MONGO_URI = "mongodb+srv://qajgvalencia:BUxIhYb4nDlfH4DV@cluster0.h07iggq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_DB_NAME = "test"

# Database Initialization
def init_db():
    """Initialize local SQLite database tables."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id TEXT PRIMARY KEY,
        username TEXT,
        email TEXT,
        password TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS biomass_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ownerId TEXT,
        recordId TEXT,
        shrimpCount INTEGER,
        biomass REAL,
        feedMeasurement REAL,
        dateTime TEXT,
        synced INTEGER DEFAULT 0
    )
    """)
    conn.commit()
    conn.close()

# QR Handshake Functions
def create_qr_session():
    """Generates a unique ID and puts it in MongoDB to wait for a mobile scan."""
    session_id = str(uuid.uuid4())
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGO_DB_NAME]
        db["qrsessions"].insert_one({
            "sessionId": session_id,
            "userId": None,
            "status": "pending",
            "createdAt": datetime.datetime.utcnow() 
        })
        return session_id
    except Exception as e:
        print("MongoDB QR Session Error:", e)
        return None
    
def poll_for_login(session_id):
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = client[MONGO_DB_NAME]
        session = db["qrsessions"].find_one({"sessionId": session_id})
        
        if session and session.get("userId"):
            user_id = session["userId"]
            # Try to find user by ObjectId or by String
            user_data = db["users"].find_one({"_id": ObjectId(str(user_id))})
            
            if user_data:
                cache_user(str(user_id), user_data['username'], user_data['email'], user_data['password'])
                return str(user_id)
    except Exception as e:
        print(f"Polling error: {e}")
    return None

# Local User Caching
def cache_user(uid, username, email, hashed_pw):
    """Save user locally so they can log in offline next time."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    INSERT OR IGNORE INTO users(id, username, email, password)
    VALUES(?,?,?,?)
    """, (uid, username, email, hashed_pw))
    conn.commit()
    conn.close()

# Biomass Record Handling (REQUIRED BY UI)
def save_biomass_record(owner_id, shrimp_count, biomass, feed_measurement):
    conn = sqlite3.connect(DB_PATH)
    record_id = str(uuid.uuid4())
    date_time = datetime.datetime.now().isoformat()
    conn.execute("""
    INSERT INTO biomass_records(ownerId, recordId, shrimpCount, biomass, feedMeasurement, dateTime, synced)
    VALUES(?,?,?,?,?, ?,0)
    """, (owner_id, record_id, shrimp_count, biomass, feed_measurement, date_time))
    conn.commit()
    conn.close()

def get_all_records(owner_id):
    """Retrieve all local records belonging to a specific user."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM biomass_records WHERE ownerId=? ORDER BY id DESC",
        (owner_id,)
    ).fetchall()
    conn.close()
    return rows

def get_last_record(owner_id=None):
    conn = sqlite3.connect(DB_PATH)
    if owner_id:
        row = conn.execute(
            "SELECT * FROM biomass_records WHERE ownerId=? ORDER BY id DESC LIMIT 1",
            (owner_id,)
        ).fetchone()
    else:
        row = conn.execute("SELECT * FROM biomass_records ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row

def delete_record(record_id, owner_id):
    """Delete a specific record locally (and try MongoDB if synced)."""
    conn = sqlite3.connect(DB_PATH)
    record = conn.execute(
        "SELECT recordId, synced FROM biomass_records WHERE id=? AND ownerId=?",
        (record_id, owner_id)
    ).fetchone()

    if not record:
        conn.close()
        return

    record_uuid, synced = record
    conn.execute("DELETE FROM biomass_records WHERE id=? AND ownerId=?", (record_id, owner_id))
    conn.commit()
    conn.close()

    if synced == 1:
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
            db = client[MONGO_DB_NAME]
            db["biomassrecords"].delete_one({"recordId": record_uuid})
        except Exception:
            pass

def sync_biomass_records(owner_id):
    """Sync only the current user's unsynced records to MongoDB Atlas."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT ownerId, recordId, shrimpCount, biomass, feedMeasurement, dateTime
        FROM biomass_records
        WHERE synced=0 AND ownerId=?
    """, (owner_id,)).fetchall()

    if not rows:
        conn.close()
        return 0

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000)
        db = client[MONGO_DB_NAME]          
        col = db["biomassrecords"]           

        docs = []
        for (o_id, r_id, count, bio, feed, dt) in rows:
            dt_obj = datetime.datetime.fromisoformat(dt).astimezone(datetime.timezone.utc)
            docs.append({
                "ownerId": ObjectId(str(o_id)) if len(str(o_id)) == 24 else str(o_id),
                "recordId": r_id,
                "shrimpCount": count,
                "biomass": bio,
                "feedMeasurement": feed,
                "dateTime": datetime.datetime.fromisoformat(dt)
            })

        if docs:
            col.insert_many(docs)
            conn.execute("UPDATE biomass_records SET synced=1 WHERE ownerId=?", (owner_id,))
            conn.commit()
        n = len(docs)
    except Exception:
        n = 0

    conn.close()
    return n

def verify_user_credentials(identifier, password):
    """Primary check against MongoDB Atlas, falls back to Local for offline support."""
    import bcrypt
    
    # 1. TRY MONGODB ATLAS FIRST (Main Database)
    try:
        # Use existing MONGO_URI and MONGO_DB_NAME
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGO_DB_NAME]
        
        # Look for user by email or username in your main cloud database
        user_data = db["users"].find_one({
            "$or": [{"email": identifier}, {"username": identifier}]
        })

        if user_data:
            hashed_pw = user_data['password']
            # Convert to bytes if stored as string
            pw_bytes = hashed_pw if isinstance(hashed_pw, bytes) else hashed_pw.encode('utf-8')
            
            if bcrypt.checkpw(password.encode('utf-8'), pw_bytes):
                uid = str(user_data['_id'])
                # Update local cache so they can log in even if internet is down later
                cache_user(uid, user_data['username'], user_data['email'], hashed_pw)
                return uid
    except Exception as e:
        print(f"MongoDB Login Error (checking local instead): {e}")

    # 2. OFFLINE FALLBACK: Check Local SQLite
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, password FROM users WHERE username=? OR email=?", 
        (identifier, identifier)
    ).fetchone()
    conn.close()

    if row:
        user_id, hashed_pw = row
        try:
            pw_bytes = hashed_pw if isinstance(hashed_pw, bytes) else hashed_pw.encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), pw_bytes):
                return user_id
        except Exception as e:
            print(f"Local login fallback error: {e}")

    return None