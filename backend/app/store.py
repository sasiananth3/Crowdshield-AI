import json
from pathlib import Path
import sqlite3
import threading


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, job_id TEXT, timestamp REAL, data TEXT);
            CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, job_id TEXT, data TEXT, acknowledged INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS alert_verifications (id TEXT PRIMARY KEY, job_id TEXT, data TEXT NOT NULL);
        """)
        self.db.commit()

    def save_job(self, job):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO jobs VALUES (?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                (job["id"], json.dumps(job)),
            )

    def jobs(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT data FROM jobs ORDER BY rowid DESC LIMIT 100"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def sample(self, job_id, sample):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO samples(job_id,timestamp,data) VALUES (?,?,?)",
                (job_id, sample["timestamp"], json.dumps(sample)),
            )

    def history(self, job_id):
        with self.lock:
            rows = self.db.execute(
                "SELECT data FROM samples WHERE job_id=? ORDER BY timestamp LIMIT 10000",
                (job_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def add_alert(self, alert):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO alerts(id,job_id,data) VALUES (?,?,?)",
                (alert["id"], alert["job_id"], json.dumps(alert)),
            )

    def alerts(self, job_id=None):
        with self.lock:
            if job_id:
                rows = self.db.execute(
                    "SELECT data, acknowledged FROM alerts WHERE job_id=? ORDER BY rowid DESC LIMIT 200",
                    (job_id,),
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT data, acknowledged FROM alerts ORDER BY rowid DESC LIMIT 200"
                ).fetchall()
        return [{**json.loads(row[0]), "acknowledged": bool(row[1])} for row in rows]

    def acknowledge(self, alert_id):
        with self.lock, self.db:
            return (
                self.db.execute(
                    "UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,)
                ).rowcount
                > 0
            )

    def add_alert_verification(self, verification):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO alert_verifications(id,job_id,data) VALUES (?,?,?)",
                (
                    verification["id"],
                    verification["job_id"],
                    json.dumps(verification),
                ),
            )

    def alert_verifications(self, job_id=None):
        with self.lock:
            if job_id:
                rows = self.db.execute(
                    "SELECT data FROM alert_verifications WHERE job_id=? ORDER BY rowid DESC LIMIT 200",
                    (job_id,),
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT data FROM alert_verifications ORDER BY rowid DESC LIMIT 200"
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self):
        with self.lock:
            self.db.close()
