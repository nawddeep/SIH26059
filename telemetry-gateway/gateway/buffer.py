"""Durable SQLite store-and-forward queue: schema plus insert/claim/mark-sent helpers.

This file is the whole reason the gateway can survive a link outage. Records go
to disk the moment they are parsed, and are only marked sent once the shore end
has actually taken them. Nothing lives solely in memory, so a dropped link, a
killed process or a power cut costs at most the records still in flight.

Concurrency note: ingest.py writes and sender.py reads/updates, in two separate
processes, against one file. WAL mode is what makes that safe - it lets a reader
work while a writer commits, instead of the two locking each other out.

Priority tiers used throughout:
    1 HIGH   position fixes and anything carrying an alert - always goes first
    2 MEDIUM routine weather and engine telemetry
    3 LOW    verbose or redundant detail - only when there is spare bandwidth
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3

PRIORITY_NAMES = {PRIORITY_HIGH: "HIGH", PRIORITY_MEDIUM: "MED", PRIORITY_LOW: "LOW"}

# Kept beside this file rather than at an absolute path, so the folder can be
# copied to any machine - or any user's home on the Pi - and still work.
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "buffer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    priority     INTEGER NOT NULL,
    payload_json TEXT    NOT NULL,
    sent_flag    INTEGER NOT NULL DEFAULT 0,
    sent_at      TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0
);

-- The sender's only query is "unsent, best priority first, oldest first".
-- This index makes that a lookup rather than a scan of the whole backlog,
-- which matters once an outage has left tens of thousands of rows queued.
CREATE INDEX IF NOT EXISTS idx_queue_unsent
    ON queue (sent_flag, priority, id);
"""


def db_path() -> Path:
    """Where the queue lives. TELEMETRY_DB overrides it for tests or a demo reset."""
    override = os.environ.get("TELEMETRY_DB")
    return Path(override) if override else DEFAULT_DB_PATH


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the queue, creating it if needed, with settings tuned for a Pi."""
    target = Path(path) if path else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False because the callers hand their blocking commits to
    # asyncio.to_thread, which runs them on whichever worker thread is free. That
    # is only safe because each caller awaits one commit at a time, so a single
    # connection is never touched concurrently - it just moves between threads.
    conn = sqlite3.connect(
        str(target), timeout=10.0, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row

    # WAL lets the sender read while ingest commits - without it the two
    # processes would serialise on a single write lock and stall each other.
    conn.execute("PRAGMA journal_mode=WAL")
    # NORMAL is durable across process and OS crashes, which is what this has to
    # survive. FULL would also survive a power cut, at the cost of an fsync per
    # commit - punishing on an SD card, for a failure mode the demo never hits.
    conn.execute("PRAGMA synchronous=NORMAL")
    # Rather than failing instantly on a locked database, wait for the peer.
    conn.execute("PRAGMA busy_timeout=5000")

    conn.executescript(SCHEMA)
    return conn


def insert(conn: sqlite3.Connection, source: str, timestamp: str,
           priority: int, payload: dict) -> int:
    """Queue one record. Returns the row id, which doubles as its sequence number."""
    cur = conn.execute(
        "INSERT INTO queue (source, timestamp, priority, payload_json) VALUES (?, ?, ?, ?)",
        (source, timestamp, priority, json.dumps(payload, separators=(",", ":"))),
    )
    return int(cur.lastrowid)


def insert_many(conn: sqlite3.Connection, records: list) -> int:
    """Queue a batch in one transaction.

    Batching is not premature optimisation here: one commit per record means one
    WAL sync per record, and on an SD card that is the difference between the
    buffer keeping up with four feeds and falling behind them.

    Each record is (source, timestamp, priority, payload_dict).
    """
    if not records:
        return 0
    rows = [
        (source, timestamp, priority, json.dumps(payload, separators=(",", ":")))
        for source, timestamp, priority, payload in records
    ]
    with conn:
        conn.executemany(
            "INSERT INTO queue (source, timestamp, priority, payload_json) VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def select_unsent(conn: sqlite3.Connection, limit: int = 50) -> list:
    """The next records to transmit: highest priority first, oldest first within it.

    This ordering is the entire point of the priority scheme. After an outage the
    backlog may be thousands of rows, and this is what guarantees a position fix
    queued a second ago still beats a low-priority row queued an hour ago.
    """
    return list(conn.execute(
        """SELECT id, source, timestamp, priority, payload_json, attempts
             FROM queue
            WHERE sent_flag = 0
            ORDER BY priority ASC, id ASC
            LIMIT ?""",
        (limit,),
    ))


def mark_sent(conn: sqlite3.Connection, ids: list) -> int:
    """Mark rows delivered. Called only after the shore end has taken them.

    Marking after rather than before is what makes a crash mid-send safe: the
    worst case is re-sending a record the shore already has, and the shore
    de-duplicates on (source, seq). Marking first would lose it instead, and a
    lost position fix is far worse than a duplicated one.
    """
    if not ids:
        return 0
    with conn:
        conn.executemany(
            "UPDATE queue SET sent_flag = 1, sent_at = datetime('now') WHERE id = ?",
            [(i,) for i in ids],
        )
    return len(ids)


def bump_attempts(conn: sqlite3.Connection, ids: list) -> None:
    """Record that delivery was tried and failed, for visibility during a demo."""
    if not ids:
        return
    with conn:
        conn.executemany(
            "UPDATE queue SET attempts = attempts + 1 WHERE id = ?", [(i,) for i in ids]
        )


def stats(conn: sqlite3.Connection) -> dict:
    """A snapshot of the queue: totals, backlog, and breakdowns for the console."""
    total = conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    unsent = conn.execute("SELECT COUNT(*) FROM queue WHERE sent_flag = 0").fetchone()[0]

    by_source = {
        row["source"]: {"total": row["n"], "unsent": row["u"]}
        for row in conn.execute(
            """SELECT source, COUNT(*) AS n,
                      SUM(CASE WHEN sent_flag = 0 THEN 1 ELSE 0 END) AS u
                 FROM queue GROUP BY source ORDER BY source"""
        )
    }
    by_priority = {
        int(row["priority"]): {"total": row["n"], "unsent": row["u"]}
        for row in conn.execute(
            """SELECT priority, COUNT(*) AS n,
                      SUM(CASE WHEN sent_flag = 0 THEN 1 ELSE 0 END) AS u
                 FROM queue GROUP BY priority ORDER BY priority"""
        )
    }
    return {
        "total": total,
        "sent": total - unsent,
        "unsent": unsent,
        "by_source": by_source,
        "by_priority": by_priority,
    }


def format_stats(st: dict) -> str:
    """One-line queue summary for logs."""
    parts = [f"buffered={st['total']}", f"sent={st['sent']}", f"backlog={st['unsent']}"]
    if st["by_priority"]:
        tiers = " ".join(
            f"{PRIORITY_NAMES.get(p, p)}:{v['unsent']}/{v['total']}"
            for p, v in sorted(st["by_priority"].items())
        )
        parts.append(f"[{tiers}]")
    return "  ".join(parts)


if __name__ == "__main__":
    # Running this file directly prints the current queue state - the quickest
    # way to check the buffer during a demo without opening a SQLite shell.
    import sys

    conn = connect()
    st = stats(conn)
    print(f"database : {db_path()}")
    print(f"summary  : {format_stats(st)}")
    print("\nby source:")
    for src, v in st["by_source"].items():
        print(f"  {src:10s} total={v['total']:6d}  unsent={v['unsent']:6d}")
    print("\nby priority:")
    for p, v in sorted(st["by_priority"].items()):
        print(f"  {PRIORITY_NAMES.get(p, p):10s} total={v['total']:6d}  unsent={v['unsent']:6d}")
    if "--peek" in sys.argv:
        print("\nnext 10 to send:")
        for row in select_unsent(conn, 10):
            print(f"  #{row['id']:<6} {PRIORITY_NAMES.get(row['priority']):4s} "
                  f"{row['source']:8s} {row['payload_json'][:64]}")
    conn.close()
