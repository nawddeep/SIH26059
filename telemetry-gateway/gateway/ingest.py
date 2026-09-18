"""Concurrent multi-feed client: reads every ship instrument, normalises it, and buffers it.

One asyncio task per instrument, all running at once, each reading its own TCP
stream line by line. Every line is parsed into one common record shape:

    (source, timestamp, priority, payload)

and handed to a single writer task that commits batches to SQLite.

Two design choices worth defending to a mentor:

  * One writer, not four. sqlite3 calls block, and four tasks writing directly
    would both contend on the database lock and stall the event loop - which
    means a slow disk would stop us *reading* the instruments. Funnelling
    through one task keeps every reader responsive and lets writes batch.

  * A feed that dies must not take the others with it. Each task owns its own
    reconnect loop with exponential backoff, so unplugging one instrument
    leaves the other three ingesting normally.

The ship's address is configurable because ship-sim eventually moves to another
machine; set SHIP_HOST to point at it.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import operator
import os
import signal
from datetime import datetime, timezone

import buffer

SHIP_HOST = os.environ.get("SHIP_HOST", "127.0.0.1")

# Backpressure bound. If the writer ever falls this far behind the readers,
# something is badly wrong and growing the queue further would just trade a
# visible problem for an out-of-memory kill - fatal on a 1 GB Pi.
QUEUE_MAXSIZE = 10_000

BATCH_MAX = 100          # rows per transaction
BATCH_WAIT_S = 0.5       # ...or commit early after this long
STATS_EVERY_S = 15.0

RECONNECT_MIN_S = 1.0
RECONNECT_MAX_S = 30.0

counters = {"parsed": 0, "written": 0, "dropped": 0, "parse_errors": 0}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def log(tag: str, message: str) -> None:
    print(f"[{tag:<8}] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


# --------------------------------------------------------------- NMEA parsing
def nmea_checksum_ok(sentence: str) -> bool:
    """Verify the XOR checksum. A corrupt sentence is dropped, not guessed at."""
    if not sentence.startswith("$") or "*" not in sentence:
        return False
    body, _, checksum = sentence[1:].partition("*")
    try:
        expected = int(checksum[:2], 16)
    except ValueError:
        return False
    return functools.reduce(operator.xor, (ord(c) for c in body), 0) == expected


def nmea_to_decimal(value: str, hemisphere: str) -> float | None:
    """NMEA ddmm.mmmm / dddmm.mmmm plus hemisphere -> signed decimal degrees."""
    if not value or not hemisphere:
        return None
    try:
        point = value.index(".")
    except ValueError:
        return None
    # Degrees are everything before the last two digits ahead of the decimal
    # point, which is what makes one function work for both lat (dd) and lon (ddd).
    degrees = float(value[: point - 2])
    minutes = float(value[point - 2 :])
    decimal = degrees + minutes / 60.0
    return -decimal if hemisphere.upper() in ("S", "W") else decimal


def parse_gps(line: str):
    """One NMEA sentence -> a list of (priority, payload).

    RMC carries position, speed and course: that is the navigational fix, and it
    goes out at high priority. GGA repeats the same position with fix-quality
    detail attached - useful for diagnostics, redundant for navigation - so it
    is tagged low and rides along only when there is bandwidth to spare. That
    split is what gives the priority queue something real to do.
    """
    line = line.strip()
    if not nmea_checksum_ok(line):
        return []

    fields = line[1:].split("*")[0].split(",")
    kind = fields[0]

    if kind.endswith("RMC") and len(fields) >= 10:
        if fields[2] != "A":                      # V = navigation receiver warning
            return []
        lat = nmea_to_decimal(fields[3], fields[4])
        lon = nmea_to_decimal(fields[5], fields[6])
        if lat is None or lon is None:
            return []
        return [(buffer.PRIORITY_HIGH, {
            "type": "position_fix",
            "sentence": kind,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "speed_kn": float(fields[7]) if fields[7] else None,
            "course_deg": float(fields[8]) if fields[8] else None,
            "utc": fields[1],
        })]

    if kind.endswith("GGA") and len(fields) >= 10:
        lat = nmea_to_decimal(fields[2], fields[3])
        lon = nmea_to_decimal(fields[4], fields[5])
        if lat is None or lon is None:
            return []
        return [(buffer.PRIORITY_LOW, {
            "type": "fix_quality",
            "sentence": kind,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "fix_quality": int(fields[6]) if fields[6] else None,
            "satellites": int(fields[7]) if fields[7] else None,
            "hdop": float(fields[8]) if fields[8] else None,
            "altitude_m": float(fields[9]) if fields[9] else None,
            "utc": fields[1],
        })]

    return []


# --------------------------------------------------------------- JSON parsing
def parse_ais(line: str):
    """AIS position reports. Position is navigationally urgent, so: high."""
    record = json.loads(line)
    return [(buffer.PRIORITY_HIGH, record)]


def parse_weather(line: str):
    """Weather. Routine unless the station flagged an alert, which promotes it."""
    record = json.loads(line)
    priority = buffer.PRIORITY_HIGH if record.get("alerts") else buffer.PRIORITY_MEDIUM
    return [(priority, record)]


def parse_engine(line: str):
    """Engine telemetry. Same rule: routine, unless it is shouting about something."""
    record = json.loads(line)
    priority = buffer.PRIORITY_HIGH if record.get("alerts") else buffer.PRIORITY_MEDIUM
    return [(priority, record)]


FEEDS = [
    ("gps", 5001, parse_gps),
    ("ais", 5002, parse_ais),
    ("weather", 5003, parse_weather),
    ("engine", 5004, parse_engine),
]


# ------------------------------------------------------------------- ingestion
async def feed_task(source: str, port: int, parser, queue: asyncio.Queue,
                    stop: asyncio.Event) -> None:
    """Read one instrument forever, reconnecting on its own if the link drops."""
    backoff = RECONNECT_MIN_S

    while not stop.is_set():
        reader = writer = None
        try:
            log(source, f"connecting to {SHIP_HOST}:{port} ...")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(SHIP_HOST, port), timeout=10.0
            )
            log(source, f"connected to {SHIP_HOST}:{port}")
            backoff = RECONNECT_MIN_S          # a good connection resets the backoff

            while not stop.is_set():
                raw = await reader.readline()
                if not raw:                     # instrument closed the stream
                    log(source, "feed closed by the instrument")
                    break

                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                try:
                    for priority, payload in parser(text):
                        counters["parsed"] += 1
                        record = (source, now_iso(), priority, payload)
                        try:
                            queue.put_nowait(record)
                        except asyncio.QueueFull:
                            counters["dropped"] += 1
                            if counters["dropped"] % 100 == 1:
                                log(source, f"WARNING buffer queue full, dropping "
                                            f"(total dropped {counters['dropped']})")
                except (json.JSONDecodeError, ValueError, IndexError, KeyError) as exc:
                    # One malformed line must never kill the feed.
                    counters["parse_errors"] += 1
                    if counters["parse_errors"] % 50 == 1:
                        log(source, f"parse error ({type(exc).__name__}): {text[:60]}")

        except (OSError, asyncio.TimeoutError) as exc:
            log(source, f"link down ({type(exc).__name__}: {exc}); "
                        f"retrying in {backoff:.0f}s")
        except Exception as exc:                # never let one feed kill the process
            log(source, f"UNEXPECTED {type(exc).__name__}: {exc}; retrying in {backoff:.0f}s")
        finally:
            if writer is not None:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

        if stop.is_set():
            break
        # Exponential backoff, capped: a dead instrument should not become a
        # busy-loop hammering a socket that is never coming back.
        try:
            await asyncio.wait_for(stop.wait(), timeout=backoff)
        except asyncio.TimeoutError:
            pass
        backoff = min(backoff * 2.0, RECONNECT_MAX_S)

    log(source, "feed task stopped")


async def writer_task(queue: asyncio.Queue, stop: asyncio.Event) -> None:
    """The only thing that touches SQLite. Batches commits to spare the SD card."""
    conn = buffer.connect()
    log("buffer", f"queue database at {buffer.db_path()}")
    log("buffer", f"opening state: {buffer.format_stats(buffer.stats(conn))}")

    try:
        while not (stop.is_set() and queue.empty()):
            batch = []
            try:
                # Block for the first item so an idle gateway does not spin.
                first = await asyncio.wait_for(queue.get(), timeout=BATCH_WAIT_S)
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            # Then take whatever else is already waiting, up to the batch cap.
            while len(batch) < BATCH_MAX:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                # to_thread keeps the blocking commit off the event loop, so the
                # feeds keep reading while the disk write happens.
                written = await asyncio.to_thread(buffer.insert_many, conn, batch)
                counters["written"] += written
            except Exception as exc:
                log("buffer", f"WRITE FAILED ({type(exc).__name__}: {exc}) - "
                              f"{len(batch)} records lost from this batch")
    finally:
        conn.close()
        log("buffer", "writer stopped")


async def stats_task(queue: asyncio.Queue, stop: asyncio.Event) -> None:
    """Periodic one-line summary, so a demo audience can watch the buffer fill."""
    conn = buffer.connect()
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=STATS_EVERY_S)
                break
            except asyncio.TimeoutError:
                pass
            st = await asyncio.to_thread(buffer.stats, conn)
            log("stats", f"{buffer.format_stats(st)}  "
                         f"parsed={counters['parsed']} written={counters['written']} "
                         f"dropped={counters['dropped']} errors={counters['parse_errors']} "
                         f"inflight={queue.qsize()}")
    finally:
        conn.close()


async def main() -> None:
    log("ingest", f"gateway ingestion starting - ship at {SHIP_HOST}")
    log("ingest", f"feeds: " + ", ".join(f"{s}:{p}" for s, p, _ in FEEDS))

    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    tasks = [asyncio.create_task(feed_task(s, p, fn, queue, stop)) for s, p, fn in FEEDS]
    tasks.append(asyncio.create_task(writer_task(queue, stop)))
    tasks.append(asyncio.create_task(stats_task(queue, stop)))

    await stop.wait()
    log("ingest", "shutdown requested, draining...")
    await asyncio.gather(*tasks, return_exceptions=True)

    conn = buffer.connect()
    log("ingest", f"final: {buffer.format_stats(buffer.stats(conn))}")
    conn.close()
    log("ingest", "stopped cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
