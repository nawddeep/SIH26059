"""Shore-side receiver (NCPOR): accepts the gateway's uplink on TCP :6000 and logs what arrives.

This stands in for mission control. It deliberately does almost nothing clever -
it timestamps arrivals, writes them to disk, and keeps a tally - because its job
in the demo is to be the independent witness that the gateway's claims are true.

Two of those tallies matter:
  * per-source and per-priority counts, which show priority ordering worked
  * duplicate detection by (source, seq), which is how "no loss, no duplication
    across a link outage" stops being an assertion and becomes a measurement
"""
import argparse
import asyncio
import json
import os
import signal
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

HOST = "0.0.0.0"
PORT = 6000
LABEL = "[SHORE]"

PRIORITY_NAMES = {1: "HIGH", 2: "MED", 3: "LOW"}

# Optional: push each record on to the navigation dashboard's API. Set with
# --forward, e.g. --forward http://192.168.1.20:8600/api/telemetry/ingest
forward_url = None
forward_failures = 0

received_total = 0
by_source = Counter()
by_priority = Counter()
seen_keys = set()
duplicates = 0
first_arrival = None
log_file = None


def log(message):
    print(f"{LABEL} {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def summarise_payload(source, payload):
    """A short, readable digest of whatever this feed sent."""
    if not isinstance(payload, dict):
        return str(payload)[:70]
    if source == "gps":
        return (f"{payload.get('lat', '?'):+.5f},{payload.get('lon', '?'):+.5f} "
                f"{payload.get('speed_kn', '?')} kn"
                if isinstance(payload.get("lat"), (int, float)) else str(payload)[:70])
    if source == "ais":
        return f"mmsi {payload.get('mmsi')} {str(payload.get('name', ''))[:20]}"
    if source == "weather":
        return (f"wind {payload.get('wind_speed_kn')} kn  "
                f"vis {payload.get('visibility_km')} km")
    if source == "engine":
        return (f"{payload.get('rpm')} rpm  tank {payload.get('fuel_remaining_pct')}%")
    return str(payload)[:70]


def forward(envelope):
    """Best-effort hand-off to the dashboard API.

    Deliberately fire-and-forget and deliberately *after* the ACK. The uplink's
    job is to get data off the ship; a shore-side dashboard being slow, down or
    absent must never stall that, and must never cause a record to go
    unacknowledged and be re-sent. Failures are counted and logged sparsely,
    never raised.
    """
    global forward_failures
    if not forward_url:
        return
    try:
        req = urllib.request.Request(
            forward_url,
            data=json.dumps(envelope).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2.0).close()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        forward_failures += 1
        if forward_failures % 25 == 1:
            log(f"forward to dashboard failed ({type(exc).__name__}) - "
                f"{forward_failures} so far; uplink unaffected")


def record_arrival(line):
    """Parse one uplinked line, tally it, and return (summary, seq).

    The seq goes back to the caller so it can acknowledge the record. That
    acknowledgement is what makes the gateway's "no data loss" claim true
    rather than hopeful - see the note in handle_client.
    """
    global received_total, duplicates, first_arrival

    arrived = datetime.now(timezone.utc)
    if first_arrival is None:
        first_arrival = arrived

    try:
        envelope = json.loads(line)
    except json.JSONDecodeError:
        # Anything that is not our envelope still gets logged rather than dropped.
        received_total += 1
        by_source["raw"] += 1
        return f"RAW  {line[:90]}", None

    source = envelope.get("source", "unknown")
    priority = envelope.get("priority", 0)
    seq = envelope.get("seq")
    payload = envelope.get("payload", {})

    is_duplicate = False
    if seq is not None:
        key = (source, seq)
        if key in seen_keys:
            duplicates += 1
            is_duplicate = True
        else:
            seen_keys.add(key)

    received_total += 1
    by_source[source] += 1
    by_priority[priority] += 1

    if log_file:
        log_file.write(json.dumps({
            "received_at": arrived.isoformat(timespec="milliseconds"),
            "envelope": envelope,
        }) + "\n")
        log_file.flush()

    tag = PRIORITY_NAMES.get(priority, f"P{priority}")
    dup = "  << DUPLICATE" if is_duplicate else ""
    return (f"{tag:4s} #{seq if seq is not None else '-':<6} {source:8s} "
            f"{summarise_payload(source, payload)}{dup}", seq)


async def handle_client(reader, writer):
    peer = writer.get_extra_info("peername")
    log(f"gateway connected {peer}")
    try:
        while True:
            line = await reader.readline()
            if not line:                       # peer closed the connection
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                summary, seq = record_arrival(text)
                log(summary)
                # Acknowledge only after the record is counted and on disk.
                # The gateway marks nothing delivered until this arrives,
                # because a successful sendall() only proves the data reached
                # a kernel socket buffer - it says nothing about whether it
                # ever crossed the satellite link.
                if seq is not None:
                    writer.write(f"ACK {seq}\n".encode())
                    await writer.drain()
                # Only now, with the gateway already satisfied, tell the dashboard.
                if forward_url:
                    try:
                        await asyncio.to_thread(forward, json.loads(text))
                    except Exception:
                        pass
            except Exception as exc:
                log(f"ERROR handling record: {type(exc).__name__}: {exc}")
    except (ConnectionResetError, BrokenPipeError, OSError) as exc:
        log(f"gateway link dropped: {type(exc).__name__}")
    finally:
        log(f"gateway gone      {peer}  (received {received_total} records so far)")
        try:
            writer.close()
        except OSError:
            pass


def print_summary():
    log("-" * 58)
    log(f"TOTAL RECEIVED : {received_total}")
    log(f"UNIQUE RECORDS : {len(seen_keys)}")
    log(f"DUPLICATES     : {duplicates}")
    if by_priority:
        breakdown = "  ".join(
            f"{PRIORITY_NAMES.get(p, f'P{p}')}={c}" for p, c in sorted(by_priority.items())
        )
        log(f"BY PRIORITY    : {breakdown}")
    if by_source:
        breakdown = "  ".join(f"{s}={c}" for s, c in sorted(by_source.items()))
        log(f"BY SOURCE      : {breakdown}")
    log("-" * 58)


async def main():
    parser = argparse.ArgumentParser(description="Shore-side uplink receiver")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--log", default="shore_received.jsonl",
                        help="file to append received records to")
    parser.add_argument("--forward", default=os.environ.get("FORWARD_URL", ""),
                        help="dashboard ingest URL, e.g. "
                             "http://<host>:8600/api/telemetry/ingest")
    args = parser.parse_args()

    global forward_url
    forward_url = args.forward or None

    global log_file
    log_file = open(args.log, "a", buffering=1)

    server = await asyncio.start_server(handle_client, HOST, args.port)
    log(f"listening on :{args.port}  (shore / NCPOR receiver)")
    log(f"appending to {os.path.abspath(args.log)}")
    log(f"forwarding to {forward_url}" if forward_url
        else "not forwarding to a dashboard (pass --forward URL to enable)")

    stop = asyncio.Event()

    def request_stop():
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_stop)

    async with server:
        await stop.wait()

    print_summary()
    log("shutting down")
    if log_file:
        log_file.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_summary()
