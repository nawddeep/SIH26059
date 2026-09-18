"""Uplink drainer: forwards buffered records to shore in priority order over the constrained link.

Runs as its own process, deliberately. Ingestion must keep buffering while this
is stuck retrying a dead satellite link, and the only way to guarantee that is
for the two never to share a thread of control. They meet only at the SQLite
queue.

Delivery guarantee: at-least-once, never at-most-once.
    A record is marked sent only after the SHORE END has acknowledged it by
    sequence number. This is not paranoia: a successful sendall() only proves
    the bytes reached a local kernel socket buffer. Under a dead satellite link
    the write still succeeds, the bytes sit in that buffer, and the gateway
    would happily mark records delivered that never left the ship. Measured on
    a 100%-loss link, that bug lost 6 of 578 records while reporting success.

    So: send the batch, wait for "ACK <seq>" per record, mark only what came
    back. Anything unacknowledged stays queued and is re-sent on reconnect; the
    shore de-duplicates on (source, seq). Losing a position fix is far worse
    than sending one twice.

THROTTLING THE LINK (do this outside Python, at the OS level)
------------------------------------------------------------
Nothing in this file simulates bad bandwidth. Faking it in application code
would prove nothing, because the point is that the *real* socket behaves badly.
Linux `tc` with `netem` does it properly, on the interface the traffic uses.

Ship-sim and pi-scraper both run on this Pi, so uplink traffic goes over
loopback. Throttle `lo`, never `wlan0` - your SSH session is on wlan0 and
throttling it will cut you off mid-demo.

    # Iridium Certus-class: ~32 kbit/s, 700 ms latency, 5% loss
    sudo /sbin/tc qdisc add dev lo root netem rate 32kbit delay 700ms loss 5%

    # Inspect what is currently applied
    sudo /sbin/tc qdisc show dev lo

    # Total outage - drop everything (the "we lost the satellite" moment)
    sudo /sbin/tc qdisc change dev lo root netem loss 100%

    # Back to the degraded-but-working link
    sudo /sbin/tc qdisc change dev lo root netem rate 32kbit delay 700ms loss 5%

    # Remove all shaping, back to full speed
    sudo /sbin/tc qdisc del dev lo root

If ship-sim later moves to a separate machine, throttle the real interface
carrying the traffic (`wlan0`/`eth0`) on the *sending* side instead of `lo`.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from datetime import datetime

import buffer

SHORE_HOST = os.environ.get("SHORE_HOST", "127.0.0.1")
SHORE_PORT = int(os.environ.get("SHORE_PORT", "6000"))

BATCH_SIZE = 20          # rows pulled from the queue per round
IDLE_SLEEP_S = 1.0       # pause when the backlog is empty
SOCKET_TIMEOUT_S = 20.0  # generous: a throttled link is slow, not dead
ACK_TIMEOUT_S = 30.0     # must exceed the worst round trip the link can impose
RECONNECT_MIN_S = 1.0
RECONNECT_MAX_S = 15.0
STATS_EVERY_S = 15.0

running = True
counters = {"sent": 0, "bytes": 0, "failures": 0, "reconnects": 0}


def log(message: str) -> None:
    print(f"[sender  ] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def stop(*_args) -> None:
    global running
    running = False
    log("shutdown requested")


def envelope(row) -> bytes:
    """One queue row -> one newline-delimited JSON message for the shore end.

    `seq` is the queue's own row id. It is monotonic and never reused, which is
    what lets the shore de-duplicate re-sends after a link failure.
    """
    return (json.dumps({
        "seq": row["id"],
        "source": row["source"],
        "timestamp": row["timestamp"],
        "priority": row["priority"],
        "payload": json.loads(row["payload_json"]),
    }, separators=(",", ":")) + "\n").encode()


def connect_to_shore():
    """One attempt. Returns a socket, or None if the link is down."""
    try:
        sock = socket.create_connection((SHORE_HOST, SHORE_PORT), timeout=SOCKET_TIMEOUT_S)
        sock.settimeout(SOCKET_TIMEOUT_S)
        # A buffered reader for the acknowledgement stream coming back.
        return sock, sock.makefile("rb")
    except OSError:
        return None


def main() -> int:
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    conn = buffer.connect()
    log(f"uplink target {SHORE_HOST}:{SHORE_PORT}")
    log(f"queue at {buffer.db_path()}")
    log(f"opening state: {buffer.format_stats(buffer.stats(conn))}")

    sock = None
    backoff = RECONNECT_MIN_S
    last_stats = time.time()

    while running:
        # --- make sure we have a link ------------------------------------
        if sock is None:
            link = connect_to_shore()
            if link is None:
                log(f"uplink DOWN - no route to {SHORE_HOST}:{SHORE_PORT}, "
                    f"retrying in {backoff:.0f}s (backlog keeps growing, nothing lost)")
                slept = 0.0
                while running and slept < backoff:
                    time.sleep(0.25)
                    slept += 0.25
                backoff = min(backoff * 2.0, RECONNECT_MAX_S)
                continue
            sock, ack_reader = link
            counters["reconnects"] += 1
            backoff = RECONNECT_MIN_S
            log(f"uplink UP (connection #{counters['reconnects']})")

        # --- take the next batch, best priority first ---------------------
        rows = buffer.select_unsent(conn, BATCH_SIZE)
        if not rows:
            time.sleep(IDLE_SLEEP_S)
            if time.time() - last_stats >= STATS_EVERY_S:
                log(f"idle - {buffer.format_stats(buffer.stats(conn))}")
                last_stats = time.time()
            continue

        expected = set()
        acked: list[int] = []
        t0 = time.time()
        try:
            for row in rows:
                blob = envelope(row)
                sock.sendall(blob)
                expected.add(int(row["id"]))
                counters["bytes"] += len(blob)

            # Now wait for shore to confirm. Nothing is marked until it does.
            deadline = time.time() + ACK_TIMEOUT_S
            while expected and time.time() < deadline:
                line = ack_reader.readline()
                if not line:
                    raise OSError("shore closed the connection")
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith("ACK "):
                    seq = int(text[4:])
                    if seq in expected:
                        expected.discard(seq)
                        acked.append(seq)
            if expected:
                raise socket.timeout(
                    f"{len(expected)} of {len(rows)} records unacknowledged"
                )
        except (OSError, socket.timeout, ValueError) as exc:
            # Only acknowledged records are marked. The rest stay queued and go
            # again on reconnect - the shore discards any it already holds.
            counters["failures"] += 1
            log(f"uplink FAILED ({type(exc).__name__}: {exc}) - "
                f"{len(acked)}/{len(rows)} acknowledged; "
                f"{len(rows) - len(acked)} stay queued for re-send")
            buffer.mark_sent(conn, acked)
            counters["sent"] += len(acked)
            buffer.bump_attempts(conn, sorted(expected))
            try:
                sock.close()
            except OSError:
                pass
            sock = None
            continue

        delivered = acked
        buffer.mark_sent(conn, delivered)
        counters["sent"] += len(delivered)

        elapsed = max(time.time() - t0, 1e-6)
        tiers = {}
        for row in rows:
            name = buffer.PRIORITY_NAMES.get(row["priority"], "?")
            tiers[name] = tiers.get(name, 0) + 1
        mix = " ".join(f"{k}:{v}" for k, v in sorted(tiers.items()))
        log(f"sent {len(delivered):3d} [{mix}] in {elapsed:5.2f}s "
            f"({len(delivered)/elapsed:6.1f} rec/s)  "
            f"{buffer.format_stats(buffer.stats(conn))}")

        if time.time() - last_stats >= STATS_EVERY_S:
            last_stats = time.time()

    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass

    log(f"final: {buffer.format_stats(buffer.stats(conn))}")
    log(f"totals: sent={counters['sent']} bytes={counters['bytes']} "
        f"failures={counters['failures']} reconnects={counters['reconnects']}")
    conn.close()
    log("stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
