# Telemetry gateway

Gets vessel data off a research ship and ashore over a satellite link that is
narrowband, high-latency and frequently absent.

Below roughly 70°S there is little or no geostationary coverage, so polar
vessels fall back on narrowband LEO links that are slow, costly per megabyte,
and unavailable for stretches at a time. The forecasting and routing models in
this repository run ashore; this is how data reaches them.

## Layout

```
gateway/     runs aboard (developed on a Raspberry Pi 3)
  ingest.py    four concurrent feed clients -> parse -> prioritise -> buffer
  buffer.py    durable SQLite queue (WAL), the store in store-and-forward
  sender.py    drains the queue to shore in priority order, ACK-gated
  run.sh       start|stop|status|reset

shore/       runs ashore
  shore_listener.py  receives, acknowledges, de-duplicates, optionally
                     forwards to the navigation dashboard
  run_all.sh         start|stop|status
```

The two halves share no imports, so they run on separate machines. Point the
gateway at the ship with `SHIP_HOST` and at the shore with `SHORE_HOST`.

## Priority tiers

Assigned at ingestion, so the sender never re-inspects payloads.

| Tier | Carries | Why |
|------|---------|-----|
| 1 HIGH | GPS `$GPRMC`, AIS positions, anything with `alerts` | Navigational and safety-critical; small payloads |
| 2 MED | Routine weather and engine telemetry | Useful, not urgent |
| 3 LOW | GPS `$GPGGA` | Same position as RMC plus fix-quality detail; redundant for navigation |

## Delivery guarantee

At-least-once, never at-most-once. A record is marked sent only after the shore
end returns `ACK <seq>`.

This is not belt-and-braces. A successful `sendall()` proves only that bytes
reached a local kernel socket buffer - under a dead link the write still
succeeds. Measured before the acknowledgement was added: **6 of 578 records
silently lost while the gateway reported success.** Shore de-duplicates on
`(source, seq)`, so a re-send after an outage is harmless.

## Running it

```bash
# ashore
cd shore && ./run_all.sh

# aboard
cd gateway && ./run.sh reset && ./run.sh

# inspect the queue at any time
cd gateway && python3 buffer.py          # totals by source and priority
cd gateway && python3 buffer.py --peek   # next 10 records to transmit
```

To forward telemetry into the navigation dashboard, give the shore listener the
API:

```bash
FORWARD_URL="http://<dashboard-host>:8600/api/telemetry/ingest" ./run_all.sh
```

That feeds the vessel marker and telemetry strip on the map. It is display only
and deliberately not an input to the models: telemetry is current, while the
processed ice archive ends 2018-12-31, so anything derived from both would pair
a real position with an eight-year-old environment.

## Degrading the link for testing

Shaping is done at the OS level with `tc`/`netem`, never faked in Python - the
point is that a real socket behaves badly.

Shape only the uplink port. A blanket rule on an interface also degrades the
ship's own instrument bus, which is physically wrong: aboard a real vessel the
internal NMEA network is unaffected by satellite weather.

```bash
# Iridium Certus-class, applied to loopback, uplink port only
sudo /sbin/tc qdisc add dev lo root handle 1: prio
sudo /sbin/tc qdisc add dev lo parent 1:3 handle 30: netem rate 32kbit delay 700ms loss 5%
sudo /sbin/tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 6000 0xffff flowid 1:3

# total outage, then restore, then remove
sudo /sbin/tc qdisc change dev lo parent 1:3 handle 30: netem loss 100%
sudo /sbin/tc qdisc change dev lo parent 1:3 handle 30: netem rate 32kbit delay 700ms loss 5%
sudo /sbin/tc qdisc del dev lo root
```

Never shape `wlan0`/`eth0` on a machine you reach over SSH.

At 32 kbit/s the link carries roughly 20 records/second, so a steady throttle
alone will not build a backlog at ordinary instrument rates. Backlog growth
comes from the outage, not the throttle.

## Data sources

There are none in this folder. The four simulated instrument emitters that
originally drove it were removed, because they generated GPS, AIS, weather and
engine values from trig functions and noise, and keeping them invites the
pipeline being described as carrying real telemetry.

What remains was never simulated: the parsers (NMEA 0183 with checksum
validation, JSON), the queue, the priority scheme and the acknowledgement
protocol. Point `ingest.py` at a real instrument feed and nothing downstream
changes.
