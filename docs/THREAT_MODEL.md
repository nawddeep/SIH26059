# Threat model — telemetry gateway

The gateway accepts data from shipboard instruments and forwards it ashore over
a link it does not control. This is what it defends against, what it does not,
and which test proves each claim.

**Scope:** `telemetry-gateway/`. **Not in scope:** the route planner or the
models, which never take telemetry as an input.

## Trust boundaries

```
  ship's instrument bus          the link            shore
 ┌───────────────────┐                        ┌──────────────┐
 │  TRUSTED          │ ──────────────────────►│  TRUSTED     │
 │  (physical access │      UNTRUSTED         │  (operator   │
 │   to the vessel)  │      hostile, lossy    │   controlled)│
 └───────────────────┘                        └──────────────┘
```

The instrument bus is trusted because anyone who can inject onto it already has
physical access to the ship — at which point the telemetry is not the problem.
The **link is untrusted**: narrowband, lossy, and not confidential.

## Threats, mitigations, tests

| # | Threat | Mitigation | Test |
|---|---|---|---|
| 1 | **Malformed / fake NMEA** | XOR checksum validated on every sentence; failures dropped, never guessed at. Malformed JSON caught per line, the feed continues | `integration/test_end_to_end.py` |
| 2 | **Duplicate packets** | Shore de-duplicates on `(source, seq)`; `seq` is the queue row id, monotonic and never reused | `test_shore_listener_acknowledges_and_deduplicates` |
| 3 | **Replay attack** | ⚠️ **Not mitigated.** Sequence numbers detect *accidental* repeats, not a deliberate replay by an attacker who can write to the link. See gaps below | — |
| 4 | **Packet corruption in transit** | TCP checksums catch transport corruption; NMEA checksums catch bus corruption; JSON parse failures are counted and skipped | `test_shore_wire_format_reaches_backend_state` |
| 5 | **Connection loss mid-transmission** | Records are marked sent only after `ACK <seq>`. Anything unacknowledged stays queued and is re-sent | `test_unacknowledged_records_are_not_marked_sent` |
| 6 | **Sender process crash** | SQLite WAL; `sent_flag` set only post-ACK. Verified by `SIGKILL` mid-flight: zero loss, duplicates de-duplicated at shore | measured, §Verified |
| 7 | **Shore unreachable or crashed** | Sender retries with exponential backoff (1→15 s) and never blocks ingestion. The buffer grows; nothing is dropped | measured, §Verified |
| 8 | **Buffer growth without bound** | `QUEUE_MAXSIZE = 10,000` in-flight records; beyond that, records are dropped **and counted**, because an OOM kill on a 1 GB Pi loses everything | `test_buffer_preserves_priority_order` |
| 9 | **Slow consumer stalling an instrument** | Emitter writes use a 2 s drain timeout; a stalled client is dropped rather than stalling the feed | — |
| 10 | **Unauthorised sender to shore** | ⚠️ **Not mitigated.** The shore listener accepts any TCP connection. See gaps | — |
| 11 | **Eavesdropping** | ⚠️ **Not mitigated.** Plaintext JSON over TCP | — |
| 12 | **Disk exhaustion from the queue** | ⚠️ **Partially mitigated.** The queue is bounded in memory but rows accumulate on disk during a long outage; no retention policy | — |

## Verified by measurement

Two claims are measurements, not assertions:

**No loss across a total link outage.** With `netem loss 100%` applied to the
uplink port, the backlog grew to 73 records while ingestion continued
undisturbed. On restore the gateway's `sent` count matched shore's unique record
count exactly; the extra arrivals were re-sends and were de-duplicated.

**No loss across an uncontrolled process kill.** `SIGKILL` on the sender mid-flight
with a 37-record backlog: after restart, gateway 370 marked sent, shore 371
unique. Shore held **one more** than the gateway claimed — a record that arrived
while its ACK was in flight. That asymmetry is the correct direction: erring
toward duplication, never toward loss.

**A measured failure, for the record.** Before acknowledgement was added, the
gateway marked records sent on a successful `sendall()`. That call only proves
bytes reached a local kernel socket buffer — under a dead link it still succeeds.
Measured: **6 of 578 records silently lost while reporting success.** This is why
delivery is ACK-gated.

## Known gaps

Stated rather than glossed, because a threat model that lists only solved
problems is marketing.

1. **No authentication.** Shore accepts any connection that can reach port 6000.
   An attacker on the network can inject fabricated positions.
   *Mitigation if deployed:* mutual TLS, or a pre-shared key per vessel.
2. **No confidentiality.** Plaintext. Vessel position is operationally sensitive.
   *Mitigation if deployed:* TLS, accepting the handshake cost on a 700 ms link.
3. **No replay protection.** Sequence numbers are per-gateway-session and reset
   on a database wipe, so a replayed capture would be accepted after a reset.
   *Mitigation if deployed:* monotonic timestamps plus a signature.
4. **No integrity guarantee.** Nothing detects an attacker modifying a record in
   transit — only accidental corruption.
   *Mitigation if deployed:* HMAC per record, cheap enough at this data rate.
5. **No disk retention policy.** A long outage grows `buffer.db` without bound.
6. **The instrument bus is trusted implicitly**, consistent with how real NMEA
   0183 works — it has no authentication either.

## Why these gaps are acceptable here

This is a prototype demonstrating **reliable delivery over a degraded link**, not
a hardened deployment. The gaps are all in confidentiality, authenticity and
retention — orthogonal to the delivery guarantee, and each addressable with
standard transport security without redesigning the buffer or the priority
scheme.

Deploying this on a real vessel without items 1–4 would be irresponsible.
