# Local recording — design notes

Why the migrations exist, and how audio actually travels from the desktop to a transcript.
Written for someone picking this up cold.

---

# 1. The three migrations — why each exists

*(A fourth, `alter_bot_session_type`, simply registers the `LOCAL` session type.)*

## `alter_botevent_event_type` — the "Local Session Ended" event

### What is an "enum value"?

An **enum** is a fixed list of allowed named values, each stored as a small number.

```
BOT_PUT_IN_WAITING_ROOM = 3
BOT_JOINED_MEETING      = 4
...
LOCAL_SESSION_ENDED     = 104   ← the one we added
```

The **database stores just the number** (`104` — tiny and fast to index), while the **code reads
the name** (`LOCAL_SESSION_ENDED`). You get readable code and a compact, valid-by-construction
column: nobody can write `"local sesion ended"` with a typo, because only declared values exist.

### How states actually change

A bot's life is a **state machine**. You **cannot** simply set a state. You create an **event**,
and a table in the code decides whether that event is legal:

```
VALID_TRANSITIONS = {
    LOCAL_SESSION_ENDED: { from: READY,  to: ENDED },
    ...
}
```

Meaning: *"this event may only be applied to a session in READY, and it moves it to ENDED."*
Try it from any other state and it is rejected. That is what stops a session going from
"deleted" back to "recording".

### Who declares events?

Whoever witnesses the thing happening:

| Who | Events they create |
|---|---|
| The **bot pod** (running in the cloud) | "I'm joining", "I'm in the waiting room", "I left the meeting" |
| The **cleanup cron** | `FATAL_ERROR` when a bot stops heart-beating |
| The **API** | when you dispatch a bot |

### And for a local session?

**There is no pod** — nothing is running in the cloud to report anything. So the event is created
by **our own server code**: the `finalize_local_session` task, which runs when either

1. the user presses **Stop**, or
2. the **idle reaper** decides the session was abandoned (app crashed, laptop closed).

Both paths call the same function, which creates `LOCAL_SESSION_ENDED` and moves the session
**`READY` → `ENDED`**.

### What other states can a local session have?

**Only three, ever** — verified by walking the transition table:

| State | Meaning |
|---|---|
| `READY` | live — currently recording |
| `ENDED` | finished |
| `DATA_DELETED` | deleted |

It can **never** reach `JOINING`, `WAITING_ROOM`, `FATAL_ERROR` or the other 16 — those all
require events only a bot pod can create. That is why we map to friendly words (`recording`,
`paused`, `processing`…) using a *second* field, the recording's own state.

> **The problem it solves, in one line:** without this event there was no legal way to finish a
> local session — and since deletion requires a finished session, **users could not delete their
> own recordings.**

## `bot_owner_user_id` — the owner column

### Why nullable?

1. **Old rows can't have an owner.** There are thousands of existing bots and nobody knows who
   created them. A `NOT NULL` column would demand a value for every one.
2. **Some bots legitimately have no owner.** API customers dispatch without a user token — those
   meetings belong to a company integration, not a person.
3. **It's instant.** In Postgres, adding a *nullable* column with no default is a
   **metadata-only** change — no table rewrite, no lock. Adding `NOT NULL` with a default can
   rewrite the entire table and block writes while it does.

### What it's for, simply

It is a **name tag on the meeting** saying *"this belongs to user 60."*

### How it pulls a person's transcripts together

The data is a tree:

```
Bot (the meeting)        ← we stamp owner_user_id HERE
 └── Recording
      └── Utterances     ← the actual transcript lines
```

Transcripts don't need their own owner tag, because **you can only reach them through the
meeting**. Stamp the top of the tree and everything beneath is automatically scoped: *"give me
meetings where owner = 60"* → and only those transcripts are reachable. One column protects the
whole subtree.

> **Why it's needed:** without it the server literally cannot answer *"which meetings are mine?"*
> — the information doesn't exist anywhere in the schema. That is the whole basis of cross-device
> history.

## `replace_bot_owner_index` — the sorted index

### What an index is

Think of a book's index: instead of reading all 500 pages to find "photosynthesis", you check a
pre-sorted list and jump straight there. A database index is the same — a **small, pre-sorted
copy** of a few columns plus a pointer to the full row.

### Why the old one wasn't enough

Our query is always: **"my meetings, newest first, 25 per page."** That is *two* jobs — **find**
yours, and **sort** them by date. The old index only helped with finding, so Postgres loaded
**all 1,200** of a user's meetings, sorted them, and threw away all but 25.

Measured on 6,000 rows:

| | owner-only index | composite index |
|---|---|---|
| Rows actually read | **1,200** | **25** |
| Disk pages touched | 119 | **6** |
| Execution | 0.906 ms | 0.204 ms |

The new index stores rows **already sorted by owner, then date** — so the database jumps to your
section and reads 25 rows in order. **No sorting, and the other 1,175 are never touched.** The
real point is not today's speed: the old plan gets slower every time you record, the new one
stays flat at 10 meetings or 10,000.

### Where `created_at` and `id` come from

Both **already exist** — we are not collecting anything new:

* **`created_at`** — Django stamps it automatically the moment a row is created
* **`id`** — the auto-incrementing primary key every row already has

We are just telling the index to keep them in sorted order.

### Why *both*, and why `DESC`?

* **`created_at DESC`** = newest first, which is how history should open.
* **`id DESC`** = the **tiebreak**, and it is not optional.

Paging works by remembering "you stopped at this meeting; give me the next 25 after it." If
**two meetings share the exact same `created_at`**, the database has no way to know which came
first — so the same row can appear on **two pages**, or be **skipped entirely**. Since `id` is
unique, adding it makes the order completely unambiguous.

---

# 2. The audio path, and how failures are handled

## How the desktop knows to retry

**Every chunk gets an explicit answer.** The upload is a normal HTTP request:

* **`202`** → safely stored, move on
* **anything else** — timeout, network drop, 5xx → **not confirmed**, so resend that same chunk
  with the **same `sequence` number**

If the first attempt actually *did* arrive and only the reply was lost, the server sees a
`sequence` it has already processed and **throws the duplicate away**. So retrying is always
safe, and the desktop never has to know whether the original made it.

> This is precisely why we chose chunked HTTP over a WebSocket — HTTP gives you a per-chunk
> receipt for free. With a socket you would have to build that acknowledgement system yourself.

## Retries at every layer

| Stage | Protection |
|---|---|
| Desktop → server | Desktop retries the same chunk; server drops duplicates by `sequence` |
| Server → Redis | Chunk is stored before returning `202`; the response means "safely queued" |
| Drain task | `max_retries=3`, exponential backoff |
| Finalize task | `max_retries=5` |
| Transcription | `max_retries=6`, backoff, and on final failure writes `failure_data` |

**If a chunk never arrives at all:** the next chunk carries a higher `offset_ms`, so the server
sees a **gap** and fills it with **actual silence** before the speech detector. The timeline stays
correct and the two sides of the gap are not glued into one run-on sentence. That audio is lost,
but the transcript is not corrupted.

**If transcription fails permanently:** after 6 attempts the error is written to `failure_data`
and the meeting reports **`partially_failed`** — the rest of the transcript stays usable.

**Two honest gaps:**

1. If the drain task exhausts its 3 retries, that batch stays in Redis until its 1-hour TTL and
   no utterance is ever created. Since no utterance exists, nothing is "pending" — so the status
   reads **done** rather than flagging a problem.
2. If Redis is lost, queued-but-unprocessed audio goes with it. Anything already turned into an
   utterance is safe in Postgres.

## Where the audio actually goes

```
1. Desktop  ──HTTPS──►  Attendee web server (gunicorn)
2. Web      ──────────►  Redis:  stores the chunk in a queue
3. Web      ──────────►  Redis:  drops a "job ticket"
4. Web      ──202──────►  Desktop        ← returns immediately, ~95 ms
                          (recording never waits on transcription)
5. Worker   ◄─────────   Redis:  picks up the job ticket
6. Worker   ◄─────────   Redis:  reads the queued chunks
7. Worker   ──────────►  cuts sentences → saves → sends to ElevenLabs
```

**Redis is doing two separate jobs here:**

1. **The parking lot** — holds uploaded audio until a worker is ready, plus the "tail" (the
   half-finished sentence carried between chunks).
2. **The noticeboard** — Celery's message broker. The web server pins up a job ticket; workers
   watch for tickets.

**Where do the workers come from?** They are **separate long-running processes** — a different
container/deployment (`attendee-worker`) from the web server. They do nothing but watch Redis for
tickets and execute them. That separation is the point: the web server stays free to accept
uploads while workers do the slow work of transcription.

## The upload contract

```
POST /api/v1/local_sessions/{session_id}/audio
Authorization: Token <project api key>
X-User-Token:  <team.day jwt>
Content-Type:  multipart/form-data

source=mic|system   sample_rate=16000   offset_ms=3000   sequence=3   file=<raw bytes>
```

Audio format — raw, no container:

* **16-bit PCM, mono, little-endian, no WAV header**
* sample rate ∈ **8000 / 16000 / 32000 / 48000** (16 kHz is what we use)
* must be an **even** number of bytes (a whole number of samples)
* max **1,000,000 bytes** per chunk; ~1 second at 16 kHz is 32 KB

## What `offset_ms` is, and why it exists

**`offset_ms` = how many milliseconds into the recording this chunk starts.** First chunk `0`,
next `1000`, next `2000`.

Three jobs:

1. **It refuses to trust the client's clock.** If we used a real timestamp and someone's laptop
   clock was an hour off, every word would land at the wrong time. A relative offset is correct
   even on a badly-set machine.
2. **It reveals gaps.** If one chunk ends at 5000 ms and the next starts at 9000 ms, the server
   *knows* 4 seconds are missing and fills them with real silence — so the speech detector sees a
   genuine pause instead of gluing two separate sentences together.
3. **It positions the transcript.** It is how each line gets its place on the timeline, and how
   resuming after a pause continues from exactly the right point.
