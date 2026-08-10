"""System prompt + transcript/context formatting for meeting summarization.

The SAME prompt is used for bot meetings and local recordings (it already handles missing attendee
lists and uncertain speakers); only the context we feed differs — a bot meeting has real participant
names, a local recording has "You"/system-audio labels. Everything here is pure (no I/O), so it is
unit-tested directly. See docs/meeting-summaries.
"""

# Longest transcript (in characters) we send to the model. Longer inputs are truncated head+tail so
# a very long meeting cannot overflow the context window or the cost budget.
TRANSCRIPT_CHAR_BUDGET = 48000
# Below this a transcript is too short to be worth summarizing (silent/empty/test recordings).
MIN_TRANSCRIPT_CHARS = 200
_TRUNCATION_MARKER = "\n\n[... transcript truncated for length ...]\n\n"

# The meeting-notes instruction, verbatim. The machine-readable output envelope is added separately
# (OUTPUT_CONTRACT) so this stays the single source of truth for *what* to produce.
NOTES_INSTRUCTION = """# Meeting Notes Extraction Prompt

You are given a meeting transcript (which may be from an automatic speech recognition system and may contain garbled names, mixed languages, or transcription errors).

Produce **one structured notes document** from this transcript.

## Rules

- **Be exhaustive** — lose no detail from the transcript. Prefer over-inclusion to omission.
- **Base notes strictly on transcript content. Never invent facts.**
- **Speaker identification is required.** Make your best effort using any attendee list provided, email domains, transcript cues, and content. Where uncertain, mark it *(speaker uncertain)*.
- **ASR artifacts** — if names or words are clearly garbled, note the likely intended word in brackets, e.g. "Kawika" [ASR: "Kamika"]. Never change the transcript words, only annotate.
- Order topic sections roughly by when each topic first arose, so the document loosely mirrors the conversation's flow.
- Where chronology matters within a topic (a position changed, a decision was reversed, an objection redirected the discussion), note that evolution inline — e.g., "*initially proposed X, revised to Y after [person]'s concern about Z*."
- Omit any sub-section that has no content (don't write "None"). Exception: the three consolidated lists at the end — if one is empty, state "None recorded" so it's clear nothing was missed.

## Response format

Follow this structure exactly:

```markdown
# Meeting Notes — [meeting title/subject], [date if known]

## Speakers
| Speaker | Identified as | Basis | Confidence |
|---|---|---|---|
| Speaker 1 | [Name, role if known] | [attendee list / email domain / transcript cue / content] | High / Medium / Low |

Unresolved: [any voices you could not attribute]

## Meeting Flow
[5–10 lines, chronological: phases of the discussion, who raised what,
major turns or pivots. The narrative arc — no details, those go below.]

## Topics

### 1. [Topic name]
**Context / discussion**
- [Detail — attributed: "Name said/argued/asked..."]

**Decisions**
- [Decision — who made it, and any conditions attached]

**Proposals**
- [Proposal — by whom, status: accepted / rejected / parked]

**Action items**
- [ ] [Owner] — [task] — [deadline if stated, else "no deadline stated"]

**Open questions**
- [Question — raised by whom, why it's unresolved]

### 2. [Next topic]
[...same sub-structure...]

## Consolidated Summary

### All decisions
1. [Decision] *(Topic N)*

### All action items
| Owner | Task | Deadline | Topic |
|---|---|---|---|

### Open questions / parked items
1. [Item] *(Topic N)*
```"""

# Wraps the notes document in a JSON envelope so we can extract a short title and the notes body
# reliably. Mentions "JSON" explicitly, which Azure requires when response_format is json_object.
OUTPUT_CONTRACT = """## Output

Return ONLY a JSON object with exactly two keys, and nothing else — no code fences, no commentary:

- "title": a single line of 10-20 words (never more than 20) that best defines this meeting.
- "notes": a string containing the full markdown notes document exactly as specified above."""

SYSTEM_PROMPT = NOTES_INSTRUCTION + "\n\n" + OUTPUT_CONTRACT


def format_transcript(utterances):
    """Render utterances as ``Speaker: text`` lines, preserving the given order.

    utterances: iterable of dicts with ``speaker_name`` (may be blank/None) and ``text``. Blank
    speakers become "Unknown speaker"; empty lines are dropped.
    """
    lines = []
    for utterance in utterances:
        speaker = (utterance.get("speaker_name") or "").strip() or "Unknown speaker"
        text = (utterance.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def truncate_transcript(text, budget=TRANSCRIPT_CHAR_BUDGET):
    """Keep the head and tail within ``budget`` so both the opening and the wrap-up survive."""
    if len(text) <= budget:
        return text
    keep = (budget - len(_TRUNCATION_MARKER)) // 2
    return text[:keep] + _TRUNCATION_MARKER + text[-keep:]


def is_long_enough(transcript_text):
    """True when there is enough transcript to be worth summarizing."""
    return len((transcript_text or "").strip()) >= MIN_TRANSCRIPT_CHARS


def build_context_block(context):
    """A short "Meeting context" header from whatever fields are known (all optional).

    A list/tuple value (e.g. participants) is joined into a comma-separated string so the model
    never sees Python's ``['A', 'B']`` repr.
    """
    labels = [("title", "Title"), ("date", "Date"), ("duration", "Duration"), ("platform", "Platform"), ("participants", "Participants")]
    lines = []
    for key, label in labels:
        value = context.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        lines.append(f"{label}: {value}")
    return "Meeting context:\n" + "\n".join(lines) if lines else ""


def build_user_message(utterances, context=None):
    """The user turn: an optional context header followed by the (truncated) transcript."""
    transcript = truncate_transcript(format_transcript(utterances))
    header = build_context_block(context or {})
    if header:
        return f"{header}\n\n---\n\nTranscript:\n{transcript}"
    return f"Transcript:\n{transcript}"
