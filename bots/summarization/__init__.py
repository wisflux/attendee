"""Meeting title/summary generation (see docs/meeting-summaries).

Pure prompt/formatting/parsing helpers live here so they can be unit-tested without a network; the
only I/O is the thin Azure call in azure_client.summarize.
"""
