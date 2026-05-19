# Discover Model Settings

A bootstrap skill the agent runs on itself the first time a new model is seen.

## What it does

For the currently-loaded model on the configured LM provider, this skill probes a
small candidate set of `reasoning_effort` values (`None`, `"off"`, `"low"`,
`"medium"`, `"high"`) by sending a fixed canary prompt to the OpenAI-compatible
`/v1/chat/completions` endpoint. For each candidate it records:

- whether the call succeeded (HTTP 2xx),
- elapsed wall-clock seconds,
- `completion_tokens_details.reasoning_tokens` (LM Studio reports this; the
  field is also defined in the OpenAI spec for reasoning models).

It then picks the lowest-reasoning value that the model actually honored —
i.e. the smallest `reasoning_tokens` among successful candidates — and
persists the result to `sessions/<model-slug>/model_settings.ipynb` as a
Papermill-parameterized notebook whose `parameters` cell *is* the canonical
record. Re-executing that notebook re-validates the settings.

## Why no `dspy.LM`

This skill must work **before** the agent has good LM settings (that's what
it's trying to discover). It therefore uses raw `urllib` HTTP requests with
a short per-request timeout, never `dspy.LM`. Each probe has a hard wall
clock so a hung thinking model can't stall discovery.

## Outputs

- `outputs/result.json` — full probe results + recommended settings.
- `sessions/<model-slug>/model_settings.ipynb` — Papermill notebook whose
  parameters cell holds the recommended `reasoning_effort` (and the
  `supports_reasoning_effort` boolean). `init_notebook()` reads this file
  on startup.
