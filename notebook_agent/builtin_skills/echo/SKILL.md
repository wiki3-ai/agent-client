# Echo Skill

## Purpose

Echo the input `message` back unchanged. This is the canonical built-in skill
used by the notebook agent's end-to-end and acceptance tests.

## Inputs

- `message` (string, required): the text to echo.

## Outputs

Writes `outputs/result.json` containing:

```json
{"message": "<the same message>"}
```

## Behaviour

The skill takes a single parameter `message` and writes a result whose
`message` field equals the input. No external services are required.
