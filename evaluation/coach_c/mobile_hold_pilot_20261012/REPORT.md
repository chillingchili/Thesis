# Bounded automatic-settling pilot

Eight preselected failure/control clips; one replay each.

```json
{
  "baseline": {
    "clips": 8,
    "status": {
      "NO_OUTPUT": 3,
      "CORRECTIVE": 2,
      "PARTIAL": 2,
      "INSUFFICIENT_DATA": 1
    },
    "coaching": 2,
    "no_output": 3,
    "late": [
      [
        "C005",
        3126
      ]
    ]
  },
  "hold_650ms": {
    "clips": 8,
    "status": {
      "NO_OUTPUT": 3,
      "CORRECTIVE": 2,
      "PARTIAL": 2,
      "POSITIVE": 1
    },
    "coaching": 3,
    "no_output": 3,
    "late": [
      [
        "C005",
        3106
      ]
    ]
  },
  "decision": "Not adopted: one additional positive output, unchanged three missing results, 3.106-second late result, no independent validity or repeatability evidence. Default remains 350ms."
}
```

The 30-clip primary mobile development result remains 18 corrective outputs. This selected pilot is not a new held-out evaluation.
