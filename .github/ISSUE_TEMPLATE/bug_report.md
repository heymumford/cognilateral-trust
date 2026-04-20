---
name: Bug report
about: Something broken, producing the wrong result, or failing silently
title: ''
labels: bug
assignees: ''
---

## What happened

_One or two sentences. What did you expect, what did you get._

## Reproduction

```python
# Minimal code that reproduces the bug
from cognilateral_trust import evaluate_trust

result = evaluate_trust(0.7)
# ...
```

Or, if the bug is in the CLI / hosted API:

```bash
$ trust-check 0.7
# actual output
```

## Environment

- `cognilateral-trust` version: (output of `pip show cognilateral-trust | grep Version`)
- Python version: (output of `python --version`)
- OS: POSIX / Windows
- Installed via: pip / uv / poetry / other

## Logs / Traceback

```
paste here
```

## What I already tried

_Optional. If you already ruled something out, say so — saves a round-trip._
