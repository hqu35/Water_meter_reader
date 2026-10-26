<!-- recursive-summary:start -->
## Directory overview

Experiment settings are kept separate from Python implementation.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |
| [stage1.json](stage1.json) | Defines the existing cs539-stage1 session, real-data subset sizes (2,400/400/800), seed 20261022, YOLO11n/AdamW training settings and frozen Dryad archive identity/checksum. |

### Summary

The worker reads this JSON through its config module. Changing this file changes the experiment definition; the recursive-summary workflow only documents it.

<!-- recursive-summary:end -->
