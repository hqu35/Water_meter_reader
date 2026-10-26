<!-- recursive-summary:start -->
## Directory overview

Active Stage I package for one-class register localization, from reviewed real data through training, held-out evaluation and single-photo prediction. Run it from the repository root with python3 -m stage1_python_worker.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |
| [\_\_init\_\_.py](__init__.py) | Declares the Stage I package and its purpose. |
| [\_\_main\_\_.py](__main__.py) | Single executable package entry point; delegates to cli.main(). |
| [cli.py](cli.py) | Defines local worker phases plus predict and colab commands, parses arguments and dispatches each operation once. |
| [colab.py](colab.py) | Synchronizes this package to the existing named session, transfers the approved subset, invokes the same worker remotely and fetches verified results. |
| [config.py](config.py) | Loads configs/stage1.json, derives repository/data/output paths and the actual package name, writes the YOLO dataset definition and inventories source files. |
| [inference.py](inference.py) | Loads an existing checkpoint for one real photograph, saves predicted boxes/confidences and crops the highest-confidence register. |

### Subdirectories

| Directory | Role |
|---|---|
| [data/](data/README.md) | Archive inspection, deterministic conversion/export, review sheets and the mandatory approval gate. |
| [evaluation/](evaluation/README.md) | Held-out AP evaluation, fixed-threshold matching and prediction visualizations. |
| [training/](training/README.md) | T4 verification, unchanged YOLO training arguments and checkpoint display-name normalization. |
| [utils/](utils/README.md) | Shared JSON/checksum/ZIP operations and result packaging. |

### Summary

The CLI coordinates focused modules rather than implementing a second pipeline. Preparation exports data and audit records; training requires valid visual approval; validation selects the best checkpoint; evaluation reports held-out performance. Colab is transport/execution infrastructure, and inference consumes the resulting checkpoint.

<!-- recursive-summary:end -->
