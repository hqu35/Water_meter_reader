<!-- recursive-summary:start -->
## Directory overview

Small shared file operations used across the worker, without model or dataset-selection logic.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |
| [\_\_init\_\_.py](__init__.py) | Declares the shared-utilities subpackage. |
| [io.py](io.py) | Reads/writes JSON, hashes files in bounded chunks and rejects ZIP paths escaping an extraction root. |
| [packaging.py](packaging.py) | Copies audit metadata into an experiment snapshot, records artifact hashes and creates stage1_results.zip and its retrieval report without bundling the image dataset. |

### Summary

These helpers keep serialization, hashing, extraction and result packaging consistent across local and Colab operations. They neither choose model hyperparameters nor perform training.

<!-- recursive-summary:end -->
