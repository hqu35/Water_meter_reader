<!-- recursive-summary:start -->
## Directory overview

Turn the verified Dryad image/mask archive into a reproducible, reviewed YOLO register-localization dataset.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |
| [\_\_init\_\_.py](__init__.py) | Declares the data-preparation subpackage. |
| [archive.py](archive.py) | Pairs exact official split/parent/filenames, decodes RGB images and binary masks, checks dimensions/encoding and writes an inspection report. |
| [download.py](download.py) | Streams or reuses the original frozen Dryad archive and enforces its configured SHA-256 before use. |
| [preparation.py](preparation.py) | Computes half-open mask bounding boxes, normalizes YOLO coordinates, preserves image/mask bytes and creates deterministic test/validation/train records with duplicate/empty-mask exclusions. |
| [review.py](review.py) | Selects 32 review examples, renders original/mask/box contact sheets, records explicit approval and verifies that reviewed records/files remain unchanged. |
| [validation.py](validation.py) | Checks serialized coordinate round trips and foreground coverage; verifies image/label/mask hashes, split uniqueness and exact directory membership. |

### Summary

The data flow is archive verification → exact pairing → seeded subset/export → review sheets → explicit approval. The training gate rechecks the approved bytes and directory contents, so generated image/label/mask leaves must not receive extra documentation files.

<!-- recursive-summary:end -->
