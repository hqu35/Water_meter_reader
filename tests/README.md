<!-- recursive-summary:start -->
## Directory overview

Regression checks use tiny generated fixtures and mocked model/Colab calls rather than training or contacting a live GPU.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |
| [stage1\_fixtures.py](stage1_fixtures.py) | Shared ZIP and reviewed-dataset fixture builders, including JPEG bytes behind .png names, empty masks and duplicate candidates; these are test-only inputs. |
| [test\_stage1\_colab.py](test_stage1_colab.py) | Simulates source sync, subset upload/restore, remote package launch and result retrieval; checks chunk integrity, unsafe ZIP paths and reviewer requirements. |
| [test\_stage1\_data.py](test_stage1_data.py) | Checks mask boxes, YOLO coordinates, exact pairing, seeded split/review selections, empty/duplicate replacement and approval invalidation. |
| [test\_stage1\_execution.py](test_stage1_execution.py) | Checks CLI dispatch, frozen training/evaluation arguments, IoU matching, checkpoint-normalization control flow, result packaging and single-image cropping. |

### Subdirectories

| Directory | Role |
|---|---|
| [fixtures/](fixtures/README.md) | Captured pre-refactor argument/selection contracts and their provenance explanation. |

### Summary

Run python3 -m unittest discover -s tests -v from the repository root. The suite catches changes to workflow contracts and file handling; it does not establish real T4 numerical equivalence or replace manual annotation inspection.

<!-- recursive-summary:end -->
