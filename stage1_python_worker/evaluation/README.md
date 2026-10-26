<!-- recursive-summary:start -->
## Directory overview

Evaluate the validation-selected checkpoint on held-out real images and show both ordinary and difficult predictions.

### Files

| File | Role |
|---|---|
| [README.md](README.md) | Directory documentation and this managed overview. |
| [\_\_init\_\_.py](__init__.py) | Declares the evaluation subpackage. |
| [evaluator.py](evaluator.py) | Runs test AP at low confidence, bounds prediction batches and verifies filename ordering; separately computes IoU matching and counts at confidence 0.25 / IoU 0.50. |
| [visualization.py](visualization.py) | Draws green ground-truth/red prediction boxes for eight worst-IoU examples and eight seeded random examples. |

### Summary

The evaluation produces heldout_metrics.json, test_predictions.json, operating_point.json and prediction sheets. AP/PR summaries and the fixed-threshold operating-point counts are distinct outputs; the test set does not select the checkpoint.

<!-- recursive-summary:end -->
