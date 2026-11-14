# Water Meter Orientation Classifier

This phase classifies a geometrically rectified **full water-meter image** as upright or upside down. Stage I first locates the register and rotates the full frame so the register is horizontal. ViT-Small then resolves the remaining 0°/180° ambiguity. The model takes the full frame, including the meter housing, before any orientation correction or register crop. This experiment does not change the production pipeline.

## Where to start

| Path | Purpose |
| --- | --- |
| [`vit_orientation/`](vit_orientation/README.md) | Model, preprocessing, dataset preparation, training, evaluation, and review utilities. |
| [`dataset_2000/`](dataset_2000/README.md) | Historical gallery package of 2,000 register crops, including four rejected examples. Its manifest connects crop IDs, orientation marks, and source evidence. These crops are not the full-frame ViT inputs. |
| [`PROVENANCE_AUDIT.md`](PROVENANCE_AUDIT.md) | Explains how historical Stage I rotations and saved classifier decisions relate crop labels to full-frame inputs. |
| `outputs/` | Local datasets, review galleries, checkpoints, metrics, and logs. This directory and `dataset_2000/images/` are ignored by Git. |

## Completed ViT-Small run

The validation-selected checkpoint is [`best.pt`](outputs/vit_small_orientation_20261112/best.pt) from epoch **6**. Training ran locally on Apple MPS and stopped after 11 epochs. The frozen, grouped split used 2,394 training, 513 validation, and 513 test image files. Each file contributes its original orientation and a 180° counterpart; the pair stays in the same split.

- Best validation accuracy: **98.6355%**.
- Held-out test accuracy: **99.7076%** (1,023 of 1,026 orientation samples, from 513 files and 359 source groups).
- See [the run report](outputs/vit_small_orientation_20261112/RESULTS.md) for the confusion matrix, training curves, latency measurements, and reproducibility details. The [frozen split manifest](outputs/datasets/vit_small_reviewed_20261112_v2/split_manifest.json) and [training log](outputs/logs/vit_small_orientation_20261112.log) are local artifacts.

These metrics describe this experiment. The combined training dataset uses reviewed Dryad-derived full frames and a Roboflow v3 release that already contained resized images and augmented variants. Model scores are not calibrated probabilities.

## Separate 100-image review

Open the [unseen-ID review gallery](outputs/unseen_orientation_100_20261113/index.html) to inspect 100 Dryad images sampled after excluding previously traced IDs. The gallery displays the geometrically rectified full frames **before** the classifier's correction. All 100 were processed; the model predicted **73 normal** and **27 upside down**. Those are prediction counts. The [pilot summary](outputs/unseen_orientation_100_20261113/summary.json) does not report human-verified accuracy.

## Run inference

From this directory, with the dependencies in [`vit_orientation/requirements.txt`](vit_orientation/requirements.txt) installed:

```python
from PIL import Image
from vit_orientation.model import OrientationPredictor

predictor = OrientationPredictor(
    "outputs/vit_small_orientation_20261112/best.pt", device="cpu"
)
with Image.open("rectified_full_meter.png") as image:
    angle, confidence = predictor.predict_orientation(image)
    print(angle, confidence)  # 0 = upright; 180 = upside down
```

Use `predictor.correct_orientation(image)` to return a copy or a 180° rotation at the original resolution. For a new training run, follow the [training setup](vit_orientation/README.md): prepare a verified, grouped split, launch it with `vit_orientation/launch.sh`, then evaluate the selected checkpoint separately with `python -m vit_orientation.evaluate outputs/<run_name>`.
