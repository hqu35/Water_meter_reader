# ViT-Small full-meter orientation experiment

**Status: training and held-out evaluation complete. See `../outputs/vit_small_orientation_20261112/RESULTS.md` for measured results and limitations.** The reviewed `dataset_2000` contains register crops. Recovered Dryad IDs locate the original images, but do not establish the rotation relating a reviewed crop to a rectified full-meter image. The existing integration outputs retained only 10 `corrected_full.png` files, with 2 overlapping the selection. The original images and masks were recovered by ID and used with the existing Stage I rectification to reconstruct 1,996 full images in `outputs/reconstruction/images/`. A crop-to-frame 0/180 template comparison proposes the relative polarity; all results remain review-required and are recorded in `outputs/reconstruction/proposals.jsonl`. Matching scores are not calibrated, and source mappings include near-identical matches. Do not treat these proposals as verified labels without validation.

Roboflow v3 was downloaded to `outputs/datasets/roboflow-orientation-v3.zip` (ignored by Git, SHA-256 `cf19c204c15453b7adaba795cc251352cbf32ceae2f1bd99c0cfaa1d0502e4fb`). Its own README says images were stretched to 640 x 640 and brightness/exposure variants were generated before export. Thus the v3 archive is not unaugmented, aspect-ratio-preserving original data. Filename-based grouping suggests variants of the same source cross its supplied splits; this heuristic needs review. Source images and grouping provenance are required for the strict protocol.

## Required input

Provide a CSV with `path,label,source,group_id,label_evidence`. Every path must identify the **full geometrically rectified meter image**, with a verified label applying to that specific image (`0` correct, `1` upside down). `source` must include `dryad` and `roboflow`. `group_id` must identify the underlying original image globally across both sources, so all variants of it have the same group even across datasets. Exclude unknown labels. Do not use `dataset_2000` crops as full images. The script checks fields, hashes, and grouping; it cannot independently verify labels, full-frame content, or source identity.

From `Phase_orientation_classifier` with a native arm64 Python environment:

```
python -m venv .venv
.venv/bin/pip install -r vit_orientation/requirements.txt
.venv/bin/python -m vit_orientation.prepare verified_full_images.csv outputs/splits/run01.json
PYTHON_BIN="$PWD/.venv/bin/python" vit_orientation/launch.sh outputs/splits/run01.json vit_run01
```

`launch.sh` starts training under `nohup`, writes the PID and a persistent log path, and refuses to reuse a run name. Use `tail -f outputs/logs/vit_run01.log` to monitor, or `kill PID` to stop. It does not automatically evaluate the test split. To evaluate the selected checkpoint once after training:

```
.venv/bin/python -m vit_orientation.evaluate outputs/vit_run01
```

Inference:

```python
from PIL import Image
from vit_orientation.model import OrientationPredictor
predictor = OrientationPredictor('outputs/vit_run01/best.pt')
image = Image.open('full_meter.png')
orientation, confidence = predictor.predict_orientation(image)
corrected_image = predictor.correct_orientation(image)
```

The 180-degree counterpart is generated on the fly after grouping and splitting; photometric jitter is train-only. Inference uses the checkpoint's pretrained normalization and aspect-ratio-preserving resize/pad. The training script selects the checkpoint by validation accuracy; evaluation reports held-out metrics, plots, and forward-pass latency. MPS and an untrained forward pass were verified on Apple M4 Pro outside the sandbox. The requested pretrained checkpoint was downloaded and loaded. Training, held-out evaluation, latency measurement and inference correction checks have now completed for the reviewed-data run. See the run report for measured results.

Run basic grouping and rejection tests with `python -m unittest -v vit_orientation.test_data`. Check group and source provenance independently before training: file hashes detect exact duplicate bytes only, not re-encoded or transformed duplicates.

## Historical-label audit

See `../PROVENANCE_AUDIT.md` for the verified pre/post-classifier lineage, saved-decision replay results, and the label inversion required when training on pre-classifier rectified full images. The blanket `review_required` status in the earlier generic reconstruction must not be interpreted as a requirement to repeat existing human orientation labeling.

## Reviewed-data run

The reviewed local labels are now applied in `outputs/orientation_label_review/revisions/review_2468f6812375`. The combined manifest and grouped source metadata are frozen under `outputs/datasets/vit_small_reviewed_20261112_v2`. The run excludes the conflicting Roboflow `dbg_83` group and exact duplicate files. Training progress is logged at `outputs/logs/vit_small_orientation_20261112.log`. Test evaluation remains a separate command after training is finalized.

## Completed experiment

The selected checkpoint is `outputs/vit_small_orientation_20261112/best.pt` (epoch 6). Validation accuracy: 98.6355%; held-out test accuracy: 99.7076% on 1,026 orientation samples from 513 files / 359 source groups. `RESULTS.md` inside that run directory documents metrics, figures, measured latency, original/augmented sample counts and limits. Test metrics are saved once and the evaluation command refuses to overwrite them. The production pipeline is unchanged.
