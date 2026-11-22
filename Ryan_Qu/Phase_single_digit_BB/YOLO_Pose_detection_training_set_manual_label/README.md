# Water Meter Digit-Slot YOLO-Pose Dataset

This dataset contains 664 real, reviewed register crops and 3,389 digit-slot instances: 524 train, 40 validation and 100 held-out test images. No model training was performed. Each digit is class `0` (`digit_slot`), with an axis-aligned detection box and four keypoints ordered TL, BL, BR, TR. The provided images are the current orientation-corrected shared crops; labels use their pixel coordinate system.

`prepare_dataset.py` reads `complete_annotations_for_training.json`, the current `shared_dataset/clear_1500` and `clear_noise_500` images, and the `dataset_2000` orientation manifest. Its default input locations are relative to the repository. It requires Python 3 and Pillow. To reproduce without replacing these outputs:

```bash
python3 prepare_dataset.py --output /path/to/new/output
```

The converter keeps all slots from the same original Dryad image ID in one split. The selected set contains one record per source ID and no known augmented variants; no independent near-duplicate or physical-meter grouping was established. Random seed 42 determines the split and the 25 sampled visualizations.

## Orientation alignment

The annotation for `train18154` predates an additional human orientation correction. The converter keeps the current shared crop unchanged and rotates its annotation coordinates 180 degrees around the pixel-center image extent, reorders the corners back to TL, BL, BR, TR, and reverses the digit sequence. This is recorded in `split_manifest.json` and `conversion_summary.json`. The additional `visualizations/train18154_orientation_alignment.png` shows that alignment. No images are rotated a second time.

## Colab transfer

Copy this directory to `/content/watermeter_pose_dataset` before using `watermeter_pose.yaml`; its `path` points there. If copied elsewhere, change `path` in the YAML to the actual dataset root. The YAML uses `kpt_shape: [4, 2]` without visibility flags.

`conversion_summary.json` records counts and problems; `split_manifest.json` records the assigned split and original source ID. `visualizations/` contains 25 sampled annotated images and one additional orientation-alignment check. Labels were checked for paired files, bounds, four keypoints, positive boxes and expected instance counts. Visualizations support inspection; they do not independently certify annotation accuracy or model performance.
