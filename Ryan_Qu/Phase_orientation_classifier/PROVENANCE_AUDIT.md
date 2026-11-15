# Orientation crop provenance audit

The reviewed 1,996-image pool mixes historical processing paths. Recovered original IDs are sufficient to locate input images; replay must follow each record's historical branch. The previous generic mask-plus-template-match reconstruction is a proposal set and is not the authoritative replay.

| Reviewed crop stage | Count |
| --- | ---: |
| After legacy orientation-classifier correction | 1,651 |
| Before orientation classification (GT original crops) | 38 |
| GT rectification with a saved 0/180 replay rotation | 307 |

The 1,651 post-correction cases comprise 1,508 selected shared-dataset images (916 detector, 1 lower-threshold detector, 591 cached crops), plus 143 qualified images traced to corrected outputs. The other qualified images consist of 38 original GT crops and 307 exact GT replay mappings. The 143 include 76 historically near-identical ID mappings; these retain their original mapping evidence and must not be relabeled as exact ID matches.

## Source evidence

Snapshots and machine-readable per-image lineage are preserved under `outputs/provenance_audit/`:

- `source_snapshot/process_shared_crops.py`, lines 30–41: full-image rectification, legacy orientation classification, optional full-image 180-degree flip, then crop and save.
- `source_snapshot/clear_blurry_process.py`, lines 29–44: the same sequence, with a branch reusing saved corrected full images.
- `source_snapshot/match_gt_reprocessed.py`: GT labels, recorded 0/180 rotation and exact crop matching.
- `source_snapshot/map_qualified_sources.py`: exact artifact matches and the original/corrected folder distinction.
- `crop_lineage.json` and `summary.json`: coverage of all 1,996 reviewed images.

## Replay checks performed

With the historical local OpenCV 4.12.0 environment, all **307 of 307** saved GT replay mappings reproduced the reviewed crop pixels exactly. Running the same replay with a newer OpenCV 5.0 environment retained all dimensions but introduced maximum channel differences of 1–4. This was a library-version interpolation difference, not a new orientation label.

Nine samples were replayed using the saved YOLO checkpoint, CPU inference, historical settings and saved legacy orientation decisions; no online classifier was called. All nine reproduced the crop dimensions and direction. Six were pixel-identical; three had maximum channel differences of 1, 1 and 2. These detector results are sample checks, not an exhaustive replay of all detector-derived images. Results are saved in `detector_replay_checks.json`; GT results in `gt_replay_checks.json`.

## Label transfer

A user-reviewed crop's orientation label applies directly to its corresponding full image at the same stage of processing. This does not require repeating human labeling.

For a full image *before* the old classifier's correction, when the historical extra flip is known, use:

`label_before = label_reviewed_crop XOR int(historical_180_flip)`

For example, if the saved old classifier applied 180 degrees and the user-reviewed resulting crop is upright, the pre-classifier rectified full image is upside down. For full images after that same historical correction, retain the reviewed crop label directly. Preserve which stage the training input represents. Replay the recorded geometric branch and saved classifier decision rather than calling a potentially changed online model or estimating polarity through new template matching.

This audit does not modify image files, human labels, splits, model checkpoints, or the production pipeline. Training has not been launched as part of this audit.
