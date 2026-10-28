# Manual review of the orientation pilot

The pilot's `index.html` is an English-only, offline review interface. The
100 original crops, corrected crops, and source photographs are embedded in the
HTML, so it can be opened without a server or an internet connection.

Current file:
`outputs/pilots/paddle_orientation/gt_train100_003/index.html`

```bash
open outputs/pilots/paddle_orientation/gt_train100_003/index.html
```

## What to label

Judge the **original crop on the left**, not the model-corrected crop:

- **0° — Original is upright:** no rotation is required.
- **180° — Original is upside down:** the original requires a 180° rotation.
- **Uncertain:** orientation cannot be determined reliably.
- **Unusable crop:** the register is incorrectly cropped, severely obscured, or missing.

A separate crop-issue checkbox preserves cases where orientation is identifiable
but the crop still has problems. Add optional notes and a reviewer name/alias.
The full photograph can be expanded for context. Zoom enlarges the preview only;
it cannot recover missing image detail.

Keys: `0` for upright, `1` for 180°, `U` for uncertain, `B` for unusable;
left/right arrows navigate. Shortcuts do not run while typing in an input.
Use the sidebar to search/filter samples, or choose **Next unreviewed**. Automatic
advance is optional. Any decision can be changed or marked unreviewed again.

## Save and export

Progress is automatically saved in the current browser using a pilot-specific
fingerprint. File-URL storage may vary by browser; export JSON regularly as a
backup. Opening the same file in another browser does not transfer local storage.

- **Export review JSON:** preferred file to send back for data analysis. It
  includes all 100 IDs, immutable model predictions/confidences, source and crop
  paths/hashes, manual decisions, crop flags, notes, reviewer and timestamps.
- **Export CSV:** spreadsheet-readable rows, including unreviewed samples. Notes
  preserve quotes/newlines; formula-like strings are escaped for spreadsheets.
  JSON retains the exact original text and is the authoritative exchange format.
- **Export portable HTML:** embeds images and current review state in one file.
  It can be reopened offline to continue reviewing and exporting.
- **Import review JSON:** restores an exported review for this same pilot.
  Incompatible fingerprints, duplicate/unknown IDs, invalid labels, or changed
  model predictions are rejected before existing progress is replaced.

`review_decision` is one of `unreviewed`, `0`, `180`, `uncertain`, `bad_crop`.
`manual_orientation_deg` and `model_decision_correct` are null when no definite
orientation can be judged. The latter compares your decision with the model; it
is not an independently established correctness label. Crop issues are flagged
separately; `eligible_for_orientation_analysis` requires a definite human
orientation, successful model inference and no crop flag. Analysis can explicitly
choose another inclusion policy from the raw fields later.

No aggregate orientation accuracy is displayed. Unreviewed, uncertain and unusable
samples are not silently interpreted as correct. The dataset, model predictions,
and production inference/training/evaluation remain unchanged.

## Rebuild and verification

Rebuild the interface from existing pilot outputs without running PaddleOCR:

```bash
python3 -m stage1_python_worker.obb.orientation_review \
  outputs/pilots/paddle_orientation/gt_train100_003
```

Future pilot runs generate this interface automatically. The former static
contact-sheet gallery is retained as `index.contact-sheets.html` in the current run.
The HTML template is included in the worker's source upload inventory.

```bash
python3 -m unittest discover -s tests -v
node tests/review_state_checks.cjs
```

Browser QA uses separate fixture IDs and storage keys, never human labels on the
real 100 samples. All three export files were downloaded and inspected. Refresh
persistence and the import/export state logic were checked. Browser-automated file
selection for import was restricted by extension permissions; these permissions
were not changed. Manual file selection uses the page's normal file input.
