# Classical CV digit-strip experiment

## Result

The test processed **21/21** original crops. The line-support heuristic flagged **18/21** as having four plausible separators; this is an uncalibrated image-evidence test, not human verification. **0/21** passed the combined geometric requirement that all five minimum-area OBBs lie inside the original input image and do not overlap. All 21 cases had at least one OBB outside the original crop; 8 also had adjacent OBB overlap. Each output is provisional. No manual digit-level ground-truth annotations were supplied; accuracy and IoU are not measured.

For each image, five candidate YOLO OBB labels refer **only to the corresponding padded image** in `yolo_candidates/images/`, not the original. Padding preserves the candidate geometry rather than clipping corners; boxes may extend into blank padding, overlap one another, or locate the wrong strip. Do not treat them as ground truth or use them as training labels without human correction.

The five source IDs 292, 304, 305, 307 and 308 were not present in the prior `qualified` subset. They were included because the task explicitly requested every sample in 290–310. No samples were missing or ambiguous: duplicate copies in `qualified/` were byte-identical to the top-level LoRA_training originals. The original images range from 114×37 to 513×112 pixels.

## Method and limits

1. Resize to 400 pixels wide; grayscale, Gaussian smoothing, estimate local dark contrast, and morphological opening to enhance vertically continuous features. Add a weaker vertical-gradient term. Aggregate along candidate slanted lines with medians/means.
2. Search 17 shared slopes between -0.22 and +0.22, and use dynamic programming for four ordered separators with soft spacing preferences. This permits uneven spacing but assumes a shared straight slope; it does not model curved separators or perspective convergence.
3. Estimate each separator width from its local evidence peak. Search near the upper and lower image edges for horizontal window boundaries. Use the **crop's left and right edges** as provisional outer bounds, not independently verified strip edges. Fit `cv2.minAreaRect` to each contour. The exact resulting coordinates and quality flags are in per-image `result.json` files.
4. Dark digit strokes and dirt can resemble separators; top and bottom window boundaries and actual separator widths are often uncertain. The current OBB construction can extend outside the image or overlap, even when underlying contours are disjoint. Support score is not a calibrated confidence or proof of correct separators.

`diagnostics.png` shows grayscale (upper left), local-dark response (upper right), vertical opening (lower left), and line evidence (lower right). `separators.png` shows fitted lines and horizontal window estimates; `obb.png` shows five provisional OBBs. `contact_sheet.png` compares each source to its candidates.

## Per-sample status

“Support” is the internal prominence-and-spacing heuristic only; “Geometry” checks bounds and nonoverlap. Neither column indicates visual correctness.

| ID | Processing | Four-line support | Five in-bounds, nonoverlapping OBBs | Runtime (s) |
|---:|---|---|---|---:|
| 290 | processed | supported | fail | 0.094 |
| 291 | processed | supported | fail | 0.060 |
| 292 | processed | supported | fail | 0.065 |
| 293 | processed | supported | fail | 0.062 |
| 294 | processed | supported | fail | 0.070 |
| 295 | processed | supported | fail | 0.059 |
| 296 | processed | supported | fail | 0.074 |
| 297 | processed | supported | fail | 0.092 |
| 298 | processed | supported | fail | 0.052 |
| 299 | processed | supported | fail | 0.058 |
| 300 | processed | supported | fail | 0.074 |
| 301 | processed | weak | fail | 0.066 |
| 302 | processed | weak | fail | 0.079 |
| 303 | processed | supported | fail | 0.078 |
| 304 | processed | supported | fail | 0.071 |
| 305 | processed | supported | fail | 0.087 |
| 306 | processed | weak | fail | 0.070 |
| 307 | processed | supported | fail | 0.098 |
| 308 | processed | supported | fail | 0.068 |
| 309 | processed | supported | fail | 0.064 |
| 310 | processed | supported | fail | 0.071 |

Median per-image runtime was **0.070 s**, range 0.052–0.098 s, on this local machine; runtime includes preprocessing, fitting, rendering and per-sample writes, but not contact-sheet or sensitivity generation.

## Visual observations and perturbation stability

On visual inspection, 291 has candidates that roughly follow several digits but the first OBB reaches beyond the input. In 302 some fitted boundaries cut through or misgroup digits; 298 contains a dirty obscured first region; 306 is very blurry; 307 contains narrow 1-shaped strokes that can be confused with separators. Sample 304 is upside down. These examples are qualitative observations, not scored labels.

`sensitivity.json` reports controlled changes to samples 291, 302 and 306. It compares the four fitted separator midpoints to the model's own unperturbed output; rotation results are mapped back with the inverse affine transform. It is a stability measurement, not correctness. Gaussian blur (sigma 1.5) produced mean shifts of 5.25%, 0.00% and 13.13% of image width, respectively. A 40% brightness reduction yielded 1.75%, 0.00% and 0.37%. Rotations of ±5° ranged from 0.21% to 12.31%. The no-shift outcomes do not establish correctness, and this three-image probe does not establish robustness to lighting or separator contrast generally.

This prototype does not justify replacing YOLO-OBB or automatically accepting labels. It may generate rough human-review proposals, but only if users are shown the original image, candidate lines, geometry flags, and can correct or reject every box.

## Reproduction

Create an environment with `numpy`, `scipy`, `Pillow`, and `opencv-python-headless`. No neural network weights or model downloads are needed. Run from this directory with a **fresh, empty output path**:

```bash
python -m pip install numpy scipy pillow opencv-python-headless
python segment.py --input-dir ../../../LoRA_training --output-dir /tmp/watermeter_classical_cv_run
python sensitivity.py --input-dir ../../../LoRA_training --output /tmp/watermeter_classical_cv_run/sensitivity.json
```

Parameters: `--work-width` (default 400), `--max-slope` (0.22), `--spacing-weight` (0.6), and `--min-prominence` (0.25). The threshold is an exploratory heuristic and has not been calibrated against manual annotations. Source images are never modified; output paths must be empty to avoid overwriting prior results.
