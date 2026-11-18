# Register-cell annotation: automatic proposals with manual fallback

Open or share **`index.html`**, a standalone offline HTML file containing the same 1,996 orientation-labeled register images. All interface text and exports are English. Source images, IDs, dataset selection, orientation handling, and five/six-digit labels are unchanged.

## Workflow

1. Mark the register outline: **TL → BL → BR → TR**.
2. Mark digit 1's right edge: **BR → TR**. Its left corners reuse the register TL/BL. After these six Space presses, automatic equal-spacing proposals appear and are saved.
3. If the proposals look satisfactory, press **period (`.`)** to keep them and continue to the next image. Six valid inputs now count as a complete automatic proposal; no reference-only clicks are required.
4. If the proposals need correction, press **Space at digit 2 TL**. Saving this **seventh point** immediately discards all extrapolated boxes and switches this image to **manual fallback**. Mouse movement alone does not switch modes.
5. Continue digit 2 **BL → BR → TR**, then mark every remaining digit **TL → BL → BR → TR**. Digit 1 and the register outline are retained. No widths, gaps, shared edges, or last-cell right corners are inferred for the remaining cells.

Manual fallback requires **22 total points for five digits** or **26 for six**, including the register outline. While a manual cell is incomplete, only its selected points/partial outline are shown; it is not exported as a complete quadrilateral. The other unmarked cells have no predicted boxes.

## Geometry and controls

The existing image stays unwarped. Digit TL/TR points snap to the top rail; BL/BR points snap to the bottom rail. The guide follows the average register side shear `(mean(dx/dy), 1)` and preserves the crosshair-to-saved-point mapping at original image resolution.

Automatic mode computes top and bottom widths/gaps independently, using `gap = (rail_length - digit_count * first_cell_width) / (digit_count - 1)`. Its last cell reaches the register right edge. Negative gaps or zero-width cells are rejected. **Manual fallback does not enforce equal widths or gaps**; it remains available when automatic spacing is infeasible. Cells are checked for convexity, corner order, image bounds, and rail membership. Geometric validity does not verify digit boundary accuracy.

- **Space:** record the colored crosshair center. Point 7 starts manual fallback.
- **C:** undo a manual point. Undoing back to six points restores the automatic proposals.
- **R:** reset this image after confirmation.
- **Comma / period:** previous / next image, without Shift. ID/index jump controls remain available.
- Autosave/resume and JSON import/export preserve the current mode and incomplete manual work.

## Existing annotations and export

The previous standalone tool is preserved byte-for-byte in `index-equal-spacing-reference-legacy.html`.

The current tool reads existing equal-spacing browser records without rewriting their storage namespace. It also accepts `watermeter-guided-equal-spacing-export-v1` JSON imports, including the user's existing 250-image export. The first six geometry inputs are reused, reproducing the same automatic boxes. Old points 7/8 were references, so they are **not silently converted into manual annotations**. The complete previous record, including those references, is retained under `previous_equal_spacing_record` and exported with the new record. Only a newly saved seventh point triggers manual fallback. The original downloaded JSON is never modified.

New records use `watermeter-guided-hybrid-cells-v1`; exports use `watermeter-guided-hybrid-cells-export-v1` and filename `watermeter_hybrid_cell_annotations.json`. They include:

- `annotation_mode`: `automatic` or `manual_fallback`.
- `manual_points`, their roles, register corners, and remaining digit inputs.
- `digit_quadrilaterals` in **TL/BL/BR/TR** order; only complete cells are included.
- `digit_box_sources` distinguishing predictions from manually defined cells.
- Original-resolution coordinates, source hashes, orientation correction, spacing when applicable, and validation status.

`complete` means the current workflow's required inputs and geometry are complete. It does not certify ground truth. Automatic boxes remain explicitly unverified. Older fully manual/guided formats remain separate; the legacy export button provides backups where browser storage is visible. Local-file storage varies by browser, so keep JSON backups.

## Source and verification

`geometry.js` implements projection, validation, automatic geometry, and the point-7 manual branch. `guided_app.js` handles capture, drawing, compatible storage, and import/export. `template.html` defines the English UI; `build.py` embeds the verified dataset and scripts in one file.

```bash
node geometry.test.js
node guided_app.test.js
python build.py --output index-new.html
```

Tests cover both digit counts, automatic spacing, immediate removal of predicted overlays after point 7, unchanged first-cell geometry, unequal manual widths/gaps, 22/26-point completion, undo across the mode transition, partial-mode resume, cursor alignment, import/export, old 6/8-point preservation, and storage errors. The actual 250-image export was checked read-only: all 250 automatic geometries reproduce the previous coordinates. These checks do not constitute a live browser UI test.

The builder refuses to overwrite existing outputs. Verify a new build before replacing `index.html`.
