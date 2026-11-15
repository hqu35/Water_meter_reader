# Manual digit-box overlays

11 original-size register images with all 55 manually annotated single-digit quadrilaterals drawn using OpenCV. Bright colors distinguish digit cells, and dots mark their corners. No cropping or rectification is applied.

- `train*_digit_boxes.png`: annotated images preserving original image IDs and resolution.
- `overview.png`: combined visual overview.
- `annotations_original.json`: unchanged source annotation export.
- `draw_digit_boxes.py`: reproducible overlay generation script; select a new output directory when rerunning.

All supplied records specify identity orientation correction. Source checksums and dimensions were verified before drawing. The previous digit-crop outputs have been replaced by these overlays; original dataset images remain unchanged.
