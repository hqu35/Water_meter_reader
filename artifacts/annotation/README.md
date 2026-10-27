# OBB annotation review — approval pending

Open [index.html](index.html) in a browser, or on macOS run
`open artifacts/annotation/index.html` from the repository root.

- [Sheet 1](obb_review_01.jpg): 8 training examples.
- [Sheet 2](obb_review_02.jpg): 8 training examples.
- [Sheet 3](obb_review_03.jpg): 8 validation examples.
- [Sheet 4](obb_review_04.jpg): 8 official-test examples.

Each row shows original photo, segmentation foreground + red oriented rectangle,
and long-axis-aligned crop. Verify alignment, edge digits and crop geometry.
180-degree text polarity remains ambiguous. Source `train_train17696` already
truncates digits at the image boundary; padding cannot restore them.

These are valid JPEG files and stay visible to Git. The image move does not grant
approval. Only after explicit human review should `approve --reviewer NAME
--confirm-reviewed` be run. JSON binds the relocated sheet hashes.
