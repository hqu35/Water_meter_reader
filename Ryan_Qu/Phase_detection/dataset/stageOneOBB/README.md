# Stage I oriented-register dataset

3,600 real examples: 2,400 train, 400 validation and 800 official test. Training
and validation come only from the official training pool. Source identities and
hashes are stored in `metadata/source_selection.json`.

- `images/{train,val,test}/`: photos; 89 boundary cases have documented padding.
- `masks/{train,val,test}/`: aligned binary foreground masks.
- `labels/{train,val,test}/`: `0 x1 y1 x2 y2 x3 y3 x4 y4`, normalized corners.
- `metadata/`: source selection, split manifest, validation and future human approval.
- `data.yaml`: machine-specific YOLO split paths.

Review sheets are in [artifacts/annotation/](../../artifacts/annotation/README.md).
Human approval is pending. Data, labels and JSON remain ignored; no README files
are inserted into exact-membership image/mask/label leaves.
