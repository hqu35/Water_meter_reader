# Stage I oriented-register dataset

3,600 real examples: 2,400 train, 400 validation and 800 official test. Training
and validation come only from the official training pool. Source identities and
hashes are stored in `metadata/source_selection.json`.

- `images/{train,val,test}/`: photos; 89 boundary cases have documented padding.
- `masks/{train,val,test}/`: aligned binary foreground masks.
- `labels/{train,val,test}/`: `0 x1 y1 x2 y2 x3 y3 x4 y4`, normalized corners.
- `metadata/`: source selection and split manifest; legacy runs may also contain historical review records.
- `data.yaml`: machine-specific YOLO split paths.

Historical sheets are in [register_BB_annotation_inspection/](../../register_BB_annotation_inspection/annotation/README.md).
Use `python -m stage1_python_worker validate` for an explicit dataset check, or
`preview --output outputs/obb_preview` with the same worker to render new sheets.
Neither is an approval requirement. Images, masks, labels and metadata remain
local.
