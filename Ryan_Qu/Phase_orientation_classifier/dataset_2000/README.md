# Orientation classifier dataset: 2,000 reviewed-gallery crops

The package retains all 2,000 original crops. Use only the 1,996 images in `images/upright_assumed/` and `images/upside_down/` as potential classifier inputs; the four user-rejected samples are isolated in `images/discarded/` and have no training label. Images are byte-identical copies and have not been rotated. The dataset has not been split into train/validation/test.

The user explicitly marked 73 images upside down. The other 1,923 training-eligible images were not marked; their provisional `upright` label is an assumption based on absence of a mark, not proof of individual human verification. Whether every image was inspected is unknown. The four excluded source IDs are train11264, train1355, train14871, train14855.

`manifest.json` and `manifest.csv` map original Dryad IDs, prior local crop filenames, package paths, scenario labels, source-mapping method, SHA-256 hashes, orientation mark status, exclusion status, and label evidence. 

`orientation_marks_original.json`, `source_selection_original.json`, and `source_selection_summary.json` preserve inputs for reproducibility. When available, the JSON manifest also retains each sample's previous Stage I processing record, including fallback and orientation model information. Some `qualified`-derived source IDs were recovered through near-identical historical crop matching rather than exact original filename records; see `source_selectino_original.json / source_mapping_method`. All source images came from the official Dryad training pool.
