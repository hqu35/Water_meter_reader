# Clear register crop selection

Open `index.html` for the categorized gallery. `clear_1500/qualified_replicate` contains 488 byte-identical copies from `qualified`; `clear_1500/new_dataset` contains 1012 additional crops. `clear_noise_500` contains 500 clear-labeled crops with exactly one noise label and blurry=0. The noise categories are {'dial-stained': 293, 'reflective': 95, 'dark': 36, 'soil-covered': 76}.

Noise means blurry, dial-stained, soil-covered, dark, or reflective. Six-digit is a meter-type label, not noise. The labels are the original Dryad train-set labels and are not a guarantee of visual quality. Sources were selected without replacement using seed 42, prioritizing already-processed crops. All original files are preserved. Existing Stage I train/val sources and the previous LoRA candidate pool were excluded from newly selected source IDs to reduce duplicate/rejected-image reuse. No official test images were used.

Qualified image 752.png was excluded because its original source could not be matched reliably. For source recovery, original or recomputed crop pixels were matched exactly when possible; a small subset was matched to near-identical recomputed crops with very large margins over other candidates. Details are recorded in `selection_manifest.json`.

New crop processing: {'prior_stage1_crop': 591, 'stage1_detector': 920, 'stage1_detector_low_confidence': 1}. Stage I OBB, full-image rectification and Roboflow orientation were used; where the OBB detector failed, a nonempty original segmentation mask was used as a fallback after dimension validation. Individual outputs and fallback status are recorded in `.processing_records/`. Both the mask labels and predictions require human review.
