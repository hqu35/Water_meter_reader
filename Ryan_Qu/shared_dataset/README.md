# Shared orientation-corrected register crops

The two current image categories are clear_1500/ and clear_noise_500/. Open
index.html to browse the 1,996 images. Filenames are original Dryad IDs.

The baseline is Phase_orientation_classifier/dataset_2000/manifest.json and
its images. Each upright image was copied without changing pixels; each image
marked upside down was rotated exactly 180 degrees with OpenCV. The four
discarded images were excluded. The historical directory names are retained:
clear_1500 contains 1,498 images and clear_noise_500 contains 498.

- selected_image_sources_and_original_ids.json: current image IDs, category,
  baseline image path, author labels, and the applied orientation correction.
- selected_image_counts_and_categories.json: current category and correction
  counts and number excluded.
- image_crop_processing_records/: historical records from the earlier image
  selection and processing; paths inside those records describe the old layout.
- reviewed_lora_register_crops/: the earlier LoRA crop pool retained for
  historical source-recovery scripts. It is not the current 1,996-image set.

The baseline orientation marks are human-provided; these images should still be
inspected for crop quality. Historical annotation exports and the standalone
annotation page use their original baseline images and display-time rotation;
do not automatically reinterpret their stored coordinates in another frame.
