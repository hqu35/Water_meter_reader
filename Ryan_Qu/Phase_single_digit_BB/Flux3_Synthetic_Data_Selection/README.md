# FLUX.3 appearance reference selection

The worker samples the orientation-corrected shared_dataset/clear_1500/ and
shared_dataset/clear_noise_500/ categories. These contain 1,498 and 498 eligible
images respectively; four discarded baseline images are excluded.

Run: python3 reference_selector.py --n 100 --seed 539

Use --dataset-root for a different copy of shared_dataset and --output for a
different manifest directory. Re-running overwrites those two manifests. N must
be even and at least 18; each category must supply N/2 images.

Manifest paths are relative to shared_dataset. List order maps to image0
through image8; the separate geometry mask is supplied later as image9.
Task counts use 40/40/20 largest-remainder rounding, with ties resolved in
clear-only, clear-noise-only, mixed order. Mixed compositions alternate.
