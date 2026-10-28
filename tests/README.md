# OBB tests

Run `python -m unittest discover -s tests -v`. `test_stage1_obb.py` checks pixel-edge
rectangles, permutations/winding, padding, normalized serialization, aligned crops,
polygon IoU, training settings, source-only preparation, human approval and remote
payload routing. `stage1_fixtures.py` builds source ZIP fixtures and identity lists;
these generated patterns are only test inputs and never model-training data.

`test_orientation_pilot.py` verifies pilot ordering, parser validation, exact pixel correction, sample-failure continuation and no-overwrite behavior.

`test_orientation_review.py` checks image embedding, prediction preservation, data fingerprints and safe template generation. `node tests/review_state_checks.cjs` verifies the actual JavaScript import/export and review-state functions without a browser or npm dependencies.
