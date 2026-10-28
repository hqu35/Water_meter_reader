# OBB tests

Run `python -m unittest discover -s tests -v`. `test_stage1_obb.py` checks pixel-edge
rectangles, permutations/winding, padding, normalized serialization, aligned crops,
polygon IoU, training settings, source-only preparation, human approval and remote
payload routing. `stage1_fixtures.py` builds source ZIP fixtures and identity lists;
these generated patterns are only test inputs and never model-training data.

`test_orientation_pilot.py` verifies pilot ordering, parser validation, exact pixel correction, sample-failure continuation and no-overwrite behavior.

The following optional historical review checks require the locally preserved, Git-ignored `stage1_python_worker/obb/orientation_review.html` template. `test_orientation_review.py` checks image embedding, prediction preservation, data fingerprints and safe template generation. `node tests/review_state_checks.cjs` verifies the actual JavaScript import/export and review-state functions without a browser or npm dependencies.

`test_stage1_integration.py` checks full-image coordinate synchronization and
orientation failure handling. `test_orientation_evaluation.py` checks the local
Label Studio workflow and requires the locally preserved, Git-ignored
`orientation_evaluation/label_studio.xml`.
