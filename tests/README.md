# OBB tests

Run `python -m unittest discover -s tests -v`. `test_stage1_obb.py` checks pixel-edge
rectangles, permutations/winding, padding, normalized serialization, aligned crops,
polygon IoU, training settings, source-only preparation, human approval and remote
payload routing. `stage1_fixtures.py` builds source ZIP fixtures and identity lists;
these generated patterns are only test inputs and never model-training data.
