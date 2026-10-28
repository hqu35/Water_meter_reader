# Dataset storage

`stageOneOBB/` contains the local 2,400/400/800 oriented-register subset and
metadata. `wordWheelSource/` contains a symlink to the original Dryad ZIP and
`selection.json`, the exact source identity inventory with hashes and split
membership. Both contain ignored local data; back them up to reproduce the exact
experiment. The source inventory is also copied into the final run metadata.
See [Stage I reproduction](../docs/STAGE1_OBB.md#reproducibility-and-local-artifacts).
