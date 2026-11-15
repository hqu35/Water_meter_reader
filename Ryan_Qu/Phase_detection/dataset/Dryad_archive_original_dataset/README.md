# Word-wheel source material

`Word-Wheel_Water_Meter_Dataset.zip` links to the original downloaded Dryad archive.
`selection.json` preserves the exact deterministic 3,600 source image/mask pairs,
source byte/pixel hashes, sizes, seed and official split identities. It contains no
converted box labels. The ZIP and inventory remain local and ignored.

Run `python -m stage1_python_worker prepare --archive /path/to/source.zip
--selection /path/to/selection.json` under a fresh `--root` to rebuild. Source hashes
are checked before conversion. The inventory is also copied into the prepared
metadata so remote integrity checks need no raw ZIP transfer.
