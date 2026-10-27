# Source data utilities

`archive.py` pairs original photographs/masks in the ZIP, checks dimensions and
encoding, and inspects oriented rectangles. `download.py` obtains/checks the frozen
Dryad source archive. `validation.py` rejects missing or extra images/masks/labels.
Conversion and human review are implemented once in `../obb/`.
