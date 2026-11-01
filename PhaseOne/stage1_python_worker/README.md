# Stage I worker

Run `python -m stage1_python_worker --help`. All detector commands execute OBB.

- `cli.py`, `__main__.py`: one command entry point; optional `obb` prefix accepted.
- `config.py`: repository identity and source checksums/upload inventory.
- `obb/`: oriented conversion, review, training, evaluation, inference and transport.
- `data/`: source archive pairing, download and exact directory membership.
- `training/`: T4 CUDA check, epoch reporting and checkpoint class-name normalization.
- `utils/`: JSON, checksums and safe ZIP extraction.
