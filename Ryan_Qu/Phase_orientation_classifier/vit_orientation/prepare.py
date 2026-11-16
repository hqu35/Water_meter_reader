"""Create a frozen split only from independently verified full-image labels."""

import argparse
from .data import prepare

p = argparse.ArgumentParser()
p.add_argument("labels_csv")
p.add_argument("output")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
m = prepare(a.labels_csv, a.output, a.seed)
for split in ("train", "val", "test"):
    n = sum(r["split"] == split for r in m["records"])
    print(f"{split}: {n} originals; {2 * n} virtual samples")
