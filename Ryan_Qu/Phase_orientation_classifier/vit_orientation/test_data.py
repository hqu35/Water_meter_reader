"""Small deterministic group-splitting and label-safety checks."""

import csv
import tempfile
import unittest
from pathlib import Path
from .data import prepare


class SplitTests(unittest.TestCase):
    def test_group_splits_and_label_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "labels.csv"
            rows = []
            for source in ("dryad", "roboflow"):
                for i in range(6):
                    for j in range(2):
                        image = root / f"{source}{i}_{j}.png"
                        image.write_bytes(str(image).encode())
                        rows.append(
                            dict(
                                path=str(image),
                                label=str(j),
                                source=source,
                                group_id=str(i),
                                label_evidence="reviewed",
                            )
                        )
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0])
                w.writeheader()
                w.writerows(rows)
            a = prepare(path, root / "a.json")
            b = prepare(path, root / "b.json")
            self.assertEqual(a, b)
            groups = {}
            for r in a["records"]:
                k = (r["source"], r["group_id"])
                groups.setdefault(k, set()).add(r["split"])
            self.assertTrue(all(len(s) == 1 for s in groups.values()))
            rows[0]["label"] = "unknown"
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0])
                w.writeheader()
                w.writerows(rows)
            with self.assertRaises(ValueError):
                prepare(path, root / "invalid.json")


if __name__ == "__main__":
    unittest.main()
