"""Select a balanced appearance pool and plan FLUX.3 reference combinations."""

import argparse
import json
import random
from pathlib import Path

DEFAULT_COUNT = 100
DEFAULT_SEED = 539
REFERENCES_PER_TASK = 9
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "shared_dataset",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    if args.n % 2 or args.n < REFERENCES_PER_TASK * 2:
        parser.error(
            "N must be even and at least 18 so each category can supply nine distinct references"
        )

    root = args.dataset_root.resolve()
    categories = {
        "clear": root / "clear_1500",
        "clear_noise": root / "clear_noise_500",
    }
    available = {}
    for name, folder in categories.items():
        if not folder.is_dir():
            parser.error(f"Dataset category directory does not exist: {folder}")
        available[name] = sorted(
            path.relative_to(root).as_posix()
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(available[name]) < args.n // 2:
            parser.error(
                f"{name} has {len(available[name])} images; {args.n // 2} required"
            )

    rng = random.Random(args.seed)
    pool = {name: rng.sample(paths, args.n // 2) for name, paths in available.items()}
    # Largest-remainder apportionment; ties follow the group order below.
    groups = ("clear_only", "clear_noise_only", "mixed")
    weights = (4, 4, 2)
    counts = [args.n * weight // 10 for weight in weights]
    order = sorted(range(len(groups)), key=lambda i: (-(args.n * weights[i] % 10), i))
    for index in order[: args.n - sum(counts)]:
        counts[index] += 1

    tasks = []
    for group, count in zip(groups, counts):
        for index in range(count):
            if group == "clear_only":
                clear_count = REFERENCES_PER_TASK
            elif group == "clear_noise_only":
                clear_count = 0
            else:
                clear_count = REFERENCES_PER_TASK // 2 + index % 2
            noise_count = REFERENCES_PER_TASK - clear_count
            references = rng.sample(pool["clear"], clear_count)
            references += rng.sample(pool["clear_noise"], noise_count)
            rng.shuffle(references)
            tasks.append(
                {
                    "task_id": f"{group}_{index + 1:05d}",
                    "group": group,
                    "seed": args.seed,
                    "appearance_refs": references,
                    "clear_count": clear_count,
                    "clear_noise_count": noise_count,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    pool_manifest = {
        "n": args.n,
        "seed": args.seed,
        "dataset_root": "shared_dataset",
        "available_counts": {name: len(paths) for name, paths in available.items()},
        "clear": pool["clear"],
        "clear_noise": pool["clear_noise"],
    }
    (args.output / "selected_reference_pool.json").write_text(
        json.dumps(pool_manifest, indent=2) + "\n"
    )
    (args.output / "reference_combinations.jsonl").write_text(
        "".join(json.dumps(task) + "\n" for task in tasks)
    )
    print(f"Selected {args.n} references; wrote {len(tasks)} tasks to {args.output}")
    print(
        f"Available: {pool_manifest['available_counts']}; task counts: {dict(zip(groups, counts))}"
    )


if __name__ == "__main__":
    main()
