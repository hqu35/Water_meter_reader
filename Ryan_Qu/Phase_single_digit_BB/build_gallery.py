"""Build an offline, self-contained review page from existing predictions."""

import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/exp0_pretrained_obb"


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output.resolve()
    predictions = json.loads((OUTPUT / "raw_predictions.json").read_text())
    selected = json.loads((OUTPUT / "selected_samples.json").read_text())
    meta = {r["sample_id"]: r for r in selected["samples"]}
    rows = predictions["samples"]
    assert len(rows) == 100 and set(meta) == {r["sample_id"] for r in rows}
    enriched = []
    for row in rows:
        item = dict(row)
        item["scenario_labels"] = meta[row["sample_id"]]["scenario_labels"]
        enriched.append(item)
    gallery = OUTPUT / "gallery"
    gallery.mkdir(exist_ok=False)
    template = (ROOT / "review_template.html").read_text()
    payload = json.dumps({"samples": enriched}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    (gallery / "index.html").write_text(template.replace("__DATA__", payload))
    counts = collections.Counter(r["box_count"] for r in rows)
    scores = [d["confidence"] for r in rows for d in r["detections"]]
    bins = {
        "0.10-0.25": sum(0.1 <= s < 0.25 for s in scores),
        "0.25-0.50": sum(0.25 <= s < 0.5 for s in scores),
        "0.50-0.75": sum(0.5 <= s < 0.75 for s in scores),
        "0.75-1.00": sum(0.75 <= s <= 1 for s in scores),
    }
    groups = collections.Counter(meta[r["sample_id"]]["selection_group"] for r in rows)
    labels = collections.Counter(d["class_name"] for r in rows for d in r["detections"])
    summary = {
        "samples": 100,
        "detection_count_distribution": dict(sorted(counts.items())),
        "total_boxes": len(scores),
        "confidence_min": min(scores),
        "confidence_median": statistics.median(scores),
        "confidence_mean": statistics.mean(scores),
        "confidence_max": max(scores),
        "confidence_bins": bins,
        "predicted_classes": dict(sorted(labels.items())),
        "selection_groups": dict(groups),
        "manual_review": {"good": None, "partial": None, "failed": None, "reviewed": 0},
        "note": "No digit-level ground truth; no accuracy, mAP or IoU has been measured.",
    }
    (OUTPUT / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Phase II Experiment 0 — zero-shot OBB digit detection",
        "",
        "100 human-reviewed orientation-corrected Dryad samples. The scenario-based sample is deliberately stratified and is not a population estimate.",
        "",
        f"Model: `ntarekp/watermeter-yolov8-obb/stage2_digits.pt`; SHA-256 `{predictions['model_sha256']}`.",
        f"Input size {predictions['input_imgsz']}; confidence threshold {predictions['confidence_threshold']}; Ultralytics OBB NMS IoU {predictions['nms_iou']}. No extra grouping was applied.",
        "",
        f"Total predictions: {len(scores)}; median confidence {statistics.median(scores):.3f}, mean {statistics.mean(scores):.3f}, range {min(scores):.3f}–{max(scores):.3f}.",
        "",
        "## Predicted box counts",
        "",
        "| Boxes per image | Images |",
        "|---:|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += [
        "",
        "## Confidence distribution",
        "",
        "| Confidence | Boxes |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in bins.items()]
    lines += [
        "",
        f"Selection groups: {dict(groups)}. These are priority groups; source scenario flags overlap.",
        "",
        "## Review status",
        "",
        "Manual localization review has not been performed: good, partial and failed counts are pending. No localization-performance comparison across meter types or image-quality groups is available. There are no digit-level ground-truth boxes, so mAP and IoU are not measured.",
        "",
        "Initial visual spot checks (not a systematic review): `sample_000019` is severely obscured yet receives several boxes; `sample_000193` has overlapping predictions near a transition digit; some visually readable registers, such as `sample_000147`, show close alignment. These observations do not establish error rates.",
        "",
        "The `u` class means unclear according to the publisher; it is not a physical digit slot. Multiple boxes in a slot can be legitimate for transition states or may be false positives; review is needed.",
        "",
        "This checkpoint is worth reviewing as a possible fine-tuning starting point, but do not select it or claim reliable localization before completing the manual gallery review and obtaining digit-level ground truth.",
    ]
    (OUTPUT / "evaluation_summary.md").write_text("\n".join(lines) + "\n")
    print("Gallery, statistics and summary created")


if __name__ == "__main__":
    main()
