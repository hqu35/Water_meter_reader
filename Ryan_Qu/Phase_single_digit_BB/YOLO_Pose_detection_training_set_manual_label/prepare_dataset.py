#!/usr/bin/env python3
"""Convert reviewed register quadrilaterals to a four-keypoint YOLO-Pose dataset."""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RYAN = HERE.parents[1]


def quad_is_valid(points, width, height):
    if len(points) != 4 or any(len(p) != 2 for p in points):
        return False
    if any(not math.isfinite(v) for point in points for v in point):
        return False
    if any(x < 0 or x >= width or y < 0 or y >= height for x, y in points):
        return False
    crosses = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(c > 0 for c in crosses) or all(c < 0 for c in crosses)


def make_label(points, width, height):
    xs, ys = zip(*points)
    xc = (min(xs) + max(xs)) / (2 * width)
    yc = (min(ys) + max(ys)) / (2 * height)
    bw = (max(xs) - min(xs)) / width
    bh = (max(ys) - min(ys)) / height
    values = [xc, yc, bw, bh] + [v for x, y in points for v in (x / width, y / height)]
    if bw <= 0 or bh <= 0 or any(v < 0 or v > 1 for v in values):
        raise ValueError('Invalid normalized box or keypoints')
    return '0 ' + ' '.join(f'{v:.10f}' for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=HERE / 'complete_annotations_for_training.json')
    parser.add_argument('--images', type=Path, default=RYAN / 'shared_dataset')
    parser.add_argument('--baseline', type=Path, default=RYAN / 'Phase_orientation_classifier/dataset_2000/manifest.json')
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = args.output
    if any((output / name).exists() for name in ('images', 'labels', 'split_manifest.json')):
        raise SystemExit('Output dataset already exists. Choose a new --output directory.')
    output.mkdir(parents=True, exist_ok=True)
    annotation_data = json.loads(args.annotations.read_text())
    records = annotation_data['annotations']
    baseline = {r['original_id']: r for r in json.loads(args.baseline.read_text())['records']}
    errors, prepared = [], []
    seen = set()
    for record in records:
        image_id = record['image_id']
        if image_id in seen:
            errors.append(f'{image_id}: duplicate image ID')
            continue
        seen.add(image_id)
        paths = [args.images / folder / f'{image_id}.png' for folder in ('clear_1500', 'clear_noise_500')]
        paths = [path for path in paths if path.is_file()]
        if len(paths) != 1 or image_id not in baseline:
            errors.append(f'{image_id}: expected one shared image and a baseline record, found {len(paths)} images')
            continue
        if record.get('status') != 'complete' or record.get('coordinate_space') != 'full_resolution_orientation_corrected_image' or record.get('digit_point_order') != ['TL', 'BL', 'BR', 'TR']:
            errors.append(f'{image_id}: incompatible annotation status or coordinate convention')
            continue
        with Image.open(paths[0]) as im:
            width, height = im.size
        if [width, height] != record['corrected_image_size_wh']:
            errors.append(f'{image_id}: image dimensions differ from annotation')
            continue
        quadrilaterals = record['digit_quadrilaterals']
        if len(quadrilaterals) != record['digit_count'] or not all(quad_is_valid(q, width, height) for q in quadrilaterals):
            errors.append(f'{image_id}: invalid quadrilateral or digit count')
            continue
        # The shared image is corrected according to the latest human review. If the
        # old annotation used the opposite orientation, transform coordinates, not
        # the image. Rotate corners and reorder them back to TL, BL, BR, TR.
        annotation_flip = record['orientation_correction'] == 'rotate_180'
        current_flip = baseline[image_id]['orientation_label'] == 'upside_down'
        transformed = annotation_flip != current_flip
        if transformed:
            quadrilaterals = [
                [[width - 1 - q[i][0], height - 1 - q[i][1]] for i in (2, 3, 0, 1)]
                for q in reversed(quadrilaterals)
            ]
        if not all(quad_is_valid(q, width, height) for q in quadrilaterals):
            errors.append(f'{image_id}: geometry invalid after orientation alignment')
            continue
        try:
            labels = [make_label(q, width, height) for q in quadrilaterals]
        except ValueError as exc:
            errors.append(f'{image_id}: {exc}')
            continue
        prepared.append({'id': image_id, 'path': paths[0], 'width': width, 'height': height,
                         'quads': quadrilaterals, 'labels': labels, 'orientation_alignment_180': transformed,
                         'group': paths[0].parent.name, 'mode': record.get('annotation_mode')})
    if errors:
        print('\n'.join(errors))
        raise SystemExit(f'{len(errors)} invalid records; dataset not written')
    if len(prepared) < 103:
        raise SystemExit('At least 103 images are needed for a 100-image held-out test split.')
    # Each image ID identifies one original Dryad image. No augmented copies are
    # included; assigning the entire image keeps all of its digit slots together.
    prepared.sort(key=lambda r: r['id'])
    random.Random(args.seed).shuffle(prepared)
    val_count = 40 if len(prepared) == 664 else max(1, round((len(prepared) - 100) * 40 / 564))
    test_count = 100
    train_count = len(prepared) - val_count - test_count
    splits = {'train': prepared[:train_count], 'val': prepared[train_count:train_count + val_count], 'test': prepared[train_count + val_count:]}
    manifest = {'seed': args.seed, 'source': 'reviewed register crops; original Dryad image ID is the split group',
                'grouping_note': 'No augmented variants identified in this selection.', 'splits': {}}
    for split, images in splits.items():
        image_dir = output / 'images' / split
        label_dir = output / 'labels' / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        manifest['splits'][split] = []
        for item in images:
            shutil.copyfile(item['path'], image_dir / f"{item['id']}.png")
            (label_dir / f"{item['id']}.txt").write_text('\n'.join(item['labels']) + '\n')
            manifest['splits'][split].append({'image_id': item['id'], 'source_group_id': item['id'],
                'source_category': item['group'], 'digit_count': len(item['quads']),
                'annotation_mode': item['mode'], 'annotation_coordinates_rotated_180': item['orientation_alignment_180']})
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'watermeter_pose.yaml').write_text('path: /content/watermeter_pose_dataset\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: digit_slot\nkpt_shape: [4, 2]\n')
    visual_dir = output / 'visualizations'
    visual_dir.mkdir()
    samples = random.Random(args.seed).sample(prepared, min(25, len(prepared)))
    visual_items = [(item, f"{item['id']}.png") for item in samples]
    visual_items += [(item, f"{item['id']}_orientation_alignment.png")
                     for item in prepared if item['orientation_alignment_180']]
    for item, filename in visual_items:
        with Image.open(item['path']) as source:
            image = source.convert('RGB')
        # Scale up for inspection without affecting labels or stored images.
        factor = max(1, math.ceil(900 / item['width']))
        image = image.resize((item['width'] * factor, item['height'] * factor))
        draw = ImageDraw.Draw(image)
        for q in item['quads']:
            scaled = [(x * factor, y * factor) for x, y in q]
            xs, ys = zip(*scaled)
            draw.rectangle((min(xs), min(ys), max(xs), max(ys)), outline='yellow', width=2)
            draw.line(scaled + [scaled[0]], fill='cyan', width=2)
            for (x, y), name in zip(scaled, ('TL', 'BL', 'BR', 'TR')):
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='red')
                draw.text((x + 3, y - 12), name, fill='red', stroke_width=1, stroke_fill='white')
        image.save(visual_dir / filename)
    report = {'images': len(prepared), 'split_counts': {s: len(v) for s, v in splits.items()},
              'digit_instances': sum(len(r['quads']) for r in prepared), 'missing_or_invalid': errors,
              'coordinate_alignment_180': [r['id'] for r in prepared if r['orientation_alignment_180']],
              'seed': args.seed, 'visualizations': min(25, len(prepared)),
              'additional_visualization': 'visualizations/train18154_orientation_alignment.png'}
    (output / 'conversion_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
