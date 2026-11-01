"""Pixel-edge rectangles and orientation-preserving register rectification."""

import cv2
import numpy as np


def ordered_corners(points):
    """Clockwise in image coordinates, long edge first, directed right/down.

    Corner zero is the local top-left, not necessarily the image's topmost point.
    This fixes winding and a deterministic 180-degree convention, not text polarity.
    """
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite corners")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    candidates = []
    for index in range(4):
        q = np.roll(points, -index, axis=0)
        u, v = q[1] - q[0], q[2] - q[1]
        width, height = np.linalg.norm(u), np.linalg.norm(v)
        if min(width, height) < 1e-7:
            continue
        if width + 1e-4 < height:
            continue
        if u[0] < -1e-6 or (abs(u[0]) <= 1e-6 and u[1] < 0):
            continue
        candidates.append((abs(np.arctan2(u[1], u[0])), q))
    if not candidates:
        raise ValueError("Degenerate rectangle")
    q = min(candidates, key=lambda pair: pair[0])[1]
    u, v = q[1] - q[0], q[2] - q[1]
    if u[0] * v[1] - u[1] * v[0] <= 0:
        raise ValueError("Corners must have positive winding")
    if abs(np.dot(u, v)) > 1e-4 * np.linalg.norm(u) * np.linalg.norm(v):
        raise ValueError("Corners are not rectangular")
    if not np.allclose(q[0] + q[2], q[1] + q[3], atol=1e-3):
        raise ValueError("Corners are not a parallelogram")
    return q


def rectangle_from_mask(mask):
    """Fit all foreground pixel cells, including isolated/boundary pixels.

    A pixel (x,y) occupies [x,x+1] x [y,y+1]. Expanding the foreground hull
    by the unit square avoids clipping the last row/column or a one-pixel mask.
    """
    foreground = np.asarray(mask) != 0
    if foreground.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    y, x = np.nonzero(foreground)
    if not len(x):
        raise ValueError("Empty foreground mask")
    centers = np.column_stack([x, y]).astype(np.float32)
    hull = cv2.convexHull(centers).reshape(-1, 2)
    offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    cells = (hull[:, None, :] + offsets).reshape(-1, 2)
    return ordered_corners(cv2.boxPoints(cv2.minAreaRect(cells)))


def required_padding(corners, width, height):
    """Never clip a rotated rectangle into a non-rectangle to normalize it."""
    low = np.asarray(corners).min(axis=0)
    high = np.asarray(corners).max(axis=0)
    return [
        int(np.ceil(max(0, -low[0] - 1e-4))),
        int(np.ceil(max(0, -low[1] - 1e-4))),
        int(np.ceil(max(0, high[0] - width - 1e-4))),
        int(np.ceil(max(0, high[1] - height - 1e-4))),
    ]


def normalized_corners(corners, width, height):
    q = ordered_corners(corners) / [width, height]
    if (q < -1e-6).any() or (q > 1 + 1e-6).any():
        raise ValueError("Rectangle requires padding before label normalization")
    return np.clip(q, 0, 1)


def mask_is_covered(mask, corners, tolerance=2e-3):
    """Check all four cell corners of the foreground convex hull, not just centers."""
    y, x = np.nonzero(np.asarray(mask))
    if not len(x):
        return False
    hull = cv2.convexHull(np.column_stack([x, y]).astype(np.float32)).reshape(-1, 2)
    cells = (hull[:, None, :] + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])).reshape(
        -1, 2
    )
    q = ordered_corners(corners)
    for start, end in zip(q, np.roll(q, -1, axis=0)):
        edge = end - start
        delta = cells - start
        distance = (edge[0] * delta[:, 1] - edge[1] * delta[:, 0]) / np.linalg.norm(
            edge
        )
        if distance.min() < -tolerance:
            return False
    return True


def aligned_crop(image, corners, margin_fraction=0.03):
    """Warp a rectangle's long axis horizontally without reflection or transpose.

    Coordinates use pixel edges; OpenCV samples pixel centers. The half-pixel
    translations below reconcile those conventions. Border pixels are replicated
    if the predicted rectangle/margin goes outside the source image.
    """
    if not 0 <= margin_fraction <= 0.5:
        raise ValueError("Margin fraction must be between 0 and 0.5")
    q = ordered_corners(corners)
    center = q.mean(axis=0)
    expanded = center + (q - center) * (1 + 2 * margin_fraction)
    width = max(1, int(np.ceil(np.linalg.norm(expanded[1] - expanded[0]))))
    height = max(1, int(np.ceil(np.linalg.norm(expanded[2] - expanded[1]))))
    destination = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], np.float32
    )
    edge_transform = cv2.getPerspectiveTransform(
        expanded.astype(np.float32), destination
    )
    to_edges = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]])
    to_centers = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]])
    transform = to_centers @ edge_transform @ to_edges
    crop = cv2.warpPerspective(
        np.asarray(image),
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    h, w = np.asarray(image).shape[:2]
    report = {
        "corners_xy": q.tolist(),
        "sampling_corners_xy": expanded.tolist(),
        "source_edge_to_crop_edge": edge_transform.tolist(),
        "width": width,
        "height": height,
        "margin_fraction": margin_fraction,
        "orientation_ambiguous_180": True,
        "near_square": bool(
            np.linalg.norm(q[1] - q[0]) / np.linalg.norm(q[2] - q[1]) < 1.1
        ),
        "border_replicated": bool((expanded < 0).any() or (expanded > [w, h]).any()),
    }
    return crop, report


def rotated_iou(first, second):
    """Exact polygon IoU via float64 convex clipping (no enclosing-box approximation).

    Avoid float32 intersectConvexConvex instability for almost coincident edges.
    """
    first, second = ordered_corners(first), ordered_corners(second)
    polygon = list(first)
    for start, end in zip(second, np.roll(second, -1, axis=0)):
        edge = end - start

        def distance(point):
            delta = point - start
            return edge[0] * delta[1] - edge[1] * delta[0]

        previous = polygon[-1] if polygon else None
        clipped = []
        for current in polygon:
            old_distance, new_distance = distance(previous), distance(current)
            old_inside, new_inside = old_distance >= 0, new_distance >= 0
            if old_inside != new_inside:
                fraction = old_distance / (old_distance - new_distance)
                clipped.append(previous + fraction * (current - previous))
            if new_inside:
                clipped.append(current)
            previous = current
        polygon = clipped

    def area(points):
        if len(points) < 3:
            return 0.0
        points = np.asarray(points) - np.asarray(points).mean(axis=0)
        return (
            abs(
                float(
                    np.sum(
                        points[:, 0] * np.roll(points[:, 1], -1)
                        - points[:, 1] * np.roll(points[:, 0], -1)
                    )
                )
            )
            / 2
        )

    intersection = area(polygon)
    union = area(first) + area(second) - intersection
    return float(np.clip(intersection / union, 0, 1)) if union > 0 else 0.0
