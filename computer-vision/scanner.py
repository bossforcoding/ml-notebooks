"""Find a sheet of paper in a photo and rectify it with a homography.

The pipeline: segment the bright sheet, detect its borders with the Hough
transform, intersect them to obtain the four corners, and warp the image so
that the sheet becomes a fronto-parallel A4 rectangle.
"""

import itertools

import numpy as np
from scipy import ndimage as ndi
from skimage import color, feature, filters, io, measure, morphology, transform
from skimage.transform import hough_line, hough_line_peaks

A4_RATIO = 297 / 210


def load_gray(path, scale=0.25):
    image = io.imread(path)
    return transform.rescale(color.rgb2gray(image), scale, anti_aliasing=True)


def sheet_mask(gray):
    """Largest bright region, with the handwriting filled in."""
    smooth = filters.gaussian(gray, 3)
    mask = smooth > filters.threshold_otsu(smooth)
    mask = ndi.binary_fill_holes(morphology.binary_closing(mask, morphology.disk(10)))
    labels = measure.label(mask)
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return labels == largest


def border_lines(mask, num_peaks=8):
    """Hough lines (angle, distance) along the border of the sheet mask."""
    edges = feature.canny(mask.astype(float), sigma=2)
    h, theta, dist = hough_line(edges, np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False))
    _, angles, dists = hough_line_peaks(h, theta, dist, num_peaks=num_peaks, min_distance=20, min_angle=8,
                                        threshold=0.15 * h.max())
    return edges, h, theta, dist, list(zip(angles, dists))


def intersect(l1, l2):
    """Intersection of two lines given in Hough normal form x cos(a) + y sin(a) = d."""
    (a1, d1), (a2, d2) = l1, l2
    A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [d1, d2])          # (x, y)


def order_corners(pts):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, float)
    centre = pts.mean(0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]                 # counter-clockwise starting from the left
    start = np.argmin(pts.sum(1))                 # top-left has the smallest x + y
    return np.roll(pts, -start, axis=0)


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def find_corners(mask, lines, min_overlap=0.9):
    """Choose two pairs of roughly parallel lines whose intersections best match the sheet mask.

    Returns the ordered corners, or None if no candidate quadrilateral covers the mask well
    (for example when part of the sheet is outside the photo).
    """
    h, w = mask.shape
    area = mask.sum()
    best, best_score = None, 0.0
    for quad in itertools.combinations(lines, 4):
        for pairing in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
            (i, j), (k, m) = pairing
            parallel = lambda a, b: abs(np.sin(quad[a][0] - quad[b][0])) < np.sin(np.radians(35))
            if not (parallel(i, j) and parallel(k, m)) or parallel(i, k):
                continue
            pts = [intersect(quad[p], quad[q]) for p in (i, j) for q in (k, m)]
            if any(p is None for p in pts):
                continue
            pts = order_corners(pts)
            if np.any(pts < -0.05 * max(h, w)) or np.any(pts[:, 0] > 1.05 * w) or np.any(pts[:, 1] > 1.05 * h):
                continue
            quad_area = polygon_area(pts)
            score = min(quad_area, area) / max(quad_area, area)
            if score > best_score:
                best, best_score = pts, score
    return (best, best_score) if best_score >= min_overlap else (None, best_score)


def rectify(gray, corners, width=700):
    """Warp the quadrilateral to an A4 rectangle (portrait or landscape, following the photo)."""
    tl, tr, br, bl = corners
    w_est = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h_est = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if h_est >= w_est:
        out_w, out_h = width, int(width * A4_RATIO)
    else:
        out_w, out_h = int(width * A4_RATIO), width
    target = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], float)
    tform = transform.ProjectiveTransform()
    tform.estimate(target, corners)               # maps output coordinates to input coordinates
    return transform.warp(gray, tform, output_shape=(out_h, out_w)), tform
