"""Colour segmentation and calibrated tabletop projection helpers."""

from dataclasses import dataclass
from math import cos, log, sin
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy


@dataclass(frozen=True)
class ColourDetection:
    """A named tabletop item detected in one RGB image."""

    name: str
    category: str
    pixel: Tuple[float, float]
    area_px: float
    size_px: Tuple[float, float]
    confidence: float


HSV_RANGES = {
    "red": (
        ((0, 100, 70), (10, 255, 255)),
        ((170, 100, 70), (180, 255, 255)),
    ),
    "blue": (
        ((100, 110, 60), (130, 255, 255)),
    ),
}


def detect_sorting_items(
    image_bgr: numpy.ndarray,
    min_cube_area_px: float = 120.0,
    min_zone_area_px: float = 900.0,
    kernel_size: int = 5,
) -> Dict[str, ColourDetection]:
    """
    Find one cube and one target zone per colour in an overhead image.

    Target zones are flat rectangles whereas a cube presents a comparatively
    square top/side silhouette to camera_main.  Use that shape cue before
    contour area: at the oblique Gazebo camera angle a cube can expose two
    faces and become larger in pixels than its flat marker.  One contour is
    never published twice as both a cube and a target zone.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {}
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    kernel = numpy.ones((kernel_size, kernel_size), dtype=numpy.uint8)
    results: Dict[str, ColourDetection] = {}
    for colour, ranges in HSV_RANGES.items():
        mask = _mask_for_ranges(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = [
            contour for contour in contours
            if cv2.contourArea(contour) >= min_cube_area_px
        ]
        if not candidates:
            continue
        cube_contour = min(candidates, key=_square_shape_score)
        cube = _make_detection(colour, cube_contour, "cube")
        if cube is not None:
            results[cube.name] = cube
        zone_candidates = [
            contour for contour in candidates
            if contour is not cube_contour
            and cv2.contourArea(contour) >= min_zone_area_px
        ]
        if zone_candidates:
            zone = _make_detection(
                colour, max(zone_candidates, key=_zone_shape_score), "target_zone"
            )
            if zone is not None:
                results[zone.name] = zone
    return results


def _square_shape_score(contour) -> tuple[float, float]:
    """Rank a compact square-like cube contour ahead of a flat zone."""
    _, _, width, height = cv2.boundingRect(contour)
    short_side = max(1.0, float(min(width, height)))
    aspect_score = abs(log(float(max(width, height)) / short_side))
    # Area is only a deterministic tie-breaker, never the classifier.
    return aspect_score, -float(cv2.contourArea(contour))


def _zone_shape_score(contour) -> tuple[float, float]:
    """Prefer the more elongated, broad flat marker among remaining contours."""
    _, _, width, height = cv2.boundingRect(contour)
    short_side = max(1.0, float(min(width, height)))
    return float(max(width, height)) / short_side, float(cv2.contourArea(contour))


def project_pixel_to_table(
    pixel: Tuple[float, float],
    camera_matrix: Iterable[float],
    translation_base_m: Iterable[float],
    rotation_rpy: Iterable[float],
    table_plane_z_m: float,
) -> Optional[Tuple[float, float, float]]:
    """Intersect a CameraInfo pixel ray with the calibrated table plane."""
    matrix = list(camera_matrix)
    if len(matrix) < 6:
        return None
    fx, fy, cx, cy = matrix[0], matrix[4], matrix[2], matrix[5]
    if fx == 0.0 or fy == 0.0:
        return None
    u, v = pixel
    ray_camera = numpy.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    translation = numpy.asarray(list(translation_base_m), dtype=float)
    if translation.shape != (3,):
        return None
    ray_base = rotation_matrix(rotation_rpy) @ ray_camera
    if abs(ray_base[2]) < 1e-6:
        return None
    scale = (float(table_plane_z_m) - translation[2]) / ray_base[2]
    if scale <= 0.0:
        return None
    point = translation + scale * ray_base
    return float(point[0]), float(point[1]), float(point[2])


def rotation_matrix(rotation_rpy: Iterable[float]) -> numpy.ndarray:
    """Return the camera-to-base rotation matrix from roll/pitch/yaw."""
    roll, pitch, yaw = [float(value) for value in rotation_rpy]
    rotation_x = numpy.array([
        [1.0, 0.0, 0.0],
        [0.0, cos(roll), -sin(roll)],
        [0.0, sin(roll), cos(roll)],
    ])
    rotation_y = numpy.array([
        [cos(pitch), 0.0, sin(pitch)],
        [0.0, 1.0, 0.0],
        [-sin(pitch), 0.0, cos(pitch)],
    ])
    rotation_z = numpy.array([
        [cos(yaw), -sin(yaw), 0.0],
        [sin(yaw), cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rotation_z @ rotation_y @ rotation_x


def _mask_for_ranges(hsv, ranges) -> numpy.ndarray:
    mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(
            hsv,
            numpy.asarray(lower, dtype=numpy.uint8),
            numpy.asarray(upper, dtype=numpy.uint8),
        )
    return mask


def _make_detection(
    colour: str,
    contour,
    category: str,
) -> Optional[ColourDetection]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0.0:
        return None
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]
    _, _, width, height = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    name = f"{colour}_{'cube' if category == 'cube' else 'target_zone'}"
    confidence = min(1.0, area / (900.0 if category == "cube" else 4000.0))
    return ColourDetection(
        name=name,
        category=category,
        pixel=(float(u), float(v)),
        area_px=area,
        size_px=(float(width), float(height)),
        confidence=confidence,
    )
