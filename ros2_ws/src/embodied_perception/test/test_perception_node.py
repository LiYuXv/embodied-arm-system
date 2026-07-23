"""Unit tests for colour-independent overhead perception."""

import math

import cv2
import numpy

from embodied_perception.colour_localizer import (
    detect_sorting_items,
    project_pixel_to_table,
)


def test_detects_red_and_blue_cubes_and_target_zones() -> None:
    """Shape separates compact cubes from larger flat coloured zones."""
    image = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
    cv2.rectangle(image, (80, 100), (120, 140), (0, 0, 255), -1)
    cv2.rectangle(image, (220, 90), (340, 180), (0, 0, 255), -1)
    cv2.rectangle(image, (80, 300), (120, 340), (255, 0, 0), -1)
    cv2.rectangle(image, (220, 280), (340, 370), (255, 0, 0), -1)

    detections = detect_sorting_items(image, 100.0, 1000.0)

    assert set(detections) == {
        "red_cube", "red_target_zone", "blue_cube", "blue_target_zone"
    }
    assert detections["red_cube"].pixel == (100.0, 120.0)
    assert detections["blue_target_zone"].category == "target_zone"


def test_projects_camera_info_pixel_onto_base_link_table_plane() -> None:
    """The camera ray must produce a tabletop coordinate, not image pixels."""
    point = project_pixel_to_table(
        (320.0, 240.0),
        [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
        [0.2, -0.1, 1.0],
        [math.pi, 0.0, 0.0],
        0.03,
    )

    assert point is not None
    assert numpy.allclose(point, (0.2, -0.1, 0.03))
