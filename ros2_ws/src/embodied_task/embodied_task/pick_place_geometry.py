"""Geometry-only dynamic pose generation for visual pick-and-place tasks."""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class PoseSpec:
    """A TCP pose expressed in ``base_link``."""

    position: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]


def build_pick_place_poses(
    object_xy: Iterable[float],
    region_xy: Iterable[float],
    config: Dict[str, object],
) -> Dict[str, PoseSpec]:
    """
    Build all six TCP poses from the current visual XY coordinates.

    ``jaw_center_offset_m`` is the rigid vector from the public MoveIt TCP to
    the centre between the two finger contact pads.  The selected downward
    tool orientation makes this offset part of every computed pose rather than
    incorrectly aiming the TCP at the cube centre.
    """
    object_x, object_y = [float(value) for value in object_xy]
    region_x, region_y = [float(value) for value in region_xy]
    object_orientation = tuple(
        float(value)
        for value in config.get("object_orientation_xyzw", config["orientation_xyzw"])
    )
    region_orientation = tuple(
        float(value)
        for value in config.get("region_orientation_xyzw", object_orientation)
    )
    table_z = float(config["table_plane_z_m"])
    cube_height = float(config["cube_height_m"])
    approach_height = float(config["approach_height_m"])
    lift_height = float(config["lift_height_m"])
    retreat_height = float(config["retreat_height_m"])
    grasp_band_height_offset = float(
        config.get("grasp_band_height_offset_m", 0.0)
    )
    offset = tuple(float(value) for value in config["jaw_center_offset_m"])
    approach_vector = tuple(
        float(value)
        for value in config.get("approach_vector_m", (0.0, 0.0, approach_height))
    )
    side_approach_vector = tuple(
        float(value)
        for value in config.get("side_approach_vector_m", (0.0, 0.0, 0.0))
    )
    # The vendor jaw collision pad is 42 mm tall while the cube is 45 mm
    # tall.  On the raised support its exact mid-band leaves only millimetres
    # above the support top, so normal trajectory tolerance makes the fingers
    # strike the pad before the cube.  A small, scene-configured upward shift
    # keeps the grasp inside the cube side faces while preserving clearance.
    object_contact_z = table_z + cube_height / 2.0 + grasp_band_height_offset
    place_contact_z = table_z + cube_height / 2.0 + grasp_band_height_offset

    def tcp_for_jaw_center(
        x: float, y: float, z: float, orientation: Tuple[float, float, float, float]
    ) -> PoseSpec:
        rotated_offset = _rotate_vector(offset, orientation)
        return PoseSpec(
            position=(
                x - rotated_offset[0],
                y - rotated_offset[1],
                z - rotated_offset[2],
            ),
            orientation_xyzw=orientation,
        )

    return {
        "object_approach": tcp_for_jaw_center(
            object_x + side_approach_vector[0],
            object_y + side_approach_vector[1],
            object_contact_z + approach_vector[2], object_orientation
        ),
        "object_side_pregrasp": tcp_for_jaw_center(
            object_x + side_approach_vector[0],
            object_y + side_approach_vector[1], object_contact_z,
            object_orientation
        ),
        "object_grasp": tcp_for_jaw_center(
            object_x, object_y, object_contact_z, object_orientation
        ),
        "object_lift": tcp_for_jaw_center(
            object_x, object_y, object_contact_z + lift_height, object_orientation
        ),
        "region_approach": tcp_for_jaw_center(
            region_x + approach_vector[0], region_y + approach_vector[1],
            place_contact_z + approach_vector[2], region_orientation
        ),
        "region_place": tcp_for_jaw_center(
            region_x, region_y, place_contact_z, region_orientation
        ),
        "region_retreat": tcp_for_jaw_center(
            region_x, region_y, place_contact_z + retreat_height, region_orientation
        ),
    }


def _rotate_vector(
    vector: Tuple[float, float, float],
    quaternion: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    """Rotate a local vector by an XYZW quaternion without ROS dependencies."""
    x, y, z, w = quaternion
    vx, vy, vz = vector
    # q * vector * q^-1, expanded for a unit quaternion.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
