"""Tests for vision-driven dynamic pick-and-place pose construction."""

from embodied_task.pick_place_geometry import build_pick_place_poses


CONFIG = {
    "table_plane_z_m": 0.03,
    "cube_height_m": 0.07,
    "approach_height_m": 0.12,
    "lift_height_m": 0.16,
    "retreat_height_m": 0.16,
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "jaw_center_offset_m": [0.0, 0.0, -0.092],
}


def test_all_pick_place_poses_follow_current_visual_coordinates() -> None:
    """Approach, grasp, lift, place, and retreat are not named fixed poses."""
    poses = build_pick_place_poses((-0.40, 0.18), (-0.72, 0.18), CONFIG)

    assert set(poses) == {
        "object_approach", "object_side_pregrasp", "object_grasp", "object_lift",
        "region_approach", "region_place", "region_retreat",
    }
    assert poses["object_approach"].position[:2] == (-0.40, 0.18)
    assert poses["object_grasp"].position[:2] == (-0.40, 0.18)
    assert poses["region_place"].position[:2] == (-0.72, 0.18)
    assert poses["object_approach"].position[2] > poses["object_grasp"].position[2]
    assert poses["object_lift"].position[2] > poses["object_grasp"].position[2]
    assert poses["region_retreat"].position[2] > poses["region_place"].position[2]
    assert poses["object_grasp"].position[2] == 0.157


def test_visual_update_recomputes_every_corresponding_xy() -> None:
    """A new detection changes both grasp and placement pose families."""
    first = build_pick_place_poses((-0.40, 0.18), (-0.72, 0.18), CONFIG)
    second = build_pick_place_poses((-0.35, -0.15), (-0.66, -0.15), CONFIG)

    assert first["object_grasp"].position[:2] != second["object_grasp"].position[:2]
    assert first["region_approach"].position[:2] != second["region_approach"].position[:2]
