"""Unit tests for MoveIt pose-goal construction.

These tests intentionally inspect serialized MoveIt messages. A valid message
prevents regressions in the pose-constraint path before a live test is run.
"""

from pathlib import Path

import pytest
import yaml
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

from embodied_motion.moveit_goal_builder import MoveItGoalBuilder


@pytest.fixture
def builder():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "embodied_motion"
        / "config"
        / "motion_config.yaml"
    )
    with config_path.open(encoding="utf-8") as config_file:
        return MoveItGoalBuilder(yaml.safe_load(config_file))


def make_pose() -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = 0.30
    pose.pose.position.y = -0.02
    pose.pose.position.z = 0.28
    pose.pose.orientation.w = 1.0
    return pose


def test_pose_goal_uses_a_spherical_position_region(builder):
    """The target point and tolerance radius must be expressed in base_link."""
    pose = make_pose()
    goal = builder.build_pose_goal(pose, 0.2, 0.2)

    constraints = goal.request.goal_constraints[0]
    position = constraints.position_constraints[0]

    assert position.header.frame_id == "base_link"
    assert position.link_name == "end_effector"
    primitive = position.constraint_region.primitives[0]
    region_pose = position.constraint_region.primitive_poses[0]
    assert primitive.type == SolidPrimitive.SPHERE
    assert primitive.dimensions == pytest.approx([0.001])
    assert region_pose.position.x == pytest.approx(pose.pose.position.x)
    assert region_pose.position.y == pytest.approx(pose.pose.position.y)
    # The KDL chain now terminates at the public TCP, so no second offset is
    # applied while constraints are constructed.
    assert region_pose.position.z == pytest.approx(pose.pose.position.z)
    assert region_pose.orientation.w == 1.0


def test_pose_goal_uses_complete_orientation_constraint(builder):
    """The pose goal constrains every orientation axis for 6-DOF IK."""
    pose = make_pose()
    goal = builder.build_pose_goal(pose, 0.2, 0.2)

    orientation = goal.request.goal_constraints[0].orientation_constraints[0]

    assert orientation.header.frame_id == "base_link"
    assert orientation.link_name == "end_effector"
    assert orientation.orientation.w == 1.0
    assert orientation.absolute_x_axis_tolerance == pytest.approx(0.01)
    assert orientation.absolute_y_axis_tolerance == pytest.approx(0.01)
    assert orientation.absolute_z_axis_tolerance == pytest.approx(0.01)
    assert goal.request.start_state.is_diff is True


def test_pose_goal_rejects_a_non_normalized_quaternion(builder):
    pose = make_pose()
    pose.pose.orientation.w = 2.0

    with pytest.raises(ValueError, match="normalized"):
        builder.build_pose_goal(pose, 0.2, 0.2)
