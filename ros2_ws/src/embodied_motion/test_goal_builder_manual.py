from pathlib import Path

import yaml
from geometry_msgs.msg import PoseStamped

from embodied_motion.moveit_goal_builder import MoveItGoalBuilder


def main() -> None:
    config_path = Path(
        "embodied_motion/config/motion_config.yaml"
    )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    builder = MoveItGoalBuilder(config)

    # 1. 合法末端位姿
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = 0.30
    pose.pose.position.y = 0.00
    pose.pose.position.z = 0.30
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0

    goal = builder.build_pose_goal(
        target_pose=pose,
        velocity_scale=0.2,
        acceleration_scale=0.2,
    )

    assert goal.request.group_name == "arm"
    assert goal.request.planner_id == "RRTConnect"
    assert len(goal.request.goal_constraints) == 1

    constraints = goal.request.goal_constraints[0]

    assert len(constraints.position_constraints) == 1
    assert len(constraints.orientation_constraints) == 1
    assert goal.planning_options.plan_only is False

    print("合法 Pose Goal 构造成功")
    print("规划组:", goal.request.group_name)
    print("规划器:", goal.request.planner_id)
    print(
        "位置约束链接:",
        constraints.position_constraints[0].link_name,
    )
    print(
        "姿态约束链接:",
        constraints.orientation_constraints[0].link_name,
    )

    # 2. 非法坐标系
    invalid_frame_pose = PoseStamped()
    invalid_frame_pose.header.frame_id = "camera_link"
    invalid_frame_pose.pose.orientation.w = 1.0

    try:
        builder.build_pose_goal(
            target_pose=invalid_frame_pose,
            velocity_scale=0.2,
            acceleration_scale=0.2,
        )
        raise AssertionError("非法坐标系未被拒绝")
    except ValueError as error:
        print("非法坐标系已正确拒绝:", error)

    # 3. 全零四元数
    zero_quaternion_pose = PoseStamped()
    zero_quaternion_pose.header.frame_id = "base_link"
    zero_quaternion_pose.pose.orientation.x = 0.0
    zero_quaternion_pose.pose.orientation.y = 0.0
    zero_quaternion_pose.pose.orientation.z = 0.0
    zero_quaternion_pose.pose.orientation.w = 0.0

    try:
        builder.build_pose_goal(
            target_pose=zero_quaternion_pose,
            velocity_scale=0.2,
            acceleration_scale=0.2,
        )
        raise AssertionError("全零四元数未被拒绝")
    except ValueError as error:
        print("全零四元数已正确拒绝:", error)

    # 4. 未归一化四元数
    unnormalized_pose = PoseStamped()
    unnormalized_pose.header.frame_id = "base_link"
    unnormalized_pose.pose.orientation.x = 0.0
    unnormalized_pose.pose.orientation.y = 0.0
    unnormalized_pose.pose.orientation.z = 0.0
    unnormalized_pose.pose.orientation.w = 2.0

    try:
        builder.build_pose_goal(
            target_pose=unnormalized_pose,
            velocity_scale=0.2,
            acceleration_scale=0.2,
        )
        raise AssertionError("未归一化四元数未被拒绝")
    except ValueError as error:
        print("未归一化四元数已正确拒绝:", error)

    # 5. 非法速度比例
    try:
        builder.build_pose_goal(
            target_pose=pose,
            velocity_scale=1.5,
            acceleration_scale=0.2,
        )
        raise AssertionError("非法速度比例未被拒绝")
    except ValueError as error:
        print("非法速度比例已正确拒绝:", error)

    print("MoveItGoalBuilder 校验全部通过")


if __name__ == "__main__":
    main()
