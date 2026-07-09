import os
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from embodied_interfaces.srv import MoveNamedPose

class MotionExecutorNode(Node):
    def __init__(self):
        super().__init__('motion_executor_node')

        self.config = self.load_motion_config()

        self.robot_config = self.config.get('robot', {})
        self.controllers_config = self.config.get('controllers', {})
        self.named_poses = self.config.get('named_poses', {})
        self.motion_config = self.config.get('motion', {})

        self.print_config_summary()
        self.create_action_clients()
        self.check_action_servers()
        self.go_named_pose_service = self.create_service(
            MoveNamedPose,
            '/motion/go_named_pose',
            self.handle_go_named_pose
        )

        self.get_logger().info('Service ready: /motion/go_named_pose')

    def load_motion_config(self):
        package_share_dir = get_package_share_directory('embodied_motion')
        config_path = os.path.join(package_share_dir, 'config', 'motion_config.yaml')

        self.get_logger().info(f'Loading motion config from: {config_path}')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Motion config file not found: {config_path}')

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def print_config_summary(self):
        robot_name = self.robot_config.get('name', 'unknown')
        base_frame = self.robot_config.get('base_frame', 'unknown')

        arm_config = self.controllers_config.get('arm', {})
        gripper_config = self.controllers_config.get('gripper', {})

        arm_action = arm_config.get('action_name', 'unknown')
        gripper_action = gripper_config.get('action_name', 'unknown')
        arm_joints = arm_config.get('joint_names', [])

        self.get_logger().info('========== Motion Executor Config ==========')
        self.get_logger().info(f'Robot name: {robot_name}')
        self.get_logger().info(f'Base frame: {base_frame}')
        self.get_logger().info(f'Arm action: {arm_action}')
        self.get_logger().info(f'Gripper action: {gripper_action}')
        self.get_logger().info(f'Arm joints: {arm_joints}')

        self.get_logger().info('Named poses:')
        for pose_name, pose_data in self.named_poses.items():
            self.get_logger().info(f'  - {pose_name}: {pose_data.get("positions", [])}')

        self.get_logger().info('===========================================')
    
    def create_action_clients(self):
        arm_config = self.controllers_config.get('arm', {})
        gripper_config = self.controllers_config.get('gripper', {})

        self.arm_action_name = arm_config.get('action_name', '/arm_controller/follow_joint_trajectory')
        self.gripper_action_name = gripper_config.get('action_name', '/gripper_controller/follow_joint_trajectory')

        self.arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.arm_action_name
        )

        self.gripper_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.gripper_action_name
        )

    def check_action_servers(self):
        self.get_logger().info('Checking action servers...')

        arm_ready = self.arm_action_client.wait_for_server(timeout_sec=2.0)
        gripper_ready = self.gripper_action_client.wait_for_server(timeout_sec=2.0)

        if arm_ready:
            self.get_logger().info(f'Arm action server connected: {self.arm_action_name}')
        else:
            self.get_logger().warn(f'Arm action server not available: {self.arm_action_name}')

        if gripper_ready:
            self.get_logger().info(f'Gripper action server connected: {self.gripper_action_name}')
        else:
            self.get_logger().warn(f'Gripper action server not available: {self.gripper_action_name}')

    def build_joint_trajectory_goal(self, joint_names, positions, duration_sec):
        goal_msg = FollowJointTrajectory.Goal()

        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec - int(duration_sec)) * 1e9)
        )

        trajectory.points.append(point)
        goal_msg.trajectory = trajectory

        return goal_msg

    def send_named_pose(self, pose_name):
        if pose_name not in self.named_poses:
            self.get_logger().error(f'Named pose not found: {pose_name}')
            return False

        arm_config = self.controllers_config.get('arm', {})
        joint_names = arm_config.get('joint_names', [])
        positions = self.named_poses[pose_name].get('positions', [])
        duration_sec = float(self.motion_config.get('default_duration_sec', 3.0))

        if len(joint_names) != len(positions):
            self.get_logger().error(
                f'Joint count mismatch: {len(joint_names)} joint names, '
                f'{len(positions)} positions'
            )
            return False

        self.get_logger().info(f'Sending named pose: {pose_name}')
        self.get_logger().info(f'Joint names: {joint_names}')
        self.get_logger().info(f'Positions: {positions}')
        self.get_logger().info(f'Duration: {duration_sec} sec')

        goal_msg = self.build_joint_trajectory_goal(
            joint_names=joint_names,
            positions=positions,
            duration_sec=duration_sec
        )

        send_goal_future = self.arm_action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.on_goal_response)

        return True

    def handle_go_named_pose(self, request, response):
        pose_name = request.pose_name.strip()

        self.get_logger().info(f'Received go_named_pose request: {pose_name}')

        if not pose_name:
            response.success = False
            response.message = 'pose_name is empty'
            return response

        if pose_name not in self.named_poses:
            response.success = False
            response.message = f'Unknown named pose: {pose_name}'
            self.get_logger().error(response.message)
            return response

        success = self.send_named_pose(pose_name)

        response.success = bool(success)
        response.message = f'Goal sent for named pose: {pose_name}' if success else f'Failed to send named pose: {pose_name}'

        return response

    def on_goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Motion goal rejected by arm controller')
            return

        self.get_logger().info('Motion goal accepted by arm controller')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_motion_result)

    def on_motion_result(self, future):
        result = future.result().result
        error_code = result.error_code
        error_string = result.error_string

        if error_code == 0:
            self.get_logger().info('Motion execution succeeded')
        else:
            self.get_logger().error(
                f'Motion execution failed, error_code={error_code}, '
                f'error_string="{error_string}"'
            )

def main(args=None):
    rclpy.init(args=args)

    node = MotionExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
