"""Non-moving IK scan for the flat red-zone placement pose."""

from math import cos, radians, sin

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node


class PlacementIkScan(Node):
    def __init__(self):
        super().__init__("placement_ik_scan")
        self.client = self.create_client(GetPositionIK, "/compute_ik")

    def run(self):
        if not self.client.wait_for_service(timeout_sec=8.0):
            raise RuntimeError("/compute_ik unavailable")
        # MotionExecutor converts a level public TCP target to the active
        # l5_l6_urdf_asm tip by adding the fixed 74 mm TCP offset in Z.
        for target_x in (-0.280, -0.300, -0.320, -0.340, -0.360, -0.380, -0.400, -0.420):
            for yaw_deg in (0,):
                half = radians(yaw_deg) / 2.0
                for tcp_z in (0.229, 0.239, 0.320):
                    request = GetPositionIK.Request()
                    ik = request.ik_request
                    ik.group_name = "arm"
                    ik.ik_link_name = "l5_l6_urdf_asm"
                    ik.pose_stamped = PoseStamped()
                    ik.pose_stamped.header.frame_id = "base_link"
                    ik.pose_stamped.pose.position.x = target_x
                    ik.pose_stamped.pose.position.y = 0.100
                    ik.pose_stamped.pose.position.z = tcp_z + 0.074
                    ik.pose_stamped.pose.orientation.z = sin(half)
                    ik.pose_stamped.pose.orientation.w = cos(half)
                    ik.timeout.sec = 2
                    ik.robot_state.joint_state.name = [
                        "L1_joint", "L2_joint", "L3_joint", "L4_joint",
                        "L5_joint", "L6_joint",
                    ]
                    ik.robot_state.joint_state.position = [
                        0.0, 0.785, -0.785, 0.0, 0.0, 0.0,
                    ]
                    ik.robot_state.is_diff = False
                    future = self.client.call_async(request)
                    rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
                    response = future.result() if future.done() else None
                    code = response.error_code.val if response is not None else -1
                    if code == MoveItErrorCodes.SUCCESS:
                        print(f"x={target_x:.3f} yaw={yaw_deg:+d} tcp_z={tcp_z:.3f} success")


def main():
    rclpy.init()
    node = PlacementIkScan()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
