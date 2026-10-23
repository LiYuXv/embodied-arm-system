#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point, PoseStamped, Pose
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, PlanningScene, CollisionObject
)
from std_srvs.srv import Empty
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header, Float64MultiArray
from control_msgs.action import GripperCommand
from tf2_ros import TransformListener, Buffer
import time
import threading
import math
import numpy as np
try:
    if not hasattr(np, 'float'):
        np.float = float
except Exception:
    pass
from tf_transformations import quaternion_from_matrix


class RobustPicker(Node):
    def __init__(self):
        super().__init__('robust_apple_picker')

        # ---------------- 通用参数 ----------------
        self.GRIPPER_LENGTH        = -0.02
        self.APPLE_RADIUS          = 0.03
        self.PROTECTION_RADIUS     = 0.06
        self.HOVER_OFFSET          = 0.08
        self.CUBE_ORIGIN_X         = 0.3
        self.CUBE_ORIGIN_Y         = 0.0
        self.CUBE_ORIGIN_Z         = 0.44

        # 交接点 & 右臂放置点
        self.HANDOVER_POS          = (0.362, 0.020, 0.563)
        self.PLACE_POS             = (0.35, -0.2, 0.44)

        # 缓存 & 状态
        self.cached_target         = None
        self.cached_z_surface      = None
        self.last_executed_target  = None
        self.is_attached_left      = False   # 新增
        self.is_attached_right     = False   # 原名 is_attached

        # ---------- 订阅 / 发布 ----------
        self.sub          = self.create_subscription(Point, '/apple_location', self.coord_callback, 10)
        self.scene_pub    = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.cube_pub     = self.create_publisher(Float64MultiArray, '/cube_controller/commands', 10)

        self.tf_buffer    = Buffer()
        self.tf_listener  = TransformListener(self.tf_buffer, self)

        # 调试关节
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joints  = {}
        self.debug_timer     = self.create_timer(1.0, self.print_finger_debug)

        # ---------- MoveIt / Gripper 客户端 ----------
        # 右臂（原有）
        self._move_client_right  = ActionClient(self, MoveGroup, 'move_action')
        self._gripper_client_right = ActionClient(self, GripperCommand, '/right_hand_controller/gripper_cmd')
        self.group_name_right    = "right_arm"
        self.ee_link_right       = "openarm_right_hand_tcp"

        # 左臂（新增）
        self._move_client_left   = ActionClient(self, MoveGroup, 'move_action')
        self._gripper_client_left = ActionClient(self, GripperCommand, '/left_hand_controller/gripper_cmd')
        self.group_name_left     = "left_arm"
        self.ee_link_left        = "openarm_left_hand_tcp"

        self.is_busy = True
        self.get_logger().info("Waiting for servers...")
        self._move_client_right.wait_for_server()
        self._gripper_client_right.wait_for_server()
        self._move_client_left.wait_for_server()
        self._gripper_client_left.wait_for_server()
        self.get_logger().info("All servers ready")

        # 启动线程
        threading.Thread(target=self.magic_loop, daemon=True).start()
        threading.Thread(target=self.init_robot, daemon=True).start()

    # =============================================================
    #  魔方跟随  ->  同时支持左右臂（根据 is_attached_* 开关）
    # =============================================================
    def teleport_cube(self, world_x, world_y, world_z):
        joint_x = world_x - self.CUBE_ORIGIN_X
        joint_y = world_y - self.CUBE_ORIGIN_Y
        joint_z = world_z - self.CUBE_ORIGIN_Z
        msg = Float64MultiArray()
        msg.data = [float(joint_x), float(joint_y), float(joint_z)]
        self.cube_pub.publish(msg)

    def magic_loop(self):
        rate = self.create_rate(500)
        while rclpy.ok():
            if self.is_attached_left:     # 优先左臂
                self._teleport_follow_ee(self.ee_link_left)
            elif self.is_attached_right:  # 右臂
                self._teleport_follow_ee(self.ee_link_right)
            rate.sleep()

    # def _teleport_follow_ee(self, ee_link):
    #     try:
    #         if self.tf_buffer.can_transform('world', ee_link, rclpy.time.Time()):
    #             t = self.tf_buffer.lookup_transform('world', ee_link, rclpy.time.Time())
    #             tx = t.transform.translation.x
    #             ty = t.transform.translation.y
    #             tz = t.transform.translation.z
    #             self.teleport_cube(tx, ty, tz)
    #     except Exception:
    #         pass
    def _teleport_follow_ee(self, ee_link):
        try:
            # 修改点 1: 不要显式检查 can_transform，直接 try-except lookup 更快
            # 修改点 2: 使用 rclpy.time.Time(seconds=0) 获取“最新可用”的变换
            # 这消除了因等待数据同步产生的延迟
            
            # 注意：rclpy.time.Time(seconds=0) 是获取最新帧的标准写法
            t = self.tf_buffer.lookup_transform(
                'world', 
                ee_link, 
                rclpy.time.Time(seconds=0) 
            )
            
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            tz = t.transform.translation.z
            
            self.teleport_cube(tx, ty, tz)
            
        except Exception as e:
            # 只有在刚启动TF树还没建立时会进入这里，运行稳定后几乎不会报错
            # self.get_logger().warn(f"TF Error: {e}") 
            pass

    # =============================================================
    #                    调试打印（不变）
    # =============================================================
    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def print_finger_debug(self):
        targets = ["openarm_right_finger_joint1", "openarm_right_finger_joint2",
                   "openarm_left_finger_joint1",  "openarm_left_finger_joint2"]
        info = "🖐️ [Finger Debug] "
        for n in targets:
            if n in self.current_joints:
                info += f"{n.split('_')[-2][:1]}_{n.split('_')[-1][-1]}:{self.current_joints[n]:.3f} | "
        self.get_logger().info(info)

    # =============================================================
    #                      初始化
    # =============================================================
    def init_robot(self):
        time.sleep(2.0)
        self.get_logger().info(">>> 左右臂回到 ready ...")
        self.move_to_named_target_right("ready")
        self.move_to_named_target_left("ready")
        self.get_logger().info("=== 就绪，等待视觉 ===")
        self.is_busy = False
        if self.cached_target:
            self.check_and_start_pick()

    # =============================================================
    #                    主触发逻辑（不变）
    # =============================================================
    def coord_callback(self, msg):
        if abs(msg.x) < 0.01 and abs(msg.y) < 0.01:
            return
        self.cached_target = msg
        self.cached_z_surface = msg.z
        self.check_and_start_pick()

    def check_and_start_pick(self):
        if self.is_busy or self.cached_target is None:
            return
        target = self.cached_target
        if self.last_executed_target is not None:
            dist = math.sqrt((target.x - self.last_executed_target.x)**2 +
                             (target.y - self.last_executed_target.y)**2)
            if dist < 0.05:
                return
        self.cached_target = None
        self.is_busy = True
        self.last_executed_target = target
        self.get_logger().info(
            f">>> 🚀 启动协同抓取: ({target.x:.2f}, {target.y:.2f}, {target.z:.2f})")
        threading.Thread(target=self.execute_collab_pick,
                         args=(target.x, target.y, target.z), daemon=True).start()

    # =============================================================
    #               三步协作主流程（新增）
    # =============================================================
    def execute_collab_pick(self, x, y, z_surface):
        # 1. 左臂抓
        if not self.left_arm_pick(x, y, z_surface):
            self.is_busy = False
            return
        # 2. 空中递
        if not self.handover():
            self.is_busy = False
            return
        # 3. 右臂放
        if not self.right_arm_place():
            self.is_busy = False
            return
        self.get_logger().info("🎉 协作任务完成")
        time.sleep(1.0)
        self.is_busy = False

    # =============================================================
    #                       左臂抓取
    # =============================================================
    def left_arm_pick(self, x, y, z_surface):
        # 0. 先记录目标姿态，后面全程用它
        approach = np.array([0., 0., -1.])
        q_target = self._quat_from_approach(approach)          # <<< 统一姿态
        dx = 0.00
        dy = -0.00
        # 1. 悬停（带姿态！）
        hover_z = z_surface + self.HOVER_OFFSET
        if not self.move_to_pose_left(x + dx, y + dy, hover_z, tolerance=0.001, quat=q_target):  # <<<
            return False
        self.get_logger().info(f"-----------x: {x}  y:{y}  z: {hover_z}--------------")
        self.get_logger().info(f"-----------x: {x}  y:{y}  z: {hover_z}--------------")
        self.get_logger().info(f"-----------x: {x}  y:{y}  z: {hover_z}--------------")
        # 2. 张开爪
        self.open_left_gripper()

        # 3. 立刻拆掉保护球，再规划下降
        self.remove_protection_sphere()                       # <<< 提前拆

        # 4. 垂直下降（姿态与悬停完全一致，MoveIt 不会绕远）
        target_z = z_surface + self.GRIPPER_LENGTH
        if not self.move_to_pose_left(x + dx, y + dy, target_z, tolerance=0.01, quat=q_target):  # <<<
            # 失败也要先抬起来再退出
            self.move_to_pose_left(x + dx, y + dy, hover_z, tolerance=0.01, quat=q_target)
            return False
        self.get_logger().info(f"-----------x: {x}  y:{y}  z: {target_z}--------------")
        self.get_logger().info(f"-----------x: {x}  y:{y}  z: {target_z}--------------")
        self.get_logger().info(f"-----------x: {x}  y:{y}  z: {target_z}--------------")
        # 5. 闭合、抬升
        self.close_left_gripper()
        self.is_attached_left = True
        self.move_to_pose_left(x + dx, y + dy, hover_z, tolerance=0.01, quat=q_target)  # <<<
        return True

    # =============================================================
    #                      空中递接
    # =============================================================
    # 左臂在交接点的姿态：手心向右（+Y）
    def _quat_left_handover(self):
        # z_axis = +Y,  y_axis = +Z,  x_axis = +X
        z = np.array([0., 1., 0.])
        y = np.array([1., 0., 0.])
        x = np.cross(y, z)
        rot = np.eye(4)
        rot[0:3, 0], rot[0:3, 1], rot[0:3, 2] = x, y, z
        return quaternion_from_matrix(rot)

    # 右臂在交接点的姿态：手心向左（-Y）
    def _quat_right_handover(self):
        z = np.array([0., -1., 0.])
        y = np.array([0., 0., -1.])
        x = np.cross(y, z)
        rot = np.eye(4)
        rot[0:3, 0], rot[0:3, 1], rot[0:3, 2] = x, y, z
        return quaternion_from_matrix(rot)
    
    def handover(self):
        # --- 1. 计算姿态 ---
        q_left  = self._quat_left_handover()
        q_right = self._quat_right_handover()

        # --- 2. 左臂先到交接点（手心向右） ---
        if not self.move_to_pose_left(*self.HANDOVER_POS, tolerance=0.02, quat=q_right):
            return False

        # --- 3. 右臂再到交接点（手心向左） ---
        self.open_right_gripper()
        if not self.move_to_pose_right(*self.HANDOVER_POS, tolerance=0.03, quat=q_left):
            return False

        # --- 4. 换手：左松右夹 ---
        self.close_right_gripper()
        self.is_attached_left  = False
        self.is_attached_right = True
        time.sleep(2)
        self.open_left_gripper()
        time.sleep(0.5)

        return True

    # =============================================================
    #                       右臂放置
    # =============================================================
    def right_arm_place(self):
        hover = (self.PLACE_POS[0], self.PLACE_POS[1], self.PLACE_POS[2] + self.HOVER_OFFSET)
        approach = np.array([0., 0., -1.])
        q = self._quat_from_approach(approach)
        if not self.move_to_pose_right(*hover, tolerance=0.02, quat=q):
            return False
        if not self.move_to_pose_right(*self.PLACE_POS, tolerance=0.01, quat=q):
            return False
        self.is_attached_right = False
        self.open_right_gripper()
        self.move_to_pose_right(*hover, tolerance=0.02, quat=q)
        self.move_to_named_target_right("ready")
        self.move_to_named_target_left("ready")
        return True

    # =============================================================
    #                 夹爪控制（左右对称）
    # =============================================================
    def open_left_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.07
        goal.command.max_effort = -1.0
        self._gripper_client_left.send_goal_async(goal)
        time.sleep(0.5)

    def close_left_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.03
        goal.command.max_effort = 1.0
        self._gripper_client_left.send_goal_async(goal)
        time.sleep(2.0)

    def open_right_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.07
        goal.command.max_effort = -1.0
        self._gripper_client_right.send_goal_async(goal)
        time.sleep(0.5)

    def close_right_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 0.03
        goal.command.max_effort = 1.0
        self._gripper_client_right.send_goal_async(goal)
        time.sleep(2.0)

    # =============================================================
    #                     MoveIt 封装
    # =============================================================
    # -------------- 右臂 --------------
    def move_to_named_target_right(self, state_name="ready"):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name_right
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.05      # 限制速度 10%
        goal.request.max_acceleration_scaling_factor = 0.05  # 限制加速度 10%
        # goal.request.max_velocity_scaling_factor = 1.0
        # goal.request.max_acceleration_scaling_factor = 1.0
        goal.request.workspace_parameters.header.frame_id = "world"
        c = Constraints()
        if state_name == "ready":
            joints = ["openarm_right_joint{}".format(i) for i in range(1, 8)]
            vals   = [0.0, 1.5, 0.0, 1.5, 0.0, 0.0, 0.0]
        else:
            self.get_logger().warn("Unknown named target")
            return False
        for j, v in zip(joints, vals):
            from moveit_msgs.msg import JointConstraint
            jc = JointConstraint(joint_name=j, position=v,
                                 tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            c.joint_constraints.append(jc)
        goal.request.goal_constraints.append(c)
        return self._send_goal_and_wait_right(goal)

    def move_to_pose_right(self, x, y, z, tolerance=0.01, quat=None):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name_right
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.05
        goal.request.max_acceleration_scaling_factor = 0.05
        # goal.request.max_velocity_scaling_factor = 1.0
        # goal.request.max_acceleration_scaling_factor = 1.0
        c = Constraints()
        c.name = "pose_goal_right"
        pc = PositionConstraint()
        pc.header.frame_id = "world"
        pc.link_name = self.ee_link_right
        pc.weight = 1.0
        bv = BoundingVolume()
        sp = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[tolerance]*3)
        bv.primitives.append(sp)
        p = PoseStamped()
        p.pose.position.x, p.pose.position.y, p.pose.position.z = x, y, z
        bv.primitive_poses.append(p.pose)
        pc.constraint_region = bv
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = "world"
        oc.link_name = self.ee_link_right
        if quat is None:
            oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = 1.0, 0.0, 0.0, 0.0
        else:
            oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = quat
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        goal.request.goal_constraints.append(c)
        return self._send_goal_and_wait_right(goal)

    def _send_goal_and_wait_right(self, goal):
        future = self._move_client_right.send_goal_async(goal)
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("MoveIt(right) 拒绝请求")
            return False
        res_future = goal_handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)
        result = res_future.result().result          # 这里才是 MoveGroup.Result
        return result.error_code.val == 1

    # -------------- 左臂 --------------
    def move_to_named_target_left(self, state_name="ready"):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name_left
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.workspace_parameters.header.frame_id = "world"
        c = Constraints()
        if state_name == "ready":
            joints = ["openarm_left_joint{}".format(i) for i in range(1, 8)]
            vals   = [0.0, -1.5, 0.0, 1.5, 0.0, 0.0, 0.0]
        else:
            self.get_logger().warn("Unknown named target")
            return False
        for j, v in zip(joints, vals):
            from moveit_msgs.msg import JointConstraint
            jc = JointConstraint(joint_name=j, position=v,
                                 tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            c.joint_constraints.append(jc)
        goal.request.goal_constraints.append(c)
        return self._send_goal_and_wait_left(goal)

    def move_to_pose_left(self, x, y, z, tolerance=0.01, quat=None):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name_left
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.05
        goal.request.max_acceleration_scaling_factor = 0.05
        # goal.request.max_velocity_scaling_factor = 1.0
        # goal.request.max_acceleration_scaling_factor = 1.0
        c = Constraints()
        c.name = "pose_goal_left"
        pc = PositionConstraint()
        pc.header.frame_id = "world"
        pc.link_name = self.ee_link_left
        pc.weight = 1.0
        bv = BoundingVolume()
        sp = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[tolerance]*3)
        bv.primitives.append(sp)
        p = PoseStamped()
        p.pose.position.x, p.pose.position.y, p.pose.position.z = x, y, z
        bv.primitive_poses.append(p.pose)
        pc.constraint_region = bv
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = "world"
        oc.link_name = self.ee_link_left
        if quat is None:
            oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = 1.0, 0.0, 0.0, 0.0
        else:
            oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = quat
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        goal.request.goal_constraints.append(c)
        return self._send_goal_and_wait_left(goal)

    def _send_goal_and_wait_left(self, goal):
        future = self._move_client_left.send_goal_async(goal)
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("MoveIt(left) 拒绝请求")
            return False
        res_future = goal_handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)
        result = res_future.result().result
        return result.error_code.val == 1

    # =============================================================
    #                     工具函数
    # =============================================================
    def _quat_from_approach(self, approach):
        z = approach / np.linalg.norm(approach)
        y = np.cross(z, [0, 0, 1])
        if np.linalg.norm(y) < 0.3:
            y = np.cross(z, [0, 1, 0])
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        rot = np.eye(4)
        rot[0:3, 0], rot[0:3, 1], rot[0:3, 2] = x, y, z
        return quaternion_from_matrix(rot)

    # =============================================================
    #                     场景碰撞球
    # =============================================================
    def add_protection_sphere(self, x, y, z_surface):
        z_center = z_surface - self.APPLE_RADIUS
        self.get_logger().info(
            f"🛡️ 添加保护球: R={self.PROTECTION_RADIUS}, Center=({x:.2f}, {y:.2f}, {z_center:.2f})")
        co = CollisionObject()
        co.header = Header(frame_id='world', stamp=self.get_clock().now().to_msg())
        co.id = 'apple_protection_sphere'
        co.operation = CollisionObject.ADD
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[self.PROTECTION_RADIUS])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z_center
        pose.orientation.w = 1.0
        co.primitives.append(sphere)
        co.primitive_poses.append(pose)
        scene_msg = PlanningScene()
        scene_msg.world.collision_objects.append(co)
        scene_msg.is_diff = True
        self.scene_pub.publish(scene_msg)
        time.sleep(0.5)

    def remove_protection_sphere(self):
        self.get_logger().info("🗑️ 移除保护球")
        co = CollisionObject()
        co.header = Header(frame_id='world', stamp=self.get_clock().now().to_msg())
        co.id = 'apple_protection_sphere'
        co.operation = CollisionObject.REMOVE
        scene_msg = PlanningScene()
        scene_msg.world.collision_objects.append(co)
        scene_msg.is_diff = True
        self.scene_pub.publish(scene_msg)
        time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)
    node = RobustPicker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()