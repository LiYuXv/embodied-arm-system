import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import tf2_ros
import tf2_geometry_msgs 

class UniversalDetector(Node):
    def __init__(self):
        super().__init__('universal_detector_node')
        
        self.depth_topic = '/my_cam/depth'
        
        # 相机内参 (保持不变)
        self.width = 640
        self.height = 480
        # self.fovy_deg = 60.0
        self.fovy_deg = 45.0
        self.fy = (self.height / 2.0) / math.tan(math.radians(self.fovy_deg / 2.0))
        self.fx = self.fy 
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        # 初始化 TF 监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, self.depth_topic, self.listener_callback, 10)
        
        self.coord_pub = self.create_publisher(Point, '/apple_location', 10)
        
        # --- 修改点1：状态变量 ---
        self.fixed_world_point = None  # 用于存储锁定的坐标
        self.lock_time = None          # 用于记录锁定的起始时间
        
        self.get_logger().info("=== 绝对坐标检测器启动 ===")

    def listener_callback(self, msg):
        # ==========================================
        # 阶段 A: 已经锁定目标 -> 持续广播 + 限时日志
        # ==========================================
        if self.fixed_world_point is not None:
            # 1. 持续发布之前锁定的坐标 (不管现在图像里有没有东西)
            self.coord_pub.publish(self.fixed_world_point)
            
            # 2. 计算锁定后的时长
            time_since_lock = (self.get_clock().now() - self.lock_time).nanoseconds / 1e9
            
            # 3. 只有在前 5 秒内，且每隔 1 秒打印一次日志
            if time_since_lock < 5.0:
                self.get_logger().info(
                    f"🔒 [已锁定] 持续广播坐标: x={self.fixed_world_point.x:.2f}, y={self.fixed_world_point.y:.2f}, z={self.fixed_world_point.z:.2f}",
                    throttle_duration_sec=1.0  # ROS2自带功能：限制打印频率为1秒1次
                )
            return  # 直接返回，不再进行图像处理，节省资源

        # ==========================================
        # 阶段 B: 尚未锁定 -> 寻找目标
        # ==========================================
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_image = np.nan_to_num(cv_image, nan=10.0)

            # 简单的深度分割提取物体 (0.3m ~ 1.5m 范围内)
            mask = (cv_image < 0.75) & (cv_image > 0.3)
            mask_uint8 = (mask * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 过滤太小的噪点
                if cv2.contourArea(largest_contour) < 50: return

                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    u = int(M["m10"] / M["m00"])
                    v = int(M["m01"] / M["m00"])
                    d = cv_image[v, u]

                    if 0.3 < d < 1.5:
                        # 1. 计算相机坐标系坐标
                        x_cam = (u - self.cx) * d / self.fx
                        y_cam = (v - self.cy) * d / self.fy
                        z_cam = d

                        # 2. 构建 Stamped Point
                        point_cam = PointStamped()
                        point_cam.header.frame_id = "camera_optical_frame" 
                        point_cam.header.stamp = self.get_clock().now().to_msg()
                        point_cam.point.x = float(x_cam)
                        point_cam.point.y = float(y_cam)
                        point_cam.point.z = float(z_cam)

                        # 3. 查询 TF 并转换到 World
                        try:
                            transform = self.tf_buffer.lookup_transform(
                                'world', 
                                'camera_optical_frame', 
                                rclpy.time.Time(), 
                                timeout=rclpy.duration.Duration(seconds=1.0)
                            )
                            
                            point_world = tf2_geometry_msgs.do_transform_point(point_cam, transform)
                            
                            # --- 修改点2：锁定坐标 ---
                            self.fixed_world_point = point_world.point # 永久保存这个坐标
                            self.lock_time = self.get_clock().now()    # 记录开始时间
                            
                            # 立即发布一次
                            self.coord_pub.publish(self.fixed_world_point)
                            
                            self.get_logger().info(
                                f"🎯 目标初次锁定! World:({point_world.point.x:.2f}, {point_world.point.y:.2f}, {point_world.point.z:.2f})"
                            )
                            
                        except tf2_ros.LookupException:
                            self.get_logger().warn("TF 查找失败: 尚未收到 camera_optical_frame 的变换")
                        except tf2_ros.ExtrapolationException:
                            self.get_logger().warn("TF 时间不同步")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UniversalDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()