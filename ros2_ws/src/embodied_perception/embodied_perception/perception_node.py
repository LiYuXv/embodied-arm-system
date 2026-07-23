"""Camera-main colour perception for the Gazebo sorting workcell."""

from dataclasses import replace
from typing import Optional

import numpy

import rclpy
from cv_bridge import CvBridge
from embodied_interfaces.msg import DetectedObject, DetectedObjectArray
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from embodied_perception.colour_localizer import (
    detect_sorting_items,
    project_pixel_to_table,
)


class PerceptionNode(Node):
    """Publish calibrated red/blue cubes and target zones from camera_main."""

    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("rgb_topic", "/camera_main/image_raw")
        self.declare_parameter("camera_info_topic", "/camera_main/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("min_cube_area_px", 120.0)
        self.declare_parameter("min_zone_area_px", 900.0)
        self.declare_parameter("morphology_kernel_size", 5)
        self.declare_parameter(
            "camera_translation_base_m", [-1.107069, 0.736844, 0.471686]
        )
        self.declare_parameter(
            "camera_rotation_rpy", [-2.048928, 0.051416, -2.589323]
        )
        self.declare_parameter("table_plane_z_m", -0.02)
        self.declare_parameter("cube_top_plane_z_m", 0.05)
        self.declare_parameter("target_zone_plane_z_m", -0.004)
        self.declare_parameter("image_to_base_homography", [0.0])

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_translation = list(
            self.get_parameter("camera_translation_base_m").value
        )
        self.camera_rotation = list(
            self.get_parameter("camera_rotation_rpy").value
        )
        self.table_plane_z = float(self.get_parameter("table_plane_z_m").value)
        self.cube_top_plane_z = float(
            self.get_parameter("cube_top_plane_z_m").value
        )
        self.target_zone_plane_z = float(
            self.get_parameter("target_zone_plane_z_m").value
        )
        homography_values = list(
            self.get_parameter("image_to_base_homography").value
        )
        # Calibrated for camera_main and the raised 2x2 sorting board.  This
        # maps current image pixels into base_link; it is not a per-object
        # pose fallback and therefore continues to respond to visual motion.
        default_homography = [
            0.011679817586, 0.038269035091, -6.164150801294,
            0.004489959975, -0.027546018363, 2.639551343672,
            0.000337339241, -0.041559291884, 1.0,
        ]
        self.image_to_base_homography = (
            numpy.asarray(homography_values, dtype=float).reshape(3, 3)
            if len(homography_values) == 9
            else numpy.asarray(default_homography, dtype=float).reshape(3, 3)
        )
        self.bridge = CvBridge()
        self.latest_rgb: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.rgb_message_count = 0
        self.camera_info_message_count = 0

        self.rgb_subscription = self.create_subscription(
            # Gazebo Classic camera plugins publish reliable image streams.
            # A reliable subscription is required here; sensor-data's
            # best-effort profile can remain unmatched on some DDS setups.
            Image, self.rgb_topic, self._handle_rgb_image, 10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._handle_camera_info,
            10,
        )
        self.detected_objects_publisher = self.create_publisher(
            DetectedObjectArray, "/detected_objects", 10
        )
        rate = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self.publish_timer = self.create_timer(1.0 / rate, self._publish_detections)
        self.get_logger().info(
            "Perception ready: camera_main HSV detection publishes red_cube, "
            "blue_cube, red_target_zone, blue_target_zone in base_link"
        )

    def _handle_rgb_image(self, message: Image) -> None:
        self.latest_rgb = message
        self.rgb_message_count += 1

    def _handle_camera_info(self, message: CameraInfo) -> None:
        self.latest_camera_info = message
        self.camera_info_message_count += 1

    def _camera_inputs_ready(self) -> bool:
        return self.latest_rgb is not None and self.latest_camera_info is not None

    def _get_output_frame(self) -> str:
        return self.base_frame

    def _publish_detections(self) -> None:
        message = DetectedObjectArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        if not self._camera_inputs_ready():
            self.detected_objects_publisher.publish(message)
            return
        try:
            image = self.bridge.imgmsg_to_cv2(self.latest_rgb, "bgr8")
            detections = detect_sorting_items(
                image,
                min_cube_area_px=float(self.get_parameter("min_cube_area_px").value),
                min_zone_area_px=float(self.get_parameter("min_zone_area_px").value),
                kernel_size=int(self.get_parameter("morphology_kernel_size").value),
            )
            projected_detections = []
            for detection in detections.values():
                point = self._project_detection(detection)
                if point is None:
                    self.get_logger().warning(
                        f"{detection.name} pixel={detection.pixel} cannot be "
                        "projected onto the table plane"
                    )
                    continue
                projected_detections.append((detection, point))
            for detection, point in projected_detections:
                message.objects.append(self._to_message(detection, point))
                self.get_logger().debug(
                    f"Detected {detection.name}: pixel=({detection.pixel[0]:.1f}, "
                    f"{detection.pixel[1]:.1f}), base_link=({point[0]:.3f}, "
                    f"{point[1]:.3f}, {point[2]:.3f})"
                )
        except Exception as error:
            self.get_logger().error(f"camera_main HSV detection failed: {error}")
        self.detected_objects_publisher.publish(message)

    def _classify_compact_layout_rows(self, projected_detections):
        """
        Keep cube/zone labels stable when a rotated cube has more pixels.

        Contour area normally separates the small cube from its larger marker.
        In the compact raised board a 45-degree cube exposes two side faces and
        can temporarily occupy more colour pixels than its flat marker.  The
        two item classes still occupy distinct *observed scene rows*: pickup
        cubes have the smaller base_link y coordinate and target zones the
        larger one.  Apply that geometric scene rule only when both same-colour
        contours are present; it is not a pose fallback and remains fully
        driven by the current camera image/homography.
        """
        by_colour = {}
        for detection, point in projected_detections:
            colour = detection.name.split("_", 1)[0]
            by_colour.setdefault(colour, []).append((detection, point))

        relabelled = []
        for colour, entries in by_colour.items():
            cube_entry = next(
                (entry for entry in entries if entry[0].category == "cube"), None
            )
            zone_entry = next(
                (entry for entry in entries if entry[0].category == "target_zone"),
                None,
            )
            if cube_entry is not None and zone_entry is not None:
                near_entry, far_entry = sorted(
                    (cube_entry, zone_entry), key=lambda entry: entry[1][1]
                )
                relabelled.append((
                    replace(near_entry[0], name=f"{colour}_cube", category="cube"),
                    (
                        near_entry[1][0], near_entry[1][1],
                        self._projection_plane_for("cube"),
                    ),
                ))
                relabelled.append((
                    replace(
                        far_entry[0], name=f"{colour}_target_zone",
                        category="target_zone",
                    ),
                    (
                        far_entry[1][0], far_entry[1][1],
                        self._projection_plane_for("target_zone"),
                    ),
                ))
            else:
                relabelled.extend(entries)
        return relabelled

    def _project_detection(self, detection):
        """Project an image point through the configured camera calibration."""
        plane_z = self._projection_plane_for(detection.category)
        if self.image_to_base_homography is not None:
            image_point = numpy.asarray(
                [detection.pixel[0], detection.pixel[1], 1.0], dtype=float
            )
            mapped = self.image_to_base_homography @ image_point
            if abs(mapped[2]) < 1e-9:
                return None
            return (float(mapped[0] / mapped[2]),
                    float(mapped[1] / mapped[2]), plane_z)
        return project_pixel_to_table(
            detection.pixel,
            self.latest_camera_info.k,
            self.camera_translation,
            self.camera_rotation,
            plane_z,
        )

    def _projection_plane_for(self, category: str) -> float:
        """Use the visible cube top or the static marker surface as needed."""
        if category == "cube":
            return self.cube_top_plane_z
        if category == "target_zone":
            return self.target_zone_plane_z
        return self.table_plane_z

    def _to_message(self, detection, point) -> DetectedObject:
        message = DetectedObject()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        message.name = detection.name
        message.category = detection.category
        message.pose.header = message.header
        message.pose.pose.position.x = point[0]
        message.pose.pose.position.y = point[1]
        message.pose.pose.position.z = point[2]
        message.pose.pose.orientation.w = 1.0
        message.size.x = detection.size_px[0]
        message.size.y = detection.size_px[1]
        message.pixel_center.x = detection.pixel[0]
        message.pixel_center.y = detection.pixel[1]
        message.confidence = detection.confidence
        return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
