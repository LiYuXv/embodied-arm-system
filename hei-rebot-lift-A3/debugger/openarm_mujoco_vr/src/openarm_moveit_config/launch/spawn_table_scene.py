import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Header
import time

class TableSpawner(Node):
    def __init__(self):
        super().__init__('table_spawner')
        self.pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        time.sleep(2.0)          # 等 move_group 起来
        self.spawn_table()
        self.get_logger().info('圆柱体桌子已加入 PlanningScene')

    def spawn_table(self):
        co = CollisionObject()
        co.header = Header(frame_id='world', stamp=self.get_clock().now().to_msg())
        co.id = 'ornate_table'

        # 一个圆柱体 = 桌面 + 四条腿全部“融”进去
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.CYLINDER
        sp.dimensions = [0.4, 0.35]   # 半径 0.37 m，高度 0.42 m

        co.primitives.append(sp)

        p = Pose()
        p.position  = Point(x=0.5, y=0.0, z=0.2)  # 底面在 z=0，高 0.42，则中心 z=0.21
        p.orientation.w = 1.0
        co.primitive_poses.append(p)
        co.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True
        self.pub.publish(scene)

def main():
    rclpy.init()
    TableSpawner().destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()