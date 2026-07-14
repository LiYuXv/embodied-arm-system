"""Run the EDULITE A3 with its wrist-mounted RGB-D route in Gazebo Classic."""

from gazebo_classic_sim.classic_bringup import build_classic_launch


def generate_launch_description():
    return build_classic_launch(
        "el_a3_workbench.world",
        camera_mode="rgbd",
        perception_config="rgbd_perception.yaml",
    )
