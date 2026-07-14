"""Run the EDULITE A3 with the separate two-RGB-camera Classic route."""

from gazebo_classic_sim.classic_bringup import build_classic_launch


def generate_launch_description():
    return build_classic_launch(
        "classic_dual_rgb_workbench.world",
        camera_mode="none",
        perception_config="dual_rgb_perception.yaml",
    )
