"""Launch the control-only EDULITE A3 Gazebo Classic validation."""

from gazebo_classic_sim.classic_bringup import build_classic_launch


def generate_launch_description():
    return build_classic_launch("el_a3_workbench.world")
