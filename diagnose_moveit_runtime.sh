#!/usr/bin/env bash

set -u

REPO_ROOT="$HOME/embodied-arm-system"
OUTPUT_FILE="$REPO_ROOT/moveit_runtime_diagnosis.txt"
PACKAGE_NAME="el_a3_moveit_config"

print_section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

run_command() {
    echo
    echo "\$ $*"
    "$@" || true
}

exec > >(tee "$OUTPUT_FILE") 2>&1

print_section "1. 基本信息"

run_command date
run_command uname -a
run_command ros2 --version
run_command ros2 node list

print_section "2. MoveIt 配置包位置"

PACKAGE_PREFIX="$(
    ros2 pkg prefix "$PACKAGE_NAME" 2>/dev/null || true
)"

echo "Package: $PACKAGE_NAME"
echo "Prefix:  $PACKAGE_PREFIX"

if [[ -z "$PACKAGE_PREFIX" ]]; then
    echo "错误：没有找到 $PACKAGE_NAME"
else
    SHARE_DIRECTORY="$PACKAGE_PREFIX/share/$PACKAGE_NAME"

    echo "Share directory: $SHARE_DIRECTORY"

    if [[ -d "$SHARE_DIRECTORY" ]]; then
        find -L "$SHARE_DIRECTORY" \
            -maxdepth 3 \
            -type f \
            | sort
    else
        echo "错误：配置目录不存在"
    fi
fi

print_section "3. 已安装的 kinematics.yaml"

if [[ -n "$PACKAGE_PREFIX" ]]; then
    while IFS= read -r FILE_PATH; do
        [[ -z "$FILE_PATH" ]] && continue

        echo
        echo "--------------------"
        echo "文件：$FILE_PATH"
        echo "--------------------"

        cat "$FILE_PATH"
    done < <(
        # colcon --symlink-install creates symlinked config files. Follow
        # them so this report does not incorrectly claim they are missing.
        find -L "$PACKAGE_PREFIX/share/$PACKAGE_NAME" \
            -type f \
            -name "*kinematics*.yaml" \
            2>/dev/null \
            | sort
    )
fi

print_section "4. 源码目录中的运动学配置"

SOURCE_ROOT="$REPO_ROOT/third_party/EDULITE_A3/el_a3_ros"

if [[ -d "$SOURCE_ROOT" ]]; then
    while IFS= read -r FILE_PATH; do
        [[ -z "$FILE_PATH" ]] && continue

        echo
        echo "--------------------"
        echo "文件：$FILE_PATH"
        echo "--------------------"

        cat "$FILE_PATH"
    done < <(
        find -L "$SOURCE_ROOT" \
            -type f \
            \( \
                -name "*kinematics*.yaml" \
                -o -name "*.srdf" \
            \) \
            2>/dev/null \
            | sort
    )
else
    echo "没有找到源码目录：$SOURCE_ROOT"
fi

print_section "5. /move_group 运动学参数列表"

PARAMETER_LIST="$(
    ros2 param list /move_group 2>/dev/null \
        | sed 's/^[[:space:]]*//'
)"

if [[ -z "$PARAMETER_LIST" ]]; then
    echo "错误：无法读取 /move_group 参数"
    echo "请确认 MoveIt2 demo.launch.py 正在运行"
else
    echo "$PARAMETER_LIST" \
        | grep -E \
            '(^arm\.|kinematics|planning_pipeline|planner)' \
        || true
fi

print_section "6. /move_group 运动学参数值"

MATCHED_PARAMETERS="$(
    echo "$PARAMETER_LIST" \
        | grep -E \
            '(^arm\.|kinematics|planning_pipeline|planner)' \
        || true
)"

while IFS= read -r PARAMETER_NAME; do
    [[ -z "$PARAMETER_NAME" ]] && continue

    echo
    echo "参数：$PARAMETER_NAME"

    ros2 param get \
        /move_group \
        "$PARAMETER_NAME" \
        || true
done <<< "$MATCHED_PARAMETERS"

print_section "7. 重点 Pick IK 参数"

IMPORTANT_PARAMETERS=(
    "arm.kinematics_solver"
    "arm.kinematics_solver_timeout"
    "arm.kinematics_solver_attempts"
    "arm.mode"
    "arm.stop_optimization_on_valid_solution"
    "arm.position_scale"
    "arm.rotation_scale"
    "arm.position_threshold"
    "arm.orientation_threshold"
    "arm.cost_threshold"
    "arm.minimal_displacement_weight"
    "arm.gd_step_size"
)

for PARAMETER_NAME in "${IMPORTANT_PARAMETERS[@]}"; do
    echo
    echo "参数：$PARAMETER_NAME"

    if echo "$PARAMETER_LIST" \
        | grep -Fxq "$PARAMETER_NAME"; then

        ros2 param get \
            /move_group \
            "$PARAMETER_NAME" \
            || true
    else
        echo "Parameter not set"
    fi
done

print_section "8. MoveIt 服务与 Action"

run_command ros2 service list -t
run_command ros2 action list -t
run_command ros2 action info /move_action

print_section "诊断完成"

echo "诊断报告已保存到："
echo "$OUTPUT_FILE"
