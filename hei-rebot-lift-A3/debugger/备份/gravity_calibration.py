"""EL-A3 重力补偿标定 - 基于最小二乘法"""

import numpy as np
import logging
import time
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gravity_calib")


class GravityCompParams:
    """重力补偿参数"""
    def __init__(self):
        self.sin_coeff = 0.0
        self.cos_coeff = 0.0
        self.offset = 0.0
        self.rmse = 0.0
        self.r_squared = 0.0


class GravityCalibrator:
    """重力补偿标定器"""

    JOINT_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    
    SAFE_HOME = [0.5, 0.785, -0.785, 0.5, 0.5, 0.5]
    
    JOINT_LIMITS = {
        'L1': (-2.79, 2.79),
        'L2': (-3.14, 3.14),
        'L3': (-3.14, 3.14),
        'L4': (-3.14, 3.14),
        'L5': (-3.14, 3.14),
        'L6': (-3.14, 3.14),
    }

    def __init__(self, arm_worker=None):
        self.arm_worker = arm_worker
        self.params: Dict[str, GravityCompParams] = {}
        for name in self.JOINT_NAMES:
            self.params[name] = GravityCompParams()

    def generate_test_angles(self, joint_idx: int, num_angles: int = 7) -> List[float]:
        """生成测试角度"""
        home_angle = self.SAFE_HOME[joint_idx]
        zero_margin = 0.3
        max_angle = 1.8
        
        if home_angle >= 0:
            lower = max(zero_margin, home_angle - 0.8)
            upper = min(max_angle, home_angle + 1.0)
        else:
            lower = max(-max_angle, home_angle - 1.0)
            upper = min(-zero_margin, home_angle + 0.8)
        
        if lower >= upper:
            lower, upper = home_angle - 0.5, home_angle + 0.5
        
        return np.linspace(lower, upper, num_angles).tolist()

    def collect_torque(self, joint_idx: int, target_angle: float, 
                     fixed_positions: List[float], 
                     samples: int = 50) -> Tuple[float, float, float]:
        """采集力矩样本"""
        if self.arm_worker is None:
            return self._simulate_torque(joint_idx, target_angle)
        
        target = fixed_positions.copy()
        target[joint_idx] = target_angle
        
        logger.info(f"    移动到: {target_angle:.3f} rad ({np.degrees(target_angle):.1f}°)")
        
        self.arm_worker.submit_command("move_to", target, 4.0)
        time.sleep(2.0)
        
        positions_list = []
        efforts_list = []
        for _ in range(samples):
            pos = self.arm_worker.data_buffer.get_latest_positions()
            eff = self.arm_worker.data_buffer.get_latest_efforts()
            if pos and eff:
                positions_list.append(pos[joint_idx] if joint_idx < len(pos) else 0.0)
                efforts_list.append(eff[joint_idx] if joint_idx < len(eff) else 0.0)
            time.sleep(0.02)
        
        if len(efforts_list) < 10:
            return self._simulate_torque(joint_idx, target_angle)
        
        efforts = np.array(efforts_list)
        mean_effort = np.mean(efforts)
        std_effort = np.std(efforts)
        
        inlier_mask = np.abs(efforts - mean_effort) < 2.5 * std_effort
        filtered = efforts[inlier_mask] if np.sum(inlier_mask) > 5 else efforts
        
        final_mean = float(np.mean(filtered))
        final_std = float(np.std(filtered))
        actual_angle = float(np.mean(positions_list))
        
        logger.info(f"      力矩: {final_mean:.4f} ± {final_std:.4f} Nm")
        
        return final_mean, final_std, actual_angle

    def _simulate_torque(self, joint_idx: int, angle: float) -> Tuple[float, float, float]:
        """模拟力矩数据"""
        mass = [0.465, 0.147, 0.947, 0.407, 0.547, 0.032]
        com = [0.102, -0.086, -0.020, 0.004, 0.0, 0.0]
        
        torque = mass[joint_idx] * 9.81 * com[joint_idx] * np.cos(angle)
        torque += np.random.normal(0, 0.03)
        
        return torque, 0.02, angle

    def fit_gravity_params(self, angles: List[float], torques: List[float]) -> Dict:
        """最小二乘法拟合 τ = a*sin(θ) + b*cos(θ) + c"""
        angles = np.array(angles)
        torques = np.array(torques)
        
        A = np.column_stack([
            np.sin(angles),
            np.cos(angles),
            np.ones_like(angles)
        ])
        
        params, residuals, rank, s = np.linalg.lstsq(A, torques, rcond=None)
        
        result = {
            'sin_coeff': float(params[0]),
            'cos_coeff': float(params[1]),
            'offset': float(params[2]),
        }
        
        predicted = A @ params
        errors = torques - predicted
        result['rmse'] = float(np.sqrt(np.mean(errors ** 2)))
        
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((torques - np.mean(torques)) ** 2)
        result['r_squared'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        
        return result

    def calibrate_single_joint(self, joint_idx: int, 
                          fixed_positions: List[float] = None) -> Dict:
        """标定单个关节"""
        if fixed_positions is None:
            fixed_positions = self.SAFE_HOME.copy()
        
        joint_name = self.JOINT_NAMES[joint_idx]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"  标定关节: {joint_name}")
        logger.info(f"{'='*50}")
        
        test_angles = self.generate_test_angles(joint_idx)
        angles = []
        torques = []
        
        for i, angle in enumerate(test_angles):
            logger.info(f"\n  采样点 {i+1}/{len(test_angles)}:")
            
            torque, std, actual = self.collect_torque(joint_idx, angle, fixed_positions)
            if torque is not None:
                angles.append(actual)
                torques.append(torque)
        
        if len(angles) < 3:
            logger.error("数据点不足，无法拟合")
            return {}
        
        result = self.fit_gravity_params(angles, torques)
        
        name = self.JOINT_NAMES[joint_idx]
        self.params[name].sin_coeff = result['sin_coeff']
        self.params[name].cos_coeff = result['cos_coeff']
        self.params[name].offset = result['offset']
        self.params[name].rmse = result['rmse']
        self.params[name].r_squared = result['r_squared']
        
        logger.info(f"\n  拟合结果:")
        logger.info(f"    sin_coeff = {result['sin_coeff']:.4f}")
        logger.info(f"    cos_coeff = {result['cos_coeff']:.4f}")
        logger.info(f"    offset    = {result['offset']:.4f}")
        logger.info(f"    RMSE      = {result['rmse']:.4f} Nm")
        logger.info(f"    R²        = {result['r_squared']:.4f}")
        
        return result

    def run_full_calibration(self) -> Dict:
        """运行完整标定"""
        logger.info("\n" + "="*60)
        logger.info("  EL-A3 重力补偿标定")
        logger.info("="*60)
        logger.info(f"  标定顺序: L6→L5→L4→L3→L2→L1")
        logger.info(f"  Home位置: {self.SAFE_HOME}")
        
        results = {}
        
        for joint_idx in [5, 4, 3, 2, 1, 0]:
            result = self.calibrate_single_joint(joint_idx)
            if result:
                results[self.JOINT_NAMES[joint_idx]] = result
        
        return results

    def generate_xacro_params(self) -> str:
        """生成XACRO参数"""
        lines = []
        for i, name in enumerate(self.JOINT_NAMES):
            p = self.params[name]
            lines.append(f'<param name="gravity_comp_{name}_sin">{p.sin_coeff:.4f}</param>')
            lines.append(f'<param name="gravity_comp_{name}_cos">{p.cos_coeff:.4f}</param>')
            lines.append(f'<param name="gravity_comp_{name}_offset">{p.offset:.4f}</param>')
            lines.append("")
        return "\n".join(lines)


def run_gravity_calibration():
    """运行重力补偿标定（模拟模式）"""
    logger.info("="*60)
    logger.info("  重力补偿标定 (模拟模式)")
    logger.info("="*60)
    
    calibrator = GravityCalibrator()
    results = calibrator.run_full_calibration()
    
    logger.info("\n" + "="*60)
    logger.info("  标定结果汇总")
    logger.info("="*60)
    logger.info(f"\n{'关节':<8} {'sin_coeff':>12} {'cos_coeff':>12} {'offset':>12} {'RMSE':>10} {'R²':>8}")
    logger.info("-"*60)
    
    for name in GravityCalibrator.JOINT_NAMES:
        p = calibrator.params[name]
        logger.info(f"{name:<8} {p.sin_coeff:>12.4f} {p.cos_coeff:>12.4f} {p.offset:>12.4f} {p.rmse:>10.4f} {p.r_squared:>8.4f}")
    
    logger.info("\n生成的 XACRO 参数:")
    logger.info(calibrator.generate_xacro_params())
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    if args.quick:
        run_gravity_calibration()
    else:
        logger.info("Use --quick to run calibration")