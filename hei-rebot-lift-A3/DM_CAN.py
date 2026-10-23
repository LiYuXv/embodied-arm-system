from time import sleep
import numpy as np
from enum import IntEnum
from struct import unpack
from struct import pack
import math
import time
import threading        # NEW
import logging
logger = logging.getLogger("motor_status")
# 将以下枚举类定义移到文件开头，在 MotorControl 类之前

class DM_Motor_Type(IntEnum):
    DM4310 = 0
    DM4310_48V = 1
    DM4340 = 2
    DM4340_48V = 3
    DM6006 = 4
    DM8006 = 5
    DM8009 = 6
    DM10010L = 7
    DM10010 = 8
    DMH3510 = 9
    DMH6215 = 10
    DMG6220 = 11

class DM_variable(IntEnum):
    UV_Value = 0
    KT_Value = 1
    OT_Value = 2
    OC_Value = 3
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7
    ESC_ID = 8
    TIMEOUT = 9
    CTRL_MODE = 10
    Damp = 11
    Inertia = 12
    hw_ver = 13
    sw_ver = 14
    SN = 15
    NPP = 16
    Rs = 17
    LS = 18
    Flux = 19
    Gr = 20
    PMAX = 21
    VMAX = 22
    TMAX = 23
    I_BW = 24
    KP_ASR = 25
    KI_ASR = 26
    KP_APR = 27
    KI_APR = 28
    OV_Value = 29
    GREF = 30
    Deta = 31
    V_BW = 32
    IQ_c1 = 33
    VL_c1 = 34
    can_br = 35
    sub_ver = 36
    u_off = 50
    v_off = 51
    k1 = 52
    k2 = 53
    m_off = 54
    dir = 55
    p_m = 80
    xout = 81

class Control_Type(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    Torque_Pos = 4

# 电机状态枚举类
class Motor_Status(IntEnum):
    DISABLED = 0x0      # 失能
    ENABLED = 0x1       # 使能
    OVERVOLTAGE = 0x8   # 超压
    UNDERVOLTAGE = 0x9  # 欠压
    OVERCURRENT = 0xA   # 过电流
    MOS_OVERTEMP = 0xB  # MOS过温
    COIL_OVERTEMP = 0xC # 电机线圈过温
    COMM_LOST = 0xD     # 通讯丢失
    OVERLOAD = 0xE      # 过载

class MotorControl:
    send_data_frame = np.array(
        [0x55, 0xAA, 0x1e, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0x00, 0x08, 0x00,
         0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0x00], np.uint8)
    
    # 电机的最大位置 (Q_MAX)、速度 (DQ_MAX) 和力矩 (TAU_MAX)
    #                4310           4310_48        4340           4340_48
    Limit_Param = [[12.5, 30, 10], [12.5, 50, 10], [12.5, 8, 28], [12.5, 10, 28],
                   # 6006           8006           8009            10010L         10010
                   [12.5, 45, 20], [12.5, 45, 40], [12.5, 45, 54], [12.5, 25, 200], [12.5, 20, 200],
                   # H3510            DMG6215      DMH6220
                   [12.5 , 280 , 1],[12.5 , 45 , 10],[12.5 , 45 , 10]]

    # 添加电压限制配置 (欠压值, 过压值)
    Voltage_Limits = {
        DM_Motor_Type.DM4310: (18.0, 30.0),        # 24V电机
        DM_Motor_Type.DM4310_48V: (36.0, 60.0),    # 48V电机
        DM_Motor_Type.DM4340: (18.0, 30.0),        # 24V电机
        DM_Motor_Type.DM4340_48V: (36.0, 60.0),    # 48V电机
        DM_Motor_Type.DM6006: (18.0, 30.0),        # 24V电机
        DM_Motor_Type.DM8006: (18.0, 30.0),        # 24V电机
        DM_Motor_Type.DM8009: (30.0, 50.0),        # 48V电机 - 根据你的8009电机设置
        DM_Motor_Type.DM10010L: (36.0, 60.0),      # 48V电机
        DM_Motor_Type.DM10010: (36.0, 60.0),       # 48V电机
        DM_Motor_Type.DMH3510: (18.0, 30.0),       # 24V电机
        DM_Motor_Type.DMH6215: (18.0, 30.0),       # 24V电机
        DM_Motor_Type.DMG6220: (18.0, 30.0)        # 24V电机
    }

    def __init__(self, serial_device):
        """
        define MotorControl object 定义电机控制对象
        :param serial_device: serial object 串口对象
        """
        self.serial_ = serial_device
        self.motors_map = dict()
        self.data_save = bytes()  # save data
        if self.serial_.is_open:  # open the serial port
            print("Serial port is open")
            serial_device.close()
        self.serial_.open()
        self._lock = threading.Lock()   # NEW：线程互斥锁

    def get_motor_voltage_limits(self, motor_type):
        """
        获取电机的电压限制
        :param motor_type: 电机类型
        :return: (欠压值, 过压值)
        """
        return self.Voltage_Limits.get(motor_type, (18.0, 30.0))  # 默认24V电机限制

    def controlMIT(self, DM_Motor, kp: float, kd: float, q: float, dq: float, tau: float):
        """
        MIT Control Mode Function 达妙电机MIT控制模式函数
        :param DM_Motor: Motor object 电机对象
        :param kp: kp
        :param kd:  kd
        :param q:  position  期望位置
        :param dq:  velocity  期望速度
        :param tau: torque  期望力矩
        :return: None
        """
        if DM_Motor.SlaveID not in self.motors_map:
            print("controlMIT ERROR : Motor ID not found")
            return
        kp_uint = float_to_uint(kp, 0, 500, 12)
        kd_uint = float_to_uint(kd, 0, 5, 12)
        MotorType = DM_Motor.MotorType
        Q_MAX = self.Limit_Param[MotorType][0]
        DQ_MAX = self.Limit_Param[MotorType][1]
        TAU_MAX = self.Limit_Param[MotorType][2]
        q_uint = float_to_uint(q, -Q_MAX, Q_MAX, 16)
        dq_uint = float_to_uint(dq, -DQ_MAX, DQ_MAX, 12)
        tau_uint = float_to_uint(tau, -TAU_MAX, TAU_MAX, 12)
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        data_buf[0] = (q_uint >> 8) & 0xff
        data_buf[1] = q_uint & 0xff
        data_buf[2] = dq_uint >> 4
        data_buf[3] = ((dq_uint & 0xf) << 4) | ((kp_uint >> 8) & 0xf)
        data_buf[4] = kp_uint & 0xff
        data_buf[5] = kd_uint >> 4
        data_buf[6] = ((kd_uint & 0xf) << 4) | ((tau_uint >> 8) & 0xf)
        data_buf[7] = tau_uint & 0xff
        self.__send_data(DM_Motor.SlaveID, data_buf)
        self.recv()  # receive the data from serial port

    def control_delay(self, DM_Motor, kp: float, kd: float, q: float, dq: float, tau: float, delay: float):
        """
        MIT Control Mode Function with delay 达妙电机MIT控制模式函数带延迟
        :param DM_Motor: Motor object 电机对象
        :param kp: kp
        :param kd: kd
        :param q:  position  期望位置
        :param dq:  velocity  期望速度
        :param tau: torque  期望力矩
        :param delay: delay time 延迟时间 单位秒
        """
        self.controlMIT(DM_Motor, kp, kd, q, dq, tau)
        sleep(delay)

    def control_Pos_Vel(self, Motor, P_desired: float, V_desired: float):
        """
        control the motor in position and velocity control mode 电机位置速度控制模式
        :param Motor: Motor object 电机对象
        :param P_desired: desired position 期望位置
        :param V_desired: desired velocity 期望速度
        :return: None
        """
        if Motor.SlaveID not in self.motors_map:
            print("Control Pos_Vel Error : Motor ID not found")
            return
        motorid = 0x100 + Motor.SlaveID
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        P_desired_uint8s = float_to_uint8s(P_desired)
        V_desired_uint8s = float_to_uint8s(V_desired)
        data_buf[0:4] = P_desired_uint8s
        data_buf[4:8] = V_desired_uint8s
        self.__send_data(motorid, data_buf)
        # time.sleep(0.001)
        self.recv()  # receive the data from serial port

    def control_Vel(self, Motor, Vel_desired):
        """
        control the motor in velocity control mode 电机速度控制模式
        :param Motor: Motor object 电机对象
        :param Vel_desired: desired velocity 期望速度
        """
        if Motor.SlaveID not in self.motors_map:
            print("control_VEL ERROR : Motor ID not found")
            return
        motorid = 0x200 + Motor.SlaveID
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        Vel_desired_uint8s = float_to_uint8s(Vel_desired)
        data_buf[0:4] = Vel_desired_uint8s
        self.__send_data(motorid, data_buf)
        self.recv()  # receive the data from serial port

    def control_pos_force(self, Motor, Pos_des: float, Vel_des, i_des):
        """
        control the motor in EMIT control mode 电机力位混合模式
        :param Pos_des: desired position rad  期望位置 单位为rad
        :param Vel_des: desired velocity rad/s  期望速度 为放大100倍
        :param i_des: desired current rang 0-10000 期望电流标幺值放大10000倍
        电流标幺值：实际电流值除以最大电流值，最大电流见上电打印
        """
        if Motor.SlaveID not in self.motors_map:
            print("control_pos_vel ERROR : Motor ID not found")
            return
        motorid = 0x300 + Motor.SlaveID
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        Pos_desired_uint8s = float_to_uint8s(Pos_des)
        data_buf[0:4] = Pos_desired_uint8s
        Vel_uint = np.uint16(Vel_des)
        ides_uint = np.uint16(i_des)
        data_buf[4] = Vel_uint & 0xff
        data_buf[5] = Vel_uint >> 8
        data_buf[6] = ides_uint & 0xff
        data_buf[7] = ides_uint >> 8
        self.__send_data(motorid, data_buf)
        self.recv()  # receive the data from serial port

    def enable(self, Motor):
        """
        enable motor 使能电机
        最好在上电后几秒后再使能电机
        :param Motor: Motor object 电机对象
        """
        self.__control_cmd(Motor, np.uint8(0xFC))
        sleep(0.1)
        self.recv()  # receive the data from serial port

    def enable_old(self, Motor ,ControlMode):
        """
        enable motor old firmware 使能电机旧版本固件，这个是为了旧版本电机固件的兼容性
        可恶的旧版本固件使能需要加上偏移量
        最好在上电后几秒后再使能电机
        :param Motor: Motor object 电机对象
        """
        data_buf = np.array([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfc], np.uint8)
        enable_id = ((int(ControlMode)-1) << 2) + Motor.SlaveID
        self.__send_data(enable_id, data_buf)
        sleep(0.1)
        self.recv()  # receive the data from serial port

    def disable(self, Motor):
        """
        disable motor 失能电机
        :param Motor: Motor object 电机对象
        """
        self.__control_cmd(Motor, np.uint8(0xFD))
        sleep(0.01)

    def set_zero_position(self, Motor):
        """
        set the zero position of the motor 设置电机0位
        :param Motor: Motor object 电机对象
        """
        self.__control_cmd(Motor, np.uint8(0xFE))
        sleep(0.1)
        self.recv()  # receive the data from serial port

    def recv(self):
        # 把上次没有解析完的剩下的也放进来
        data_recv = b''.join([self.data_save, self.serial_.read_all()])
        packets = self.__extract_packets(data_recv)
        for packet in packets:
            data = packet[7:15]
            CANID = (packet[6] << 24) | (packet[5] << 16) | (packet[4] << 8) | packet[3]
            CMD = packet[1]
            self.__process_packet(data, CANID, CMD)

    def recv_set_param_data(self):
        data_recv = self.serial_.read_all()
        packets = self.__extract_packets(data_recv)
        for packet in packets:
            data = packet[7:15]
            CANID = (packet[6] << 24) | (packet[5] << 16) | (packet[4] << 8) | packet[3]
            CMD = packet[1]
            self.__process_set_param_packet(data, CANID, CMD)
    def __process_packet(self, data, CANID, CMD):
        if CMD == 0x11:
            if CANID != 0x00:
                if CANID in self.motors_map:
                    # 解析状态信息: D[0] = MST_ID, D[1] = ID|ERR<<4
                    motor_id = data[1] & 0x0F      # 电机ID (低4位)
                    status = (data[1] >> 4) & 0x0F  # 状态 (高4位)
                    
                    q_uint = np.uint16((np.uint16(data[1]) << 8) | data[2])
                    dq_uint = np.uint16((np.uint16(data[3]) << 4) | (data[4] >> 4))
                    tau_uint = np.uint16(((data[4] & 0xf) << 8) | data[5])
                    # 新增：解析温度数据
                    t_mos = float(data[6])      # D[6] = T_MOS，直接为摄氏度
                    t_rotor = float(data[7])    # D[7] = T_Rotor，直接为摄氏度
                    
                    MotorType_recv = self.motors_map[CANID].MotorType
                    Q_MAX = self.Limit_Param[MotorType_recv][0]
                    DQ_MAX = self.Limit_Param[MotorType_recv][1]
                    TAU_MAX = self.Limit_Param[MotorType_recv][2]
                    recv_q = uint_to_float(q_uint, -Q_MAX, Q_MAX, 16)
                    recv_dq = uint_to_float(dq_uint, -DQ_MAX, DQ_MAX, 12)
                    recv_tau = uint_to_float(tau_uint, -TAU_MAX, TAU_MAX, 12)
                    # 更新电机数据，包括温度和状态
                    self.motors_map[CANID].recv_data(recv_q, recv_dq, recv_tau, t_mos, t_rotor, motor_id, status)
            else:
                MasterID = data[0] & 0x0f
                if MasterID in self.motors_map:
                    # 解析状态信息: D[0] = MST_ID, D[1] = ID|ERR<<4  
                    motor_id = data[1] & 0x0F      # 电机ID (低4位)
                    status = (data[1] >> 4) & 0x0F  # 状态 (高4位)
                    
                    q_uint = np.uint16((np.uint16(data[1]) << 8) | data[2])
                    dq_uint = np.uint16((np.uint16(data[3]) << 4) | (data[4] >> 4))
                    tau_uint = np.uint16(((data[4] & 0xf) << 8) | data[5])
                    # 新增：解析温度数据
                    t_mos = float(data[6])      # D[6] = T_MOS，直接为摄氏度
                    t_rotor = float(data[7])    # D[7] = T_Rotor，直接为摄氏度
                    
                    MotorType_recv = self.motors_map[MasterID].MotorType
                    Q_MAX = self.Limit_Param[MotorType_recv][0]
                    DQ_MAX = self.Limit_Param[MotorType_recv][1]
                    TAU_MAX = self.Limit_Param[MotorType_recv][2]
                    recv_q = uint_to_float(q_uint, -Q_MAX, Q_MAX, 16)
                    recv_dq = uint_to_float(dq_uint, -DQ_MAX, DQ_MAX, 12)
                    recv_tau = uint_to_float(tau_uint, -TAU_MAX, TAU_MAX, 12)
                    # 更新电机数据，包括温度和状态
                    self.motors_map[MasterID].recv_data(recv_q, recv_dq, recv_tau, t_mos, t_rotor, motor_id, status)


    def __process_set_param_packet(self, data, CANID, CMD):
        if CMD == 0x11 and (data[2] == 0x33 or data[2] == 0x55):
            masterid=CANID
            slaveId = ((data[1] << 8) | data[0])
            if CANID==0x00:  #防止有人把MasterID设为0稳一手
                masterid=slaveId

            if masterid not in self.motors_map:
                if slaveId not in self.motors_map:
                    return
                else:
                    masterid=slaveId

            RID = data[3]
            # 读取参数得到的数据
            if is_in_ranges(RID):
                #uint32类型
                num = uint8s_to_uint32(data[4], data[5], data[6], data[7])
                self.motors_map[masterid].temp_param_dict[RID] = num

            else:
                #float类型
                num = uint8s_to_float(data[4], data[5], data[6], data[7])
                self.motors_map[masterid].temp_param_dict[RID] = num


    def addMotor(self, Motor):
        """
        add motor to the motor control object 添加电机到电机控制对象
        :param Motor: Motor object 电机对象
        """
        self.motors_map[Motor.SlaveID] = Motor
        if Motor.MasterID != 0:
            self.motors_map[Motor.MasterID] = Motor
        return True

    def __control_cmd(self, Motor, cmd: np.uint8):
        data_buf = np.array([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, cmd], np.uint8)
        self.__send_data(Motor.SlaveID, data_buf)

    def __send_data(self, motor_id, data):
        """
        send data to the motor 发送数据到电机
        :param motor_id:
        :param data:
        :return:
        """
        self.send_data_frame[13] = motor_id & 0xff
        self.send_data_frame[14] = (motor_id >> 8)& 0xff  #id high 8 bits
        self.send_data_frame[21:29] = data
        self.serial_.write(bytes(self.send_data_frame.T))

    def __read_RID_param(self, Motor, RID):
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8)& 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0x33, np.uint8(RID), 0x00, 0x00, 0x00, 0x00], np.uint8)
        self.__send_data(0x7FF, data_buf)

    def __write_motor_param(self, Motor, RID, data):
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8)& 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0x55, np.uint8(RID), 0x00, 0x00, 0x00, 0x00], np.uint8)
        if not is_in_ranges(RID):
            # data is float
            data_buf[4:8] = float_to_uint8s(data)
        else:
            # data is int
            data_buf[4:8] = data_to_uint8s(int(data))
        self.__send_data(0x7FF, data_buf)
    def switchControlMode(self, Motor, ControlMode):
        """极速版本：预期1-15ms"""
        RID = 10
        max_retries = 5
    
        self.__write_motor_param(Motor, RID, np.uint8(ControlMode))
    
        for i in range(max_retries):
            wait_time = [0.001, 0.002, 0.003, 0.005, 0.008][i]
            sleep(wait_time)
        
            try:
                self.recv_set_param_data()
                if (Motor.SlaveID in self.motors_map and 
                    RID in self.motors_map[Motor.SlaveID].temp_param_dict and
                    self.motors_map[Motor.SlaveID].temp_param_dict[RID] == ControlMode):
                    return True
            except Exception:
                continue
    
        return False
    # def switchControlMode原始版本(self, Motor, ControlMode):
    #     """
    #     switch the control mode of the motor 切换电机控制模式
    #     :param Motor: Motor object 电机对象
    #     :param ControlMode: Control_Type 电机控制模式 example:MIT:Control_Type.MIT MIT模式
    #     """
    #     max_retries = 10
    #     retry_interval = 0.05  #retry times
    #     RID = 10
    #     self.__write_motor_param(Motor, RID, np.uint8(ControlMode))
    #     for _ in range(max_retries):
    #         sleep(retry_interval)
    #         self.recv_set_param_data()
    #         if Motor.SlaveID in self.motors_map:
    #             if RID in self.motors_map[Motor.SlaveID].temp_param_dict:
    #                 if self.motors_map[Motor.SlaveID].temp_param_dict[RID] == ControlMode:
    #                     return True
    #                 else:
    #                     return False
    #     return False

    def save_motor_param(self, Motor):
        """
        save the all parameter  to flash 保存所有电机参数
        :param Motor: Motor object 电机对象
        :return:
        """
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8)& 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        self.disable(Motor)  # before save disable the motor
        self.__send_data(0x7FF, data_buf)
        sleep(0.001)

    def change_limit_param(self, Motor_Type, PMAX, VMAX, TMAX):
        """
        change the PMAX VMAX TMAX of the motor 改变电机的PMAX VMAX TMAX
        :param Motor_Type:
        :param PMAX: 电机的PMAX
        :param VMAX: 电机的VMAX
        :param TMAX: 电机的TMAX
        :return:
        """
        self.Limit_Param[Motor_Type][0] = PMAX
        self.Limit_Param[Motor_Type][1] = VMAX
        self.Limit_Param[Motor_Type][2] = TMAX

    def refresh_motor_status(self,Motor):
        """
        get the motor status 获得电机状态
        """
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8) & 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0xCC, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        self.__send_data(0x7FF, data_buf)
        self.recv()  # receive the data from serial port

    def change_motor_param(self, Motor, RID, data):
        """
        change the RID of the motor 改变电机的参数
        :param Motor: Motor object 电机对象
        :param RID: DM_variable 电机参数
        :param data: 电机参数的值
        :return: True or False ,True means success, False means fail
        """
        max_retries = 20
        retry_interval = 0.05  #retry times

        self.__write_motor_param(Motor, RID, data)
        for _ in range(max_retries):
            self.recv_set_param_data()
            if Motor.SlaveID in self.motors_map and RID in self.motors_map[Motor.SlaveID].temp_param_dict:
                if abs(self.motors_map[Motor.SlaveID].temp_param_dict[RID] - data) < 0.1:
                    return True
                else:
                    return False
            sleep(retry_interval)
        return False

    def read_motor_param(self, Motor, RID):
        """
        read only the RID of the motor 读取电机的内部信息例如 版本号等
        :param Motor: Motor object 电机对象
        :param RID: DM_variable 电机参数
        :return: 电机参数的值
        """
        max_retries = 20
        retry_interval = 0.05  #retry times
        self.__read_RID_param(Motor, RID)
        for _ in range(max_retries):
            sleep(retry_interval)
            self.recv_set_param_data()
            if Motor.SlaveID in self.motors_map:
                if RID in self.motors_map[Motor.SlaveID].temp_param_dict:
                    return self.motors_map[Motor.SlaveID].temp_param_dict[RID]
                else:
                    return None
        return None

    # -------------------------------------------------
    # Extract packets from the serial data
    def __extract_packets(self, data):
        frames = []
        header = 0xAA
        tail = 0x55
        frame_length = 16
        i = 0
        remainder_pos = 0

        while i <= len(data) - frame_length:
            if data[i] == header and data[i + frame_length - 1] == tail:
                frame = data[i:i + frame_length]
                frames.append(frame)
                i += frame_length
                remainder_pos = i
            else:
                i += 1
        self.data_save = data[remainder_pos:]
        return frames


# Motor类增加温度和状态相关属性和方法
class Motor:
    def __init__(self, MotorType, SlaveID, MasterID):
        """
        define Motor object 定义电机对象
        :param MotorType: Motor type 电机类型
        :param SlaveID: CANID 电机ID
        :param MasterID: MasterID 主机ID 建议不要设为0
        """
        self.Pd = float(0)
        self.Vd = float(0)
        self.state_q = float(0)
        self.state_dq = float(0)
        self.state_tau = float(0)
        # 新增温度相关属性
        self.state_t_mos = float(0)      # MOS管温度
        self.state_t_rotor = float(0)    # 电机线圈温度
        # 新增状态相关属性
        self.motor_status = Motor_Status.DISABLED  # 电机状态
        self.motor_id = 0                # 电机ID (从反馈帧获取)
        self.SlaveID = SlaveID
        self.MasterID = MasterID
        self.MotorType = MotorType
        self.isEnable = False
        self.NowControlMode = Control_Type.MIT
        self.temp_param_dict = {}
        
        # 添加电压状态记录
        self.voltage_status_history = []
        self.voltage_check_enabled = True

    def check_voltage_status(self, mc_instance):
        """
        检查电压状态是否合理
        :param mc_instance: MotorControl实例，用于获取电压限制
        :return: (是否正常, 诊断信息)
        """
        if not self.voltage_check_enabled:
            return True, "电压检查已禁用"
            
        # 获取电机的电压限制
        uv_limit, ov_limit = mc_instance.get_motor_voltage_limits(self.MotorType)
        
        # 记录当前状态和时间
        current_status = {
            'timestamp': time.time(),
            'status': self.motor_status,
            'status_text': self.getMotorStatusText()
        }
        self.voltage_status_history.append(current_status)
        
        # 只保留最近100条记录
        if len(self.voltage_status_history) > 100:
            self.voltage_status_history.pop(0)
        
        # 检查当前状态
        if self.motor_status == Motor_Status.OVERVOLTAGE:
            # 检查是否可能是误报（基于电机型号和电压限制）
            motor_name = getattr(DM_Motor_Type, 'DM8009', 'Unknown')
            if self.MotorType == DM_Motor_Type.DM8009:
                # 8009电机是48V电机，36V电源不应该触发过压
                return False, f"电机报告过压，但{self.MotorType.name}是48V电机，36V电源不应该过压。可能是误报或硬件问题。"
            else:
                return False, f"电机报告过压。{self.MotorType.name}的电压限制: 欠压{uv_limit}V, 过压{ov_limit}V"
                
        elif self.motor_status == Motor_Status.UNDERVOLTAGE:
            if self.MotorType == DM_Motor_Type.DM8009:
                # 8009电机是48V电机，36V电源可能触发欠压
                return False, f"电机报告欠压。{self.MotorType.name}是48V电机，36V电源可能低于工作电压。建议使用48V电源。"
            else:
                return False, f"电机报告欠压。{self.MotorType.name}的电压限制: 欠压{uv_limit}V, 过压{ov_limit}V"
        
        return True, "电压状态正常"

    def get_voltage_status_info(self):
        """获取电压状态详细信息"""
        uv_limit, ov_limit = self.mc_instance.get_motor_voltage_limits(self.MotorType) if hasattr(self, 'mc_instance') else (18.0, 30.0)
        
        info = f"电机型号: {self.MotorType.name}\n"
        info += f"电压限制: 欠压{uv_limit}V, 过压{ov_limit}V\n"
        info += f"当前状态: {self.getMotorStatusText()}\n"
        
        if self.voltage_status_history:
            recent_errors = [s for s in self.voltage_status_history[-5:] 
                           if s['status'] in [Motor_Status.OVERVOLTAGE, Motor_Status.UNDERVOLTAGE]]
            if recent_errors:
                info += f"最近电压异常: {len(recent_errors)}次\n"
        
        return info

    def disable_voltage_check(self):
        """禁用电压检查（用于调试或特殊情况下）"""
        self.voltage_check_enabled = False
        print("电压检查已禁用")

    def enable_voltage_check(self):
        """启用电压检查"""
        self.voltage_check_enabled = True
        print("电压检查已启用")
    def recv_data(self, q: float, dq: float, tau: float, t_mos: float = 0, t_rotor: float = 0, 
                  motor_id: int = 0, status: int = 0):
        """
        接收电机数据，包括温度和状态信息
        """
        self.state_q = q
        self.state_dq = dq
        self.state_tau = tau
        self.state_t_mos = t_mos
        self.state_t_rotor = t_rotor
        self.motor_id = motor_id
        try:
            self.motor_status = Motor_Status(status)
        except ValueError:
            self.motor_status = status  # 如果是未知状态，直接存储数值
    # def recv_data(self, q: float, dq: float, tau: float, t_mos: float = 0, t_rotor: float = 0, 
    #               motor_id: int = 0, status: int = 0):
    #     """
    #     接收电机数据，包括温度和状态信息
    #     """
    #     # 记录旧状态
    #     old_status = self.motor_status
        
    #     # 更新数据
    #     self.state_q = q
    #     self.state_dq = dq
    #     self.state_tau = tau
    #     self.state_t_mos = t_mos
    #     self.state_t_rotor = t_rotor
    #     self.motor_id = motor_id
        
    #     try:
    #         self.motor_status = Motor_Status(status)
    #     except ValueError:
    #         self.motor_status = status  # 如果是未知状态，直接存储数值
        
    #     # 状态变化时打印调试信息
    #     # if old_status != self.motor_status:
    #     #     logger.info(f"电机 {self.SlaveID} 状态变化: {old_status} -> {self.motor_status}")
    def getPosition(self):
        """
        get the position of the motor 获取电机位置
        :return: the position of the motor 电机位置
        """
        return self.state_q

    def getVelocity(self):
        """
        get the velocity of the motor 获取电机速度
        :return: the velocity of the motor 电机速度
        """
        return self.state_dq

    def getTorque(self):
        """
        get the torque of the motor 获取电机力矩
        :return: the torque of the motor 电机力矩
        """
        return self.state_tau

    def getT_MOS(self):
        """
        获取MOS管温度
        :return: MOS管温度(摄氏度)
        """
        return self.state_t_mos

    def getT_Rotor(self):
        """
        获取电机线圈温度
        :return: 电机线圈温度(摄氏度)
        """
        return self.state_t_rotor

    def getMotorStatus(self):
        """
        获取电机状态
        :return: 电机状态(Motor_Status枚举或数值)
        """
        return self.motor_status

    def getMotorStatusText(self):
        """
        获取电机状态的文字描述
        :return: 状态描述字符串
        """
        status_dict = {
            Motor_Status.DISABLED: "失能",
            Motor_Status.ENABLED: "使能", 
            Motor_Status.OVERVOLTAGE: "超压",
            Motor_Status.UNDERVOLTAGE: "欠压",
            Motor_Status.OVERCURRENT: "过电流",
            Motor_Status.MOS_OVERTEMP: "MOS过温",
            Motor_Status.COIL_OVERTEMP: "电机线圈过温",
            Motor_Status.COMM_LOST: "通讯丢失",
            Motor_Status.OVERLOAD: "过载"
        }
        return status_dict.get(self.motor_status, f"未知状态({self.motor_status})")

    def getMotorStatusIcon(self):
        """
        获取电机状态的图标
        :return: 状态对应的图标字符串
        """
        status_icons = {
            Motor_Status.DISABLED: "⭕",      # 失能
            Motor_Status.ENABLED: "✅",       # 使能
            Motor_Status.OVERVOLTAGE: "⚡",   # 超压
            Motor_Status.UNDERVOLTAGE: "🔋",  # 欠压
            Motor_Status.OVERCURRENT: "⚠️",   # 过电流
            Motor_Status.MOS_OVERTEMP: "🔥",  # MOS过温
            Motor_Status.COIL_OVERTEMP: "🌡️", # 电机线圈过温
            Motor_Status.COMM_LOST: "📡",     # 通讯丢失
            Motor_Status.OVERLOAD: "🚫"      # 过载
        }
        return status_icons.get(self.motor_status, "❓")

    def isHealthy(self):
        """
        判断电机是否处于健康状态
        :return: True表示健康，False表示有异常
        """
        healthy_states = [Motor_Status.DISABLED, Motor_Status.ENABLED]
        return self.motor_status in healthy_states

    def hasError(self):
        """
        判断电机是否有错误
        :return: True表示有错误，False表示正常
        """
        error_states = [Motor_Status.OVERVOLTAGE, Motor_Status.UNDERVOLTAGE, 
                       Motor_Status.OVERCURRENT, Motor_Status.MOS_OVERTEMP,
                       Motor_Status.COIL_OVERTEMP, Motor_Status.COMM_LOST, 
                       Motor_Status.OVERLOAD]
        return self.motor_status in error_states

    def printStatus(self):
        """
        打印电机完整状态信息
        """
        print(f"🔧 电机 ID {self.SlaveID} 状态报告:")
        print(f"   状态: {self.getMotorStatusIcon()} {self.getMotorStatusText()}")
        print(f"   位置: {self.state_q:.3f} rad")
        print(f"   速度: {self.state_dq:.3f} rad/s") 
        print(f"   扭矩: {self.state_tau:.3f} Nm")
        print(f"   MOS温度: {self.state_t_mos:.1f} ℃")
        print(f"   线圈温度: {self.state_t_rotor:.1f} ℃")
        if self.hasError():
            print(f"   ⚠️  警告: 电机处于异常状态!")

    def getParam(self, RID):
        """
        get the parameter of the motor 获取电机内部的参数，需要提前读取
        :param RID: DM_variable 电机参数
        :return: the parameter of the motor 电机参数
        
        使用技巧:
        1. 调用read_motor_param()读取参数后，参数会临时存储在temp_param_dict中
        2. 然后可以通过getParam()快速获取已读取的参数
        3. 常用参数示例:
           - DM_variable.hw_ver: 硬件版本
           - DM_variable.sw_ver: 软件版本  
           - DM_variable.PMAX: 最大位置
           - DM_variable.VMAX: 最大速度
           - DM_variable.TMAX: 最大扭矩
           - DM_variable.CTRL_MODE: 控制模式
        """
        if RID in self.temp_param_dict:
            return self.temp_param_dict[RID]
        else:
            return None

    def getParamText(self, RID):
        """
        获取参数的文字描述
        :param RID: DM_variable 电机参数
        :return: 参数名称和值的描述
        """
        param_names = {
            DM_variable.UV_Value: "欠压值",
            DM_variable.KT_Value: "扭矩常数", 
            DM_variable.OT_Value: "过温值",
            DM_variable.OC_Value: "过流值",
            DM_variable.ACC: "加速度",
            DM_variable.DEC: "减速度",
            DM_variable.MAX_SPD: "最大速度",
            DM_variable.MST_ID: "主机ID",
            DM_variable.ESC_ID: "从机ID",
            DM_variable.TIMEOUT: "超时时间",
            DM_variable.CTRL_MODE: "控制模式",
            DM_variable.hw_ver: "硬件版本",
            DM_variable.sw_ver: "软件版本",
            DM_variable.PMAX: "最大位置",
            DM_variable.VMAX: "最大速度", 
            DM_variable.TMAX: "最大扭矩"
        }
        
        param_name = param_names.get(RID, f"参数{RID}")
        param_value = self.getParam(RID)
        
        if param_value is not None:
            return f"{param_name}: {param_value}"
        else:
            return f"{param_name}: 未读取"

def LIMIT_MIN_MAX(x, min, max):
    if x <= min:
        x = min
    elif x > max:
        x = max


def float_to_uint(x: float, x_min: float, x_max: float, bits):
    LIMIT_MIN_MAX(x, x_min, x_max)
    span = x_max - x_min
    data_norm = (x - x_min) / span
    return np.uint16(data_norm * ((1 << bits) - 1))


def uint_to_float(x: np.uint16, min: float, max: float, bits):
    span = max - min
    data_norm = float(x) / ((1 << bits) - 1)
    temp = data_norm * span + min
    return np.float32(temp)


def float_to_uint8s(value):
    # Pack the float into 4 bytes
    packed = pack('f', value)
    # Unpack the bytes into four uint8 values
    return unpack('4B', packed)


def data_to_uint8s(value):
    # Check if the value is within the range of uint32
    if isinstance(value, int) and (0 <= value <= 0xFFFFFFFF):
        # Pack the uint32 into 4 bytes
        packed = pack('I', value)
    else:
        raise ValueError("Value must be an integer within the range of uint32")

    # Unpack the bytes into four uint8 values
    return unpack('4B', packed)


def is_in_ranges(number):
    """
    check if the number is in the range of uint32
    :param number:
    :return:
    """
    if (7 <= number <= 10) or (13 <= number <= 16) or (35 <= number <= 36):
        return True
    return False


def uint8s_to_uint32(byte1, byte2, byte3, byte4):
    # Pack the four uint8 values into a single uint32 value in little-endian order
    packed = pack('<4B', byte1, byte2, byte3, byte4)
    # Unpack the packed bytes into a uint32 value
    return unpack('<I', packed)[0]


def uint8s_to_float(byte1, byte2, byte3, byte4):
    # Pack the four uint8 values into a single float value in little-endian order
    packed = pack('<4B', byte1, byte2, byte3, byte4)
    # Unpack the packed bytes into a float value
    return unpack('<f', packed)[0]


def print_hex(data):
    hex_values = [f'{byte:02X}' for byte in data]
    print(' '.join(hex_values))


def get_enum_by_index(index, enum_class):
    try:
        return enum_class(index)
    except ValueError:
        return None

class Control_Type(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    Torque_Pos = 4









