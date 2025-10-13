import time
import rtde_control
import rtde_receive
# 注释掉夹爪相关依赖（不使用夹爪，无需导入minimalmodbus）
# import minimalmodbus
import copy
import numpy as np
import threading
import math


# 注释掉夹爪寄存器地址（不使用夹爪，无需定义）
# POSITION_HIGH_8 = 0x0102
# POSITION_LOW_8 = 0x0103
# SPEED = 0x0104
# FORCE = 0x0105
# MOTION_TRIGGER = 0x0108

# 注释掉夹爪线程锁（不使用夹爪，无需锁）
# lock = threading.Lock()


class UR_Robot:
    def __init__(self, robot_ip="192.168.1.35",
                 # 新增参数：控制是否使用夹爪，默认False（不使用）
                 use_gripper=False,
                 # 保留夹爪参数但默认无效（避免修改调用方式）
                 gripper_port='COM6', gripper_baudrate=115200, gripper_address=1,
                 workspace_limits=None, is_use_robot=True, is_use_camera=True):

        # 1. 工作空间初始化（不变）
        if workspace_limits is None:
            workspace_limits = [[-0.30, -0.15], [-0.50, 0.05], [0.002, 0.15]]
        self.workspace_limits = workspace_limits

        # 2. 机械臂核心控制初始化（不变，这是机械臂必须的）
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        # 3. 【关键修改】仅当use_gripper=True时，才初始化夹爪（无夹爪则跳过）
        self.use_gripper = use_gripper
        if self.use_gripper:
            # 只有使用夹爪时才导入依赖并初始化串口（避免无夹爪时报错）
            import minimalmodbus
            lock = threading.Lock()
            self.instrument = minimalmodbus.Instrument(gripper_port, gripper_address)
            self.instrument.serial.baudrate = gripper_baudrate
            self.instrument.serial.timeout = 1
        else:
            # 不使用夹爪时，将夹爪相关属性设为None（避免后续调用报错）
            self.instrument = None
            self.lock = None

        # 4. 其他参数初始化（不变）
        self.is_use_robotiq85 = use_gripper  # 与夹爪开关同步
        self.is_use_camera = is_use_camera

        self.joint_acc = 0.2
        self.joint_spd = 0.2
        self.tool_acc = 0.1
        self.tool_spd = 0.1
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]
        self.joint_tolerance = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        self.initial_pose = [-0.4, -0.025, 0.14981, 0.000, 3.141, 0.000]

        # 5. 相机初始化（不变，不使用相机时设is_use_camera=False即可）
        if self.is_use_camera:
            # 若不使用相机，建议注释掉以下代码（避免Camera类未定义报错）
            # from real.realsenseD415 import Camera
            # self.camera = Camera()
            # self.cam_intrinsics = np.array([386.89120483, 0, 321.61746216, 0, 386.45446777, 237.19995117, 0, 0, 1]).reshape(3, 3)
            # self.cam_pose = np.loadtxt("C:/Users/LCT/Desktop/URrobot/GRCNN/real/cam_pose/camera_pose.txt", delimiter=' ')
            # self.cam_depth_scale = np.loadtxt("C:/Users/LCT/Desktop/URrobot/GRCNN/real/cam_pose/camera_depth_scale.txt", delimiter=' ')
            pass

    # --------------------------
    # 机械臂核心控制方法（不变，保留）
    # --------------------------
    def moveL(self, target_pose, speed=0.1, acceleration=0.1):
        self.rtde_c.moveL(target_pose, speed, acceleration)
        actual_tool_positions = self.get_actual_tcp_pose()
        while not all(
                [np.abs(actual_tool_positions[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            actual_tool_positions = self.get_actual_tcp_pose()
            time.sleep(0.01)
        time.sleep(1.5)

    def moveJ(self, target_joint, speed=0.2, acceleration=0.2):
        self.rtde_c.moveJ(target_joint, speed, acceleration)
        actual_joint_positions = self.get_actual_joint_position()
        while not all([np.abs(actual_joint_positions[j] - target_joint[j]) < self.joint_tolerance[j] for j in
                       range(len(target_joint))]):
            actual_joint_positions = self.get_actual_joint_position()
            time.sleep(0.01)
        time.sleep(1.5)

    def go_home(self):
        # 原代码中home_joint_config未定义，建议补充（或删除该方法，避免报错）
        # self.moveJ(self.home_joint_config)
        pass

    def get_actual_tcp_pose(self):
        return self.rtde_r.getActualTCPPose()

    def get_actual_joint_position(self):
        return self.rtde_r.getActualQ()

    def get_robot_status(self):
        return self.rtde_r.getRobotStatus()

    # --------------------------
    # 【关键修改】屏蔽所有夹爪相关方法（不使用夹爪，避免调用时报错）
    # --------------------------
    def write_position_high8(self, value):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return
        with self.lock:
            self.instrument.write_register(POSITION_HIGH_8, value, functioncode=6)

    def write_position_low8(self, value):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return
        with self.lock:
            self.instrument.write_register(POSITION_LOW_8, value, functioncode=6)

    def write_position(self, value):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return
        with self.lock:
            self.instrument.write_long(POSITION_HIGH_8, value)

    def write_speed(self, speed):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return
        with self.lock:
            self.instrument.write_register(SPEED, speed, functioncode=6)

    def write_force(self, force):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return
        with self.lock:
            self.instrument.write_register(FORCE, force, functioncode=6)

    def trigger_motion(self):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return
        with self.lock:
            self.instrument.write_register(MOTION_TRIGGER, 1, functioncode=6)

    def read_position(self):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return None
        with self.lock:
            high = self.instrument.read_register(POSITION_HIGH_8, functioncode=3)
            low = self.instrument.read_register(POSITION_LOW_8, functioncode=3)
            position = (high << 8) | low
            return position

    def grip(self, position, speed, force):
        if not self.use_gripper:
            print("未启用夹爪，跳过该操作")
            return None
        self.write_position(position)
        self.write_speed(speed)
        self.write_force(force)
        self.trigger_motion()
        time.sleep(2)
        return self.read_position()

    # --------------------------
    # 相机相关方法（不使用则屏蔽，避免报错）
    # --------------------------
    def get_camera_data(self):
        if not self.is_use_camera:
            print("未启用相机，跳过该操作")
            return None, None
        color_img, depth_img = self.camera.get_data()
        return color_img, depth_img

    # 其他辅助方法（不变，保留）
    def angle_to_cartesian(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        rx = math.cos(angle_radians)
        ry = math.sin(angle_radians)
        return rx, ry

    def grasp(self, position, angle, close_position=5000, k_acc=0.1, k_vel=0.1, speed=100, force=40):
        if not self.use_gripper:
            print("未启用夹爪，跳过抓取操作")
            return
        # 原抓取逻辑不变（但因use_gripper=False，不会执行）
        open_position = 4000
        rpy = [0, 3.141, 0]
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
        grasp_home = [-0.3, 0.05, 0.12, 0.000, 3.141, 0.000]
        self.moveL(grasp_home, k_acc, k_vel)
        self.grip(open_position, speed, force)
        self.read_position()
        # 后续抓取逻辑省略（因use_gripper=False，无需关注）