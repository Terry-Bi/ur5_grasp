import time
import os
import datetime
import cv2
import numpy as np
import math
import configparser
from ultralytics import YOLO
from ur5_robot import UR_Robot
from real.realsenseD415 import Camera


class YOLORobotPoseCalculator:
    def __init__(self, cam2end_path, camera_params_path, yolo_model_path, robot, camera):
        # 设备对象
        self.robot = robot
        self.camera = camera  # Realsense深度相机对象

        # 1. 加载参数
        self.cam2end = self.load_cam2end(cam2end_path)
        self.end2cam = self.calculate_transform_inverse(self.cam2end)
        self.camera_matrix, self.dist_coeffs = self.load_camera_params(camera_params_path)

        # 2. 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)

        # 3. 目标配置
        self.target_class_id = 1  # 目标类别ID
        self.conf_threshold = 0.3
        self.iou_threshold = 0.3

        # 类别检测标记
        self.found_target = False
        self.last_robot_pose = None  # 存储最新的解算结果
        self.last_center_depth = None  # 新增：存储最新中心点深度值（米）

        # 中心点绘制参数
        self.center_outer_radius = 10
        self.center_inner_radius = 3
        self.center_outer_color = (0, 0, 0)
        self.center_inner_color = (0, 255, 255)
        self.center_outer_thickness = 2
        self.center_inner_thickness = -1

        # moveL运动参数
        self.moveL_speed = 0.1  # 线性运动速度(m/s)
        self.moveL_acc = 0.1  # 线性运动加速度(m/s²)
        self.moveL_offset = [-0.05, -0.5, 0.61, 0.034, -3.108, 0.630]  # 目标位姿偏移量

        # 安全范围配置
        self.SAFE_POSITION_RANGE = {
            "X": [-0.8, 0.2],
            "Y": [-0.6, 0.6],
            "Z": [0.05, 0.8]
        }
        self.SAFE_ORIENTATION_RANGE = {
            "Roll": [-math.pi, math.pi],
            "Pitch": [-math.pi / 2, math.pi / 2],
            "Yaw": [-math.pi, math.pi]
        }

    # 相机参数加载方法（不变）
    def load_camera_params(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"相机参数文件不存在：{file_path}")

        config = configparser.ConfigParser()
        config.read(file_path)

        # 读取内参（fx, fy为焦距；cx, cy为主点坐标）
        fx = float(config.get('camera_parameters', 'fx'))
        fy = float(config.get('camera_parameters', 'fy'))
        cx = float(config.get('camera_parameters', 'cx'))
        cy = float(config.get('camera_parameters', 'cy'))

        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)

        # 读取畸变系数（Realsense通常已做畸变校正，此处保留兼容）
        k1 = float(config.get('distortion_parameters', 'k1', fallback=0.0))
        k2 = float(config.get('distortion_parameters', 'k2', fallback=0.0))
        p1 = float(config.get('distortion_parameters', 'p1', fallback=0.0))
        p2 = float(config.get('distortion_parameters', 'p2', fallback=0.0))
        k3 = float(config.get('distortion_parameters', 'k3', fallback=0.0))

        dist_coeffs = np.array([[k1, k2, p1, p2, k3]], dtype=np.float32)
        return camera_matrix, dist_coeffs

    # 相机到末端执行器变换矩阵加载方法（不变）
    def load_cam2end(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"cam2end文件不存在：{file_path}")

        cam2end = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                row = list(map(float, line.split()))
                if len(row) == 4:
                    row = [x / 1000.0 for x in row]  # 毫米→米
                    cam2end.append(row)

        if len(cam2end) != 4:
            raise ValueError(f"cam2end矩阵需4行，当前{len(cam2end)}行")

        return np.array(cam2end, dtype=np.float32)

    # 变换矩阵求逆方法（不变）
    def calculate_transform_inverse(self, transform_mat):
        R = transform_mat[:3, :3]
        t = transform_mat[:3, 3].reshape(3, 1)
        R_inv = R.T
        t_inv = -np.dot(R_inv, t)
        inv_transform = np.eye(4, dtype=np.float32)
        inv_transform[:3, :3] = R_inv
        inv_transform[:3, 3] = t_inv.flatten()
        return inv_transform

    # 目标检测（只保留中心点，不变）
    def yolo_detect_target(self, color_image):
        display_img = color_image.copy()
        target_center = None  # 只需要目标中心点（2D像素坐标）

        try:
            # YOLO检测（语义分割模式，用于定位目标区域）
            results = self.yolo_model(
                color_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=640
            )
            result = results[0]

            # 初始化检测状态
            self.found_target = False

            # 可视化检测结果并提取中心点
            if result.masks is not None:
                for mask, cls, conf in zip(result.masks.data, result.boxes.cls, result.boxes.conf):
                    cls_id = int(cls)
                    confidence = float(conf)

                    # 只处理目标类别
                    if cls_id != self.target_class_id:
                        continue

                    self.found_target = True  # 标记为找到目标

                    # 绘制掩码（可视化目标区域）
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    if mask_np.shape != (color_image.shape[0], color_image.shape[1]):
                        mask_np = cv2.resize(mask_np, (color_image.shape[1], color_image.shape[0]))

                    color = (0, 255, 0)  # 目标类别用绿色
                    mask_3d = np.stack([mask_np] * 3, axis=-1) / 255.0
                    display_img = cv2.addWeighted(
                        display_img, 1,
                        (mask_3d * color).astype(np.uint8), 0.3, 0
                    )

                    # 标记类别和置信度
                    x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy()
                    cv2.putText(display_img, f"cls:{cls_id} ({confidence:.2f})",
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

                    # 提取目标中心点（重心）
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(display_img, [largest_contour], -1, (255, 0, 0), 2)

                        # 计算重心（目标中心点的2D像素坐标）
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:  # 避免除以0
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            target_center = (cX, cY)  # 最终获取的2D中心点

                            # 绘制中心点（突出显示）
                            cv2.circle(display_img, (cX, cY), self.center_outer_radius,
                                       self.center_outer_color, self.center_outer_thickness)
                            cv2.circle(display_img, (cX, cY), self.center_inner_radius,
                                       self.center_inner_color, self.center_inner_thickness)

                            # 标注中心点像素坐标
                            center_text = f"Center: ({cX}, {cY})"
                            cv2.putText(display_img, center_text,
                                        (cX + 15, cY - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            print(f"检测出错: {str(e)}")
            return display_img, None  # 只返回图像和中心点（无角点）

        return display_img, target_center  # 移除角点返回，只保留中心点

    # -------------------------- 关键修改1：保存深度值到类变量，用于后续显示 --------------------------
    def get_obj2cam_pose(self, target_center_2d, depth_image):
        """
        输入：目标中心点2D像素坐标、深度图
        输出：目标中心点在相机坐标系下的3D坐标（X, Y, Z）
        新增：将深度值保存到self.last_center_depth，用于实时显示
        """
        # 1. 空值检查
        if target_center_2d is None or depth_image is None:
            print("警告: 中心点或深度图为空，无法计算相机坐标系坐标")
            self.last_center_depth = None  # 无深度时置空
            return None

        cX, cY = target_center_2d  # 中心点2D像素坐标（u, v）
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]  # 相机焦距
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]  # 相机主点坐标

        # 2. 获取中心点的深度值（Realsense深度图单位为毫米，需转换为米）
        depth_roi_size = 5  # 深度采样区域大小（奇数，避免偏移）
        half_roi = depth_roi_size // 2

        # 确保采样区域在深度图范围内
        depth_h, depth_w = depth_image.shape
        start_x = max(0, cX - half_roi)
        end_x = min(depth_w, cX + half_roi + 1)
        start_y = max(0, cY - half_roi)
        end_y = min(depth_h, cY + half_roi + 1)

        # 提取ROI深度区域并过滤无效值（Realsense无效深度为0）
        depth_roi = depth_image[start_y:end_y, start_x:end_x]
        valid_depth = depth_roi[depth_roi > 0]  # 过滤0值（无效深度）

        if len(valid_depth) == 0:
            print(f"警告: 中心点({cX},{cY})周围无有效深度值")
            self.last_center_depth = None  # 无有效深度时置空
            return None

        # 取有效深度的平均值（降低噪声影响）
        avg_depth = np.mean(valid_depth) / 1000.0  # 毫米 → 米
        self.last_center_depth = round(avg_depth, 4)  # 新增：保存深度值（保留4位小数）

        # 3. 核心公式：将像素坐标（u, v）转换为相机坐标系3D坐标（X, Y, Z）
        X_cam = (cX - cx) * avg_depth / fx  # 相机坐标系X轴（右为正）
        Y_cam = (cY - cy) * avg_depth / fy  # 相机坐标系Y轴（下为正）
        Z_cam = avg_depth                   # 相机坐标系Z轴（前为正，与深度方向一致）

        # 返回相机坐标系下的3D坐标（格式：[X, Y, Z]）
        return np.array([X_cam, Y_cam, Z_cam], dtype=np.float32)

    # 欧拉角转旋转矩阵（不变）
    def euler_to_rotation_matrix(self, rpy):
        roll, pitch, yaw = rpy
        R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        return np.dot(R_z, np.dot(R_y, R_x))

    # 获取末端到基座的位姿（不变）
    def get_end2base_pose(self):
        try:
            tcp_pose = self.robot.get_actual_tcp_pose()
            if not (-2 < tcp_pose[0] < 2 and -2 < tcp_pose[1] < 2 and -2 < tcp_pose[2] < 2):
                print("警告: 机械臂位姿超出合理范围")
                return None

            x, y, z = tcp_pose[0], tcp_pose[1], tcp_pose[2]
            rx, ry, rz = tcp_pose[3], tcp_pose[4], tcp_pose[5]

            end2base = np.eye(4, dtype=np.float32)
            end2base[:3, :3] = self.euler_to_rotation_matrix([rx, ry, rz])
            end2base[:3, 3] = [x, y, z]
            return end2base
        except Exception as e:
            print(f"获取机械臂位姿出错: {str(e)}")
            return None

    # 计算基座坐标系位姿（不变）
    def calculate_robot_pose(self, target_center_2d, depth_image):
        # 1. 计算目标中心点在相机坐标系下的3D坐标（核心修改：已在get_obj2cam_pose中保存深度值）
        center_in_cam = self.get_obj2cam_pose(target_center_2d, depth_image)
        if center_in_cam is None:
            self.last_robot_pose = None
            return None

        # 2. 获取机械臂末端到基座的位姿（原有逻辑不变）
        end2base = self.get_end2base_pose()
        if end2base is None:
            self.last_robot_pose = None
            return None

        # 3. 坐标系转换：相机→末端→基座（原有逻辑不变）
        center_homogeneous = np.append(center_in_cam, 1.0)  # 格式：[X, Y, Z, 1]
        center_in_end = self.end2cam @ center_homogeneous  # 相机坐标系 → 末端坐标系
        center_in_base = end2base @ center_in_end          # 末端坐标系 → 基座坐标系

        # 4. 提取基座坐标系下的位置和姿态（姿态可根据需求调整）
        position = center_in_base[:3]  # 基座坐标系下的3D位置（X, Y, Z）
        orientation = [0, -180, 0]     # 手动配置姿态（根据实际抓取需求调整，单位：度）

        # 5. 保存最新解算结果
        self.last_robot_pose = {
            "position": position,
            "orientation": orientation,
            "tcp_pose": self.robot.get_actual_tcp_pose() if self.robot else None
        }
        return self.last_robot_pose

    # 安全校验（不变）
    def check_safe_pose(self, modified_position, modified_orientation_rad):
        if modified_position is None or modified_orientation_rad is None:
            return False, "❌ 位姿数据为空"

        x, y, z = modified_position
        if not (self.SAFE_POSITION_RANGE["X"][0] <= x <= self.SAFE_POSITION_RANGE["X"][1]):
            msg = f"❌ X坐标 {x:.4f}m 超出安全范围 [{self.SAFE_POSITION_RANGE['X'][0]:.4f}, {self.SAFE_POSITION_RANGE['X'][1]:.4f}]"
            return False, msg
        if not (self.SAFE_POSITION_RANGE["Y"][0] <= y <= self.SAFE_POSITION_RANGE["Y"][1]):
            msg = f"❌ Y坐标 {y:.4f}m 超出安全范围 [{self.SAFE_POSITION_RANGE['Y'][0]:.4f}, {self.SAFE_POSITION_RANGE['Y'][1]:.4f}]"
            return False, msg
        if not (self.SAFE_POSITION_RANGE["Z"][0] <= z <= self.SAFE_POSITION_RANGE["Z"][1]):
            msg = f"❌ Z坐标 {z:.4f}m 超出安全范围 [{self.SAFE_POSITION_RANGE['Z'][0]:.4f}, {self.SAFE_POSITION_RANGE['Z'][1]:.4f}]"
            return False, msg

        roll_deg = math.degrees(modified_orientation_rad[0])
        pitch_deg = math.degrees(modified_orientation_rad[1])
        yaw_deg = math.degrees(modified_orientation_rad[2])

        roll_min_deg = math.degrees(self.SAFE_ORIENTATION_RANGE["Roll"][0])
        roll_max_deg = math.degrees(self.SAFE_ORIENTATION_RANGE["Roll"][1])
        pitch_min_deg = math.degrees(self.SAFE_ORIENTATION_RANGE["Pitch"][0])
        pitch_max_deg = math.degrees(self.SAFE_ORIENTATION_RANGE["Pitch"][1])
        yaw_min_deg = math.degrees(self.SAFE_ORIENTATION_RANGE["Yaw"][0])
        yaw_max_deg = math.degrees(self.SAFE_ORIENTATION_RANGE["Yaw"][1])

        if not (roll_min_deg <= roll_deg <= roll_max_deg):
            msg = f"❌ Roll角 {roll_deg:.2f}° 超出安全范围 [{roll_min_deg:.2f}, {roll_max_deg:.2f}]"
            return False, msg
        if not (pitch_min_deg <= pitch_deg <= pitch_max_deg):
            msg = f"❌ Pitch角 {pitch_deg:.2f}° 超出安全范围 [{pitch_min_deg:.2f}, {pitch_max_deg:.2f}]"
            return False, msg
        if not (yaw_min_deg <= yaw_deg <= yaw_max_deg):
            msg = f"❌ Yaw角 {yaw_deg:.2f}° 超出安全范围 [{yaw_min_deg:.2f}, {yaw_max_deg:.2f}]"
            return False, msg

        msg = f"✅ 位姿安全：位置[{x:.4f},{y:.4f},{z:.4f}]m | 姿态[{roll_deg:.2f},{pitch_deg:.2f},{yaw_deg:.2f}]°"
        return True, msg

    # 生成moveL目标位姿（不变）
    def generate_moveL_target(self, robot_pose):
        if robot_pose is None:
            return None

        # 基于中心点坐标计算目标位姿（叠加偏移量）
        target_x = robot_pose["position"][0] + self.moveL_offset[0]
        target_y = robot_pose["position"][1] + self.moveL_offset[1]
        target_z = robot_pose["position"][2] + self.moveL_offset[2]

        target_roll = math.radians(robot_pose["orientation"][0] + self.moveL_offset[3])
        target_pitch = math.radians(robot_pose["orientation"][1] + self.moveL_offset[4])
        target_yaw = math.radians(robot_pose["orientation"][2] + self.moveL_offset[5])

        return [
            round(target_x, 4),
            round(target_y, 4),
            round(target_z, 4),
            round(target_roll, 4),
            round(target_pitch, 4),
            round(target_yaw, 4)
        ]

    # 执行moveL运动（不变）
    def execute_moveL(self, robot_pose):
        if robot_pose is None:
            print("❌ 无法执行moveL：机器人位姿为空")
            return False

        moveL_target = self.generate_moveL_target(robot_pose)
        if moveL_target is None:
            print("❌ 无法生成moveL目标位姿（无有效目标）")
            return False

        try:
            print("\n" + "=" * 60)
            print(f"📢 执行moveL运动至目标中心点位姿:")
            print(f"  X: {moveL_target[0]:.4f} m")
            print(f"  Y: {moveL_target[1]:.4f} m")
            print(f"  Z: {moveL_target[2]:.4f} m")
            print(f"  Roll: {math.degrees(moveL_target[3]):.2f}°")
            print(f"  Pitch: {math.degrees(moveL_target[4]):.2f}°")
            print(f"  Yaw: {math.degrees(moveL_target[5]):.2f}°")
            print(f"  运动参数: 速度={self.moveL_speed}m/s, 加速度={self.moveL_acc}m/s²")
            print("=" * 60)

            self.robot.moveL(
                target_pose=moveL_target,
                speed=self.moveL_speed,
                acceleration=self.moveL_acc
            )

            print("✅ moveL运动完成\n")
            return True
        except Exception as e:
            print(f"❌ moveL运动失败: {str(e)}")
            return False

    # -------------------------- 关键修改2：移除黑色背景矩形+新增深度信息显示 --------------------------
    def draw_pose_info(self, image, robot_pose, fps, target_center, moveL_status=None):
        # 显示类别信息
        cv2.putText(image, f"Target Class: {self.target_class_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 显示检测状态
        status_text = "Found Target" if self.found_target else "No Target"
        status_color = (0, 255, 0) if self.found_target else (0, 0, 255)
        cv2.putText(image, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # 显示中心点像素坐标
        if target_center is not None:
            cv2.putText(image, f"Center Pixel: ({target_center[0]}, {target_center[1]})",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            cv2.putText(image, "Center Pixel: None", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # -------------------------- 新增：实时显示中心点深度信息 --------------------------
        depth_y = 120  # 深度信息显示Y坐标（在中心点坐标下方）
        if self.last_center_depth is not None:
            # 深度有效：显示绿色文本
            cv2.putText(image, f"Center Depth: {self.last_center_depth:.4f} m",
                        (10, depth_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 深度无效：显示红色文本
            cv2.putText(image, "Center Depth: Invalid", (10, depth_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示解算后的中心点6个参数（移除黑色背景矩形，直接绘制文本）
        params_y = 150  # 参数显示起始Y坐标（在深度信息下方）
        param_spacing = 25  # 参数行间距

        # 移除原黑色背景矩形代码，避免遮挡图像
        if robot_pose is not None:
            # 位置参数 (X, Y, Z) - 目标中心点坐标（青色文本，提高可读性）
            cv2.putText(image, f"X: {robot_pose['position'][0]:.4f} m",
                        (10, params_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Y: {robot_pose['position'][1]:.4f} m",
                        (10, params_y + param_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Z: {robot_pose['position'][2]:.4f} m",
                        (10, params_y + 2 * param_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 姿态参数（黄色文本）
            cv2.putText(image, f"Roll: {robot_pose['orientation'][0]:.2f}°",
                        (10, params_y + 3 * param_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Pitch: {robot_pose['orientation'][1]:.2f}°",
                        (10, params_y + 4 * param_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Yaw: {robot_pose['orientation'][2]:.2f}°",
                        (10, params_y + 5 * param_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(image, "No pose data available",
                        (10, params_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示moveL状态
        moveL_y = params_y + 6 * param_spacing
        if moveL_status is not None:
            moveL_text = "moveL: Running" if moveL_status == "running" else "moveL: Done"
            moveL_color = (0, 255, 255) if moveL_status == "running" else (0, 255, 0)
            cv2.putText(image, moveL_text, (10, moveL_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, moveL_color, 2)
        else:
            cv2.putText(image, "moveL: Ready", (10, moveL_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示FPS
        cv2.putText(image, f"FPS: {fps:.1f}", (10, moveL_y + param_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 操作提示
        cv2.putText(image, "Press 'n' to moveL | 's' to save | 'q' to quit",
                    (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image


def main():
    # 配置参数（根据实际路径修改）
    CAM2END_PATH = "cam2end_20251010_zhoutaobi.txt"
    CAMERA_PARAMS_PATH = "camera_20251010_zhoutaobi.ini"
    YOLO_MODEL_PATH = "seg_arm_body_20251010_zhoutaobi.pt"
    ROBOT_IP = "192.168.1.35"
    SAVE_DIR = "./yolo_robot_pose_results"
    TARGET_CLASS_ID = 1

    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        # 初始化机械臂和深度相机
        robot = UR_Robot(
            robot_ip=ROBOT_IP,
            gripper_port=False,
            is_use_robot=True,
            is_use_camera=True
        )
        camera = Camera(width=640, height=480, fps=30)  # Realsense相机（需确保驱动正常）

        # 初始化计算器
        calculator = YOLORobotPoseCalculator(
            cam2end_path=CAM2END_PATH,
            camera_params_path=CAMERA_PARAMS_PATH,
            yolo_model_path=YOLO_MODEL_PATH,
            robot=robot,
            camera=camera
        )
        calculator.target_class_id = TARGET_CLASS_ID  # 设置目标类别

        # 调整运动参数（根据实际需求修改）
        calculator.moveL_speed = 0.15
        calculator.moveL_acc = 0.15
        calculator.moveL_offset = [0, 0, 0.08, 0, 0, 0]  # 位姿偏移量（根据抓取需求微调）

        print("系统初始化成功，开始检测...")
        print(f"目标类别ID: {TARGET_CLASS_ID}")
        print("=" * 60)
        print("操作说明：")
        print("  - 按 'n' 键：执行moveL运动到目标中心点")
        print("  - 按 's' 键：保存当前帧数据")
        print("  - 按 'q' 键：退出程序")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"初始化失败：{e}")
        return

    frame_count = 0
    start_time = time.time()
    save_count = 0
    moveL_status = None

    # 主循环（不变，深度值已在calculate_robot_pose中自动更新）
    while True:
        # 1. 获取深度相机数据（color图用于检测，depth图用于深度值获取）
        color_image, depth_image = camera.get_data()
        if color_image is None or depth_image is None:
            time.sleep(0.1)
            print("警告: 未获取到相机数据，重试...")
            continue

        # 2. 检测目标（只返回图像和中心点2D坐标）
        display_img, target_center = calculator.yolo_detect_target(color_image)

        # 3. 计算目标中心点在基座坐标系下的位姿（传入中心点+深度图，自动更新深度值）
        robot_pose = None
        if target_center is not None:
            robot_pose = calculator.calculate_robot_pose(target_center, depth_image)
        else:
            print("未检测到有效目标中心点，无法计算位姿")

        # 4. 计算FPS并绘制信息（自动显示深度值）
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        display_img = calculator.draw_pose_info(display_img, robot_pose, fps, target_center, moveL_status)
        cv2.imshow("Detection + moveL Control (Depth-based)", display_img)

        # 5. 键盘事件处理（不变）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n程序退出")
            break
        elif key == ord('s'):
            save_count += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            color_save_path = os.path.join(SAVE_DIR, f"color_{timestamp}_{save_count:03d}.jpg")
            cv2.imwrite(color_save_path, display_img)
            print(f"已保存第{save_count}组数据: {color_save_path}")
        elif key == ord('n'):
            if robot_pose is None:
                print("❌ 无法执行moveL：未检测到有效目标位姿")
                continue

            # 运动前更新状态并显示
            moveL_status = "running"
            cv2.imshow("Detection + moveL Control (Depth-based)", display_img)
            cv2.waitKey(1)

            # 调整目标位姿（根据实际抓取需求修改Z轴高度和姿态）
            modified_robot_pose = {
                "position": robot_pose["position"].copy(),
                "orientation": robot_pose["orientation"].copy(),
                "tcp_pose": robot_pose["tcp_pose"]
            }
            modified_robot_pose["position"][2] = 0.4  # 调整抓取高度（避免碰撞）
            modified_robot_pose["orientation"] = [0, -180, 0]  # 固定姿态

            # 安全校验
            modified_orientation_rad = [
                math.radians(modified_robot_pose["orientation"][0]),
                math.radians(modified_robot_pose["orientation"][1]),
                math.radians(modified_robot_pose["orientation"][2])
            ]
            is_safe, safe_msg = calculator.check_safe_pose(
                modified_position=modified_robot_pose["position"],
                modified_orientation_rad=modified_orientation_rad
            )
            print(safe_msg)

            if not is_safe:
                print("❌ 位姿不安全，取消moveL运动")
                moveL_status = None
                continue

            # 执行运动
            success = calculator.execute_moveL(modified_robot_pose)
            moveL_status = "done" if success else None

    # 释放资源
    cv2.destroyAllWindows()
    if hasattr(robot, 'rtde_c') and robot.rtde_c:
        robot.rtde_c.stopScript()


if __name__ == "__main__":
    main()