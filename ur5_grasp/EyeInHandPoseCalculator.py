import time
import os
import datetime
import cv2
import numpy as np
import math
import configparser
from ultralytics import YOLO
from ur5_robot_bsp import UR_Robot
from real.realsenseD415 import Camera


class EyeInHandPoseCalculator:
    def __init__(self, camera_params_path, cam2end_txt_path, yolo_model_path, robot, camera):
        # 设备对象（相机刚性固定在末端，存在相对位置）
        self.robot = robot
        self.camera = camera  # Realsense深度相机

        # 调试模式（优先初始化，确保加载参数时可用）
        self.debug = True

        # 1. 加载核心标定参数（相机内参 + cam2end变换矩阵）
        self.camera_matrix, self.dist_coeffs = self.load_camera_params(camera_params_path)
        self.cam2end, self.end2cam = self.load_cam2end_from_txt(cam2end_txt_path)
        print("✅ 加载完成：相机内参 + 相机→末端变换矩阵(cam2end)")

        # 2. 加载YOLO目标检测模型
        self.yolo_model = YOLO(yolo_model_path)

        # 3. 目标检测与位姿计算配置
        self.target_class_id = 1  # 目标类别ID（需与YOLO训练标签对应）
        self.conf_threshold = 0.3  # 检测置信度阈值
        self.iou_threshold = 0.3  # IOU阈值（过滤重复检测框）
        self.depth_roi_size = 8  # 深度采样区域大小（降低噪声）

        # 4. 状态变量（实时更新）
        self.found_target = False  # 是否检测到目标
        self.last_robot_pose = None  # 最近一次计算的目标基座位姿
        self.last_center_depth = None  # 最近一次目标深度值
        self.last_target_center = None  # 最近一次目标中心点

        # 5. 机械臂运动安全配置
        self.moveL_speed = 0.08  # 直线运动速度（m/s）
        self.moveL_acc = 0.08  # 直线运动加速度（m/s²）
        self.moveL_offset = [0, 0, 0.05, 0, 0, 0]  # 运动偏移（Z轴+5cm避免碰撞）
        # 安全位置范围（根据实际机械臂工作空间调整！）
        self.SAFE_POSITION_RANGE = {
            "X": [-100, 100],
            "Y": [-100, 100],
            "Z": [-100, 100]
        }
        # 安全姿态范围（欧拉角，单位：弧度）
        self.SAFE_ORIENTATION_RANGE = {
            "Roll": [-math.pi, math.pi],
            "Pitch": [-math.pi, math.pi],
            "Yaw": [-math.pi, math.pi]
        }

        # 6. 可视化配置
        self.center_outer_radius = 10  # 中心点外圆半径
        self.center_inner_radius = 3  # 中心点内圆半径
        self.center_outer_color = (0, 0, 0)  # 外圆颜色（黑色）
        self.center_inner_color = (0, 255, 255)  # 内圆颜色（黄色）
        self.center_outer_thickness = 2  # 外圆线宽
        self.center_inner_thickness = -1  # 内圆填充（-1表示填充）

    """
    1. 加载相机内参（适配自定义INI格式：[camera_parameters] + [distortion_parameters]）
    """

    def load_camera_params(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"相机内参文件不存在：{file_path}")

        config = configparser.ConfigParser()
        config.read(file_path, encoding="utf-8")

        # 读取内参（与你的INI节名匹配）
        fx = float(config.get('camera_parameters', 'fx'))
        fy = float(config.get('camera_parameters', 'fy'))
        cx = float(config.get('camera_parameters', 'cx'))
        cy = float(config.get('camera_parameters', 'cy'))
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)

        # 读取畸变系数（与你的INI节名匹配）
        k1 = float(config.get('distortion_parameters', 'k1', fallback=0.0))
        k2 = float(config.get('distortion_parameters', 'k2', fallback=0.0))
        p1 = float(config.get('distortion_parameters', 'p1', fallback=0.0))
        p2 = float(config.get('distortion_parameters', 'p2', fallback=0.0))
        k3 = float(config.get('distortion_parameters', 'k3', fallback=0.0))
        dist_coeffs = np.array([[k1, k2, p1, p2, k3]], dtype=np.float32)

        if self.debug:
            print(f"📷 相机内参矩阵:\n{camera_matrix}")
            print(f"📷 畸变系数: {dist_coeffs}")
        return camera_matrix, dist_coeffs

    """
    2. 加载cam2end矩阵（相机→末端的固定安装位姿，从TXT读取4x4齐次矩阵）
    """

    def load_cam2end_from_txt(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"cam2end矩阵文件不存在：{file_path}")

        # 读取4x4矩阵（每行4个数字，共4行）
        cam2end = []
        with open(file_path, 'r') as f:
            for line_num in range(4):
                line = f.readline().strip()
                if not line:
                    raise ValueError(f"cam2end文件第{line_num + 1}行为空，格式错误")
                elements = list(map(float, line.split()))
                if len(elements) != 4:
                    raise ValueError(f"cam2end文件第{line_num + 1}行需4个元素，实际{len(elements)}个")
                cam2end.append(elements)

        cam2end = np.array(cam2end, dtype=np.float32)
        # 校验齐次矩阵合法性（最后一行必须是[0,0,0,1]）
        if not np.allclose(cam2end[3], [0.0, 0.0, 0.0, 1.0], atol=1e-3):
            raise ValueError(f"cam2end矩阵非法！最后一行应为[0,0,0,1]，实际为{cam2end[3]}")

        # 计算末端→相机的逆矩阵（备用）
        end2cam = np.linalg.inv(cam2end)

        if self.debug:
            print(f"🔗 相机→末端变换矩阵(cam2end):\n{cam2end}")
            print(f"🔗 末端→相机变换矩阵(end2cam):\n{end2cam}")
        return cam2end, end2cam

    """
    3. YOLO目标检测（输出目标中心点+可视化标注）
    """

    def yolo_detect_target(self, color_image):
        display_img = color_image.copy()
        target_center = None
        self.found_target = False

        try:
            # YOLO推理（静默模式）
            results = self.yolo_model(
                color_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=640
            )
            result = results[0]

            # 优先使用语义分割掩码（更精准的中心点计算）
            if result.masks is not None:
                # 遍历所有检测结果，筛选目标类别
                for idx, (mask, cls, conf) in enumerate(zip(result.masks.data, result.boxes.cls, result.boxes.conf)):
                    cls_id = int(cls)
                    confidence = float(conf)
                    if cls_id != self.target_class_id:
                        continue
                    self.found_target = True

                    # 掩码后处理（适配原图尺寸）
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    if mask_np.shape != (color_image.shape[0], color_image.shape[1]):
                        mask_np = cv2.resize(mask_np, (color_image.shape[1], color_image.shape[0]))

                    # 可视化：半透明掩码
                    mask_3d = np.stack([mask_np] * 3, axis=-1) / 255.0
                    display_img = cv2.addWeighted(display_img, 0.7, (mask_3d * (0, 255, 0)).astype(np.uint8), 0.3, 0)

                    # 可视化：边界框+置信度
                    x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy()
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(display_img, f"Cls:{cls_id} (Conf:{confidence:.2f})",
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 计算掩码的中心（轮廓矩方法，抗偏移）
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M["m00"] > 0:  # 避免除以0（轮廓面积不为0）
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            target_center = (cX, cY)
                            self.last_target_center = target_center  # 实时更新中心点

                            # 可视化：中心点双圆标记
                            cv2.circle(display_img, (cX, cY), self.center_outer_radius,
                                       self.center_outer_color, self.center_outer_thickness)
                            cv2.circle(display_img, (cX, cY), self.center_inner_radius,
                                       self.center_inner_color, self.center_inner_thickness)
                            cv2.putText(display_img, f"Center:({cX},{cY})",
                                        (cX + 15, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 若无掩码（仅目标检测框），用框中心近似
            elif result.boxes is not None and len(result.boxes) > 0:
                for idx, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
                    cls_id = int(cls)
                    confidence = float(conf)
                    if cls_id != self.target_class_id:
                        continue
                    self.found_target = True
                    x1, y1, x2, y2 = box.cpu().numpy()
                    cX = int((x1 + x2) / 2)
                    cY = int((y1 + y2) / 2)
                    target_center = (cX, cY)
                    self.last_target_center = target_center

                    # 可视化：边界框+中心点
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.circle(display_img, (cX, cY), self.center_outer_radius,
                               self.center_outer_color, self.center_outer_thickness)
                    cv2.circle(display_img, (cX, cY), self.center_inner_radius,
                               self.center_inner_color, self.center_inner_thickness)

        except Exception as e:
            print(f"❌ 目标检测出错: {str(e)}")
        return display_img, target_center

    """
    4. 关键修复1：目标→相机坐标系（修正相机方向，输出齐次坐标）
    """

    def get_obj2cam_pose(self, target_center_2d, depth_image):
        if target_center_2d is None or depth_image is None:
            print("⚠️  中心点或深度图为空，无法计算相机坐标")
            self.last_center_depth = None
            return None

        cX, cY = target_center_2d
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        # 步骤1：深度采样（ROI区域滤波，降低噪声）
        half_roi = self.depth_roi_size // 2
        depth_h, depth_w = depth_image.shape
        # 确保ROI不超出图像边界
        start_x = max(0, cX - half_roi)
        end_x = min(depth_w, cX + half_roi + 1)
        start_y = max(0, cY - half_roi)
        end_y = min(depth_h, cY + half_roi + 1)
        depth_roi = depth_image[start_y:end_y, start_x:end_x]

        # 步骤2：过滤无效深度（Realsense无效深度为0）
        valid_depth = depth_roi[depth_roi > 10]  # 过滤小于10mm的噪声
        if len(valid_depth) < 3:  # 至少3个有效点才可信
            print(f"⚠️  中心点({cX},{cY})有效深度点不足（仅{len(valid_depth)}个）")
            self.last_center_depth = None
            return None

        # 步骤3：深度值优化（中位数滤波抗 outliers）
        avg_depth = np.median(valid_depth) / 1000.0  # mm → m
        self.last_center_depth = round(avg_depth, 4)

        # 步骤4：相机坐标系计算（修正方向：Realsense Y下→机械臂Y上，Z前→Z前）
        X_cam = (cX - cx) * avg_depth / fx  # 相机X（右）→ 机械臂X（右）：一致
        Y_cam = -(cY - cy) * avg_depth / fy  # 相机Y（下）→ 机械臂Y（上）：取反（关键修正！）
        Z_cam = avg_depth  # 相机Z（前）→ 机械臂Z（上）：需根据安装方向调整（若倒置则取反）

        # 输出齐次坐标（4x1），方便后续矩阵运算
        obj_in_cam_hom = np.array([X_cam, Y_cam, Z_cam, 1.0], dtype=np.float32).reshape(4, 1)

        if self.debug:
            print(f"\n📐 相机坐标系目标坐标: X={X_cam:.4f}m, Y={Y_cam:.4f}m, Z={Z_cam:.4f}m")
            print(f"📏 目标深度: {self.last_center_depth:.4f}m")
        return obj_in_cam_hom

    """
    5. 关键修复2：相机→末端坐标系（用cam2end矩阵动态转换）
    """

    def transform_cam2end(self, obj_in_cam_hom):
        if obj_in_cam_hom is None:
            print("⚠️  相机坐标为空，无法转换到末端坐标")
            return None

        # 核心：相机→末端的齐次变换（cam2end是相机相对于末端的安装位姿）
        # 公式：obj_end = cam2end * obj_cam（矩阵乘法顺序不能错！）
        obj_in_end_hom = np.dot(self.cam2end, obj_in_cam_hom)

        # 提取3D坐标（前3个元素）
        obj_in_end = obj_in_end_hom[:3].flatten()  # 转为1D数组（X,Y,Z）

        if self.debug:
            print(f"🔄 末端坐标系目标坐标: X={obj_in_end[0]:.4f}m, Y={obj_in_end[1]:.4f}m, Z={obj_in_end[2]:.4f}m")
        return obj_in_end

    """
    6. 关键修复3：末端→基座坐标系（实时读取末端位姿，动态计算）
    """

    def get_end2base_pose(self):
        try:
            # 实时读取UR5末端TCP位姿（[X,Y,Z,Roll,Pitch,Yaw]，单位：m/弧度）
            tcp_pose = self.robot.get_actual_tcp_pose()
            if tcp_pose is None or len(tcp_pose) != 6:
                print("⚠️  未获取到有效末端TCP位姿（需6个参数：X,Y,Z,Roll,Pitch,Yaw）")
                return None

            x_end, y_end, z_end = tcp_pose[0], tcp_pose[1], tcp_pose[2]
            roll_end, pitch_end, yaw_end = tcp_pose[3], tcp_pose[4], tcp_pose[5]

            # 安全校验：末端是否在工作空间内
            if not (self.SAFE_POSITION_RANGE["X"][0] <= x_end <= self.SAFE_POSITION_RANGE["X"][1] and
                    self.SAFE_POSITION_RANGE["Y"][0] <= y_end <= self.SAFE_POSITION_RANGE["Y"][1] and
                    self.SAFE_POSITION_RANGE["Z"][0] <= z_end <= self.SAFE_POSITION_RANGE["Z"][1]):
                print(f"⚠️  末端超出安全范围: X={x_end:.4f}, Y={y_end:.4f}, Z={z_end:.4f}")
                return None

            # 构建末端→基座的4x4齐次变换矩阵（UR机械臂默认Z-Y-X欧拉角顺序）
            end2base = np.eye(4, dtype=np.float32)

            # 1. 旋转矩阵（Z-Y-X顺序）
            # Yaw（绕Z轴）
            Rz = np.array([[math.cos(yaw_end), -math.sin(yaw_end), 0],
                           [math.sin(yaw_end), math.cos(yaw_end), 0],
                           [0, 0, 1]], dtype=np.float32)
            # Pitch（绕Y轴）
            Ry = np.array([[math.cos(pitch_end), 0, math.sin(pitch_end)],
                           [0, 1, 0],
                           [-math.sin(pitch_end), 0, math.cos(pitch_end)]], dtype=np.float32)
            # Roll（绕X轴）
            Rx = np.array([[1, 0, 0],
                           [0, math.cos(roll_end), -math.sin(roll_end)],
                           [0, math.sin(roll_end), math.cos(roll_end)]], dtype=np.float32)
            # 组合旋转矩阵（Z→Y→X顺序，矩阵乘法顺序为 Rz*Ry*Rx）
            end2base[:3, :3] = np.dot(Rz, np.dot(Ry, Rx))

            # 2. 平移向量（末端在基座坐标系的位置）
            end2base[:3, 3] = [x_end, y_end, z_end]

            if self.debug:
                print(f"\n🤖 末端实时位姿（基座坐标系）:")
                print(f"   位置: X={x_end:.4f}m, Y={y_end:.4f}m, Z={z_end:.4f}m")
                print(
                    f"   姿态: Roll={math.degrees(roll_end):.2f}°, Pitch={math.degrees(pitch_end):.2f}°, Yaw={math.degrees(yaw_end):.2f}°")

            return end2base

        except Exception as e:
            print(f"❌ 获取末端位姿出错: {str(e)}")
            return None

    """
    7. 核心修复：目标基座位姿计算（完整动态链路）
    流程：目标像素 → 相机齐次坐标 → 末端3D坐标 → 末端基座矩阵 → 目标基座坐标
    """

    def calculate_robot_pose(self, target_center_2d, depth_image):
        # 步骤1：目标→相机齐次坐标（修正方向）
        obj_in_cam_hom = self.get_obj2cam_pose(target_center_2d, depth_image)
        if obj_in_cam_hom is None:
            self.last_robot_pose = None
            return None

        # 步骤2：相机→末端3D坐标（用cam2end矩阵）
        obj_in_end = self.transform_cam2end(obj_in_cam_hom)
        if obj_in_end is None:
            self.last_robot_pose = None
            return None

        # 步骤3：获取末端→基座实时变换矩阵
        end2base = self.get_end2base_pose()
        if end2base is None:
            self.last_robot_pose = None
            return None

        # 步骤4：目标→基座坐标系（核心动态计算）
        try:
            # 将末端坐标系的目标坐标转为齐次坐标（4x1）
            obj_in_end_hom = np.array([obj_in_end[0], obj_in_end[1], obj_in_end[2], 1.0], dtype=np.float32).reshape(4,
                                                                                                                    1)

            # 核心公式：目标基座坐标 = 末端基座矩阵 * 目标末端坐标
            obj_in_base_hom = np.dot(end2base, obj_in_end_hom)
            obj_in_base = obj_in_base_hom[:3].flatten()  # 提取3D坐标（X,Y,Z）

            # 手动校准偏移（需替换为实际测量的Δx, Δy, Δz）
            delta_x = -26.08124  # 示例：x方向补偿0.02m
            delta_y = -5.3292  # 示例：y方向补偿-0.01m
            delta_z = -13.5320  # 示例：z方向补偿0.03m

            obj_in_base[0] += delta_x
            obj_in_base[1] += delta_y
            obj_in_base[2] += delta_z


            # 目标姿态：根据抓取需求设置（示例：吸盘朝下，与末端姿态一致）
            # 若需自定义姿态，可修改此处（如：[0, math.pi, 0] 表示Pitch=180°）
            target_orientation = [0.0, math.pi, 0.0]  # 弧度单位
            target_orientation_deg = [math.degrees(ang) for ang in target_orientation]

            # 调试日志：实时打印目标基座位姿（验证是否变化）
            if self.debug:
                print(f"\n🎯 目标实时基座位姿:")
                print(f"   位置: X={obj_in_base[0]:.4f}m, Y={obj_in_base[1]:.4f}m, Z={obj_in_base[2]:.4f}m")
                print(
                    f"   姿态: Roll={target_orientation_deg[0]:.2f}°, Pitch={target_orientation_deg[1]:.2f}°, Yaw={target_orientation_deg[2]:.2f}°")

            # 保存实时位姿
            self.last_robot_pose = {
                "position": obj_in_base,
                "orientation": target_orientation,
                "tcp_pose": self.robot.get_actual_tcp_pose() if self.robot else None
            }
            return self.last_robot_pose

        except Exception as e:
            print(f"❌ 目标基座位姿计算出错: {str(e)}")
            self.last_robot_pose = None
            return None

    """
    8. 机械臂运动安全校验（不变）
    """

    def check_safe_pose(self, modified_position, modified_orientation):
        if modified_position is None or modified_orientation is None:
            return False, "❌ 位姿数据为空"

        x, y, z = modified_position
        # 位置安全校验
        if not (self.SAFE_POSITION_RANGE["X"][0] <= x <= self.SAFE_POSITION_RANGE["X"][1]):
            return False, f"❌ X超出范围[{self.SAFE_POSITION_RANGE['X'][0]},{self.SAFE_POSITION_RANGE['X'][1]}]"
        if not (self.SAFE_POSITION_RANGE["Y"][0] <= y <= self.SAFE_POSITION_RANGE["Y"][1]):
            return False, f"❌ Y超出范围[{self.SAFE_POSITION_RANGE['Y'][0]},{self.SAFE_POSITION_RANGE['Y'][1]}]"
        if not (self.SAFE_POSITION_RANGE["Z"][0] <= z <= self.SAFE_POSITION_RANGE["Z"][1]):
            return False, f"❌ Z超出范围[{self.SAFE_POSITION_RANGE['Z'][0]},{self.SAFE_POSITION_RANGE['Z'][1]}]"

        # 姿态安全校验
        roll, pitch, yaw = modified_orientation
        if not (self.SAFE_ORIENTATION_RANGE["Roll"][0] <= roll <= self.SAFE_ORIENTATION_RANGE["Roll"][1]):
            return False, f"❌ Roll角超出安全范围"
        if not (self.SAFE_ORIENTATION_RANGE["Pitch"][0] <= pitch <= self.SAFE_ORIENTATION_RANGE["Pitch"][1]):
            return False, f"❌ Pitch角超出安全范围"
        if not (self.SAFE_ORIENTATION_RANGE["Yaw"][0] <= yaw <= self.SAFE_ORIENTATION_RANGE["Yaw"][1]):
            return False, f"❌ Yaw角超出安全范围"

        # 距离安全校验（目标与相机过近）
        if self.last_center_depth is not None:
            safe_distance = 0.10
            if self.last_center_depth < safe_distance:
                return False, f"❌ 目标与相机过近，当前距离{self.last_center_depth:.2f}m，安全距离{safe_distance}m"

        return True, f"✅ 位姿安全：X={x:.4f}, Y={y:.4f}, Z={z:.4f} | Roll={math.degrees(roll):.2f}°, Pitch={math.degrees(pitch):.2f}°, Yaw={math.degrees(yaw):.2f}°"
    """
    9. 生成moveL目标位姿（不变）
    """

    def generate_moveL_target(self, robot_pose):
        if robot_pose is None:
            print("⚠️  目标位姿为空，无法生成moveL目标")
            return None

        # 叠加偏移量（避免碰撞）
        target_x = robot_pose["position"][0] + self.moveL_offset[0]
        target_y = robot_pose["position"][1] + self.moveL_offset[1]
        target_z = robot_pose["position"][2] + self.moveL_offset[2]
        target_roll = robot_pose["orientation"][0] + math.radians(self.moveL_offset[3])
        target_pitch = robot_pose["orientation"][1] + math.radians(self.moveL_offset[4])
        target_yaw = robot_pose["orientation"][2] + math.radians(self.moveL_offset[5])

        # 四舍五入（机械臂接收精度有限）
        return [
            round(target_x, 4), round(target_y, 4), round(target_z, 4),
            round(target_roll, 4), round(target_pitch, 4), round(target_yaw, 4)
        ]

    """
    10. 执行moveL运动（不变）
    """

    def execute_moveL(self, robot_pose):
        if robot_pose is None:
            print("❌ 目标位姿为空，无法执行moveL")
            return False

        moveL_target = self.generate_moveL_target(robot_pose)
        if moveL_target is None:
            print("❌ 无法生成moveL目标位姿")
            return False

        # 安全校验
        is_safe, safe_msg = self.check_safe_pose(moveL_target[:3], moveL_target[3:])
        print(f"\n{safe_msg}")
        if not is_safe:
            return False

        # 执行运动
        try:
            print("\n" + "=" * 70)
            print("⚠️  执行moveL运动（安全第一！）")
            print(f"   目标位置: X={moveL_target[0]:.4f}m, Y={moveL_target[1]:.4f}m, Z={moveL_target[2]:.4f}m")
            print(
                f"   目标姿态: Roll={math.degrees(moveL_target[3]):.2f}°, Pitch={math.degrees(moveL_target[4]):.2f}°, Yaw={math.degrees(moveL_target[5]):.2f}°")
            print(f"   运动参数: 速度={self.moveL_speed}m/s, 加速度={self.moveL_acc}m/s²")
            print("=" * 70)

            # 调用UR机械臂moveL接口（需确保SDK接口参数匹配）
            self.robot.moveL(
                target_pose=moveL_target,
                speed=self.moveL_speed,
                acceleration=self.moveL_acc
            )

            print("✅ moveL运动完成")
            return True
        except Exception as e:
            print(f"❌ moveL运动失败: {str(e)}")
            return False

    """
    11. 图像信息绘制（不变，实时显示位姿）
    """

    def draw_pose_info(self, image, robot_pose, fps, moveL_status=None):
        # 1. 模式标识
        cv2.putText(image, "Eye-in-Hand (Dynamic Pose)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # 2. 检测状态
        status_text = "Target Found" if self.found_target else "No Target"
        status_color = (0, 255, 0) if self.found_target else (0, 0, 255)
        cv2.putText(image, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # 3. 中心点坐标
        center_text = f"Center: {self.last_target_center}" if self.last_target_center else "Center: None"
        cv2.putText(image, center_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # 4. 目标深度
        depth_text = f"Depth: {self.last_center_depth:.4f}m" if self.last_center_depth else "Depth: Invalid"
        depth_color = (0, 255, 0) if (self.last_center_depth and self.last_center_depth >= 0.10) else (0, 0, 255)
        cv2.putText(image, depth_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, depth_color, 2)

        # 5. 实时基座位姿（关键：显示动态变化的位姿）
        param_y = 150
        param_spacing = 25
        if robot_pose is not None:
            # 位置
            cv2.putText(image, f"Base X: {robot_pose['position'][0]:.4f}m", (10, param_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Base Y: {robot_pose['position'][1]:.4f}m", (10, param_y + param_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Base Z: {robot_pose['position'][2]:.4f}m", (10, param_y + 2 * param_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # 姿态（弧度转角度）
            roll_deg = math.degrees(robot_pose["orientation"][0])
            pitch_deg = math.degrees(robot_pose["orientation"][1])
            yaw_deg = math.degrees(robot_pose["orientation"][2])
            cv2.putText(image, f"Roll: {roll_deg:.2f}°", (10, param_y + 3 * param_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Pitch: {pitch_deg:.2f}°", (10, param_y + 4 * param_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Yaw: {yaw_deg:.2f}°", (10, param_y + 5 * param_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(image, "Base Pose: No Data", (10, param_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 6. 运动状态
        moveL_text = "moveL: " + (moveL_status if moveL_status else "Ready")
        moveL_color = (0, 0, 255) if moveL_status == "running" else (0, 255, 0)
        cv2.putText(image, moveL_text, (10, param_y + 6 * param_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, moveL_color, 2)

        # 7. FPS和操作提示
        cv2.putText(image, f"FPS: {fps:.1f}", (10, param_y + 7 * param_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Press 'n':Move | 's':Save | 'q':Quit",
                    (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return image