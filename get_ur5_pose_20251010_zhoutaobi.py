import time
import os
import datetime
import cv2
import numpy as np
import math
import configparser  # 用于解析普通INI文件
import keyboard
from ur5_robot import UR_Robot
from real.realsenseD415 import Camera


class RobotPoseCalculator:
    def __init__(self, cam2end_path, camera_params_path, robot, camera):
        """初始化位姿计算器（适配眼在手上模式）"""
        # 设备对象
        self.robot = robot
        self.camera = camera

        # 加载手眼标定结果（相机→末端），并计算末端→相机的逆矩阵（眼在手上核心修正）
        self.cam2end = self.load_cam2end(cam2end_path)
        self.end2cam = self.calculate_transform_inverse(self.cam2end)  # 新增：末端到相机的变换矩阵

        # 加载相机内参和畸变参数（适配普通INI格式）
        self.camera_matrix, self.dist_coeffs = self.load_camera_params(camera_params_path)

        # 目标3D模型点（需与实际目标尺寸一致，单位：米）
        self.target_3d_points = self.get_target_3d_model()

    def load_cam2end(self, file_path):
        """加载相机到末端的变换矩阵，并处理单位转换（毫米→米，标定文件常见格式）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"cam2end矩阵文件不存在: {file_path}")

        cam2end = []
        with open(file_path, 'r') as f:
            for line in f:
                # 跳过空行或注释行（避免读取无效数据）
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                row = list(map(float, line.split()))
                if len(row) == 4:  # 确保是4x4变换矩阵的一行
                    # 关键：标定文件常以毫米为单位，转换为米（若已为米，注释此行）
                    row = [x / 1000.0 for x in row]
                    cam2end.append(row)

        # 验证矩阵维度（必须是4x4）
        if len(cam2end) != 4:
            raise ValueError(f"cam2end矩阵需为4行，当前读取到{len(cam2end)}行，请检查标定文件格式")

        cam2end_mat = np.array(cam2end, dtype=np.float32)
        print("cam2end矩阵加载完成（已转换为米单位）:")
        print(cam2end_mat)
        return cam2end_mat

    def calculate_transform_inverse(self, transform_mat):
        """计算4x4变换矩阵的逆矩阵（眼在手上模式必需）
        原理：变换矩阵逆 = [R^T, -R^T*t; 0, 1]，其中R是旋转矩阵，t是平移向量
        """
        # 提取旋转矩阵和平移向量
        R = transform_mat[:3, :3]
        t = transform_mat[:3, 3].reshape(3, 1)  # 转为列向量

        # 旋转矩阵的逆 = 旋转矩阵的转置（正交矩阵特性）
        R_inv = R.T
        # 平移向量的逆 = -R^T * t
        t_inv = -np.dot(R_inv, t)

        # 构建逆变换矩阵
        inv_transform = np.eye(4, dtype=np.float32)
        inv_transform[:3, :3] = R_inv
        inv_transform[:3, 3] = t_inv.flatten()  # 转回行向量

        print("末端到相机的逆变换矩阵（end2cam）:")
        print(inv_transform)
        return inv_transform

    def load_camera_params(self, file_path):
        """加载普通INI格式的相机参数（保持原逻辑，确保兼容性）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"相机参数文件不存在: {file_path}")

        config = configparser.ConfigParser()
        config.read(file_path)

        # 读取内参
        try:
            fx = float(config.get('camera_parameters', 'fx'))
            fy = float(config.get('camera_parameters', 'fy'))
            cx = float(config.get('camera_parameters', 'cx'))
            cy = float(config.get('camera_parameters', 'cy'))

            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        except Exception as e:
            raise ValueError(f"解析相机内参失败: {str(e)}")

        # 读取畸变参数，兼容k3缺失的情况
        try:
            k1 = float(config.get('distortion_parameters', 'k1', fallback=0.0))
            k2 = float(config.get('distortion_parameters', 'k2', fallback=0.0))
            p1 = float(config.get('distortion_parameters', 'p1', fallback=0.0))
            p2 = float(config.get('distortion_parameters', 'p2', fallback=0.0))
            k3 = float(config.get('distortion_parameters', 'k3', fallback=0.0))  # 默认为0

            dist_coeffs = np.array([[k1, k2, p1, p2, k3]], dtype=np.float32)
        except Exception as e:
            raise ValueError(f"解析畸变参数失败: {str(e)}")

        print("相机参数加载成功:")
        print(f"内参矩阵:\n{camera_matrix}")
        print(f"畸变参数: k1={k1}, k2={k2}, p1={p1}, p2={p2}, k3={k3}")

        return camera_matrix, dist_coeffs

    def get_target_3d_model(self):
        """定义目标物体的3D模型点（需根据实际目标修改尺寸！）"""
        # 示例：10cmx10cm的正方形靶标（单位：米）
        # 若您的目标是其他尺寸（如5cm、20cm），请修改target_size的值
        target_size = 0.1  # 单位：米
        return np.array([
            [-target_size / 2, -target_size / 2, 0],  # 左上角（靶标坐标系）
            [target_size / 2, -target_size / 2, 0],  # 右上角
            [target_size / 2, target_size / 2, 0],  # 右下角
            [-target_size / 2, target_size / 2, 0]  # 左下角
        ], dtype=np.float32)

    def detect_target(self, image):
        """检测图像中的目标（保持原逻辑，若目标不是四边形需修改）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 优化边缘检测（降低噪声影响）
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)  # sigma从0改为1，增强去噪
        edges = cv2.Canny(blurred, 40, 120)  # 调整阈值，适配更多场景

        # 寻找轮廓
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # 寻找四边形轮廓（假设目标是四边形，如棋盘格、方形工件）
        for contour in contours:
            # 过滤过小的轮廓（避免检测到噪声）
            if cv2.contourArea(contour) < 500:  # 面积小于500像素的轮廓忽略
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:  # 找到四边形
                points = approx.reshape(4, 2).astype(np.float32)
                return self.order_points(points)

        return None  # 未检测到目标

    def order_points(self, pts):
        """对四边形的四个点进行排序（保持原逻辑，确保与3D模型点顺序对应）"""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上角（对应3D模型第0个点）
        rect[2] = pts[np.argmax(s)]  # 右下角（对应3D模型第2个点）

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上角（对应3D模型第1个点）
        rect[3] = pts[np.argmax(diff)]  # 左下角（对应3D模型第3个点）

        return rect

    def get_obj2cam_pose(self, image):
        """计算目标相对于相机的位姿（保持原逻辑，确保参数正确）"""
        image_points = self.detect_target(image)
        if image_points is None:
            return None

        # 使用solvePnP计算位姿（采用ITERATIVE迭代法，精度更高）
        _, rvec, tvec = cv2.solvePnP(
            self.target_3d_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE  # 新增：指定迭代法，提升位姿计算精度
        )

        # 转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 构建4x4变换矩阵（目标→相机）
        obj2cam = np.eye(4, dtype=np.float32)
        obj2cam[:3, :3] = R
        obj2cam[:3, 3] = tvec.flatten()

        return obj2cam

    def euler_to_rotation_matrix(self, rpy):
        """将欧拉角转换为旋转矩阵（保持原逻辑，确保与UR机械臂坐标系一致）"""
        roll, pitch, yaw = rpy

        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])

        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        # UR机械臂默认旋转顺序为Z→Y→X，与该组合顺序一致
        return np.dot(R_z, np.dot(R_y, R_x))

    def get_end2base_pose(self):
        """从机器人获取末端相对于基座的位姿（保持原逻辑，确保数据有效）"""
        try:
            tcp_pose = self.robot.get_actual_tcp_pose()
            # 验证TCP位姿数据有效性（UR机械臂工作范围通常在X:0.1~1.0, Y:-0.5~0.5, Z:0.1~1.0）
            if not (-2 < tcp_pose[0] < 2 and -2 < tcp_pose[1] < 2 and -2 < tcp_pose[2] < 2):
                print(f"警告：TCP位姿异常 {tcp_pose}，可能是机器人连接问题")
                return None

            x, y, z = tcp_pose[0], tcp_pose[1], tcp_pose[2]
            rx, ry, rz = tcp_pose[3], tcp_pose[4], tcp_pose[5]  # UR返回的是弧度

            # 构建末端→基座的变换矩阵
            end2base = np.eye(4, dtype=np.float32)
            end2base[:3, :3] = self.euler_to_rotation_matrix([rx, ry, rz])
            end2base[:3, 3] = [x, y, z]

            return end2base
        except Exception as e:
            print(f"获取机器人末端位姿失败: {e}")
            return None

    def calculate_robot_pose(self, image):
        """计算目标相对于机器人基座的位姿（核心修正：眼在手上变换链）"""
        # 1. 计算目标→相机的位姿
        obj2cam = self.get_obj2cam_pose(image)
        if obj2cam is None:
            print("位姿计算失败：未检测到目标")
            return None

        # 2. 计算末端→基座的位姿
        end2base = self.get_end2base_pose()
        if end2base is None:
            print("位姿计算失败：无法获取末端位姿")
            return None

        # 3. 眼在手上模式核心变换链（修正前：end2base @ cam2end @ obj2cam）
        # 修正后：目标→相机→末端→基座 → 对应矩阵乘法顺序：end2base @ end2cam @ obj2cam
        obj2base = end2base @ self.end2cam @ obj2cam

        # 提取位置和姿态（保持原逻辑）
        position = obj2base[:3, 3]
        rotation_matrix = obj2base[:3, :3]

        # 旋转矩阵转欧拉角（保持原逻辑，转换为角度便于查看）
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = 0

        orientation = [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

        # 验证位置是否在合理范围（UR机械臂工作范围通常为±1.5米内）
        if any(abs(coord) > 2 for coord in position):
            print(f"警告：目标位置异常 {position}，可能是单位或变换链错误")

        return {
            'position': position,  # 单位：米
            'orientation': orientation,  # 单位：度
            'tcp_pose': self.robot.get_actual_tcp_pose()
        }


def main():
    # 配置文件路径（请确认路径正确）
    cam2end_path = "cam2end_20251010_zhoutaobi.txt"
    camera_params_path = "camera_20251010_zhoutaobi.ini"

    # 初始化UR机械臂（保持原逻辑）
    robot = UR_Robot(
        robot_ip="192.168.1.35",
        gripper_port=False,
        is_use_robot=True,
        is_use_camera=True
    )

    # 初始化Realsense相机（保持原逻辑）
    try:
        camera = Camera(width=640, height=480, fps=30)
        print("相机初始化成功")
    except Exception as e:
        print(f"相机初始化失败: {e}")
        return

    try:
        # 初始化位姿计算器（新增逆矩阵计算，无需额外参数）
        calculator = RobotPoseCalculator(
            cam2end_path,
            camera_params_path,
            robot,
            camera
        )
        print("位姿计算器初始化成功（眼在手上模式）")

        # 创建保存文件夹（保持原逻辑）
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"pose_results_{current_time}"
        os.makedirs(folder_name, exist_ok=True)
        print(f"数据将保存到: {folder_name}")

        # 操作提示（保持原逻辑）
        print("\n=== 操作说明 ===")
        print("1. 实时显示框展示相机画面")
        print("2. 按下 'y' 键：拍摄照片并解算位姿")
        print("3. 按下 'q' 键：退出程序")
        print("===============\n")

        count = 0
        print("实时显示已启动，等待操作...")

        while True:
            # 获取相机画面（保持原逻辑）
            color_image, depth_image = camera.get_data()
            display_image = color_image.copy()
            blackwhite_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 绘制提示文字（保持原逻辑）
            cv2.putText(
                display_image,
                "Press 'y' to calculate | 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display_image,
                f"Count: {count}",
                (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # 检测目标并可视化（保持原逻辑）
            target_points = calculator.detect_target(blackwhite_image)
            if target_points is not None:
                for (x, y) in target_points:
                    cv2.circle(display_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                for i in range(4):
                    cv2.line(display_image,
                             (int(target_points[i][0]), int(target_points[i][1])),
                             (int(target_points[(i + 1) % 4][0]), int(target_points[(i + 1) % 4][1])),
                             (255, 0, 0), 2)
                cv2.putText(display_image, "Target detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_image, "No target", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示画面（保持原逻辑）
            cv2.imshow("Camera View", display_image)

            # 键盘交互（保持原逻辑）
            if keyboard.is_pressed('y'):
                count += 1
                # 保存图像（保持原逻辑）
                color_filename = f"{folder_name}/color_{count:04d}.jpg"
                cv2.imwrite(color_filename, color_image)
                bw_filename = f"{folder_name}/bw_{count:04d}.jpg"
                cv2.imwrite(bw_filename, blackwhite_image)
                depth_filename = f"{folder_name}/depth_{count:04d}.png"
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(depth_filename, depth_colormap)

                # 计算位姿（调用修正后的方法）
                print(f"\n[计算中] 第{count}次...")
                robot_pose = calculator.calculate_robot_pose(blackwhite_image)

                if robot_pose is not None:
                    # 保存结果（保持原逻辑）
                    pose_file = f"{folder_name}/poses.txt"
                    with open(pose_file, "a") as f:
                        f.write(f"=== 第{count}次 ===\n")
                        f.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(
                            f"位置 (米): X={robot_pose['position'][0]:.4f}, Y={robot_pose['position'][1]:.4f}, Z={robot_pose['position'][2]:.4f}\n")
                        f.write(
                            f"姿态 (度): Roll={robot_pose['orientation'][0]:.2f}, Pitch={robot_pose['orientation'][1]:.2f}, Yaw={robot_pose['orientation'][2]:.2f}\n")
                        f.write(f"TCP位姿: {robot_pose['tcp_pose']}\n\n")

                    # 打印结果（保持原逻辑）
                    print(f"[成功] 第{count}次")
                    print(
                        f"位置: X={robot_pose['position'][0]:.4f}, Y={robot_pose['position'][1]:.4f}, Z={robot_pose['position'][2]:.4f}")
                    print(
                        f"姿态: Roll={robot_pose['orientation'][0]:.2f}, Pitch={robot_pose['orientation'][1]:.2f}, Yaw={robot_pose['orientation'][2]:.2f}")
                else:
                    print(f"[失败] 第{count}次，未检测到目标或末端位姿异常")

                # 防止长按连续触发（保持原逻辑）
                time.sleep(0.3)

            elif keyboard.is_pressed('q'):
                print("\n退出程序...")
                break

            # 窗口刷新（保持原逻辑）
            cv2.waitKey(1)

    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        # 资源清理（保持原逻辑）
        cv2.destroyAllWindows()
        robot.rtde_c.stopScript()
        print("资源已释放，程序退出")


if __name__ == "__main__":
    main()