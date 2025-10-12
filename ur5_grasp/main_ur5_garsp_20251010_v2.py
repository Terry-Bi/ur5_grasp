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
from EyeInHandPoseCalculator import EyeInHandPoseCalculator

def main():
    # 配置参数（新增cam2end TXT文件路径）
    CAMERA_PARAMS_PATH = "camera_20251010_zhoutaobi_v2.ini"
    CAM2END_TXT_PATH = "cam2end_20251010_zhoutaobi_v2.txt" 
    YOLO_MODEL_PATH = "seg_arm_body_20251010_zhoutaobi.pt"
    ROBOT_IP = "192.168.1.35"
    SAVE_DIR = "./eye_in_hand_simple_results"
    TARGET_CLASS_ID = 1

    # 检查必要文件是否存在
    required_files = [CAMERA_PARAMS_PATH, CAM2END_TXT_PATH, YOLO_MODEL_PATH]
    for file in required_files:
        if not os.path.exists(file):
            print(f"错误：必要文件不存在 - {file}")
            return

    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        # 初始化机械臂和相机
        robot = UR_Robot(
            robot_ip=ROBOT_IP,
            gripper_port=False,
            is_use_robot=True,
            is_use_camera=True
        )
        camera = Camera(width=640, height=480, fps=30)

        # 初始化计算器（新增传入cam2end_txt_path参数）
        calculator = EyeInHandPoseCalculator(
            camera_params_path=CAMERA_PARAMS_PATH,
            cam2end_txt_path=CAM2END_TXT_PATH,  # 新增：传入cam2end TXT文件路径
            yolo_model_path=YOLO_MODEL_PATH,
            robot=robot,
            camera=camera
        )
        calculator.target_class_id = TARGET_CLASS_ID

        print("\n" + "=" * 70)
        print("✅ 眼在手上系统初始化成功（含cam2end矩阵）")
        print(f"🎯 目标类别ID: {TARGET_CLASS_ID}")
        print(f"📂 加载的cam2end矩阵文件: {CAM2END_TXT_PATH}")
        print("⚠️  安全提示：运动前请确认目标与末端距离>12cm，避免碰撞！")
        print("=" * 70)
        print("操作说明：")
        print("  - 按 'n' 键：执行moveL运动（速度已降低）")
        print("  - 按 's' 键：保存当前帧数据")
        print("  - 按 'q' 键：退出程序")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"初始化失败：{e}")
        return

    frame_count = 0
    start_time = time.time()
    save_count = 0
    moveL_status = None

    while True:
        # 获取相机数据（彩色图+深度图）
        color_image, depth_image = camera.get_data()
        if color_image is None or depth_image is None:
            time.sleep(0.1)
            print("警告: 未获取到相机数据，重试...")
            continue

        # 检测目标中心点
        display_img, target_center = calculator.yolo_detect_target(color_image)
        calculator.last_target_center = target_center  # 保存中心点用于可视化
        print("target_center")
        print(target_center)
        # 计算基座坐标系位姿
        robot_pose = None
        if target_center is not None:
            robot_pose = calculator.calculate_robot_pose(target_center, depth_image)
        else:
            print("未检测到有效目标中心点，无法计算位姿")

        # 更新FPS并绘制信息
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        # 修改draw_pose_info调用参数，与类定义保持一致
        display_img = calculator.draw_pose_info(display_img, robot_pose, fps, moveL_status)
        cv2.imshow("Eye-in-Hand (With cam2end) Detection & Control", display_img)

        # 键盘事件处理
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

            moveL_status = "running"
            cv2.imshow("Eye-in-Hand (With cam2end) Detection & Control", display_img)
            cv2.waitKey(1)

            # 调整目标位姿（抬高Z轴，进一步降低碰撞风险）
            modified_robot_pose = {
                "position": robot_pose["position"].copy(),
                "orientation": robot_pose["orientation"].copy(),
                "tcp_pose": robot_pose["tcp_pose"]
            }
            modified_robot_pose["position"][2] = 0.6  # Z轴最低0.6m
            # 姿态已在类中使用弧度，无需再转换
            modified_robot_pose["orientation"] = [0, math.pi, 0]  # 固定姿态（弧度）

            # 安全校验（直接使用弧度姿态）
            is_safe, safe_msg = calculator.check_safe_pose(
                modified_position=modified_robot_pose["position"],
                modified_orientation=modified_robot_pose["orientation"]
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
    