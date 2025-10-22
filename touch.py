#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UR机器人视觉抓取脚本
功能：通过RealSense相机采集图像，鼠标点击目标位置后，机器人自动抓取并放置到指定区域
依赖：UR_Robot主类、RealSense相机驱动、OpenCV、NumPy
"""

# 导入依赖库
import numpy as np
import time
import cv2
# 导入自定义相机类（RealSense D415）和机器人控制主类
from bsp.camera_bsp.realsenseD415 import Camera
from bsp.robot_bsp.UR_Robot import UR_Robot


# --------------------------  用户配置选项  --------------------------
# 机器人IP（需与UR机器人实际IP一致，UR默认网段为192.168.1.x）
tcp_host_ip = '192.168.1.35'
# 工具末端姿态（RPY角，弧度制，需根据实际工具安装方向调整）
tool_orientation = [0.000, 3.141, 0.000]
# 机器人抓取后放置的目标位置
box_position = [-0.200, -0.38, 0.05, 0.000, 3.141, 0.000]
# ---------------------------------------------------------------------


# --------------------------  机器人初始化与复位  --------------------------
# 初始化UR机器人对象（参数与主类__init__匹配）
robot = UR_Robot(robot_ip=tcp_host_ip)

# 机器人初始位置（抓取前的"Home位"，避免碰撞）
grasp_home = [-0.200, -0.38, 0.25, 0.000, 3.141, 0.000]
# 控制机器人直线运动到初始位置
robot.moveL(grasp_home)

# 打开夹爪（position=0为完全打开，speed/force可根据夹爪型号调整）
open_position = 5000
robot.grip(position=open_position, speed=100, force=40)

# 降低机器人关节运动速度和加速度（提升平稳性）
robot.joint_acc = 0.2  # 关节加速度（rad/s²）
robot.joint_spd = 0.2  # 关节速度（rad/s）
# 工具笛卡尔空间运动参数（m/s、m/s²）
robot.tool_spd = 0.1
robot.tool_acc = 0.1


# --------------------------  鼠标点击回调函数  --------------------------
# 全局变量：存储鼠标点击的像素坐标（初始为空）
click_point_pix = ()
# 先获取一帧相机数据，用于后续点击坐标计算
camera_color_img, camera_depth_img = robot.get_camera_data()

def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_point_pix, camera_color_img, camera_depth_img
        click_point_pix = (x, y)
        print(f"已点击像素坐标：({x}, {y})")

        # click_point_pix = (358,199)

        # print(f"已点击像素坐标：" +click_point_pix)

        # 1. 计算点击位置的相机三维坐标（含深度Z）
        click_z = camera_depth_img[y][x] * robot.cam_depth_scale
        if click_z <= 0:  # 过滤无效深度（≤0表示无数据）
            print("点击位置无有效深度数据，跳过运动")
            return
        # 相机内参：主点(cx, cy)、焦距(fx, fy)
        cx, cy = robot.cam_intrinsics[0][2], robot.cam_intrinsics[1][2]
        fx, fy = robot.cam_intrinsics[0][0], robot.cam_intrinsics[1][1]
        click_x = (x - cx) * click_z / fx
        click_y = (y - cy) * click_z / fy
        click_point_cam = np.asarray([click_x, click_y, click_z])
        click_point_cam.shape = (3, 1)  # 转为列向量，适配矩阵运算

        # 2. 相机坐标 → 机器人基系坐标（刚体变换：R*cam + T）
        camera2robot_pose = robot.cam_pose  # 手眼标定的变换矩阵（4×4）
        target_position = np.dot(camera2robot_pose[0:3, 0:3], click_point_cam) + camera2robot_pose[0:3, 3:]
        target_position = target_position[0:3, 0]  # 提取X, Y, Z

        # 3. 构造抓取目标姿态（位置+指定姿态）
        target_pose = [
            target_position[0],
            target_position[1],
            target_position[2],
            tool_orientation[0],
            tool_orientation[1],
            tool_orientation[2]
        ]

        # 4. 安全检查：限制坐标在工作空间内（避免碰撞）
        x_min, x_max = robot.workspace_limits[0]
        y_min, y_max = robot.workspace_limits[1]
        z_min, z_max = robot.workspace_limits[2]
        target_pose[0] = max(x_min, min(target_pose[0], x_max))
        target_pose[1] = max(y_min, min(target_pose[1], y_max))
        target_pose[2] = max(z_min, min(target_pose[2], z_max))
        print(f"机器人基系下目标位置：{target_pose}")

        try:
            # 5. 移动到目标上方“接近位置”（抬高Z轴10cm，避免碰撞）
            approach_pose = target_pose.copy()
            approach_pose[2] += 0.1  # 抬高Z轴0.1米
            robot.moveL(approach_pose, speed=robot.tool_spd, acceleration=robot.tool_acc)
            print("已移动到目标上方接近位置")

            # 6. 低速移动到“抓取位置”（精细操作）
            robot.moveL(target_pose, speed=robot.tool_spd * 0.5, acceleration=robot.tool_acc * 0.5)
            print("已移动到抓取位置")

            # 7. 关闭夹爪（抓取物体）
            close_position = 11000  # 夹爪闭合位置（根据实际夹爪型号调整）
            robot.grip(position=close_position, speed=50, force=60)
            time.sleep(1)  # 等待夹爪动作完成
            print("夹爪已闭合，完成抓取")

            # 8. 回到“接近位置”（带物体上升）
            robot.moveL(approach_pose, speed=robot.tool_spd, acceleration=robot.tool_acc)
            print("已抓取，回到接近位置")

            # 9. 移动到“放置位置”
            robot.moveL(box_position, speed=robot.tool_spd, acceleration=robot.tool_acc)
            print("已移动到放置位置")

            # 10. 打开夹爪（释放物体）
            robot.grip(position=open_position, speed=100, force=40)
            time.sleep(1)  # 等待夹爪动作完成
            # print("夹爪已打开，完成放置")

            # 11. 回到“初始位置”
            robot.moveL(grasp_home, speed=robot.tool_spd, acceleration=robot.tool_acc)
            print("已回到初始位置")

        except Exception as e:
            print(f"运动过程中出错：{e}")
            # 出错时尝试回到初始位置并打开夹爪
            robot.moveL(grasp_home, speed=robot.tool_spd, acceleration=robot.tool_acc)
            robot.grip(position=open_position, speed=100, force=40)


# --------------------------  图像显示与交互循环  --------------------------
# 创建OpenCV窗口
cv2.namedWindow('彩色图像（点击目标位置执行抓取）', cv2.WINDOW_NORMAL)
cv2.namedWindow('深度图像', cv2.WINDOW_NORMAL)

# 绑定鼠标回调函数
cv2.setMouseCallback('彩色图像（点击目标位置执行抓取）', mouseclick_callback)

print("提示：在'彩色图像'窗口中点击目标物体，机器人将自动抓取并放置；按键盘'c'键退出程序")

# 循环获取并显示图像
while True:
    camera_color_img, camera_depth_img = robot.get_camera_data()
    bgr_img = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)  # 转为OpenCV显示格式（BGR）

    # 标记鼠标点击位置
    if len(click_point_pix) != 0:
        cv2.circle(bgr_img, click_point_pix, 7, (0, 0, 255), 2)

    cv2.imshow('彩色图像（点击目标位置执行抓取）', bgr_img)
    cv2.imshow('深度图像', camera_depth_img)

    # 按'c'键退出
    if cv2.waitKey(1) == ord('c'):
        print("已按下'c'键，程序即将退出")
        break


# --------------------------  程序退出清理  --------------------------
cv2.destroyAllWindows()
# 确保机器人回到初始位置
robot.moveL(grasp_home)
# 确保夹爪打开
robot.grip(position=open_position, speed=100, force=40)
print("程序已退出，机器人已复位")