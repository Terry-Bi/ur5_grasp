import time
import os
import datetime
import cv2
import numpy as np
import math
import configparser
from ultralytics import YOLO
from bsp.robot_bsp.ur5_robot import UR_Robot
from bsp.camera_bsp.realsenseD415 import Camera
from app.eysinhandposecalculator.EyeInHandPoseCalculator_v2 import EyeInHandPoseCalculator
from app.yolodetect.yolo_detect import yolo_seg


def main():
    
    robot = UR_Robot(
            robot_ip="192.168.1.35",
            gripper_port=False,
            is_use_robot=True,
            is_use_camera=True
        )
    camera = Camera(width=640, height=480)

    YOLO_MODEL_PATH = "seg_arm_body_20251010_zhoutaobi.pt"
    yolo = yolo_seg(yolo_model_path=YOLO_MODEL_PATH)  
    print("完成导入yolo模型")
   #导入相机参数、转换矩阵
    calc = EyeInHandPoseCalculator('camera.yaml', 'cam2end_20251010_zhoutaobi_v2.txt')
    
    while True:
        # 获取图像
        color_image, depth_image = camera.get_data()
        if color_image is None or depth_image is None:
            time.sleep(0.1)
            print("警告: 未获取到相机数据，重试...")
            continue
        
        # 目标检测
        display_img, target_center = yolo.yolo_detect_target(color_image)
        
        #存在问题！！！！#
        imshow = cv2.resize(display_img, (960, 720))
        cv2.imshow("yolo_detect", imshow)
        
        yolo.last_target_center = target_center  # 保存中心点
        print("目标检测完成...")
        print("target_center")
        print(target_center)
        
        #姿态解算
        if yolo.last_target_center is not None:
            T_end2base = np.array(robot.get_actual_tcp_pose())  # 先变 ndarray
            res = calc.call(yolo.last_target_center, depth_image, T_end2base)
        else:
            print("【警告】未检测到目标，跳过位姿计算")
            continue
        print("目标姿态解算完成...")
        print("当前姿态")
        print(res)

if __name__ == "__main__":
    main()