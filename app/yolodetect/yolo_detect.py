import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_visualization import yolo_draw
class yolo_seg():
    
    def __init__(self,
                 yolo_model_path):
        
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

        self.center_outer_radius = 10  # 中心点外圆半径
        self.center_inner_radius = 3  # 中心点内圆半径
        self.center_outer_color = (0, 0, 0)  # 外圆颜色（黑色）
        self.center_inner_color = (0, 255, 255)  # 内圆颜色（黄色）
        self.center_outer_thickness = 2  # 外圆线宽
        self.center_inner_thickness = -1  # 内圆填充（-1表示填充）



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
            mask_np = None
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
                display_img = yolo_visualization.yolo_draw(display_img, mask_np, cX, cY)            
                return display_img, target_center 
            
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
                display_img = yolo_visualization.yolo_draw(display_img, mask_np, cX, cY)  
                
                return display_img, target_center 
            
        except Exception as e:
            print(f"❌ 目标检测出错: {str(e)}")
        return display_img, target_center        