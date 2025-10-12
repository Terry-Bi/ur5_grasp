from ultralytics import YOLO
import numpy as np
import cv2
import os
from pathlib import Path
import time

# 加载分割模型
model = YOLO("seg_arm_body_20251010_zhoutaobi.pt")

# 输出目录配置（用于保存截图）
output_dir = "./realtime_segment_results"
os.makedirs(output_dir, exist_ok=True)

# 定义类别对应的BGR颜色（3通道，匹配图像格式）
class_colors = {
    0: (0, 0, 255),    # class0 → 红色
    2: (255, 0, 0),    # class2 → 蓝色
    3: (255, 255, 255),# class3 → 白色
    "default": (255, 255, 255)  # 默认→白色
}
alpha = 0.4  # 分割色透明度（0=完全透明，1=完全不透明）

# 中心点标记参数
OUTER_RADIUS = 16
INNER_RADIUS = 1
OUTER_COLOR = (0, 0, 0)        # 黑色外圆
INNER_COLOR = (0, 255, 255)    # 黄色内圆
OUTER_THICKNESS = 2
INNER_THICKNESS = -1  # 填充

# 边缘平滑参数
BLUR_KERNEL = (7, 7)
ERODE_ITERATIONS = 1
DILATE_ITERATIONS = 1
KERNEL_SIZE = (3, 3)
kernel = np.ones(KERNEL_SIZE, np.uint8)

def process_frame(frame, model):
    """处理单帧图像，返回分割分割和中心点标记"""
    # 模型预测
    results = model(frame, conf=0.8, verbose=False)
    result = results[0]  # 取第一个结果
    
    base_img = frame.copy()
    masks = result.masks

    if masks is not None:
        for i, (mask, cls) in enumerate(zip(masks.data, result.boxes.cls)):
            class_id = int(cls)
            color = class_colors.get(class_id, class_colors["default"])

            # 掩码处理
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            if mask_np.shape != (frame.shape[0], frame.shape[1]):
                mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            else:
                mask_resized = mask_np

            # 边缘平滑处理
            mask_processed = cv2.erode(mask_resized, kernel, iterations=ERODE_ITERATIONS)
            mask_processed = cv2.dilate(mask_processed, kernel, iterations=DILATE_ITERATIONS)
            mask_processed = cv2.GaussianBlur(mask_processed, BLUR_KERNEL, 0)
            _, mask_processed = cv2.threshold(mask_processed, 127, 255, cv2.THRESH_BINARY)

            # 叠加透明分割色
            seg_region = mask_processed == 255
            base_img[seg_region] = (
                base_img[seg_region] * (1 - alpha) +
                np.array(color) * alpha
            ).astype(np.uint8)

            # 计算并绘制中心点
            contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # 绘制中心点
                    cv2.circle(base_img, (cX, cY), OUTER_RADIUS, OUTER_COLOR, OUTER_THICKNESS)
                    cv2.circle(base_img, (cX, cY), INNER_RADIUS, INNER_COLOR, INNER_THICKNESS)
                    
                    # 绘制坐标文本
                    text = f"({cX}, {cY})"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = cX + 10
                    text_y = cY - 10
                    cv2.rectangle(base_img,
                                 (text_x, text_y - text_size[1] - 2),
                                 (text_x + text_size[0], text_y + 2),
                                 OUTER_COLOR, -1)
                    cv2.putText(base_img, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    
    return base_img

def main():
    # 打开摄像头（0为默认摄像头，可根据需要修改）
    cap = cv2.VideoCapture(1)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率（根据实际设备调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("实时监测开始 - 按 's' 保存当前帧，按 'q' 退出")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            print("无法获取图像帧")
            break
        
        # 处理帧
        processed_frame = process_frame(frame, model)
        
        # 计算并显示帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示操作提示
        cv2.putText(processed_frame, "Press 's' to save, 'q' to quit", 
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示处理后的图像
        cv2.imshow("实时目标分割与中心点检测", processed_frame)
        
        # 键盘交互
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 退出
            break
        elif key == ord('s'):  # 保存当前帧
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(save_path, processed_frame)
            print(f"已保存图像: {save_path}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()
    