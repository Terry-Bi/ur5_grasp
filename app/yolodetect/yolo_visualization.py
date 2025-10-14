import numpy as np
import cv2

center_outer_radius = 10  # 中心点外圆半径
center_inner_radius = 3  # 中心点内圆半径
center_outer_color = (0, 0, 0)  # 外圆颜色（黑色）
center_inner_color = (0, 255, 255)  # 内圆颜色（黄色）
center_outer_thickness = 2  # 外圆线宽
center_inner_thickness = -1  # 内圆填充（-1表示填充）

def yolo_draw(display_img, mask_np, cX, cY):
# 优先使用语义分割掩码（更精准的中心点计算）
    if mask_np is not None:
        # 可视化：半透明掩码
        mask_3d = np.stack([mask_np] * 3, axis=-1) / 255.0
        display_img = cv2.addWeighted(display_img, 0.7, (mask_3d * (0, 255, 0)).astype(np.uint8), 0.3, 0)

            # 可视化：中心点双圆标记
        cv2.circle(display_img, (cX, cY), center_outer_radius,
                    center_outer_color, center_outer_thickness)
        cv2.circle(display_img, (cX, cY), center_inner_radius,
                    center_inner_color, center_inner_thickness)
        cv2.putText(display_img, f"Center:({cX},{cY})",
                    (cX + 15, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return display_img
    
        # 若无掩码（仅目标检测框），用框中心近似
    elif mask_np is None:
            # 可视化：边界框+中心点
        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.circle(display_img, (cX, cY), center_outer_radius,
                center_outer_color, center_outer_thickness)
        cv2.circle(display_img, (cX, cY), center_inner_radius,
                center_inner_color, center_inner_thickness)
        return display_img
