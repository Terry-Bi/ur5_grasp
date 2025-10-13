from real.realsenseD415 import Camera



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