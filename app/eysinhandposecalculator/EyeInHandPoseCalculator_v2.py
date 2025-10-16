from __future__ import annotations
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from scipy.spatial.transform import Rotation as R

# ---------- 工具 ----------
def rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues 向量 -> 3×3 旋转矩阵"""
    return cv2.Rodrigues(rvec)[0]

def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R,t -> 4×4 齐次矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    """齐次矩阵求逆"""
    R, t = T[:3, :3], T[:3, 3]
    Tinv = np.eye(4, dtype=T.dtype)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def pose_to_transform(pose):
    x, y, z, rx, ry, rz = pose
    # 创建旋转对象（假设欧拉角顺序为 'xyz'）
    rotation = R.from_euler('xyz', [rx, ry, rz])
    rot_mat = rotation.as_matrix()
    # 构造变换矩阵（平移部分单位为mm）
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = [x, y, z]
    return transform

# ---------- 核心 ----------
class EyeInHandPoseCalculator:
    """
    输入:
        color_image  – H×W×3 uint8
        depth_image  – H×W  uint16 (单位 mm)
    输出:
        dict  {position: (x,y,z) [mm], orientation: (rx,ry,rz) [rad]}
    """

    def __init__(self,
                 yaml_path: str | Path,
                 cam2end_txt: str | Path,
                 target_class_id: int = 1,
                 conf_thresh: float = 0.3,
                 depth_roi: int = 8):
        # 1. 相机内参
        with open(yaml_path, 'r', encoding='utf-8') as f:
            c = yaml.safe_load(f)
        self.K = np.array(c['camera_matrix'], dtype=np.float64)  # 3×3（像素单位）
        self.D = np.array(c['dist_coeffs'], dtype=np.float64)   # 1×5
        # 移除深度缩放（直接使用mm单位）

        # 2. 外参 cam->end（确保平移部分单位为mm）
        self.T_cam2end = np.loadtxt(cam2end_txt, delimiter=None)  # 4×4
        assert self.T_cam2end.shape == (4, 4)
        self.T_end2cam = T_inv(self.T_cam2end)

        # 3. 检测参数
        self.target_id = target_class_id
        self.conf = conf_thresh
        self.roi = depth_roi

    # ---- 深度滤波（保留mm单位）----
    def _query_depth(self, depth: np.ndarray, u: int, v: int) -> Optional[float]:
        roi = self.roi
        h, w = depth.shape
        x1 = max(0, u - roi//2)
        y1 = max(0, v - roi//2)
        x2 = min(w, x1 + roi)
        y2 = min(h, y1 + roi)
        patch = depth[y1:y2, x1:x2]
        valid = patch[patch > 0]
        if valid.size < 3:
            return None
        return float(np.median(valid))  # 直接返回mm单位，不做缩放

    # ---- 2D -> 3D（输出mm单位）----
    def pixel_to_camera(self, u: int, v: int, z: float) -> np.ndarray:
        """返回 3×1 相机坐标 (mm)"""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)  # z为mm，x、y同步为mm

    # ---- 主入口 ----
    def call(self,  target_center: tuple, depth: np.ndarray,
             T_end2base: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        T_end2base: 当前时刻 4×4 末端->基座齐次矩阵（平移单位需为mm）
        返回:  {position: (x,y,z) [mm], orientation: (rx,ry,rz) [rad]}
        """
        u, v = target_center
        u, v = int(round(u)), int(round(v))  # 确保像素坐标为整数
        z = self._query_depth(depth, u, v)
        if z is None:
            return None

        # 若T_end2base由位姿转换而来，确保输入的位姿平移单位为mm
        T_end2base = pose_to_transform(T_end2base)

        # 1. 目标->相机（mm单位）
        P_cam = self.pixel_to_camera(u, v, z)
        T_cam_obj = Rt_to_T(np.eye(3), P_cam)

        # 2. 目标->末端（mm单位）
        T_end_obj = self.T_cam2end @ T_cam_obj

        # 3. 目标->基座（mm单位）
        T_base_obj = T_end2base @ T_end_obj
        t = T_base_obj[:3, 3]  # 位置单位为mm
        R_mat = T_base_obj[:3, :3]
        rvec = cv2.Rodrigues(R_mat)[0].flatten()  # 姿态仍为弧度

        return dict(position=t, orientation=rvec)

# ---------- 使用示例 ----------
if __name__ == '__main__':
    import cv2
    # 确保cam2end.txt的平移单位为mm
    calc = EyeInHandPoseCalculator('camera.yaml', 'cam2end.txt')
    # 假设T_end2base的平移单位为mm（此处用单位矩阵模拟）
    T_end2base = np.eye(4)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取图像")
            break

        # 模拟移动的目标中心点（像素坐标）
        h, w = frame.shape[:2]
        center_x = int(w/2 + 100 * np.sin(cv2.getTickCount()/1000000))
        center_y = int(h/2 + 100 * np.cos(cv2.getTickCount()/1000000))
        target_center = (center_x, center_y)
        cv2.circle(frame, target_center, 5, (0, 255, 0), -1)

        # 伪深度图像（mm单位）
        depth = np.random.randint(500, 1500, (h, w), dtype=np.uint16)
        res = calc.call(target_center, depth, T_end2base)

        # 打印结果（位置单位为mm）
        if res:
            print(f"位置(mm): {np.round(res['position'], 2)}, 姿态(rad): {np.round(res['orientation'], 3)}")

        cv2.imshow('color', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()