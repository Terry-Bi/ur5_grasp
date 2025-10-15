
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

def vec2hom4x4(vec):
    """
    vec: 长度 7  (x,y,z,qx,qy,qz,qw)  四元数
    返回 4×4 齐次矩阵
    """
    t = vec[:3]
    quat = vec[3:7]
    rot = R.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3]  = t
    return T

# ---------- 核心 ----------
class EyeInHandPoseCalculator:
    """
    输入:
        color_image  – H×W×3 uint8
        depth_image  – H×W  uint16 (单位 mm)
    输出:
        dict  {position: (x,y,z) [m], orientation: (rx,ry,rz) [rad]}
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
        self.K = np.array(c['camera_matrix'], dtype=np.float64)  # 3×3
        self.D = np.array(c['dist_coeffs'], dtype=np.float64)   # 1×5
        self.depth_scale = float(c.get('depth_scale', 0.001))   # mm→m

        # 2. 外参 cam->end
        self.T_cam2end = np.loadtxt(cam2end_txt, delimiter=None)  # 4×4
        assert self.T_cam2end.shape == (4, 4)
        self.T_end2cam = T_inv(self.T_cam2end)

        # 3. 检测参数
        self.target_id = target_class_id
        self.conf = conf_thresh
        self.roi = depth_roi


    # ---- 深度滤波 ----
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
        return float(np.median(valid)) * self.depth_scale  # m

    # ---- 2D -> 3D ----
    def pixel_to_camera(self, u: int, v: int, z: float) -> np.ndarray:
        """返回 3×1 相机坐标 (m)"""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    # ---- 主入口 ----
    def call(self,  target_center: tuple, depth: np.ndarray,
             T_end2base: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        T_end2base: 当前时刻 4×4 末端->基座齐次矩阵（由机器人 SDK 给出）
        返回:  {position: (x,y,z), orientation: (rx,ry,rz)}  全部基座系/弧度
        """

        print('T_end2base shape:', T_end2base.shape)
        
        u, v = target_center
        z = self._query_depth(depth, u, v)
        if z is None:
            return None

        # 1. 目标->相机
        P_cam = self.pixel_to_camera(u, v, z)          # 3×1
        T_cam_obj = Rt_to_T(np.eye(3), P_cam)          # 4×4

        # 2. 目标->末端
        T_end_obj = self.T_cam2end @ T_cam_obj

        # 3. 目标->基座
        T_base_obj = T_end2base @ T_end_obj
        t = T_base_obj[:3, 3]                          # 3×1
        R = T_base_obj[:3, :3]
        rvec = cv2.Rodrigues(R)[0].flatten()           # 3×1 轴角

        return dict(position=t, orientation=rvec)

# ---------- 使用示例 ----------
if __name__ == '__main__':
    import cv2
    calc = EyeInHandPoseCalculator('camera.yaml', 'cam2end.txt')
    # 假设机器人给了当前 T_end2base
    T_end2base = np.eye(4)

    cap = cv2.VideoCapture(0)
    while True:
        ret, target_center = cap.read()
        if not ret:
            break
        depth = np.random.randint(200, 1500, (480, 640), dtype=np.uint16)  # 伪深度
        res = calc.call(target_center, depth, T_end2base)
        print(res)
        cv2.imshow('color', color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break