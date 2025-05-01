import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import time
from collections import OrderedDict
from scipy.optimize import least_squares
import math


# =============================================================================
# 1. Kinematik Fonksiyonları
# =============================================================================
def deg2rad(angle_deg):
    return angle_deg * math.pi / 180.0


def rot_x(angle_rad):
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ])


def rot_y(angle_rad):
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ])

def rot_z(angle_rad):
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ca, -sa, 0],
        [sa, ca, 0],
        [0, 0, 1]
    ])


def hom_trans(R, t):
    """Verilen 3x3 rotasyon matrisi R ve 3x1 öteleme vektörü t için 4x4 homojen dönüşüm."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_hom_T(T, points):
    """
    T: 4x4 homojen dönüşüm matrisi
    points: Nx3 boyutlu nokta dizisi
    """
    N = points.shape[0]
    points_h = np.hstack((points, np.ones((N, 1))))
    transformed = (T @ points_h.T).T
    return transformed[:, :3]


# Pan-Tilt taban dönüşümü (dünya -> pan-tilt)
T_c_w = np.eye(4)

# Pan–Tilt tabanı, Kamera çerçevesinde (x,y,z)=(-0.148, 0.025, 0.0135)
t_p_c = np.array([-0.110, 0.155, 0.0081])
T_p_c = hom_trans(np.eye(3), t_p_c)

# Dünya -> Pan–Tilt dönüşüm
T_p_w = T_p_c @ T_c_w

# =============================================================================
# 2. Pan–Tilt Mekanizması – İleri Kinematik Parametreleri
# =============================================================================
pan_link = 0.46  # Pan kol uzunluğu (z ekseninde)
tilt_link = 0.55  # Tilt kol uzunluğu (z ekseninde)
laser_offset = 0.001  # Lazer ofseti (pan–tilt tabanında, yerel x yönünde)


def forward_kinematics_full(theta_deg, phi_deg):
    """
    Verilen pan (theta) ve tilt (phi) açılarında (ölçülen değerler; nötrde 90° kabul),
    pan–tilt taban çerçevesine göre:
      - p_laser: lazer emitter konumu (3,)
      - d_laser: lazerin çıkış ışını yönü (birim vektör, 3,)
    """
    # Etkin açılar (radyan cinsinde)
    theta_eff = deg2rad(theta_deg - 90)
    phi_eff = deg2rad(phi_deg - 90)

    # Pan dönüşü: -y ekseni etrafında dönüş (ölçülen açının ofsetlenmiş hali)
    R_pan = rot_y(-theta_eff)
    T_pan = hom_trans(R_pan, np.array([0, 0, 0]))

    # Pan kol ötelemesi (z ekseni boyunca)
    T_panL = hom_trans(np.eye(3), np.array([0, 0, pan_link]))

    # Tilt dönüşü: +x ekseni etrafında dönüş
    R_tilt = rot_x(phi_eff)
    T_tilt = hom_trans(R_tilt, np.array([0, 0, 0]))

    # Tilt kol ötelemesi (z ekseni boyunca)
    T_tiltL = hom_trans(np.eye(3), np.array([0, 0, tilt_link]))

    # Lazer ofseti: pan–tilt tabanında, yerel x yönünde
    T_laser = hom_trans(np.eye(3), np.array([laser_offset, 0, 0]))

    # Zincirin homojen dönüşümü (sağdan sola uygulanır)
    T_total = T_laser @ T_tiltL @ T_tilt @ T_panL @ T_pan

    # Lazer emitter konumunu hesaplayalım:
    p0 = np.array([0, 0, 0, 1])
    p_laser = (T_total @ p0)[:3]

    # Lazer ışını yönü: cihazın "nötr"deki yönü (örneğin, yerel [0,0,1]) dönüşümü
    R_total = T_total[:3, :3]
    d_laser = R_total @ np.array([0, 0, 1])
    d_laser = d_laser / np.linalg.norm(d_laser)

    return p_laser, d_laser


def ik_error(angles, target):
    """
    angles: [theta_deg, phi_deg] (ölçülen değerler, nötrde 90°)
    target: Pan–Tilt çerçevesinde hedef nokta (3,)

    Hata vektörü: lazer ışını yönü (d_laser) ile, lazer emitter'ından hedefe giden birim vektör arasındaki fark.
    """
    theta, phi = angles
    p_laser, d_laser = forward_kinematics_full(theta, phi)
    vec = target - p_laser
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        vec_norm = vec
    else:
        vec_norm = vec / norm
    return d_laser - vec_norm


def inverse_kinematics(target, init_guess=[90.0, 90.0]):
    """
    target: Pan–Tilt çerçevesinde hedef nokta (3,)
    init_guess: Başlangıç açısı tahmini (ölçülen değerler; nötrde 90°)
    """
    res = least_squares(ik_error, x0=init_guess, args=(target,))
    theta_sol, phi_sol = res.x
    return theta_sol, phi_sol



# =============================================================================
# 2. Nesne Takip: CentroidTracker
# =============================================================================
class CentroidTracker:
    def _init_(self, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, c):
        self.objects[self.nextObjectID] = c
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, oid):
        del self.objects[oid];
        del self.disappeared[oid]

    def update(self, centroids):
        if not centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared: self.deregister(oid)
            return self.objects
        inC = np.array(centroids)
        if not self.objects:
            for c in inC: self.register(c)
        else:
            oIDs = list(self.objects.keys());
            oC = list(self.objects.values())
            D = np.linalg.norm(np.array(oC)[:, None] - inC, axis=2)
            rows = D.min(axis=1).argsort();
            cols = D.argmin(axis=1)
            usedR = set();
            usedC = set()
            for r, c in zip(rows, cols):
                if r in usedR or c in usedC: continue
                oid = oIDs[r];
                self.objects[oid] = inC[c];
                self.disappeared[oid] = 0
                usedR.add(r);
                usedC.add(c)
            for r in set(range(D.shape[0])) - usedR:
                oid = oIDs[r];
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared: self.deregister(oid)
            for c in set(range(D.shape[1])) - usedC: self.register(inC[c])
        return self.objects


# =============================================================================
# 3. Setup: Seri Port ve RealSense
# =============================================================================
pan_offset, tilt_offset = 0,0
ser_port = 'COM7';
baud = 115200
arduino = serial.Serial(ser_port, baud, timeout=1);
time.sleep(2)


def send_pan_tilt(p, t):
    cmd = f"{p:.2f},{t:.2f}\n";
    arduino.write(cmd.encode())
    print(f"→ Pan={p:.2f}, Tilt={t:.2f}")


pipeline = rs.pipeline();
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

# HSV kırmızı maskesi
lr1, ur1 = np.array([0, 120, 70]), np.array([10, 255, 255])
lr2, ur2 = np.array([170, 120, 70]), np.array([180, 255, 255])
ct = CentroidTracker(maxDisappeared=15)

# Manual kontrol değişkenleri
manual_mode = False
current_pan, current_tilt = 90.0, 90.0
pan_step, tilt_step = 1.0, 1.0

# Home pozisyon
send_pan_tilt(current_pan + pan_offset, current_tilt + tilt_offset)

# Ana Döngü
try:
    while True:
        frames = pipeline.wait_for_frames();
        af = align.process(frames)
        cf = af.get_color_frame();
        df = af.get_depth_frame()
        if not cf or not df: continue
        img = np.asanyarray(cf.get_data());
        depth = np.asanyarray(df.get_data()) * 0.001
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lr1, ur1), cv2.inRange(hsv, lr2, ur2))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroid = None
        if cnts:
            c = max(cnts, key=cv2.contourArea);
            M = cv2.moments(c)
            if M['m00'] > 0:
                centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                cv2.circle(img, centroid, 5, (0, 255, 0), -1)

        # IK her frame hesaplama
        if centroid:
            Z = depth[centroid[1], centroid[0]]
            X = (centroid[0] - cx) / fx * Z;
            Y = (centroid[1] - cy) / fy * Z
            tgt = apply_hom_T(T_p_w, np.array([[X, Y, Z]]))[0]
            pan_calc, tilt_calc = inverse_kinematics(tgt)
            ik_text = f"IK -> Pan:{pan_calc:.2f}, Tilt:{tilt_calc:.2f}"
        else:
            ik_text = "Hedef yok"

        cv2.putText(img, ik_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Tuş yakalama
        key = cv2.waitKeyEx(1)
        if key == ord('q'): break
        if key == ord('f'):
            manual_mode = not manual_mode;
            time.sleep(0.2)

        if manual_mode:
            cv2.putText(img, "MANUEL MOD", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if key == 2424832:
                current_pan -= pan_step  # ←
            elif key == 2555904:
                current_pan += pan_step  # →
            elif key == 2490368:
                current_tilt += tilt_step  # ↑
            elif key == 2621440:
                current_tilt -= tilt_step  # ↓
            send_pan_tilt(current_pan + pan_offset, current_tilt + tilt_offset)
        else:
            cv2.imshow("Mask", mask)
            cv2.imshow("Balon Takip", img)

finally:
    pipeline.stop();
    cv2.destroyAllWindows();
    arduino.close()
