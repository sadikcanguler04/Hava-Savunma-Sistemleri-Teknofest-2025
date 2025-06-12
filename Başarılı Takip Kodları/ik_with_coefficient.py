import os
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import time
from scipy.optimize import least_squares

# =============================================================================
# Ayarlar
# =============================================================================
CALIB_FILE = r'C:\Users\ABDULLAH\Desktop\calibrationn.npz'

# =============================================================================
# Kalibrasyon katsayılarını yükle
# =============================================================================
if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    pan_coeff  = data['pan_coeff']
    tilt_coeff = data['tilt_coeff']
    print(f"Calibration loaded: pan_coeff={pan_coeff}, tilt_coeff={tilt_coeff}")
else:
    raise FileNotFoundError(f"{CALIB_FILE} bulunamadı. Önce kalibrasyon yapıp bu dosyayı oluşturun.")

# =============================================================================
# Yardımcı Fonksiyonlar: Derece-radyan dönüşümü, rotasyon matrisleri, homojen dönüşüm
# =============================================================================
def deg2rad(angle_deg):
    return angle_deg * np.pi / 180.0

def rad2deg(angle_rad):
    return angle_rad * 180.0 / np.pi

def rot_x(angle_rad):
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa, ca]])

def rot_y(angle_rad):
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]])

def rot_z(angle_rad):
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]])

def hom_trans(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]   = t
    return T

def apply_hom_T(T, points):
    pts_h = np.hstack((points, np.ones((points.shape[0], 1))))
    return (T @ pts_h.T).T[:, :3]

# =============================================================================
# Dünya -> Kamera -> Pan–Tilt dönüşümleri
# =============================================================================
T_c_w = np.eye(4)
t_p_c = np.array([-0.148, 0.025, 0.0135])
T_p_c = hom_trans(np.eye(3), t_p_c)
T_p_w = T_p_c @ T_c_w

# =============================================================================
# Mekanik parametreler
# =============================================================================
pan_link     = 0.05
tilt_link    = 0.06
laser_offset = 0.025

def forward_kinematics_full(theta_deg, phi_deg):
    theta_eff = deg2rad(theta_deg - 90)
    phi_eff   = deg2rad(phi_deg   - 90)

    R_pan  = rot_y(-theta_eff)
    T_pan  = hom_trans(R_pan, np.zeros(3))
    T_panL = hom_trans(np.eye(3), [0,0,pan_link])

    R_tilt = rot_x(phi_eff)
    T_tilt = hom_trans(R_tilt, np.zeros(3))
    T_tiltL= hom_trans(np.eye(3), [0,0,tilt_link])

    T_laser= hom_trans(np.eye(3), [laser_offset,0,0])
    T_tot  = T_laser @ T_tiltL @ T_tilt @ T_panL @ T_pan

    p_laser = (T_tot @ np.array([0,0,0,1]))[:3]
    d_laser = T_tot[:3, :3] @ np.array([0,0,1])
    d_laser /= np.linalg.norm(d_laser)

    return p_laser, d_laser

def ik_error(angles, target):
    p_l, d_l = forward_kinematics_full(*angles)
    vec = target - p_l
    n   = np.linalg.norm(vec)
    vec_n = vec/n if n>1e-6 else vec
    return d_l - vec_n

def inverse_kinematics(target, init_guess=[90.0, 90.0]):
    res = least_squares(ik_error, x0=init_guess, args=(target,))
    return res.x[0], res.x[1]

# =============================================================================
# Arduino bağlantısı
# =============================================================================
arduino = serial.Serial('COM7', 115200)
time.sleep(2)

def send_pan_tilt(pan_angle, tilt_angle):
    cmd = f"{pan_angle:.2f},{tilt_angle:.2f}\n"
    arduino.write(cmd.encode())
    print(f"Pan: {pan_angle:.2f}, Tilt: {tilt_angle:.2f} gönderildi.")
    resp = arduino.readline().decode().strip()
    print(f"Arduino yanıtı: {resp}")

# =============================================================================
# RealSense konfigürasyonu
# =============================================================================
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile  = pipeline.start(config)

align_to = rs.stream.color
align    = rs.align(align_to)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

# HSV kırmızı maskeleri
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# =============================================================================
# Ana Döngü (otomatik, kalibre edilmiş)
# =============================================================================
try:
    current_enemy_id = None
    frame_count = 0
    objects = {}
    next_enemy_id = 1
    disappear_threshold = 10

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 400:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                depth = depth_frame.get_distance(int(x), int(y))
                detected.append(((int(x),int(y)), int(r), depth))

        frame_count += 1
        # Tracker update
        for center, radius, depth in detected:
            matched = False
            for obj_id, data in objects.items():
                if dist(center, data['position']) < 130:
                    objects[obj_id].update({'position': center, 'radius':radius, 'depth':depth, 'last_seen':frame_count})
                    matched = True
                    break
            if not matched:
                objects[f"Red-{next_enemy_id}"] = {'position':center,'radius':radius,'depth':depth,'last_seen':frame_count}
                next_enemy_id += 1

        # Remove disappeared
        to_delete = [oid for oid,d in objects.items() if frame_count-d['last_seen']>disappear_threshold]
        for oid in to_delete:
            del objects[oid]

        # Choose closest
        if current_enemy_id not in objects:
            closest, min_d = None, float('inf')
            for oid,data in objects.items():
                d = dist(data['position'], (cx,cy))
                if d<min_d:
                    closest, min_d = oid, d
            current_enemy_id = closest

        if current_enemy_id and current_enemy_id in objects:
            data = objects[current_enemy_id]
            x, y = data['position']
            Z = data['depth']

            # 3D target in Pan–Tilt frame
            X = (x-cx)/fx * Z
            Y = (y-cy)/fy * Z
            pP = apply_hom_T(T_p_w, np.array([[X,Y,Z]]))[0]

            # IK ve kalibrasyon
            pan_raw, tilt_raw = inverse_kinematics(pP, [90.0,90.0])
            pan_cal  = pan_raw  + np.polyval(pan_coeff,  Z)
            tilt_cal = tilt_raw + np.polyval(tilt_coeff, Z)

            # Görsel çıktı
            cv2.circle(frame, (x,y), data['radius'], (0,0,255), 2)
            cv2.putText(frame, f"Pan:{pan_cal:.1f} Tilt:{tilt_cal:.1f}", (x-50,y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
            cv2.putText(frame, f"Dist:{Z:.2f}m", (x-50,y+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

            # Arduino'ya gönder
            send_pan_tilt(pan_cal, tilt_cal)

        cv2.imshow("Calibrated RealSense", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    arduino.close()
