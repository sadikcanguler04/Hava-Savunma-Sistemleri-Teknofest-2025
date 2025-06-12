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
CALIB_FILE = 'calibration.npz'

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
# Dünya -> Kamera -> Pan-Tilt dönüşümleri
# =============================================================================
T_c_w = np.eye(4)
t_p_c = np.array([-0.148, 0.025, 0.0135])
T_p_c = hom_trans(np.eye(3), t_p_c)
T_p_w = T_p_c @ T_c_w

# =============================================================================
# Mekanik parametreler
# =============================================================================
pan_link    = 0.05
tilt_link   = 0.06
laser_offset= 0.025

def forward_kinematics_full(t_deg, p_deg):
    t_eff = deg2rad(t_deg - 90)
    p_eff = deg2rad(p_deg - 90)
    R_pan  = rot_y(-t_eff)
    T_pan  = hom_trans(R_pan, np.zeros(3))
    T_panL = hom_trans(np.eye(3), [0,0,pan_link])
    R_tilt = rot_x(p_eff)
    T_tilt = hom_trans(R_tilt, np.zeros(3))
    T_tiltL= hom_trans(np.eye(3), [0,0,tilt_link])
    T_laser= hom_trans(np.eye(3), [laser_offset,0,0])
    T_tot  = T_laser @ T_tiltL @ T_tilt @ T_panL @ T_pan
    p_laser= (T_tot @ np.array([0,0,0,1]))[:3]
    d_laser= T_tot[:3,:3] @ np.array([0,0,1])
    d_laser/= np.linalg.norm(d_laser)
    return p_laser, d_laser

def ik_error(angs, target):
    p_l, d_l = forward_kinematics_full(*angs)
    vec = target - p_l
    n   = np.linalg.norm(vec)
    vec_n = vec / n if n > 1e-6 else vec
    return d_l - vec_n

def inverse_kinematics(target, guess=[90,90]):
    sol = least_squares(ik_error, x0=guess, args=(target,))
    return sol.x[0], sol.x[1]

# =============================================================================
# Arduino bağlantısı
# =============================================================================
arduino = serial.Serial('COM7', 115200)
time.sleep(2)

def send_pan_tilt(p, t):
    arduino.write(f"{p:.2f},{t:.2f}\n".encode())
    print(f"[ARD] Pan:{p:.2f}, Tilt:{t:.2f}")
    print("[ARD]>", arduino.readline().decode().strip())

# =============================================================================
# RealSense konfigürasyonu
# =============================================================================
pipe   = rs.pipeline()
cfg    = rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16,  30)
profile= pipe.start(cfg)
align  = rs.align(rs.stream.color)
intr   = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

# HSV kırmızı maskeleri
low1, up1 = np.array([0, 120,  70]), np.array([10, 255, 255])
low2, up2 = np.array([170,120, 70]), np.array([180,255,255])

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# =============================================================================
# Kalibrasyon ve Düzeltme
# =============================================================================
calibrating = True
depths      = []
pan_errs    = []
tilt_errs   = []
pan_coeff   = []
tilt_coeff  = []

# Eğer daha önce kaydedilmiş kalibrasyon varsa yükle
if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    pan_coeff  = data['pan_coeff']
    tilt_coeff = data['tilt_coeff']
    calibrating = False
    print(f"Önceki kalibrasyon yüklendi: pan_coeff={pan_coeff}, tilt_coeff={tilt_coeff}")

def fit_calibration(degree=2):
    global pan_coeff, tilt_coeff, calibrating
    pan_coeff  = np.polyfit(depths, pan_errs, degree)
    tilt_coeff = np.polyfit(depths, tilt_errs, degree)
    # Dosyaya kaydet
    np.savez(CALIB_FILE, pan_coeff=pan_coeff, tilt_coeff=tilt_coeff)
    print(f"Pan coeffs kaydedildi: {pan_coeff}")
    print(f"Tilt coeffs kaydedildi: {tilt_coeff}")
    calibrating = False

# =============================================================================
# Ana Döngü
# =============================================================================
pan_man, tilt_man = 90.0, 90.0
auto_mode = False
print("Başlangıç manuel: Pan=90, Tilt=90")

try:
    while True:
        frames = pipe.wait_for_frames()
        aligned= align.process(frames)
        df     = aligned.get_depth_frame()
        cf     = aligned.get_color_frame()
        if not df or not cf:
            continue

        img = np.asanyarray(cf.get_data())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, low1, up1)
        m2  = cv2.inRange(hsv, low2, up2)
        mask= cv2.bitwise_or(m1, m2)
        cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        det = []
        for c in cnts:
            if cv2.contourArea(c) > 400:
                (x, y), r = cv2.minEnclosingCircle(c)
                z = df.get_distance(int(x), int(y))
                det.append(((int(x),int(y)), int(r), z))

        if det:
            ctr, r, Z = min(det, key=lambda t: dist(t[0], (cx, cy)))
            X = (ctr[0]-cx)/fx * Z
            Y = (ctr[1]-cy)/fy * Z
            pP = apply_hom_T(T_p_w, np.array([[X,Y,Z]]))[0]
            pan_auto, tilt_auto = inverse_kinematics(pP, [90,90])

            if not calibrating:
                pan_auto  += np.polyval(pan_coeff, Z)
                tilt_auto += np.polyval(tilt_coeff, Z)

            if auto_mode:
                send_pan_tilt(pan_auto, tilt_auto)

            cv2.circle(img, ctr, r, (0,0,255), 2)
            cv2.putText(img, f"Pan:{pan_auto:.1f} Tilt:{tilt_auto:.1f}",
                        (ctr[0]-50, ctr[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
            cv2.putText(img, f"Dist:{Z:.2f}m",
                        (ctr[0]-50, ctr[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)

        cv2.putText(img, f"Man Pan:{pan_man:.1f} Tilt:{tilt_man:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.imshow("RS", img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('s') and det:
            err_p = pan_man - pan_auto
            err_t = tilt_man - tilt_auto
            depths.append(Z)
            pan_errs.append(err_p)
            tilt_errs.append(err_t)
            print(f"Örnek eklendi: Dist={Z:.3f}, ErrPan={err_p:.3f}, ErrTilt={err_t:.3f}")
        elif key == ord('f'):
            fit_calibration(degree=2)
        elif key == ord('m'):
            auto_mode = not auto_mode
            print(f"Auto mode: {auto_mode}")
        elif key in [100,97,120,119]:
            if key == 100: pan_man  = max(pan_man  - 0.5, 0)
            if key == 97: pan_man  = min(pan_man  + 0.5, 180)
            if key == 120: tilt_man= max(tilt_man - 0.5, 0)
            if key == 119: tilt_man= min(tilt_man + 0.5, 180)
            send_pan_tilt(pan_man, tilt_man)

finally:
    pipe.stop()
    cv2.destroyAllWindows()
    arduino.close()
