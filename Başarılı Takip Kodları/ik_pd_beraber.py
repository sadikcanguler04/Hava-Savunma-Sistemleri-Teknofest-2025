import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import time
from collections import OrderedDict
from scipy.optimize import least_squares

##############################
# Ortak Ayarlar & Fonksiyonlar
##############################

# Arduino için pan/tilt açılarındaki ofsetler
pan_aci = -12
tilt_aci = -6

# Arduino bağlantısı
arduino = serial.Serial('COM6', 115200)
time.sleep(2)

def send_pan_tilt(pan_angle, tilt_angle):
    cmd = f"{pan_angle:.2f},{tilt_angle:.2f}\n"
    arduino.write(cmd.encode())
    print(f"Pan: {pan_angle:.2f}, Tilt: {tilt_angle:.2f} gönderildi.")
    if arduino.in_waiting:
        response = arduino.readline().decode().strip()
        print(f"Arduino yanıtı: {response}")

# RealSense Kamera Ayarları
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy

# Sistem başlatma zamanı
start_time = time.time()

# PD kontrol parametreleri (PD mod için)
current_pan_angle = 90.0
current_tilt_angle = 90.0
Kp_pan = 0.05
Kd_pan = 0.01
Kp_tilt = 0.05
Kd_tilt = 0.01
last_error_pan = 0.0
last_error_tilt = 0.0
last_time = time.time()
error_threshold = 0.5  # Küçük hata olduğunda güncelleme yapılmaz

# Blend (geçiş) faktörü ayarları
# alpha = 0  => Tamamen PD mod
# alpha = 1  => Tamamen IK mod
alpha = 0.0  
transition_rate = 0.01  # Geçiş hızı (0 ile 1 arasında)

# Lazer tespitinin sürekliliğini ölçmek için sayaç
laser_detected_count = 0
laser_detection_threshold = 5  # Örneğin, 5 ardışık karede lazer tespit edilirse PD mod kesinleşir

##############################
# Lazer Takibi için CentroidTracker (PD mod)
##############################
class CentroidTracker:
    def _init_(self, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, centroids):
        if len(centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.array(centroids)
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(D.shape[0])) - usedRows
            unusedCols = set(range(D.shape[1])) - usedCols
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            for col in unusedCols:
                self.register(inputCentroids[col])
        return self.objects

ct = CentroidTracker(maxDisappeared=15)

##############################
# Blob Detector Ayarları (Lazer tespiti için)
##############################
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255   # Beyaz noktalar
# Yeni parametreler: daha geniş eşik aralıkları, daha esnek alan ve geometrik parametreler
params.minThreshold = 10  
params.maxThreshold = 220  
params.thresholdStep = 10
params.filterByArea = True
params.minArea = 5         # Küçük noktalar için alt sınır
params.maxArea = 10000     # Üst sınır artırıldı
params.filterByCircularity = True
params.minCircularity = 0.7  
params.filterByInertia = True
params.minInertiaRatio = 0.5
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)

##############################
# Balon Tespiti için HSV Ayarları (kırmızı)
##############################
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])

##############################
# İleri/Ters Kinematik & Rotasyon Matrisi Fonksiyonları (IK mod için)
##############################
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
    return np.array([[ca, 0, sa],
                     [0, 1, 0],
                     [-sa, 0, ca]])

def rot_z(angle_rad):
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ca, -sa, 0],
                     [sa, ca, 0],
                     [0, 0, 1]])

def hom_trans(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def apply_hom_T(T, points):
    N = points.shape[0]
    points_h = np.hstack((points, np.ones((N,1))))
    transformed = (T @ points_h.T).T
    return transformed[:, :3]

# Dünya -> Kamera (T_c_w) varsayılan olarak kimlik matrisi
T_c_w = np.eye(4)
# Pan–Tilt taban konumu (kamera çerçevesinde)
t_p_c = np.array([-0.148, 0.025, 0.0135])
T_p_c = hom_trans(np.eye(3), t_p_c)
# Dünya -> Pan–Tilt dönüşümü (T_p_w)
T_p_w = T_p_c @ T_c_w

# İleri kinematik: Verilen pan (theta) ve tilt (phi) açılarında (nötrde 90° kabul)
# lazer emitter konumunu ve lazer ışını yönünü hesaplar.
pan_link = 0.05   # pan kol uzunluğu
tilt_link = 0.06  # tilt kol uzunluğu
laser_offset = 0.025  # lazer ofseti

def forward_kinematics_full(theta_deg, phi_deg):
    theta_eff = deg2rad(theta_deg - 90)
    phi_eff   = deg2rad(phi_deg - 90)
    
    R_pan = rot_y(-theta_eff)
    T_pan = hom_trans(R_pan, np.array([0, 0, 0]))
    T_panL = hom_trans(np.eye(3), np.array([0, 0, pan_link]))
    
    R_tilt = rot_x(phi_eff)
    T_tilt = hom_trans(R_tilt, np.array([0, 0, 0]))
    T_tiltL = hom_trans(np.eye(3), np.array([0, 0, tilt_link]))
    
    T_laser = hom_trans(np.eye(3), np.array([laser_offset, 0, 0]))
    
    T_total = T_laser @ T_tiltL @ T_tilt @ T_panL @ T_pan
    p0 = np.array([0, 0, 0, 1])
    p_laser = (T_total @ p0)[:3]
    R_total = T_total[:3, :3]
    d_laser = R_total @ np.array([0, 0, 1])
    d_laser = d_laser / np.linalg.norm(d_laser)
    
    return p_laser, d_laser

def ik_error(angles, target):
    theta, phi = angles
    p_laser, d_laser = forward_kinematics_full(theta, phi)
    vec = target - p_laser
    norm = np.linalg.norm(vec)
    vec_norm = vec if norm < 1e-6 else vec / norm
    return d_laser - vec_norm

def inverse_kinematics(target, init_guess=[90.0, 90.0]):
    res = least_squares(ik_error, x0=init_guess, args=(target,))
    theta_sol, phi_sol = res.x
    return theta_sol, phi_sol

##############################
# Ana Döngü: Görüntü İşleme, Nesne Tespiti, Mod Seçimi (PD veya IK) ve Blend Edilmiş Kontrol
##############################
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sistem başlatma gecikmesi (3 saniye)
    if time.time() - start_time < 3.0:
        cv2.putText(frame, "Sistem baslatiliyor... (3 sn bekleyin)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Entegre Kontrol", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    ##############################
    # A) Lazer Tespiti (Yeni HSV ve Morfolojik İşlemler ile)
    ##############################
    # Lazer için uygun HSV aralığı (uzak mesafe koşulları için doygunluk/parlaklık değerleri biraz indirildi)
    lower_green = np.array([24, 35, 204])
    upper_green = np.array([44, 123, 255])
    mask1 = cv2.inRange(hsv, lower_green, upper_green)
    lower_green1 = np.array([50, 27, 166])
    upper_green1 = np.array([70, 120, 200])
    mask2 = cv2.inRange(hsv, lower_green1, upper_green1)
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Morfolojik işlemler: dilatasyon, açma ve kapanma
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    keypoints = detector.detect(mask_blurred)
    laser_centroid = None
    if keypoints:
        kp = keypoints[0]
        laser_centroid = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(frame, laser_centroid, int(kp.size/2), (255, 0, 0), 2)
        objects = ct.update([laser_centroid])
        for objectID, centroid in objects.items():
            cv2.putText(frame, f"Laser {objectID}", (centroid[0]-10, centroid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
    else:
        cv2.putText(frame, "Lazer bulunamadi", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    ##############################
    # B) Balon Tespiti (Kırmızı) - 2D konum ve derinlik
    ##############################
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_mask = cv2.GaussianBlur(red_mask, (5,5), 0)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    balloon_centroid = None
    balloon_depth = None
    if red_contours:
        valid_contours = [cnt for cnt in red_contours if cv2.contourArea(cnt) > 600]
        if valid_contours:
            cnt = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_balloon = int(M["m10"] / M["m00"])
                cy_balloon = int(M["m01"] / M["m00"])
                balloon_centroid = (cx_balloon, cy_balloon)
                balloon_depth = depth_frame.get_distance(cx_balloon, cy_balloon)
                cv2.circle(frame, balloon_centroid, 6, (0, 0, 255), -1)
                cv2.putText(frame, "Balon", (cx_balloon - 20, cy_balloon - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Balon bulunamadi", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    ##############################
    # C) Mod Seçimi: PD (lazer varsa) veya IK (lazer tespit edilemezse) ve blend edelim
    ##############################
    # PD mod açılarından (mevcut açı değerleri)
    pd_pan = current_pan_angle
    pd_tilt = current_tilt_angle

    if balloon_centroid is not None:
        if laser_centroid is not None:
            mode = "PD Mod (Lazer referansli)"
            error_x = balloon_centroid[0] - laser_centroid[0]
            error_y = balloon_centroid[1] - laser_centroid[1]
            error_angle_pan = (error_x / fx) * (180.0 / np.pi)
            error_angle_tilt = (error_y / fy) * (180.0 / np.pi)
            current_time = time.time()
            dt = current_time - last_time if current_time - last_time > 0 else 1e-3
            if abs(error_angle_pan) > error_threshold or abs(error_angle_tilt) > error_threshold:
                d_error_pan = (error_angle_pan - last_error_pan) / dt
                d_error_tilt = (error_angle_tilt - last_error_tilt) / dt
                output_pan = Kp_pan * error_angle_pan + Kd_pan * d_error_pan
                output_tilt = Kp_tilt * error_angle_tilt + Kd_tilt * d_error_tilt
                pd_pan -= output_pan
                pd_tilt -= output_tilt
                last_error_pan = error_angle_pan
                last_error_tilt = error_angle_tilt
                last_time = current_time
        else:
            mode = "IK Mod (Lazer yok, ters kinematik)"
            if balloon_depth is not None and balloon_depth > 0:
                x, y = balloon_centroid
                X = (x - cx) / fx * balloon_depth
                Y = (y - cy) / fy * balloon_depth
                Z = balloon_depth
                pW_target = np.array([[X, Y, Z]])
                pP_target = apply_hom_T(T_p_w, pW_target)
                target_pt = pP_target[0]
                ik_pan, ik_tilt = inverse_kinematics(target_pt, init_guess=[90.0, 90.0])
            else:
                mode = "IK Mod (Balon derinligi yok)"
                ik_pan = pd_pan
                ik_tilt = pd_tilt
    else:
        mode = "Balon tespit edilemedi"
        ik_pan = pd_pan
        ik_tilt = pd_tilt

    # Lazer tespiti durumuna göre, sürekli tespit varsa sayaç arttırılır.
    if laser_centroid is not None:
        laser_detected_count += 1
    else:
        laser_detected_count = 0

    # Eğer lazer belirli sayıda ardışık karede tespit edildiyse, alpha hemen 0 yapılır (PD mod kesinleşir)
    if laser_detected_count >= laser_detection_threshold:
        alpha = 0.0
    else:
        alpha = min(1.0, alpha + transition_rate)

    smooth_pan = (1 - alpha) * pd_pan + alpha * ik_pan
    smooth_tilt = (1 - alpha) * pd_tilt + alpha * ik_tilt

    cv2.putText(frame, f"Mod: {mode}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Alpha: {alpha:.2f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Pan: {smooth_pan:.2f}, Tilt: {smooth_tilt:.2f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    current_pan_angle = smooth_pan
    current_tilt_angle = smooth_tilt
    send_pan_tilt(smooth_pan + pan_aci, smooth_tilt + tilt_aci)

    ##############################
    # D) Görüntülerin Gösterilmesi
    ##############################
    cv2.imshow("Entegre Kontrol", frame)
    cv2.imshow("Lazer Maskesi", mask_blurred)
    cv2.imshow("Balon Maskesi", red_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
arduino.close()
