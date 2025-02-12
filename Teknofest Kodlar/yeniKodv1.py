import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import time
from scipy.optimize import least_squares


# =============================================================================
# Yardımcı Fonksiyonlar: Derece-radyan dönüşümü, rotasyon matrisleri, homojen dönüşüm
# =============================================================================

def deg2rad(angle_deg):
    return angle_deg * np.pi / 180.0


def rad2deg(angle_rad):
    return angle_rad * 180.0 / np.pi


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


# =============================================================================
# 1. Dünya -> Kamera ve Kamera -> Pan–Tilt Dönüşümü
# =============================================================================
# Kamera, Dünya çerçevesinde (0,0,0) ve açılar 0 kabul ediliyor.
T_c_w = np.eye(4)

# Pan–Tilt tabanı, Kamera çerçevesinde (x,y,z)=(-0.148, 0.025, 0.0135)
t_p_c = np.array([-0.148, 0.025, 0.0135])
T_p_c = hom_trans(np.eye(3), t_p_c)

# Dünya -> Pan–Tilt dönüşüm
T_p_w = T_p_c @ T_c_w

# =============================================================================
# 2. Pan–Tilt Mekanizması – İleri Kinematik Parametreleri
# =============================================================================
pan_link = 0.05  # Pan kol uzunluğu (z ekseninde)
tilt_link = 0.06  # Tilt kol uzunluğu (z ekseninde)
laser_offset = 0.025  # Lazer ofseti (pan–tilt tabanında, yerel x yönünde)


# Not: Ölçülen açılarda "nötr" durum 90° olduğundan, ileri kinematikte:
#    theta_eff = theta_measured - 90,  phi_eff = phi_measured - 90

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


# =============================================================================
# 3. Ters Kinematik: Lazer ışını yönü ile "hedefe giden" yönün hizalanmasını sağla
# =============================================================================

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
# Arduino Bağlantısı ve Pan-Tilt Komutları
# =============================================================================

arduino = serial.Serial('COM6', 115200)
time.sleep(2)  # Arduino'nun hazır olmasını bekle


def send_pan_tilt(pan_angle, tilt_angle):
    """Pan ve tilt açılarını Arduino'ya gönderir."""
    cmd = f"{pan_angle:.2f},{tilt_angle:.2f}\n"
    arduino.write(cmd.encode())
    print(f"Pan: {pan_angle:.2f}, Tilt: {tilt_angle:.2f} gönderildi.")
    response = arduino.readline().decode().strip()
    print(f"Arduino yanıtı: {response}")


# =============================================================================
# RealSense Kamera Yapılandırması
# =============================================================================

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

# =============================================================================
# Nesne Takibi için Parametreler
# =============================================================================

objects = {}
next_enemy_id = 1
disappear_threshold = 10
frame_count = 0

# HSV için kırmızı eşik değerleri
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_area(radius):
    return math.pi * (radius ** 2)


# =============================================================================
# Ana Döngü
# =============================================================================

try:
    current_enemy_id = None
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # HSV renk dönüşümü
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Kırmızı maske oluştur
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Kırmızı nesneleri bulma
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected_centers = []

        for cnt in red_contours:
            if cv2.contourArea(cnt) > 1100:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                area = calculate_area(radius)
                depth = depth_frame.get_distance(center[0], center[1])
                detected_centers.append((center, radius, area, depth))

        # Nesne takip bilgilerini güncelle
        frame_count += 1
        for center, radius, area, depth in detected_centers:
            matched = False
            for obj_id, data in objects.items():
                if calculate_distance(center, data['position']) < 130:
                    objects[obj_id].update({
                        'position': center,
                        'radius': radius,
                        'area': area,
                        'depth': depth,
                        'last_seen': frame_count
                    })
                    matched = True
                    break
            if not matched:
                objects[f"Red-{next_enemy_id}"] = {
                    'position': center,
                    'radius': radius,
                    'area': area,
                    'depth': depth,
                    'last_seen': frame_count
                }
                next_enemy_id += 1

        # Kaybolan nesneleri kaldır
        to_remove = [obj_id for obj_id, data in objects.items() if
                     frame_count - data['last_seen'] > disappear_threshold]
        for obj_id in to_remove:
            del objects[obj_id]

        # En yakın enemy'i seç
        if current_enemy_id not in objects:
            closest_enemy = None
            min_distance = float('inf')
            for obj_id, data in objects.items():
                distance = calculate_distance(data['position'], (cx, cy))
                if distance < min_distance:
                    closest_enemy = obj_id
                    min_distance = distance
            current_enemy_id = closest_enemy

        if current_enemy_id and current_enemy_id in objects:
            data = objects[current_enemy_id]
            x, y = data['position']
            depth = data['depth']
            # Kamera koordinatlarında 3D nokta hesaplama
            X = (x - cx) / fx * depth
            Y = (y - cy) / fy * depth
            Z = depth

            # 1) Dünya koordinatlarında hedef nokta (W çerçevesinde)
            pW_target = np.array([[X, Y, Z]])
            # 2) Kamera -> Pan–Tilt dönüşümü: (P çerçevesinde)
            pP_target = apply_hom_T(T_p_w, pW_target)  # (1,3)
            target_pt = pP_target[0]
            print("Hedef nokta (Pan-Tilt frame):", target_pt)

            # 3) Ters kinematik çözümü (başlangıç tahmini: nötrde 90°)
            pan_deg, tilt_deg = inverse_kinematics(target_pt, init_guess=[90.0, 90.0])
            print(f"Çözüm açılar => Pan (theta) = {pan_deg:.4f}°, Tilt (phi) = {tilt_deg:.4f}°")

            # 4) İleri kinematikle hesaplanan lazer emitter konumunu kontrol edelim
            p_laser, _ = forward_kinematics_full(pan_deg, tilt_deg)
            hata = p_laser - target_pt
            print("İleri kinematik sonucu lazer ucu (P çerçevesinde) =", p_laser)
            print("Hedefe olan hata =", hata)

            # Görsel çıktı: hedef üzerinde daire ve açı bilgisi yazdırma
            cv2.circle(frame, (x, y), int(data['radius']), (0, 0, 255), 2)
            cv2.putText(frame, f"Pan: {pan_deg:.2f}, Tilt: {tilt_deg:.2f}", (x - 50, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"ID: {current_enemy_id}", (x - 50, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Arduino'ya açıları gönder
            send_pan_tilt(pan_deg, tilt_deg)

        # Görüntüyü göster
        cv2.imshow("RealSense Kamera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Kaynakları serbest bırakma
    pipeline.stop()
    cv2.destroyAllWindows()
    arduino.close()
