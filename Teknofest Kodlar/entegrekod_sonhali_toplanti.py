import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import time

from numpy import ones
from scipy.optimize import least_squares


# -------------------------------------------------------------------
#  Yardımcı Fonksiyonlar
# -------------------------------------------------------------------


# Arduino bağlantısı
arduino = serial.Serial('COM5', 115200)
time.sleep(2)  # Arduino'nun hazır olmasını bekle


#def send_pan_tilt(pan_angle, tilt_angle):
#    """Pan ve tilt açılarını Arduino'ya gönderir."""
#    arduino.write(f"{pan_angle},{tilt_angle}\n".encode())
#    print(f"Pan: {pan_angle}, Tilt: {tilt_angle} gönderildi.")
#    response = arduino.readline().decode().strip()
#    print(f"Arduino yanıtı: {response}")

#Yeni ve 2 tane pan tilt için seri haberleşme kodu
def send_angles(pan1, tilt1, pan2, tilt2):
    command = f"{pan1},{tilt1},{pan2},{tilt2}\n"
    arduino.write(command.encode()) 
    response = arduino.readline().decode().strip() 
    print("Arduino:", response)



#sistemimiz pan oldugu icin şu an sadece kamera seri haberleşmesi
def camera_pan_send_angle(angle):
    command = f"{angle}\n"
    arduino.write(command.encode())  # Açı değerini gönder
    response = arduino.readline().decode().strip()  # Arduino'nun cevabını oku
    print("Arduino:", response)


#Pan tilt Mekanizmalarımızın kamerayla arasındaki mesafeleri (offset)

#işaretleyici laser için değerler
pantilt_x_offset= 0
pantilt_y_offset= 0
pantilt_z_offset= 0

#Vurucu laser için ekledik bu 2.olan değerleri
pantilt_x_offset2= 0
pantilt_y_offset2= 0
pantilt_z_offset2= 0

#kamera son pozisyonun acisini tutan degerler
#yakında kamerayı döndürdüğümüz bir sistem ekleyeceğimizde bu değerler üzerinden gideceğiz
camera_degree_pan = 0
camera_degree_tilt = 0
camera_degree_roll = 0





def deg2rad(angle_deg):
    """Derece -> Radyan dönüşümü."""
    return angle_deg * np.pi / 180.0


def rot_x(angle_deg):
    """X ekseni etrafında (sağ el kuralı) angle_deg kadar döndürme matrisi (3x3)."""
    a = deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ])
    return R


def rot_y(angle_deg):
    """Y ekseni etrafında angle_deg kadar döndürme matrisi (3x3)."""
    a = deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ])
    return R


def rot_z(angle_deg):
    """Z ekseni etrafında angle_deg kadar döndürme matrisi (3x3)."""
    a = deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([
        [ca, -sa, 0],
        [sa, ca, 0],
        [0, 0, 1]
    ])
    return R


def hom_trans(R, t):
    """
    3x3 bir rotasyon matrisi R ve 3x1 bir öteleme vektörü t verildiğinde,
    4x4'lük homojen dönüşüm matrisi döndür.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_hom_T(T, points):
    """
    T: 4x4'lük homojen dönüşüm matrisi
    points: Nx3 veya Nx4 boyutlu nokta dizisi (numpy array)

    Dönüşüm sonucu Nx3 boyutlu çıktı verir.
    """
    # Noktaları Nx4 haline getirelim (homojen koordinata genişlet)
    if points.shape[1] == 3:

        points_h = np.hstack([points, ones])  # (N,4)
    else:
        points_h = points  # Zaten 4 bileşenli

    transformed_h = (T @ points_h.T).T  # (N,4)
    return transformed_h[:, :3]  # 3 bileşene indir.


# -------------------------------------------------------------------
#  1. Dünya -> Kamera dönüşümü
# -------------------------------------------------------------------
# Kamera Dünya orijininde (0,0,0), fakat Euler açıları:
#   - Z: +0.5 derece
#   - X: -2.7 derece
#   - Y: +0.05 derece
# Sıralama varsayımı: R_C_W = R_y(0.05) * R_x(-2.7) * R_z(0.5)
# (Dikkat: Euler açı sıralaması projeye göre değişebilir.)

R_c_w = rot_y(camera_degree_pan) @ rot_x(camera_degree_tilt) @ rot_z(camera_degree_roll)
t_c_w = np.array([0, 0, 0])  # Kameranın dünyadaki konumu
T_c_w = hom_trans(R_c_w, t_c_w)  # 4x4 homojen matris

# -------------------------------------------------------------------
#  2. Kamera -> PanTilt dönüşümü
# -------------------------------------------------------------------
# PanTilt tabanı kamera çerçevesinde (10, -2, 5),
# Eksenleri kamerayla paralel => rotasyon yok => R_p_c = I
t_p_c = np.array([pantilt_x_offset,pantilt_y_offset, pantilt_z_offset])  # Dikkat: homojen matris son sütunda eksi işaretli olur.
R_p_c = np.eye(3)
T_p_c = hom_trans(R_p_c, t_p_c)

#2.pan tilt mekanizması için (vurucu laser pan-tilt --> kamera dönüşümü)
t_p_c2 = np.array([pantilt_x_offset2,pantilt_y_offset2, pantilt_z_offset2])  # Dikkat: homojen matris son sütunda eksi işaretli olur.
R_p_c2 = np.eye(3)
T_p_c2 = hom_trans(R_p_c, t_p_c)


# Toplam Dünya -> PanTilt dönüşümü
# Dikkat: p^P = T_p_c * ( T_c_w * p^W )
T_p_w = T_p_c @ T_c_w

T_p_w2 = T_p_c2 @ T_c_w


# -------------------------------------------------------------------
#  3. Pan-Tilt-Lazer Parametreleri (İleri Kinematik)
# -------------------------------------------------------------------
# - Pan ekseni: -Y etrafında dönüyor (açı: theta)
# - Pan kolu: 5 birim (Z ekseni boyunca)
# - Tilt ekseni: +X etrafında dönüyor (açı: phi)
# - Tilt kolu: 4 birim (Z ekseni boyunca)
# - Lazer ofseti: +X yönünde 3 birim
#
# Lazerin ucunun, Pan-Tilt taban çerçevesine göre konumu:
#
#   T_laser(P->laser) = Trans(3,0,0)*Trans(0,0,4)*Rot_x(phi)*Trans(0,0,5)*Rot_{-y}(theta)
#
# Bu matris ile (0,0,0,1) noktasını çarparsak lazer ucunun (x,y,z) konumunu elde ederiz.

def rot_my(angle_deg):
    """-Y ekseni etrafında dönme (pan ekseninin -y olduğunu varsayıyoruz)."""
    # Rot_{-y}(theta) = R_y(-theta)
    return rot_y(-angle_deg)


def forward_kinematics(theta_deg, phi_deg):
    """
    Verilen pan (theta) ve tilt (phi) açıları için
    lazer ucunun pan-tilt taban çerçevesindeki [x, y, z] konumunu döndür.
    """
    # Homojen matrisleri sırayla çarpalım
    # 1) Rot_{-y}(theta)
    # 2) Trans(0,0,5)
    # 3) Rot_x(phi)
    # 4) Trans(0,0,4)
    # 5) Trans(3,0,0)

    # (4x4 olarak ilerleyeceğiz)
    T_pan = hom_trans(rot_my(theta_deg), np.array([0, 0, 0]))  # sadece rot
    T_panL = hom_trans(np.eye(3), np.array([0, 0, 5]))  # pan kolu
    T_tilt = hom_trans(rot_x(phi_deg), np.array([0, 0, 0]))
    T_tiltL = hom_trans(np.eye(3), np.array([0, 0, 4]))
    T_laser = hom_trans(np.eye(3), np.array([3, 0, 0]))

    # Nihai çarpım (sağdan sola uygulanmasına dikkat, ama numpy'da soldan sağ matris çarpımıyla tutarlı olsun diye)
    T_total = T_laser @ T_tiltL @ T_tilt @ T_panL @ T_pan

    # Lazer ucunun (0,0,0) noktası homojen formda:
    p0 = np.array([[0, 0, 0, 1]]).T  # shape: (4,1)
    p_laser = T_total @ p0  # shape: (4,1)

    return p_laser[:3, 0]  # [x, y, z]

#2.pan tilt yani vurucu laser için forward kinematic fonksiyonu
def forward_kinematics2(theta_deg, phi_deg):
    """
    Verilen pan (theta) ve tilt (phi) açıları için
    lazer ucunun pan-tilt taban çerçevesindeki [x, y, z] konumunu döndür.
    """
    # Homojen matrisleri sırayla çarpalım
    # 1) Rot_{-y}(theta)
    # 2) Trans(0,0,5)
    # 3) Rot_x(phi)
    # 4) Trans(0,0,4)
    # 5) Trans(3,0,0)

    # (4x4 olarak ilerleyeceğiz)
    T_pan2 = hom_trans(rot_my(theta_deg), np.array([0, 0, 0]))  # sadece rot
    T_panL2 = hom_trans(np.eye(3), np.array([0, 0, 5]))  # pan kolu
    T_tilt2 = hom_trans(rot_x(phi_deg), np.array([0, 0, 0]))
    T_tiltL2 = hom_trans(np.eye(3), np.array([0, 0, 4]))
    T_laser2 = hom_trans(np.eye(3), np.array([3, 0, 0]))

    # Nihai çarpım (sağdan sola uygulanmasına dikkat, ama numpy'da soldan sağ matris çarpımıyla tutarlı olsun diye)
    T_total2 = T_laser2 @ T_tiltL2 @ T_tilt2 @ T_panL2 @ T_pan2

    # Lazer ucunun (0,0,0) noktası homojen formda:
    p02 = np.array([[0, 0, 0, 1]]).T  # shape: (4,1)
    p_laser2 = T_total2 @ p02  # shape: (4,1)

    return p_laser2[:3, 0]  # [x, y, z]




# -------------------------------------------------------------------
#  4. Ters Kinematik (Inverse Kinematics) - Basit bir numerik çözüm
# -------------------------------------------------------------------
# Elde etmek istediğimiz: forward_kinematics(theta, phi) = (xP, yP, zP)
#
# Burada, xP,yP,zP pan-tilt taban çerçevesindeki hedef nokta.
# Aşağıdaki fonksiyon, hata vektörünü döndürüyor. Bunu least_squares'e vereceğiz.

def ik_error(angles, target_pos):
    """
    angles: [theta_deg, phi_deg]
    target_pos: [xP, yP, zP]
    Döndürülen sonuç: forward_kinematics(...) - target_pos
    """
    theta, phi = angles
    fk = forward_kinematics(theta, phi)
    return fk - target_pos


def inverse_kinematics(target_pos, init_guess=[0, 0]):
    """
    Numerik yöntemle (Levenberg-Marquardt vs.) ters kinematik çöz.
    target_pos: (xP, yP, zP) => pan-tilt taban çerçevesinde hedef
    init_guess: [theta0, phi0] ilk açılar
    """
    res = least_squares(ik_error, x0=init_guess, args=(target_pos,))
    theta_sol, phi_sol = res.x
    return theta_sol, phi_sol


# RealSense yapılandırması
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

# Nesne takip bilgileri
objects = {}
next_enemy_id = 1
disappear_threshold = 10
frame_count = 0

# Renk eşik değerleri
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_area(radius):
    return math.pi * (radius ** 2)


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

        # Takip edilen enemy'i güncelle
        frame_count += 1
        for center, radius, area, depth in detected_centers:
            matched = False
            for obj_id, data in objects.items():
                distance = calculate_distance(center, data['position'])
                if distance < 130:
                    objects[obj_id].update(
                        {'position': center, 'radius': radius, 'area': area, 'depth': depth, 'last_seen': frame_count})
                    matched = True
                    break
            if not matched:
                objects[f"Red-{next_enemy_id}"] = {'position': center, 'radius': radius, 'area': area, 'depth': depth,
                                                   'last_seen': frame_count}
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
            # Buraadaki X,Y ve Z koordinatları dünya koordinatları
            X = (x - cx) / fx * depth
            Y = (y - cy) / fy * depth
            Z = depth

            if __name__ == "__main__":
                # 1) Dünya koordinatlarında hedef nokta:

                pW_target = np.array([[X, Y, Z]])  # shape: (1,3)


                # 2) Bunu pan-tilt çerçevesine dönüştür:
                pP_target = apply_hom_T(T_p_w, pW_target)  # shape: (1,3)
                xP, yP, zP = pP_target[0]
                print("Hedef nokta 1.(Pan-Tilt frame): ", (xP, yP, zP))



                pP_target2 = apply_hom_T(T_p_w, pW_target)  # shape: (1,3)
                xP2, yP2, zP2 = pP_target2[0]
                print("Hedef nokta 2.(Pan-Tilt frame): ", (xP2, yP2, zP2))

                # 3) Ters kinematik çözümü:
                theta_guess, phi_guess = 0.0, 0.0  # Bir başlangıç açısı tahmini
               
               #  pan_deg, tilt_deg = inverse_kinematics([xP, yP, zP], init_guess=[theta_guess, phi_guess])

                pan_deg_laser1, tilt_deg_laser1 = inverse_kinematics([xP, yP, zP], init_guess=[theta_guess, phi_guess])


                pan_deg_laser2, tilt_deg_laser2 = inverse_kinematics([xP2, yP2, zP2], init_guess=[theta_guess, phi_guess])

                # 4) İleri kinematikle kontrol edelim
                pP_laser = forward_kinematics(pan_deg_laser1, tilt_deg_laser1)


                pP_laser2 = forward_kinematics2(pan_deg_laser2, tilt_deg_laser2)



            cv2.circle(frame, (x, y), int(data['radius']), (0, 0, 255), 2)
            cv2.putText(frame, f"Pan: {pan_deg:.2f}, Tilt: {tilt_deg:.2f}", (x - 50, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"ID: {current_enemy_id}", (x - 50, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

           # send_pan_tilt(pan_deg, tilt_deg)
            #2 pan tilt için arduino seri haberleşme
            send_angles(pan_deg_laser1, tilt_deg_laser1, pan_deg_laser2, tilt_deg_laser2)

        cv2.imshow("RealSense Kamera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
