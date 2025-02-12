import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import time

# Arduino bağlantısı
arduino = serial.Serial('COM5', 115200)
time.sleep(2)  # Arduino'nun hazır olmasını bekle

def send_pan_tilt(pan_angle, tilt_angle):
    """Pan ve tilt açılarını Arduino'ya gönderir."""
    arduino.write(f"{pan_angle},{tilt_angle}\n".encode())
    print(f"Pan: {pan_angle}, Tilt: {tilt_angle} gönderildi.")
    response = arduino.readline().decode().strip()
    print(f"Arduino yanıtı: {response}")

# Pan-tilt taban noktasının kamera koordinatları (metre)
X_pt = -0.12
Y_pt = -0.35
Z_pt = 0.0735

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
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_area(radius):
    return math.pi * (radius**2)

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
                    objects[obj_id].update({'position': center, 'radius': radius, 'area': area, 'depth': depth, 'last_seen': frame_count})
                    matched = True
                    break
            if not matched:
                objects[f"Red-{next_enemy_id}"] = {'position': center, 'radius': radius, 'area': area, 'depth': depth, 'last_seen': frame_count}
                next_enemy_id += 1

        # Kaybolan nesneleri kaldır
        to_remove = [obj_id for obj_id, data in objects.items() if frame_count - data['last_seen'] > disappear_threshold]
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
            X = (x - cx) / fx * depth
            Y = (y - cy) / fy * depth
            Z = depth

            VX = X - X_pt
            VY = Y - Y_pt
            VZ = Z - Z_pt

            pan = math.atan2(VZ, VX)
            h_dist = math.sqrt(VX**2 + VZ**2)
            tilt_raw = math.atan2(VY, h_dist)
            tilt = tilt_raw + math.pi / 2

            pan = (pan + math.pi) % (2 * math.pi) if pan < 0 else pan
            tilt = max(0, min(math.pi, tilt))

            pan_deg = math.degrees(pan)
            tilt_deg = math.degrees(tilt)

            cv2.circle(frame, (x, y), int(data['radius']), (0, 0, 255), 2)
            cv2.putText(frame, f"Pan: {pan_deg:.2f}, Tilt: {tilt_deg:.2f}", (x - 50, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"ID: {current_enemy_id}", (x - 50, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            send_pan_tilt(pan_deg, tilt_deg)

        cv2.imshow("RealSense Kamera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
