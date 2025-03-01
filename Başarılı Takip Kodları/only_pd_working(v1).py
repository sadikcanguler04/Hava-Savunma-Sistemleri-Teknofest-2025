import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import time
from collections import OrderedDict

# -----------------------------
# 1. Lazer Takibi İçin Yardımcı Sınıf (CentroidTracker)
# -----------------------------
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

# -----------------------------
# 2. Arduino ve Pan-Tilt Kontrolü
# -----------------------------
pan_aci = -7
tilt_aci = 6

arduino = serial.Serial('COM6', 115200)
time.sleep(2)

def send_pan_tilt(pan_angle, tilt_angle):
    cmd = f"{pan_angle:.2f},{tilt_angle:.2f}\n"
    arduino.write(cmd.encode())
    print(f"Pan: {pan_angle:.2f}, Tilt: {tilt_angle:.2f} gönderildi.")
    if arduino.in_waiting:
        response = arduino.readline().decode().strip()
        print(f"Arduino yanıtı: {response}")

# -----------------------------
# 3. PD Kontrol Parametreleri
# -----------------------------
current_pan_angle = 90.0
current_tilt_angle = 90.0

# PD kazançları (sisteminizin dinamiğine göre ayarlayın)
Kp_pan = 0.05
Kd_pan = 0.01
Kp_tilt = 0.05
Kd_tilt = 0.01

last_error_pan = 0.0
last_error_tilt = 0.0
last_time = time.time()
error_threshold = 0.5  # Küçük hata varsa güncelleme yapmayız

# -----------------------------
# 4. RealSense Kamera Yapılandırması
# -----------------------------
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

# -----------------------------
# 5. Lazer Tespiti İçin Blob Detector Parametreleri (Sizin örneğinizdeki gibi)
# -----------------------------
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255   # Binary maskede beyaz noktalar
params.minThreshold = 0
params.maxThreshold = 256
params.filterByArea = True
params.minArea = 1       # Uzak mesafedeki küçük noktalar için
params.maxArea = 5000    # Çok büyük alanları ele
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByInertia = True
params.minInertiaRatio = 0.5
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)

# Balon için kırmızı HSV eşik değerleri
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])

ct = CentroidTracker(maxDisappeared=15)
send_pan_tilt(current_pan_angle + pan_aci, current_tilt_angle + tilt_aci)
start_time = time.time()

# -----------------------------
# 6. Ana Döngü: Görüntü İşleme, Nesne Tespiti ve PD Kontrolü
# -----------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()  # Opsiyonel
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Başlangıçta 3 sn bekleme (sistem stabilizasyonu için)
        if time.time() - start_time < 3.0:
            cv2.putText(frame, "Sistem baslatiliyor... (3 sn bekleyin)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Entegre Kontrol", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # -----------------------------
        # A) Lazer Tespiti: HSV aralığı kullanılarak maske oluşturuluyor
        # -----------------------------
        # Sizin verdiğiniz filtre değerlerini kullanıyoruz
        lower_green = np.array([24, 35, 204])
        upper_green = np.array([44, 123, 255])
        lower_green1 = np.array([50, 27, 166])
        upper_green1 = np.array([70, 120, 200])
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        mask2 = cv2.inRange(hsv, lower_green1, upper_green1)
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Morfolojik işlemler: Dilatasyon, açma, kapanma
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        # SimpleBlobDetector ile lazer tespiti
        keypoints = detector.detect(mask_blurred)
        laser_centroid = None
        if keypoints:
            # Eğer birden fazla varsa, en büyük ya da ilkini kullanabiliriz
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # -----------------------------
        # B) Balon Tespiti (Kırmızı renk)
        # -----------------------------
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        balloon_centroid = None
        if red_contours:
            valid_contours = [cnt for cnt in red_contours if cv2.contourArea(cnt) > 600]
            if valid_contours:
                cnt = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx_balloon = int(M["m10"] / M["m00"])
                    cy_balloon = int(M["m01"] / M["m00"])
                    balloon_centroid = (cx_balloon, cy_balloon)
                    cv2.circle(frame, balloon_centroid, 6, (0, 0, 255), -1)
                    cv2.putText(frame, "Balon", (cx_balloon - 20, cy_balloon - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Balon bulunamadi", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # -----------------------------
        # C) PD Kontrol: Sadece hem lazer hem balon tespit edildiyse
        # -----------------------------
        if laser_centroid is not None and balloon_centroid is not None:
            # Hata vektörü: balon merkezinin lazer merkezine göre konumu
            error_x = balloon_centroid[0] - laser_centroid[0]
            error_y = balloon_centroid[1] - laser_centroid[1]
            error_angle_pan = (error_x / fx) * (180.0 / np.pi)
            error_angle_tilt = (error_y / fy) * (180.0 / np.pi)
            current_time = time.time()
            dt = current_time - last_time if current_time - last_time > 0 else 1e-3

            # PD kontrol: NOT! Burada açılar güncellenirken hata yönünü ters alıyoruz
            if abs(error_angle_pan) > error_threshold or abs(error_angle_tilt) > error_threshold:
                d_error_pan = (error_angle_pan - last_error_pan) / dt
                d_error_tilt = (error_angle_tilt - last_error_tilt) / dt
                output_pan = Kp_pan * error_angle_pan + Kd_pan * d_error_pan
                output_tilt = Kp_tilt * error_angle_tilt + Kd_tilt * d_error_tilt
                # Hata doğru yönde düzeltilmesi için çıkarıyoruz
                current_pan_angle -= output_pan
                current_tilt_angle -= output_tilt
                last_error_pan = error_angle_pan
                last_error_tilt = error_angle_tilt
                last_time = current_time

            cv2.putText(frame, f"Hata X: {error_x:.1f}, Y: {error_y:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Pan: {current_pan_angle:.2f}, Tilt: {current_tilt_angle:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            send_pan_tilt(current_pan_angle + pan_aci, current_tilt_angle + tilt_aci)
        else:
            cv2.putText(frame, "PD kontrol: Balon veya lazer bulunamadi", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # -----------------------------
        # D) Görüntüleri Göster
        # -----------------------------
        cv2.imshow("Entegre Kontrol", frame)
        cv2.imshow("Lazer Maskesi", mask_blurred)
        cv2.imshow("Balon Maskesi", red_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    arduino.close()
