import pyrealsense2 as rs
import numpy as np
import cv2

# RealSense pipeline ve konfigürasyon ayarları
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# SimpleBlobDetector parametrelerini ayarlıyoruz
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255  # Binary maskede beyaz noktaları tespit edeceğiz
params.minThreshold = 0
params.maxThreshold = 256
params.filterByArea = True
params.minArea = 1      # Uzak mesafedeki küçük noktalar için
params.maxArea = 5000   # Çok büyük alanları eleyebilirsiniz
params.filterByCircularity = True
params.minCircularity = 0.7  # Lazer noktasının daireselliği
params.filterByInertia = True
params.minInertiaRatio = 0.5
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)

try:
    while True:
        # Frame setini al
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Renk frame'ini numpy array formatına çevir
        frame = np.asanyarray(color_frame.get_data())

        # BGR görüntüyü HSV renk uzayına dönüştür
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Lazer için uygun HSV aralığı (uzak mesafe koşulları için doygunluk/parlaklık değerleri biraz indirildi)
        lower_green = np.array([24, 35, 204])
        upper_green = np.array([44, 123, 255])
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        lower_green1 = np.array([50, 27, 166])
        upper_green1 = np.array([70, 120, 200])
        mask2 = cv2.inRange(hsv, lower_green1, upper_green1)
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Morfolojik işlemler:
        # 1. Dilatasyon: Noktanın büyümesini sağlamak için
        # 2. Açma (MORPH_OPEN): Küçük gürültüleri temizler (erosyon + dilatasyon)
        # 3. Kapanma (MORPH_CLOSE): Küçük delikleri doldurur (dilatasyon + erozyon)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Gaussian Blur ile gürültüyü azalt
        mask_blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        # Blob detektörü ile lazer noktasını tespit et
        keypoints = detector.detect(mask_blurred)

        # Tespit edilen noktaların etrafına daire çizmek yerine,
        # stabil maskeleme sonucunu ve orijinal görüntüyü gösteriyoruz.
        cv2.imshow("combined_mask", mask_blurred)
        cv2.imshow("Original", frame)

        # 'Esc' tuşuna basılırsa çıkış yap
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
