import cv2
import numpy as np
import pyrealsense2 as rs

# Global değişkenler
clicked_hsv = []            # Her tıklamada alınan HSV değeri (3 değer)
max_clicks = 10             # Toplam izin verilen tıklama sayısı
click_count = 0             # Yapılan tıklama sayısı
# Her tıklamada alınan HSV değer etrafında oluşturulacak toleranslar (H: ±10, S: ±50, V: ±50)
tolerance = np.array([10, 50, 50])

# Global frame'ler: döngüde güncellenen görüntü ve HSV görüntüsü
global_frame_display = None
global_frame_hsv = None

def mouse_callback(event, x, y, flags, param):
    global click_count, clicked_hsv, global_frame_hsv, global_frame_display

    # Sadece sol fare tuşuyla tıklama ve toplam tıklama sayısı 10'dan küçükse
    if event == cv2.EVENT_LBUTTONDOWN and click_count < max_clicks:
        # Eğer global HSV frame mevcutsa, (x,y) koordinatındaki HSV değerini al
        if global_frame_hsv is not None:
            hsv_value = global_frame_hsv[y, x]
            clicked_hsv.append(hsv_value)
            click_count += 1

            # Görüntü üzerinde tıklanan noktayı işaretle
            cv2.circle(global_frame_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Kamera", global_frame_display)

            print(f"{click_count}. nokta HSV: {hsv_value}")

            # 10 tıklama tamamlandığında işlemi gerçekleştir
            if click_count == max_clicks:
                print("10 nokta seçildi. HSV aralığı hesaplanıyor...")

                # Her tıklamadan alınan HSV değeri etrafında tolerans ekleyip alt/üst sınırları belirle
                clicked_hsv_np = np.array(clicked_hsv, dtype=np.int32)
                lower_bounds = clicked_hsv_np - tolerance
                upper_bounds = clicked_hsv_np + tolerance

                # OpenCV için HSV geçerli aralıkları: H: 0-179, S: 0-255, V: 0-255
                lower_bounds[:, 0] = np.clip(lower_bounds[:, 0], 0, 179)
                lower_bounds[:, 1:] = np.clip(lower_bounds[:, 1:], 0, 255)
                upper_bounds[:, 0] = np.clip(upper_bounds[:, 0], 0, 179)
                upper_bounds[:, 1:] = np.clip(upper_bounds[:, 1:], 0, 255)

                # Tüm noktaların alt sınırlarının ortalaması ve üst sınırlarının ortalaması
                final_lower = np.mean(lower_bounds, axis=0).astype(np.uint8)
                final_upper = np.mean(upper_bounds, axis=0).astype(np.uint8)

                print("Hesaplanan Lower (alt) HSV sınırı:", final_lower)
                print("Hesaplanan Upper (üst) HSV sınırı:", final_upper)
                print("Artık tıklama alınmayacaktır.")

# Intel RealSense kamerasını başlat
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow("Kamera")
cv2.setMouseCallback("Kamera", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Kameradan gelen renkli görüntüyü numpy dizisine çevir
        frame = np.asanyarray(color_frame.get_data())

        # Görüntüyü kopyalayarak ekrana gösterilecek versiyonu oluştur
        global_frame_display = frame.copy()

        # HSV dönüşümü
        global_frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Eğer 10 tıklama yapılmamışsa güncel görüntüyü göster
        if click_count < max_clicks:
            cv2.imshow("Kamera", global_frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC tuşuna basılırsa çıkış yap
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
