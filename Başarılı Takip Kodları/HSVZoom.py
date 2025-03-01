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

# Zoom özellikleri için gerekli değişkenler
is_zoomed = False           # Şu an zoom modunda mıyız?
zoom_roi = None             # Zoom yapılacak bölgenin koordinatları
zoom_factor = 5             # Zoom faktörü - ne kadar büyüteceğiz
zoom_size = 200             # Zoom penceresinin boyutu (piksel olarak)
temp_click_point = None     # Geçici ilk tıklama noktası
zoom_image = None           # Zoom yapılmış görüntü

def main_window_callback(event, x, y, flags, param):
    global click_count, is_zoomed, zoom_roi, temp_click_point
    
    # Sadece sol fare tuşuyla tıklama ve toplam tıklama sayısı 10'dan küçükse
    if event == cv2.EVENT_LBUTTONDOWN and click_count < max_clicks and not is_zoomed:
        # İlk tıklama noktasını kaydet
        temp_click_point = (x, y)
        
        # Zoom bölgesini belirle
        half_size = zoom_size // (2 * zoom_factor)
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(global_frame_display.shape[1], x + half_size)
        y2 = min(global_frame_display.shape[0], y + half_size)
        
        zoom_roi = (x1, y1, x2, y2)
        is_zoomed = True
        
        # Zoom yapılmış görüntüyü hazırla ve göster
        show_zoomed_image()

def zoom_window_callback(event, x, y, flags, param):
    global click_count, clicked_hsv, global_frame_hsv, global_frame_display
    global is_zoomed, zoom_roi, zoom_image
    
    # Zoom penceresinde sol tıklama yapıldıysa
    if event == cv2.EVENT_LBUTTONDOWN and is_zoomed and click_count < max_clicks:
        # Zoom yapılmış görüntüdeki tıklama koordinatlarını gerçek görüntüdeki koordinatlara dönüştür
        real_x = zoom_roi[0] + x // zoom_factor
        real_y = zoom_roi[1] + y // zoom_factor
        
        # Koordinatlar görüntü sınırları içinde olmalı
        real_x = min(max(0, real_x), global_frame_hsv.shape[1] - 1)
        real_y = min(max(0, real_y), global_frame_hsv.shape[0] - 1)
        
        # HSV değerini al
        hsv_value = global_frame_hsv[real_y, real_x]
        clicked_hsv.append(hsv_value)
        click_count += 1
        
        # Orijinal görüntüde işaretleme yap
        cv2.circle(global_frame_display, (real_x, real_y), 5, (0, 0, 255), -1)
        
        # Zoom görüntüsünde de işaretleme yap
        cv2.circle(zoom_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Zoom Görüntüsü", zoom_image)
        
        print(f"{click_count}. nokta HSV: {hsv_value}")
        
        # Zoom modundan çık
        is_zoomed = False
        
        # Normal görüntüyü göster
        cv2.imshow("Kamera", global_frame_display)
        cv2.destroyWindow("Zoom Görüntüsü")
        
        # 10 tıklama tamamlandığında işlemi gerçekleştir
        if click_count == max_clicks:
            calculate_hsv_range()

def calculate_hsv_range():
    global clicked_hsv
    
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

def show_zoomed_image():
    global global_frame_display, zoom_roi, temp_click_point, zoom_image
    
    if zoom_roi is None or global_frame_display is None:
        return
    
    # Zoom yapılacak bölgeyi kırp
    x1, y1, x2, y2 = zoom_roi
    roi = global_frame_display[y1:y2, x1:x2]
    
    # Görüntüyü büyüt
    if roi.size > 0:  # Boş olmadığından emin ol
        zoom_image = cv2.resize(roi, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
        
        # Yakınlaştırılmış görüntünün kenarlarını belli etmek için çerçeve çiz
        cv2.rectangle(zoom_image, (0, 0), (zoom_size-1, zoom_size-1), (0, 255, 0), 2)
        
        # Orta noktayı belirten bir artı çiz
        center = zoom_size // 2
        cv2.line(zoom_image, (center-10, center), (center+10, center), (0, 255, 0), 1)
        cv2.line(zoom_image, (center, center-10), (center, center+10), (0, 255, 0), 1)
        
        # Geçici tıklama noktasını orijinal görüntüde göster
        cv2.circle(global_frame_display, temp_click_point, 3, (0, 255, 0), -1)
        
        # Zoom görüntüsü için pencere oluştur ve mouse callback ekle
        cv2.namedWindow("Zoom Görüntüsü")
        cv2.setMouseCallback("Zoom Görüntüsü", zoom_window_callback)
        cv2.imshow("Zoom Görüntüsü", zoom_image)

# Intel RealSense kamerasını başlat
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow("Kamera")
cv2.setMouseCallback("Kamera", main_window_callback)

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
        
        # Zoom modunda değilsek güncel görüntüyü göster
        if not is_zoomed:
            # Daha önce seçilmiş noktaları tekrar çiz
            for i in range(click_count):
                if i < len(clicked_hsv):
                    # Seçilen pixellerin pozisyonunu bulmak için gerçek koordinatları kullanmalıyız
                    # Bu örnekte koordinatları kaydediyoruz (gerçek projede bunu eklemeniz gerekir)
                    x = 100 + i * 20  # Örnek koordinat (bu değiştirilmeli)
                    y = 100           # Örnek koordinat (bu değiştirilmeli)
                    cv2.circle(global_frame_display, (x, y), 5, (0, 0, 255), -1)
            
            cv2.imshow("Kamera", global_frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC tuşuna basılırsa çıkış yap
            break
        elif key == ord('c'):  # 'c' tuşuna basılırsa zoom modundan çık
            is_zoomed = False
            cv2.imshow("Kamera", global_frame_display)
            cv2.destroyWindow("Zoom Görüntüsü")
        elif key == ord('r'):  # 'r' tuşuna basılırsa işlemi sıfırla
            clicked_hsv = []
            click_count = 0
            is_zoomed = False
            cv2.imshow("Kamera", global_frame_display)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
