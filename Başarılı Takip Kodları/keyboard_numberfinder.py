import cv2
import numpy as np

# Boş bir pencere oluşturun
cv2.namedWindow("Key Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Key Detector", 300, 100)
cv2.imshow("Key Detector", 255 * np.ones((100, 300), dtype=np.uint8))

print("Bir tuşa basın (çıkmak için 'q').")

while True:
    key = cv2.waitKey(0)  # tuşa basılana kadar bekle
    print("Key code:", key)
    if key == ord('q'):
        print("Çıkılıyor.")
        break

cv2.destroyAllWindows()
