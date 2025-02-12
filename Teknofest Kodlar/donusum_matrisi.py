import numpy as np
from scipy.optimize import least_squares

# -------------------------------------------------------------------
#  Yardımcı Fonksiyonlar
# -------------------------------------------------------------------

def deg2rad(angle_deg):
    """Derece -> Radyan dönüşümü."""
    return angle_deg * np.pi / 180.0

def rot_x(angle_deg):
    """X ekseni etrafında (sağ el kuralı) angle_deg kadar döndürme matrisi (3x3)."""
    a = deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([
        [1,   0,    0   ],
        [0,  ca,  -sa   ],
        [0,  sa,   ca   ]
    ])
    return R

def rot_y(angle_deg):
    """Y ekseni etrafında angle_deg kadar döndürme matrisi (3x3)."""
    a = deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([
        [ ca,  0,  sa ],
        [  0,  1,   0 ],
        [-sa,  0,  ca ]
    ])
    return R

def rot_z(angle_deg):
    """Z ekseni etrafında angle_deg kadar döndürme matrisi (3x3)."""
    a = deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([
        [ ca, -sa,  0 ],
        [ sa,  ca,  0 ],
        [  0,   0,  1 ]
    ])
    return R

def hom_trans(R, t):
    """
    3x3 bir rotasyon matrisi R ve 3x1 bir öteleme vektörü t verildiğinde,
    4x4'lük homojen dönüşüm matrisi döndür.
    """
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,  3] = t
    return T

def apply_hom_T(T, points):
    """
    T: 4x4'lük homojen dönüşüm matrisi
    points: Nx3 veya Nx4 boyutlu nokta dizisi (numpy array)
    
    Dönüşüm sonucu Nx3 boyutlu çıktı verir.
    """
    # Noktaları Nx4 haline getirelim (homojen koordinata genişlet)
    if points.shape[1] == 3:
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack([points, ones])  # (N,4)
    else:
        points_h = points  # Zaten 4 bileşenli
    
    transformed_h = (T @ points_h.T).T  # (N,4)
    return transformed_h[:,:3]  # 3 bileşene indir.

# -------------------------------------------------------------------
#  1. Dünya -> Kamera dönüşümü
# -------------------------------------------------------------------
# Kamera Dünya orijininde (0,0,0), fakat Euler açıları:
#   - Z: +0.5 derece
#   - X: -2.7 derece
#   - Y: +0.05 derece
# Sıralama varsayımı: R_C_W = R_y(0.05) * R_x(-2.7) * R_z(0.5)
# (Dikkat: Euler açı sıralaması projeye göre değişebilir.)

R_c_w = rot_y(0.05) @ rot_x(-2.7) @ rot_z(0.5)
t_c_w = np.array([0, 0, 0])   # Kameranın dünyadaki konumu
T_c_w = hom_trans(R_c_w, t_c_w)  # 4x4 homojen matris

# -------------------------------------------------------------------
#  2. Kamera -> PanTilt dönüşümü
# -------------------------------------------------------------------
# PanTilt tabanı kamera çerçevesinde (10, -2, 5),
# Eksenleri kamerayla paralel => rotasyon yok => R_p_c = I
t_p_c = np.array([-10, 2, -5])  # Dikkat: homojen matris son sütunda eksi işaretli olur.
R_p_c = np.eye(3)
T_p_c = hom_trans(R_p_c, t_p_c)

# Toplam Dünya -> PanTilt dönüşümü
# Dikkat: p^P = T_p_c * ( T_c_w * p^W )
T_p_w = T_p_c @ T_c_w

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
    T_pan   = hom_trans(rot_my(theta_deg), np.array([0,0,0]))  # sadece rot
    T_panL  = hom_trans(np.eye(3),         np.array([0,0,5]))  # pan kolu
    T_tilt  = hom_trans(rot_x(phi_deg),    np.array([0,0,0]))
    T_tiltL = hom_trans(np.eye(3),         np.array([0,0,4]))
    T_laser = hom_trans(np.eye(3),         np.array([3,0,0]))
    
    # Nihai çarpım (sağdan sola uygulanmasına dikkat, ama numpy'da soldan sağ matris çarpımıyla tutarlı olsun diye)
    T_total = T_laser @ T_tiltL @ T_tilt @ T_panL @ T_pan
    
    # Lazer ucunun (0,0,0) noktası homojen formda:
    p0 = np.array([[0, 0, 0, 1]]).T  # shape: (4,1)
    p_laser = T_total @ p0           # shape: (4,1)
    
    return p_laser[:3, 0]  # [x, y, z]

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

def inverse_kinematics(target_pos, init_guess=[0,0]):
    """
    Numerik yöntemle (Levenberg-Marquardt vs.) ters kinematik çöz.
    target_pos: (xP, yP, zP) => pan-tilt taban çerçevesinde hedef
    init_guess: [theta0, phi0] ilk açılar
    """
    res = least_squares(ik_error, x0=init_guess, args=(target_pos,))
    theta_sol, phi_sol = res.x
    return theta_sol, phi_sol


# -------------------------------------------------------------------
#  DEMO: (30, 30, 30) noktasını işaretlemek
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Dünya koordinatlarında hedef nokta:
    pW_target = np.array([[30, 30, 30]])  # shape: (1,3)
    
    # 2) Bunu pan-tilt çerçevesine dönüştür:
    pP_target = apply_hom_T(T_p_w, pW_target)  # shape: (1,3)
    xP, yP, zP = pP_target[0]
    print("Hedef nokta (Pan-Tilt frame): ", (xP, yP, zP))
    
    # 3) Ters kinematik çözümü:
    theta_guess, phi_guess = 0.0, 0.0  # Bir başlangıç açısı tahmini
    theta_sol, phi_sol = inverse_kinematics([xP, yP, zP], init_guess=[theta_guess, phi_guess])
    
    print(f"Çözüm açılar => Pan (theta) = {theta_sol:.4f} deg,  Tilt (phi) = {phi_sol:.4f} deg")
    
    # 4) İleri kinematikle kontrol edelim
    pP_laser = forward_kinematics(theta_sol, phi_sol)
    print("İleri kinematik sonucu lazer ucu (P çerçevesinde) =", pP_laser)
    print("Hedefe olan hata =", pP_laser - np.array([xP, yP, zP]))
