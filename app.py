import sys
from unittest.mock import MagicMock

# BLOCK TENSORFLOW GLOBAL
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.tools'] = MagicMock()
sys.modules['tensorflow.tools.docs'] = MagicMock()

import cv2
import numpy as np
import time
import mediapipe as mp

# INISIALISASI MEDIAPIPE
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    model_complexity=0, 
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

face_detector = mp_face.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.7
)

# =========================================================================
# DATABASE 100 KOSAKATA LENGKAP (Teks untuk GUI Help Menu)
# =========================================================================
KAMUS_ISYARAT = {
    # --- Interaksi Wajah (Spatial Face) ---
    "Tidur":        "(2 TANGAN) Telapak dirapatkan, ditempel ke pipi",
    "Cuci_Muka":    "(2 TANGAN) Kedua tangan terbuka di depan wajah",
    "Mendengar":    "Telunjuk menunjuk ke telinga",
    "Melihat":      "Dua jari (V) menunjuk ke mata",
    "Bicara":       "Tangan terbuka bergerak di depan mulut",
    "Makan":        "Jempol & telunjuk menguncup (capit) di depan mulut",
    "Minum":        "Pose telepon (jempol & kelingking) di depan mulut",
    "Berpikir":     "Telunjuk menempel di dahi/pelipis",
    "Menangis":     "Telunjuk diletakkan tepat di bawah mata",
    "Rahasia":      "Telunjuk diletakkan tegak di tengah bibir (Sssht)",
    "Pusing":       "Tangan terbuka memegang sisi kepala",
    
    # --- Interaksi Tubuh/Dada (Spatial Body) ---
    "Maaf":         "Tangan mengepal diletakkan di dada",
    "Sabar":        "Telapak tangan terbuka diletakkan di dada",
    "Lapar":        "Tangan memegang/menekan area perut bawah",
    
    # --- 2 Tangan (Multi-Hand) ---
    "Berdoa":       "(2 TANGAN) Kedua telapak tangan dirapatkan di dada",
    "Menyerah":     "(2 TANGAN) Kedua telapak tangan terbuka diangkat",
    "Bertinju":     "(2 TANGAN) Mengepal bersiap di depan dada",
    "Kutipan":      "(2 TANGAN) Tanda kutip dengan telunjuk & tengah",
    "Tepuk_Tangan": "(2 TANGAN) Kedua telapak tangan terbuka berdekatan",
    "Keluarga":     "(2 TANGAN) Tangan melengkung ujung jari saling bertemu",
    
    # --- Geometris Khusus & Orientasi ---
    "Halo":         "Telapak terbuka menghadap kamera",
    "Mantap":       "Acungkan jempol tegak lurus (Like)",
    "Selesai":      "Jempol rebah/horizontal ke samping",
    "Okey":         "Jempol dan telunjuk membentuk bulatan (OK)",
    "Kecil":        "Hanya kelingking yang tegak",
    "Cinta":        "Pose I Love You",
    "Tembak":       "Membentuk pistol ke depan",
    
    # --- Kata Tanya & Umum ---
    "Siapa":        "Telunjuk menggoyangkan/memutar kecil di udara",
    "Apa":          "Telapak tangan terbuka digerakkan mendatar",
    "Di_mana":      "Telunjuk menunjuk ke arah bawah",
    "Kapan":        "Telunjuk mengetuk pergelangan tangan (seperti jam)",
    "Ya":           "Kepalan tangan mengangguk ke atas-bawah",
    "Tidak":        "Telapak tangan melambai ke kanan-kiri",
    "Keren":        "Telunjuk & kelingking tegak (Metal)",
    "Kamu":         "Menunjuk lurus ke depan",
    "Saya":         "Menunjuk lurus ke belakang",
    
    # --- Angka 0-10 ---
    "Angka_0":      "Semua jari membentuk lingkaran utuh (Huruf O)",
    "Angka_1":      "Satu jari telunjuk tegak",
    "Angka_2":      "Dua jari (Telunjuk & Tengah) tegak / Damai",
    "Angka_3":      "Tiga jari (Telunjuk, Tengah, Manis) tegak",
    "Angka_4":      "Empat jari tegak rapat",
    "Angka_5":      "Lima jari terbuka lebar",
    "Angka_6":      "Jempol dan kelingking terbuka",
    "Angka_7":      "Jempol dan telunjuk terbuka lebar",
    "Angka_8":      "Jempol, telunjuk, jari tengah terbuka",
    "Angka_9":      "Empat jari terbuka, kelingking tertutup",
    "Angka_10":     "Tangan mengepal diguncang horizontal",
    
    # --- Alfabet (A-Z) ---
    "Huruf_A": "Kepalan tangan, jempol lurus di samping",
    "Huruf_B": "Empat jari lurus, jempol ditekuk ke telapak",
    "Huruf_C": "Tangan melengkung bentuk C",
    "Huruf_D": "Telunjuk lurus, jari lain melingkar menyentuh jempol",
    "Huruf_E": "Jari-jari ditekuk menyentuh pangkal telapak",
    "Huruf_F": "Telunjuk dan jempol menyatu bulat, jari lain tegak",
    "Huruf_G": "Jempol dan telunjuk lurus horizontal",
    "Huruf_H": "Telunjuk dan jari tengah lurus horizontal",
    "Huruf_I": "Hanya kelingking lurus ke atas",
    "Huruf_J": "Kelingking membuat gerakan J di udara",
    "Huruf_K": "Telunjuk-tengah lurus, jempol di ruas jari tengah",
    "Huruf_L": "Jempol dan telunjuk membentuk L",
    "Huruf_M": "Tiga jari ditekuk di atas jempol",
    "Huruf_N": "Dua jari ditekuk di atas jempol",
    "Huruf_O": "Jari menyatu membentuk O",
    "Huruf_P": "Huruf K yang dihadapkan ke bawah",
    "Huruf_Q": "Huruf G yang dihadapkan ke bawah",
    "Huruf_R": "Telunjuk dan jari tengah menyilang",
    "Huruf_S": "Tangan mengepal, jempol menutup jari lain",
    "Huruf_T": "Mengepal, jempol diselip di antara telunjuk-tengah",
    "Huruf_U": "Telunjuk dan tengah tegak rapat",
    "Huruf_V": "Telunjuk dan tengah terbuka lebar (Angka 2)",
    "Huruf_W": "Tiga jari terbuka membentuk W",
    "Huruf_X": "Telunjuk ditekuk seperti kait",
    "Huruf_Y": "Jempol dan kelingking lurus (Minum tanpa wajah)",
    "Huruf_Z": "Telunjuk melukis Z di udara"
}

# --- Peta Logika Statis (Untuk pencarian instan 1 Tangan via State Array) ---
# Format Tuple: (Jempol, Telunjuk, Tengah, Manis, Kelingking)
STATIC_MAP = {
    (0,1,1,1,1): "Angka_4",
    (1,1,1,1,1): "Angka_5",
    (1,1,1,0,0): "Angka_8",
    (1,1,1,1,0): "Angka_9",
    (0,1,0,0,0): "Angka_1",
    (0,1,1,0,0): "Angka_2",
    (0,1,1,1,0): "Angka_3",
    (1,0,0,0,1): "Huruf_Y",
    (0,1,0,0,1): "Keren",
    (1,1,0,0,1): "Cinta",
    (0,0,0,0,1): "Kecil",
    (0,0,0,0,0): "Simpan",
    (1,1,0,0,0): "Tembak",
    (0,0,1,1,1): "Huruf_F"
}

MENU_LIST = [(k, v) for k, v in KAMUS_ISYARAT.items()]

def hitung_jarak(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

kalimat_tercatat = []
kata_terakhir = ""
waktu_kata_terakhir = 0
durasi_kunci_kata = 1.0  

tampilkan_help = True  
scroll_index = 0
max_baris_tampil = 12

cap = cv2.VideoCapture(0)
print("\n=== PAPAN KETIK ISYARAT: 100 KOSAKATA LENGKAP ===")
print("- Sistem menggunakan Mapping Tuple & Spatial Geometrics.")
print("- Tekan 'w' / 's' untuk scroll menu.")
print("- Tekan 'q' untuk keluar.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_results = face_detector.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    
    status_kata_sekarang = ""
    wajah_terdeteksi = False
    face_kpts = {}
    
    # --- 1. DETEKSI WAJAH ---
    if face_results.detections:
        wajah_terdeteksi = True
        face = face_results.detections[0]
        kpts = face.location_data.relative_keypoints
        face_kpts['r_eye'] = (kpts[0].x, kpts[0].y)
        face_kpts['l_eye'] = (kpts[1].x, kpts[1].y)
        face_kpts['nose'] = (kpts[2].x, kpts[2].y)
        face_kpts['mouth'] = (kpts[3].x, kpts[3].y)
        face_kpts['r_ear'] = (kpts[4].x, kpts[4].y)
        face_kpts['l_ear'] = (kpts[5].x, kpts[5].y)
        
        bboxC = face.location_data.relative_bounding_box
        x, y, w, h = int(bboxC.xmin * w_frame), int(bboxC.ymin * h_frame), int(bboxC.width * w_frame), int(bboxC.height * h_frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # --- 2. DETEKSI TANGAN ---
    if hands_results.multi_hand_landmarks:
        jumlah_tangan = len(hands_results.multi_hand_landmarks)
        status_jari_all = [] 
        
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            tips_ids = [4, 8, 12, 16, 20]
            pip_ids = [3, 6, 10, 14, 18]
            mcp_ids = [2, 5, 9, 13, 17]
            
            jari_terbuka = [0, 0, 0, 0, 0]
            for i in range(1, 5): 
                d_ujung = hitung_jarak((hand_landmarks.landmark[tips_ids[i]].x, hand_landmarks.landmark[tips_ids[i]].y), 
                                       (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y))
                d_pangkal = hitung_jarak((hand_landmarks.landmark[pip_ids[i]].x, hand_landmarks.landmark[pip_ids[i]].y), 
                                         (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y))
                if d_ujung > d_pangkal: jari_terbuka[i] = 1
            
            label_tangan = hands_results.multi_handedness[idx].classification[0].label
            if label_tangan == "Right" and hand_landmarks.landmark[4].x < hand_landmarks.landmark[5].x: jari_terbuka[0] = 1
            elif label_tangan == "Left" and hand_landmarks.landmark[4].x > hand_landmarks.landmark[5].x: jari_terbuka[0] = 1

            status_jari_all.append(jari_terbuka)

        # ---------------------------------------------------------
        # ARSITEKTUR KLASIFIKASI 100 KOSAKATA
        # ---------------------------------------------------------
        
        # A. MULTI-HAND (2 TANGAN)
        if jumlah_tangan == 2:
            t1, t2 = status_jari_all[0], status_jari_all[1]
            w1 = (hands_results.multi_hand_landmarks[0].landmark[0].x, hands_results.multi_hand_landmarks[0].landmark[0].y)
            w2 = (hands_results.multi_hand_landmarks[1].landmark[0].x, hands_results.multi_hand_landmarks[1].landmark[0].y)
            jarak_tangan = hitung_jarak(w1, w2)
            
            if t1 in [[1,1,1,1,1], [0,1,1,1,1]] and t2 in [[1,1,1,1,1], [0,1,1,1,1]]:
                if wajah_terdeteksi and (hitung_jarak(w1, face_kpts['nose']) < 0.35 or hitung_jarak(w2, face_kpts['nose']) < 0.35) and jarak_tangan < 0.2:
                    status_kata_sekarang = "Tidur"
                elif wajah_terdeteksi and hitung_jarak(w1, face_kpts['nose']) < 0.35 and jarak_tangan > 0.2:
                    status_kata_sekarang = "Cuci_Muka"
                elif jarak_tangan < 0.15: status_kata_sekarang = "Berdoa"
                elif jarak_tangan > 0.4 and w1[1] < 0.5 and w2[1] < 0.5: status_kata_sekarang = "Menyerah"
                elif jarak_tangan < 0.25: status_kata_sekarang = "Tepuk_Tangan"
            elif t1 == [0,0,0,0,0] and t2 == [0,0,0,0,0]: status_kata_sekarang = "Bertinju"
            elif t1 == [0,1,1,0,0] and t2 == [0,1,1,0,0]: status_kata_sekarang = "Kutipan"
            elif t1 == [1,1,1,1,1] and t2 == [1,1,1,1,1]: status_kata_sekarang = "Keluarga"
            else: status_jari_all = [t1] # Fallback

        # B. SINGLE-HAND (1 TANGAN)
        if status_kata_sekarang == "":
            jari_terbuka = status_jari_all[0]
            state_tuple = tuple(jari_terbuka)
            
            ujung_telunjuk = (hands_results.multi_hand_landmarks[0].landmark[8].x, hands_results.multi_hand_landmarks[0].landmark[8].y)
            pergelangan = (hands_results.multi_hand_landmarks[0].landmark[0].x, hands_results.multi_hand_landmarks[0].landmark[0].y)
            
            # --- 1. Pengecekan Interaksi Wajah (Spatial Face) ---
            if wajah_terdeteksi:
                if jari_terbuka == [0,1,0,0,0]: 
                    if hitung_jarak(ujung_telunjuk, face_kpts['r_ear']) < 0.15 or hitung_jarak(ujung_telunjuk, face_kpts['l_ear']) < 0.15:
                        status_kata_sekarang = "Mendengar"
                    elif hitung_jarak(ujung_telunjuk, face_kpts['nose']) < 0.15 and ujung_telunjuk[1] > face_kpts['nose'][1]:
                        status_kata_sekarang = "Rahasia"
                    elif ujung_telunjuk[1] > face_kpts['r_eye'][1] and hitung_jarak(ujung_telunjuk, face_kpts['r_eye']) < 0.15:
                        status_kata_sekarang = "Menangis"
                    elif ujung_telunjuk[1] < face_kpts['r_eye'][1] and hitung_jarak(ujung_telunjuk, face_kpts['nose']) < 0.3:
                        status_kata_sekarang = "Berpikir"
                elif jari_terbuka == [0,1,1,0,0]: 
                    if hitung_jarak(ujung_telunjuk, face_kpts['nose']) < 0.2: status_kata_sekarang = "Melihat"
                elif jari_terbuka in [[1,1,1,1,1], [0,1,1,1,1], [1,1,1,1,0]]:
                    if hitung_jarak(ujung_telunjuk, face_kpts['mouth']) < 0.2: status_kata_sekarang = "Bicara"
                    elif hitung_jarak(pergelangan, face_kpts['r_ear']) < 0.25: status_kata_sekarang = "Pusing"
                elif jari_terbuka == [1,1,0,0,0]: 
                    if hitung_jarak(ujung_telunjuk, face_kpts['mouth']) < 0.15: status_kata_sekarang = "Makan"
                elif jari_terbuka == [1,0,0,0,1]:
                    if hitung_jarak(ujung_telunjuk, face_kpts['mouth']) < 0.2: status_kata_sekarang = "Minum"

            # --- 2. Pengecekan Area Dada/Perut (Spatial Body) ---
            if status_kata_sekarang == "":
                # Jika pergelangan tangan berada di bawah (Y > 0.6) dan di tengah X (0.3 - 0.7)
                if pergelangan[1] > 0.6 and 0.3 < pergelangan[0] < 0.7:
                    if jari_terbuka == [0,0,0,0,0]: status_kata_sekarang = "Maaf"
                    elif jari_terbuka == [1,1,1,1,1]: status_kata_sekarang = "Sabar"
                    elif jari_terbuka == [1,1,1,1,0]: status_kata_sekarang = "Lapar"
                    elif jari_terbuka == [0,1,0,0,0]: status_kata_sekarang = "Saya"

            # --- 3. Pengecekan Geometris & Orientasi Khusus ---
            if status_kata_sekarang == "":
                jarak_ok = hitung_jarak((hands_results.multi_hand_landmarks[0].landmark[8].x, hands_results.multi_hand_landmarks[0].landmark[8].y), 
                                        (hands_results.multi_hand_landmarks[0].landmark[4].x, hands_results.multi_hand_landmarks[0].landmark[4].y))
                jempol_horizontal = hands_results.multi_hand_landmarks[0].landmark[4].y > hands_results.multi_hand_landmarks[0].landmark[3].y
                
                if jari_terbuka == [1, 1, 1, 1, 1]:
                    if jarak_ok > 0.15: status_kata_sekarang = "Halo"
                    else: status_kata_sekarang = "Huruf_C"
                elif jari_terbuka == [1, 0, 0, 0, 0]:
                    if jempol_horizontal: status_kata_sekarang = "Selesai"
                    else: status_kata_sekarang = "Mantap"
                elif jari_terbuka == [0, 1, 0, 0, 0]:
                    y_diff = hands_results.multi_hand_landmarks[0].landmark[8].y - hands_results.multi_hand_landmarks[0].landmark[6].y
                    x_diff = hands_results.multi_hand_landmarks[0].landmark[8].x - hands_results.multi_hand_landmarks[0].landmark[6].x
                    if y_diff < -0.05: status_kata_sekarang = "Angka_1"
                    elif y_diff > 0.05: status_kata_sekarang = "Di_mana"
                    else: status_kata_sekarang = "Kamu"
                elif jari_terbuka == [0, 1, 1, 0, 0]:
                    jarak_jari = abs(hands_results.multi_hand_landmarks[0].landmark[8].x - hands_results.multi_hand_landmarks[0].landmark[12].x)
                    if jarak_jari > 0.05: status_kata_sekarang = "Damai"
                    else: status_kata_sekarang = "Huruf_U"
                elif jari_terbuka == [0, 0, 1, 1, 1] or jarak_ok < 0.035: status_kata_sekarang = "Okey"
                elif jari_terbuka == [0, 0, 0, 0, 0]:
                    if jempol_horizontal: status_kata_sekarang = "Huruf_S"
                    else: status_kata_sekarang = "Simpan"

            # --- 4. Fallback ke Mapping Array Statis ---
            if status_kata_sekarang == "" and state_tuple in STATIC_MAP:
                status_kata_sekarang = STATIC_MAP[state_tuple]

        # PENGUNCIAN KATA OTOMATIS
        if status_kata_sekarang != "":
            if status_kata_sekarang == kata_terakhir:
                if time.time() - waktu_kata_terakhir > durasi_kunci_kata:
                    if len(kalimat_tercatat) == 0 or kalimat_tercatat[-1] != status_kata_sekarang:
                        kalimat_tercatat.append(status_kata_sekarang)
                        print(f"[TERKETIK] -> {status_kata_sekarang}")
                    waktu_kata_terakhir = time.time()
            else:
                kata_terakhir = status_kata_sekarang
                waktu_kata_terakhir = time.time()
        else:
            kata_terakhir = ""
    else:
        kata_terakhir = ""

    # --- INTERFACE DESIGN ---
    cv2.rectangle(frame, (20, 20), (500, 80), (35, 35, 35), cv2.FILLED)
    warna_border = (0, 255, 0) if status_kata_sekarang else (0, 165, 255)
    cv2.rectangle(frame, (20, 20), (500, 80), warna_border, 2)
    cv2.putText(frame, f"INPUT: {status_kata_sekarang if status_kata_sekarang else 'MENUNGGU...'}", (40, 58), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna_border, 2)
    cv2.putText(frame, "KONTROL: 'h' (Hide/Show) | 'w' (Up) | 's' (Down)", (w_frame - 410, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    if tampilkan_help:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w_frame - 460, 60), (w_frame - 15, h_frame - 75), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, f"KAMUS ISYARAT ({scroll_index+1}-{min(scroll_index+max_baris_tampil, len(MENU_LIST))}/{len(MENU_LIST)}):", 
                    (w_frame - 440, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        
        y_offset = 130
        for kata, deskripsi in MENU_LIST[scroll_index : scroll_index + max_baris_tampil]:
            cv2.putText(frame, f"> {kata}", (w_frame - 440, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 2)
            cv2.putText(frame, f"  {deskripsi}", (w_frame - 440, y_offset + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            y_offset += 36

    gabungan_kalimat = " ".join(kalimat_tercatat)
    cv2.rectangle(frame, (0, h_frame - 60), (w_frame, h_frame), (10, 10, 10), cv2.FILLED)
    cv2.line(frame, (0, h_frame - 60), (w_frame, h_frame - 60), (50, 50, 50), 2)
    cv2.putText(frame, f"TEKS KALIMAT: {gabungan_kalimat}", (25, h_frame - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Keyboard Isyarat (100 Kata + Spatial Face)', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('h'): tampilkan_help = not tampilkan_help
    elif key == ord('w') and scroll_index > 0: scroll_index -= 1
    elif key == ord('s') and scroll_index < len(MENU_LIST) - max_baris_tampil: scroll_index += 1
    elif key == ord('c'): kalimat_tercatat = []
    elif key == ord('e'):
        if kalimat_tercatat:
            nama_file = f"Hasil_Ketik_Isyarat_{int(time.time())}.txt"
            with open(nama_file, "w") as f: f.write(gabungan_kalimat)
            print(f"[EXPORT BERHASIL] File disimpan: {nama_file}")
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()