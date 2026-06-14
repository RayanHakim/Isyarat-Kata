import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7)
cap = cv2.VideoCapture(0)

print("\n=== REKAM ISYARAT (CVZONE VERSION) ===")
print("Tekan 's' untuk cetak koordinat ke terminal. Tekan 'q' untuk keluar.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, draw=True, flipType=False)
    
    current_data = None
    status = "Tangan TIDAK terdeteksi"
    
    if hands:
        status = "Tangan SIAP (Tekan 's')"
        hand1 = hands[0]
        lmList = hand1["lmList"]
        
        base_x, base_y, base_z = lmList[0]
        current_data = []
        for lm in lmList:
            current_data.append(lm[0] - base_x)
            current_data.append(lm[1] - base_y)
            current_data.append(lm[2] - base_z)
            
    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if hands else (0, 0, 255), 2)
    cv2.imshow('Kolektor Data', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and current_data is not None:
        print("\n--- COPY ARRAY INI ---")
        print(f"np.array({current_data}),")
        print("----------------------\n")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()