import cv2
import os
import hashlib
import numpy as np
import time
from datetime import datetime

# Folder setup
EMPLOYEE_FOLDER = 'employee'
os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Helper to capture and save an image and coordinates
def capture_and_save(cap, emp_id):
    face_data = {}
    face_detected_time = None
    saved = False

    while not saved:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eye_coords = []
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eye_x, eye_y = x + ex, y + ey
                eye_coords.append((eye_x, eye_y))
                cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (255, 255, 0), 2)
                cv2.putText(frame, f"Eye: ({eye_x},{eye_y})", (eye_x, eye_y - 5),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: ({x},{y})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if face_detected_time is None:
                face_detected_time = time.time()
                print(f"[INFO] Face detected. Saving image in 5 seconds...")

            elapsed = time.time() - face_detected_time
            if elapsed >= 5:
                # Save image
                img_filename = f'emp_{emp_id}.jpg'
                txt_filename = f'emp_{emp_id}.txt'
                img_path = os.path.join(EMPLOYEE_FOLDER, img_filename)
                txt_path = os.path.join(EMPLOYEE_FOLDER, txt_filename)

                cv2.imwrite(img_path, frame)
                print(f"[INFO] Saved {img_filename}")

                # Save coordinates
                with open(txt_path, 'w') as f:
                    f.write(f"Face: x={x}, y={y}, w={w}, h={h}\n")
                    for i, (eye_x, eye_y) in enumerate(eye_coords):
                        f.write(f"Eye {i+1}: x={eye_x}, y={eye_y}\n")
                    f.write(f"Timestamp: {datetime.now()}\n")

                face_data['image'] = frame
                face_data['face'] = (x, y, w, h)
                face_data['eyes'] = eye_coords
                saved = True

        cv2.imshow(f"Panel {emp_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return face_data

# Compare face and eye coordinates
def compare_coordinates(data1, data2):
    print("\n=== COORDINATE COMPARISON ===")
    f1 = data1['face']
    f2 = data2['face']
    print(f"Face Difference: dx={abs(f1[0]-f2[0])}, dy={abs(f1[1]-f2[1])}")

    eyes1 = data1['eyes']
    eyes2 = data2['eyes']

    min_eyes = min(len(eyes1), len(eyes2))
    for i in range(min_eyes):
        ex1, ey1 = eyes1[i]
        ex2, ey2 = eyes2[i]
        print(f"Eye {i+1} Diff: dx={abs(ex1 - ex2)}, dy={abs(ey1 - ey2)}")

# Main
cap = cv2.VideoCapture(0)
print("[STEP 1] Capturing first face in 5 seconds...")
data1 = capture_and_save(cap, emp_id=1)

print("\n[STEP 2] Capturing second face in 5 seconds...")
data2 = capture_and_save(cap, emp_id=2)

# Close camera and windows
cap.release()
cv2.destroyAllWindows()

# Combine and display both panels side by side
if data1 and data2:
    img1 = cv2.resize(data1['image'], (400, 400))
    img2 = cv2.resize(data2['image'], (400, 400))
    combined = np.hstack((img1, img2))
    cv2.imshow("Panel 1 (Left) vs Panel 2 (Right)", combined)
    compare_coordinates(data1, data2)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Face data missing. Could not compare.")
