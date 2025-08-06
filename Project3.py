import cv2
import os
import numpy as np
import time
from datetime import datetime
import re

# Folder setup
COMPARE_FOLDER = 'Compare'
os.makedirs(COMPARE_FOLDER, exist_ok=True)

# Get next comparison ID
def get_next_comparison_id():
    existing = os.listdir(COMPARE_FOLDER)
    ids = []
    pattern = re.compile(r'comparison(\d+)')
    for file in existing:
        match = pattern.match(file)
        if match:
            ids.append(int(match.group(1)))
    return max(ids, default=0) + 1

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture and save image/coordinates
def capture_and_save(cap, comparison_id, tag):
    face_data = {}
    face_detected_time = None
    saved = False

    while not saved:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eye_coords = []
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eye_x, eye_y = x + ex, y + ey
                eye_coords.append((eye_x, eye_y))
                cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (255, 255, 0), 2)
                cv2.putText(frame, f"Eye: ({eye_x},{eye_y})", (eye_x, eye_y - 5),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: ({x},{y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if face_detected_time is None:
                face_detected_time = time.time()
                print(f"[INFO] Face detected. Saving image in 5 seconds...")

            elapsed = time.time() - face_detected_time
            if elapsed >= 5:
                # Save image and coordinates
                img_name = f'comparison{comparison_id}{tag}.jpg'
                txt_name = f'comparison{comparison_id}{tag}.txt'
                img_path = os.path.join(COMPARE_FOLDER, img_name)
                txt_path = os.path.join(COMPARE_FOLDER, txt_name)

                cv2.imwrite(img_path, frame)
                print(f"[INFO] Saved {img_name}")

                with open(txt_path, 'w') as f:
                    f.write(f"Face: x={x}, y={y}, w={w}, h={h}\n")
                    for i, (eye_x, eye_y) in enumerate(eye_coords):
                        f.write(f"Eye {i + 1}: x={eye_x}, y={eye_y}\n")
                    f.write(f"Timestamp: {datetime.now()}\n")

                face_data['image'] = frame
                face_data['face'] = (x, y, w, h)
                face_data['eyes'] = eye_coords
                saved = True

        cv2.imshow(f"Panel {tag.upper()}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return face_data

# Compare coordinates
def compare_coordinates(data1, data2):
    print("\n=== COORDINATE COMPARISON ===")
    f1 = data1['face']
    f2 = data2['face']
    dx_face = abs(f1[0] - f2[0])
    dy_face = abs(f1[1] - f2[1])
    print(f"Face Difference: dx={dx_face}, dy={dy_face}")

    eyes1 = data1['eyes']
    eyes2 = data2['eyes']
    eye_diffs = []

    min_eyes = min(len(eyes1), len(eyes2))
    for i in range(min_eyes):
        ex1, ey1 = eyes1[i]
        ex2, ey2 = eyes2[i]
        dx_eye = abs(ex1 - ex2)
        dy_eye = abs(ey1 - ey2)
        eye_diffs.append((dx_eye, dy_eye))
        print(f"Eye {i + 1} Diff: dx={dx_eye}, dy={dy_eye}")

    return dx_face, dy_face, eye_diffs

# === Main ===
comparison_id = get_next_comparison_id()
cap = cv2.VideoCapture(0)

print(f"[STEP 1] Capturing face A for comparison {comparison_id}...")
data1 = capture_and_save(cap, comparison_id, 'a')

print(f"\n[STEP 2] Capturing face B for comparison {comparison_id}...")
data2 = capture_and_save(cap, comparison_id, 'b')

cap.release()
cv2.destroyAllWindows()

# Show result and save side-by-side + difference
if data1 and data2:
    img1 = cv2.resize(data1['image'], (400, 400))
    img2 = cv2.resize(data2['image'], (400, 400))
    combined = np.hstack((img1, img2))
    cv2.imshow("Comparison", combined)

    dx_face, dy_face, eye_diffs = compare_coordinates(data1, data2)

    img_path = os.path.join(COMPARE_FOLDER, f"comparison{comparison_id}.jpg")
    txt_path = os.path.join(COMPARE_FOLDER, f"comparison{comparison_id}.txt")

    cv2.imwrite(img_path, combined)
    print(f"[INFO] Saved combined image as {img_path}")

    with open(txt_path, 'w') as f:
        f.write("=== COORDINATE COMPARISON ===\n")
        f.write(f"Face Difference: dx={dx_face}, dy={dy_face}\n")
        for i, (dx_eye, dy_eye) in enumerate(eye_diffs):
            f.write(f"Eye {i+1} Diff: dx={dx_eye}, dy={dy_eye}\n")
        f.write(f"\nTimestamp: {datetime.now()}\n")
    print(f"[INFO] Saved comparison details as {txt_path}")

    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not complete comparison – missing face data.")
