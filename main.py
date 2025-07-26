import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
import threading

# Load known faces
known_face_encodings = []
known_ids = []
known_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join("known_faces", filename)
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            parts = os.path.splitext(filename)[0].split('_')
            known_ids.append(parts[0])
            known_names.append(parts[1])

# Load or create attendance DataFrame
if os.path.exists("attendance_log.xlsx"):
    df = pd.read_excel("attendance_log.xlsx")
else:
    df = pd.DataFrame(columns=["ID", "Name", "In Time", "Out Time"])

lock = threading.Lock()

def log_terminal_and_excel(student_id, name, in_time=None, out_time=None):
    global df

    with lock:
        if in_time:
            print(f"[IN] {student_id} - {name} at {in_time}")
            df = pd.concat([df, pd.DataFrame([[student_id, name, in_time, None]], columns=df.columns)], ignore_index=True)
        elif out_time:
            print(f"[OUT] {student_id} - {name} at {out_time}")
            # Update corresponding in-time entry
            idx = df[(df["ID"] == student_id) & (df["Out Time"].isna())].last_valid_index()
            if idx is not None:
                df.at[idx, "Out Time"] = out_time

        # Save to Excel
        df.to_excel("attendance_log.xlsx", index=False)

def handle_camera(cam_index, cam_type):
    print(f"[INFO] Starting Camera {cam_index} for {cam_type.upper()} tracking...")
    cap = cv2.VideoCapture(cam_index)

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                student_id = known_ids[best_match_index]
                name = known_names[best_match_index]

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if cam_type == "in":
                    existing = df[(df["ID"] == student_id) & (df["Out Time"].isna())]
                    if existing.empty:
                        log_terminal_and_excel(student_id, name, in_time=now)
                elif cam_type == "out":
                    existing = df[(df["ID"] == student_id) & (df["Out Time"].isna())]
                    if not existing.empty:
                        log_terminal_and_excel(student_id, name, out_time=now)

                top, right, bottom, left = face_location
                color = (0, 255, 0) if cam_type == "in" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, f"{student_id} - {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow(f"Camera {cam_index} - {cam_type.upper()}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {cam_index} - {cam_type.upper()}")

# Start both cameras in threads
t1 = threading.Thread(target=handle_camera, args=(0, "in"))
t2 = threading.Thread(target=handle_camera, args=(1, "out"))

t1.start()
t2.start()

t1.join()
t2.join()

print("✅ Attendance session completed.")
