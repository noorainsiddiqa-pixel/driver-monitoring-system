import cv2
import mediapipe as mp

print("🚀 Script started...", flush=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera NOT opening", flush=True)
    exit()
else:
   print("✅ Camera opened", flush=True)

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received", flush=True)
        break

    print("📸 Frame received", flush=True)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        print("🙂 Face detected", flush=True)

        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0,255,0), -1)
    else:
        print("😐 No face detected", flush=True)

    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) == 27:
        print("🚪 Exiting loop", flush=True)
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Script ended", flush=True)