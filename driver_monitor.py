import cv2
import time

# -------------------------
# Initialize OpenCV cascade classifiers
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# -------------------------
# OpenCV camera
# -------------------------
cap = cv2.VideoCapture(0)

# EAR threshold & time tracking
DROWSY_TIME_THRESHOLD = 5.0  # 5 seconds
eye_closure_start_time = None
is_eyes_closed = False

# -------------------------

print("🚀 Starting Driver Monitoring System...")
print(f"⏱️ Eyes must be closed for {DROWSY_TIME_THRESHOLD} seconds to trigger drowsiness alert")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received", flush=True)
        break

    print("📸 Frame received", flush=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        print("🙂 Face detected", flush=True)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                # Eyes detected
                eyes_closed = False
                print(f"👁️ Eyes detected: {len(eyes)} eye(s)", flush=True)
            else:
                eyes_closed = len(eyes) < 2
                if eyes_closed:
                    if not is_eyes_closed:
                        # Eyes just closed
                        eye_closure_start_time = time.time()
                        is_eyes_closed = True
                        print("👁️ Eyes appear to be closed", flush=True)
                    
                    closure_duration = time.time() - eye_closure_start_time
                    print(f"Eyes closed for: {closure_duration:.1f}s", flush=True)
                    
                    if closure_duration >= DROWSY_TIME_THRESHOLD:
                        cv2.putText(frame, "😴 DROWSY!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)
                        print(f"⚠️ Drowsiness detected! Eyes closed for {closure_duration:.1f}s", flush=True)
                else:
                    if is_eyes_closed:
                        if eye_closure_start_time:
                            closure_duration = time.time() - eye_closure_start_time
                            print(f"👁️ Eyes opened after {closure_duration:.1f}s", flush=True)
                        is_eyes_closed = False
                        eye_closure_start_time = None
                    
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw eyes rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    else:
        print("😐 No face detected", flush=True)

    cv2.imshow("Driver Monitoring", frame)

    # Press Esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("🚪 Exiting Driver Monitoring", flush=True)
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Driver Monitoring System ended", flush=True)