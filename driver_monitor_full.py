import cv2
from ultralytics import YOLO
import winsound
import time
from twilio.rest import Client
import threading

# -------------------------
# Twilio SMS Setup
# -------------------------
TWILIO_SID = "AC99f46166f97a395c286cc9b7ab0a6b4d"
TWILIO_AUTH_TOKEN = "7a4701a5e84732a995df445f9ed11a70"
TWILIO_PHONE = "+13168679169"  # Twilio number
MY_PHONE = "+917676376189"    # Your mobile number

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_sms_alert(message):
    try:
        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=MY_PHONE
        )
        print(f"📩 SMS sent successfully! SID: {msg.sid}")
    except Exception as e:
        print(f"❌ SMS sending failed: {str(e)}")
        import traceback
        traceback.print_exc()

def trigger_alert():
    """Trigger sound and SMS alert in a separate thread"""
    try:
        # Play loud beep multiple times
        print("🔔 ALERT: Multiple beeps!")
        for i in range(3):
            winsound.Beep(1500, 500)  # High pitch, 0.5 sec
            time.sleep(0.2)
        print("✅ Beep completed")
    except Exception as e:
        print(f"❌ Beep failed: {e}")
    
    # Send SMS in background thread
    sms_thread = threading.Thread(target=send_sms_alert, args=("⚠️ DROWSINESS ALERT! Eyes closed for too long!",))
    sms_thread.daemon = True
    sms_thread.start()

# -------------------------
# Initialize OpenCV cascade classifiers
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# -------------------------
# Initialize YOLO for phone detection
# -------------------------
yolo_model = YOLO("yolov8n.pt")  # small YOLO model

# -------------------------
# Camera setup
# -------------------------
cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.25
COUNTER = 0
CONSEC_FRAMES = 3  # Very sensitive - detect within 3 frames
drowsy_alerted = False  # Debounce for alarm/SMS

print("🚀 Starting Full Driver Monitoring System...")

# -------------------------
# Function to compute EAR
# -------------------------

# -------------------------
# Main Loop
# -------------------------
drowsy_alerted = False
eye_closure_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    # -------------------------
    # Night vision enhancement
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

    # -------------------------
    # Face & eye detection
    # -------------------------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    drowsy_detected = False

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            print(f"Eyes detected: {len(eyes)}, COUNTER: {COUNTER}")
            
            if len(eyes) >= 1:
                # At least one eye detected - eyes are likely open
                COUNTER = 0
                eye_closure_start_time = None
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                # No eyes detected - eyes might be closed
                if eye_closure_start_time is None:
                    eye_closure_start_time = time.time()
                
                closure_duration = time.time() - eye_closure_start_time
                COUNTER += 1
                cv2.putText(frame, f"😴 DROWSY! {closure_duration:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                print(f"⚠️ Eyes closed! COUNTER: {COUNTER}/{CONSEC_FRAMES}, Duration: {closure_duration:.1f}s")
                
                # Trigger if counter reaches threshold OR if closed for 5 seconds
                if COUNTER >= CONSEC_FRAMES or closure_duration >= 5.0:
                    drowsy_detected = True
                    print(f"🚨 DROWSINESS TRIGGERED! (COUNTER={COUNTER}, Duration={closure_duration:.1f}s)")

    # -------------------------
    # Drowsiness alert (alarm + SMS) - debounced
    # -------------------------
    if drowsy_detected and not drowsy_alerted:
        print("🚨 DROWSINESS DETECTED - TRIGGERING ALERT!")
        trigger_alert()
        drowsy_alerted = True
        eye_closure_start_time = None  # Reset timer
        COUNTER = 0  # Reset counter
    elif not drowsy_detected:
        drowsy_alerted = False

    # -------------------------
    # Phone detection with YOLO
    # -------------------------
    yolo_results = yolo_model(frame)
    for r in yolo_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            if label in ["cell phone", "phone"]:
                cv2.putText(frame, "📵 PHONE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                print("📵 Phone detected!")

    # -------------------------
    # Display
    # -------------------------
    cv2.imshow("Driver Monitoring Full", frame)

    # Press Esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("🚪 Exiting Full Driver Monitoring System")
        break

cap.release()
cv2.destroyAllWindows()