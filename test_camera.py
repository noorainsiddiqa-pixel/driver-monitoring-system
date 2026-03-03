import cv2

print("🚀 Testing camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera NOT opening")
    exit()
else:
    print("✅ Camera opened")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        print("🚪 Exiting camera test")
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera test ended")