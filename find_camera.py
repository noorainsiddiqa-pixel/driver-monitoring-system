import cv2

print("🔍 Searching for available camera devices...\n")

for i in range(10):  # Try camera indices 0-9
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        ret, frame = cap.read()
        if ret:
            print(f"   ✓ Can read frames from camera {i}")
            print(f"   Frame size: {frame.shape}")
        else:
            print(f"   ✗ Cannot read frames from camera {i}")
        cap.release()
    else:
        break

if i == 0:
    print("❌ No camera devices found!")
else:
    print(f"\n✅ Search complete. Tried indices 0-{i-1}")
