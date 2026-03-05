🚗 Driver Monitoring System

📋 Project Overview
The **Driver Monitoring System** is a real-time computer vision application that monitors driver drowsiness and phone usage. It provides instant alerts using audio beeps and SMS notifications to improve road safety.

This system uses **OpenCV, YOLOv8, and Twilio** to detect:  
- Driver drowsiness (eyes closed for more than 5 seconds)  
- Phone usage while driving  
- Night vision support for low-light conditions  



🎯 Features
- ✅ Real-time face and eye detection using Haar cascades  
- ✅ Drowsiness detection logic with timers and consecutive frame checks  
- ✅ Phone detection using YOLOv8  
- ✅ Audio alert (beep) when drowsiness is detected  
- ✅ SMS alert to a predefined phone number using Twilio  
- ✅ Night vision image enhancement for low-light detection  


 🛠️ Technologies Used
- Python 3.x  
- OpenCV  
- YOLOv8  
- Twilio API  
- Winsound (for audio alerts)  
- Threading (for non-blocking SMS sending) 

  Installation

1. **Clone the repository:**

git clone https://github.com/noorainsiddiqa-pixel/driver-monitoring-system.git
cd driver-monitoring-system

2.Create a virtual environment:

python -m venv .venv
.\.venv\Scripts\activate  # Windows

3.Install dependencies:

pip install -r requirements.txt
If you are on Python 3.14, switch to Python 3.10 or 3.11 as pygame and other libraries may not work on 3.14.


USAGE

Connect your camera.
Run the system:

python driver_monitor_full.py

The camera feed will open.
Alerts will trigger when:
Eyes are closed for too long (drowsiness)
Phone usage is detected
Press ESC to exit the system.

PROJECT STRUCTURE

driver-monitoring-system/
├── driver_monitor_full.py      # Main driver monitoring script
├── driver_monitor.py           # Modular driver monitoring code
├── main.py                     # Testing script
├── test_camera.py              # Camera test
├── find_camera.py              # Camera detection helper
├── yolov8n.pt                  # YOLOv8 model weights
├── Driver Monitoring System.pdf # Full project report
├── REPORT.md                   # Markdown version of report
├── README.md                   # This file
└── requirements.txt            # Python dependencies
