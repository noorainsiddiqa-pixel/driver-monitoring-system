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
