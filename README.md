# Driver Monitoring System

A real-time computer vision application that monitors driver drowsiness, phone usage, and fatigue using advanced AI techniques.

## Features

- **Drowsiness Detection**: EAR (Eye Aspect Ratio) based detection with temporal analysis
- **Phone Detection**: YOLOv8-powered cell phone detection while driving
- **Yawning Detection**: Mouth aspect ratio analysis for fatigue monitoring
- **Night Vision**: Automatic low-light enhancement using CLAHE
- **Audio Alerts**: Distinctive beep patterns for different alert types
- **SMS Alerts**: Twilio integration for real-time notifications
- **Visual Overlays**: Real-time status indicators and alert banners

## Installation

### Prerequisites

- Python 3.8+
- Webcam or video capture device
- (Optional) Twilio account for SMS alerts

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/noorainsiddiqa-pixel/driver-monitoring-system.git
   cd driver-monitoring-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If you are on Python 3.14, switch to Python 3.10 or 3.11 as pygame and other libraries may not work on 3.14.

4. **Configure environment (optional - for SMS alerts)**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your Twilio credentials:
   ```
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_PHONE_NUMBER=+1234567890
   ALERT_RECIPIENT_NUMBER=+0987654321
   ```

## Usage

### Basic Usage

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Send test SMS |
| `r` | Reset alert cooldowns |

### Using a Different Camera

```bash
# Use secondary camera
python main.py --camera 1

# Use video file
python main.py --camera video.mp4
```

## Configuration

Edit `.env` or modify `src/config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EAR_THRESHOLD` | 0.28 | Eye closure threshold |
| `PHONE_CONFIDENCE_THRESHOLD` | 0.5 | Phone detection confidence |
| `DROWSINESS_FRAMES` | 15 | Frames to confirm drowsiness (~0.5s) |
| `NIGHT_VISION_THRESHOLD` | 50 | Low-light brightness threshold |
| `ALERT_COOLDOWN` | 30 | Frames between repeated alerts |

## Project Structure

```
driver-monitoring-system/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration and thresholds
│   ├── drowsiness.py        # EAR-based drowsiness detection
│   ├── phone_detect.py      # YOLOv8 phone detection
│   ├── alerts.py            # Audio and SMS alerts
│   └── utils.py             # Image processing utilities
├── main.py                  # Main entry point
├── test_camera.py           # Camera test utility
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## How It Works

### Drowsiness Detection

1. MediaPipe Face Mesh detects 468 facial landmarks
2. Eye Aspect Ratio (EAR) is calculated for both eyes
3. EAR history is tracked over ~15 frames
4. Drowsiness is triggered when average EAR < threshold for sustained period

### Phone Detection

1. YOLOv8 (nano model) runs inference on each frame
2. Detects "cell phone" class from COCO dataset (class 67)
3. Temporal smoothing ensures consistent detection
4. Visual bounding box overlay with confidence score

### Night Vision

1. Average frame brightness is calculated
2. If below threshold, CLAHE enhancement is applied
3. Gamma correction brightens the image further
4. Green tint overlay indicates night mode active

## Troubleshooting

### Camera Not Opening

- Check if camera is connected
- Try a different camera index: `python main.py --camera 1`
- Ensure no other application is using the camera

### SMS Alerts Not Working

- Verify `.env` file exists with correct credentials
- Check Twilio account balance
- Ensure phone numbers are in E.164 format (+1234567890)

### Low FPS

- Reduce camera resolution in `config.py`
- Use YOLOv8 nano model (default) for faster inference
- Close other applications using GPU/CPU

### "No module named 'src'" Error

- Ensure you're running from the project root directory
- Check that `src/__init__.py` exists

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8+ | Computer vision |
| mediapipe | 0.10+ | Face mesh detection |
| ultralytics | 8.0+ | YOLOv8 phone detection |
| twilio | 8.0+ | SMS notifications |
| pygame | 2.5+ | Audio alerts |
| numpy | 1.24+ | Numerical operations |
| python-dotenv | 1.0+ | Environment management |

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh detection
- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [Twilio](https://twilio.com/) for SMS notifications
