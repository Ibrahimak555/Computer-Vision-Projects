# People Proximity Alert System

This system uses YOLOv5 for real-time object detection to detect people and estimate their distance from the camera. It provides visual and audio alerts when people are within defined distance in meters.

## Features
- Real-time people detection using YOLOv5
- Distance estimation based on bounding box height
- Alerts when people are within proximity
- Webcam input support

## Setup
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python people_proximity_alert.py
```

## Usage
- Press 'q' to quit the application
- The system will show a window with real-time video feed
- People within proximity range will be highlighted in red with a "Proximity Alert" message and an alert will be sounded

## Notes
- The distance estimation is approximate and may need calibration based on your specific camera
- The system uses the default webcam (0), change the camera index if needed
- The focal length parameter may need adjustment based on your camera's specifications
