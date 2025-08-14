import cv2
from ultralytics import YOLO
from pygame import mixer

class ProximityAlert:
    def __init__(self):
        # Initialize YOLOv5s model
        self.model = YOLO('yolov5su.pt')
        
        # Initialize camera
        self.vid = cv2.VideoCapture(0)
        
        # Audio setup for alert sound
        mixer.init()
        mixer.music.load("alert.wav")
        
        # Distance parameters
        self.proximity_threshold = 3.0  # meters
        self.human_height = 1.7     # meters (average human height)
        self.focal_length = 550     # pixels (adjust based on camera)
        self.detected_frames = 0
        
    def calculate_distance(self, bbox_height):
        return (self.human_height * self.focal_length) / bbox_height
    
    def play_beep(self):
	# Play 'music.wav' every 40 frames 
        if self.detected_frames % 40 == 0:
            mixer.music.play()
    
    def run(self):
        while True:
            ret, frame = self.vid.read()
            
            if not ret:
                break

            # Process frame with YOLO
            results = self.model(frame)
            
            # Get detections
            boxes = results[0].boxes
            
            # Draw detections and calculate distances
            for box in boxes:
                class_id = box.cls.item()
                if class_id == 0:  # 0 is the class ID for 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox_height = y2 - y1
                    
                    # Calculate distance
                    distance = self.calculate_distance(bbox_height)
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Default BB_colour is Green
                    text = f"{distance:.1f}m"
                    
                    # If person is less than proximity threshold
                    if distance <= self.proximity_threshold:
                        color = (0, 0, 255)  # Red for alert
                        text = f"!Proximity Alert! {distance:.1f}m"
                        self.play_beep()  # Play alert sound
                        self.detected_frames += 1  # Used for playing alert sound from 'music.wav'
                    else:
                        self.detected_frames = 0
                    
                    # Draw box and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display the frame
            cv2.imshow("People Proximity Alert - Press 'q' to exit", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    def close_window(self):
        self.vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    alert_system = ProximityAlert()

    try:
        alert_system.run()
    except KeyboardInterrupt:
        pass

    alert_system.close_window()