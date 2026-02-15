import cv2, os

class Camera:
    def __init__(self):
        # 1. Look for the Virtual Loopback (created by your .sh script)
        if os.path.exists("/dev/video10"):
            print("Camera: Found Loopback at /dev/video10")
            self.cap = cv2.VideoCapture(10, cv2.CAP_V4L2)
        
        # 2. Fallback to direct CSI/USB Hardware
        elif os.path.exists("/dev/video0"):
            print("Camera: Using hardware at /dev/video0")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
        else:
            print("Camera: No standard video nodes found.")
            self.cap = cv2.VideoCapture(0)

        # Ensure resolution matches your detector logic
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame(self):
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()