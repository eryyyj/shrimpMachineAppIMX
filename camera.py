import cv2, os

class Camera:
    def __init__(self):
        self.index = 10
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            print(f"FAILED to open /dev/video{self.index}")
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        return None

    def release(self):
        if self.cap:
            self.cap.release()