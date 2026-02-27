import time
import cv2
from ultralytics import YOLO

class ShrimpDetector:
    def __init__(self, model_path="models/best_ncnn_model", conf=0.25, imgsz=640):
        print(f"Loading NCNN model from: {model_path}")
        try:
            self.model = YOLO(model_path, task="detect")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load model: {e}")
            self.model = None

        self.conf = conf
        self.imgsz = imgsz
        self.total_count = 0
        self.counted_ids = set()
        self.line_ratio = 0.3 

    def detect(self, frame, draw=True):
        if self.model is None:
            return self.total_count, frame

        h, w = frame.shape[:2]
        
        # We REMOVED the cvtColor(GRAY) line here to fix the "axes don't match" error
        start = time.time()
        
        results = self.model.track(
            frame,  # Pass the color frame directly
            imgsz=self.imgsz,
            conf=self.conf,
            device="cpu",
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]

        inference_time = (time.time() - start) * 1000
        line_x = int(w * self.line_ratio)
        vis = frame.copy()

        cv2.line(vis, (line_x, 0), (line_x, h), (0, 255, 255), 2)

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)

                if cx > line_x and track_id not in self.counted_ids:
                    self.total_count += 1
                    self.counted_ids.add(track_id)

                if draw:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"ID:{int(track_id)}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps = int(1000 / inference_time) if inference_time > 0 else 0
        cv2.putText(vis, f"FPS: {fps} | Count: {self.total_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return self.total_count, vis

    def reset_total_count(self):
        self.total_count = 0
        self.counted_ids = set()