import time
import os
import cv2
from ultralytics import YOLO
import numpy as np

class ShrimpDetector:
    def __init__(self, model_path="models/YOLOShrimpV1.1_ncnn_model", conf_thresh=0.25, imgsz=416):
        """
        Initialize NCNN model.
        CRITICAL: imgsz must match the size used during 'yolo export'.
        """
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz 

        try:
            print(f"?? Loading NCNN model from: {model_path}...")
            # Load the model
            self.model = YOLO(model_path, task='detect')
            print(f"? Successfully loaded NCNN model!")
        except Exception as e:
            print(f"? Failed to load model: {e}")
            self.model = None

        # --- Tracking Setup ---
        self.total_count = 0
        self.count_log_file = "shrimp_count.txt"
        self.load_count()

        # Counting line X-coordinate (will be set in detect loop based on frame width)
        self.counting_line_x = 0  
        self.line_ratio = 0.3     # Line at 30% of the screen width
        
        # Tracking State
        self.active_tracks = {}   # {id: [cx, cy, frames_unseen]}
        self.next_track_id = 0
        self.counted_track_ids = set()
        
        # Tracking Tunables
        self.max_distance = 100       # Adjusted for 720p resolution
        self.max_disappeared_frames = 20

    def load_count(self):
        try:
            if os.path.exists(self.count_log_file):
                with open(self.count_log_file, 'r') as f:
                    self.total_count = int(f.read())
        except:
            self.total_count = 0

    def save_count(self):
        with open(self.count_log_file, 'w') as f:
            f.write(str(self.total_count))

    def reset_total_count(self):
        self.total_count = 0
        self.counted_track_ids.clear()
        self.active_tracks.clear()
        self.next_track_id = 0
        self.save_count()

    def _update_tracker(self, detections):
        """
        Updates tracks. Detections must be in (x1, y1, x2, y2) ORIGINAL frame scale.
        """
        current_centers = []
        for (x1, y1, x2, y2) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_centers.append((cx, cy))

        if not current_centers:
            for tid in list(self.active_tracks.keys()):
                self.active_tracks[tid][2] += 1
            return

        if not self.active_tracks:
            for (cx, cy) in current_centers:
                self.active_tracks[self.next_track_id] = [cx, cy, 0]
                self.next_track_id += 1
            return

        # Distance Matrix Logic
        track_ids = list(self.active_tracks.keys())
        track_positions = np.array([self.active_tracks[tid][:2] for tid in track_ids])
        dist_matrix = np.linalg.norm(track_positions[:, np.newaxis] - current_centers, axis=2)
        
        matched_track_ids = set()
        unmatched_center_indices = set(range(len(current_centers)))
        
        for i, tid in enumerate(track_ids):
            if dist_matrix.shape[1] == 0: break
            min_dist_idx = np.argmin(dist_matrix[i, :])
            if dist_matrix[i, min_dist_idx] < self.max_distance:
                # Match found
                new_cx, new_cy = current_centers[min_dist_idx]
                old_cx, old_cy, _ = self.active_tracks[tid]
                
                # --- Counting Logic (Left to Right crossing) ---
                if old_cx < self.counting_line_x and new_cx >= self.counting_line_x:
                    if tid not in self.counted_track_ids:
                        self.total_count += 1
                        self.counted_track_ids.add(tid)
                        print(f"?? Counted! Total: {self.total_count}")
                
                self.active_tracks[tid] = [new_cx, new_cy, 0]
                matched_track_ids.add(tid)
                
                if min_dist_idx in unmatched_center_indices:
                    unmatched_center_indices.remove(min_dist_idx)
                dist_matrix[:, min_dist_idx] = np.inf 

        # Handle unmatched/lost
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.active_tracks[tid][2] += 1
                if self.active_tracks[tid][2] > self.max_disappeared_frames:
                    del self.active_tracks[tid]
                    self.counted_track_ids.discard(tid)

        for idx in unmatched_center_indices:
            cx, cy = current_centers[idx]
            self.active_tracks[self.next_track_id] = [cx, cy, 0]
            self.next_track_id += 1

    def detect(self, frame, draw=True):
        if self.model is None:
            return self.total_count, frame

        h, w = frame.shape[:2]
        self.counting_line_x = int(w * self.line_ratio)

        start_time = time.time()
        
        # -------------------------------------------------------------------------
        # CRITICAL FIX: Pass imgsz=416 here!
        # This tells Ultralytics to resize the 1280x720 frame to 416x416 BEFORE inference
        # and then automatically scale the boxes BACK to 1280x720.
        # -------------------------------------------------------------------------
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf_thresh, device='cpu', verbose=False)
        
        inference_time = (time.time() - start_time) * 1000

        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Ultralytics automatically returns these in Original Frame (1280x720) coords
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append((x1, y1, x2, y2))

        # Update Tracker
        self._update_tracker(detections)

        # Visualization
        vis_frame = frame.copy()
        if draw:
            # Draw Counting Line
            cv2.line(vis_frame, (self.counting_line_x, 0), (self.counting_line_x, h), (0, 255, 255), 2)

            # Draw Boxes
            for (x1, y1, x2, y2) in detections:
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw Tracks
            # Draw Tracks (ONLY if they were seen in THIS frame)
            for tid, (cx, cy, unseen_frames) in self.active_tracks.items():
                if unseen_frames == 0:  # <--- ONLY DRAW IF SEEN RIGHT NOW
                    if 0 <= cx < w and 0 <= cy < h:
                        color = (0, 0, 255) if tid in self.counted_track_ids else (255, 0, 0)
                        cv2.circle(vis_frame, (cx, cy), 5, color, -1)
                        cv2.putText(vis_frame, str(tid), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw Stats
            fps = int(1000 / inference_time) if inference_time > 0 else 0
            stats = f"FPS: {fps} | Count: {self.total_count}"
            
            # Text background
            (text_w, text_h), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(vis_frame, (10, 10), (10 + text_w + 10, 10 + text_h + 20), (0,0,0), -1)
            cv2.putText(vis_frame, stats, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return self.total_count, cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------
# MAIN EXECUTION BLOCK (Run this to test)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure this points to your NCNN FOLDER
    detector = ShrimpDetector("models/YOLOShrimpV1.1_ncnn_model", conf_thresh=0.25, imgsz=416)
    
    # Use the camera path we found earlier
    cap = cv2.VideoCapture(10) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Could not open video device 10. Trying 0...")
        cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            count, vis_frame_rgb = detector.detect(frame, draw=True)
            
            # Convert RGB back to BGR for OpenCV display
            cv2.imshow("NCNN Shrimp Detector", cv2.cvtColor(vis_frame_rgb, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) == 27: # ESC to quit
                break
    finally:
        detector.save_count()
        cap.release()
        cv2.destroyAllWindows()