import sys
import math
import time
import threading
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# --- Global Configurations (Replaces argparse) ---
CONFIG = {
    "threshold": 0.55,
    "iou": 0.65,
    "max_detections": 10,
    "bbox_normalization": False,
    "bbox_order": "yx",
    "preserve_aspect_ratio": False
}

# --- Global Variables ---
picam2 = None
imx500 = None
intrinsics = None
is_running = False

last_results = None
latest_frame = None

# Tracking Variables
tracked_objects = {}
next_object_id = 0
total_shrimp_count = 0
MAX_DISTANCE = 80 
MAX_DISAPPEARED = 100

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    global last_results
    bbox_normalization = CONFIG["bbox_normalization"]
    bbox_order = CONFIG["bbox_order"]
    threshold = CONFIG["threshold"]
    iou = CONFIG["iou"]
    max_detections = CONFIG["max_detections"]

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_results

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_results = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_results

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if hasattr(intrinsics, 'ignore_dash_labels') and intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(request, stream="main"):
    """Draw the detections and save the frame for the UI."""
    global tracked_objects, next_object_id, total_shrimp_count, latest_frame

    # 1. Immediately extract and copy the frame to avoid read-only errors
    with MappedArray(request, stream) as m:
        frame = m.array.copy()

    detections = last_results
    if detections is None:
        latest_frame = frame
        return

    height, width = frame.shape[:2]
    split_x = int(width * 0.70)

    # 2. Draw on the COPIED 'frame', NOT the read-only 'm.array'
    cv2.line(frame, (split_x, 0), (split_x, height), (255, 0, 0, 255), 2) 
    cv2.putText(frame, "Detection Area", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1)
    cv2.putText(frame, "Count Area", (split_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1)

    current_centroids = []

    for detection in detections:
        x, y, w, h = detection.box
        cx, cy = int(x + w/2), int(y + h/2)
        current_centroids.append((cx, cy, x, y, w, h))

        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0, 255), thickness=1)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255, 255), -1)
# 3. Robust Centroid Tracking Logic
    if len(current_centroids) == 0:
        for obj_id in list(tracked_objects.keys()):
            tracked_objects[obj_id]['disappeared'] += 1
            if tracked_objects[obj_id]['disappeared'] > MAX_DISAPPEARED:
                del tracked_objects[obj_id]
    else:
        if len(tracked_objects) == 0:
            for cx, cy, x, y, w, h in current_centroids:
                tracked_objects[next_object_id] = {'centroid': (cx, cy), 'counted': False, 'disappeared': 0}
                if cx > split_x:
                    tracked_objects[next_object_id]['counted'] = True
                next_object_id += 1
        else:
            used_centroids = set()
            used_ids = set()
            
            distances = []
            for i, (cx, cy, x, y, w, h) in enumerate(current_centroids):
                for obj_id, data in tracked_objects.items():
                    prev_cx, prev_cy = data['centroid']
                    dist = math.hypot(cx - prev_cx, cy - prev_cy)
                    if dist <= MAX_DISTANCE:
                        distances.append((dist, obj_id, i))
                        
            distances.sort(key=lambda item: item[0])

            for dist, obj_id, i in distances:
                if obj_id in used_ids or i in used_centroids:
                    continue
                    
                used_ids.add(obj_id)
                used_centroids.add(i)
                
                cx, cy = current_centroids[i][0], current_centroids[i][1]
                prev_cx = tracked_objects[obj_id]['centroid'][0]
                
                tracked_objects[obj_id]['centroid'] = (cx, cy)
                tracked_objects[obj_id]['disappeared'] = 0 
                
                # Count happens here (Crossing left to right)
                if prev_cx <= split_x and cx > split_x and not tracked_objects[obj_id]['counted']:
                    total_shrimp_count += 1
                    tracked_objects[obj_id]['counted'] = True
            
            for obj_id in list(tracked_objects.keys()):
                if obj_id not in used_ids:
                    tracked_objects[obj_id]['disappeared'] += 1
                    if tracked_objects[obj_id]['disappeared'] > MAX_DISAPPEARED:
                        del tracked_objects[obj_id]
                        
            for i, (cx, cy, x, y, w, h) in enumerate(current_centroids):
                if i not in used_centroids:
                    tracked_objects[next_object_id] = {'centroid': (cx, cy), 'counted': False, 'disappeared': 0}
                    if cx > split_x:
                        tracked_objects[next_object_id]['counted'] = True
                    next_object_id += 1
            
    # Export the modified frame to our global variable so the UI can grab it
    latest_frame = frame

# --- API FUNCTIONS FOR THE UI ---
def _metadata_loop():
    """Background thread to continuously capture AI inference metadata."""
    global is_running
    while is_running:
        try:
            meta = picam2.capture_metadata()
            if meta:
                parse_detections(meta)
        except Exception as e:
            time.sleep(0.01)

def init_camera(model_path):
    """Initializes the camera, loads the model, and starts the background streams."""
    global picam2, imx500, intrinsics, is_running

    if picam2 is not None:
        return # Camera already running

    imx500 = IMX500(model_path)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"

    # Default Labels
    try:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    except Exception:
        intrinsics.labels = []
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    
    # Configure stream to output RGB formats required by PyQt
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}, 
        controls={"FrameRate": intrinsics.inference_rate}, 
        buffer_count=12
    )

    picam2.start(config, show_preview=False) # We handle preview in the UI now
    picam2.set_controls({"ScalerCrop": (0, 500, 4056, 2040)})
    
    if CONFIG["preserve_aspect_ratio"]:
        imx500.set_auto_aspect_ratio()

    # Hook up the callback for drawing
    picam2.pre_callback = draw_detections
    
    # Start the background polling thread
    is_running = True
    threading.Thread(target=_metadata_loop, daemon=True).start()
    print("AI Camera Initialized and Running.")

def get_latest_frame():
    """Returns the most recent annotated frame and the shrimp count."""
    global latest_frame, total_shrimp_count
    return latest_frame, total_shrimp_count

def reset_count():
    """Resets the tracking algorithms and shrimp count."""
    global total_shrimp_count, tracked_objects, next_object_id
    total_shrimp_count = 0
    tracked_objects.clear()
    next_object_id = 0

def stop_camera():
    """Safely shuts down the AI hardware."""
    global picam2, is_running
    is_running = False
    if picam2 is not None:
        picam2.stop()
        picam2.close()
        picam2 = None
        print("AI Camera Stopped.")