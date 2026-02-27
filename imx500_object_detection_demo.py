import argparse
import sys
import math # Added for centroid distance calculations
from functools import lru_cache

import cv2

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []

# --- New Global Variables for Tracking and Counting ---
tracked_objects = {}
next_object_id = 0
total_shrimp_count = 0
# The maximum pixel distance a shrimp can move between frames to still be considered the same shrimp. 
# You may need to tune this based on your camera's FPS and the water flow speed.
MAX_DISTANCE = 80 
MAX_DISAPPEARED = 100
# ----------------------------------------------------


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
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

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections

@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    global tracked_objects, next_object_id, total_shrimp_count

    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    
    with MappedArray(request, stream) as m:
        # Get frame dimensions to calculate the 70/30 split
        height, width = m.array.shape[:2]
        split_x = int(width * 0.70)

        # 1. Draw the division line and zone labels
        cv2.line(m.array, (split_x, 0), (split_x, height), (255, 0, 0, 255), 2) 
        cv2.putText(m.array, "Detection Area", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1)
        cv2.putText(m.array, "Count Area", (split_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1)

        current_centroids = []

        # 2. Extract boxes, calculate centroids, and draw them
        for detection in detections:
            x, y, w, h = detection.box
            cx, cy = int(x + w/2), int(y + h/2)
            current_centroids.append((cx, cy, x, y, w, h))

            cv2.rectangle(m.array, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0, 255), thickness=1)
            cv2.circle(m.array, (cx, cy), 3, (0, 0, 255, 255), -1)
            cv2.circle(m.array, (cx, cy), 3, (0, 0, 255, 255), -1)

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
                        # FIX: Mark as counted so it doesn't trigger later, but DO NOT add to total count.
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
                    
                    # This is the ONLY place a count should happen (Crossing from left to right)
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
                            # FIX: Mark as counted so it doesn't trigger later, but DO NOT add to total count.
                            tracked_objects[next_object_id]['counted'] = True
                        next_object_id += 1
                
 # 4. Display the Live Count
        count_text = f"Live Count: {total_shrimp_count}"
        cv2.putText(m.array, count_text, (split_x + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255, 255), 2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    picam2.set_controls({"ScalerCrop": (0, 500, 4056, 2040)})

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    while True:
        last_results = parse_detections(picam2.capture_metadata())