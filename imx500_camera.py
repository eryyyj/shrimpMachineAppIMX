"""
IMX500 Raspberry Pi AI Camera integration for shrimp counting.
Provides hardware-accelerated object detection via picamera2.
"""

import math
import time
import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# Tracking constants - tuned for fast-moving shrimp and detection gaps
MAX_DISTANCE = 140          # Max pixel movement between frames to match same shrimp
MAX_DISTANCE_REAPPEAR = 220 # Larger threshold when matching after detection gap
MAX_DISAPPEARED = 100       # Frames before removing lost tracks
NEAR_LINE_PX = 100          # New object in count area within this of line = likely crossed during gap

# Area split (Detection Area on left, Count Area on right)
DETECTION_AREA_RATIO = 0.50  # 50% detection area, 50% count area

# De-duplication of counts near the Count Area line
RECENT_COUNT_TIME = 1.5     # Seconds within which repeated counts near same spot are treated as duplicates
RECENT_COUNT_DISTANCE = 80  # Max pixel distance for a duplicate count relative to last crossing

# Exposure / zoom crop: high shutter speed reduces motion blur; AnalogueGain compensates for darkness
EXPOSURE_TIME_US = 7500     # 7.5ms shutter (5000-10000 for 30fps)
ANALOGUE_GAIN = 5.0         # Compensate for short exposure (tune for your lighting)
# Zoom-in crop to remove unused top/bottom areas (x, y, width, height)
SCALER_CROP = (0, 400, 4056, 2200)

# Default config (custom shrimp detection model)
DEFAULT_MODEL = "/home/hiponpd/my_custom_model/network.rpk"
DEFAULT_LABELS = "/home/hiponpd/Downloads/best_imx_model/labels.txt"
DEFAULT_FPS = 30
DEFAULT_THRESHOLD = 0.10
DEFAULT_IOU = 0.65
DEFAULT_MAX_DETECTIONS = 10


class Detection:
    """Detection with bounding box converted to ISP output coordinates."""

    def __init__(self, coords, category, conf, metadata, imx500, picam2):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


# Singleton: only one IMX500 camera instance app-wide to prevent hardware conflicts
_camera_instance = None
_camera_lock = None

try:
    import threading
    _camera_lock = threading.Lock()
except ImportError:
    pass


def get_imx500_camera(**kwargs):
    """Get or create the singleton IMX500Camera instance."""
    global _camera_instance
    if _camera_lock:
        with _camera_lock:
            if _camera_instance is None:
                _camera_instance = IMX500Camera(**kwargs)
            return _camera_instance
    if _camera_instance is None:
        _camera_instance = IMX500Camera(**kwargs)
    return _camera_instance


def close_imx500_camera():
    """Release camera resources when leaving the biomass page."""
    global _camera_instance
    if _camera_lock:
        with _camera_lock:
            if _camera_instance:
                _camera_instance.close()
                _camera_instance = None
    else:
        if _camera_instance:
            _camera_instance.close()
            _camera_instance = None


class IMX500Camera:
    """
    IMX500 camera with hardware-accelerated object detection.
    Runs inference on the NPU; centroid tracking for shrimp counting.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        labels_path: str | None = DEFAULT_LABELS,
        fps: int = DEFAULT_FPS,
        threshold: float = DEFAULT_THRESHOLD,
        iou: float = DEFAULT_IOU,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        bbox_normalization: bool = True,
        ignore_dash_labels: bool = True,
        bbox_order: str = "xy",
    ):
        self.imx500 = None
        self.picam2 = None
        self.intrinsics = None
        self.config = {
            "model_path": model_path,
            "labels_path": labels_path,
            "fps": fps,
            "threshold": threshold,
            "iou": iou,
            "max_detections": max_detections,
        }
        self.last_detections = []
        self.last_results = None
        self.tracked_objects = {}
        self.next_object_id = 0
        self.total_shrimp_count = 0
        self.recent_counts = []  # (timestamp, cx, cy) for recent line-cross events
        self._fw_uploaded = False  # Ensure network firmware is uploaded only once per instance

        # IMX500 must be created before Picamera2
        self.imx500 = IMX500(model_path)
        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"
        elif self.intrinsics.task != "object detection":
            raise ValueError("Network is not an object detection task")

        # Override intrinsics from config (matches demo: --bbox-normalization --ignore-dash-labels --bbox-order xy --fps 30)
        self.intrinsics.threshold = threshold
        self.intrinsics.iou = iou
        self.intrinsics.max_detections = max_detections
        self.intrinsics.bbox_normalization = bbox_normalization
        self.intrinsics.ignore_dash_labels = ignore_dash_labels
        self.intrinsics.bbox_order = bbox_order
        self.intrinsics.inference_rate = fps
        self.intrinsics.fps = fps

        if labels_path:
            with open(labels_path) as f:
                self.intrinsics.labels = f.read().splitlines()

        # Fallback labels if not provided
        if self.intrinsics.labels is None:
            try:
                with open("assets/coco_labels.txt") as f:
                    self.intrinsics.labels = f.read().splitlines()
            except FileNotFoundError:
                self.intrinsics.labels = ["object"]

        self.intrinsics.update_with_defaults()

        # Create Picamera2; will be recreated in start() after close() releases it.
        self.picam2 = Picamera2(self.imx500.camera_num)
        self._closed = False

    def _get_labels(self):
        labels = self.intrinsics.labels or []
        if getattr(self.intrinsics, "ignore_dash_labels", False):
            labels = [l for l in labels if l and l != "-"]
        return labels

    def _parse_detections(self, metadata: dict):
        """Parse inference metadata into Detection objects."""
        thresh = self.config["threshold"]
        iou_val = self.config["iou"]
        max_dets = self.config["max_detections"]
        intrinsics = self.intrinsics

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = self.imx500.get_input_size()

        if np_outputs is None:
            return self.last_detections

        if getattr(intrinsics, "postprocess", None) == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0],
                conf=thresh,
                iou_thres=iou_val,
                max_out_dets=max_dets,
            )[0]
            from picamera2.devices.imx500.postprocess import scale_boxes

            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = (
                np_outputs[0][0],
                np_outputs[1][0],
                np_outputs[2][0],
            )
            if getattr(intrinsics, "bbox_normalization", False):
                boxes = boxes / input_h
            if getattr(intrinsics, "bbox_order", "yx") == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

        self.last_detections = [
            Detection(box, int(cat), float(score), metadata, self.imx500, self.picam2)
            for box, score, cat in zip(boxes, scores, classes)
            if score > thresh
        ]
        return self.last_detections

    def _prune_recent_counts(self, now: float):
        """Drop old recent-count entries outside the de-duplication window."""
        cutoff = now - (RECENT_COUNT_TIME * 2.0)
        self.recent_counts = [
            (ts, cx, cy) for (ts, cx, cy) in self.recent_counts if ts >= cutoff
        ]

    def _is_duplicate_count(self, cx: int, cy: int, now: float) -> bool:
        """Return True if a new count at (cx, cy) is likely the same shrimp."""
        for ts, prev_cx, prev_cy in self.recent_counts:
            if now - ts > RECENT_COUNT_TIME:
                continue
            if math.hypot(cx - prev_cx, cy - prev_cy) <= RECENT_COUNT_DISTANCE:
                return True
        return False

    def _register_count(self, cx: int, cy: int):
        """
        Register a new shrimp count, with short-term spatial/temporal de-duplication.
        Returns True if the global count was incremented, False if treated as duplicate.
        """
        now = time.time()
        self._prune_recent_counts(now)
        if self._is_duplicate_count(cx, cy, now):
            return False

        self.total_shrimp_count += 1
        self.recent_counts.append((now, cx, cy))
        return True

    def _draw_detections(self, request, stream="main"):
        """Pre-callback: draw detections and run centroid tracking."""
        detections = self.last_results
        if detections is None:
            return

        with MappedArray(request, stream) as m:
            height, width = m.array.shape[:2]
            # 30% Detection Area (left), 70% Count Area (right)
            split_x = int(width * DETECTION_AREA_RATIO)

            has_alpha = (m.array.ndim == 3 and m.array.shape[2] == 4)
            blue = (255, 0, 0, 255) if has_alpha else (255, 0, 0)
            white = (255, 255, 255, 255) if has_alpha else (255, 255, 255)
            green = (0, 255, 0, 255) if has_alpha else (0, 255, 0)
            red = (0, 0, 255, 255) if has_alpha else (0, 0, 255)
            yellow = (0, 255, 255, 255) if has_alpha else (0, 255, 255)

            cv2.line(m.array, (split_x, 0), (split_x, height), blue, 2)
            cv2.putText(
                m.array, "Detection Area", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1
            )
            cv2.putText(
                m.array, "Count Area", (split_x + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1
            )

            current_centroids = []
            for det in detections:
                x, y, w, h = det.box
                cx, cy = int(x + w / 2), int(y + h / 2)
                current_centroids.append((cx, cy, x, y, w, h))
                cv2.rectangle(
                    m.array,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    green,
                    1,
                )
                cv2.circle(m.array, (cx, cy), 3, red, -1)

            # Original centroid tracking / counting logic (line crossing + near-line)
            if len(current_centroids) == 0:
                for obj_id in list(self.tracked_objects.keys()):
                    self.tracked_objects[obj_id]["disappeared"] += 1
                    if self.tracked_objects[obj_id]["disappeared"] > MAX_DISAPPEARED:
                        del self.tracked_objects[obj_id]
            else:
                if len(self.tracked_objects) == 0:
                    for cx, cy, x, y, w, h in current_centroids:
                        self.tracked_objects[self.next_object_id] = {
                            "centroid": (cx, cy),
                            "counted": False,
                            "disappeared": 0,
                        }
                        if cx > split_x:
                            self.tracked_objects[self.next_object_id]["counted"] = True
                        self.next_object_id += 1
                else:
                    used_centroids = set()
                    used_ids = set()
                    distances = []
                    for i, (cx, cy, x, y, w, h) in enumerate(current_centroids):
                        for obj_id, data in self.tracked_objects.items():
                            prev_cx, prev_cy = data["centroid"]
                            dist = math.hypot(cx - prev_cx, cy - prev_cy)
                            max_d = (
                                MAX_DISTANCE_REAPPEAR
                                if data["disappeared"] > 0
                                else MAX_DISTANCE
                            )
                            if dist <= max_d:
                                distances.append((dist, obj_id, i))
                    distances.sort(key=lambda item: item[0])

                    for dist, obj_id, i in distances:
                        if obj_id in used_ids or i in used_centroids:
                            continue
                        used_ids.add(obj_id)
                        used_centroids.add(i)
                        cx, cy = current_centroids[i][0], current_centroids[i][1]
                        prev_cx = self.tracked_objects[obj_id]["centroid"][0]
                        self.tracked_objects[obj_id]["centroid"] = (cx, cy)
                        self.tracked_objects[obj_id]["disappeared"] = 0
                        if (
                            prev_cx <= split_x
                            and cx > split_x
                            and not self.tracked_objects[obj_id]["counted"]
                        ):
                            self.total_shrimp_count += 1
                            self.tracked_objects[obj_id]["counted"] = True

                    for obj_id in list(self.tracked_objects.keys()):
                        if obj_id not in used_ids:
                            self.tracked_objects[obj_id]["disappeared"] += 1
                            if self.tracked_objects[obj_id]["disappeared"] > MAX_DISAPPEARED:
                                del self.tracked_objects[obj_id]

                    for i, (cx, cy, x, y, w, h) in enumerate(current_centroids):
                        if i not in used_centroids:
                            is_count_area = cx > split_x
                            near_line = is_count_area and (cx - split_x) < NEAR_LINE_PX
                            self.tracked_objects[self.next_object_id] = {
                                "centroid": (cx, cy),
                                "counted": is_count_area,
                                "disappeared": 0,
                            }
                            if near_line:
                                self.total_shrimp_count += 1
                                self.tracked_objects[self.next_object_id]["counted"] = True
                            elif is_count_area:
                                self.tracked_objects[self.next_object_id]["counted"] = True
                            self.next_object_id += 1

            cv2.putText(
                m.array, f"Live Count: {self.total_shrimp_count}",
                (split_x + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2
            )

            if getattr(self.intrinsics, "preserve_aspect_ratio", False):
                b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
                cv2.putText(
                    m.array, "ROI", (b_x + 5, b_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                )
                cv2.rectangle(
                    m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0), 1
                )

    def start(self):
        """Start camera and inference pipeline."""
        # Recreate Picamera2 after close() released it (avoids libcamera "Configured state" error).
        recreated = False
        if self.picam2 is None or self._closed:
            self.picam2 = Picamera2(self.imx500.camera_num)
            self._closed = False
            recreated = True
            # If we had to re-acquire the camera, allow firmware upload again.
            self._fw_uploaded = False
        ir = getattr(
            self.intrinsics,
            "inference_rate",
            getattr(self.intrinsics, "fps", 10),
        )
        # Force BGR output so OpenCV/UI channel order is correct (avoids "blue skin" tint).
        config = self.picam2.create_preview_configuration(
            main={"format": "BGR888"},
            controls={"FrameRate": ir},
            buffer_count=12,
        )
        # Upload network firmware only when (re)acquiring the camera.
        if recreated and not self._fw_uploaded:
            self.imx500.show_network_fw_progress_bar()
            self._fw_uploaded = True
        self.picam2.start(config, show_preview=False)
        # ScalerCrop (zoom in) + manual exposure to reduce motion blur; AnalogueGain compensates for darkness
        self.picam2.set_controls({
            "ScalerCrop": SCALER_CROP,
            "ExposureTime": EXPOSURE_TIME_US,
            "AnalogueGain": ANALOGUE_GAIN,
        })
        if getattr(self.intrinsics, "preserve_aspect_ratio", False):
            self.imx500.set_auto_aspect_ratio()
        self.picam2.pre_callback = lambda req, s="main": self._draw_detections(req, s)
        self.last_results = None

    def capture_frame_and_count(self):
        """Capture one frame and return (frame_array, shrimp_count). Returns (None, 0) on error."""
        if not self.picam2:
            return None, 0
        try:
            request = self.picam2.capture_request()
            metadata = request.get_metadata()
            self.last_results = self._parse_detections(metadata)
            with MappedArray(request, "main") as m:
                frame = m.array.copy()
            request.release()
            return frame, self.total_shrimp_count
        except Exception:
            return None, self.total_shrimp_count

    def stop(self):
        """Stop camera for start/stop cycles within a session."""
        if self.picam2 and not self._closed:
            try:
                self.picam2.stop()
                time.sleep(0.5)
            except Exception:
                pass

    def close(self):
        """Fully release camera and IMX500 resources when leaving the biomass page."""
        if self.picam2 and not self._closed:
            try:
                self.picam2.pre_callback = None
                self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None
            self._closed = True
            time.sleep(1.5)  # Allow libcamera to fully release before re-acquire

    def is_closed(self):
        return self._closed

    def reset_count(self):
        """Reset tracking and shrimp count."""
        self.tracked_objects.clear()
        self.next_object_id = 0
        self.total_shrimp_count = 0
        self.recent_counts.clear()


class IMX500Worker(QThread):
    """Background worker that captures frames and emits them to the UI."""

    frame_ready = pyqtSignal(object, int)  # (frame: np.ndarray | None, count: int)
    error = pyqtSignal(str)

    def __init__(self, camera: IMX500Camera | None, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._stop_requested = False

    def run(self):
        if self.camera is None:
            self.frame_ready.emit(None, 0)
            return
        try:
            self.camera.start()
        except Exception as exc:
            self.frame_ready.emit(None, 0)
            self.error.emit(f"Camera start failed: {exc}")
            return

        while not self._stop_requested:
            frame, count = self.camera.capture_frame_and_count()
            self.frame_ready.emit(frame, count)
            if frame is None:
                time.sleep(0.1)

        try:
            self.camera.stop()
        except Exception as exc:
            self.error.emit(f"Camera stop failed: {exc}")

    def request_stop(self):
        self._stop_requested = True
