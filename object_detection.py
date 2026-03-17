"""
object_detection.py
-------------------
TensorFlow Lite object detection module for the Raspberry Pi 4 + Camera Module 2.
Designed to integrate with the autonomous RC car project.

Features:
  - Threaded capture + inference pipeline (non-blocking for motor/steering loop)
  - EfficientDet-Lite0 by default (or any SSD-style TFLite model)
  - Configurable confidence threshold and target label filtering
  - OpenCV overlay with bounding boxes and labels
  - Simple callback hook: on_detection(label, confidence, bbox)

Dependencies:
  pip install tflite-runtime opencv-python picamera2 numpy

Model download (EfficientDet-Lite0):
  wget https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
  wget https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/tasks/python/vision/object_detector/efficientdet_lite0_labels.txt
"""

import threading
import time
import cv2
import numpy as np

from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH      = "efficientdet_lite0.tflite"
LABELS_PATH     = "efficientdet_lite0_labels.txt"
CAPTURE_SIZE    = (640, 480)   # Camera capture resolution
INPUT_SIZE      = (320, 320)   # Model input size (EfficientDet-Lite0 expects 320x320)
CONF_THRESHOLD  = 0.45         # Minimum confidence to report a detection
MAX_DETECTIONS  = 5            # Max number of boxes to draw per frame

# Labels to actively react to (set to None to report all classes)
TARGET_LABELS = {"stop sign", "traffic light", "person", "car"}


# ---------------------------------------------------------------------------
# Label loader
# ---------------------------------------------------------------------------

def load_labels(path: str) -> list[str]:
    """Load class labels from a plain-text file (one label per line)."""
    with open(path, "r") as f:
        lines = f.read().splitlines()
    # Strip optional leading index (e.g. "0 person" → "person")
    labels = []
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        labels.append(parts[-1] if parts else "")
    return labels


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Threaded TFLite object detector.

    Usage
    -----
    detector = ObjectDetector(on_detection=my_callback)
    detector.start()
    ...
    latest = detector.get_latest_frame()   # annotated BGR frame for display
    detector.stop()

    Callback signature
    ------------------
    def my_callback(label: str, confidence: float, bbox: tuple):
        # bbox = (x_min, y_min, x_max, y_max) in pixel coords of CAPTURE_SIZE
        ...
    """

    def __init__(
        self,
        model_path:      str        = MODEL_PATH,
        labels_path:     str        = LABELS_PATH,
        capture_size:    tuple      = CAPTURE_SIZE,
        input_size:      tuple      = INPUT_SIZE,
        conf_threshold:  float      = CONF_THRESHOLD,
        max_detections:  int        = MAX_DETECTIONS,
        target_labels:   set | None = TARGET_LABELS,
        on_detection                = None,
    ):
        self.capture_size   = capture_size
        self.input_size     = input_size
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.target_labels  = target_labels
        self.on_detection   = on_detection  # optional callback

        # Load labels
        self.labels = load_labels(labels_path)

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self._input_details  = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

        # Determine whether the model expects uint8 or float32 input
        self._input_dtype = self._input_details[0]["dtype"]

        # Threading state
        self._thread       = None
        self._running      = False
        self._lock         = threading.Lock()
        self._latest_frame = None   # annotated BGR frame
        self._latest_dets  = []     # list of (label, conf, bbox) from last frame

        # Camera
        self._picam2 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the camera and the background detection thread."""
        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": self.capture_size, "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(0.5)  # allow camera to warm up

        self._running = True
        self._thread  = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        print("[ObjectDetector] Started.")

    def stop(self):
        """Stop the detection thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._picam2:
            self._picam2.stop()
        print("[ObjectDetector] Stopped.")

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recent annotated BGR frame (thread-safe)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_detections(self) -> list:
        """Return the most recent list of (label, confidence, bbox) tuples."""
        with self._lock:
            return list(self._latest_dets)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detection_loop(self):
        while self._running:
            # 1. Capture frame (RGB numpy array from Picamera2)
            frame_rgb = self._picam2.capture_array()

            # 2. Run inference
            detections = self._run_inference(frame_rgb)

            # 3. Filter by confidence and optional target labels
            filtered = [
                d for d in detections
                if d[1] >= self.conf_threshold
                and (self.target_labels is None or d[0] in self.target_labels)
            ]
            filtered = filtered[:self.max_detections]

            # 4. Fire callback for each valid detection
            if self.on_detection:
                for label, conf, bbox in filtered:
                    self.on_detection(label, conf, bbox)

            # 5. Draw bounding boxes onto a BGR copy for display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            annotated  = self._draw_detections(frame_bgr, filtered)

            # 6. Store results thread-safely
            with self._lock:
                self._latest_frame = annotated
                self._latest_dets  = filtered

    def _run_inference(self, frame_rgb: np.ndarray) -> list:
        """
        Preprocess frame, run TFLite interpreter, and parse raw SSD outputs.

        This model produces 2 tensors (no built-in NMS):
          StatefulPartitionedCall:0  (1, 19206, 4)  – box coords [y_min, x_min, y_max, x_max] normalised
          StatefulPartitionedCall:1  (1, 19206, 90) – raw logits, one per class per anchor

        Returns list of (label, confidence, bbox) after sigmoid + NMS.
        bbox = (x_min_px, y_min_px, x_max_px, y_max_px)
        """
        cap_h, cap_w = self.capture_size[1], self.capture_size[0]
        in_h,  in_w  = self.input_size

        # --- 1. Preprocess ---
        resized = cv2.resize(frame_rgb, (in_w, in_h))
        if self._input_dtype == np.uint8:
            input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
        else:
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        self.interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # --- 2. Retrieve tensors by shape (name order is not guaranteed) ---
        all_tensors = [
            self.interpreter.get_tensor(d["index"])
            for d in self._output_details
        ]

        boxes_raw  = None   # (19206, 4)
        scores_raw = None   # (19206, 90)

        for t in all_tensors:
            if t.ndim == 3 and t.shape[2] == 4:
                boxes_raw = t[0]       # (N, 4)
            elif t.ndim == 3 and t.shape[2] > 4:
                scores_raw = t[0]      # (N, num_classes)

        if boxes_raw is None or scores_raw is None:
            print("[ObjectDetector] Warning: could not identify output tensors.")
            return []

        # --- 3. Sigmoid → probabilities, find best class per anchor ---
        probs = 1.0 / (1.0 + np.exp(-scores_raw.astype(np.float32)))  # (N, 90)

        best_class_scores = probs.max(axis=1)   # (N,)
        best_class_ids    = probs.argmax(axis=1) # (N,)

        # --- 4. Pre-filter by confidence threshold ---
        keep = np.where(best_class_scores >= self.conf_threshold)[0]
        if len(keep) == 0:
            return []

        filtered_boxes   = boxes_raw[keep]            # (M, 4)
        filtered_scores  = best_class_scores[keep]    # (M,)
        filtered_classes = best_class_ids[keep]       # (M,)

        # --- 5. Convert [y_min, x_min, y_max, x_max] → pixel rects for NMS ---
        # cv2.dnn.NMSBoxes expects [x, y, w, h] in pixels
        rects = []
        for y_min, x_min, y_max, x_max in filtered_boxes:
            x1 = int(np.clip(x_min * cap_w, 0, cap_w))
            y1 = int(np.clip(y_min * cap_h, 0, cap_h))
            x2 = int(np.clip(x_max * cap_w, 0, cap_w))
            y2 = int(np.clip(y_max * cap_h, 0, cap_h))
            rects.append([x1, y1, x2 - x1, y2 - y1])   # [x, y, w, h]

        # --- 6. NMS (IoU threshold 0.45) ---
        nms_indices = cv2.dnn.NMSBoxes(
            rects,
            filtered_scores.tolist(),
            self.conf_threshold,
            0.45,           # IoU threshold
        )

        if len(nms_indices) == 0:
            return []

        # cv2 returns shape (N, 1) in older versions, flat list in newer
        nms_indices = np.array(nms_indices).flatten()

        # --- 7. Build final detection list ---
        detections = []
        for idx in nms_indices[:self.max_detections]:
            class_id = int(filtered_classes[idx])
            score    = float(filtered_scores[idx])
            label    = self.labels[class_id] if class_id < len(self.labels) else str(class_id)

            x, y, w, h = rects[idx]
            bbox = (x, y, x + w, y + h)   # (x_min, y_min, x_max, y_max)

            detections.append((label, score, bbox))

        return detections

    def _draw_detections(
        self,
        frame_bgr:  np.ndarray,
        detections: list,
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto a BGR frame."""
        for label, conf, (x1, y1, x2, y2) in detections:
            color = self._label_color(label)

            # Bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            # Label background + text
            text     = f"{label}  {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y  = max(y1 - 6, th + baseline)
            cv2.rectangle(
                frame_bgr,
                (x1, label_y - th - baseline),
                (x1 + tw + 4, label_y + baseline),
                color, cv2.FILLED,
            )
            cv2.putText(
                frame_bgr, text,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # FPS indicator (top-left)
        cv2.putText(
            frame_bgr, "TFLite Detect",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return frame_bgr

    @staticmethod
    def _label_color(label: str) -> tuple:
        """Return a consistent BGR color for a given label string."""
        palette = [
            (220,  80,  80),  # blue
            ( 80, 200,  80),  # green
            ( 80,  80, 220),  # red
            (220, 180,  40),  # cyan
            (180,  40, 220),  # magenta
            ( 40, 220, 180),  # yellow-green
        ]
        return palette[hash(label) % len(palette)]


# ---------------------------------------------------------------------------
# RC car integration helper
# ---------------------------------------------------------------------------

class RCCarDetectionMixin:
    """
    Drop-in mixin that wires ObjectDetector to the RC car's action system.

    Subclass your main RC car controller with this and call
    `self._init_detection()` in __init__, then `self._start_detection()` /
    `self._stop_detection()` in your start/stop lifecycle.

    Override `_handle_detection()` to map detections to motor commands.
    """

    def _init_detection(
        self,
        model_path:  str   = MODEL_PATH,
        labels_path: str   = LABELS_PATH,
        conf:        float = CONF_THRESHOLD,
    ):
        self._detector = ObjectDetector(
            model_path      = model_path,
            labels_path     = labels_path,
            conf_threshold  = conf,
            target_labels   = TARGET_LABELS,
            on_detection    = self._handle_detection,
        )

    def _start_detection(self):
        self._detector.start()

    def _stop_detection(self):
        self._detector.stop()

    def _handle_detection(self, label: str, confidence: float, bbox: tuple):
        """
        Override this in your RC car class.

        Example:
            if label == "stop sign" and confidence > 0.6:
                self.stop_motors()
                time.sleep(2)
                self.resume()
        """
        print(f"[Detection] {label:20s}  conf={confidence:.2f}  bbox={bbox}")


# ---------------------------------------------------------------------------
# Standalone demo (run directly on the Pi to test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Starting standalone detection demo. Press 'q' to quit.")

    detector = ObjectDetector(
        on_detection=lambda lbl, conf, bbox: print(
            f"  >> {lbl:<20s} {conf:.0%}  {bbox}"
        )
    )
    detector.start()

    # OpenCV display loop (runs on the main thread)
    while True:
        frame = detector.get_latest_frame()
        if frame is not None:
            cv2.imshow("TFLite Object Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    detector.stop()
    cv2.destroyAllWindows()
    sys.exit(0)
