"""
object_detection.py  —  TFLite object detection for Raspberry Pi 4 + Camera Module 2
Model: SSD MobileNet v1 INT8 quant (detect.tflite + labelmap.txt)
"""

# MUST be set before importing tflite — disables XNNPACK which fuses the
# 4 SSD output tensors into 2 and breaks post-invoke tensor access.
import os
os.environ["TFLITE_DISABLE_XNNPACK"] = "1"

import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH     = "detect.tflite"
LABELS_PATH    = "labelmap.txt"
CAPTURE_SIZE   = (640, 480)   # camera resolution
INPUT_SIZE     = (300, 300)   # SSD MobileNet v1 expects 300×300
CONF_THRESHOLD = 0.5          # raise to reduce false positives
MAX_DETECTIONS = 5

# Only fire the callback for these labels (None = all classes)
TARGET_LABELS = {"stop sign", "traffic light", "person", "car"}

# BGR colours per label for bounding boxes
COLOURS = [
    (220, 80,  80),  (80,  200, 80),  (80,  80,  220),
    (220, 180, 40),  (180, 40,  220), (40,  220, 180),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_labels(path: str) -> list[str]:
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    # labelmap.txt has a background entry ("???" or blank) at index 0 — skip it
    if lines and lines[0] in ("???", ""):
        lines = lines[1:]
    return lines


def label_colour(label: str) -> tuple:
    return COLOURS[hash(label) % len(COLOURS)]


# ── ObjectDetector ────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Threaded TFLite object detector.

    detector = ObjectDetector(on_detection=my_callback)
    detector.start()
    frame   = detector.get_latest_frame()      # annotated BGR frame
    results = detector.get_latest_detections() # [(label, conf, bbox), ...]
    detector.stop()

    Callback: on_detection(label: str, confidence: float, bbox: tuple)
              bbox = (x_min, y_min, x_max, y_max) in pixels
    """

    def __init__(
        self,
        model_path:     str        = MODEL_PATH,
        labels_path:    str        = LABELS_PATH,
        capture_size:   tuple      = CAPTURE_SIZE,
        input_size:     tuple      = INPUT_SIZE,
        conf_threshold: float      = CONF_THRESHOLD,
        max_detections: int        = MAX_DETECTIONS,
        target_labels:  set | None = TARGET_LABELS,
        on_detection                = None,
    ):
        self.capture_size   = capture_size
        self.input_size     = input_size
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.target_labels  = target_labels
        self.on_detection   = on_detection

        self.labels = load_labels(labels_path)

        # ── Load interpreter ──────────────────────────────────────────────────
        self.interp = tflite.Interpreter(model_path=model_path, num_threads=4)
        self.interp.allocate_tensors()

        self._in_idx = self.interp.get_input_details()[0]["index"]
        self._in_dtype = self.interp.get_input_details()[0]["dtype"]

        # Cache output tensor indices by name at load time.
        # get_output_details() is reliable before invoke(); indices don't change.
        name_map = {d["name"]: d["index"] for d in self.interp.get_output_details()}

        # Expected names for SSD MobileNet / TFLite_Detection_PostProcess
        expected = [
            "TFLite_Detection_PostProcess",    # boxes    (1,N,4)
            "TFLite_Detection_PostProcess:1",  # classes  (1,N)
            "TFLite_Detection_PostProcess:2",  # scores   (1,N)
            "TFLite_Detection_PostProcess:3",  # count    (1,)
        ]
        if all(n in name_map for n in expected):
            self._idx_boxes   = name_map[expected[0]]
            self._idx_classes = name_map[expected[1]]
            self._idx_scores  = name_map[expected[2]]
            self._idx_count   = name_map[expected[3]]
        else:
            # Fallback: sort by index and assign positionally
            print("[ObjectDetector] Warning: expected tensor names not found, "
                  f"falling back to positional order. Found: {list(name_map)}")
            sorted_idx = sorted(self.interp.get_output_details(),
                                key=lambda d: d["index"])
            self._idx_boxes   = sorted_idx[0]["index"]
            self._idx_classes = sorted_idx[1]["index"]
            self._idx_scores  = sorted_idx[2]["index"]
            self._idx_count   = sorted_idx[3]["index"] if len(sorted_idx) > 3 else None

        # ── Threading ─────────────────────────────────────────────────────────
        self._lock         = threading.Lock()
        self._running      = False
        self._thread       = None
        self._latest_frame = None
        self._latest_dets  = []
        self._fps          = 0.0
        self._cam          = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._cam = Picamera2()
        cfg = self._cam.create_preview_configuration(
            main={"size": self.capture_size, "format": "RGB888"}
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(0.5)

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[ObjectDetector] Started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cam:
            self._cam.stop()
        print("[ObjectDetector] Stopped.")

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_detections(self) -> list:
        with self._lock:
            return list(self._latest_dets)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            t0 = time.time()

            frame_rgb  = self._cam.capture_array()
            detections = self._infer(frame_rgb)

            # Filter
            filtered = [
                d for d in detections
                if d[1] >= self.conf_threshold
                and (self.target_labels is None or d[0] in self.target_labels)
            ][:self.max_detections]

            # Callback
            if self.on_detection:
                for det in filtered:
                    self.on_detection(*det)

            # Annotate
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            annotated = self._annotate(frame_bgr, filtered, self._fps)

            fps = 1.0 / max(time.time() - t0, 1e-6)

            with self._lock:
                self._latest_frame = annotated
                self._latest_dets  = filtered
                self._fps          = fps

    def _infer(self, frame_rgb: np.ndarray) -> list:
        cap_h, cap_w = self.capture_size[1], self.capture_size[0]
        in_w, in_h   = self.input_size

        # Preprocess
        resized = cv2.resize(frame_rgb, (in_w, in_h))
        blob    = np.expand_dims(
            resized.astype(self._in_dtype if self._in_dtype == np.uint8
                           else np.float32),
            axis=0,
        )
        if self._in_dtype != np.uint8:
            blob = blob / 255.0

        # Inference
        self.interp.set_tensor(self._in_idx, blob)
        self.interp.invoke()

        # Read outputs using cached indices
        boxes   = self.interp.get_tensor(self._idx_boxes)[0]    # (N, 4)
        classes = self.interp.get_tensor(self._idx_classes)[0]  # (N,)
        scores  = self.interp.get_tensor(self._idx_scores)[0]   # (N,)
        count   = int(
            self.interp.get_tensor(self._idx_count)[0]
            if self._idx_count is not None
            else len(scores)
        )

        detections = []
        for i in range(count):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            class_id = int(classes[i])
            label    = (self.labels[class_id]
                        if class_id < len(self.labels) else str(class_id))

            y_min, x_min, y_max, x_max = boxes[i]
            bbox = (
                int(np.clip(x_min * cap_w, 0, cap_w - 1)),
                int(np.clip(y_min * cap_h, 0, cap_h - 1)),
                int(np.clip(x_max * cap_w, 0, cap_w - 1)),
                int(np.clip(y_max * cap_h, 0, cap_h - 1)),
            )
            detections.append((label, score, bbox))

        return detections

    @staticmethod
    def _annotate(frame: np.ndarray, detections: list, fps: float) -> np.ndarray:
        for label, conf, (x1, y1, x2, y2) in detections:
            colour = label_colour(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            text = f"{label}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1 - 6, th + bl)
            cv2.rectangle(frame, (x1, ty - th - bl), (x1 + tw + 4, ty + bl),
                          colour, cv2.FILLED)
            cv2.putText(frame, text, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return frame


# ── RC car mixin ──────────────────────────────────────────────────────────────

class RCCarDetectionMixin:
    """
    Bolt-on mixin for the RC car controller.

    class RCCar(RCCarDetectionMixin):
        def __init__(self):
            self._init_detection()

        def _handle_detection(self, label, confidence, bbox):
            if label == "stop sign" and confidence > 0.6:
                self.stop_motors()
                time.sleep(2)
                self.resume()
    """

    def _init_detection(self, **kwargs):
        self._detector = ObjectDetector(on_detection=self._handle_detection, **kwargs)

    def _start_detection(self):
        self._detector.start()

    def _stop_detection(self):
        self._detector.stop()

    def _handle_detection(self, label: str, confidence: float, bbox: tuple):
        print(f"[Detection] {label:<20s}  conf={confidence:.2f}  bbox={bbox}")


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    def on_det(label, conf, bbox):
        print(f"  {label:<20s} {conf:.0%}  {bbox}")

    detector = ObjectDetector(on_detection=on_det)
    detector.start()

    print("Running — press 'q' in the window or Ctrl-C to quit.")
    try:
        while True:
            frame = detector.get_latest_frame()
            if frame is not None:
                cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
        cv2.destroyAllWindows()
    sys.exit(0)
