# source env/bin/activate
# https://ignorantofthings.com/receiving-infrared-on-the-raspberry-pi-with-python/
# sudo ir-keytable -p all   (enable all protocols)
# sudo ir-keytable -t       (test)

# ── XNNPACK must be disabled before any tflite import ────────────────────────
import os
os.environ["TFLITE_DISABLE_XNNPACK"] = "1"

# ── Imports ───────────────────────────────────────────────────────────────────
import sys
import threading
import time
from time import sleep

import board
import cv2
import evdev
import numpy as np
import RPi.GPIO as GPIO
import serial
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

# ── OLED display setup ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 128, 64
i2c  = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C)
oled.fill(0)
oled.show()
image = Image.new("1", (oled.width, oled.height))
draw  = ImageDraw.Draw(image)
draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
font  = ImageFont.load_default()

# ── Serial ────────────────────────────────────────────────────────────────────
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# ── Object detection config ───────────────────────────────────────────────────
MODEL_PATH     = "detect.tflite"
LABELS_PATH    = "labelmap.txt"
CAPTURE_SIZE   = (640, 480)
INPUT_SIZE     = (300, 300)
CONF_THRESHOLD = 0.5
MAX_DETECTIONS = 5
TARGET_LABELS  = {"stop sign", "traffic light", "person", "car"}
COLOURS        = [
    (220,  80,  80), (80,  200,  80), (80,   80, 220),
    (220, 180,  40), (180,  40, 220), (40,  220, 180),
]

# ── Stop sign behaviour config ────────────────────────────────────────────────
STOP_SIGN_CONF      = 0.6    # confidence required to trigger a stop
STOP_SIGN_DURATION  = 3.0    # seconds to hold throttle=0 after detection
STOP_SIGN_COOLDOWN  = 5.0    # seconds before the same sign can trigger again


# ═══════════════════════════════════════════════════════════════════════════════
# Object detector
# ═══════════════════════════════════════════════════════════════════════════════

def load_labels(path: str) -> list:
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    if lines and lines[0] in ("???", ""):
        lines = lines[1:]
    return lines


def label_colour(label: str) -> tuple:
    return COLOURS[hash(label) % len(COLOURS)]


class ObjectDetector:
    """
    Threaded TFLite object detector.

    Accepts an optional existing Picamera2 instance so it can share the
    camera with other modes rather than opening a second one.

    detector = ObjectDetector(on_detection=my_callback)
    detector.start()
    frame   = detector.get_latest_frame()       # annotated BGR frame
    results = detector.get_latest_detections()  # [(label, conf, bbox), ...]
    detector.stop()

    Callback signature:
        on_detection(label: str, confidence: float, bbox: tuple)
        bbox = (x_min, y_min, x_max, y_max) in pixels
    """

    def __init__(
        self,
        model_path:     str   = MODEL_PATH,
        labels_path:    str   = LABELS_PATH,
        capture_size:   tuple = CAPTURE_SIZE,
        input_size:     tuple = INPUT_SIZE,
        conf_threshold: float = CONF_THRESHOLD,
        max_detections: int   = MAX_DETECTIONS,
        target_labels         = TARGET_LABELS,
        on_detection          = None,
        cam                   = None,
    ):
        self.capture_size   = capture_size
        self.input_size     = input_size
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.target_labels  = target_labels
        self.on_detection   = on_detection

        self.labels = load_labels(labels_path)

        # TFLite interpreter
        self.interp = tflite.Interpreter(model_path=model_path, num_threads=4)
        self.interp.allocate_tensors()

        self._in_idx   = self.interp.get_input_details()[0]["index"]
        self._in_dtype = self.interp.get_input_details()[0]["dtype"]

        # Cache output tensor indices by name
        name_map = {d["name"]: d["index"] for d in self.interp.get_output_details()}
        expected = [
            "TFLite_Detection_PostProcess",
            "TFLite_Detection_PostProcess:1",
            "TFLite_Detection_PostProcess:2",
            "TFLite_Detection_PostProcess:3",
        ]
        if all(n in name_map for n in expected):
            self._idx_boxes   = name_map[expected[0]]
            self._idx_classes = name_map[expected[1]]
            self._idx_scores  = name_map[expected[2]]
            self._idx_count   = name_map[expected[3]]
        else:
            print("[ObjectDetector] Warning: expected tensor names not found, "
                  f"falling back to positional order. Found: {list(name_map)}")
            s = sorted(self.interp.get_output_details(), key=lambda d: d["index"])
            self._idx_boxes   = s[0]["index"]
            self._idx_classes = s[1]["index"]
            self._idx_scores  = s[2]["index"]
            self._idx_count   = s[3]["index"] if len(s) > 3 else None

        # Camera
        self._cam       = cam
        self._owns_cam  = (cam is None)

        # Threading
        self._lock         = threading.Lock()
        self._running      = False
        self._thread       = None
        self._latest_frame = None
        self._latest_dets  = []
        self._fps          = 0.0

    def start(self):
        if self._owns_cam:
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
        if self._owns_cam and self._cam:
            self._cam.stop()
        print("[ObjectDetector] Stopped.")

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_detections(self):
        with self._lock:
            return list(self._latest_dets)

    def _loop(self):
        while self._running:
            try:
                t0        = time.time()
                frame_rgb = self._cam.capture_array()
                dets      = self._infer(frame_rgb)

                filtered = [
                    d for d in dets
                    if d[1] >= self.conf_threshold
                    and (self.target_labels is None or d[0] in self.target_labels)
                ][:self.max_detections]

                if self.on_detection:
                    for det in filtered:
                        self.on_detection(*det)

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                annotated = self._annotate(frame_bgr, filtered, self._fps)
                fps       = 1.0 / max(time.time() - t0, 1e-6)

                with self._lock:
                    self._latest_frame = annotated
                    self._latest_dets  = filtered
                    self._fps          = fps
            except Exception as e:
                print(f"[ObjectDetector] Frame error: {e}")
                time.sleep(0.1)

    def _infer(self, frame_rgb: np.ndarray) -> list:
        cap_h, cap_w = self.capture_size[1], self.capture_size[0]
        in_w, in_h   = self.input_size

        resized = cv2.resize(frame_rgb, (in_w, in_h))
        blob    = np.expand_dims(
            resized.astype(np.uint8 if self._in_dtype == np.uint8 else np.float32),
            axis=0,
        )
        if self._in_dtype != np.uint8:
            blob = blob / 255.0

        self.interp.set_tensor(self._in_idx, blob)
        self.interp.invoke()

        boxes   = self.interp.get_tensor(self._idx_boxes)[0]
        classes = self.interp.get_tensor(self._idx_classes)[0]
        scores  = self.interp.get_tensor(self._idx_scores)[0]
        count   = int(
            self.interp.get_tensor(self._idx_count)[0]
            if self._idx_count is not None else len(scores)
        )

        detections = []
        for i in range(count):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue
            class_id = int(classes[i])
            label    = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
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
    def _annotate(frame, detections, fps):
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


# ═══════════════════════════════════════════════════════════════════════════════
# IR remote helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_ir_device():
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if dev.name == "gpio_ir_recv":
            print("Using device", dev.path)
            return dev
    print("No IR device found!")
    sys.exit()


def get_last_event(dev):
    last_event = None
    try:
        for event in dev.read():
            if event.value > 0:
                last_event = event
    except BlockingIOError:
        pass
    return last_event


def flush_ir_events(dev):
    """Drain queued IR events so stale repeats don't exit a mode immediately."""
    sleep(0.20)
    try:
        for _ in dev.read():
            pass
    except BlockingIOError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# System helpers
# ═══════════════════════════════════════════════════════════════════════════════

def safe_stop():
    """Emergency stop — always safe to call, never raises."""
    try:
        ser.write(b'0,0\n')
    except Exception:
        pass


def update_controls(throttle, steering):
    try:
        ser.write(f'{throttle},{steering}\n'.encode())
    except Exception as e:
        print(f"[Serial] Write failed: {e}")


def update_display(mode, throttle, steering):
    try:
        draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
        draw.text((0,  0), "ECE 4415 Group 4",        font=font, fill=255)
        draw.text((0, 15), f"Mode: {mode}",            font=font, fill=255)
        draw.text((0, 35), f"Throttle: {throttle}",    font=font, fill=255)
        draw.text((0, 45), f"Steering: {steering}",    font=font, fill=255)
        oled.image(image)
        oled.show()
    except Exception as e:
        print(f"[OLED] Display update failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Lane keeping  (PI-controlled, robust, any-colour lanes)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Tunable constants ─────────────────────────────────────────────────────────
_LK_ROI_TOP_FRAC    = 0.55    # ignore top 55 % of frame (sky / far background)
_LK_ROI_TOP_INSET   = 0.10    # trapezoid inset at the top edge (each side)
_LK_BLUR_KERNEL     = 5       # Gaussian blur kernel size (must be odd)
_LK_CANNY_LOW       = 50
_LK_CANNY_HIGH      = 150
_LK_HOUGH_THRESH    = 30      # minimum Hough votes to accept a line
_LK_HOUGH_MIN_LEN   = 40      # minimum segment length (px)
_LK_HOUGH_MAX_GAP   = 80      # maximum gap between collinear segments (px)
_LK_SLOPE_MIN       = 0.3     # |slope| below this -> discard (near-horizontal)
_LK_SLOPE_MAX       = 3.0     # |slope| above this -> discard (near-vertical)
_LK_LANE_WIDTH_FRAC = 0.60    # prior lane width as a fraction of frame width
_LK_KP              = 0.40    # proportional gain
_LK_KI              = 0.008   # integral gain  (keep small to avoid windup)
_LK_INTEG_CLAMP     = 150.0   # anti-windup clamp  (pixel * seconds)
_LK_STEER_CENTER    = 0       # steering neutral  (matches Arduino protocol)
_LK_STEER_MAX       = 100     # max steering magnitude each side of centre
_LK_THROTTLE_BASE   = 80      # cruise throttle (0-100)
_LK_THROTTLE_MIN    = 30      # floor throttle during hard turns
_LK_THROTTLE_SCALE  = 0.40    # throttle reduction per unit of |steering|
_LK_MAX_LOSS_FRAMES = 15      # consecutive no-detection frames before safe-stop


def _lk_build_roi(h: int, w: int) -> np.ndarray:
    """Trapezoid mask covering only the lower portion of the frame."""
    top_y  = int(h * _LK_ROI_TOP_FRAC)
    inset  = int(w * _LK_ROI_TOP_INSET)
    poly   = np.array([[(0, h), (inset, top_y),
                         (w - inset, top_y), (w, h)]], dtype=np.int32)
    mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, poly, 255)
    return mask


def _lk_detect_edges(frame_bgr: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Grayscale -> blur -> Canny -> ROI mask."""
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (_LK_BLUR_KERNEL, _LK_BLUR_KERNEL), 0)
    edges   = cv2.Canny(blurred, _LK_CANNY_LOW, _LK_CANNY_HIGH)
    return cv2.bitwise_and(edges, edges, mask=roi_mask)


def _lk_classify_lines(lines, width: int):
    """Split Hough segments into left-lane and right-lane buckets."""
    cx           = width / 2.0
    left_params  = []
    right_params = []

    if lines is None:
        return left_params, right_params

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if not (_LK_SLOPE_MIN <= abs(slope) <= _LK_SLOPE_MAX):
            continue

        seg_cx = (x1 + x2) / 2.0

        if slope < 0 and seg_cx < cx:
            left_params.append((slope, intercept))
        elif slope > 0 and seg_cx > cx:
            right_params.append((slope, intercept))

    return left_params, right_params


def _lk_average_line(params, height: int, roi_top: int):
    """Average (slope, intercept) pairs into one line extrapolated to frame edges."""
    if not params:
        return None

    m = float(np.mean([p[0] for p in params]))
    b = float(np.mean([p[1] for p in params]))

    if abs(m) < 1e-6:
        return None

    y_bot = height
    y_top = roi_top
    x_bot = int((y_bot - b) / m)
    x_top = int((y_top - b) / m)
    return (x_bot, y_bot, x_top, y_top)


def _lk_compute_error(left_line, right_line, width: int):
    """Lateral error in pixels (positive = car right of centre)."""
    cx               = width / 2.0
    lane_width_prior = width * _LK_LANE_WIDTH_FRAC

    if left_line is not None and right_line is not None:
        lane_cx = (left_line[0] + right_line[0]) / 2.0
    elif left_line is not None:
        lane_cx = left_line[0] + lane_width_prior / 2.0
    elif right_line is not None:
        lane_cx = right_line[0] - lane_width_prior / 2.0
    else:
        return None

    return cx - lane_cx


def lane_keep(device, picam2):
    """PI-controlled lane keeping mode."""
    flush_ir_events(device)

    h, w     = 480, 640
    roi_mask = _lk_build_roi(h, w)
    roi_top  = int(h * _LK_ROI_TOP_FRAC)

    # PI state
    integral = 0.0
    prev_t   = time.monotonic()

    # Lane-loss tracking
    loss_frames   = 0
    last_steering = _LK_STEER_CENTER
    last_throttle = _LK_THROTTLE_MIN

    print("Lane keep ACTIVE")

    try:
        while get_last_event(device) is None:

            t_now  = time.monotonic()
            dt     = max(t_now - prev_t, 1e-4)
            prev_t = t_now

            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            edges      = _lk_detect_edges(frame_bgr, roi_mask)
            raw_lines  = cv2.HoughLinesP(
                edges,
                rho           = 1,
                theta         = np.pi / 180,
                threshold     = _LK_HOUGH_THRESH,
                minLineLength = _LK_HOUGH_MIN_LEN,
                maxLineGap    = _LK_HOUGH_MAX_GAP,
            )
            left_p, right_p = _lk_classify_lines(raw_lines, w)
            left_line       = _lk_average_line(left_p,  h, roi_top)
            right_line      = _lk_average_line(right_p, h, roi_top)
            error           = _lk_compute_error(left_line, right_line, w)

            if error is not None:
                loss_frames = 0

                integral += error * dt
                integral  = float(np.clip(integral,
                                          -_LK_INTEG_CLAMP, _LK_INTEG_CLAMP))

                raw_output = _LK_KP * error + _LK_KI * integral
                steering   = int(np.clip(-raw_output,
                                         -_LK_STEER_MAX, _LK_STEER_MAX))

                throttle = max(
                    int(_LK_THROTTLE_BASE - abs(steering) * _LK_THROTTLE_SCALE),
                    _LK_THROTTLE_MIN,
                )
                last_steering = steering
                last_throttle = throttle

            else:
                loss_frames += 1

                if loss_frames <= _LK_MAX_LOSS_FRAMES:
                    steering = last_steering
                    throttle = last_throttle
                else:
                    integral  = 0.0
                    steering  = _LK_STEER_CENTER
                    throttle  = 0
                    print("[LaneKeep] Lane lost — safe stop")

            update_controls(throttle, steering)
            update_display("Lane keep", throttle, steering)

            # Debug visualisation
            vis = frame_bgr.copy()

            for line, col in [(left_line,  (255, 100,   0)),
                               (right_line, (  0, 100, 255))]:
                if line is not None:
                    cv2.line(vis, (line[0], line[1]), (line[2], line[3]),
                             col, 4, cv2.LINE_AA)

            bot_y = h - 15
            if error is not None:
                det_cx = int(w / 2.0 - error)
                cv2.circle(vis, (det_cx, bot_y), 10, (0, 255, 255), -1)
                cv2.line(vis, (w // 2, bot_y), (det_cx, bot_y),
                         (0, 255, 255), 2)
            cv2.circle(vis, (w // 2, bot_y), 5, (255, 255, 0), -1)

            status = ("BOTH"  if left_line  and right_line else
                      "LEFT"  if left_line  else
                      "RIGHT" if right_line else "NONE")
            err_str = f"{error:.1f}" if error is not None else "--"
            cv2.putText(vis,
                        f"Lines:{status}  Err:{err_str}px  S:{steering}  T:{throttle}",
                        (8, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Lane Detection", vis)
            print(f"Lines:{status:<5}  Err:{err_str:>7}px"
                  f"  Steer:{steering:4d}  Throttle:{throttle:3d}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(0.05)

    finally:
        update_controls(0, _LK_STEER_CENTER)
        cv2.destroyWindow("Lane Detection")
        print("Lane keep INACTIVE")


# ═══════════════════════════════════════════════════════════════════════════════
# Object detection mode  —  with stop sign handling
# ═══════════════════════════════════════════════════════════════════════════════

class StopSignController:
    """Tracks stop-sign detections and enforces a timed stop + cooldown."""

    def __init__(
        self,
        conf_threshold: float = STOP_SIGN_CONF,
        stop_duration:  float = STOP_SIGN_DURATION,
        cooldown:       float = STOP_SIGN_COOLDOWN,
    ):
        self._conf      = conf_threshold
        self._duration  = stop_duration
        self._cooldown  = cooldown
        self._stop_until    = 0.0
        self._cooldown_until = 0.0
        self._lock      = threading.Lock()

    def on_detection(self, label: str, confidence: float, bbox: tuple):
        if label != "stop sign":
            return
        if confidence < self._conf:
            return
        now = time.time()
        with self._lock:
            if now < self._cooldown_until:
                return
            print(f"[StopSign] Detected ({confidence:.0%}) — stopping for "
                  f"{self._duration}s")
            self._stop_until     = now + self._duration
            self._cooldown_until = now + self._duration + self._cooldown

    def apply(self, throttle: int, steering: int):
        with self._lock:
            if time.time() < self._stop_until:
                return 0, 0
        return throttle, steering

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return time.time() < self._stop_until


def object_detection(device, picam2):
    """Object detection drive mode with stop sign handling."""
    flush_ir_events(device)

    stop_ctrl = StopSignController()
    detector  = ObjectDetector(
        on_detection=stop_ctrl.on_detection,
        cam=picam2,
    )
    detector.start()

    print("Object Detection ACTIVE")

    try:
        while get_last_event(device) is None:
            base_throttle = 80
            base_steering = 0

            throttle, steering = stop_ctrl.apply(base_throttle, base_steering)

            update_controls(throttle, steering)
            update_display(
                "STOP SIGN" if stop_ctrl.is_stopped else "Object Det",
                throttle, steering,
            )

            frame = detector.get_latest_frame()
            if frame is not None:
                cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(0.05)

    finally:
        update_controls(0, 0)
        detector.stop()
        cv2.destroyWindow("Object Detection")
        print("Object Detection INACTIVE")


# ═══════════════════════════════════════════════════════════════════════════════
# Object avoidance
# ═══════════════════════════════════════════════════════════════════════════════

def object_avoidance(device):
    flush_ir_events(device)

    TRIG_FRONT = 22
    ECHO_FRONT = 27
    TRIG_REAR  = 25
    ECHO_REAR  = 10

    AVOID_DISTANCE_IN = 36.0  # 3 feet
    SLOW_DISTANCE_IN  = 48.0  # 4 feet
    BASE_SPEED = 80
    MIN_SPEED  = 30

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(TRIG_FRONT, GPIO.OUT)
    GPIO.setup(ECHO_FRONT, GPIO.IN)
    GPIO.setup(TRIG_REAR,  GPIO.OUT)
    GPIO.setup(ECHO_REAR,  GPIO.IN)

    GPIO.output(TRIG_FRONT, False)
    GPIO.output(TRIG_REAR,  False)
    time.sleep(0.1)

    def get_distance_in(trig_pin, echo_pin, timeout=0.03):
        GPIO.output(trig_pin, False)
        time.sleep(0.0002)
        GPIO.output(trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(trig_pin, False)

        pulse_start   = time.time()
        timeout_start = pulse_start

        while GPIO.input(echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - timeout_start > timeout:
                return 999

        pulse_end = pulse_start
        while GPIO.input(echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - pulse_start > timeout:
                return 999

        pulse_duration = pulse_end - pulse_start
        distance_cm    = pulse_duration * 17150
        distance_in    = distance_cm / 2.54
        return round(distance_in, 2)

    def read_sensors():
        front = get_distance_in(TRIG_FRONT, ECHO_FRONT)
        time.sleep(0.01)
        rear  = get_distance_in(TRIG_REAR,  ECHO_REAR)
        return front, rear

    def compute_avoidance(front, rear, last_turn):
        if front < 12:
            steering = 80 if last_turn <= 0 else -80
            throttle = MIN_SPEED
            return (throttle, steering, steering)
        elif front < AVOID_DISTANCE_IN:
            steering = 50 if last_turn <= 0 else -50
            throttle = int(BASE_SPEED * 0.6)
            return (throttle, steering, steering)
        elif front < SLOW_DISTANCE_IN:
            steering = int(last_turn * 0.5)
            speed_factor = (front - AVOID_DISTANCE_IN) / (SLOW_DISTANCE_IN - AVOID_DISTANCE_IN)
            throttle = int(MIN_SPEED + (BASE_SPEED - MIN_SPEED) * speed_factor)
            return (throttle, steering, steering)
        else:
            return (BASE_SPEED, 0, 0)

    print("Object Avoidance ACTIVE")
    print(f"Avoidance distance: {AVOID_DISTANCE_IN} inches (3 feet)")
    last_turn = 0

    try:
        while get_last_event(device) is None:
            front, rear = read_sensors()
            throttle, steering, last_turn = compute_avoidance(front, rear, last_turn)

            update_controls(throttle, steering)
            update_display("Obj Avoid", throttle, steering)

            print(f"Front: {front:5.1f}in | Rear: {rear:5.1f}in | "
                  f"Throttle: {throttle:4d} | Steering: {steering:4d}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        update_controls(0, 0)
        print("Object Avoidance INACTIVE")


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive cruise control
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_cruise(device):
    flush_ir_events(device)

    TRIG = 23
    ECHO = 24

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    def get_distance_in(timeout=0.03):
        GPIO.output(TRIG, False)
        time.sleep(0.0002)

        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        pulse_start   = time.time()
        timeout_start = pulse_start

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
            if pulse_start - timeout_start > timeout:
                return 999

        pulse_end = pulse_start
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if pulse_end - pulse_start > timeout:
                return 999

        pulse_duration = pulse_end - pulse_start
        distance_cm    = pulse_duration * 17150
        distance_in    = distance_cm / 2.54
        return round(distance_in, 2)

    print("Adaptive Cruise Control ACTIVE")

    try:
        while get_last_event(device) is None:
            dist = get_distance_in()

            if dist >= 999:
                # Sensor error / no echo — hold current speed cautiously
                throttle = 80
            elif dist < 2:
                throttle = 0
            elif dist > 20:
                throttle = 100
            else:
                throttle = int((dist - 2) * (100.0 / 18))

            steering = 0

            update_controls(throttle, steering)
            update_display("Adaptive Cruise", throttle, steering)

            print(f"Distance: {dist} in | Throttle: {throttle}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    finally:
        update_controls(0, 0)
        print("Adaptive Cruise Control INACTIVE")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    sleep(1)

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    throttle     = 0
    steering     = 0
    mode_message = "Waiting..."
    update_controls(throttle, steering)
    update_display(mode_message, throttle, steering)

    device = get_ir_device()

    try:
        while True:
            key = get_last_event(device)
            if key is None:
                sleep(0.05)          # don't burn CPU while idle
                continue

            mode = key
            print("Received command:", mode.value)

            if mode.value == 69:                        # key 1 — lane keep
                print("Lane keep mode")
                lane_keep(device, picam2)
                mode_message = "Lane keep"

            elif mode.value == 70:                      # key 2 — object avoidance
                print("Object avoidance mode")
                object_avoidance(device)
                mode_message = "Object Avoidance"

            elif mode.value == 71:                      # key 3 — object detection
                print("Object detection mode")
                object_detection(device, picam2)
                mode_message = "Object Det"

            elif mode.value == 68:                      # key 4 — adaptive cruise
                print("Adaptive cruise mode")
                adaptive_cruise(device)
                mode_message = "Adaptive Cruise"

            elif mode.value == 24:                      # up — drive forward
                throttle = 100
                steering = 0
                mode_message = "Drive forward"

            elif mode.value == 82:                      # down — drive back
                throttle = -100
                steering = 0
                mode_message = "Drive back"

            elif mode.value == 8:                       # left
                throttle = 100
                steering = -50
                mode_message = "Steer left"

            elif mode.value == 90:                      # right
                throttle = 100
                steering = 50
                mode_message = "Steer right"

            elif mode.value == 28:                      # OK — halt
                throttle = 0
                steering = 0
                mode_message = "HALT"
                print("HALT")

            elif mode.value == 13:                      # # — shutdown
                print("Shutting down")
                update_controls(0, 0)
                update_display("Shutdown", 0, 0)
                picam2.stop()
                picam2.close()
                cv2.destroyAllWindows()
                GPIO.cleanup()
                sys.exit(0)

            else:
                throttle = 0
                steering = 0
                mode_message = "Invalid input"
                print("Unrecognized command:", mode.value)

            update_controls(throttle, steering)
            update_display(mode_message, throttle, steering)

    except KeyboardInterrupt:
        print("Interrupted — shutting down.")

    finally:
        safe_stop()
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
