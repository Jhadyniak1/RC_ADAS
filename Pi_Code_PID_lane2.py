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

    detector = ObjectDetector(cam=picam2)
    detector.start()
    detections = detector.get_latest_detections()  # [(label, conf, bbox), ...]
    frame      = detector.get_latest_frame()        # annotated BGR frame
    detector.stop()
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
        cam                   = None,
    ):
        self.capture_size   = capture_size
        self.input_size     = input_size
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.target_labels  = target_labels
        self.labels         = load_labels(labels_path)

        # TFLite interpreter
        self.interp    = tflite.Interpreter(model_path=model_path, num_threads=4)
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
            print("[ObjectDetector] Warning: falling back to positional order. "
                  f"Found: {list(name_map)}")
            s = sorted(self.interp.get_output_details(), key=lambda d: d["index"])
            self._idx_boxes   = s[0]["index"]
            self._idx_classes = s[1]["index"]
            self._idx_scores  = s[2]["index"]
            self._idx_count   = s[3]["index"] if len(s) > 3 else None

        # Camera
        self._cam      = cam
        self._owns_cam = (cam is None)

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
            t0        = time.time()
            frame_rgb = self._cam.capture_array()
            dets      = self._infer(frame_rgb)

            filtered = [
                d for d in dets
                if d[1] >= self.conf_threshold
                and (self.target_labels is None or d[0] in self.target_labels)
            ][:self.max_detections]

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            annotated = self._annotate(frame_bgr, filtered, self._fps)
            fps       = 1.0 / max(time.time() - t0, 1e-6)

            with self._lock:
                self._latest_frame = annotated
                self._latest_dets  = filtered
                self._fps          = fps

    def _infer(self, frame_rgb):
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
    sleep(0.15)
    try:
        for _ in dev.read():
            pass
    except BlockingIOError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# System helpers
# ═══════════════════════════════════════════════════════════════════════════════

def update_controls(throttle, steering):
    ser.write(f'{throttle},{steering}\n'.encode())


def update_display(mode, throttle, steering):
    draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
    draw.text((0,  0), "ECE 4415 Group 4",      font=font, fill=255)
    draw.text((0, 15), f"Mode: {mode}",          font=font, fill=255)
    draw.text((0, 35), f"Throttle: {throttle}",  font=font, fill=255)
    draw.text((0, 45), f"Steering: {steering}",  font=font, fill=255)
    oled.image(image)
    oled.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Lane keeping
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_thresholding(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)


def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array([[(0, h), (w, h), (w, h // 2), (0, h // 2)]], np.int32)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edges, mask)


def preprocess_image(frame):
    masked = adaptive_thresholding(frame)
    blur   = cv2.GaussianBlur(masked, (5, 5), 1)
    edges  = cv2.Canny(blur, 50, 150)
    return region_of_interest(edges)


def detect_lanes(frame, prev_center=None):
    h, w, _ = frame.shape
    roi = frame[h//2:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── BLUE MASK ─────────────────────────────
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ── RED MASK (two ranges in HSV) ──────────
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # ── BLACK MASK (low value, low saturation) ─
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Combine all lane colors
    mask = cv2.bitwise_or(blue_mask, red_mask)
    mask = cv2.bitwise_or(mask, black_mask)

    # ── Clean mask ─────────────────────────────
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # ── Contours ───────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_x = []
    right_x = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:  # higher threshold to reject noise
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        cx = x + cw // 2

        # Reject weird horizontal blobs
        if ch < 20:
            continue

        if cx < w * 0.5:
            left_x.append(cx)
            cv2.rectangle(roi, (x, y), (x+cw, y+ch), (0,255,0), 2)
        else:
            right_x.append(cx)
            cv2.rectangle(roi, (x, y), (x+cw, y+ch), (255,0,0), 2)

    # ── Compute centers ────────────────────────
    left_center = int(np.mean(left_x)) if left_x else None
    right_center = int(np.mean(right_x)) if right_x else None
    lane_center = w // 2

    # ── Robust center logic ────────────────────
    if left_center is not None and right_center is not None:
        detected_center = (left_center + right_center) // 2
        confidence = 1.0

    elif left_center is not None:
        detected_center = left_center + 200
        confidence = 0.5

    elif right_center is not None:
        detected_center = right_center - 200
        confidence = 0.5

    else:
        detected_center = prev_center
        confidence = 0.0

    # ── Visualization ──────────────────────────
    if detected_center is not None:
        cv2.line(roi, (detected_center, 0), (detected_center, h//2), (0,255,255), 2)

    cv2.line(roi, (lane_center, 0), (lane_center, h//2), (0,0,255), 2)

    frame[h//2:, :] = roi

    return frame, lane_center, detected_center, confidence

def lane_keep(device, picam2):
Kp = 0.8
Ki = 0.02
Kd = 0.1
integral = 0
prev_error = 0
prev_time = time.time()
smoothed_center = None
alpha = 0.2
prev_center = None
while get_last_event(device) is None:
    now = time.time() #need to verify this makes sense
    dt = now - prev_time
    prev_time = now

    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    lane_frame, lane_center, detected_center, confidence = detect_lanes(frame_bgr, prev_center)

if detected_center is not None:
    prev_center = detected_center
    if detected_center is None:
        update_controls(0, 0)
        continue

    if smoothed_center is None:
        smoothed_center = detected_center
    else:
        smoothed_center = alpha * detected_center + (1 - alpha) * smoothed_center

    error = lane_center - smoothed_center

    integral += error * dt
    integral = max(min(integral, 100), -100)

    derivative = (error - prev_error) / dt if dt > 0 else 0
    prev_error = error

    steering = Kp * error + Ki * integral + Kd * derivative
    steering = int(max(min(steering, 100), -100))

    # Adaptive speed
    base_speed = 80
    min_speed = 40
    turn_factor = abs(steering) / 100.0
    throttle = int(base_speed * (1 - 0.7 * turn_factor))
    throttle = max(throttle, min_speed)
    if confidence < 0.5:
    throttle *= 0.5  # slow down if uncertain
    update_controls(throttle, steering)
    update_display("Lane keep", throttle, steering)
        cv2.imshow("Lane Detection", lane_frame)
        print(f"Lane Error: {error}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(0.05)   # ~20 Hz

    cv2.destroyWindow("Lane Detection")



# ═══════════════════════════════════════════════════════════════════════════════
# Object detection mode
# ═══════════════════════════════════════════════════════════════════════════════

def object_detection(device, picam2, targets, conf_threshold=0.6):
    """
    Runs object detection and fires a callback when a target is seen.

    targets : dict  — { "label": callback_fn, ... }
                      callback_fn(confidence, bbox) is called on detection
    Example:
        object_detection(device, picam2, {
            "stop sign": on_stop_sign,
            "person":    on_person,
        })
    """
    detector = ObjectDetector(cam=picam2)
    detector.start()
    flush_ir_events(device)

    try:
        while get_last_event(device) is None:
            for label, confidence, bbox in detector.get_latest_detections():
                if label in targets and confidence >= conf_threshold:
                    targets[label](confidence, bbox)

            frame = detector.get_latest_frame()
            if frame is not None:
                cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            sleep(0.05)
    finally:
        detector.stop()
        cv2.destroyWindow("Object Detection")


# ═══════════════════════════════════════════════════════════════════════════════
# Stub modes
# ═══════════════════════════════════════════════════════════════════════════════

def object_avoidance(device):
    flush_ir_events(device)
    # TODO: implement
    pass


def adaptive_cruise(device):
    flush_ir_events(device)
    
    TRIG = 23
    ECHO = 24

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    def get_distance_in():
        GPIO.output(TRIG, False)
        time.sleep(0.0002)

        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        pulse_start = time.time()
        pulse_end   = time.time()

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()

        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance_cm = pulse_duration * 17150
        distance_in = distance_cm / 2.54
        return round(distance_in, 2) #I like displaying and working with inches

    print("Adaptive Cruise Control ACTIVE")

    try:
        while get_last_event(device) is None:
            dist = get_distance_in()
            if dist < 2: #right now this stops below 2 inches and peaks at 20 inches, we can adjust if needed
                speed = 0
            elif dist > 20:
                speed = 255
            else:
                speed = int((dist - 2) * (255 / 18))

            throttle = speed
            steering = 0

            update_controls(throttle, steering)
            update_display("Adaptive Cruise", throttle, steering)

            print(f"Distance: {dist} in | Throttle: {throttle}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    print("Exiting Adaptive Cruise Control")



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

    throttle     = 0
    steering     = 0
    mode_message = "Waiting..."
    update_controls(throttle, steering)
    update_display(mode_message, throttle, steering)

    device = get_ir_device()

    # ── Object detection callbacks ────────────────────────────────────────────
    def on_stop_sign(confidence, bbox):
        print(f"Stop sign detected ({confidence:.0%}) — stopping")
        update_controls(0, 0)
        update_display("STOP SIGN", 0, 0)
        sleep(3)

    def on_person(confidence, bbox):
        print(f"Person detected ({confidence:.0%}) — stopping")
        update_controls(0, 0)
        update_display("PERSON", 0, 0)
        sleep(1)

    def on_traffic_light(confidence, bbox):
        print(f"Traffic light detected ({confidence:.0%})")
        # Add your traffic light logic here

    def on_car(confidence, bbox):
        print(f"Car detected ({confidence:.0%}) — slowing down")
        update_controls(50, 0)
        update_display("CAR AHEAD", 50, 0)

    detection_targets = {
        "stop sign":     on_stop_sign,
        "person":        on_person,
        "traffic light": on_traffic_light,
        "car":           on_car,
    }
    # ─────────────────────────────────────────────────────────────────────────

    try:
        while True:
            key = get_last_event(device)
            if key is None:
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
                object_detection(device, picam2, detection_targets)
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
        update_controls(0, 0)
        picam2.stop()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()
