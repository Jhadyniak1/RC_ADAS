#source env/bin/activate
#https://ignorantofthings.com/receiving-infrared-on-the-raspberry-pi-with-python/
#sudo ir-keytable -p all (to enable all protocols)
#sudo ir-keytable -t (to test)
#!/usr/bin/python
################################Package installs######################################################
import os
os.environ["TFLITE_DISABLE_XNNPACK"] = "1"
import threading
import tflite_runtime.interpreter as tflite

import sys  
import evdev
from time import sleep
import serial

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont

import adafruit_ssd1306

import time
import smbus
from picamera2 import Picamera2
import numpy as np
import cv2
import RPi.GPIO as GPIO
############################LCD Screen Setup#############################################################
WIDTH = 128
HEIGHT = 64  
BORDER = 5
i2c = board.I2C()  # uses board.SCL and board.SDA
oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C)
# Clear display
oled.fill(0)
oled.show()
# Create blank image for drawing
# Make sure to create image with mode '1' for 1-bit color
image = Image.new("1", (oled.width, oled.height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a white background
draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
# Load default font
font = ImageFont.load_default()  ### https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
######################Serial channel initalization#################################
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  #'/dev/ttyACM0 if using the actual USB port, /dev/ttyS0 for wires'
##############################Object detection setup#####################################################################
MODEL_PATH     = "detect.tflite"
LABELS_PATH    = "labelmap.txt"
CAPTURE_SIZE   = (640, 480)   # camera resolution
INPUT_SIZE     = (300, 300)   # SSD MobileNet v1 expects 300×300
CONF_THRESHOLD = 0.5          # raise to reduce false positives
MAX_DETECTIONS = 5

# Only fire the callback for these labels (None = all classes)
TARGET_LABELS = {"stop sign", "traffic light", "person", "car"} #Adjust as needed

# BGR colours per label for bounding boxes
COLOURS = [
    (220, 80,  80),  (80,  200, 80),  (80,  80,  220),
    (220, 180, 40),  (180, 40,  220), (40,  220, 180),
]
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


######################IR Remote Setup######################################################
# returns path of gpio ir receiver device
def get_ir_device():
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if (device.name == "gpio_ir_recv"):
            print("Using device", device.path, "\n")
            return device

    print("No device found!")
    sys.exit()

# returns the most recent InputEvent instance
# returns NoneType if no events available
def get_last_event(dev):
    try:
        for event in dev.read():  # iterate through all queued events
            if (event.value > 0):
                last_event = event
    except BlockingIOError:  # no events to be read
        last_event = None

    return last_event
################System update functions########################
def update_controls(throttle, steering):
    ser.write(f'{throttle},{steering}\n'.encode())
    return 0


def update_display(mode, throttle, steering):
    draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
    text = "ECE 4415 Group 4"
    draw.text((0, 0), text, font=font, fill=255)
    text1 = "Mode:" + mode
    draw.text((0, 15), text1, font=font, fill=255)
    text2 = "Throttle: " + str(throttle)
    draw.text((0, 35), text2, font=font, fill=255)
    text3 = "Steering: " + str(steering)
    draw.text((0, 45), text3, font=font, fill=255)
    # Display image
    oled.image(image)
    oled.show()

#############Lane keep functions###################################
def adaptive_thresholding(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return binary
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width, height // 2), (0, height // 2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)

def preprocess_image(frame):
    masked = adaptive_thresholding(frame)
    blur = cv2.GaussianBlur(masked, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lanes(frame):
    edges = preprocess_image(frame)
    edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold = 50, minLineLength=50, maxLineGap=20)
    lane_center = frame.shape[1] // 2
    left_lane, right_lane = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x2 - x1) == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            mid_x = (x1 + x2) // 2
            w = frame.shape[1]

            # Tighter slope range: ignore near-horizontal and near-vertical lines
            if not (0.4 < abs(slope) < 2.5):
                continue
        
            if slope < 0 and mid_x < w * 0.6:   # left lane: negative slope, left 60%
                left_lane.append(mid_x)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            elif slope > 0 and mid_x > w * 0.4:  # right lane: positive slope, right 60%
                right_lane.append(mid_x)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

    left_center = np.mean(left_lane) if left_lane else None
    right_center = np.mean(right_lane) if right_lane else None
    detected_center = int((left_center + right_center) / 2) if left_center and right_center else lane_center
    return frame, lane_center, detected_center

# need to add PI steering not proportional steering
def lane_keep(device):
    while get_last_event(device) is None:
        throttle = 0  # initialise with safe defaults
        steering = 0
        Kp = 3 #proportional gain
        frame = picam2.capture_array()
        lane_frame, lane_center, detected_center = detect_lanes(frame)
        error = lane_center - detected_center
        if error > 10:
            throttle = 100
            steering = Kp*error
        elif error < -10:
            throttle = 100
            steering = Kp*error
        else:
            throttle = 100
            steering = 0

        update_controls(throttle, steering)
        update_display("Lane keep", throttle, steering)
        cv2.imshow("Lane Detection", lane_frame)
        print(f"Lane Error: {error}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(0.2)

##########################Object avoidance function############################
def object_avoidance():
    throttle = 0
    steering = 0
    return throttle, steering
###########################Object detection###############################
def object_detection(device):
    def on_det(label, conf, bbox):
        print(f"  {label:<20s} {conf:.0%}  {bbox}")

    detector = ObjectDetector(on_detection=on_det)
    detector.start()
    while get_last_event(device) is None:
        frame = detector.get_latest_frame()
        if frame is not None:
            cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        throttle = 0 
        steering = 0
        
        update_controls(throttle, steering)
        update_display("Object Detection", throttle, steering)
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

###############################Adaptive cruise########################################
def adaptive_cruise():
    return throttle, steering

###############Project initializations for main script###########################

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"} 
) #was previously {"size": (640, 480), "format": "RGB888"}
picam2.preview_configuration.align()
picam2.configure(config)
picam2.start()
sleep(2)
##################Main loop execution###############################
def main():
    throttle = 0
    steering = 0
    mode_message = "Unknown"  
    update_controls(throttle,steering)
    device = get_ir_device()
    draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
    text = "Waiting for input....."
    draw.text((0, 0), text, font=font, fill=255)
    # Display image
    oled.image(image)
    oled.show()
    while True:
        key = get_last_event(device)
        if key is not None:
            mode = key
            print("Received command:", mode.value, "\n")
            

            if mode.value == 69:  # Lane keep (key number 1)
                print("Lane keep mode")
                mode_message = "Lane keep"
                lane_keep(device)

            elif mode.value == 70:  # Object avoidance (key number 2)
                throttle, steering = object_avoidance() 
                print("Object avoidance mode")
                mode_message = "Object Avoidance"

            elif mode.value == 71:  # Object detection (key number 3)
                throttle, steering, object = object_detection(device)  
                print("Object detection mode")
                mode_message = "Object Detection"

            elif mode.value == 68:  # Adaptive cruise (key number 4)
                adaptive_cruise(device)  
                mode_message = "Adaptive cruise"

            elif mode.value == 24:  # Up key
                throttle = 100
                steering = 0
                mode_message = "Drive forward"

            elif mode.value == 82:  # Down key
                throttle = -100
                steering = 0
                mode_message = "Drive back"

            elif mode.value == 8:  # Left key
                throttle = 100
                steering = -50
                mode_message = "Steer left"

            elif mode.value == 90:  # Right key
                throttle = 100
                steering = 50  
                mode_message = "Steer right"

            elif mode.value == 28:  # OK
                print("HALT")
                throttle = 0
                steering = 0
                mode_message = "HALT"


            elif mode.value == 13:  # Pound sign
                print("Shutting down")
                picam2.stop()
                picam2.close()
                cv2.destroyAllWindows()
                update_controls(0, 0)
                update_display("Shutdown", 0, 0)
                sys.exit()

            else:  # Unknown command
                update_controls(0, 0)
                update_display("Invalid input", 0, 0)
                print("Unrecognized command")

            update_controls(throttle, steering)
            update_display(mode_message, throttle, steering)

        

if __name__ == "__main__":
    main()
