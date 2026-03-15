#source env/bin/activate
#https://ignorantofthings.com/receiving-infrared-on-the-raspberry-pi-with-python/
#sudo ir-keytable -p all (to enable all protocols)
#sudo ir-keytable -t (to test)
#!/usr/bin/python

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
# Load default font.
font = ImageFont.load_default()  ### https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
#font1 = ImageFont.load("arial.pil")

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  #'/dev/ttyACM0 if using the actual USB port, /dev/ttyS0 for wires'

# returns path of gpio ir receiver device
def get_ir_device():
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if (device.name == "gpio_ir_recv"):
            print("Using device", device.path, "\n")
            return device

    print("No device found!")
    sys.exit()


# raises BlockingIOError if no events available, which must be caught
def get_all_events(dev):
    return dev.read()

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

# returns the next InputEvent instance //// blocks until event is available
def get_next_event(dev):
    while(True):
        event = dev.read_one()
        if (event):
            return event

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

'''
Key, Hex, Dec
1 = 0x45, 69
2 = 0x46, 70
3 = 0x47, 71
4 = 0x44, 68
5 = 0x40, 64
6 = 0x43, 67
7 = 0x07, 7
8 = 0x15, 21
9 = 0x09, 9
0 = 0x19, 25
UP = 0x18, 24
LEFT = 0x08, 8
DOWN = 0x52, 82
RIGHT = 0x5a, 90
OK = 0x1c, 28
STAR = 0x16, 22
POUND = 0x0d, 13
'''

def adaptive_thresholding(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def preprocess_image(frame):
    masked = adaptive_thresholding(frame)
    blur = cv2.GaussianBlur(masked, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lanes(frame):
    edges = preprocess_image(frame)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=30, maxLineGap=10)
    lane_center = frame.shape[1] // 2
    left_lane, right_lane = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < -0.2:
                left_lane.append((x1 + x2) // 2)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            elif slope > 0.2:
                right_lane.append((x1 + x2) // 2)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

    left_center = np.mean(left_lane) if left_lane else None
    right_center = np.mean(right_lane) if right_lane else None
    detected_center = int((left_center + right_center) / 2) if left_center and right_center else lane_center
    return frame, lane_center, detected_center
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
sleep(2)
# need to add dynamic steering not step steering
def lane_keep(device):
    while get_last_event(device) is None:
        throttle = 0  # initialise with safe defaults
        steering = 0
        frame = picam2.capture_array()
        lane_frame, lane_center, detected_center = detect_lanes(frame)
        error = lane_center - detected_center
        if error > 20:
            throttle = 100
            steering = 50
        elif error < -20:
            throttle = 100
            steering = -50
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


def object_avoidance():
    return throttle, steering

def object_detection(device):
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    while get_last_event(device) is None:
        frame = picam2.capture_array()
        fgmask = fgbg.apply(frame)
        # Noise cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacle_detected = False
        for c in contours:
            if cv2.contourArea(c) > 2000:  # filter small noise
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                obstacle_detected = True

        throttle = 0 if obstacle_detected else 100
        steering = 0
        update_controls(throttle, steering)
        update_display("Object Detection", throttle, steering)
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def adaptive_cruise():
    return throttle, steering


throttle = 0
steering = 0


def main():
    update_controls(0,0)
    #picam2 = Picamera2()
   # picam2.start()
    sleep(2)
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
            mode_message = "Unknown"  # FIX 5: Initialise mode_message before the if/elif chain

            if mode.value == 69:  # Lane keep (key number 1)
                print("Lane keep mode")
                mode_message = "Lane keep"
                lane_keep(device)

            elif mode.value == 70:  # Object avoidance (key number 2)
                throttle, steering = object_avoidance() 
                print("Object avoidance mode")
                mode_message = "Object Avoidance"

            elif mode.value == 71:  # Object detection (key number 3)
                throttle, steering = object_detection()  
                print("Object detection mode")
                mode_message = "Object Detection"

            elif mode.value == 68:  # Adaptive cruise (key number 4)
                throttle, steering = adaptive_cruise()  
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
