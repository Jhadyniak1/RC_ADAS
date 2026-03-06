#source env/bin/activate
#https://ignorantofthings.com/receiving-infrared-on-the-raspberry-pi-with-python/
#sudo ir-keytable -p all (to enable all protocols)
#sudo ir-keytable -t (to test)
#!/usr/bin/python

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
HEIGHT = 64  # Change to 64 if needed
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
font = ImageFont.load_default(size = 14) ### https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
#font1 = ImageFont.load("arial.pil")

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) #'/dev/ttyACM0 if using the actual USB port, /dev/ttyS0 for wires'
# returns path of gpio ir receiver device
def get_ir_device():
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if (device.name == "gpio_ir_recv"):
            print("Using device", device.path, "\n")
            return device

    print("No device found!")
    sys.exit()

# returns a generator object that yields InputEvent instances
# raises BlockingIOError if no events available, which much be caught
def get_all_events(dev):
    return dev.read()

# returns the most recent InputEvent instance
# returns NoneType if no events available
def get_last_event(dev):
    try:
        for event in dev.read():	# iterate through all queued events
            if (event.value > 0):
                last_event = event
    except BlockingIOError: # no events to be read
        last_event = None

    return last_event

# returns the next InputEvent instance
# blocks until event is available
def get_next_event(dev):
    while(True):
    	event = dev.read_one()
    	if (event):
    		return event
            
def update_controls(throttle,steering):
    controls = f"{throttle},{steering}\n"
    ser.write(controls.encode('utf-8'))
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
    detected_center = int((left_center + right_center) / 2) if left_center and right_center else lane_center #lane_center
    return frame, lane_center, detected_center


#need to add dynamic steering not step steering

def lane_keep():
    picam2 = Picamera2()
    picam2.start()
    i = 0
    while i<10:
        frame = picam2.capture_array()
        lane_frame, lane_center, detected_center = detect_lanes(frame)
        error = lane_center - detected_center
         if error > 20:
            #PWM.set_motor_model(1500, 1500, -400, -400)
            throttle = 100
            steering = 50 #not sure if needs to be positive or negative
        elif error < -20:
            #PWM.set_motor_model(-400, -400, 1500, 1500)
            throttle = 100
            steering = -50 
        else:
            #PWM.set_motor_model(500, 500, 500, 500)
            throttle = 100
            steering = 0 

        cv2.imshow("Lane Detection", lane_frame)
        print(f"Lane Error: {error}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
        i = i + 1
    picam2.stop()
    cv2.destroyAllWindows()

    return throttle, steering 

def object_avoidance():


    return throttle, steering 

def object_detection():


    return throttle, steering 

def adaptive_cruise():


    return throttle, steering 

throttle = 0
steering = 0
def main():
    sleep(2)
    device = get_ir_device()
    while True:
        mode = get_last_event(device)
        if mode is not None:
            print("Received command:", mode.value, "\n")
        
            if mode.value == 69: # Lane keep (key number 1)
                {throttle, steering} = lane_keep()
                print("Lane keep mode")
                mode_message = "Lane keep"
                
            elif mode.value == 70: # Object avoidance (key number 2)
                {throttle, steering} = object_avoidance()
                print("Object avoidance mode")
                mode_message = "Object Avoidance"
                
            elif mode.value == 71: # Object detection (key number 3)
                {throttle, steering} = object_detection()
                print("Object avoidance mode")
                mode_message = "Object Detection"
                
            elif mode.value == 68: # Adaptive cruise (key number 4)
                {throttle, steering} = adaptive_cruise()
                controls = f"{throttle},{steering}\n"
                ser.write(controls.encode('utf-8'))
                mode_message = "Adaptive cruise"
                
            elif mode.value == 24: # Up key
                throttle = 150 
                steering = 0 
                mode_message = "Drive forward"
                
            elif mode.value == 82: # Down key
                throttle = -150 
                steering = 0 
                mode_message = "Drive back"
                
            elif mode.value == 8: # Left key
                throttle = 100 
                steering = -50
                mode_message = "Steer left"
            
            elif mode.value == 90: # Right key
                throttle = 100 
                steering = -50
                mode_message = "Steer right"

            elif mode.value == 13: #Pound sign
                print("Shutting down")
                sys.exit()
            
            else:  #Unknown command
                print("Unrecognized command")

            update_controls(throttle, steering)
            update_display(mode_message, throttle, steering)
            
    sleep(0.5)
if __name__ == "__main__":
    main()
