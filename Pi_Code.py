#source env/bin/activate
#https://ignorantofthings.com/receiving-infrared-on-the-raspberry-pi-with-python/
#sudo ir-keytable -p all (to enable all protocols)
#sudo ir-keytable -t (to test)
#!/usr/bin/python
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
WIDTH = 128
HEIGHT = 64  # Change to 64 if needed
BORDER = 5
i2c = board.I2C()  # uses board.SCL and board.SDA
oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C)
# Clear display.
oled.fill(0)
oled.show()
# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
image = Image.new("1", (oled.width, oled.height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a white background
draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
# Load default font.
font = ImageFont.load_default(size = 14) ### https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
#font1 = ImageFont.load("arial.pil")

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) #'/
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
    draw.text(
    (0, 0),
    text,
    font=font,
    fill=255,
)
   
    text1 = "Mode:" + mode
    draw.text(
                (0, 15),
                text1,
                font=font,
                fill=255)
    text2 = "Throttle: " + str(throttle)
    draw.text(
                (0, 35),
                text2,
                font=font,
                fill=255)
    text3 = "Steering: " + str(steering)
    draw.text(
        (0, 45),
        text3,
        font=font,
        fill=255)
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
throttle = 0
steering = 0
def main():
    #ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) #'/dev/ttyACM0 if using the actual USB port, /dev/ttyS0 for wires'
    sleep(2)
    device = get_ir_device()
    while True:
        mode = get_last_event(device)
        if mode is not None:
            print("Received command:", mode.value, "\n")
        
            if mode.value == 69: # Lane keep!
                throttle = 5 #need to be changed by the algorithm
                steering = 255 #need to be changed by the algorithm
                controls = f"{throttle},{steering}\n"
                print("Lane keep mode")
                mode_message = "Lane keep"
                
                
            elif mode.value == 70: # Object avoidance 
                throttle = 5 #need to be changed by the algorithm
                steering = 125 #need to be changed by the algorithm
                controls = f"{throttle},{steering}\n"
                ser.write(controls.encode('utf-8'))
                mode_message = "Object Avoidance"
                
            elif mode.value == 71: # Object detection 
                throttle = 6 #need to be changed by the algorithm
                steering = 255 #need to be changed by the algorithm
                controls = f"{throttle},{steering}\n"
                mode_message = "Object Detection"
                
            elif mode.value == 68: # Adaptive cruise
                throttle = 10 #need to be changed by the algorithm
                steering = 255 #need to be changed by the algorithm
                controls = f"{throttle},{steering}\n"
                ser.write(controls.encode('utf-8'))
                mode_message = "Adaptive cruise"
                
            elif mode.value == 24: # Up key
                throttle = 255 #need to be changed by the algorithm
                steering = 0 #need to be changed by the algorithm
                controls = f"{throttle},{steering}\n"
                mode_message = "Lane keep"
                
            elif mode.value == 82: # Down key
               controls = f"{11},{255}\n"
               ser.write(controls.encode('utf-8'))
               mode_message = "Lane keep"
                
            elif mode.value == 8: # Left key
                controls = f"{11},{255}\n"
                ser.write(controls.encode('utf-8'))
                mode_message = "Lane keep"
            
            elif mode.value == 90: # Right key
                controls = f"{11},{255}\n"
                ser.write(controls.encode('utf-8'))
                
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
