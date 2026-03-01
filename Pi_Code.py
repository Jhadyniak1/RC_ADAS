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
def main():
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) #'/dev/ttyACM0 if using the actual USB port, /dev/ttyS0 for wires'
    sleep(2)
    device = get_ir_device()
    while True:
        mode = get_last_event(device)
        if mode is not None:
            print("Received command:", mode.value, "\n")
        
            if mode.value == 69: # Lane keep!
                message = f"{5},{255}\n"
                print("Lane keep mode")
                ser.write(message.encode('utf-8'))
            elif mode.value == 70: # Object avoidance 
                message = f"{5},{125}\n"
                ser.write(message.encode('utf-8'))
            elif mode.value == 71: # Object detection 
                message = f"{6},{255}\n"
                ser.write(message.encode('utf-8'))
            elif mode.value == 68: # Adaptive cruise
                message = f"{10},{255}\n"
                ser.write(message.encode('utf-8'))
            elif mode.value == 24: # Up key
                message = f"{11},{255}\n"
                ser.write(message.encode('utf-8'))
            elif mode.value == 82: # Down key
               message = f"{11},{255}\n"
               ser.write(message.encode('utf-8'))
            elif mode.value == 8: # Left key
                message = f"{11},{255}\n"
                ser.write(message.encode('utf-8'))
            elif mode.value == 90: # Right key
                message = f"{11},{255}\n"
                ser.write(message.encode('utf-8'))
            elif mode.value == 13: #Pound sign
                print("Shutting down")
                sys.exit()
            else:  #Unknown command
                print("Unrecognized command")
        
        
        else:
            print("No commands received.\n")
        sleep(0.5)
if __name__ == "__main__":
    main()
