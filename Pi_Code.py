#source env/bin/activate

#!/usr/bin/python
import evdev
from time import sleep

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

def main():
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    sleep(2)
    device = get_ir_device()
    while True:
        mode = get_last_event(device)
        if mode is not None:
            print("Received command:", mode.value, "\n")
        else:
            print("No commands received.\n")
        
        if mode == 1:
            ser.write(b'1'\n)
        if mode == 2:
            ser.write(b'2'\n)
        if mode == 3:
            ser.write(b'3'\n)
        sleep(0.5)
            
if __name__ == "__main__":
    main()
