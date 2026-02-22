## install dependencies via terminal on raspi
#pip install adafruit-circuitpython-rplidar
### https://learn.adafruit.com/slamtec-rplidar-on-pi/cpython-on-raspberry-pi
import os
from math import cos, sin, pi, floor
from adafruit_rplidar import RPLidar
import matplotlib.pyplot as plt
import numpy as np


# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, BAUDRATE = 115200)

# used to scale data to fit on the screen
max_distance = 0
scan_data = [0]*360

try:
    print(lidar.info)
    for scan in lidar.iter_scans():
        for (_, angle, distance) in scan:
            scan_data[min([359, floor(angle)])] = distance
        process_data(scan_data)

except KeyboardInterrupt:
    print('Stoping.')
    lidar.stop()
    lidar.disconnect()


def process_data(data):
    global max_distance
    for angle in range(360):
        distance = data[angle]
        if distance > 0:                  # ignore initially ungathered data points
            max_distance = max([min([5000, distance]), max_distance])
            radians = angle * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            point = (160 + int(x / max_distance * 119), 120 + int(y / max_distance * 119))
            
            r = distance
            theta = radians
            
            fig, axs = plt.subplots(2, 1, figsize=(5, 8), subplot_kw={'projection': 'polar'},
                                    layout='constrained')
            ax = axs[0]
            ax.plot(theta, r)
            ax.set_rmax(10)
            ax.set_rticks([0.5, 1, 1.5, 2])  # Fewer radial ticks
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.grid(True)
            ax.set_title("Lidar live data read", va='bottom')
            plt.show()
