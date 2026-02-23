#for use on raspberru pi, the below commented out commands are to be run from terminal prior to trying to execute the program
############################
#sudo apt update
#sudo apt install python3-venv
#python3 -m venv venv
#source venv/bin/activate
#pip install rplidar-roboticia
#pip install numpy
#pip install matplotlib
#############################
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import matplotlib.pyplot as plt

lidar = RPLidar('/dev/ttyUSB0', baudrate=115200)
time.sleep(2)
MAX_DISTANCE = 3000

# Set up plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7,7))
ax.set_facecolor('#0a0c10')
fig.patch.set_facecolor('#0a0c10')
ax.set_ylim(0, MAX_DISTANCE)
ax.set_title('RPLidar A1 Live', color='cyan', pad=15)
ax.tick_params(colors='#444')
ax.grid(color='#1e2535')

angles    = np.zeros(360)
distances = np.zeros(360)
scatter   = ax.scatter(angles, distances, s=2, c='cyan', alpha=0.7)

plt.ion()
plt.show()
try:
    for scan in lidar.iter_scans(max_buf_meas=5000):
        try:
            for _, angle, distance in scan:
                if 0 < distance < MAX_DISTANCE:
                    idx = int(angle) % 360
                    angles[idx]    = np.radians(angle)
                    distances[idx] = distance

            scatter.set_offsets(np.c_[angles, distances])
            fig.canvas.flush_events()
            plt.pause(0.001)
            time.sleep(1)

        except RPLidarException:
            continue  # skip bad packets, keep going
except RPLidarException:
    pass      # skip bad packets, keep going
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
