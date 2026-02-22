from rplidar import RPLidar
import matplotlib.pyplot as plt
import numpy as np

PORT_NAME = '/dev/ttyUSB0'

lidar = RPLidar(PORT_NAME, baudrate=115200)

# Set up polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_facecolor('#0a0c10')
fig.patch.set_facecolor('#0a0c10')
ax.set_title('RPLidar Live Scan', color='cyan')
ax.tick_params(colors='gray')
scatter = ax.scatter([], [], s=2, c='cyan', alpha=0.8)

plt.ion()
plt.show()

try:
    for scan in lidar.iter_scans():
        angles    = np.radians([p[1] for p in scan])
        distances = [p[2] for p in scan]

        scatter.set_offsets(np.c_[angles, distances])
        scatter.set_array(np.array(distances))  # color by distance
        
        ax.set_ylim(0, max(distances) if distances else 1000)
        
        fig.canvas.flush_events()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    plt.close()
