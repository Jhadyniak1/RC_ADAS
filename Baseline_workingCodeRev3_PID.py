
import time
import smbus
from picamera2 import Picamera2
import numpy as np
import cv2
from pca9685 import PCA9685
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
import csv

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
    
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width, height // 2), (0, height // 2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)

def detect_lanes(frame):
    edges = preprocess_image(frame)
    edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=35, maxLineGap=10)
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
# Motor control code
class Motor:
    def __init__(self):
        self.pwm = PCA9685(0x40, debug=True)
        self.pwm.set_pwm_freq(50)
        #FL, BL, FR, BR
    def duty_range(self, duty1, duty2, duty3, duty4):
        if duty1 > 4095:
            duty1 = 4095
        elif duty1 < -4095:
            duty1 = -4095        
        if duty2 > 4095:
            duty2 = 4095
        elif duty2 < -4095:
            duty2 = -4095  
        if duty3 > 4095:
            duty3 = 4095
        elif duty3 < -4095:
            duty3 = -4095
        if duty4 > 4095:
            duty4 = 4095
        elif duty4 < -4095:
            duty4 = -4095
        return duty1,duty2,duty3,duty4
    def left_upper_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(0,0)
            self.pwm.set_motor_pwm(1,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(1,0)
            self.pwm.set_motor_pwm(0,abs(duty))
        else:
            self.pwm.set_motor_pwm(0,4095)
            self.pwm.set_motor_pwm(1,4095)
    def left_lower_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(3,0)
            self.pwm.set_motor_pwm(2,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(2,0)
            self.pwm.set_motor_pwm(3,abs(duty))
        else:
            self.pwm.set_motor_pwm(2,4095)
            self.pwm.set_motor_pwm(3,4095)
    def right_upper_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(6,0)
            self.pwm.set_motor_pwm(7,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(7,0)
            self.pwm.set_motor_pwm(6,abs(duty))
        else:
            self.pwm.set_motor_pwm(6,4095)
            self.pwm.set_motor_pwm(7,4095)
    def right_lower_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(4,0)
            self.pwm.set_motor_pwm(5,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(5,0)
            self.pwm.set_motor_pwm(4,abs(duty))
        else:
            self.pwm.set_motor_pwm(4,4095)
            self.pwm.set_motor_pwm(5,4095)
    def set_motor_model(self, duty1, duty2, duty3, duty4):
        duty1,duty2,duty3,duty4=self.duty_range(duty1,duty2,duty3,duty4)
        self.left_upper_wheel(duty1)
        self.left_lower_wheel(duty2)
        self.right_upper_wheel(duty3)
        self.right_lower_wheel(duty4)

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

class ErrorLogger:
    def __init__(self, log_to_file=True, filename="lane_errors.csv"):
        self.errors = []
        self.timestamps = []
        self.log_to_file = log_to_file
        self.filename = filename
        if log_to_file:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'Error'])

    def log(self, error, timestamp):
        self.errors.append(error)
        self.timestamps.append(timestamp)
        if self.log_to_file:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, error])

def plot_errors(logger):
    plt.figure(figsize=(10, 4))
    plt.plot(logger.timestamps, logger.errors, label='Lane Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (pixels)')
    plt.title('Lane Following Error Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def control_motors(lane_center, detected_center, pid, last_time):
    PWM = Motor()
    error = lane_center - detected_center
    current_time = time.time()
    dt = current_time - last_time if last_time else 0.1
    correction = pid.compute(error, dt)

    base_speed = 750 #tunable
    max_speed = 2000 #tunable
    min_speed = 400 #tunable

    left_speed = base_speed + correction
    print(f"Left speed command: {left_speed}")
    right_speed = base_speed - correction
    print(f"Left speed command: {right_speed}")

    left_speed = max(min_speed, min(max_speed, int(left_speed)))
    right_speed = max(min_speed, min(max_speed, int(right_speed)))

    PWM.set_motor_model(left_speed, left_speed, right_speed, right_speed)
    return error, current_time


def main():
    PWM = Motor()
    picam2 = Picamera2()
    picam2.start()

    pid = PIDController(kp=1, ki=0.75, kd=0.0)
    logger = ErrorLogger()
    last_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            lane_frame, lane_center, detected_center = detect_lanes(frame)
            error, last_time = control_motors(lane_center, detected_center, pid, last_time)
            logger.log(error, last_time)
            cv2.imshow("Lane Detection", lane_frame)
            print(f"Lane Error: {error}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("\nEnd of program")
