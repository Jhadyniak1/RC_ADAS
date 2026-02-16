import time
import smbus
from picamera2 import Picamera2
import numpy as np
import cv2
from pca9685 import PCA9685
import RPi.GPIO as GPIO

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
# Motor control code
class Motor:
    def __init__(self):
        self.pwm = PCA9685(0x40, debug=True)
        self.pwm.set_pwm_freq(50)
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

def control_motors(lane_center, detected_center):
    PWM = Motor()
    error = lane_center - detected_center
    
    if error > 20:
        PWM.set_motor_model(1500, 1500, -400, -400)
    elif error < -20:
        PWM.set_motor_model(-400, -400, 1500, 1500)
    else:
        PWM.set_motor_model(500, 500, 500, 500)
    return error

def main():
    PWM = Motor()
    picam2 = Picamera2()
    picam2.start()
    
    while True:
        frame = picam2.capture_array()
        lane_frame, lane_center, detected_center = detect_lanes(frame)
        error = control_motors(lane_center, detected_center)
        
        cv2.imshow("Lane Detection", lane_frame)
        print(f"Lane Error: {error}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
    
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("\nEnd of program")
