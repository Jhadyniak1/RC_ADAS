// ============================================================
//  4WD Differential Steering — Adafruit Motor Shield v2
//  Wiring:
//    Left  Front motor  → M1
//    Left  Rear  motor  → M2
//    Right Front motor  → M3
//    Right Rear  motor  → M4
//
//  Requires: Adafruit Motor Shield V2 Library
//  Install via Arduino Library Manager:
//    "Adafruit Motor Shield V2 Library"
// ============================================================

#include <Wire.h>
#include <Adafruit_MotorShield.h>

// ── Shield & motor objects ────────────────────────────────────
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); // default I2C addr 0x60

Adafruit_DCMotor *leftFront  = AFMS.getMotor(1);   // M1
Adafruit_DCMotor *leftRear   = AFMS.getMotor(2);   // M2
Adafruit_DCMotor *rightFront = AFMS.getMotor(3);   // M3
Adafruit_DCMotor *rightRear  = AFMS.getMotor(4);   // M4

// ── Speed constants (0–255) ───────────────────────────────────
const uint8_t BASE_SPEED  = 100;   // default drive speed
const uint8_t TURN_SPEED  = 100;   // inner wheel speed while turning
const uint8_t PIVOT_SPEED = 100;   // speed used for point turns

// ─────────────────────────────────────────────────────────────
//  Low-level helpers
// ─────────────────────────────────────────────────────────────

/** Set left-side motors to a given speed and direction. */
void setLeft(uint8_t speed, uint8_t dir) {
  leftFront->setSpeed(speed);
  leftRear->setSpeed(speed);
  leftFront->run(dir);
  leftRear->run(dir);
}

/** Set right-side motors to a given speed and direction. */
void setRight(uint8_t speed, uint8_t dir) {
  rightFront->setSpeed(speed);
  rightRear->setSpeed(speed);
  rightFront->run(dir);
  rightRear->run(dir);
}

/** Release (coast) all motors. */
void releaseAll() {
  leftFront->run(RELEASE);
  leftRear->run(RELEASE);
  rightFront->run(RELEASE);
  rightRear->run(RELEASE);
}

/**
 * Stop all motors (coast).
 */
void stopAll() {
  releaseAll();
}

/**
 * Brake all motors (active stop).
 */
void brakeAll() {
  leftFront->run(BRAKE);
  leftRear->run(BRAKE);
  rightFront->run(BRAKE);
  rightRear->run(BRAKE);
}

// ─────────────────────────────────────────────────────────────
//  Mixer — steer with a single throttle + steering value
// ─────────────────────────────────────────────────────────────

 * @param throttle  -255 (full reverse) to +255 (full forward)
 * @param steering  -255 (full left)    to +255 (full right)

void drive(int throttle, int steering) {
  // Mix
  int leftSpeed  = throttle + steering;
  int rightSpeed = throttle - steering;

  // Clamp to ±255
  leftSpeed  = constrain(leftSpeed,  -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);

  // Apply left side
  if (leftSpeed >= 0) {
    setLeft((uint8_t)leftSpeed, FORWARD);
  } else {
    setLeft((uint8_t)(-leftSpeed), BACKWARD);
  }

  // Apply right side
  if (rightSpeed >= 0) {
    setRight((uint8_t)rightSpeed, FORWARD);
  } else {
    setRight((uint8_t)(-rightSpeed), BACKWARD);
  }
}

void setup() {
Serial.begin(9600);
}

int throttle = 0;
int steering = 0;

void loop() {
// If the Pi sends something, read 
if (Serial.available()) {
String data = Serial.readStringUntil('\n');  // read until newline
int commaIndex = data.indexOf(',');          // find the comma
if (commaIndex != -1) {
throttle = data.substring(0, commaIndex).toInt();
steering = data.substring(commaIndex + 1).toInt();
  }
}
drive(throttle, steering);
}
