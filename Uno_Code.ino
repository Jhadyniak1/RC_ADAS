// ============================================================
//  4WD Differential Steering — Adafruit Motor Shield v2
//  Arduino Uno / Nano
//
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
const uint8_t BASE_SPEED  = 180;   // default drive speed
const uint8_t TURN_SPEED  = 150;   // inner wheel speed while turning
const uint8_t PIVOT_SPEED = 160;   // speed used for point turns

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

// ─────────────────────────────────────────────────────────────
//  High-level movement functions
// ─────────────────────────────────────────────────────────────

/**
 * Drive straight forward.
 * @param speed  0–255
 */
void driveForward(uint8_t speed = BASE_SPEED) {
  setLeft(speed, FORWARD);
  setRight(speed, FORWARD);
}

/**
 * Drive straight backward.
 * @param speed  0–255
 */
void driveBackward(uint8_t speed = BASE_SPEED) {
  setLeft(speed, BACKWARD);
  setRight(speed, BACKWARD);
}

/**
 * Gradual left turn while moving forward.
 * The right side drives at full speed; the left side is slowed.
 * @param outerSpeed  Speed of the faster (right) side
 * @param innerSpeed  Speed of the slower (left) side — use 0 to pivot on left wheels
 */
void turnLeft(uint8_t outerSpeed = BASE_SPEED, uint8_t innerSpeed = TURN_SPEED) {
  setLeft(innerSpeed, FORWARD);
  setRight(outerSpeed, FORWARD);
}

/**
 * Gradual right turn while moving forward.
 * The left side drives at full speed; the right side is slowed.
 */
void turnRight(uint8_t outerSpeed = BASE_SPEED, uint8_t innerSpeed = TURN_SPEED) {
  setLeft(outerSpeed, FORWARD);
  setRight(innerSpeed, FORWARD);
}

/**
 * Pivot (point-turn) left in place.
 * Left wheels reverse, right wheels forward.
 */
void pivotLeft(uint8_t speed = PIVOT_SPEED) {
  setLeft(speed, BACKWARD);
  setRight(speed, FORWARD);
}

/**
 * Pivot (point-turn) right in place.
 * Right wheels reverse, left wheels forward.
 */
void pivotRight(uint8_t speed = PIVOT_SPEED) {
  setLeft(speed, FORWARD);
  setRight(speed, BACKWARD);
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

/**
 * Mixed drive: combine throttle and steering into left/right speeds.
 *
 * @param throttle  -255 (full reverse) to +255 (full forward)
 * @param steering  -255 (full left)    to +255 (full right)
 *
 * Example sources: joystick, RC receiver, serial commands, etc.
 */
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


  // ── Example: use the mixer with fixed values ──
  // drive(200, -100);  // forward + steer left
  // delay(1000);
  // drive(0, 0);       // stop



void setup() {
  Serial.begin(9600);
    Serial.println("4WD Motor Shield v2 — Starting...");

 /// if (!AFMS.begin()) {
  ///  Serial.println("ERROR: Motor Shield not found. Check wiring/I2C address.");
   /// while (true); // halt
  }

  Serial.println("Motor Shield OK.");
}

int speed = 0;
int direction = 0;
void loop() {
 
 // If the Pi sends something, read and echo it back
  if (Serial.available()) {
      String data = Serial.readStringUntil('\n');  // read until newline

        int commaIndex = data.indexOf(',');          // find the comma

        if (commaIndex != -1) {
            value1 = data.substring(0, commaIndex).toInt();
            value2 = data.substring(commaIndex + 1).toInt();
  }
 analogWrite(value1,value2);
///  drive(speed, direction)

}
