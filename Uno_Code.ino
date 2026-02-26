void setup() {
  Serial.begin(9600);
}


void loop() {

 // If the Pi sends something, read and echo it back
  if (Serial.available()) {
    String received = Serial.readStringUntil('\n');
    Serial.println("Arduino got: " + received);
  }

  delay(1000);
}
