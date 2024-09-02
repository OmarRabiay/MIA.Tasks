#include <Wire.h>

const int MPU_ADDR = 0x68;
int16_t gyroZ;
float yaw = 0.0;
unsigned long lastTime = 0;

void setup() {
  Wire.begin();
  Serial.begin(9600);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission();
  
  lastTime = millis();
}

void loop() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x43);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);

  Wire.read(); // Ignore gyroX
  Wire.read();
  Wire.read(); // Ignore gyroY
  Wire.read();
  gyroZ = Wire.read() << 8 | Wire.read();

  float gyroZrate = gyroZ / 131.0;
  unsigned long currentTime = millis();
  float dt = (currentTime - lastTime) / 1000.0;
  lastTime = currentTime;
  yaw += gyroZrate * dt;

  Serial.print("Yaw: ");
  Serial.println(yaw);

  delay(100);
}
