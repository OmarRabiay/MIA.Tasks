class PID {
  public:
    float Kp, Ki, Kd;
    float integral;
    float previousError;
    unsigned long lastTime;

    PID(float _Kp, float _Ki, float _Kd) {
      Kp = _Kp;
      Ki = _Ki;
      Kd = _Kd;
      integral = 0;
      previousError = 0;
      lastTime = millis();
    }

    float compute(float setPoint, float currentSpeed) {
      unsigned long currentTime = millis();
      float elapsedTime = (currentTime - lastTime) / 1000.0;
      
      if (elapsedTime <= 0.0) elapsedTime = 0.01;

      float error = setPoint - currentSpeed; // e(t) = SP (setpoint) – PV (process variable)

      integral += error * elapsedTime; 
      
      integral = constrain(integral, -255, 255);
      
      float derivative = (error - previousError) / elapsedTime;
      float output = Kp * error + Ki * integral + Kd * derivative; // u(t) = Kp *e(t) + Kp * d/dt e(t)

      output = constrain(output, 0, 255);

      previousError = error;
      lastTime = currentTime;
      return output;
    }
};

class SoftStartFilter {
  public:
    float alpha;
    float smoothedOutput;

    SoftStartFilter(float _alpha) {
      alpha = _alpha;
      smoothedOutput = 0;
    }

    float smooth(float currentOutput) {
      smoothedOutput = alpha * currentOutput + (1 - alpha) * smoothedOutput;
      
      if (isinf(smoothedOutput) || isnan(smoothedOutput)) {
        smoothedOutput = 0;
      }
      
      return smoothedOutput;
    }
};

PID motorPID(1.0, 0.5, 0.1);       // Kp, Ki, Kd
SoftStartFilter softStart(0.6);     // alpha 
const int motorPin = 9;             // Motor control pin
const int setPoint = 500;           // Desired motor speed

void setup() {
  Serial.begin(9600);
  pinMode(motorPin, OUTPUT);
}

void loop() {
  float currentSpeed = analogRead(A0);
  float pidOutput = motorPID.compute(setPoint, currentSpeed);
  float smoothedOutput = softStart.smooth(pidOutput);
  analogWrite(motorPin, constrain(smoothedOutput, 0, 255));

  Serial.print("Current Speed: ");
  Serial.print(currentSpeed);
  Serial.print(" PID Output: ");
  Serial.print(pidOutput);
  Serial.print(" Smoothed Output: ");
  Serial.println(smoothedOutput);

  delay(100);
}
