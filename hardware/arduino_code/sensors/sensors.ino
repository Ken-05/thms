#include <Wire.h>

// Assume MPU6050 or ADXL345 for accelerometers
#include <Adafruit_MPU6050.h>
Adafruit_MPU6050 accelCabin;
Adafruit_MPU6050 accelBody;

// Temperature sensor pins
const int engineTempPin = A0;
const int frontBrakeTempPin = A1;
const int rearBrakeTempPin = A2;
const int alternatorTempPin = A3;
const int clutchTempPin = A4;

// Float level sensor pins
const int hydraulicLevelPin = A5;      // Analog float sensor
const int transmissionLevelPin = 2;    // Digital optical level sensor

// Variables
unsigned long lastSampleTime = 0;
const unsigned long sampleInterval = 1000;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialize accelerometers
  if (!accelCabin.begin(0x68)) {
    Serial.println("Cabin accelerometer not found!");
  }
  if (!accelBody.begin(0x69)) {
    Serial.println("Body accelerometer not found!");
  }

  pinMode(transmissionLevelPin, INPUT_PULLUP);  // Digital optical level sensor

  // CSV header
  Serial.println("timestamp_ms,engine_temp_C,front_brake_temp_C,rear_brake_temp_C,alt_temp_C,clutch_temp_C,hydraulic_level_pct,transmission_level,onboard_ax,ay,az,body_ax,ay,az");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime >= sampleInterval) {
    lastSampleTime = now;

    // Read temps (convert voltage to °C as needed)
    float engineTemp = analogToTemp(analogRead(engineTempPin));
    float frontBrakeTemp = analogToTemp(analogRead(frontBrakeTempPin));
    float rearBrakeTemp = analogToTemp(analogRead(rearBrakeTempPin));
    float altTemp = analogToTemp(analogRead(alternatorTempPin));
    float clutchTemp = analogToTempClutch(analogRead(clutchTempPin));

    // Read level sensors
    float hydraulicLevel = analogToPercent(analogRead(hydraulicLevelPin));  // percent
    int transmissionLevel = digitalRead(transmissionLevelPin);  // 0 = low, 1 = sufficient

    // Read cabin accelerometer
    sensors_event_t acX, acY, acZ;
    accelCabin.getAccelerometerSensor()->getEvent(&acX, &acY, &acZ);

    // Read body frame accelerometer
    sensors_event_t abX, abY, abZ;
    accelBody.getAccelerometerSensor()->getEvent(&abX, &abY, &abZ);

    // Print CSV line
    Serial.print(now); Serial.print(",");
    Serial.print(engineTemp); Serial.print(",");
    Serial.print(frontBrakeTemp); Serial.print(",");
    Serial.print(rearBrakeTemp); Serial.print(",");
    Serial.print(altTemp); Serial.print(",");
    Serial.print(clutchTemp); Serial.print(",");
    Serial.print(hydraulicLevel); Serial.print(",");
    Serial.print(transmissionLevel); Serial.print(",");
    Serial.print(acX.acceleration.x); Serial.print(",");
    Serial.print(acY.acceleration.y); Serial.print(",");
    Serial.print(acZ.acceleration.z); Serial.print(",");
    Serial.print(abX.acceleration.x); Serial.print(",");
    Serial.print(abY.acceleration.y); Serial.print(",");
    Serial.println(abZ.acceleration.z);
  }
}

// Convert analog sensor reading to temperature 
float analogToTemp(int raw) {
  float voltage = raw * (5.0 / 1023.0);
  return (voltage - 0.5) * 100.0;  
}
float analogToTempClutch(int raw) {
  float voltage = raw * (5.0 / 488.0);
  return (voltage - 0.5) * 100.0;  
}

// Convert analog level sensor to percent 
float analogToPercent(int raw) {
  return (raw / 627.0) * 100.0;
}