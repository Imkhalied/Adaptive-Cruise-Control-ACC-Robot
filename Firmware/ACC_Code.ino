// =====================================================
// ROBOT 2 — FOLLOWER (PID + HC-SR04 + Edge Impulse AI)
// Maintains fixed distance behind Leader.
// Emergency brake if gap < 8cm.
// PID setpoint tracks Leader's actual gap in real time.
// Edge Impulse fusion classifier ("safe_distance" /
// "too_close" / "too_far") runs alongside the PID loop
// and can override/bias motor output as a safety layer.
//
// FIXED FOR ESP32 ARDUINO CORE 3.x:
// ledcSetup() / ledcAttachPin() were removed in core 3.x.
// New API attaches LEDC directly to a pin with ledcAttach(),
// and writes duty cycle with ledcWrite(pin, duty) — no more
// separate channel numbers to manage.
//
// ── EDGE IMPULSE SETUP (read this first) ────────────
// 1. Install the library: Arduino IDE > Sketch > Include
//    Library > Add .ZIP Library... > select the ORIGINAL
//    zip (ei-ai-project-arduino-1_0_2-impulse-_1.zip),
//    not the extracted folder. That registers it correctly
//    under the name "AI_project_inferencing".
// 2. This is a SENSOR FUSION model: 8 axes/frame, 50
//    frames/window, sampled at 50 Hz (20 ms/sample) -> a
//    1-second window per inference. Its DSP block uses an
//    FFT + low-pass filter (9.7 Hz cutoff), so the 50 Hz
//    sample rate must be hit consistently or accuracy will
//    degrade — this is why the loop below is restructured
//    to be non-blocking (millis()-based) instead of using
//    delay().
// 3. IMPORTANT — VERIFY THE FEATURE ORDER: the exported
//    library does NOT embed axis names for fusion projects,
//    so the exact order of the 8 values below is an
//    assumption, not a certainty. I picked the most likely
//    layout given your hardware (HC-SR04 + MPU6050) and the
//    labels (safe_distance/too_close/too_far):
//
//      [0] distance_filtered (cm)
//      [1] distance_raw      (cm)
//      [2] accX (m/s^2)
//      [3] accY (m/s^2)
//      [4] accZ (m/s^2)
//      [5] gyroX (deg/s)
//      [6] gyroY (deg/s)
//      [7] gyroZ (deg/s)
//
//    Open your Edge Impulse project > Data acquisition, and
//    check the sensor/axis order shown for your recorded
//    samples. If it differs, reorder ONLY the 8 lines inside
//    sampleFeatures() below — nothing else needs to change.
// =====================================================

#include <Wire.h>
#include <MPU6050.h>
#include <AI_project_inferencing.h>   // Edge Impulse: AI_project

// ── HC-SR04 ──────────────────────────────────────────
#define TRIG  5
#define ECHO 18

// ── Motor Pins ──────────────────────────────────────
#define ENA  25
#define IN1  26
#define IN2  27
#define ENB  14
#define IN3  12
#define IN4  13

#define PWM_FREQ 1000
#define PWM_RES  8

// ── PID Parameters (TUNE THESE) ─────────────────────
float Kp = 3.0;
float Ki = 0.05;
float Kd = 1.5;

const float TARGET_DIST   = 20.0;  // cm — desired following gap
const float ESTOP_DIST    =  8.0;  // cm — emergency brake threshold
const float TOO_FAR_DIST  = 60.0;  // cm — max range before full speed
const float MAX_PWM       = 220.0;
const float MIN_PWM       =  60.0;

// ── PID state ───────────────────────────────────────
float prevError  = 0;
float integral   = 0;
unsigned long prevTime = 0;

// ── MPU6050 ─────────────────────────────────────────
MPU6050 mpu;
int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;

// ── Sensor noise filter ─────────────────────────────
float distFiltered = TARGET_DIST;
const float ALPHA = 0.3;   // Low-pass: 0=frozen, 1=raw

// ── Control loop timing (non-blocking) ──────────────
const unsigned long CONTROL_INTERVAL_MS = 40;  // ~25 Hz, same as before
unsigned long lastControlMs = 0;

// =====================================================
// EDGE IMPULSE — feature buffer & classifier plumbing
// =====================================================
#define EI_SAMPLE_INTERVAL_MS (1000 / EI_CLASSIFIER_FREQUENCY)  // 20 ms @ 50 Hz
static float eiFeatures[EI_CLASSIFIER_RAW_SAMPLE_COUNT * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME]; // 50*8 = 400
static size_t eiFeatureIx = 0;
unsigned long lastEiSampleMs = 0;

// Latest classifier verdict, used by the control loop
enum FollowState { STATE_SAFE, STATE_TOO_CLOSE, STATE_TOO_FAR, STATE_UNKNOWN };
FollowState eiState = STATE_UNKNOWN;
float eiConfidence = 0.0f;

// Required by the Edge Impulse SDK: feeds raw features to the classifier
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, eiFeatures + offset, length * sizeof(float));
  return 0;
}

// Grabs one frame (8 values) of sensor data for the EI feature buffer.
// See the "VERIFY THE FEATURE ORDER" note at the top of this file.
void sampleFeatures() {
  float rawDist = readDistance();
  distFiltered = ALPHA * rawDist + (1.0f - ALPHA) * distFiltered;

  mpu.getMotion6(&rawAx, &rawAy, &rawAz, &rawGx, &rawGy, &rawGz);
  float accX = (rawAx / 16384.0f) * 9.80665f;
  float accY = (rawAy / 16384.0f) * 9.80665f;
  float accZ = (rawAz / 16384.0f) * 9.80665f;
  float gyroX = rawGx / 131.0f;
  float gyroY = rawGy / 131.0f;
  float gyroZ = rawGz / 131.0f;

  if (eiFeatureIx + 8 <= EI_CLASSIFIER_RAW_SAMPLE_COUNT * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
    eiFeatures[eiFeatureIx++] = distFiltered;
    eiFeatures[eiFeatureIx++] = rawDist;
    eiFeatures[eiFeatureIx++] = accX;
    eiFeatures[eiFeatureIx++] = accY;
    eiFeatures[eiFeatureIx++] = accZ;
    eiFeatures[eiFeatureIx++] = gyroX;
    eiFeatures[eiFeatureIx++] = gyroY;
    eiFeatures[eiFeatureIx++] = gyroZ;
  }
}

// Runs once the 1-second window (400 floats) is full.
void runClassifier() {
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
  signal.get_data = &raw_feature_get_data;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.print("EI classifier error: "); Serial.println((int)err);
    return;
  }

  // Pick the highest-confidence label
  float bestValue = 0.0f;
  const char *bestLabel = "unknown";
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > bestValue) {
      bestValue = result.classification[ix].value;
      bestLabel = result.classification[ix].label;
    }
  }
  eiConfidence = bestValue;

  if (strcmp(bestLabel, "too_close") == 0) eiState = STATE_TOO_CLOSE;
  else if (strcmp(bestLabel, "too_far") == 0) eiState = STATE_TOO_FAR;
  else if (strcmp(bestLabel, "safe_distance") == 0) eiState = STATE_SAFE;
  else eiState = STATE_UNKNOWN;

  Serial.print("EI -> "); Serial.print(bestLabel);
  Serial.print(" ("); Serial.print(bestValue * 100.0f, 1); Serial.println("%)");
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // New core 3.x LEDC API: attach frequency+resolution directly to the pin.
  // Returns the actual frequency achieved (or 0 on failure) — worth checking.
  ledcAttach(ENA, PWM_FREQ, PWM_RES);
  ledcAttach(ENB, PWM_FREQ, PWM_RES);

  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()) Serial.println("MPU6050 FAIL");

  prevTime = millis();
  lastControlMs = millis();
  lastEiSampleMs = millis();
  Serial.println("Follower ready (with Edge Impulse classifier).");
}

// ── Read Distance ───────────────────────────────────
float readDistance() {
  digitalWrite(TRIG, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long dur = pulseIn(ECHO, HIGH, 25000);  // 25ms timeout
  if (dur == 0) return TOO_FAR_DIST;      // no echo = too far
  return (dur * 0.0343f) / 2.0f;
}

// ── Motor Control ───────────────────────────────────
void driveForward(int pwm) {
  pwm = constrain(pwm, 0, 255);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  ledcWrite(ENA, pwm);
  ledcWrite(ENB, pwm);
}

void driveReverse(int pwm) {
  pwm = constrain(pwm, 0, 255);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, pwm);
  ledcWrite(ENB, pwm);
}

void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  ledcWrite(ENA, 0);
  ledcWrite(ENB, 0);
}

// ── Slope feedforward (MPU6050) ──────────────────────
float slopeFeedforward() {
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);
  float az_g = az / 16384.0f;
  // Climbing (az_g>1.05) → boost speed; descending → reduce
  return constrain((az_g - 1.0f) * 40.0f, -50.0f, 50.0f);
}

// ── PID Compute ─────────────────────────────────────
float computePID(float dist) {
  unsigned long now = millis();
  float dt = (now - prevTime) / 1000.0f;
  if (dt < 0.001f) dt = 0.001f;
  prevTime = now;

  float error = dist - TARGET_DIST;  // positive = too far → speed up

  // Anti-windup: clamp integral
  if (abs(error) < 40.0f) {
    integral += error * dt;
    integral = constrain(integral, -80.0f, 80.0f);
  } else {
    integral *= 0.9f;  // decay when far off target
  }

  float derivative = (error - prevError) / dt;
  prevError = error;

  return (Kp * error) + (Ki * integral) + (Kd * derivative);
}

void loop() {
  unsigned long now = millis();

  // ── EI sampling task: 50 Hz, matches the model's training rate ──
  if (now - lastEiSampleMs >= EI_SAMPLE_INTERVAL_MS) {
    lastEiSampleMs = now;
    sampleFeatures();
    if (eiFeatureIx >= EI_CLASSIFIER_RAW_SAMPLE_COUNT * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
      runClassifier();
      eiFeatureIx = 0;  // start filling the next 1-second window
    }
  }

  // ── Control task: ~25 Hz, drives the motors ──
  if (now - lastControlMs < CONTROL_INTERVAL_MS) return;
  lastControlMs = now;

  // distFiltered is kept up to date by sampleFeatures() above (50 Hz),
  // so the control loop just reads the latest value.

  // 1. Hardware emergency stop (always wins, independent of the model)
  if (distFiltered < ESTOP_DIST) {
    stopMotors();
    integral = 0;
    Serial.println("!!! EMERGENCY STOP (HC-SR04) !!!");
    return;
  }

  // 2. AI safety layer: if the classifier is confident the gap is
  //    dangerously close, brake even if the raw threshold hasn't
  //    tripped yet (e.g. it's picking up motion cues the single
  //    ultrasonic reading misses).
  if (eiState == STATE_TOO_CLOSE && eiConfidence > 0.7f) {
    stopMotors();
    integral = 0;
    Serial.println("!!! EI SAFETY STOP (too_close) !!!");
    return;
  }

  // 3. PID (with a small bias from the classifier's "too_far" verdict)
  float pid    = computePID(distFiltered);
  float slope  = slopeFeedforward();
  float eiBias = (eiState == STATE_TOO_FAR && eiConfidence > 0.7f) ? 15.0f : 0.0f;
  float output = pid + slope + eiBias;

  // 4. Drive
  if (output > 0) {
    float pwm = constrain(abs(output), MIN_PWM, MAX_PWM);
    driveForward((int)pwm);
  } else if (output < -20) {
    driveReverse((int)constrain(abs(output), 40, 120));
  } else {
    stopMotors();
  }

  // 5. Serial Plotter (open at 115200)
  Serial.print("Dist:"); Serial.print(distFiltered);
  Serial.print(",Target:"); Serial.print(TARGET_DIST);
  Serial.print(",PID:"); Serial.print(output);
  Serial.print(",EStop:"); Serial.print(ESTOP_DIST);
  Serial.print(",EIState:"); Serial.println((int)eiState);
}
