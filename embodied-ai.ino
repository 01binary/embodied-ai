#define USE_USBCON

#include <Arduino.h>
#include <Servo.h>

#include <ros.h>
#include <embodied_ai/ActuatorCommand.h>
#include <embodied_ai/ActuatorState.h>

#define PIN_LED 19
#define PIN_SOL 22
#define PIN_SERVO_SIG 20
#define PIN_SERVO_POS A5

static uint16_t encoderMin = 0;
static uint16_t encoderMax = 1023;

// If your servo endpoints are different, adjust these.
static const int SERVO_MIN_DEG = 0;
static const int SERVO_MAX_DEG = 180;

static bool led_state = false;
static bool sol_state = false;

static bool have_servo_target = false;
static float servo_target_norm = 0.0f; // 0..1

Servo servo;

ros::NodeHandle_<ArduinoHardware, 1, 1, 128, 128> nh;

embodied_ai::ActuatorState state_msg;
ros::Publisher pub_state("actuator/state", &state_msg);

void command(const embodied_ai::ActuatorCommand& msg)
{
  led_state = msg.led;
  sol_state = msg.solenoid;

  if (msg.servo_enable) {
    have_servo_target = true;
    servo_target_norm = msg.servo_cmd;
    if (servo_target_norm < 0.0f) servo_target_norm = 0.0f;
    if (servo_target_norm > 1.0f) servo_target_norm = 1.0f;
  }
}

void feedback()
{
  uint16_t adc_raw = 0;
  float pos_norm = readServoNorm(adc_raw);

  state_msg.led = led_state;
  state_msg.solenoid = sol_state;
  state_msg.servo_pos = pos_norm;
  state_msg.adc_raw = adc_raw;
  state_msg.encoder_min = encoderMin;
  state_msg.encoder_max = encoderMax;

  pub_state.publish(&state_msg);
}

ros::Subscriber<embodied_ai::ActuatorCommand> sub_cmd("actuator/cmd", &command);

// Publish at ~10 Hz
static unsigned long last_pub_ms = 0;
static const unsigned long PUB_PERIOD_MS = 100;

static void delayWithSpin(unsigned long ms)
{
  unsigned long start = millis();
  while (millis() - start < ms) {
    nh.spinOnce();
    delay(5);
  }
}

static float readServoNorm(uint16_t& adc_raw_out)
{
  uint16_t raw = (uint16_t)analogRead(PIN_SERVO_POS);
  adc_raw_out = raw;

  int denom = (int)encoderMax - (int)encoderMin;
  if (denom <= 0) return 0.0f;

  float norm = (float)((int)raw - (int)encoderMin) / (float)denom;
  if (norm < 0.0f) norm = 0.0f;
  if (norm > 1.0f) norm = 1.0f;
  return norm;
}

static void setServoFromNorm(float norm)
{
  if (norm < 0.0f) norm = 0.0f;
  if (norm > 1.0f) norm = 1.0f;

  int deg = (int)(SERVO_MIN_DEG + norm * (SERVO_MAX_DEG - SERVO_MIN_DEG) + 0.5f);
  if (deg < SERVO_MIN_DEG) deg = SERVO_MIN_DEG;
  if (deg > SERVO_MAX_DEG) deg = SERVO_MAX_DEG;

  servo.write(deg);
}

static void initServo()
{
  servo.attach(PIN_SERVO_SIG);

  // Move to minimum, sample
  setServoFromNorm(0.0f);
  delayWithSpin(1250);
  encoderMin = (uint16_t)analogRead(PIN_SERVO_POS);

  // Move to maximum, sample
  setServoFromNorm(1.0f);
  delayWithSpin(1250);
  encoderMax = (uint16_t)analogRead(PIN_SERVO_POS);

  // Optional: swap if reversed
  if (encoderMax < encoderMin) {
    uint16_t tmp = encoderMin;
    encoderMin = encoderMax;
    encoderMax = tmp;
  }

  // Return to center
  setServoFromNorm(0.5f);
}

void setup()
{
  pinMode(PIN_SOL, OUTPUT);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_SERVO_POS, INPUT);

  // ROS serial init
  nh.getHardware()->setBaud(115200);
  nh.initNode();
  nh.advertise(pub_state);
  nh.subscribe(sub_cmd);
  delayWithSpin(500);

  // Default safe outputs
  led_state = false;
  sol_state = false;
  digitalWrite(PIN_LED, led_state ? LOW : HIGH);
  digitalWrite(PIN_SOL, sol_state ? HIGH : LOW);

  initServo();
}

void loop()
{
  // Apply outputs
  digitalWrite(PIN_LED, led_state ? LOW : HIGH);
  digitalWrite(PIN_SOL, sol_state ? HIGH : LOW);

  if (have_servo_target) {
    setServoFromNorm(servo_target_norm);
  }

  // Publish state
  unsigned long now = millis();

  if (now - last_pub_ms >= PUB_PERIOD_MS) {
    last_pub_ms = now;
    feedback();
  }

  nh.spinOnce();
  delay(5);
}
