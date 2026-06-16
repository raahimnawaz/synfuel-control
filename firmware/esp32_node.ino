// ESP32 sensor node for the synfuel reactor control loop.
//
// Reads the conditioned temperature and pressure signals on two ADC pins, converts the
// raw 12-bit codes back to physical units using the SAME analog front-end calibration as
// circuits/frontend.py, streams them as JSON telemetry over serial, and applies incoming
// coolant-setpoint commands to a PWM output that stands in for the coolant valve.
//
// Target: ESP32 dev board. Runs as-is in Wokwi (wokwi.toml + diagram.json) with
// potentiometers standing in for the analog front-end outputs. The conversion logic here
// is mirrored by bridge/virtual_esp32.py so the SIL and hardware paths match.

// --- Analog front-end calibration (mirror of circuits/frontend.py) ------------------
static const float VREF       = 3.3f;
static const int   ADC_MAX    = 4095;
static const float RTD_R0     = 100.0f;
static const float RTD_ALPHA  = 0.00385f;
static const float R_DIVIDER  = 1000.0f;
static const float GAIN_T     = 5.0f;
static const float BRIDGE_SENS = 4.0e-4f;
static const float GAIN_P     = 50.0f;

// --- Pins -----------------------------------------------------------------------------
static const int PIN_TEMP   = 34;   // ADC1_CH6 : conditioned temperature
static const int PIN_PRESS  = 35;   // ADC1_CH7 : conditioned pressure
static const int PIN_VALVE  = 25;   // PWM out  : coolant valve actuator (stand-in)

static const int    SAMPLE_HZ = 100;
static const float  COOLANT_MIN = 440.0f, COOLANT_MAX = 540.0f;

float adc_to_temp(int code) {
  float v_out = (float)code / ADC_MAX * VREF;
  float v_div = v_out / GAIN_T;
  float r = R_DIVIDER * v_div / (VREF - v_div);
  return (r / RTD_R0 - 1.0f) / RTD_ALPHA;
}

float adc_to_pressure(int code) {
  float v = (float)code / ADC_MAX * VREF;
  return v / (GAIN_P * VREF * BRIDGE_SENS);
}

// Apply a coolant-temperature setpoint as a PWM duty (lower coolant => more cooling).
void apply_coolant(float coolant_K) {
  float frac = (COOLANT_MAX - coolant_K) / (COOLANT_MAX - COOLANT_MIN);
  frac = frac < 0 ? 0 : (frac > 1 ? 1 : frac);
  ledcWrite(0, (int)(frac * 255));
}

// Minimal JSON command parser: pull the number after "coolant_K".
bool parse_coolant(const String& line, float& out) {
  int k = line.indexOf("coolant_K");
  if (k < 0) return false;
  int colon = line.indexOf(':', k);
  if (colon < 0) return false;
  out = line.substring(colon + 1).toFloat();
  return true;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  ledcSetup(0, 5000, 8);        // channel 0, 5 kHz, 8-bit
  ledcAttachPin(PIN_VALVE, 0);
}

void loop() {
  int temp_code  = analogRead(PIN_TEMP);
  int press_code = analogRead(PIN_PRESS);
  float temp_C   = adc_to_temp(temp_code);
  float press_bar = adc_to_pressure(press_code);

  // Telemetry as JSON.
  Serial.print("{\"t_ms\":");   Serial.print(millis());
  Serial.print(",\"temp_C\":"); Serial.print(temp_C, 2);
  Serial.print(",\"press_bar\":"); Serial.print(press_bar, 2);
  Serial.println("}");

  // Apply any pending coolant command.
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    float coolant_K;
    if (parse_coolant(line, coolant_K)) apply_coolant(coolant_K);
  }

  delay(1000 / SAMPLE_HZ);
}
