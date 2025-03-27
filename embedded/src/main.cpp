#include <TensorFlowLite.h>
#include <Arduino.h>
#include <Arduino_OV767X_TinyMLx.h>
#include "mug_model.h"
#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
bool is_initialized;

constexpr int kTensorArenaSize = 100 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}

//GetImage function
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
  int image_height, int channels, int8_t* image_data) {

byte data[176 * 144]; // Receiving QCIF grayscale from camera = 176 * 144 * 1

static bool g_is_camera_initialized = false;
static bool serial_is_initialized = false;

// Initialize camera if necessary
if (!g_is_camera_initialized) {
if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
return kTfLiteError;
}
g_is_camera_initialized = true;
}

// Read camera data
Camera.readFrame(data);

int min_x = (176 - 96) / 2;
int min_y = (144 - 96) / 2;
int index = 0;

// Crop 96x96 image. This lowers FOV, ideally we should downsample
for (int y = min_y; y < min_y + 96; y++) {
for (int x = min_x; x < min_x + 96; x++) {
image_data[index++] = static_cast<int8_t>(data[(y * 176) + x] - 128); // convert TF input image to signed 8-bit
}
}
return kTfLiteOk;
}

//RespondToDetection function
void RespondToDetection(tflite::ErrorReporter* error_reporter,
  int8_t person_score, int8_t no_person_score) {
if (!is_initialized) {
// Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
pinMode(LEDR, OUTPUT);
pinMode(LEDG, OUTPUT);
pinMode(LEDB, OUTPUT);
is_initialized = true;
}

// Note: The RGB LEDs on the Arduino Nano 33 BLE
// Sense are on when the pin is LOW, off when HIGH.

// Switch the person/not person LEDs off
digitalWrite(LEDG, HIGH);
digitalWrite(LEDR, HIGH);
digitalWrite(LEDB, HIGH);

// Switch on the green LED when a person is detected,
// the red when no person is detected
if (person_score > no_person_score) {
digitalWrite(LEDG, LOW);
digitalWrite(LEDR, HIGH);
} else {
digitalWrite(LEDG, HIGH);
digitalWrite(LEDR, LOW);
}

TF_LITE_REPORT_ERROR(error_reporter, "Person score: %d No person score: %d",
 person_score, no_person_score);
}


void setup() {
  Serial.begin(115200);
  delay(5000);
  Serial.println("Starting");

  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  model = tflite::GetModel(mug_model_data);
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
}

void loop() {
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }

  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(error_reporter, person_score, no_person_score);
}
