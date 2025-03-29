/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "detection_responder.h"

#include "Arduino.h"

void RespondToDetection(tflite::ErrorReporter* error_reporter,
  int8_t mug_score, int8_t no_mug_score) {
    static bool is_initialized = false;
  if (!is_initialized) {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  is_initialized = true;
  }

// Turn off LEDs
digitalWrite(LEDG, HIGH);
digitalWrite(LEDR, HIGH);
digitalWrite(LEDB, HIGH);

// Turn on LED if a mug is detected
if (mug_score > no_mug_score) {
Serial.println("Mug Detected!");
digitalWrite(LEDG, LOW);
} else {
Serial.println("No Mug Detected.");
digitalWrite(LEDR, LOW);
}

TF_LITE_REPORT_ERROR(error_reporter, "Mug Score: %d No Mug Score: %d",
   mug_score, no_mug_score);
}


#endif  // ARDUINO_EXCLUDE_CODE
