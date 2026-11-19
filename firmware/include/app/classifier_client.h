#pragma once

#include <Arduino.h>
#include <stddef.h>
#include <stdint.h>

// ESP32 Arduino helper: POST JPEG multipart/form-data to existing FastAPI
// /classify endpoint. Does not perform object detection or parse JSON.
namespace classifier_client {

// Must be called off the LVGL/UI loop (network can block up to timeout).
// Returns true on HTTP 200. Caller owns jpeg_data and must free it later.
// Example URL: http://192.168.1.100:8000/classify (NEVER 127.0.0.1 on ESP32).
bool upload_jpeg(const char* url, const uint8_t* jpeg_data, size_t jpeg_len,
                 String& json_result, int& http_status, uint32_t timeout_ms = 7000);

}  // namespace classifier_client
