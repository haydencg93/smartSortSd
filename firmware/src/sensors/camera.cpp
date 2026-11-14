#include "sensors/camera.h"

#include <Arduino.h>
#include <esp_camera.h>
#include <string.h>
#include <img_converters.h>
#include <stdlib.h>

namespace camera {

namespace {

bool initialized = false;

// Waveshare ESP32-S3-Touch-LCD-3.5 Rev 2.0
constexpr int CAMERA_PWDN = -1;
constexpr int CAMERA_RESET = -1;

constexpr int CAMERA_XCLK = 38;

// Reuse the I2C bus already initialized by the board driver.
constexpr int CAMERA_SIOD = -1;
constexpr int CAMERA_SIOC = -1;
constexpr int CAMERA_I2C_PORT = 0;

// Camera data pins.
constexpr int CAMERA_Y9 = 21;
constexpr int CAMERA_Y8 = 39;
constexpr int CAMERA_Y7 = 40;
constexpr int CAMERA_Y6 = 42;
constexpr int CAMERA_Y5 = 46;
constexpr int CAMERA_Y4 = 48;
constexpr int CAMERA_Y3 = 47;
constexpr int CAMERA_Y2 = 45;

constexpr int CAMERA_VSYNC = 17;
constexpr int CAMERA_HREF = 18;
constexpr int CAMERA_PCLK = 41;

} // anonymous namespace


bool start() {

    if (initialized) {
        return true;
    }

    Serial.println("[Camera] Starting OV2640...");

    camera_config_t config = {};

    config.pin_pwdn = CAMERA_PWDN;
    config.pin_reset = CAMERA_RESET;
    config.pin_xclk = CAMERA_XCLK;

    config.pin_sccb_sda = CAMERA_SIOD;
    config.pin_sccb_scl = CAMERA_SIOC;
    config.sccb_i2c_port = CAMERA_I2C_PORT;

    config.pin_d7 = CAMERA_Y9;
    config.pin_d6 = CAMERA_Y8;
    config.pin_d5 = CAMERA_Y7;
    config.pin_d4 = CAMERA_Y6;
    config.pin_d3 = CAMERA_Y5;
    config.pin_d2 = CAMERA_Y4;
    config.pin_d1 = CAMERA_Y3;
    config.pin_d0 = CAMERA_Y2;

    config.pin_vsync = CAMERA_VSYNC;
    config.pin_href = CAMERA_HREF;
    config.pin_pclk = CAMERA_PCLK;

    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    // RGB565 allows LVGL to display the camera frame directly.
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;

    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    const esp_err_t result = esp_camera_init(&config);

    if (result != ESP_OK) {
        Serial.printf(
            "[Camera] Initialization failed: 0x%x\n",
            static_cast<unsigned int>(result)
        );

        initialized = false;
        return false;
    }

    initialized = true;

    Serial.println("[Camera] OV2640 started.");
    Serial.println("[Camera] Resolution: 320 x 240");
    Serial.println("[Camera] Format: RGB565");

    return true;
}


void stop() {

    if (!initialized) {
        return;
    }

    Serial.println("[Camera] Stopping OV2640...");

    const esp_err_t result = esp_camera_deinit();

    if (result != ESP_OK) {
        Serial.printf(
            "[Camera] Shutdown returned error: 0x%x\n",
            static_cast<unsigned int>(result)
        );
    }

    initialized = false;

    Serial.println("[Camera] Stopped.");
}


bool is_running() {
    return initialized;
}


bool capture_rgb565(
    uint8_t* destination,
    size_t capacity,
    uint16_t& width,
    uint16_t& height
) {

    if (!initialized || destination == nullptr) {
        return false;
    }

    camera_fb_t* frame = esp_camera_fb_get();

    if (frame == nullptr) {
        Serial.println("[Camera] Frame capture failed.");
        return false;
    }

    const size_t required_size =
        static_cast<size_t>(frame->width) *
        static_cast<size_t>(frame->height) * 2;

    bool valid =
        frame->format == PIXFORMAT_RGB565 &&
        frame->len >= required_size &&
        capacity >= required_size;

    if (valid) {
        // Swap the two bytes of each RGB565 pixel to match
        // the byte order expected by the LVGL image renderer.
        for (size_t i = 0; i < required_size; i += 2) {
            destination[i]     = frame->buf[i + 1];
            destination[i + 1] = frame->buf[i];
        }

        width = frame->width;
        height = frame->height;
    }

    esp_camera_fb_return(frame);

    return valid;
}

bool capture_jpeg(
    uint8_t** jpeg_data,
    size_t& jpeg_size,
    uint8_t quality
) {

    // Initialize outputs so failure never leaves stale pointers or sizes.
    if (jpeg_data == nullptr) {
        return false;
    }

    *jpeg_data = nullptr;
    jpeg_size = 0;

    if (!initialized) {
        Serial.println("[Camera] ERROR: Cannot capture JPEG; camera is stopped.");
        return false;
    }

    camera_fb_t* frame = esp_camera_fb_get();

    if (frame == nullptr) {
        Serial.println("[Camera] ERROR: JPEG frame capture failed.");
        return false;
    }

    bool success = false;

    if (frame->format == PIXFORMAT_RGB565) {

        // The camera's original frame is passed directly to the encoder.
        // It has not been byte-swapped for LVGL.
        success = fmt2jpg(
            frame->buf,
            frame->len,
            frame->width,
            frame->height,
            PIXFORMAT_RGB565,
            quality,
            jpeg_data,
            &jpeg_size
        );
    }

    esp_camera_fb_return(frame);

    if (!success) {
        Serial.println("[Camera] ERROR: JPEG conversion failed.");

        if (*jpeg_data != nullptr) {
            free(*jpeg_data);
            *jpeg_data = nullptr;
        }

        jpeg_size = 0;
        return false;
    }

    Serial.printf(
        "[Camera] JPEG captured: %u bytes\n",
        static_cast<unsigned int>(jpeg_size)
    );

    return true;
}

} // namespace camera