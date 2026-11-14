#include "sensors/camera.h"

#include <Arduino.h>
#include <esp_camera.h>

namespace camera {

namespace {

bool initialized = false;

// Waveshare ESP32-S3-Touch-LCD-3.5 Rev 2.0
constexpr int CAMERA_PWDN = -1;
constexpr int CAMERA_RESET = -1;

constexpr int CAMERA_XCLK = 38;

// The camera shares the existing touchscreen I2C bus.
// Wire uses I2C controller 0, initialized by the board driver.
constexpr int CAMERA_SIOD = -1;
constexpr int CAMERA_SIOC = -1;
constexpr int CAMERA_I2C_PORT = 0;

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


bool begin() {

    if (initialized) {
        Serial.println("[Camera] Already initialized.");
        return true;
    }

    Serial.println("[Camera] Initializing OV2640...");

    camera_config_t config = {};

    config.pin_pwdn = CAMERA_PWDN;
    config.pin_reset = CAMERA_RESET;
    config.pin_xclk = CAMERA_XCLK;

    // Reuse the already initialized Arduino Wire I2C controller.
    config.pin_sccb_sda = CAMERA_SIOD;
    config.pin_sccb_scl = CAMERA_SIOC;
    config.sccb_i2c_port = CAMERA_I2C_PORT;

    // Camera data pins.
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

    // Camera clock configuration.
    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    // Initial test settings.
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;

    // Use one frame buffer in PSRAM.
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t result = esp_camera_init(&config);

    if (result != ESP_OK) {
        Serial.printf(
            "[Camera] Initialization FAILED. Error: 0x%x\n",
            static_cast<unsigned int>(result)
        );

        initialized = false;
        return false;
    }

    initialized = true;

    Serial.println("[Camera] OV2640 initialized successfully!");
    Serial.println("[Camera] Resolution: 320 x 240");
    Serial.println("[Camera] Format: JPEG");

    return true;
}


bool is_initialized() {
    return initialized;
}

bool capture_test() {

    if (!initialized) {
        Serial.println("[Camera] Cannot capture: camera is not initialized.");
        return false;
    }

    Serial.println("[Camera] Capturing test image...");

    // Request one frame from the camera.
    camera_fb_t* frame = esp_camera_fb_get();

    if (frame == nullptr) {
        Serial.println("[Camera] ERROR: Image capture failed.");
        return false;
    }

    Serial.println("[Camera] Image captured successfully!");

    Serial.printf("[Camera] Width: %u pixels\n", frame->width);
    Serial.printf("[Camera] Height: %u pixels\n", frame->height);
    Serial.printf("[Camera] Image size: %u bytes\n",
                  static_cast<unsigned int>(frame->len));

    bool valid = true;

    // Verify that the frame contains image data.
    if (frame->len == 0) {
        Serial.println("[Camera] ERROR: Image is empty.");
        valid = false;
    }

    // Verify that the camera returned JPEG data.
    if (frame->format != PIXFORMAT_JPEG) {
        Serial.println("[Camera] ERROR: Image is not JPEG format.");
        valid = false;
    }

    if (valid) {
        Serial.println("[Camera] Image verification PASSED!");
    } else {
        Serial.println("[Camera] Image verification FAILED!");
    }

    // Always return the frame buffer when finished.
    esp_camera_fb_return(frame);

    return valid;
}

} // namespace camera