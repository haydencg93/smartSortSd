
#include "sensors/camera.h"

#include <Arduino.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <string.h>
#include <stdlib.h>

namespace camera {

namespace {

bool initialized = false;

// Waveshare ESP32-S3-Touch-LCD-3.5 Rev 2.0
constexpr int CAMERA_PWDN = -1;
constexpr int CAMERA_RESET = -1;
constexpr int CAMERA_XCLK = 38;

// Reuse the board's initialized I2C bus.
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

enum class CameraMode {
    Preview,
    Capture
};

bool initialize(CameraMode mode, uint8_t quality = 10) {

    if (initialized) {
        return true;
    }

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

    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    if (mode == CameraMode::Preview) {
        config.pixel_format = PIXFORMAT_RGB565;
        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 12;
    } else {
        config.pixel_format = PIXFORMAT_JPEG;
        config.frame_size = FRAMESIZE_SVGA;
        config.jpeg_quality = quality;
    }

    Serial.printf(
        "[Camera] Initializing %s mode...\n",
        mode == CameraMode::Preview ? "preview" : "capture"
    );

    if (!psramFound()) {
        Serial.println("[Camera] ERROR: PSRAM not detected.");
        return false;
    }

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

    if (mode == CameraMode::Preview) {
        Serial.println("[Camera] Preview ready: 320x240 RGB565");
    } else {
        Serial.println("[Camera] Capture ready: 800x600 JPEG");
    }

    return true;
}

} // anonymous namespace


bool start() {
    return initialize(CameraMode::Preview);
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
        for (size_t i = 0; i < required_size; i += 2) {
            destination[i] = frame->buf[i + 1];
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

    if (jpeg_data == nullptr) {
        return false;
    }

    *jpeg_data = nullptr;
    jpeg_size = 0;

    // The preview timer must be stopped before this function is called.
    // Switch modes by fully releasing the preview camera first.
    if (initialized) {
        stop();
    }

    if (!initialize(CameraMode::Capture, quality)) {
        Serial.println("[Camera] ERROR: Could not initialize capture mode.");
        return false;
    }

    Serial.println("[Camera] Capturing 800x600 JPEG...");

    // Discard an initial frame so the sensor can settle after initialization.
    camera_fb_t* warmup = esp_camera_fb_get();

    if (warmup != nullptr) {
        esp_camera_fb_return(warmup);
    }

    camera_fb_t* frame = esp_camera_fb_get();

    if (frame == nullptr) {
        Serial.println("[Camera] ERROR: JPEG frame capture failed.");
        stop();
        return false;
    }

    bool valid =
        frame->format == PIXFORMAT_JPEG &&
        frame->width == 800 &&
        frame->height == 600 &&
        frame->len > 0;

    if (!valid) {
        Serial.printf(
            "[Camera] ERROR: Unexpected frame: %ux%u, format=%d, bytes=%u\n",
            static_cast<unsigned int>(frame->width),
            static_cast<unsigned int>(frame->height),
            static_cast<int>(frame->format),
            static_cast<unsigned int>(frame->len)
        );

        esp_camera_fb_return(frame);
        stop();
        return false;
    }

    // Allocate independent memory so the JPEG survives camera shutdown.
    uint8_t* copy = static_cast<uint8_t*>(
        heap_caps_malloc(
            frame->len,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        )
    );

    if (copy == nullptr) {
        Serial.println("[Camera] ERROR: JPEG memory allocation failed.");

        esp_camera_fb_return(frame);
        stop();
        return false;
    }

    memcpy(copy, frame->buf, frame->len);

    jpeg_size = frame->len;
    *jpeg_data = copy;

    esp_camera_fb_return(frame);

    Serial.printf(
        "[Camera] Native JPEG captured: %ux%u, %u bytes\n",
        800,
        600,
        static_cast<unsigned int>(jpeg_size)
    );

    return true;
}

} // namespace camera
