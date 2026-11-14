
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace camera {

    // Start the OV2640 camera.
    bool start();

    // Stop the camera and release its driver resources.
    void stop();

    // Returns whether the camera is currently running.
    bool is_running();

    // Capture one RGB565 frame and copy it into the supplied buffer.
    // The caller owns the destination buffer.
    bool capture_rgb565(
        uint8_t* destination,
        size_t capacity,
        uint16_t& width,
        uint16_t& height
    );

    // Capture one camera frame and encode it as JPEG.
    // On success, jpeg_data is allocated by the camera library.
    // The caller MUST release it using free().
    bool capture_jpeg(
        uint8_t** jpeg_data,
        size_t& jpeg_size,
        uint8_t quality = 12
    );

}
