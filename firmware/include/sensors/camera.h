#pragma once

namespace camera {

    // Initialize the OV2640 camera.
    // Returns true if initialization succeeds.
    bool begin();

    // Returns whether the camera has initialized successfully.
    bool is_initialized();

    // Capture and verify one JPEG frame.
    bool capture_test();

}