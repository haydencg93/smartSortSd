#pragma once

#include <lvgl.h>
#include <stdint.h>
#include <stddef.h>

namespace smartsort { struct Detection; }

namespace ui {

constexpr int32_t W = 480;
constexpr int32_t H = 320;

enum class Bin : uint8_t { Recycle, Trash };

enum class Reason : uint8_t {
    None,
    WrongMaterial,
    GreasyOrDirty,
    ItemNotShown,
    SomethingElse,
};

// Sent when the user confirms a result or submits a correction.
// The item pointer is valid only during the callback; copy it if retained.
struct Feedback {
    Bin shown;
    Bin correct;
    Reason reason;
    const char* item;
};

struct Callbacks {
    void (*on_idle_tapped)() = nullptr;
    void (*on_feedback)(const Feedback&) = nullptr;
};

// Call once, after LVGL and the display have been initialized.
void init(const Callbacks& cb);

void show_idle();
void show_scan_prompt();
void show_detection_boxes(const smartsort::Detection* detections, size_t count);
void clear_detection_boxes();
bool is_idle_visible();
bool capture_scan_image();
void show_analyzing();
void show_result(Bin bin, const char* item_name, float confidence, float latency_ms);
void show_special_disposal(float confidence = 0.0f, float latency_ms = 0.0f);
void show_manual_review(float confidence = 0.0f, float latency_ms = 0.0f);
void show_wifi_error();

}  // namespace ui
