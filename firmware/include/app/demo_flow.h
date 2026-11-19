#pragma once
#include <stddef.h>
#include <stdint.h>
namespace smartsort { struct Detection; }

void demo_flow_init();
void demo_flow_start_scan();
void demo_flow_cancel();
void demo_flow_update(bool camera_zone_active);
// Feed ONLY fresh results of completed YOLO inferences, on the LVGL/UI thread.
// Does not run YOLO; the model adapter still needs to be integrated.
void demo_flow_on_detections(const smartsort::Detection* detections,
                             size_t count,
                             uint32_t inference_completed_ms);
