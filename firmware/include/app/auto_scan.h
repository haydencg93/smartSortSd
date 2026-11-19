#pragma once

#include <cstdint>
#include <vector>

// Portable detection/trigger policy: compiles with ordinary C++17, Arduino, or ESP-IDF.
// Does not run YOLO, touch the camera, or perform HTTP requests.
namespace smartsort {

struct Detection {
    int class_id = -1;       // Standard COCO IDs (0 = person)
    float confidence = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // Normalized [0,1]
};

enum class ScanState {
    Searching,
    Confirming,
    CapturePending,
    WaitForRemoval
};

struct ScanSettings {
    float min_confidence = 0.55f;
    float min_box_area = 0.02f; // 2% of image area; not a validated threshold
    float zone_left = 0.20f;
    float zone_top = 0.15f;
    float zone_right = 0.80f;
    float zone_bottom = 0.85f;
    float max_center_motion = 0.20f; // max normalized displacement between detections
    uint32_t confirm_ms = 500;
    uint32_t absence_to_rearm_ms = 1000;
};

class AutoScan {
public:
    explicit AutoScan(ScanSettings settings = {});

    // Called once per completed object-detection inference, with monotonic milliseconds.
    // Returns true exactly once when a stable candidate should be photographed.
    bool update(const std::vector<Detection>& detections, uint32_t now_ms);

    // Call when capture/upload has completed OR failed. The same item cannot retrigger
    // until the detector observes sustained absence in the presentation zone.
    void capture_finished();
    void reset();
    ScanState state() const { return state_; }
    const Detection& target() const { return target_; }

private:
    bool eligible(const Detection& d) const;
    bool same_item(const Detection& a, const Detection& b) const;
    const Detection* best_eligible(const std::vector<Detection>& candidates) const;
    const Detection* best_match(const std::vector<Detection>& candidates) const;

    ScanSettings settings_;
    ScanState state_ = ScanState::Searching;
    Detection target_{};
    uint32_t first_seen_ms_ = 0;
    uint32_t absence_started_ms_ = 0;
    bool absence_started_ = false;
};

} // namespace smartsort
