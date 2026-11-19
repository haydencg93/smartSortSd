#include "app/auto_scan.h"

#include <cmath>

namespace smartsort {

AutoScan::AutoScan(ScanSettings settings) : settings_(settings) {}

bool AutoScan::eligible(const Detection& d) const {
    if (d.class_id <= 0 || d.confidence < settings_.min_confidence) {
        // COCO category 0 is person. Hand is NOT a COCO category.
        return false;
    }
    if (!(d.x1 >= 0 && d.y1 >= 0 && d.x2 <= 1 && d.y2 <= 1 &&
          d.x2 > d.x1 && d.y2 > d.y1)) {
        return false;
    }
    const float cx = (d.x1 + d.x2) * 0.5f;
    const float cy = (d.y1 + d.y2) * 0.5f;
    return (d.x2 - d.x1) * (d.y2 - d.y1) >= settings_.min_box_area &&
           cx >= settings_.zone_left && cx <= settings_.zone_right &&
           cy >= settings_.zone_top && cy <= settings_.zone_bottom;
}

bool AutoScan::same_item(const Detection& a, const Detection& b) const {
    if (a.class_id != b.class_id) return false;
    const float dx = (a.x1 + a.x2 - b.x1 - b.x2) * 0.5f;
    const float dy = (a.y1 + a.y2 - b.y1 - b.y2) * 0.5f;
    return dx * dx + dy * dy <=
           settings_.max_center_motion * settings_.max_center_motion;
}

const Detection* AutoScan::best_eligible(const std::vector<Detection>& candidates) const {
    const Detection* best = nullptr;
    for (const auto& d : candidates) {
        if (eligible(d) && (!best || d.confidence > best->confidence)) best = &d;
    }
    return best;
}

const Detection* AutoScan::best_match(const std::vector<Detection>& candidates) const {
    const Detection* best = nullptr;
    for (const auto& d : candidates) {
        if (eligible(d) && same_item(target_, d) &&
            (!best || d.confidence > best->confidence)) best = &d;
    }
    return best;
}

bool AutoScan::update(const std::vector<Detection>& detections, uint32_t now_ms) {
    if (state_ == ScanState::CapturePending) return false;

    if (state_ == ScanState::WaitForRemoval) {
        if (best_eligible(detections)) {
            absence_started_ = false;
        } else if (!absence_started_) {
            absence_started_ms_ = now_ms;
            absence_started_ = true;
        } else if (static_cast<uint32_t>(now_ms - absence_started_ms_) >=
                   settings_.absence_to_rearm_ms) {
            reset();
        }
        return false;
    }

    if (state_ == ScanState::Searching) {
        if (const Detection* candidate = best_eligible(detections)) {
            target_ = *candidate;
            first_seen_ms_ = now_ms;
            state_ = ScanState::Confirming;
        }
        return false;
    }

    if (const Detection* same = best_match(detections)) {
        target_ = *same;
        if (static_cast<uint32_t>(now_ms - first_seen_ms_) >= settings_.confirm_ms) {
            state_ = ScanState::CapturePending;
            return true;
        }
    } else {
        // A different eligible item can begin a new confirmation window immediately.
        state_ = ScanState::Searching;
        if (const Detection* candidate = best_eligible(detections)) {
            target_ = *candidate;
            first_seen_ms_ = now_ms;
            state_ = ScanState::Confirming;
        }
    }
    return false;
}

void AutoScan::capture_finished() {
    if (state_ == ScanState::CapturePending) {
        state_ = ScanState::WaitForRemoval;
        absence_started_ = false;
    }
}

void AutoScan::reset() {
    state_ = ScanState::Searching;
    target_ = {};
    first_seen_ms_ = 0;
    absence_started_ms_ = 0;
    absence_started_ = false;
}

} // namespace smartsort
