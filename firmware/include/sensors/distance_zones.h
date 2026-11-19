#pragma once

#include <stdint.h>

// Hardware-independent, deterministic policy for HC-SR04 distance samples.
// A missing echo does NOT immediately force the UI back to standby.
namespace proximity {

enum class Zone : uint8_t { Standby, Welcome, Camera };

inline const char* zone_name(Zone z) {
    switch (z) {
        case Zone::Standby: return "STANDBY";
        case Zone::Welcome: return "WELCOME";
        case Zone::Camera: return "CAMERA";
    }
    return "UNKNOWN";
}

class ZoneTracker {
public:
    Zone zone() const { return current_; }
    // Returns true only after the new zone is stable long enough to be adopted.
    bool update(float distance_in, uint32_t now_ms) {
        const bool valid = distance_in > 0.0f && distance_in <= 160.0f;
        Zone requested;
        if (valid) {
            have_valid_ = true;
            last_valid_ms_ = now_ms;
            requested = next_zone(distance_in);
        } else {
            // One or two lost echoes are common with angled objects.
            if (!have_valid_ || uint32_t(now_ms - last_valid_ms_) < kMissingEchoGraceMs)
                return false;
            requested = Zone::Standby;
        }
        if (requested == current_) {
            candidate_ = current_;
            return false;
        }
        if (requested != candidate_) {
            candidate_ = requested;
            candidate_started_ms_ = now_ms;
            return false;
        }
        if (uint32_t(now_ms - candidate_started_ms_) < kZoneDebounceMs)
            return false;
        current_ = requested;
        candidate_ = current_;
        return true;
    }

    void reset() {
        current_ = Zone::Standby;
        candidate_ = Zone::Standby;
        candidate_started_ms_ = 0;
        last_valid_ms_ = 0;
        have_valid_ = false;
    }

private:
    Zone next_zone(float d) const {
        switch (current_) {
            case Zone::Standby:
                if (d <= 12.0f) return Zone::Camera;
                if (d <= 24.0f) return Zone::Welcome;
                return Zone::Standby;
            case Zone::Welcome:
                if (d <= 12.0f) return Zone::Camera;
                if (d >= 26.0f) return Zone::Standby;
                return Zone::Welcome;
            case Zone::Camera:
                if (d >= 14.0f) return d >= 26.0f ? Zone::Standby : Zone::Welcome;
                return Zone::Camera;
        }
        return Zone::Standby;
    }

    static constexpr uint32_t kZoneDebounceMs = 350;
    static constexpr uint32_t kMissingEchoGraceMs = 1500;
    Zone current_ = Zone::Standby;
    Zone candidate_ = Zone::Standby;
    uint32_t candidate_started_ms_ = 0;
    uint32_t last_valid_ms_ = 0;
    bool have_valid_ = false;
};

} // namespace proximity
