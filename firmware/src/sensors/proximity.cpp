#include "sensors/proximity.h"
#include <Arduino.h>

namespace proximity {
namespace {
constexpr uint8_t TRIG_PIN = 9;
constexpr uint8_t ECHO_PIN = 11;
constexpr uint32_t ECHO_TIMEOUT_US = 12000; // enough for >24 inches
constexpr uint32_t MEASUREMENT_INTERVAL_MS = 200;
constexpr uint32_t PRINT_INTERVAL_MS = 1000;
ZoneTracker tracker;
ZoneCallback callback = nullptr;
float last_distance = -1.0f;
uint32_t last_measurement_ms = 0;
uint32_t last_print_ms = 0;

float measure_distance_inches() {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    const unsigned long us = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
    return us == 0 ? -1.0f : static_cast<float>(us) / 148.0f;
}
} // namespace

void begin(ZoneCallback on_zone_changed) {
    callback = on_zone_changed;
    tracker.reset();
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    digitalWrite(TRIG_PIN, LOW);
    last_measurement_ms = millis() - MEASUREMENT_INTERVAL_MS;
    last_print_ms = millis();
    Serial.println("[Proximity] Wake <=24in; camera <=12in; exit camera >=14in; standby >=26in; debounce 350ms");
}

void update() {
    const uint32_t now = millis();
    if (uint32_t(now - last_measurement_ms) < MEASUREMENT_INTERVAL_MS)
        return;
    last_measurement_ms = now;
    last_distance = measure_distance_inches();
    if (tracker.update(last_distance, millis())) {
        Serial.printf("[Proximity] Zone changed to %s (distance=%.1fin)\n",
                      zone_name(tracker.zone()), last_distance);
        if (callback) callback(tracker.zone(), last_distance);
    }
    if (uint32_t(now - last_print_ms) >= PRINT_INTERVAL_MS) {
        last_print_ms = now;
        Serial.printf("[Proximity] distance=%.1fin zone=%s\n", last_distance, zone_name(tracker.zone()));
    }
}

Zone current_zone() { return tracker.zone(); }
float latest_distance_inches() { return last_distance; }

} // namespace proximity
