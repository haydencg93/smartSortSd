#include "sensors/proximity.h"

#include <Arduino.h>

namespace proximity {
namespace {

constexpr uint8_t TRIG_PIN = 9;
constexpr uint8_t ECHO_PIN = 11;

constexpr float DETECTION_THRESHOLD_IN = 24.0f;
constexpr float THRESHOLD_TOLERANCE_IN = 2.0f;
constexpr float MAX_DETECTION_DISTANCE_IN =
    DETECTION_THRESHOLD_IN + THRESHOLD_TOLERANCE_IN;

constexpr uint32_t INACTIVITY_TIME_MS = 3000;
constexpr uint32_t ECHO_TIMEOUT_US = 30000;
constexpr uint32_t MEASUREMENT_INTERVAL_MS = 60;

PresenceCallback presence_callback = nullptr;

bool active = false;
bool timer_running = false;
bool timer_expired = false;

uint32_t inactivity_timer_start = 0;
uint32_t last_measurement_ms = 0;
uint32_t last_print_ms = 0;

float latest_distance_in = -1.0f;
bool user_detected = false;

float measure_distance_inches() {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);

    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    const unsigned long duration =
        pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);

    if (duration == 0) {
        return -1.0f;
    }

    return static_cast<float>(duration) / 148.0f;
}

void print_status(uint32_t now) {
    if (now - last_print_ms < 500) {
        return;
    }
    last_print_ms = now;

    Serial.print("Distance: ");
    if (latest_distance_in < 0.0f) {
        Serial.print("No valid echo");
    } else {
        Serial.print(latest_distance_in, 2);
        Serial.print(" in");
    }

    Serial.print(" | Active: ");
    Serial.print(active ? "ON" : "OFF");

    Serial.print(" | State: ");
    if (user_detected && active) {
        Serial.print("ACTIVE STATE");
    } else if (active && timer_running) {
        Serial.print("TIMER COUNTDOWN STATE");
    } else if (timer_expired) {
        Serial.print("TIMER EXPIRED STATE");
    } else {
        Serial.print("INACTIVE STATE");
    }

    if (timer_running) {
        Serial.print(" | Elapsed: ");
        Serial.print((now - inactivity_timer_start) / 1000.0f, 1);
        Serial.print(" s");
    }

    Serial.println();
}

}  // namespace

void begin(PresenceCallback on_presence_changed) {
    presence_callback = on_presence_changed;

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    digitalWrite(TRIG_PIN, LOW);

    last_measurement_ms = millis() - MEASUREMENT_INTERVAL_MS;

    Serial.println("Proximity sensor state monitor");
    Serial.println("Detection threshold: 26 inches");
    Serial.println("Inactivity timer: 3 seconds");
    Serial.println("--------------------------------");
}

void update() {
    uint32_t now = millis();

    if (now - last_measurement_ms >= MEASUREMENT_INTERVAL_MS) {
        last_measurement_ms = now;
        latest_distance_in = measure_distance_inches();
        user_detected =
            latest_distance_in >= 0.0f &&
            latest_distance_in <= MAX_DETECTION_DISTANCE_IN;
    }

    now = millis();

    if (user_detected) {
        // Presence activates the system and cancels any pending shutdown.
        if (!active && presence_callback != nullptr) {
            presence_callback(true);
        }

        active = true;
        timer_running = false;
        timer_expired = false;
    } else {
        // Start the countdown once, when an active system becomes clear.
        if (active && !timer_running) {
            timer_running = true;
            timer_expired = false;
            inactivity_timer_start = now;
        }

        if (timer_running &&
            now - inactivity_timer_start >= INACTIVITY_TIME_MS) {
            active = false;
            timer_running = false;
            timer_expired = true;

            if (presence_callback != nullptr) {
                presence_callback(false);
            }
        }
    }

    print_status(now);
}

}  // namespace proximity