#include <Arduino.h>
#include <lvgl.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClient.h>

#include "app/demo_flow.h"
#include "board/waveshare_touch_lcd_35.h"
#include "sensors/proximity.h"
#include "ui/ui.h"
#include "secrets.h"

namespace {
    bool wifi_error_shown = false;
    bool previous_wifi_up = false;

    uint32_t tick_cb() {
        return millis();
    }

    bool connect_wifi() {
        Serial.println();
        Serial.println("[WiFi] Connecting to Android hotspot...");

        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

        const uint32_t start_time = millis();
        constexpr uint32_t WIFI_TIMEOUT_MS = 15000;

        while (WiFi.status() != WL_CONNECTED &&
               millis() - start_time < WIFI_TIMEOUT_MS) {
            delay(500);
            Serial.print(".");
        }

        Serial.println();

        if (WiFi.status() == WL_CONNECTED) {
            Serial.println("[WiFi] Connected successfully.");
            Serial.print("[WiFi] ESP32 IP address: ");
            Serial.println(WiFi.localIP());
            Serial.print("[WiFi] Signal strength: ");
            Serial.print(WiFi.RSSI());
            Serial.println(" dBm");
        } else {
            Serial.println("[WiFi] Connection failed or timed out.");
            Serial.println("[WiFi] The kiosk will continue without WiFi.");
        }

        return WiFi.status() == WL_CONNECTED;
    }
    
    void test_classifier_api() {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[API] WiFi is not connected. Skipping API test.");
            return;
        }
        
        WiFiClient client;
        HTTPClient http;
        
        const char* url = "http://10.67.162.143:8000/health";
        
        Serial.println("[API] Testing connection to classifier...");
        
        http.setConnectTimeout(5000);
        http.setTimeout(5000);
        
        if (!http.begin(client, url)) {
            Serial.println("[API] Could not initialize HTTP connection.");
            return;
        }
        
        int status_code = http.GET();
        
        if (status_code > 0) {
            Serial.printf("[API] HTTP status: %d\n", status_code);
            Serial.print("[API] Response: ");
            Serial.println(http.getString());
        } else {
            Serial.print("[API] Request failed: ");
            Serial.println(http.errorToString(status_code));
        }
        
        http.end();
    }
}

// Proximity is the ONLY owner of standby/welcome/camera screen transitions.
// One callback per debounced zone transition; no timer-based photograph.
void on_distance_zone_changed(proximity::Zone zone, float distance_inches) {
    Serial.printf("[App] Zone=%s distance=%.1fin\n", proximity::zone_name(zone), distance_inches);

    if (zone == proximity::Zone::Standby) {
        demo_flow_cancel();
        ui::show_idle();
        board::backlight_off();
        return;
    }

    board::backlight_on();
    if (WiFi.status() != WL_CONNECTED) {
        demo_flow_cancel();
        if (!wifi_error_shown) {
            ui::show_wifi_error();
            wifi_error_shown = true;
        }
        return;
    }
    wifi_error_shown = false;

    if (zone == proximity::Zone::Welcome) {
        demo_flow_cancel();
        ui::show_idle();
    } else if (zone == proximity::Zone::Camera) {
        demo_flow_start_scan();
    }
}

void setup() {
    Serial.begin(115200);

    bool wifi_connected = connect_wifi();
    previous_wifi_up = wifi_connected;

    if(wifi_connected) {
        test_classifier_api();
    }

    lv_init();
    lv_tick_set_cb(tick_cb);
    board::init_waveshare_touch_lcd_35();

    demo_flow_init();

    proximity::begin(on_distance_zone_changed);

    if (!wifi_connected) {
        // Honor standby on startup. Display the Wi-Fi error only when someone
        // approaches the HC-SR04, not while the device should be asleep.
        Serial.println("[WiFi] Offline; error screen will appear when a user approaches.");
    }

    Serial.println("Smart Sort UI ready.");
}

void loop() {
    lv_timer_handler();
    proximity::update();
    const bool wifi_up = WiFi.status() == WL_CONNECTED;
    if (wifi_up != previous_wifi_up) {
        previous_wifi_up = wifi_up;
        if (proximity::current_zone() != proximity::Zone::Standby) {
            on_distance_zone_changed(proximity::current_zone(),
                                     proximity::latest_distance_inches());
        }
    }
    demo_flow_update(proximity::current_zone() == proximity::Zone::Camera && wifi_up);
    delay(5);
}
