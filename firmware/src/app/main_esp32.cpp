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
    uint32_t tick_cb() {
        return millis();
    }

    void connect_wifi() {
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

void on_presence_changed(bool active) {
    if (active) {
        board::backlight_on();
        ui::show_idle();
    } else {
        demo_flow_cancel();
        ui::show_idle();
        board::backlight_off();
    }
}

void setup() {
    Serial.begin(115200);

    connect_wifi();
    test_classifier_api();

    lv_init();
    lv_tick_set_cb(tick_cb);
    board::init_waveshare_touch_lcd_35();
    demo_flow_init();
    proximity::begin(on_presence_changed);
    Serial.println("Smart Sort UI ready. Tap the idle screen to run the demo flow.");
}

void loop() {
    lv_timer_handler();
    proximity::update();
    delay(5);
}
