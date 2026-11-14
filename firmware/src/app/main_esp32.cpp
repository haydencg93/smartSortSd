#include <Arduino.h>
#include <lvgl.h>

#include "app/demo_flow.h"
#include "board/waveshare_touch_lcd_35.h"
#include "sensors/proximity.h"
#include "ui/ui.h"
#include "sensors/camera.h"

namespace {
    uint32_t tick_cb() {
        return millis();
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
    lv_init();
    lv_tick_set_cb(tick_cb);
    board::init_waveshare_touch_lcd_35();

    if (!camera::begin()){
        Serial.println("WARNING: Camera initialization failed.");
    } else{
        camera::capture_test();
    }

    demo_flow_init();
    proximity::begin(on_presence_changed);
    Serial.println("Smart Sort UI ready. Tap the idle screen to run the demo flow.");
}

void loop() {
    lv_timer_handler();
    proximity::update();
    delay(5);
}
