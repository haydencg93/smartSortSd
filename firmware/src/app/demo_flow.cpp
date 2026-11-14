#include "app/demo_flow.h"
#include "ui/ui.h"

#include <stdio.h>

static lv_timer_t* pending_flow_timer = nullptr;

// Helper: run fn once after `ms` milliseconds.
static void after(uint32_t ms, lv_timer_cb_t fn) {
    if (pending_flow_timer != nullptr) {
        lv_timer_del(pending_flow_timer);
    }

    pending_flow_timer = lv_timer_create(fn, ms, nullptr);
    lv_timer_set_repeat_count(pending_flow_timer, 1);
}

// Step 3 (fake): "classifier" answers. Alternates recycle / trash so you can see both screens.
static void step_result(lv_timer_t* timer) {
    if (pending_flow_timer == timer) {
        pending_flow_timer = nullptr;
    }

    static bool recycle = true;
    if (recycle) ui::show_result(ui::Bin::Recycle, "Plastic bottle");
    else         ui::show_result(ui::Bin::Trash, "Greasy pizza box");
    recycle = !recycle;
}

// Step 2 (fake): "camera captured something".
static void step_analyze(lv_timer_t* timer) {
    if (pending_flow_timer == timer) {
        pending_flow_timer = nullptr;
    }

    if(!ui::capture_scan_image()){
        printf("[Camera] Warning: Continuing without a captured image.\n");
    }

    ui::show_analyzing();
    after(2000, step_result);
}

// Step 1 (fake): "proximity sensor fired". On the real device this comes from the HC-SR04/VL53L1X.
static void on_idle_tapped() {
    ui::show_scan_prompt();
    after(5000, step_analyze);
}

void demo_flow_start_scan(){
    on_idle_tapped();
}

void demo_flow_cancel() {
    if (pending_flow_timer != nullptr) {
        lv_timer_del(pending_flow_timer);
        pending_flow_timer = nullptr;
    }
}

// The UI reports what the user decided. On the real device: POST this to your server.
static void on_feedback(const ui::Feedback& f) {
    printf("[feedback] item=\"%s\" shown=%s correct=%s reason=%d\n",
           f.item,
           f.shown   == ui::Bin::Recycle ? "recycle" : "trash",
           f.correct == ui::Bin::Recycle ? "recycle" : "trash",
           (int)f.reason);
}

void demo_flow_init() {
    ui::Callbacks cb;
    cb.on_idle_tapped = on_idle_tapped;
    cb.on_feedback    = on_feedback;
    ui::init(cb);
}
