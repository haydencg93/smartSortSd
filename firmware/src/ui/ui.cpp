#include "ui/ui.h"
#include "sensors/camera.h"

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <stdio.h>

namespace ui {
namespace {

// ---------------------------------------------------------------------------
// Palette (taken from the mockups)
// ---------------------------------------------------------------------------
#define C_GREEN   lv_color_hex(0x2E6B57)   // headers, idle accent
#define C_MINT    lv_color_hex(0xDFF3EC)   // idle background, success banner
#define C_BLUE    lv_color_hex(0x2E5FA8)   // recycle
#define C_DARK    lv_color_hex(0x3F3F3F)   // trash
#define C_RUST    lv_color_hex(0x8B3F2A)   // correction flow headers / outlines
#define C_PEACH   lv_color_hex(0xFBE5DE)   // "That's wrong" button
#define C_GREY    lv_color_hex(0xEFEBE4)   // Cancel / Skip
#define C_WHITE   lv_color_hex(0xFFFFFF)
#define C_TEXT    lv_color_hex(0x33443D)   // body text on white

// Timeouts: if the user walks away mid-flow, fall back to Idle.
constexpr uint32_t T_SCAN_MS      = 30000;
constexpr uint32_t T_ANALYZE_MS   = 10000;
constexpr uint32_t T_RESULT_MS    = 20000;
constexpr uint32_t T_CORRECT_MS   = 20000;
constexpr uint32_t T_THANKS_MS    = 3000;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct State {
    Callbacks cb;

    Bin  shown     = Bin::Recycle;   // what the classifier said
    Bin  corrected = Bin::Recycle;   // what the user says is right
    char item[48]  = "";

    lv_timer_t* timeout = nullptr;   // the one active inactivity timer

    // screens
    lv_obj_t* scr_idle      = nullptr;
    lv_obj_t* scr_scan      = nullptr;
    lv_obj_t* scr_analyzing = nullptr;
    lv_obj_t* scr_result    = nullptr;   // used for screens 4, 5 and 8
    lv_obj_t* scr_pick      = nullptr;
    lv_obj_t* scr_reason    = nullptr;

    // widgets we need to update later
    lv_obj_t* bar          = nullptr;
    lv_obj_t* res_title    = nullptr;
    lv_obj_t* res_hint     = nullptr;
    lv_obj_t* res_arrow    = nullptr;
    lv_obj_t* res_detected = nullptr;
    lv_obj_t* res_buttons  = nullptr;
    lv_obj_t* res_ok_label = nullptr;
    lv_obj_t* res_banner   = nullptr;
    bool      arrow_right  = false;

    // Live camera preview
    lv_obj_t* preview_image = nullptr;
    lv_obj_t* preview_status = nullptr;
    lv_timer_t* preview_timer = nullptr;

    uint8_t* preview_buffer = nullptr;
    lv_image_dsc_t preview_dsc = {};

};

State S;

// ---------------------------------------------------------------------------
// Small helpers so the screen builders stay readable
// ---------------------------------------------------------------------------
lv_obj_t* new_screen(lv_color_t bg) {
    lv_obj_t* s = lv_obj_create(nullptr);          // nullptr parent = a full screen
    lv_obj_set_style_bg_color(s, bg, 0);
    lv_obj_set_style_bg_opa(s, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(s, 0, 0);
    lv_obj_set_style_radius(s, 0, 0);
    lv_obj_set_style_pad_all(s, 0, 0);
    lv_obj_remove_flag(s, LV_OBJ_FLAG_SCROLLABLE);
    return s;
}

lv_obj_t* make_label(lv_obj_t* parent, const char* text, const lv_font_t* font, lv_color_t color) {
    lv_obj_t* l = lv_label_create(parent);
    lv_label_set_text(l, text);
    lv_obj_set_style_text_font(l, font, 0);
    lv_obj_set_style_text_color(l, color, 0);
    return l;
}

// A transparent, non-interactive box (used for grouping and for the arrow canvas).
lv_obj_t* make_box(lv_obj_t* parent, int32_t x, int32_t y, int32_t w, int32_t h) {
    lv_obj_t* o = lv_obj_create(parent);
    lv_obj_set_pos(o, x, y);
    lv_obj_set_size(o, w, h);
    lv_obj_set_style_bg_opa(o, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(o, 0, 0);
    lv_obj_set_style_pad_all(o, 0, 0);
    lv_obj_set_style_radius(o, 0, 0);
    lv_obj_remove_flag(o, LV_OBJ_FLAG_SCROLLABLE);
    return o;
}

lv_obj_t* make_header(lv_obj_t* scr, const char* text, lv_color_t bg) {
    lv_obj_t* bar = make_box(scr, 0, 0, W, 56);
    lv_obj_set_style_bg_color(bar, bg, 0);
    lv_obj_set_style_bg_opa(bar, LV_OPA_COVER, 0);
    lv_obj_t* l = make_label(bar, text, &lv_font_montserrat_20, C_WHITE);
    lv_obj_center(l);
    return bar;
}

// Rounded button with centered label. Returns the button; label via lv_obj_get_child(btn, 0).
lv_obj_t* make_button(lv_obj_t* parent, const char* text, int32_t x, int32_t y, int32_t w, int32_t h,
                      lv_color_t bg, lv_color_t fg, const lv_font_t* font,
                      int32_t border_w, lv_color_t border_c,
                      lv_event_cb_t cb, void* user) {
    lv_obj_t* b = lv_button_create(parent);
    lv_obj_set_pos(b, x, y);
    lv_obj_set_size(b, w, h);
    lv_obj_set_style_bg_color(b, bg, 0);
    lv_obj_set_style_radius(b, 14, 0);
    lv_obj_set_style_shadow_width(b, 0, 0);
    lv_obj_set_style_border_width(b, border_w, 0);
    lv_obj_set_style_border_color(b, border_c, 0);
    lv_obj_set_style_opa(b, LV_OPA_70, LV_STATE_PRESSED);   // simple press feedback
    lv_obj_t* l = make_label(b, text, font, fg);
    lv_obj_center(l);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, user);
    return b;
}

// ---------------------------------------------------------------------------
// Navigation plumbing
// ---------------------------------------------------------------------------
void cancel_timeout() {
    if (S.timeout) { lv_timer_delete(S.timeout); S.timeout = nullptr; }
}

// One-shot inactivity timer. Only one exists at a time; entering a new screen replaces it.
void arm_timeout(uint32_t ms, lv_timer_cb_t cb) {
    cancel_timeout();
    S.timeout = lv_timer_create(cb, ms, nullptr);
    lv_timer_set_repeat_count(S.timeout, 1);   // LVGL deletes it after it fires
}

void go(lv_obj_t* scr, bool animate = true) {
    if (animate) lv_screen_load_anim(scr, LV_SCR_LOAD_ANIM_FADE_IN, 150, 0, false);
    else         lv_screen_load(scr);
}

void timeout_to_idle(lv_timer_t*) {
    S.timeout = nullptr;             // LVGL frees this timer after the callback
    show_idle();
}

// ---------------------------------------------------------------------------
// Screen 4/5/8: result screen (one screen, restyled per result)
// ---------------------------------------------------------------------------
void arrow_draw_cb(lv_event_t* e) {
    lv_layer_t* layer = lv_event_get_layer(e);
    lv_obj_t*   obj   = lv_event_get_target_obj(e);
    lv_area_t a;
    lv_obj_get_coords(obj, &a);

    const int32_t h    = lv_area_get_height(&a);
    const int32_t head = h;                 // arrow head is as wide as the box is tall
    const int32_t midy = (a.y1 + a.y2) / 2;
    const int32_t sh   = h / 2;             // shaft thickness

    lv_draw_rect_dsc_t rd;
    lv_draw_rect_dsc_init(&rd);
    rd.bg_color = C_WHITE;
    rd.bg_opa   = LV_OPA_COVER;

    lv_draw_triangle_dsc_t td;
    lv_draw_triangle_dsc_init(&td);
    td.bg_color = C_WHITE;
    td.bg_opa   = LV_OPA_COVER;

    lv_area_t shaft;
    shaft.y1 = midy - sh / 2;
    shaft.y2 = midy + sh / 2;

    if (S.arrow_right) {
        shaft.x1 = a.x1;         shaft.x2 = a.x2 - head + 2;
        td.p[0].x = a.x2 - head; td.p[0].y = a.y1;
        td.p[1].x = a.x2;        td.p[1].y = midy;
        td.p[2].x = a.x2 - head; td.p[2].y = a.y2;
    } else {
        shaft.x1 = a.x1 + head - 2; shaft.x2 = a.x2;
        td.p[0].x = a.x1 + head;    td.p[0].y = a.y1;
        td.p[1].x = a.x1;           td.p[1].y = midy;
        td.p[2].x = a.x1 + head;    td.p[2].y = a.y2;
    }
    lv_draw_rect(layer, &rd, &shaft);
    lv_draw_triangle(layer, &td);
}

// Restyle the result screen. corrected=true gives the "Corrected by you" version (screen 8).
void apply_result(Bin bin, bool corrected) {
    const bool rec = (bin == Bin::Recycle);
    const lv_color_t bg = rec ? C_BLUE : C_DARK;

    lv_obj_set_style_bg_color(S.scr_result, bg, 0);
    lv_label_set_text(S.res_title, rec ? "RECYCLE" : "TRASH");
    lv_label_set_text(S.res_hint,  rec ? "Place in the\nblue bin" : "Place in the\nblack bin");

    // Recycle bin is on the left, trash bin on the right (matches the mockups).
    S.arrow_right = !rec;
    lv_obj_set_pos(S.res_arrow, rec ? 30 : 330, 92);
    lv_obj_set_pos(S.res_hint,  rec ? 250 : 30, 100);
    lv_obj_invalidate(S.res_arrow);

    char buf[80];
    if (corrected) snprintf(buf, sizeof buf, "Corrected by you");
    else           snprintf(buf, sizeof buf, "Detected: %s", S.item);
    lv_label_set_text(S.res_detected, buf);

    lv_obj_set_style_text_color(S.res_ok_label, bg, 0);   // "Looks right" text matches bin colour

    if (corrected) {
        lv_obj_add_flag(S.res_buttons, LV_OBJ_FLAG_HIDDEN);
        lv_obj_remove_flag(S.res_banner, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_remove_flag(S.res_buttons, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(S.res_banner, LV_OBJ_FLAG_HIDDEN);
    }
}

// ---------------------------------------------------------------------------
// Button handlers (this is where navigation between pages happens)
// ---------------------------------------------------------------------------
void send_feedback(Bin correct, Reason reason) {
    if (S.cb.on_feedback) {
        Feedback f{S.shown, correct, reason, S.item};
        S.cb.on_feedback(f);
    }
}

// Screen 4/5 "Looks right"  ->  log confirmation, back to Idle
void on_looks_right(lv_event_t*) {
    send_feedback(S.shown, Reason::None);
    show_idle();
}

// Screen 4/5 "That's wrong"  ->  Screen 6
void on_thats_wrong(lv_event_t*) {
    arm_timeout(T_CORRECT_MS, timeout_to_idle);
    go(S.scr_pick);
}

// Screen 6 Recycle / Trash  ->  Screen 7
void on_pick_bin(lv_event_t* e) {
    S.corrected = (Bin)(intptr_t)lv_event_get_user_data(e);
    arm_timeout(T_CORRECT_MS, timeout_to_idle);
    go(S.scr_reason);
}

// Screen 6 Cancel  ->  back to the original result (no logging)
void on_cancel(lv_event_t*) {
    arm_timeout(T_RESULT_MS, timeout_to_idle);
    go(S.scr_result);
}

// Screen 7 reason / Skip  ->  log the correction, Screen 8, then Idle after 3s
void on_reason(lv_event_t* e) {
    Reason r = (Reason)(intptr_t)lv_event_get_user_data(e);
    send_feedback(S.corrected, r);
    apply_result(S.corrected, /*corrected=*/true);
    arm_timeout(T_THANKS_MS, timeout_to_idle);
    go(S.scr_result);
}

// Idle screen tap: hands off to the app (real hardware: proximity sensor does this instead)
void on_idle_tap(lv_event_t*) {
    if (S.cb.on_idle_tapped) S.cb.on_idle_tapped();
}

void bar_anim_cb(void* bar, int32_t v) { lv_bar_set_value((lv_obj_t*)bar, v, LV_ANIM_OFF); }

// ---------------------------------------------------------------------------
// Camera Preview
// ---------------------------------------------------------------------------
constexpr uint16_t PREVIEW_W = 320;
constexpr uint16_t PREVIEW_H = 240;
constexpr size_t PREVIEW_BYTES = static_cast<size_t>(PREVIEW_W) * PREVIEW_H *2;

void preview_update(lv_timer_t*){
    if (!camera::is_running() || S.preview_buffer == nullptr){
        return;
    }

    uint16_t width = 0;
    uint16_t height = 0;

    if(!camera::capture_rgb565(S.preview_buffer, PREVIEW_BYTES, width, height)) {
        return;
    }

    // The descriptor points to our persistent PSRAM buffer.
    // The camera's own frame buffer has already been returned.
    if(width != PREVIEW_W || height != PREVIEW_H){
        Serial.println("[Camera] Unexpected preview dimensions.");
        return;
    }

    lv_image_set_src(S.preview_image, &S.preview_dsc);
    lv_obj_invalidate(S.preview_image);

    //Hide the startup message after the first successful frame.
    if(S.preview_status){
        lv_obj_add_flag(S.preview_status, LV_OBJ_FLAG_HIDDEN);
    }
}

void stop_preview(){
    // Stop the LVGL timer before shutting down the camera.
    if(S.preview_timer != nullptr){
        lv_timer_delete(S.preview_timer);
        S.preview_timer = nullptr;
    }

    camera::stop();
}

void start_preview(){
    stop_preview();

    if(S.preview_buffer == nullptr){
        Serial.println("[Camera] Allocating preview buffer in PSRAM...");

        S.preview_buffer = static_cast<uint8_t*>(
            heap_caps_malloc(
                PREVIEW_BYTES,
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
            )
        );

        if (S.preview_buffer == nullptr){
            Serial.println("[Camera] ERROR: Could not allocate preview buffer.");

            lv_label_set_text(S.preview_status, "Preview memory unavailable");
            lv_obj_remove_flag(S.preview_status, LV_OBJ_FLAG_HIDDEN);
            return;
        }

        // Configure the LVGL 9 image descriptor.
        S.preview_dsc.header.magic = LV_IMAGE_HEADER_MAGIC;
        S.preview_dsc.header.cf = LV_COLOR_FORMAT_RGB565;
        S.preview_dsc.header.w = PREVIEW_W;
        S.preview_dsc.header.h = PREVIEW_H;
        S.preview_dsc.header.stride = PREVIEW_W * 2;
        S.preview_dsc.data_size = PREVIEW_BYTES;
        S.preview_dsc.data = S.preview_buffer;
    }

    lv_label_set_text(S.preview_status, "Starting camera...");
    lv_obj_remove_flag(S.preview_status, LV_OBJ_FLAG_HIDDEN);

    if(!camera::start()){
        Serial.println("[Camera] ERROR: Could not start preview");

        lv_label_set_text(S.preview_status, "Camera preview unavailable");
        return;
    }

    //Refresh the preview approximately five times per second
    S.preview_timer = lv_timer_create(preview_update, 200, nullptr);
}

// ---------------------------------------------------------------------------
// Screen builders
// ---------------------------------------------------------------------------
void build_idle() {                       // Screen 1
    S.scr_idle = new_screen(C_MINT);

    lv_obj_t* logo = make_box(S.scr_idle, (W - 80) / 2, 44, 80, 80);
    lv_obj_set_style_radius(logo, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_bg_color(logo, C_GREEN, 0);
    lv_obj_set_style_bg_opa(logo, LV_OPA_COVER, 0);
    lv_obj_center(make_label(logo, "SS", &lv_font_montserrat_28, C_WHITE));

    lv_obj_align(make_label(S.scr_idle, "Smart Sort", &lv_font_montserrat_28, C_GREEN),
                 LV_ALIGN_TOP_MID, 0, 145);
    lv_obj_align(make_label(S.scr_idle, "Step up to scan an item", &lv_font_montserrat_20, C_GREEN),
                 LV_ALIGN_TOP_MID, 0, 190);

    lv_obj_t* pill = make_box(S.scr_idle, (W - 200) / 2, 240, 200, 40);
    lv_obj_set_style_radius(pill, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_border_width(pill, 2, 0);
    lv_obj_set_style_border_color(pill, C_GREEN, 0);
    lv_obj_center(make_label(pill, "Waiting for a user", &lv_font_montserrat_16, C_GREEN));

    // Tap anywhere on the idle screen. (Children are non-clickable so taps reach the screen.)
    lv_obj_remove_flag(logo, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_remove_flag(pill, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_add_event_cb(S.scr_idle, on_idle_tap, LV_EVENT_CLICKED, nullptr);
}

void build_scan() {                       // Screen 2
    S.scr_scan = new_screen(C_WHITE);

    make_header(S.scr_scan, "Hold your item up to the camera", C_GREEN);

    // Green outline around the camera preview.
    lv_obj_t* frame = make_box(S.scr_scan, 76, 60, 328, 248);

    lv_obj_set_style_radius(frame, 14, 0);
    lv_obj_set_style_border_width(frame, 3, 0);
    lv_obj_set_style_border_color(frame, C_GREEN, 0);

    // The image itself is 320 x 240 pixels.
    S.preview_image = lv_image_create(S.scr_scan);
    lv_obj_set_pos(S.preview_image, 80, 64);

    // Message shown until the first camera frame arrives.
    S.preview_status = make_label(S.scr_scan, "Starting camera...", &lv_font_montserrat_16, C_TEXT );

    lv_obj_align(S.preview_status, LV_ALIGN_CENTER, 0, 20);
}

void build_analyzing() {                  // Screen 3
    S.scr_analyzing = new_screen(C_WHITE);
    make_header(S.scr_analyzing, "Checking your item...", C_GREEN);

    S.bar = lv_bar_create(S.scr_analyzing);
    lv_obj_set_size(S.bar, 300, 18);
    lv_obj_align(S.bar, LV_ALIGN_TOP_MID, 0, 130);
    lv_bar_set_range(S.bar, 0, 100);
    lv_obj_set_style_bg_color(S.bar, C_GREY, LV_PART_MAIN);
    lv_obj_set_style_bg_color(S.bar, C_GREEN, LV_PART_INDICATOR);

    lv_obj_align(make_label(S.scr_analyzing, "Usually under 5 seconds", &lv_font_montserrat_16, C_TEXT),
                 LV_ALIGN_TOP_MID, 0, 165);
}

void build_result() {                     // Screens 4, 5, 8 (same widgets, restyled)
    S.scr_result = new_screen(C_BLUE);

    S.res_title = make_label(S.scr_result, "RECYCLE", &lv_font_montserrat_48, C_WHITE);
    lv_obj_align(S.res_title, LV_ALIGN_TOP_MID, 0, 12);

    S.res_arrow = make_box(S.scr_result, 30, 92, 120, 80);
    lv_obj_add_event_cb(S.res_arrow, arrow_draw_cb, LV_EVENT_DRAW_MAIN, nullptr);

    S.res_hint = make_label(S.scr_result, "", &lv_font_montserrat_28, C_WHITE);
    lv_obj_set_width(S.res_hint, 200);
    lv_obj_set_style_text_align(S.res_hint, LV_TEXT_ALIGN_CENTER, 0);

    S.res_detected = make_label(S.scr_result, "", &lv_font_montserrat_16, lv_color_hex(0xD6E4F5));
    lv_obj_align(S.res_detected, LV_ALIGN_TOP_MID, 0, 196);

    // Action buttons (hidden on the "corrected" version)
    S.res_buttons = make_box(S.scr_result, 30, 236, 420, 60);
    lv_obj_t* ok = make_button(S.res_buttons, "Looks right", 0, 0, 200, 60, C_WHITE, C_BLUE,
                               &lv_font_montserrat_20, 0, C_WHITE, on_looks_right, nullptr);
    S.res_ok_label = lv_obj_get_child(ok, 0);
    make_button(S.res_buttons, "That's wrong", 220, 0, 200, 60, C_PEACH, C_RUST,
                &lv_font_montserrat_20, 2, C_RUST, on_thats_wrong, nullptr);

    // "Thanks" banner (shown instead of the buttons on the corrected version)
    S.res_banner = make_box(S.scr_result, 30, 240, 420, 52);
    lv_obj_set_style_radius(S.res_banner, 14, 0);
    lv_obj_set_style_bg_color(S.res_banner, C_MINT, 0);
    lv_obj_set_style_bg_opa(S.res_banner, LV_OPA_COVER, 0);
    lv_obj_center(make_label(S.res_banner, "Thanks! Your correction was logged.",
                             &lv_font_montserrat_16, C_GREEN));
    lv_obj_add_flag(S.res_banner, LV_OBJ_FLAG_HIDDEN);

    S.item[0] = '\0';
    apply_result(Bin::Recycle, false);
}

void build_pick() {                       // Screen 6
    S.scr_pick = new_screen(C_WHITE);
    make_header(S.scr_pick, "Which bin is correct?", C_RUST);

    make_button(S.scr_pick, "Recycle", 30, 86, 200, 110, C_BLUE, C_WHITE,
                &lv_font_montserrat_28, 0, C_BLUE, on_pick_bin, (void*)(intptr_t)Bin::Recycle);
    make_button(S.scr_pick, "Trash", 250, 86, 200, 110, C_DARK, C_WHITE,
                &lv_font_montserrat_28, 0, C_DARK, on_pick_bin, (void*)(intptr_t)Bin::Trash);
    make_button(S.scr_pick, "Cancel", (W - 170) / 2, 226, 170, 56, C_GREY, C_TEXT,
                &lv_font_montserrat_20, 2, lv_color_hex(0xBDB7AB), on_cancel, nullptr);
}

void build_reason() {                     // Screen 7
    S.scr_reason = new_screen(C_WHITE);
    make_header(S.scr_reason, "Why was it wrong? (optional)", C_RUST);

    struct Opt { const char* text; Reason r; int32_t x, y; };
    const Opt opts[] = {
        {"Wrong material", Reason::WrongMaterial, 30,  76},
        {"Greasy or dirty", Reason::GreasyOrDirty, 250, 76},
        {"Item not shown", Reason::ItemNotShown,  30,  146},
        {"Something else", Reason::SomethingElse, 250, 146},
    };
    for (const Opt& o : opts) {
        make_button(S.scr_reason, o.text, o.x, o.y, 200, 58, C_WHITE, C_RUST,
                    &lv_font_montserrat_20, 2, C_RUST, on_reason, (void*)(intptr_t)o.r);
    }
    make_button(S.scr_reason, "Skip", (W - 170) / 2, 230, 170, 56, C_GREY, C_TEXT,
                &lv_font_montserrat_20, 2, lv_color_hex(0xBDB7AB), on_reason, (void*)(intptr_t)Reason::None);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void init(const Callbacks& cb) {
    S.cb = cb;
    build_idle();
    build_scan();
    build_analyzing();
    build_result();
    build_pick();
    build_reason();
    go(S.scr_idle, /*animate=*/false);
}

void show_idle() {
    stop_preview();
    cancel_timeout();
    go(S.scr_idle);
}

void show_scan_prompt() {
    arm_timeout(T_SCAN_MS, timeout_to_idle);
    go(S.scr_scan);
    start_preview();
}

bool capture_scan_image() {
    if (!camera::is_running() || S.preview_buffer == nullptr) {
        Serial.println("[Camera] ERROR: Camera or preview buffer unavailable.");
        return false;
    }

    uint16_t width = 0;
    uint16_t height = 0;

    if (!camera::capture_rgb565(
            S.preview_buffer,
            PREVIEW_BYTES,
            width,
            height)) {
        Serial.println("[Camera] ERROR: Image capture failed.");
        return false;
    }

    if (width != PREVIEW_W || height != PREVIEW_H) {
        Serial.println("[Camera] ERROR: Unexpected image dimensions.");
        return false;
    }

    // Keep the successfully captured frame in the preview buffer.
    lv_image_set_src(S.preview_image, &S.preview_dsc);
    lv_obj_invalidate(S.preview_image);

    Serial.println("[Camera] Final image captured successfully.");
    return true;
}

void show_analyzing() {
    stop_preview();
    arm_timeout(T_ANALYZE_MS, timeout_to_idle);

    // Fake-progress bar: eases to 90% over ~4.5 s; the real result replaces this screen.
    lv_anim_delete(S.bar, bar_anim_cb);
    lv_bar_set_value(S.bar, 0, LV_ANIM_OFF);
    lv_anim_t a;
    lv_anim_init(&a);
    lv_anim_set_var(&a, S.bar);
    lv_anim_set_exec_cb(&a, bar_anim_cb);
    lv_anim_set_values(&a, 0, 90);
    lv_anim_set_duration(&a, 4500);
    lv_anim_set_path_cb(&a, lv_anim_path_ease_out);
    lv_anim_start(&a);

    go(S.scr_analyzing);
}

void show_result(Bin bin, const char* item_name) {
    stop_preview();
    S.shown = bin;
    snprintf(S.item, sizeof S.item, "%s", item_name ? item_name : "Unknown item");
    apply_result(bin, /*corrected=*/false);
    arm_timeout(T_RESULT_MS, timeout_to_idle);
    go(S.scr_result);
}

}  // namespace ui
