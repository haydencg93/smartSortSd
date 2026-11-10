#include <unity.h>
#include <lvgl.h>

#include <cstring>

#include "ui/ui.h"

namespace {

constexpr uint32_t DISPLAY_WIDTH = 480;
constexpr uint32_t DISPLAY_HEIGHT = 320;
constexpr uint32_t BUFFER_LINES = 10;

lv_color_t draw_buffer[DISPLAY_WIDTH * BUFFER_LINES];
uint32_t idle_tap_count = 0;
uint32_t feedback_count = 0;
ui::Feedback last_feedback{};
char last_item[48]{};

void flush_cb(lv_display_t* display, const lv_area_t*, uint8_t*) {
    lv_display_flush_ready(display);
}

void on_idle_tapped() {
    ++idle_tap_count;
}

void on_feedback(const ui::Feedback& feedback) {
    ++feedback_count;
    last_feedback = feedback;
    std::strncpy(last_item, feedback.item, sizeof(last_item) - 1);
    last_item[sizeof(last_item) - 1] = '\0';
    last_feedback.item = last_item;
}

lv_obj_t* find_text(lv_obj_t* root, const char* text) {
    if (lv_obj_check_type(root, &lv_label_class) &&
        std::strcmp(lv_label_get_text(root), text) == 0) {
        return root;
    }
    for (uint32_t i = 0; i < lv_obj_get_child_count(root); ++i) {
        if (lv_obj_t* match = find_text(lv_obj_get_child(root, static_cast<int32_t>(i)), text)) {
            return match;
        }
    }
    return nullptr;
}

lv_obj_t* find_button(lv_obj_t* root, const char* text) {
    if (lv_obj_check_type(root, &lv_button_class) && find_text(root, text) != nullptr) {
        return root;
    }
    for (uint32_t i = 0; i < lv_obj_get_child_count(root); ++i) {
        if (lv_obj_t* match = find_button(lv_obj_get_child(root, static_cast<int32_t>(i)), text)) {
            return match;
        }
    }
    return nullptr;
}

void advance_time(uint32_t duration_ms) {
    for (uint32_t elapsed = 0; elapsed < duration_ms; elapsed += 5) {
        lv_tick_inc(5);
        lv_timer_handler();
    }
}

void click_button(const char* text) {
    lv_obj_t* button = find_button(lv_screen_active(), text);
    TEST_ASSERT_NOT_NULL_MESSAGE(button, text);
    lv_obj_send_event(button, LV_EVENT_CLICKED, nullptr);
    advance_time(200);
}

void test_ui_navigation_and_feedback() {
    lv_init();
    lv_display_t* display = lv_display_create(DISPLAY_WIDTH, DISPLAY_HEIGHT);
    lv_display_set_flush_cb(display, flush_cb);
    lv_display_set_buffers(display, draw_buffer, nullptr, sizeof(draw_buffer),
                           LV_DISPLAY_RENDER_MODE_PARTIAL);

    ui::Callbacks callbacks;
    callbacks.on_idle_tapped = on_idle_tapped;
    callbacks.on_feedback = on_feedback;
    ui::init(callbacks);

    lv_obj_send_event(lv_screen_active(), LV_EVENT_CLICKED, nullptr);
    TEST_ASSERT_EQUAL_UINT32(1, idle_tap_count);

    ui::show_scan_prompt();
    advance_time(200);
    TEST_ASSERT_NOT_NULL(find_text(lv_screen_active(), "Barcodes scan automatically"));

    ui::show_analyzing();
    advance_time(200);
    TEST_ASSERT_NOT_NULL(find_text(lv_screen_active(), "Usually under 5 seconds"));

    ui::show_result(ui::Bin::Recycle, "Plastic bottle");
    advance_time(200);
    TEST_ASSERT_NOT_NULL(find_text(lv_screen_active(), "Detected: Plastic bottle"));
    click_button("That's wrong");
    click_button("Trash");
    click_button("Wrong material");

    TEST_ASSERT_EQUAL_UINT32(1, feedback_count);
    TEST_ASSERT_EQUAL(static_cast<int>(ui::Bin::Recycle), static_cast<int>(last_feedback.shown));
    TEST_ASSERT_EQUAL(static_cast<int>(ui::Bin::Trash), static_cast<int>(last_feedback.correct));
    TEST_ASSERT_EQUAL(static_cast<int>(ui::Reason::WrongMaterial), static_cast<int>(last_feedback.reason));
    TEST_ASSERT_EQUAL_STRING("Plastic bottle", last_feedback.item);
    TEST_ASSERT_NOT_NULL(find_text(lv_screen_active(), "Corrected by you"));

    ui::show_result(ui::Bin::Trash, "Pizza box");
    advance_time(200);
    click_button("Looks right");
    TEST_ASSERT_EQUAL_UINT32(2, feedback_count);
    TEST_ASSERT_EQUAL(static_cast<int>(ui::Bin::Trash), static_cast<int>(last_feedback.shown));
    TEST_ASSERT_EQUAL(static_cast<int>(ui::Bin::Trash), static_cast<int>(last_feedback.correct));
    TEST_ASSERT_EQUAL(static_cast<int>(ui::Reason::None), static_cast<int>(last_feedback.reason));
    TEST_ASSERT_EQUAL_STRING("Pizza box", last_feedback.item);
}

}  // namespace

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_ui_navigation_and_feedback);
    return UNITY_END();
}
