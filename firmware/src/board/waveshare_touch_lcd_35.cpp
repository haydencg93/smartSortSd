#include "board/waveshare_touch_lcd_35.h"

#include <Arduino.h>
#include <Arduino_GFX_Library.h>
#include <Wire.h>
#include <esp_heap_caps.h>
#include <lvgl.h>

#include "ui/ui.h"

namespace board {
namespace {

constexpr int32_t PANEL_W = 320;
constexpr int32_t PANEL_H = 480;
constexpr int32_t DRAW_BUF_LINES = 32;

constexpr int LCD_BACKLIGHT = 6;
constexpr int LCD_MISO = 2;
constexpr int LCD_MOSI = 1;
constexpr int LCD_SCLK = 5;
constexpr int LCD_CS = -1;
constexpr int LCD_DC = 3;
constexpr int LCD_RST = -1;

constexpr int I2C_SDA = 8;
constexpr int I2C_SCL = 7;
constexpr uint8_t IO_EXPANDER_ADDR = 0x20;
constexpr uint8_t TOUCH_ADDR = 0x38;

constexpr uint8_t TCA_OUTPUT_PORT = 0x01;
constexpr uint8_t TCA_CONFIG_PORT = 0x03;
constexpr uint8_t LCD_RESET_PIN = 1;

Arduino_ESP32SPI lcd_bus(LCD_DC, LCD_CS, LCD_SCLK, LCD_MOSI, LCD_MISO);
Arduino_ST7796 lcd(&lcd_bus, LCD_RST, 1, true, PANEL_W, PANEL_H);

bool write_i2c_register(uint8_t address, uint8_t reg, uint8_t value) {
    Wire.beginTransmission(address);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
}

bool i2c_device_present(uint8_t address) {
    Wire.beginTransmission(address);
    return Wire.endTransmission() == 0;
}

bool reset_lcd() {
    const uint8_t reset_mask = static_cast<uint8_t>(1U << LCD_RESET_PIN);
    if (!write_i2c_register(IO_EXPANDER_ADDR, TCA_OUTPUT_PORT, reset_mask) ||
        !write_i2c_register(IO_EXPANDER_ADDR, TCA_CONFIG_PORT,
                            static_cast<uint8_t>(~reset_mask))) {
        return false;
    }

    delay(10);
    if (!write_i2c_register(IO_EXPANDER_ADDR, TCA_OUTPUT_PORT, 0)) return false;
    delay(10);
    if (!write_i2c_register(IO_EXPANDER_ADDR, TCA_OUTPUT_PORT, reset_mask)) return false;
    delay(200);
    return true;
}

bool read_touch_point(uint16_t& x, uint16_t& y, bool& pressed) {
    Wire.beginTransmission(TOUCH_ADDR);
    Wire.write(0x02);
    if (Wire.endTransmission(false) != 0 ||
        Wire.requestFrom(static_cast<int>(TOUCH_ADDR), 5) != 5) {
        while (Wire.available()) Wire.read();
        return false;
    }

    const uint8_t points = Wire.read() & 0x0F;
    const uint8_t x_high = Wire.read() & 0x0F;
    const uint8_t x_low = Wire.read();
    const uint8_t y_high = Wire.read() & 0x0F;
    const uint8_t y_low = Wire.read();
    pressed = points != 0;
    if (!pressed) return true;

    x = (static_cast<uint16_t>(x_high) << 8) | x_low;
    y = (static_cast<uint16_t>(y_high) << 8) | y_low;
    return true;
}

void flush_cb(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map) {
    const uint32_t width = static_cast<uint32_t>(lv_area_get_width(area));
    const uint32_t height = static_cast<uint32_t>(lv_area_get_height(area));
    auto* pixels = reinterpret_cast<uint16_t*>(px_map);

#if defined(LV_COLOR_16_SWAP) && LV_COLOR_16_SWAP
    lcd.draw16bitBeRGBBitmap(area->x1, area->y1, pixels, width, height);
#else
    lcd.draw16bitRGBBitmap(area->x1, area->y1, pixels, width, height);
#endif
    lv_display_flush_ready(disp);
}

void touch_cb(lv_indev_t*, lv_indev_data_t* data) {
    uint16_t panel_x = 0;
    uint16_t panel_y = 0;
    bool pressed = false;
    if (!read_touch_point(panel_x, panel_y, pressed)) {
        static uint32_t last_error_log = 0;
        const uint32_t now = millis();
        if (now - last_error_log >= 5000) {
            Serial.println("WARNING: Failed to read the FT6336 touch controller over I2C.");
            last_error_log = now;
        }
        data->state = LV_INDEV_STATE_RELEASED;
        return;
    }
    if (!pressed || panel_x >= PANEL_W || panel_y >= PANEL_H) {
        data->state = LV_INDEV_STATE_RELEASED;
        return;
    }

    data->point.x = panel_y;
    data->point.y = PANEL_W - 1 - panel_x;
    data->state = LV_INDEV_STATE_PRESSED;
}

[[noreturn]] void halt_with_error(const char* message) {
    Serial.println(message);
    while (true) delay(1000);
}

}  // namespace

void init_waveshare_touch_lcd_35() {
    delay(100);
    if (!Wire.begin(I2C_SDA, I2C_SCL)) {
        halt_with_error("ERROR: Could not start I2C on GPIO 8/7.");
    }
    if (!reset_lcd()) {
        halt_with_error("ERROR: LCD reset via TCA9554 at I2C address 0x20 failed.");
    }
    if (!i2c_device_present(TOUCH_ADDR)) {
        halt_with_error("ERROR: FT6336 touch controller not found at I2C address 0x38.");
    }
    if (!lcd.begin()) {
        halt_with_error("ERROR: ST7796 display initialization failed.");
    }

    pinMode(LCD_BACKLIGHT, OUTPUT);
    digitalWrite(LCD_BACKLIGHT, HIGH);
    lcd.fillScreen(RGB565_BLACK);

    lv_display_t* disp = lv_display_create(ui::W, ui::H);
    lv_display_set_color_format(disp, LV_COLOR_FORMAT_RGB565);
    lv_display_set_flush_cb(disp, flush_cb);

    const size_t buffer_bytes =
        static_cast<size_t>(ui::W) * DRAW_BUF_LINES * sizeof(lv_color16_t);
    void* draw_buffer = heap_caps_malloc(buffer_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (draw_buffer == nullptr) {
        halt_with_error("ERROR: Could not allocate the LVGL draw buffer.");
    }
    lv_display_set_buffers(disp, draw_buffer, nullptr, buffer_bytes, LV_DISPLAY_RENDER_MODE_PARTIAL);

    lv_indev_t* touch = lv_indev_create();
    lv_indev_set_type(touch, LV_INDEV_TYPE_POINTER);
    lv_indev_set_read_cb(touch, touch_cb);
}

}  // namespace board
