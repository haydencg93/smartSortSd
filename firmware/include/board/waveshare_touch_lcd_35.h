#pragma once

namespace board {

// Initialize the Waveshare display and touch input after lv_init().
void init_waveshare_touch_lcd_35();

void backlight_on();
void backlight_off();

}  // namespace board
