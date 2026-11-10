// Desktop simulator entry point (env:sim). Opens a 480x320 window; mouse = touchscreen.
#include <lvgl.h>
#include <SDL2/SDL.h>
#include "app/demo_flow.h"
#include "ui/ui.h"

static uint32_t tick_cb() { return SDL_GetTicks(); }

int main() {
    // Initialize LVGL and SDL, create a window, and set up the mouse as a touchscreen input device.
    lv_init();
    lv_tick_set_cb(tick_cb);

    // Create the simulator display at the UI's logical landscape size.
    lv_display_t *disp = lv_sdl_window_create(ui::W, ui::H);
    lv_sdl_window_set_title(disp, "Smart Sort UI");
    lv_sdl_mouse_create();

    demo_flow_init();

    // Main loop: let LVGL process timers and SDL input.
    while (true)
    {
        lv_timer_handler();
        SDL_Delay(5);
    }
    return 0;
}
