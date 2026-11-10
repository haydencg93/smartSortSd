# Smart Sort UI

LVGL 9 touchscreen UI for the Waveshare ESP32-S3-Touch-LCD-3.5, with a desktop simulator for development.

## Project layout

```text
include/
  app/demo_flow.h
  board/waveshare_touch_lcd_35.h
  ui/ui.h
src/
  app/       Application entry points and demo flow
  board/     Waveshare display and touch drivers
  ui/        Hardware-independent LVGL screens
test/
  test_ui/   Native UI navigation and feedback tests
```

The PlatformIO source filters compile only the files used by each target: the simulator, native UI tests, or the ESP32-S3 firmware.

## Build and run

Install PlatformIO and SDL2 (`brew install sdl2` on macOS), then run from this directory:

```sh
pio run -e sim
pio run -e sim -t exec
```

Run the native UI flow tests with:

```sh
pio test -e test_native
```

For the physical board, connect it over USB and run:

```sh
pio run -e esp32s3
pio run -e esp32s3 -t upload
pio device monitor -b 115200
```

The board profile targets the 16 MB flash / 8 MB OPI PSRAM ESP32-S3R8 module. The display is an ST7796 on SPI (GPIO 1/2/3/5/6); the FT6336 touch controller and TCA9554 LCD reset expander share I2C on GPIO 8/7. Display rotation and touch coordinates are configured for the UI's 480x320 landscape layout.

The current `demo_flow` is intentionally simulated: tapping the idle display runs a sample scan/classification flow and alternates example results. The UI reports confirmation/correction through its callbacks; connect those callbacks to the real proximity sensor, camera, classifier, and backend when those parts are ready.

Hardware and pin details: [Waveshare ESP32-S3-Touch-LCD-3.5 documentation](https://docs.waveshare.com/ESP32-S3-Touch-LCD-3.5).
