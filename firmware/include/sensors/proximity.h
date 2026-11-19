#pragma once

#include "sensors/distance_zones.h"

namespace proximity {

using ZoneCallback = void (*)(Zone zone, float distance_inches);

void begin(ZoneCallback on_zone_changed);
void update();
Zone current_zone();
float latest_distance_inches(); // -1.0f = missing echo

} // namespace proximity
