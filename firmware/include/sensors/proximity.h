#pragma once

namespace proximity {

using PresenceCallback = void (*)(bool active);

void begin(PresenceCallback on_presence_changed);
void update();

}  // namespace proximity