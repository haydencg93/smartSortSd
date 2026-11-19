#include "sensors/distance_zones.h"
#include "app/auto_scan.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    using namespace proximity;
    ZoneTracker z;
    assert(z.zone() == Zone::Standby);
    assert(!z.update(23.0f, 100));
    assert(z.update(23.0f, 500));
    assert(z.zone() == Zone::Welcome);
    assert(!z.update(11.0f, 700));
    assert(z.update(11.0f, 1100));
    assert(z.zone() == Zone::Camera);
    // 12-14 inch hysteresis keeps camera open even with small echoes jitter.
    assert(!z.update(13.0f, 1300));
    assert(z.zone() == Zone::Camera);
    // Single bad echo cannot force exit.
    assert(!z.update(-1.0f, 1500));
    assert(z.zone() == Zone::Camera);
    assert(!z.update(15.0f, 1650));
    assert(z.update(15.0f, 2100));
    assert(z.zone() == Zone::Welcome);
    assert(!z.update(27.0f, 2400));
    assert(z.update(27.0f, 2800));
    assert(z.zone() == Zone::Standby);
    assert(!z.update(9.0f, 2900));
    assert(z.update(9.0f, 3400));
    assert(z.zone() == Zone::Camera);
    assert(!z.update(-1.0f, 3600));
    assert(!z.update(-1.0f, 4900));
    assert(!z.update(-1.0f, 5100));
    assert(z.update(-1.0f, 5500));
    assert(z.zone() == Zone::Standby);

    using namespace smartsort;
    AutoScan scan;
    const Detection person{0, 0.99f, .3f, .2f, .6f, .9f};
    const Detection bottle{39, 0.85f, .3f, .2f, .6f, .9f};
    assert(!scan.update({}, 0));
    assert(!scan.update({person}, 100));
    assert(!scan.update({bottle}, 200));
    assert(!scan.update({bottle, person}, 450));
    assert(!scan.update({}, 600)); // dropped item resets the confirmation
    assert(!scan.update({bottle}, 700));
    const Detection misnamed{41, 0.80f, .3f, .2f, .6f, .9f};
    assert(!scan.update({misnamed}, 1199)); // class changes; same physical item
    assert(scan.update({bottle}, 1200));
    assert(!scan.update({bottle}, 1700));
    scan.capture_finished();
    assert(!scan.update({bottle}, 2000));
    assert(!scan.update({}, 2100));
    assert(!scan.update({}, 3100));
    assert(!scan.update({bottle}, 3300)); // new confirmation, not immediate capture
    puts("PASS: 24/12in proximity zones and real-detection 500ms scan gate");
    return 0;
}
