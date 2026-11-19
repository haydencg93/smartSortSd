#include "app/auto_scan.h"
#include <cassert>
#include <iostream>

using smartsort::AutoScan;
using smartsort::Detection;
using smartsort::ScanState;

static Detection bottle(float x1 = .35f, float x2 = .65f) {
    return {39, .92f, x1, .25f, x2, .85f}; // COCO bottle = 39
}

int main() {
    AutoScan gate;
    assert(!gate.update({{0, .99f, .2f, .1f, .8f, .9f}}, 0)); // person alone
    assert(gate.state() == ScanState::Searching);
    assert(!gate.update({bottle()}, 100));
    assert(!gate.update({bottle()}, 599));
    assert(gate.update({bottle()}, 600));
    assert(!gate.update({bottle()}, 700)); // only once
    gate.capture_finished();
    assert(gate.state() == ScanState::WaitForRemoval);
    assert(!gate.update({bottle()}, 1000)); // same object still there
    assert(!gate.update({}, 1100));
    assert(!gate.update({}, 2099));
    assert(!gate.update({}, 2100));
    assert(gate.state() == ScanState::Searching);
    assert(!gate.update({bottle()}, 2200));
    assert(!gate.update({bottle(.75f, 1.0f)}, 2250)); // different location, fails match
    assert(!gate.update({}, 2300)); // resets
    assert(!gate.update({bottle()}, 2400));
    assert(!gate.update({bottle()}, 2800));
    assert(gate.update({bottle()}, 2900));
    std::cout << "PASS: person filter, 500ms confirmation, duplicate suppression, removal rearm\n";
}
