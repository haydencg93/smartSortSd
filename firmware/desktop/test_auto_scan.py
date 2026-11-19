import unittest

from auto_scan import AutoScan, Detection, State


BOTTLE = Detection(39, .90, .36, .20, .64, .80)
PERSON = Detection(0, .99, .10, .10, .90, .90)
EDGE_OBJECT = Detection(39, .9, .01, .01, .15, .30)


class GateTests(unittest.TestCase):
    def test_person_does_not_trigger(self):
        gate = AutoScan()
        for t in (0, 500, 1500, 2000):
            self.assertFalse(gate.update([PERSON], t))
        self.assertEqual(gate.state, State.SEARCHING)

    def test_waits_five_hundred_milliseconds(self):
        gate = AutoScan()
        self.assertFalse(gate.update([BOTTLE], 0))
        self.assertFalse(gate.update([BOTTLE], 499))
        self.assertTrue(gate.update([BOTTLE], 500))
        self.assertFalse(gate.update([BOTTLE], 501))

    def test_person_with_bottle_is_valid(self):
        gate = AutoScan()
        self.assertFalse(gate.update([PERSON, BOTTLE], 100))
        self.assertTrue(gate.update([PERSON, BOTTLE], 600))

    def test_out_of_zone(self):
        gate = AutoScan()
        self.assertFalse(gate.update([EDGE_OBJECT], 0))
        self.assertFalse(gate.update([EDGE_OBJECT], 1000))

    def test_requires_removal_before_rescan(self):
        gate = AutoScan()
        gate.update([BOTTLE], 0)
        self.assertTrue(gate.update([BOTTLE], 500))
        gate.capture_finished()
        for t in (1000, 3000, 5000):
            self.assertFalse(gate.update([BOTTLE], t))
        gate.update([], 6000)
        gate.update([], 7000)
        self.assertEqual(gate.state, State.SEARCHING)
        gate.update([BOTTLE], 7100)
        self.assertTrue(gate.update([BOTTLE], 7600))

    def test_loss_of_detection_resets_timer(self):
        gate = AutoScan()
        gate.update([BOTTLE], 0)
        gate.update([], 400)
        self.assertFalse(gate.update([BOTTLE], 500))
        self.assertFalse(gate.update([BOTTLE], 999))
        self.assertTrue(gate.update([BOTTLE], 1000))


if __name__ == "__main__":
    unittest.main()
