from __future__ import annotations

import unittest
from unittest.mock import patch

import humanoid_end_effector_service as ee_service


class FakeHand:
    """One-servo, in-memory hand used only by the grasp algorithm tests."""

    def __init__(self, position: int = 0, load=20, *, follows=True,
                 fail_position=False, fail_write=False):
        self.position = position
        self.load = load
        self.follows = follows
        self.fail_position = fail_position
        self.fail_write = fail_write
        self.commands = []

    def get_positions(self, ids):
        if self.fail_position:
            raise RuntimeError("position bus failure")
        return [self.position for _ in ids]

    def set_positions(self, ids, positions, speed, acc, torque):
        if self.fail_write:
            raise RuntimeError("write bus failure")
        target = int(positions[0])
        self.commands.append((tuple(ids), target, speed, acc, torque))
        if self.follows:
            self.position = target
        return True

    def get_load(self, _servo_id):
        if callable(self.load):
            return self.load()
        return self.load


def make_service(hand: FakeHand, *, close=700):
    """Build just enough service state without sockets or servo hardware."""
    service = object.__new__(ee_service.EndEffectorService)
    service.hands = {"left": hand, "right": None}
    service.healthy = {"left": True, "right": False}
    service.servo_ids = {"left": [5], "right": [0]}
    service.hand_open = {"left": 0, "right": 0}
    service.hand_close = {"left": close, "right": close}
    return service


class LoadDecodeTests(unittest.TestCase):
    def test_decodes_new_signed_and_legacy_bit10_values(self):
        cases = {
            0: 0,
            321: 321,
            -321: -321,
            (1 << 10) | 321: -321,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(
                    ee_service.decode_gripper_load(raw), expected)

    def test_missing_boolean_and_impossible_loads_are_indeterminate(self):
        for raw in (None, True, False, 1001, (1 << 10) | 1001, 0x8000):
            with self.subTest(raw=raw):
                self.assertIsNone(ee_service.decode_gripper_load(raw))


class ShutdownFailClosedTests(unittest.TestCase):
    def test_false_or_unknown_without_hold_target_sends_no_position(self):
        for verdict in (False, None):
            with self.subTest(verdict=verdict):
                hand = FakeHand()
                service = make_service(hand)
                service.grasp_hold_pos = {"left": None, "right": None}
                service.last_grasp_detected = {
                    "left": verdict, "right": None}

                service._preserve_grippers_on_shutdown()

                self.assertEqual(hand.commands, [])

    def test_known_hold_target_is_reasserted_and_never_opened(self):
        hand = FakeHand()
        service = make_service(hand)
        service.hand_open["left"] = -3000
        service.grasp_hold_pos = {"left": 77, "right": None}

        with patch.object(ee_service.time, "sleep", return_value=None):
            service._preserve_grippers_on_shutdown()

        self.assertEqual(len(hand.commands), 1)
        _ids, target, _speed, _acc, torque = hand.commands[0]
        self.assertEqual(target, 77)
        self.assertNotEqual(target, service.hand_open["left"])
        self.assertEqual(torque, ee_service.SHUTDOWN_GRIP_TORQUE)


class GraspCloseTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, hand, *, close=700, timeout=None):
        service = make_service(hand, close=close)
        patches = [patch.object(ee_service.time, "sleep", return_value=None)]
        if timeout is not None:
            patches.append(patch.object(
                ee_service, "GRASP_TIMEOUT_S", timeout))
        for active_patch in patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)
        return await service.run_grasp_close("left")

    async def test_valid_low_load_at_measured_close_is_confident_empty(self):
        result, final_position = await self._run(
            FakeHand(position=0, load=20), close=140)

        self.assertIs(result, False)
        self.assertEqual(final_position, 140)

    async def test_sustained_load_contact_is_grasp(self):
        result, final_position = await self._run(
            FakeHand(position=0, load=700), close=700)

        self.assertIs(result, True)
        self.assertGreater(final_position, 0)

    async def test_measured_stall_with_low_load_is_grasp(self):
        hand = FakeHand(position=0, load=20, follows=False)

        result, _final_position = await self._run(hand, close=700)

        self.assertIs(result, True)
        self.assertGreaterEqual(
            len(hand.commands), ee_service.GRASP_STALL_CONSECUTIVE_REQUIRED)

    async def test_initial_position_read_failure_is_indeterminate(self):
        result, _final_position = await self._run(
            FakeHand(fail_position=True))

        self.assertIsNone(result)

    async def test_write_failure_is_indeterminate(self):
        result, _final_position = await self._run(
            FakeHand(fail_write=True))

        self.assertIsNone(result)

    async def test_missing_load_is_indeterminate_not_empty(self):
        result, _final_position = await self._run(
            FakeHand(load=None), close=140)

        self.assertIsNone(result)

    async def test_timeout_is_indeterminate_not_empty(self):
        result, _final_position = await self._run(
            FakeHand(load=20), timeout=-1.0)

        self.assertIsNone(result)

    async def test_indeterminate_report_preserves_hold_and_correlates_id(self):
        service = make_service(FakeHand())
        service.grasp_hold_pos = {"left": None, "right": None}
        service.last_grasp_detected = {"left": None, "right": None}
        service.last_grasp_action_id = {"left": None, "right": None}
        service.grasping_in_progress = {"left": True, "right": False}
        reports = []

        async def indeterminate(_side):
            return None, 77

        service.run_grasp_close = indeterminate
        service.emit_status = reports.append

        await service._grasp_and_report("left", 123)

        self.assertEqual(service.grasp_hold_pos["left"], 77)
        self.assertIsNone(service.last_grasp_detected["left"])
        self.assertEqual(service.last_grasp_action_id["left"], 123)
        self.assertFalse(service.grasping_in_progress["left"])
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].state, "failed")
        self.assertEqual(reports[0].error_code, "GRASP_INDETERMINATE")
        self.assertIsNone(reports[0].grasp_detected)


if __name__ == "__main__":
    unittest.main()
