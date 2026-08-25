"""The grasp action id has to survive the hop that validates it.

WHY THIS HAS ITS OWN FILE. The bug it guards cost a complete, otherwise perfect
hardware reach. On 2026-07-30 the cuRobo tool planned in 0.14 s, passed all four
gates (39.5 mm clearance), streamed 141 waypoints, settled 1.6 deg from the
planned configuration — and then sat for 20 s waiting for a grasp verdict that
could never arrive, because the command had been thrown away one process
earlier.

The mechanism is worth stating precisely, because nothing in the commander's own
output can reveal it. real_env's EEServiceClient.send_action validates every id
with `int(action_id)` and, on a ValueError, logs a warning INTO REAL_ENV'S
TERMINAL and returns without publishing. The commander sees a gripper that never
answered. The tool was sending `uuid.uuid4().hex[:12]`.

So these tests do not check "the id looks like an int". They push the id the tool
actually generates through the REAL validation function and assert a packet came
out the other side. That is the only version of this test that would have failed
before the fix — and a test that cannot fail is not evidence.

Zeroing is the reason this survived to the last step of a mission: it passes no
id at all, takes send_action's auto-increment branch, and works perfectly. A
run's gripper can therefore calibrate flawlessly and still be unable to grasp.
"""

from __future__ import annotations

import unittest

# The failure mode below is the dangerous one: an import error here SKIPS every
# test in this file instead of failing it. Until 2026-08-22 a plain import was
# not enough — the reach tool puts <visual_servoing> on sys.path, and its own
# `common/` package shadowed the service's `from common.publisher import ...`
# under discovery, so this file front-loaded control/ and purged
# sys.modules["common"]. The control-side package is `ipc` now, nothing shadows
# it, and the hack is gone; watch the skip count whenever the suite changes.
try:
    import humanoid_end_effector_service as ee_service
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool / EE service unavailable ({type(exc).__name__}: {exc})"

# The id from the hardware log, verbatim. Not a fresh uuid4: hex[:12] is
# all-digits about once in 220 draws, so generating one here would make the
# regression test pass at random.
DROPPED_ID = "dd3a8235c87b"


class _SpyPub:
    def __init__(self):
        self.sent = []

    def publish(self, packet):
        self.sent.append(packet)


def _client(spy):
    """An EEServiceClient with the real send_action and no sockets.

    __init__ binds /tmp/ee_request.sock, which would collide with a running
    service and put a side effect in the suite; __new__ plus the three fields
    send_action touches exercises the same code with none of that.
    """
    c = ee_service.EEServiceClient.__new__(ee_service.EEServiceClient)
    c.alive = True
    c._action_id = -1
    c._pub = spy
    return c


@unittest.skipIf(R is None, _SKIP)
class GraspActionIdTests(unittest.TestCase):

    def test_the_generated_id_reaches_the_service(self):
        """THE test. Anything the tool sends must come out of send_action as a
        published packet, or the grasp is a 20 s wait for nothing."""
        spy = _SpyPub()
        got = _client(spy).send_action("left", "hand_grab",
                                       action_id=R.new_grasp_action_id())
        self.assertIsNotNone(got, "the command was DROPPED before the service")
        self.assertEqual(len(spy.sent), 1)
        act = spy.sent[0]["ee_action"]
        self.assertEqual(act["command"], "hand_grab")
        self.assertEqual(act["side"], "left")

    def test_the_id_that_was_forwarded_is_the_id_we_asked_for(self):
        """The verdict is correlated by id — the tool waits for
        grasp_action_id == its own. An id silently rewritten in transit would
        strand the same wait, just for a different reason."""
        spy = _SpyPub()
        mine = R.new_grasp_action_id()
        _client(spy).send_action("left", "hand_grab", action_id=mine)
        self.assertEqual(spy.sent[0]["ee_action"]["action_id"], mine)

    def test_a_uuid_hex_id_is_dropped_on_the_floor(self):
        """The regression, with the exact string the hardware run sent. This is
        what "no id-matched grasp verdict in 20s" looked like from inside."""
        spy = _SpyPub()
        got = _client(spy).send_action("left", "hand_grab",
                                       action_id=DROPPED_ID)
        self.assertIsNone(got)
        self.assertEqual(spy.sent, [], "a non-integer id must not be published")

    def test_consecutive_grasps_get_increasing_ids(self):
        """Two grasps in one session must not share an id: the service dedups on
        last_processed_action_id, so a repeat would be swallowed as a duplicate
        of the grasp that already happened."""
        a = R.new_grasp_action_id()
        b = R.new_grasp_action_id()
        self.assertGreater(b, a)

    def test_the_id_is_non_negative(self):
        """send_action rejects negatives on a separate branch, with its own
        warning, in the same silent way."""
        self.assertGreaterEqual(R.new_grasp_action_id(), 0)


if __name__ == "__main__":
    unittest.main()
