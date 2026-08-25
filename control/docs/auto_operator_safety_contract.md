# Auto-Operator — Safety Contract (mechanism inventory)

Generated 2026-07-18 from a 14-lens extraction workflow over `humanoid_auto_operator.py` (snapshot `auto_operator_legacy_20260718.py`, 2853 lines). Line numbers refer to that snapshot. Every mechanism keeps its ID through the refactor; the **New location** field is filled in as extraction stages land (see `auto_operator_refactor_plan.md`). Incident context: `auto_operator_incidents.md`.

| area | mechanisms |
|---|---|
| Navigation | 16 |
| Detection & target tracks | 19 |
| Gaze & cameras | 14 |
| Startup & posture | 12 |
| Workspace envelope & same-side | 23 |
| Sector staging (classic) | 31 |
| Side-home profile | 15 |
| Command-wire arbitration | 21 |
| Held-arm dynamics | 25 |
| Dual-parallel | 19 |
| Visual servo | 19 |
| Gripper / EE chain | 22 |
| Shutdown & silence | 10 |
| Gap sweep (uncategorized) | 15 |
| **total** | **261** |


## Navigation

### SAFE-NAV-001 · Hard nav output clamp (|vx|<=0.3, vy==0, |wz|<=0.2)

- **Location (legacy):** 117-119, 1209, 2708-2711 (also 52 header contract)
- **Protects against:** Overspeed/lateral/spin command destabilizing the walking policy or driving the base into the table/cube; below ~0.1 m/s the policy steps in place (comment 117) so the clamp band is also a validity band
- **Trigger:** Every tick: whatever vx_des/wz_des the phase handlers produced, including bugs upstream
- **Response:** Final unconditional np.clip to +/-CRUISE_VX and +/-WZ_MAX at 2709-2710 BEFORE the ramp; steering already pre-clamped at 1209 (wz = clip(K_STEER*err, +/-WZ_MAX)); vy is never computed at all — structurally zero
- **Constants:** CRUISE_VX=0.3 m/s, WZ_MAX=0.2 rad/s, K_STEER=0.6; vy is a hardcoded 0.0 literal in every published triple (2549, 2551, 2711)
- **State:** self._vx, self._wz
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Header safety contract (lines 52-55): outputs hard-clamped as the operator-side half of the robot-side failsafe pact
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-002 · ACCEL per-channel ramp on published nav

- **Location (legacy):** 120, 1212-1214, 2546-2547, 2709-2710
- **Protects against:** Step discontinuity in commanded velocity (leg start, arrival stop, SEARCH rewind, startup fail-closed zeroing) jerking the walking policy
- **Trigger:** Any change in desired vx/wz, including the hard zeroings — _ramp is the only writer of _vx/_wz
- **Response:** _ramp(current, desired) moves at most ACCEL_LIMIT*DT per tick; even fail-closed zeroing at 2546-2547 decelerates through the ramp rather than snapping
- **Constants:** ACCEL_LIMIT=3.0 m/s^2, DT=1/OP_RATE_HZ=0.05 s -> max step 0.15 per tick per channel; applied to BOTH vx and wz (same numeric limit as rad/s^2)
- **State:** self._vx, self._wz (ramped outputs, persist across ticks)
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-003 · DET_STALE_S blind-stop (stale detection zeroes walk)

- **Location (legacy):** 142, 1158-1160
- **Protects against:** Walking on a base-frame cube position that is geometrically rotten — base-frame positions rot as the robot moves, so an old fix steers the robot to a phantom location
- **Trigger:** In GO, active cube not fresh(now, DET_STALE_S) — no sighting within 0.5 s
- **Response:** _legs_and_gate returns (0,0,False): stand still (through the ramp), keep tracking, stay in GO; walking resumes only on a fresh sighting
- **Constants:** DET_STALE_S=0.5 s
- **State:** CubeTrack.last_seen, CubeTrack.pos
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (rig tests place cubes inside the arrival floor; walking is never exercised)
- **Incident:** Header line 52-53: 'stale or frozen detections zero the walk command'
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-004 · Frozen-monitor detection zeroing (main loop data_id watch)

- **Location (legacy):** 2796-2813
- **Protects against:** Monitor process frozen/dead while NNGSubscriber.data retains the last dict forever: re-ingesting it would keep re-stamping cube.last_seen and the robot would walk forever on dead perception (verbatim comment 2811-2812)
- **Trigger:** det_sub.data_id unchanged for >0.5 s (no new packet arrival, tracked via last_det_id/last_det_arrival)
- **Response:** det_targets = {} — tick() sees no detections at all, so DET_STALE_S then zeroes the walk and DET_LOST_RESCAN_S rewinds GO to SEARCH; fresh-vs-frozen is judged on arrival TIME, not packet content
- **Constants:** literal 0.5 s freshness window on det_sub packet ARRIVAL (unnamed; numerically equals DET_STALE_S but hardcoded separately)
- **State:** last_det_id, last_det_arrival, det_targets (main() locals)
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (tests drive AutoOperator.tick directly; main() loop untested)
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-005 · DET_LOST_RESCAN rewind GO->SEARCH

- **Location (legacy):** 143-152, 1342-1349, 688-716 (scan fallback), 453-457 (re-confirm)
- **Protects against:** Pursuing a cube unseen for seconds (moved/occluded/mis-detected) to a stale location; or a camera staring at a stale point never rediscovering its cube (comment 691-693)
- **Trigger:** In GO, (now - active.last_seen) > 3.0 s
- **Response:** Phase -> SEARCH with nav (0,0); camera claim KEPT, only the sighting must be re-confirmed (0.3 s streak filters one-frame mid-sweep flukes); _eye uses the same DET_LOST_RESCAN_S horizon (696-701) so the claiming camera falls back to scanning — that fallback is what makes SEARCH re-entry actually search
- **Constants:** DET_LOST_RESCAN_S=3.0 s; DET_UNCLAIM_S=6.0 s deliberately > DET_LOST_RESCAN_S (claiming camera gets a rescan window before the binding is up for grabs); re-entry needs confirmed() = fresh within 1.0 s AND 0.3 s uninterrupted streak
- **State:** CubeTrack.last_seen, first_seen, camera_port; self.phase
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-006 · Leg direction chosen ONCE per leg (forward/backward)

- **Location (legacy):** 541, 1205-1206, 1210, 1337-1339
- **Protects against:** Mid-leg forward/backward flapping when the bearing hovers near +/-pi/2 — alternating vx sign each tick as detection noise crosses the boundary; also avoids turn-in-place maneuvers toward rear cubes (walks backward instead)
- **Trigger:** First _legs_and_gate tick of a leg (_leg_dir is None)
- **Response:** Direction latched for the whole leg ('sim semantics' comment 1205); reset to None only at the SEARCH->GO transition (1339) when a new leg starts
- **Constants:** _leg_dir = +1.0 if |atan2(y,x)| <= pi/2 else -1.0; vx = _leg_dir * CRUISE_VX
- **State:** self._leg_dir (None between legs)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-007 · Backward-walk steering error wrap

- **Location (legacy):** 1207-1209, 676-678
- **Protects against:** For a backward leg the raw bearing sits near +/-pi; without the wrap the +pi shift jumps the error by 2pi across the cut, flipping wz sign and steering the tail the long way around (oscillation)
- **Trigger:** Every GO tick of a backward leg (_leg_dir < 0)
- **Response:** Error re-wrapped to (-pi, pi] so the tail-toward-target steering is continuous across the cut, then clamped to +/-WZ_MAX
- **Constants:** err = bearing (fwd) or _wrap(bearing + pi) (bwd); _wrap = (a+pi)%(2pi)-pi; wz = clip(K_STEER=0.6 * err, +/-WZ_MAX=0.2)
- **State:** self._leg_dir
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-008 · GEOMETRIC_FLOOR unconditional arrival stop

- **Location (legacy):** 133, 1161-1165
- **Protects against:** Walking into/over the cube (body-cube collision) if the learned gate never fires, is disabled (--no-use-gate), or is wrong
- **Trigger:** dist_xy <= 0.28 m to the active cube, checked BEFORE the learned gate each fresh GO tick
- **Response:** arrived=True immediately — no debounce needed (deterministic geometry); nav goes (0,0); the learned gate can only stop EARLIER (farther), never later than the floor
- **Constants:** GEOMETRIC_FLOOR_M=0.28 m (sim constant), on planar dist_xy = hypot(x,y)
- **State:** self._arrived
- **Flags:** fail-closed, timing-insensitive
- **Tests:** implicit only — every rig test (e.g. t_grasp_carry_side, t_retry_then_fail) places cubes at r<0.28 so this arrival path drives the phase machine; no test asserts the stop itself
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-009 · Arrival LATCH (no stop-start on the gate boundary)

- **Location (legacy):** 536, 1151-1157, 1198-1203, 1339
- **Protects against:** The gate scorer samples fresh random candidates each call, so re-deriving the verdict every tick would chatter right at the envelope edge — robot stop-starting on the boundary
- **Trigger:** Once _legs_and_gate sets _arrived=True (floor or debounced gate)
- **Response:** Every subsequent GO tick short-circuits to (0,0,True); 'once latched it never un-latches'; cleared only at the next SEARCH->GO leg start (1339)
- **Constants:** _arrived latch; sim ReachabilityGate semantics quoted at 1151-1154
- **State:** self._arrived, self._gate_hits (both reset with _leg_dir at 1339)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-010 · Learned-gate arrival + GATE_CONSECUTIVE debounce

- **Location (legacy):** 123, 130-132, 1166-1175
- **Protects against:** One lucky stochastic gate sample stopping the walk while the cube is still out of reach (or a lucky miss resetting a legitimate stop) — the scorer draws fresh random candidates per call so single verdicts chatter
- **Trigger:** GO tick beyond the geometric floor with a learned gate loaded: fresh _gate_eval sample above/below threshold
- **Response:** _gate_hits increments only on FRESH above-threshold samples, hard-resets to 0 on a fresh miss; arrival requires _gate_hits >= 5; cached (fresh=False) evals never step the counter
- **Constants:** GATE_CONSECUTIVE=5 consecutive above-threshold FRESH samples; score > gate.SCORE_THRESHOLD; spans ~0.5 s wall-clock under the 0.1 s eval throttle (was 0.25 s unthrottled)
- **State:** self._gate_hits
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_gate_throttle covers fresh-only counter stepping via _dyn_reachable/_reachable_now (FakeGate); the GO arrival path itself: none (rigs run gate=None)
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-011 · _gate_eval inference throttle + quantized cache

- **Location (legacy):** 124-132, 1780-1803
- **Protects against:** gate.score() is CPU torch at 17-37 ms/call (measured 07-17); unthrottled SEARCH scored 2-3 pending cubes EVERY 20 Hz tick (35-75 ms) and blew the 50 ms tick budget — the gaze publish jitter excited the gimbal trim loop into the violent periodic pre-reach camera shudder; also: stepping debounce counters on cached repeats would let one lucky sample latch
- **Trigger:** Any _gate_eval call (GO arrival, _dyn_reachable, _reachable_now) within 0.1 s of a cached verdict for the same 2 cm cell
- **Response:** Return the cached (score, best, fresh=False); all debounce counters (GATE_CONSECUTIVE, in_hits/out_hits) step ONLY on fresh=True so they integrate independent stochastic draws
- **Constants:** GATE_EVAL_PERIOD_S=0.1 s per target; cache key = pos quantized to 0.02 m per axis; cache cleared when len > 128; returns (score, best_arm, fresh) with fresh=False on cache hits
- **State:** self._gate_cache {quantized pos -> (score, best, t)}
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_gate_throttle: '20 back-to-back evals -> 1 network call', 'refreshes after the throttle period', 'cached evals do NOT advance the debounce counters'
- **Incident:** 07-17 pre-reach camera shudder: tick-budget overrun -> gaze jitter -> hot gimbal trim integrator (companion fix TRIM_GAIN 0.3->0.05, lines 98-104)
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-012 · Reach-latch waits for the base to physically stop

- **Location (legacy):** 1342-1353 (esp. 1351)
- **Protects against:** Freezing the base-frame reach target while the ramp is still decelerating the base — the latched median would be measured from a moving frame and the arm would reach at a smeared/offset point
- **Trigger:** GO tick with arrived=True but |ramped _vx| >= 0.02
- **Response:** Stay in GO publishing the decelerating ramp; _latch_reach_target(now) fires only once the commanded base velocity has ramped below 0.02 m/s
- **Constants:** abs(self._vx) < 0.02 m/s required in addition to arrived
- **State:** self._vx, self._arrived
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (rigs never build up _vx — cubes start inside the floor)
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-013 · --no-walk hard-zero at the wire (standing missions)

- **Location (legacy):** 2739-2743, 2792-2794, 2825-2828
- **Protects against:** A standing (balance-in-place) mission stepping — the robot must never step; docstring: cubes must sit inside the arrival gate's envelope or GO waits forever
- **Trigger:** args.walk False and not gaze_only, every loop iteration
- **Response:** Every published nav_cmd replaced with zeros at the last possible point (state machine unchanged and still exercised — 'identical to the hanging-bench behavior where legs ignore nav anyway'); continuous zero publish keeps the robot-side nav 1 s silence failsafe armed
- **Constants:** Args.walk (default True); nav = [0.0, 0.0, 0.0] substituted at 2827 AFTER tick(), immediately before nav_pub.publish
- **State:** args.walk
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (main() untested)
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-014 · Startup fail-closed: nav ramped to zero until posture check PASSES

- **Location (legacy):** 2539-2551 (guard), 2479-2483 (NaN fail-open note), 496-503
- **Protects against:** Operator relaunched against a wrong --robot-ip, before the arm homed, or with no telemetry, walking or blind-reaching from an unknown state
- **Trigger:** Any full-mission tick before the startup posture check has actually passed (it cannot pass without fresh telemetry)
- **Response:** Publish nav ramped to zero + gaze only (once seeded); NO arm packet — the robot's arm-silence failsafe holds/crawls; continues publishing so robot-side silence failsafes stay armed while nothing moves
- **Constants:** _startup_posture_ok latch; needs finite jpos[13:27] telemetry to even run; NaN posture would PASS a bare 'NaN > x' comparison, hence the isfinite gate (fail-open class, comment 2481-2483)
- **State:** self._startup_posture_ok, self._gaze_seeded, self._vx, self._wz
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_tilt_gate covers only the SystemExit tilt-refusal branch of the same startup guard; the nav-zero fail-closed path itself: none
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-015 · Continuous publication contract + zero-nav on exit

- **Location (legacy):** 85 (OP_RATE_HZ), 52-54 (contract), 2803-2844 (loop pacing), 2845-2848 (finally)
- **Protects against:** Publication gaps disarm the design: the robot-side failsafes are armed BY continuous publishing (a silent operator looks identical to a crashed one); on exit, without the final zero the robot would keep walking on the last nonzero nav_cmd until its own 1 s nav timeout
- **Trigger:** Every loop iteration; the finally block on ANY exit (Ctrl+C, DONE, crash)
- **Response:** nav published every DT tick whenever tick() returns non-None; finally: explicit zero nav_cmd + 0.1 s flush before socket teardown; gaze-only mode never binds/publishes 9873 at all (nav=None, 2781, 2527-2534) so the robot's nav silence failsafe governs
- **Constants:** OP_RATE_HZ=20.0 ('> all robot-side timeout rates'); robot-side silence failsafes: nav 1 s, arm 0.5 s, gaze 2 s; loop paced sleep(max(0, DT - elapsed)); shutdown publishes {nav_cmd:[0,0,0]} then sleeps 0.1 s
- **State:** nav_pub, t0 loop clock
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (main() untested)
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop

### SAFE-NAV-016 · Gamepad human-override assumption (robot-side)

- **Location (legacy):** 55
- **Protects against:** Operator fighting a human takeover during an incident; the design assumes the deflection pause exists in humanoid_real_env and the operator does nothing to detect or resist it
- **Trigger:** Any stick deflection on the robot-side gamepad
- **Response:** None needed operator-side (documented assumption): nav keeps publishing but the robot ignores it while deflected; a refactor must preserve BOTH halves — robot-side pause and operator-side tolerance of unacknowledged nav
- **Constants:** none operator-side — 'A human with the gamepad always wins (stick deflection pauses nav robot-side)'
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (mechanism lives robot-side; assumption untestable in this harness)
- **New location (S6):** auto_operator/mission.py (legs_and_gate, tick_go, gate_eval) + operator tick() final clamp/ramp + main() no-walk/publish loop


## Detection & target tracks

### SAFE-DET-001 · Malformed/NaN cube detection rejection (_ingest)

- **Location (legacy):** 754-761
- **Protects against:** A single NaN/malformed pos entering a track poisons the gaze reference and every downstream IK/reach target (fail-open class) — gimbal and arm commanded to garbage
- **Trigger:** Detection payload for a cube key with n_inliers < 1, pos not shape (3,), or any non-finite component
- **Response:** continue — the sighting never touches cube.pos/pos_port/hist/last_seen; track state stays at last-good
- **Constants:** n_inliers >= 1 (line 755); pos.shape == (3,); np.isfinite all-axes
- **State:** cube.pos, cube.pos_port, cube.hist, cube.last_seen (all protected)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (all test detections are well-formed, n_inliers=4)
- **Incident:** 07-15 adversarial audit: a single NaN here used to poison the gaze reference and every downstream target (fail-open class)
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-002 · Sighting-streak reset + confirmed() 0.3 s debounce

- **Location (legacy):** 448-457, 762-763 (consumers: 1300, 1692)
- **Protects against:** Latching a reach leg onto a one-frame fluke position captured mid-gimbal-sweep — arm dispatched at a phantom point
- **Trigger:** Sighting after a >1.0 s gap resets first_seen (new streak); SEARCH (1300) and secondary latch (1692) skip any cube not sighted steadily >= 0.3 s
- **Response:** Cube is not eligible for leg selection / secondary latch until the streak matures
- **Constants:** streak-gap reset threshold = 1.0 s (line 762); confirmed = fresh(now, 1.0) AND (now - first_seen) > 0.3 s
- **State:** cube.first_seen, cube.last_seen
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none directly (implicit in every reach test; t_dual_parallel_sec_first_grab relies on first_seen ordering for primary selection, not the 0.3 s streak)
- **Incident:** Extrinsic is FK of the LATEST telemetry, so a fast-moving gimbal skews the first sighting; once TRACK locks on the pose settles
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-003 · DET_STALE_S blind-stop in GO (+ fresh() horizon semantics)

- **Location (legacy):** 142, 448-451, 1158-1160
- **Protects against:** Robot keeps walking toward a rotted base-frame position (base-frame coordinates rot as the robot moves) — blind collision with the cube/table
- **Trigger:** Active cube unseen > 0.5 s during a GO tick
- **Response:** vx=wz=0 returned immediately (stand still, keep tracking); no arrival evaluation on stale data
- **Constants:** DET_STALE_S = 0.5
- **State:** cube.last_seen, cube.pos
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Comment at 142: 'detection packet older than this -> treat as blind, stop walking'; fresh() docstring: anything older is geometrically meaningless
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-004 · DET_LOST_RESCAN_S lost-target rewind GO->SEARCH (+ _eye scan fallback)

- **Location (legacy):** 143, 696-705, 1347-1349
- **Protects against:** A leg continuing against a lost target, and a camera staring forever at a stale point that would never rediscover its cube (SEARCH re-entry must actually search)
- **Trigger:** Active cube unseen > 3.0 s during GO; per-camera, claimed cube not fresh within 3.0 s
- **Response:** phase = SEARCH with the camera claim KEPT (only the sighting must re-confirm); the claiming camera reverts to scan sweep
- **Constants:** DET_LOST_RESCAN_S = 3.0
- **State:** self.phase, cube.camera_port (deliberately retained)
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Freshness gate rationale in _eye docstring (691-693): falling back to scanning is what makes SEARCH re-entry actually search
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-005 · DET_UNCLAIM_S camera-claim expiry

- **Location (legacy):** 144-152, 748-753
- **Protects against:** Camera-binding deadlock: a claim held forever on a vanished cube starves rediscovery and blocks rebinding to the camera that can actually see it
- **Trigger:** Claimed cube unseen by ANY camera for > 6.0 s
- **Response:** cube.camera_port = None (printed '[lock] ... released'); next sighting from either camera may rebind
- **Constants:** DET_UNCLAIM_S = 6.0 (deliberately > DET_LOST_RESCAN_S = 3.0)
- **State:** cube.camera_port
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Comment 146-152: claiming camera first gets a rescan window; any-camera freshness is the right clock — while the OTHER camera sees the cube the pos stays live (spotter handoff), so releasing earlier would just re-assign the claim back
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-006 · Claim assignment collision handling / over-subscription degrade

- **Location (legacy):** 768-780 (CAM_PORTS at 106)
- **Protects against:** Two cubes bound to one camera (tracking starvation) or a crash when cubes outnumber cameras mid-mission
- **Trigger:** First sighting of an unclaimed cube whose spotting camera is already claimed by another cube
- **Response:** Claim falls to the next free port in CAM_PORTS; if none free, camera_port stays None (cube tracked opportunistically, no dedicated eye) — mission continues
- **Constants:** CAM_PORTS = (5555, 5556) (5555=left, 5556=right)
- **State:** cube.camera_port; claimed_ports set
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (dual tests feed two ports/two cubes but never assert claim state)
- **Incident:** Comment 775-776: 'None = more cubes than cameras: stays unclaimed (no dedicated tracking) until a camera frees up; never crash mid-mission'
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-007 · no_claim rule for carried cubes + drop reset

- **Location (legacy):** 768-770, 1840-1854 (set), 1884 (reset on drop)
- **Protects against:** A cube riding in the gripper re-claiming a camera and starving the OTHER cube's search when only one camera survives
- **Trigger:** Hold transitions to 'carried' -> no_claim=True and camera_port released; confirmed drop (_grasp_dropped) -> no_claim=False ('the cube needs an eye again')
- **Response:** _ingest skips claim assignment for no_claim cubes (position ingest itself continues); drop restores claim eligibility
- **Constants:** cube.no_claim flag (dynamically attached, read via getattr default False)
- **State:** cube.no_claim, cube.camera_port
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_drop_detection (carried->dropped revert exercised), t_grasp_carry_side (carried transition) — no direct claim-state assertion
- **Incident:** 07-16 audit: keeping a carried cube's claim starves the other cube's search when only one camera survives
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-008 · pos_port vs camera_port separation + modal latch port binding

- **Location (legacy):** 438-439, 765-766, 1360-1362, 1393-1394
- **Protects against:** Visual-servo hand-bias b-hat learned from the WRONG camera: common-mode errors only cancel same-camera, so cross-camera mixing drives the closed-loop hand off-target near the cube
- **Trigger:** Every accepted sighting records pos_port; at latch, _reach_port = most-common measurement port across the CUBE_LATCH_WINDOW_S window
- **Response:** Servo chain binds to the measurement port for the whole leg; sightings from other ports are not usable for b-hat
- **Constants:** cube.pos_port (measuring camera) distinct from cube.camera_port (claiming camera); _reach_port = modal port of latch window
- **State:** cube.pos_port, self._reach_port
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (all tests run visual_servo=False)
- **Incident:** Monitor's targets dict is last-camera-wins per key, so the claim port can differ from the port that measured the latched fix (docstring 1360-1362; trust rule at 386-388)
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-009 · hist deque sighting history (maxlen=5)

- **Location (legacy):** 444-446, 766
- **Protects against:** A single garbage frame (1-inlier PnP glitch) at the arrival tick steering the whole leg
- **Trigger:** Every accepted sighting appends; median consumers window it by recency
- **Response:** Bounded history feeding all median-of-window computations (latch, live retrack, secondary latch)
- **Constants:** deque(maxlen=5) of (seen_time, pos, port) tuples
- **State:** cube.hist
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_inflight_retrack (indirect: retrack median must converge to the moved position)
- **Incident:** Comment 445-446: the reach latch medians over these so a single garbage frame at the arrival tick cannot poison a leg
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-010 · Median-of-window reach latch (CUBE_LATCH_WINDOW_S)

- **Location (legacy):** 420, 1355-1364, 1372
- **Protects against:** Latching the reach target off one noisy frame — the arm chases a PnP outlier at hardware speed
- **Trigger:** GO arrival confirmed AND base actually stopped (|vx| < 0.02, line 1351)
- **Response:** Reach target = per-axis median of the recent window, pulled back by the standoff; leg state reset via _reset_leg_state
- **Constants:** CUBE_LATCH_WINDOW_S = 0.7; per-axis np.median over hist entries newer than 0.7 s
- **State:** self._reach_target, self._reach_t0, self._reach_port
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_inflight_retrack ('retrack: latched at the original position' asserts the latched median)
- **Incident:** Docstring 1358-1362: a single 1-inlier PnP glitch at the arrival tick cannot poison the leg
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-011 · Empty-latch-window crash guard

- **Location (legacy):** 1365-1371
- **Protects against:** np.median over an empty stack CRASHES the operator mid-mission — uncontrolled stop with a real arm in flight
- **Trigger:** No hist entry newer than 0.7 s at the latch instant (occluded at arrival)
- **Response:** Print + phase = SEARCH to re-acquire; no latch, no crash
- **Constants:** window built from CUBE_LATCH_WINDOW_S = 0.7 recency
- **State:** self.phase
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** 07-15 audit: occlusion exactly at the arrival tick left the window empty; crash confirmed as a failure class
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-012 · Post-median side_ok + safety-envelope validation at latch

- **Location (legacy):** 1373-1384
- **Protects against:** Cross-body / out-of-envelope IK target committed at latch: the arrival frame alone chose the arm, and a lagged median can disagree — rear-midline cross-back or torso-strike reach
- **Trigger:** Latched median fails the half-space check for _reach_arm or _target_safe
- **Response:** Print which check failed, phase = SEARCH — REACH is never entered with a bad median
- **Constants:** MIDLINE_MARGIN_M = 0.03 (median[1] must clear it on the bound arm's side); _target_safe envelope (TARGET_MIN_RADIUS_M=0.16, TARGET_MAX_RADIUS_M=0.45, TARGET_Z_MIN_M=-0.40, TARGET_Z_MAX_M=0.30)
- **State:** self.phase (rewound), self._reach_target (never set)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-15 audit: rear-midline cross-back / envelope findings — the MEDIAN (what the arm actually chases) must sit clearly in the bound arm's half-space and inside the envelope
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-013 · SEARCH pre-latch eligibility gauntlet (confirmed/midline/envelope loud refusal)

- **Location (legacy):** 1299-1336 (confirmed 1300; midline 1302-1306; envelope 1309-1319)
- **Protects against:** Committing a leg to a phantom or dangerous cube: midline jitter picks the arm from a coin flip (un-audited rear cross-back), and r>0.45 or z>0 'table cubes' are the TORSO-TILT geometry-corruption signature the OOD-fooled learned gate reaches at
- **Trigger:** Per pending cube each SEARCH tick: unconfirmed, |y| <= 0.03, or _target_safe False
- **Response:** Cube excluded from `ready` (never becomes active); envelope refusal announced ONCE per cube via _sector_warned key '{key}:env' with the full diagnostic
- **Constants:** MIDLINE_MARGIN_M = 0.03; TARGET_MIN_RADIUS_M=0.16, TARGET_MAX_RADIUS_M=0.45, TARGET_Z_MIN_M=-0.40, TARGET_Z_MAX_M=0.30
- **State:** self._sector_warned, self.active (only set from ready)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (t_tilt_gate covers the separate projected-gravity startup gate, not this per-cube envelope skip)
- **Incident:** 07-15 audit (midline coin-flip); torso-tilt class observed AGAIN 07-18 14:24 — warning text points at RULE #0 (hang legs straight, torso upright)
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-014 · _target_safe coarse safety envelope (incl. non-finite fail-closed)

- **Location (legacy):** 197-212 (constants+incident), 1102-1115 (implementation)
- **Protects against:** IK has NO self-collision awareness and no reach math looked at z: a cube against the torso or held at face height became a literal IK target — two confirmed hardware-damage findings
- **Trigger:** Any point the arms may be sent to; non-finite input returns False immediately (1111-1112)
- **Response:** Unsafe treated exactly like out-of-reach: not servable, never latched, never followed
- **Constants:** TARGET_MIN_RADIUS_M = 0.16 (trunk keep-out cylinder), TARGET_MAX_RADIUS_M = 0.45 (physical arm end / torso-tilt signature, added 07-18), TARGET_Z_MIN_M = -0.40, TARGET_Z_MAX_M = 0.30 (face-height refused)
- **State:** none (pure predicate, called at every latch/retrack/servable site)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none directly (indirectly load-bearing in every reach test's in-envelope targets)
- **Incident:** 2026-07-15 adversarial audit (trunk keep-out + z-band both absent); TARGET_MAX_RADIUS_M added 07-18 against the hang-too-low torso-tilt class
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-015 · Live in-flight retrack fail-frozen validation suite

- **Location (legacy):** 1597-1628 (fresh 1611; window 1613-1616; side_ok 1618-1619; envelope+sector clamp 1620-1622; via-home bearing 1624-1628)
- **Protects against:** Mid-flight target sliding un-staged across a basin/sector boundary or across the midline — the facet-2 cross-body class (arm swept across the torso); also chasing a big bearing jump instead of routing via home
- **Trigger:** Each Cartesian-leg tick with dynamic_track: cube stale, empty window, median off-side/unsafe/left-sector, or bearing jump > 45 deg (side-home)
- **Response:** Return None — target stays FROZEN at the last validated point (occlusion => frozen behavior); only a fully validated median may move it
- **Constants:** DYN_LOST_GRACE_S = 1.0; CUBE_LATCH_WINDOW_S = 0.7; MIDLINE_MARGIN_M = 0.03; sector clamp via _sector_left hysteresis; BEARING_JUMP_DEG = 45.0 (side-home via-home rule)
- **State:** self._reach_target / _sec['target'] (only updated on validated medians)
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_inflight_retrack (follow), t_inflight_sector_clamp (refuse across sector), t_via_home_routing (via-home routing end-to-end)
- **Incident:** 07-17: latch is a body-frame point and balance sway drifts it 1-2 cm (grabbed-off-centre class); facet-2 cross-body class for the sector clamp; 07-18 hard requirement for the via-home rule
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-016 · DYN_RETRACK_M retrack deadband

- **Location (legacy):** 183-184, 1629-1632
- **Protects against:** Detection jitter continuously wiggling the arm target during a live reach/hold follow
- **Trigger:** Validated live median within 1 cm of the current target
- **Response:** Return None (no update); the target moves only on >1 cm real displacement
- **Constants:** DYN_RETRACK_M = 0.01
- **State:** current target unchanged
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_inflight_retrack (4.2 cm move is followed; deadband itself not asserted)
- **Incident:** Comment 183-184: 'a deadband so detection jitter cannot wiggle the arm'
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-017 · Secondary-latch validation gauntlet + _sec_hits streak debounce

- **Location (legacy):** 1663-1730 (eligibility 1689-1710; debounce 1711-1721, 1730; state at 529)
- **Protects against:** Committing the second arm on a chattering learned-gate verdict (gate samples fresh random candidates per call), or spawning a secondary in legacy park mode that strands the mission in HOLD forever, or a secondary median off-side/out-of-envelope
- **Trigger:** Primary trackless + no joint stream on the wire; candidate must be confirmed, off-midline on the free arm's side, front sector, _target_safe, _reachable_now, non-empty median window, and post-median side_ok+safe+sector — then 5 consecutive qualifying ticks on the SAME cube_key
- **Response:** _sec latched only after the streak; any tick without a qualifying candidate resets _sec_hits = None (streak broken)
- **Constants:** GATE_CONSECUTIVE = 5 (consecutive qualifying ticks per cube_key); MIDLINE_MARGIN_M = 0.03; CUBE_LATCH_WINDOW_S = 0.7; front-sector-only unless side_home; hold_after_reach required
- **State:** self._sec_hits [cube_key, streak], self._sec
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_front, t_dual_parallel_off, t_dual_parallel_side_gate, t_dual_parallel_sec_first_grab, t_dual_parallel_unseeded_abandon
- **Incident:** 07-17 audit (both the strand-forever guard and the chatter debounce, mirroring the arrival gate's GATE_CONSECUTIVE streak)
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-018 · SERVO_MIN_INLIERS + NaN gating on gripper-tag sightings

- **Location (legacy):** 409, 781-793
- **Protects against:** A garbage single-16mm-tag PnP pose feeding the visual-servo control loop — corrupted b-hat drives the closed-loop hand off-target right next to the cube
- **Trigger:** Gripper-base detection with n_inliers < 2, wrong shape, or non-finite pos
- **Response:** Sighting dropped; _grip[arm] keeps the last accepted measurement (servo's own SERVO_FRESH_S staleness gate then withholds updates downstream)
- **Constants:** SERVO_MIN_INLIERS = 2 (>= 2 tag faces); SERVO_GRIPPER_KEYS = {left: gripper_L_base, right: gripper_R_base}; capture_stamp defaults to now
- **State:** self._grip[arm] = {pos, port, stamp, seen}
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (all tests run visual_servo=False)
- **Incident:** Comment 782-783: single-16mm-tag PnP poses are garbage-prone and this feeds a control loop; gripper bases are measurement-only — never tracked/claimed like cubes, never a gaze target
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)

### SAFE-DET-019 · Idle-camera park-at-neutral (last-camera-wins overwrite guard)

- **Location (legacy):** 706-712 (within _eye 688-716)
- **Protects against:** A sweeping idle camera periodically crosses the arm/cube region and its grazing single-tag detections OVERWRITE the tracking camera's entries in the monitor's one-entry-per-key targets dict (last camera wins), starving the visual servo during the final approach
- **Trigger:** Camera with no fresh claimed cube while no cube is left unclaimed/pending
- **Response:** Camera parks at neutral (0,0) instead of scan-sweeping; scan resumes only while something is genuinely unfound
- **Constants:** yaw=pitch=0.0 park when nothing unclaimed/unreached is pending
- **State:** published gaze slots (GAZE_ORDER), self._scan_phase
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Comment 707-711: 'a sweeping idle camera is not harmless' — detection-stream integrity, not just idle aesthetics
- **New location (S6):** auto_operator/perception.py (ingest/claims) + operator CubeTrack + mission.latch_reach_target (median/validation)


## Gaze & cameras

### SAFE-GAZE-001 · Gaze encoder-seeding gate (no publish until seeded)

- **Location (legacy):** 566-570, 2471-2478, 2531-2532, 2548-2551, 2703-2706
- **Protects against:** Zeros-initialized first gaze packet slamming a parked gimbal to joint 0 in one control step (multi-radian gimbal slam)
- **Trigger:** First tick where telemetry joint_pos exists, len(jpos)>=31, and jpos[27:31] (gimbal encoders) are all finite; until then every tick
- **Response:** Seed self.gaze = jpos[27:31].copy() and set _gaze_seeded=True; while unseeded, ALL THREE publish sites suppress gaze_targets entirely (gaze-only returns (None,{}) at 2531-2532; pre-startup fail-closed path returns no gaze at 2548; mission publish gated at 2704) — real_env holds its last clamped reference, which is benign
- **Constants:** gimbal encoder slots jpos[27:31]; OP_RATE_HZ=20.0
- **State:** _gaze_seeded, self.gaze
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (test rig poweron_jpos() supplies finite zeros at 27:31 so the gate passes silently; nothing asserts the unseeded no-publish behavior)
- **Incident:** 07-15 audit: zeros-init first packet + parked gimbal = multi-radian slam in one control step
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-002 · _slew_gaze sender-side rate limit on published reference

- **Location (legacy):** 109-114, 680-686
- **Protects against:** A scan-to-lock acquisition jump, a +/-2pi branch change, or a first-tick offset reaching the gimbal motors in ONE control step — the robot applies gaze_reference RAW since commit 8045371 (robot-side slew was removed), so the sender is the only remaining bound
- **Trigger:** Every gaze publish (all three sites call self.gaze = self._slew_gaze(self._eye(...)))
- **Response:** Per-tick step toward target clipped to +/-GAZE_SLEW_RATE*DT = 0.075 rad/tick; self.gaze integrates the clipped step so the limit is stateful across ticks
- **Constants:** GAZE_SLEW_RATE=1.5 rad/s (mirrors sim slew cap and old robot-side limit); DT=0.05 s; max_step=0.075 rad
- **State:** self.gaze
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Robot-side slew removed in 8045371 (cam target = gaze_reference + residual, applied raw) — sender MUST bound the step
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-003 · _probe_cam_frame probed optical senses (no hardcoded axis signs)

- **Location (legacy):** 512-517, 575-596
- **Protects against:** A model re-export silently flipping a gimbal axis convention, making the camera drive the WRONG direction (positive-feedback runaway toward the yaw joint limit)
- **Trigger:** Constructor, once per camera port; every _aim_angles/_scan_angles call then multiplies by the probed sense instead of an assumed sign
- **Response:** Finite-difference probe (eps=0.01 rad) of the model FK yields (datum_bearing, yaw_sense=sign d(bearing)/d(yaw), pitch_sense=sign d(elev)/d(pitch)) per camera; 'probed, not assumed — the deploy re-export owns these conventions'
- **Constants:** eps=0.01; CAM_PORTS=(5555,5556); CAM_SITES={5555:cam_left_rgb, 5556:cam_right_rgb}; GAZE_ORDER={5555:(0,1), 5556:(2,3)}; OpenCV +z = optical axis
- **State:** _cam_frame[port] = (datum_bearing, yaw_sense, pitch_sense)
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** The deploy re-export has already flipped yaw once (07-10); left cam datum ~0, rear-mounted right cam datum ~pi
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-004 · Nearest-branch yaw unwrap onto the actual joint

- **Location (legacy):** 602-606, 632-635 (helper _wrap 676-678)
- **Protects against:** The rear-mounted right camera tracking a FRONT cube sits exactly on the +/-pi cut: detection noise would flip the commanded yaw by 2pi and send the gimbal the long way around (cable stress / violent 2pi swing) — off-principal branches are legal because yaw ROM is +/-4.71 rad
- **Trigger:** Every _aim_angles call with valid jpos (len>=31)
- **Response:** yaw = act + _wrap(yaw - act): the command is re-placed on the 2pi branch NEAREST the ACTUAL measured joint, so noise near the cut cannot flip branches
- **Constants:** yaw ROM +/-4.71 rad (comment); wrap window +/-pi
- **State:** none (uses live jpos[27+slot])
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Mirrors sim policies.py:716-735 branch-unwrap logic
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-005 · Two-pass fixed-point FK aim refinement

- **Location (legacy):** 616-631
- **Protects against:** Systematic aim error from gimbal mount axis skew (~4.7 deg measured at combined yaw + up-pitch) plus stale parallax — the camera points off-target at exactly the poses used during reach, degrading detection and the visual-servo geometry
- **Trigger:** Every _aim_angles call (after the closed-form decoupled pan-tilt solution)
- **Response:** Two fixed-point passes re-solve yaw/pitch against the FULL FK evaluated at the COMMANDED pose (j_ref seeded from live jpos, zeros fallback), killing the axis-skew coupling and re-evaluating parallax at the commanded pose
- **Constants:** 2 passes; measured skew ~4.7 deg; direction vectors normalized with 1e-6 floor; asin inputs clamped to [-1,1]
- **State:** none
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Closed form assumes an ideal decoupled pan-tilt; deploy model's mount has slight axis skew
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-006 · Gimbal integral trim: gain + anti-windup clamp + near-lock gate

- **Location (legacy):** 98-105 (constants), 538 (state), 636-642
- **Protects against:** Hot integrator around a lagged plant (gaze low-pass ~100 ms + 5%-torque sluggishness) exciting the violent periodic pre-reach camera shudder; and transit error during an acquisition swing winding the bias to the clamp
- **Trigger:** Trim integrates ONLY near lock: valid jpos AND |yaw_err|<0.3 rad AND |pitch_err|<0.3 rad; frozen otherwise (acquisition/transit)
- **Response:** Per-axis tr += TRIM_GAIN*(commanded - actual), clipped to +/-TRIM_MAX; trim added to the command every tracking tick; per-camera trim state reset never leaks between ports
- **Constants:** TRIM_GAIN=0.05/step (was 0.3 ~6/s at 20 Hz — reverted 07-17; 0.05 still cancels a 0.1 rad steady bias in ~1 s); TRIM_MAX=0.35 rad; near-lock gate 0.3 rad both axes
- **State:** _trim[port] (2-vector per camera)
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_gate_throttle (covers only the tick-jitter root cause, GATE_EVAL_PERIOD_S throttle + fresh-sample debounce); the trim gain/clamp/near-lock gate itself: none
- **Incident:** 07-17: gain 0.3 was marginal and tick jitter from unthrottled gate scoring (see GATE_EVAL_PERIOD_S) excited it into the pre-reach camera shudder; calm once REACH started
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-007 · Serpentine scan geometry + cable-unwind alternation + excursion bound

- **Location (legacy):** 89-97 (spec/constants), 537, 645-674, 715
- **Protects against:** Gimbal cable wind-up: a literal repetition of same-direction 360 deg circles would pass the +/-270 deg yaw joint limit on cycle TWO (cable damage / joint-limit crash); also motion blur if swept faster than the blur budget
- **Trigger:** Camera has no fresh claimed cube while unclaimed/stale pending cubes remain (_eye scan branch)
- **Response:** Lead-in half-turn CCW at 15 deg down (0 to +pi), then a repeating cycle of four FULL 360 deg rows at 30/45/60/15 deg down with direction ALTERNATING per row so each circle unwinds the previous — yaw bounces within exactly +/-180 deg of the camera's own zero (inside the +/-270 deg limit, less excursion than the retired front-wedge sweep); joint = s_rel * probed sense with NO wrapping so the sweep crosses +/-180 deg continuously and stays model-convention-agnostic; turnarounds land at neutral+180 deg
- **Constants:** SCAN_YAW_RATE=0.5 rad/s (blur-budgeted, well under the 1.5 robot slew cap); PANO_ROWS=(30,45,60,15 deg down); lead_t=pi/SCAN_YAW_RATE; row_t=2pi/SCAN_YAW_RATE; excursion +/-(180deg+|datum|) vs +/-270 deg joint limit
- **State:** _scan_phase (advances DT per _eye tick)
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** User spec 07-13; earlier version pinned the rear camera at rest out of cable-ROM caution — the +/-180 deg panoramic scan has since been hardware-validated for both cameras
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-008 · Per-camera claim binding + DET_UNCLAIM_S expiry + carried-cube no_claim

- **Location (legacy):** 143-152 (constants), 436-439, 698-704, 748-780, 1842-1854, 1884
- **Protects against:** Both cameras chasing one cube while the other cube goes unsearched; claim thrash near the freshness boundary; a carried (in-gripper) cube re-claiming an eye and starving the other cube's search when only one camera survives
- **Trigger:** First sighting of an unclaimed cube binds it to the spotting port, else the next free port, else None (more cubes than cameras: stays unclaimed, 'never crash mid-mission'); cube unseen by ANY camera > DET_UNCLAIM_S dissolves the claim; grasp registration sets cube.no_claim=True and releases the camera; release clears no_claim ('needs an eye again')
- **Response:** Each camera aims ONLY at its own claimed cube and ignores the rest; DET_UNCLAIM_S(6.0) deliberately > DET_LOST_RESCAN_S(3.0) so the claiming camera first gets a rescan window before the binding is up for grabs; any-camera freshness is the clock (spotter handoff: while the OTHER camera sees the cube, pos stays live and the claimer aims at it)
- **Constants:** DET_UNCLAIM_S=6.0; DET_LOST_RESCAN_S=3.0; CAM_PORTS=(5555,5556)
- **State:** cube.camera_port (claim), cube.pos_port (measuring camera — distinct because the monitor's targets dict is last-camera-wins), cube.no_claim
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none (tests inject camera_port in detections but never assert claim assignment, expiry, or gaze output)
- **Incident:** 07-16 audit: keeping a carried cube's claim starved the other cube's search when only one camera survives
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-009 · Track-fresh-else-scan fallback on stale sighting

- **Location (legacy):** 688-705 (freshness const 143); upstream monitor-dead clearing 2806-2813
- **Protects against:** A camera staring forever at a stale base-frame point: positions rot as the robot moves, so without the fallback SEARCH re-entry would never actually search (mission stall); upstream, a frozen/dead monitor whose retained dict keeps re-stamping cube.last_seen would defeat every freshness clock
- **Trigger:** Claimed cube's sighting older than DET_LOST_RESCAN_S=3.0 s -> that camera drops out of TRACK; main loop: no new detection packet for >0.5 s -> det_targets={} so last_seen stops being re-stamped
- **Response:** Camera falls back to the serpentine scan (if any unclaimed/stale pending cube exists) instead of aiming at the stale position — 'falling back to scanning is what makes SEARCH re-entry actually search'
- **Constants:** DET_LOST_RESCAN_S=3.0; monitor-dead window 0.5 s (hardcoded, main loop line 2810)
- **State:** cube.last_seen (fresh()), _scan_phase
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Monitor frozen/dead comment: retaining the last dict would keep re-stamping cube.last_seen and the robot would walk forever on dead perception
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-010 · Idle-camera neutral parking (last-camera-wins overwrite prevention)

- **Location (legacy):** 706-712
- **Protects against:** A sweeping idle camera periodically crosses the arm/cube region and its grazing single-tag detections OVERWRITE the tracking camera's entries in the monitor's one-entry-per-key targets dict (last camera wins) — starving the visual servo mid-reach
- **Trigger:** Camera has no fresh claimed cube AND nothing is left to search for (no unclaimed/stale pending cube)
- **Response:** PARK at neutral (yaw=0.0, pitch=0.0) instead of continuing the serpentine sweep — 'a sweeping idle camera is not harmless'
- **Constants:** park pose (0.0, 0.0)
- **State:** none
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Comment documents the servo-starvation mechanism via the monitor's one-entry-per-key dict
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-011 · Gaze-only mode restrictions (no walk, no reach, no 9873)

- **Location (legacy):** 37-45, 50, 519, 2527-2534, 2724, 2766, 2772-2781, 2825-2831
- **Protects against:** A gimbal bring-up/validation session accidentally commanding walking or arm motion on the real robot
- **Trigger:** --gaze-only flag
- **Response:** tick returns nav=None (main() never even constructs the 9873 publisher: nav_pub=None) and a packet carrying ONLY gaze_targets; grasp, dual_parallel, and side_home are force-ANDed off in the constructor call; the reachability gate is never loaded ('gaze-only never walks'); before encoder seeding it returns (None, {}) — publish nothing; gaze itself runs the SAME per-camera claim/track/scan logic as the full mission
- **Constants:** gaze_only flag; nav_pub=None
- **State:** self.gaze_only
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Mode exists to bring up / validate the gimbal on the real robot BEFORE enabling the legs, and to test multi-cube camera locking
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-012 · Pre-startup fail-closed publish: gaze-only output until posture check passes

- **Location (legacy):** 2539-2551
- **Protects against:** An operator relaunched against a wrong --robot-ip, or before the arm homed, driving a blind Cartesian reach; meanwhile the gimbal must not go dark (robot-side gaze silence failsafe) or stop searching
- **Trigger:** Full mission tick with _startup_posture_ok still False (no/stale telemetry, or posture check not yet passed)
- **Response:** Nav ramped to zero, NO arm packet (the robot's 0.5 s arm-silence failsafe holds/crawls), and ONLY gaze_targets published — and only if _gaze_seeded (else empty packet). Eyes keep the claim/track/scan loop running so the 2 s robot-side gaze failsafe stays armed
- **Constants:** robot-side failsafes: nav 1 s, arm 0.5 s, gaze 2 s (header 52-54)
- **State:** _startup_posture_ok, _gaze_seeded, _vx, _wz
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none for the degraded gaze-only output (all mk_rig tests pass through this gate with a valid posture; t_tilt_gate covers only the refuse branch)
- **Incident:** FAIL CLOSED comment block: 'a full mission may not command anything until the startup posture check has actually PASSED'
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-013 · NaN/malformed detection drop (gaze-reference poisoning guard)

- **Location (legacy):** 754-761 (cube tracks), 786-789 (same drop for gripper tags)
- **Protects against:** A single NaN entering a cube track used to poison the gaze reference and every downstream target (fail-open class) — NaN propagates through _aim_angles, _slew_gaze, and the published gimbal reference
- **Trigger:** Detection entry with n_inliers < 1, pos shape != (3,), or any non-finite component
- **Response:** Entry is dropped before touching cube.pos/last_seen — 'malformed/NaN detection: must never enter a track'; identical guard on the gripper-tag stream
- **Constants:** n_inliers >= 1 (cubes) / SERVO_MIN_INLIERS (gripper tags); shape (3,); np.isfinite
- **State:** cube.pos, cube.last_seen (protected)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-15 audit, fail-open class
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)

### SAFE-GAZE-014 · Continuous 20 Hz gaze publish keeps the robot-side 2 s failsafe armed; DONE holds robot-side

- **Location (legacy):** 52-54, 85-86, 2703-2706
- **Protects against:** Robot-side gaze silence failsafe (2 s) tripping mid-mission from a slow operator loop, or an uncontrolled reference after mission end
- **Trigger:** Every tick: OP_RATE_HZ=20.0 chosen explicitly > all robot-side timeout rates; on Phase.DONE the eye deliberately stops
- **Response:** gaze_targets rides every 9874 packet in every phase except DONE ('eye runs in every phase except DONE — then reference holds robot-side'): mission end freezes the gimbal at its last slewed reference rather than continuing to scan
- **Constants:** OP_RATE_HZ=20.0; robot-side silence failsafes nav 1 s / arm 0.5 s / gaze 2 s
- **State:** self.phase, self.gaze
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Header safety block: every silence failsafe stays armed because we publish continuously at OP_RATE_HZ
- **New location (S6):** auto_operator/gaze.py + operator tick() (encoder seeding, slew application)


## Startup & posture

### SAFE-STARTUP-001 · Telemetry freshness gate feeding all startup guards

- **Location (legacy):** 2817-2822 (main loop)
- **Protects against:** Frozen/restarted 9870 telemetry stream keeps feeding yesterday's encoders to every posture guard — all startup checks would pass on stale data (fail-open) and a blind mission would start
- **Trigger:** time.monotonic() - last_tel_arrival > 0.5 s (tel_sub.data holds the LAST packet forever)
- **Response:** telemetry dict replaced with {} so tick() sees NO telemetry at all; every downstream guard (posture, tilt, gaze seed, home settle) then fails CLOSED
- **Constants:** staleness window = 0.5 s (unnamed literal, line 2822)
- **State:** last_tel_id, last_tel_arrival
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** 07-15 audit: tel_sub retains the last packet forever, so a dead stream silently satisfied every posture guard
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-002 · Fail-closed no-command-before-posture-pass branch

- **Location (legacy):** 2539-2551 (branch), 496-503 (_startup_posture_ok init)
- **Protects against:** An operator relaunched against a wrong --robot-ip, or before the arm homed, drives a blind Cartesian reach from an unknown arm posture
- **Trigger:** self._startup_posture_ok still False (the posture check needs finite telemetry to even run, so no/NaN telemetry keeps it False forever)
- **Response:** nav ramped to zero via self._ramp, packet carries gaze_targets ONLY (or {} if gaze unseeded), NO arm packet — the robot's arm-silence failsafe holds/crawls
- **Constants:** none (pure state gate)
- **State:** _startup_posture_ok, _vx, _wz, _gaze_seeded
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (all tests construct full telemetry at power-on pose from tick 1)
- **Incident:** Comment 2539-2544: a full mission may not command anything until the startup posture check has actually PASSED; the demo walk test has the same guard
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-003 · Startup posture gate: min-delta vs POWERON and raised-front homes

- **Location (legacy):** 2479-2517 (gate), 304-310 (STAGE_JOINTS['front']), 323-325 (POWERON_JOINTS)
- **Protects against:** Reaching from an unknown/off-home arm posture crosses IK configuration basins with no staircase protection — arm strikes the torso (07-15 cross-body jump class)
- **Trigger:** per-arm min(max|jpos - front|, max|jpos - poweron|) > FRONT_HOME_TOL_RAD for left slice 13:20 (sign +1) or right slice 20:27 (sign -1), evaluated once on first finite telemetry
- **Response:** raise SystemExit('[operator] STARTUP REFUSED: arm(s) not at the power-on pose ... Home the arm first: power-cycle, let the failsafe crawl finish, or run humanoid_stage_walk_test.py')
- **Constants:** FRONT_HOME_TOL_RAD=0.35 rad; STAGE_JOINTS['front']=[-0.1943,-0.6844,-0.3430,1.6406,-0.4615,-0.6902,-0.5096]; POWERON_JOINTS=[-0.1786,-0.6451,-0.4061,1.4804,-0.4229,-0.5632,-0.5089]
- **State:** _startup_posture_ok
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none for the refusal path (every test starts at poweron_jpos() so the gate passes silently; no off-home refusal test exists)
- **Incident:** 07-17 raised stations: TWO homes accepted because gravity sag (~5 deg at 0.8 torque) plus the deliberate 9.2 deg power-on->new-front offset would otherwise eat the 0.35 rad margin (min-delta over both)
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-004 · FRONT_HOME_TOL_RAD + _arm_near_front fail-closed home verification

- **Location (legacy):** 213-217 (constant), 2338-2354 (_arm_near_front), consumers: startup gate 2509, _idle_arms_home 2356-2368, per-arm REST gate 2435
- **Protects against:** Garbage/NaN joint telemetry classified as 'arm at home' would let the keep-alive publish a Cartesian REST that drags a side/rear-basin arm across the torso
- **Trigger:** jpos is None, len(jpos)<31, or non-finite arm segment — OR max|seg - sgn*home| > FRONT_HOME_TOL_RAD vs every known home (front, poweron, + SIDE_HOME_JOINTS in side-home mode)
- **Response:** returns False (fails CLOSED): the arm is treated as NOT home — idle REST withheld, arm held in place or stream silent
- **Constants:** FRONT_HOME_TOL_RAD=0.35 rad — real PD-vs-gravity sag of the HELD pose measured ±14 deg (07-15); SIDE differs 85 deg, REAR 155 deg so discrimination is untouched
- **State:** _jpos (cached per tick at 2468)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_glide_no_mans_land exercises the near-home-False mid-glide region; none for the NaN/short-jpos fail-closed path
- **Incident:** 07-15 audit fail-open class: NaN > x is False, so a plain comparison would pass a garbage posture as 'home'
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-005 · isfinite gates on startup telemetry slices

- **Location (legacy):** 2471-2472 (gaze seed jpos[27:31]), 2479-2483 (posture gate jpos[13:27]); same pattern at 1270 (home settle), 2348 (_arm_near_front), 2452 (_arm_packet hold-in-place), 1825 (_tilt_deg)
- **Protects against:** NaN posture would PASS the startup check (NaN > x is False — fail-open) and NaN encoders would seed the gaze reference, so a corrupted first packet arms the mission or slams the gimbal
- **Trigger:** np.all(np.isfinite(...)) False on the relevant telemetry slice
- **Response:** the entire seeding/posture block is skipped: _startup_posture_ok/_gaze_seeded stay False and control falls through to the fail-closed no-command branch (2539-2551)
- **Constants:** slice indices: gimbal jpos[27:31], arms jpos[13:27], left 13:20 / right 20:27
- **State:** _startup_posture_ok, _gaze_seeded
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (no test injects NaN/short joint_pos)
- **Incident:** Comment 2481-2483 documents the fail-open class explicitly: 'without the isfinite gate a NaN posture would PASS the check — fail-open'
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-006 · Startup tilt refuse/warn (projected_gravity)

- **Location (legacy):** 2484-2497 (gate), 245-253 (constants), 1816-1830 (_tilt_deg)
- **Protects against:** A tilted torso corrupts the ENTIRE body-frame geometry chain: table cubes read z=+0.2 chest height at r_xy≈0.5, the OOD-fooled learned gate waves them through, and mink's use_yaw_frame IK diverges from body-frame commands so arms never land where pointed
- **Trigger:** _tilt_deg = degrees(arccos(-g_z/|g|)) from telemetry projected_gravity, evaluated inside the startup posture block: > TILT_REFUSE_DEG refuses, > TILT_WARN_DEG warns
- **Response:** refuse: SystemExit('STARTUP REFUSED: torso N deg off vertical ... Fix the HANG/STANCE (RULE #0: legs straight, torso upright) and restart'); warn band: print advising to verify the first [reach] latched z is NEGATIVE. CAVEAT: _tilt_deg returns None on absent/malformed projected_gravity (older real_env) and the gate is then silently SKIPPED — fail-OPEN for a missing field
- **Constants:** TILT_REFUSE_DEG=15.0, TILT_WARN_DEG=8.0
- **State:** _startup_posture_ok (gate runs once inside its block)
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_tilt_gate (30-deg torso REFUSED via SystemExit; 5-deg passes); none for the missing-projected_gravity fail-open caveat
- **Incident:** 07-16 + three 07-17/18 recurrences of the hang-too-low class (bent legs -> torso tilt -> geometry corruption); grasp-carry MEMORY RULE #0: hang legs straight — 3 failed hardware sessions traced to this
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-007 · Continuous mid-mission tilt watch

- **Location (legacy):** 2518-2524
- **Protects against:** Hang/stance degrades MID-mission (robot sags on the hang, legs bend) after the startup gate already passed — latched body-frame targets silently corrupt
- **Trigger:** every 5.0 s (throttle via _tilt_diag_t), _tilt_deg(telemetry) > TILT_WARN_DEG
- **Response:** warn-only print: '[tilt] torso N deg off vertical — body-frame geometry suspect (latched targets may be corrupted)'; no automatic abort — operator must intervene
- **Constants:** TILT_WARN_DEG=8.0; diag period 5.0 s (unnamed literal)
- **State:** _tilt_diag_t (lazily created via getattr default -1e9)
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Comment 2519: 'the hang/stance can degrade MID-mission' — follow-on to the 07-16/18 tilt incident class
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-008 · Home-settle gate before the FIRST reach

- **Location (legacy):** 1257-1292 (gate in leg-start logic), 262-267 (constants), 531-532 (state init)
- **Protects against:** Reaching from the power-on pose mid-glide starts the Cartesian approach from an unvalidated seed — approach geometry computed from an arm that is not where the plan assumes
- **Trigger:** any non-held arm's MEASURED EE (pos_actual, isfinite-gated) farther than HOME_SETTLE_TOL_M from rest_ee[arm]; measured not commanded — 'the glide publishing the target is not the arm being there'
- **Response:** new reach legs blocked (return; diag print every 3 s); on both-arms-near: _home_settled=True + announcement; after HOME_SETTLE_TIMEOUT_S: proceeds anyway with LOUD warning '(EE telemetry gap or an obstructed glide) — proceeding anyway; watch the first reach' (deliberate fail-open to avoid stranding the mission forever)
- **Constants:** HOME_SETTLE_TOL_M=0.06 m, HOME_SETTLE_TIMEOUT_S=20.0 s (glide is ~5 s); diag throttle 3.0 s
- **State:** _home_settled, _home_wait_t0, _home_wait_diag_t
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none directly — implicitly gates the first reach in every test (mk_rig's perfect Cartesian tracking settles fast); the 20 s timeout/warning path is untested
- **Incident:** User 07-18 hard requirement: 'home first, THEN grab — never reach from the power-on pose'
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-009 · Home glide: measured-EE seeding, 8 cm/s ramp, announcement

- **Location (legacy):** 2370-2393 (_home_glide), 1420-1435 (_gentle_approach), 346-350 (rate), 2455-2457 (re-seed reset), 86 (DT)
- **Protects against:** Publishing rest_ee RAW lets the robot-side IK sweep the whole distance at its 0.25 m/s cap — a violent ~40 cm sweep to the side home ('arms shot out to the sides at startup', hardware 07-18)
- **Trigger:** idle/parked arm not at home at mission start (keep-alive REST from MISSION START per 07-17 user spec, lines 2650-2659) or dyn-retract homing
- **Response:** straight-line ramp seeded from the MEASURED EE at STAGE_REACH_EE_RATE; no pos_actual seed => pub None => caller goes stream-SILENT (robot holds pose) — the ramp is never seeded at the goal; glides > 0.03 m are announced ('[home] {arm} arm gliding to SIDE/REST home (N cm at 8 cm/s, ~Ns)'); _home_stream[arm]=None re-seeds whenever the slot is owned by reach/hold/carry
- **Constants:** STAGE_REACH_EE_RATE=0.08 m/s, per-tick step=STAGE_REACH_EE_RATE*DT (DT=1/OP_RATE_HZ, 20 Hz), announce threshold 0.03 m (unnamed)
- **State:** _home_stream{left,right}, _ee_act, side_home/rest_ee
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_startup_lift (rise to REST + per-tick step <= 8 cm/s * DT), t_side_home_basic (same assertions for SIDE home)
- **Incident:** Hardware 07-18: side-home arms shot out at the robot's 0.25 m/s cap because rest was published raw
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-010 · Active-glide exemption to the near-home gate (no-man's-land)

- **Location (legacy):** 2356-2368 (_idle_arms_home), 2432-2457 (_arm_packet idle slots)
- **Protects against:** Two-sided: (a) gating the glide on _arm_near_front FROZE it 8 cm out (mid-glide joints sit between power-on and side-home, near-home False for every reference — hardware 07-18: pos_target stuck at [0.27, 0.29], arms never reached the side); (b) removing the gate entirely would drag a stranded side/rear arm home across the torso in one Cartesian packet
- **Trigger:** idle-arm slot: _arm_near_front(arm) OR _home_stream[arm] is not None (an ACTIVE glide keeps publishing to completion); a STRANDED arm (stream None, off-home) falls to the else-branch
- **Response:** active glide publishes its ramp point regardless of the posture gate (the glide itself is the safety mechanism: seeded from the measured EE, 8 cm/s, straight line); stranded arm is held IN PLACE at its finite measured EE, or the whole packet returns None (stream silence, real_env holds last target) when no safe value exists
- **Constants:** FRONT_HOME_TOL_RAD=0.35 (via _arm_near_front), STAGE_REACH_EE_RATE=0.08
- **State:** _home_stream, _ee_act, _jpos, _held, _sec
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_glide_no_mans_land (glide completes with joints planted mid-way between POWERON_JOINTS and SIDE_HOME_JOINTS)
- **Incident:** Hardware 07-18 glide-freeze; 07-15 audit: phase-handler packets used to bypass the _idle_arms_home guard (one un-staged Cartesian REST = catapult class)
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-011 · Gaze reference seeding from actual gimbal encoders

- **Location (legacy):** 2471-2478 (seed), 566-570 (init/rationale), 2531-2532 and 2548-2551 (no-publish-until-seeded branches)
- **Protects against:** Zeros-init first gaze packet slams a parked gimbal multiple radians to 0 in one control step
- **Trigger:** _gaze_seeded False until jpos has >=31 finite entries; seed copies jpos[27:31] into self.gaze before the first publish
- **Response:** until seeded NO gaze_targets are published in ANY mode (gaze-only returns (None, {}); pre-posture branch returns nav + {}): real_env holds its last clamped reference — benign
- **Constants:** gimbal slice jpos[27:31]
- **State:** _gaze_seeded, self.gaze
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (rig telemetry has finite zeros at jpos[27:31] so seeding always succeeds silently)
- **Incident:** 07-15 audit: zeros-init + a parked gimbal = a multi-radian slam in one step
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)

### SAFE-STARTUP-012 · Gaze-only mode exemptions (structurally cannot command arms/nav)

- **Location (legacy):** 2479-2480 (posture+tilt gate skipped: 'and not self.gaze_only'), 2527-2534 (nav=None, packet carries ONLY gaze_targets), 2766 (no reachability gate), 2772-2778 (grasp/dual_parallel/side_home force-disabled), 2781 (nav_pub=None — 9873 never created), 2825 (nav publish guarded), 37-50 (docstring)
- **Protects against:** Without the structural lockout, the exemption from the startup posture/tilt gates would let a gaze-only launch (arms possibly off-home) emit arm or nav commands
- **Trigger:** self.gaze_only True (constructed from --gaze-only)
- **Response:** startup posture gate and tilt refuse legitimately SKIPPED because the mode can never walk/reach/grasp: tick() returns nav=None and a gaze_targets-only packet (still gated on _gaze_seeded); main() never constructs the 9873 publisher and ANDs grasp, dual_parallel, side_home with 'not gaze_only'
- **Constants:** none (mode flag)
- **State:** gaze_only, _gaze_seeded
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (test rig never constructs gaze_only=True)
- **Incident:** Docstring 37-50: reduced mode that ONLY tracks pre-registered cubes with the head — needs neither --use-ik nor 9873
- **New location (S6):** operator tick() startup block + auto_operator/safety.py (tilt_deg) + mission.tick_search (home-settle gate)


## Workspace envelope & same-side

### SAFE-ENVELOPE-001 · _target_safe envelope predicate (r band + z band + isfinite)

- **Location (legacy):** 1103-1115 (constants 197-212)
- **Protects against:** IK has no self-collision awareness and reach math ignored z: a cube against the torso or at face height becomes a literal IK target (hardware damage); r>0.45 is also the torso-tilt geometry-corruption signature that the OOD-fooled learned gate reaches at
- **Trigger:** any candidate arm point with non-finite coords, r_xy outside [0.16, 0.45] m, or z outside [-0.40, 0.30] m
- **Response:** returns False; caller treats the point exactly like out-of-reach — never latched, never followed, never reached
- **Constants:** TARGET_MIN_RADIUS_M=0.16, TARGET_MAX_RADIUS_M=0.45, TARGET_Z_MIN_M=-0.40, TARGET_Z_MAX_M=0.30
- **State:** none (pure static function)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none directly (only pass-path exercised via t_grasp_carry_side / t_inflight_retrack)
- **Incident:** 2026-07-15 adversarial audit: trunk keep-out and z-band both absent, two confirmed hardware-damage findings; TARGET_MAX_RADIUS_M added 07-18 after chest-height z=+0.2/r_xy~0.5 corrupted cubes were reached at again (07-18 14:24)
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-002 · SEARCH candidate envelope filter

- **Location (legacy):** 1309-1319
- **Protects against:** starting a mission leg toward a torso/face-height/tilt-corrupted target
- **Trigger:** confirmed pending cube with _target_safe(c.pos) False during SEARCH
- **Response:** cube skipped (never becomes active); loud once-per-cube warning (key '{key}:env' in _sector_warned) citing RULE #0 (legs straight, torso upright) and the torso-tilt signature
- **Constants:** TARGET_MIN_RADIUS_M=0.16, TARGET_MAX_RADIUS_M=0.45, TARGET_Z_MIN_M=-0.40, TARGET_Z_MAX_M=0.30
- **State:** self._sector_warned
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-18: hang-too-low torso tilt made cubes read chest height; the learned gate is OOD there and would have let the mission reach
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-003 · SEARCH midline dead-band filter

- **Location (legacy):** 1302-1306
- **Protects against:** detection jitter at y~0 coin-flips the arm choice and (rear especially) commits an un-audited cross-back reach
- **Trigger:** pending cube with |pos_y| <= 0.03 m
- **Response:** cube skipped until it sits clearly in one half-space; never selected as active
- **Constants:** MIDLINE_MARGIN_M=0.03
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-15 adversarial audit (cross-back reach class)
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-004 · SEARCH hard same-side arm-free binding

- **Location (legacy):** 1297, 1307-1308 (contract in docstring 1222-1225, header line 30)
- **Protects against:** an arm crossing the midline to serve the other half-space cube while its own-side arm is busy (arm-vs-arm collision; IK has no avoidance)
- **Trigger:** cube whose half-space arm ('left' if y>0 else 'right') is currently in self._held
- **Response:** cube stays in the pending queue and waits; only the matching free arm may ever pursue it
- **Constants:** MIDLINE_MARGIN_M=0.03 (side decided by sign of y beyond it)
- **State:** self._held keys
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none for the wait case (t_dual_serialized only exercises the pass path routing each cube to its own-side arm)
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-005 · GO arrival on-midline requeue

- **Location (legacy):** 1182-1189
- **Protects against:** arm choice riding one noisy frame's sign of y at the arrival tick
- **Trigger:** gate/geometric arrival fires while active cube |y| <= 0.03 m
- **Response:** abort the leg, phase -> SEARCH (re-queue); no arm is bound, no reach target latched
- **Constants:** MIDLINE_MARGIN_M=0.03
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-15 audit: side pick would ride one noisy frame's sign
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-006 · GO arrival busy-side-arm requeue

- **Location (legacy):** 1190-1197
- **Protects against:** reaching a cube on a half-space whose arm already holds the other cube — would force a cross-body reach or double-book the arm
- **Trigger:** arrival with side arm ('left' if y>0) already present in self._held
- **Response:** abort leg, phase -> SEARCH re-queue; logs that the one-cube-per-side placement contract was broken
- **Constants:** MIDLINE_MARGIN_M=0.03 (side derivation)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-007 · GO hard same-side binding overrides gate best_arm

- **Location (legacy):** 1176-1203 (override log 1198-1200, bind 1201-1202)
- **Protects against:** the side-agnostic learned gate crowning the cross-body arm -> arms cross the midline (no arm-vs-arm avoidance in IK)
- **Trigger:** gate_fired with gate.best_arm != half-space side of the arrived cube
- **Response:** disagreement is logged only; _reach_arm is bound to the half-space side arm regardless — the gate decides WHEN we are close enough, never WHICH arm
- **Constants:** MIDLINE_MARGIN_M=0.03, GATE_CONSECUTIVE=5, GEOMETRIC_FLOOR_M=0.28
- **State:** self._reach_arm, self._arrived, self._gate_hits
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** dual-cube collision safety rule; user places one cube per half-space
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-008 · Latch-median side + envelope re-validation

- **Location (legacy):** 1372-1384
- **Protects against:** arrival frame alone chose the arm; a lagged median can sit on the wrong half-space or outside the envelope -> rear-midline cross-back reach
- **Trigger:** per-axis median of last 0.7 s of sightings has median_y not strictly beyond +0.03 (left arm) / -0.03 (right arm), or fails _target_safe
- **Response:** leg aborted before REACH, phase -> SEARCH re-acquire
- **Constants:** MIDLINE_MARGIN_M=0.03, CUBE_LATCH_WINDOW_S=0.7
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (pass path only in t_inflight_retrack)
- **Incident:** 07-15 audit: rear-midline cross-back + envelope findings
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-009 · Empty latch-window crash guard + requeue

- **Location (legacy):** 1363-1371
- **Protects against:** np.median over an empty stack CRASHES the operator mid-mission with arms live
- **Trigger:** occlusion exactly at the arrival tick -> zero sightings inside the 0.7 s latch window
- **Response:** phase -> SEARCH re-acquire instead of computing the median
- **Constants:** CUBE_LATCH_WINDOW_S=0.7
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** 07-15 audit crash finding
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-010 · In-flight retrack validation bundle (_live_retrack)

- **Location (legacy):** 1597-1632 (guards 1611-1630); call sites 1499-1503 (primary REACH), 1741-1745 (secondary)
- **Protects against:** the live-follow target sliding across the midline, out of the envelope, or across a sector basin boundary UN-staged (facet-2 cross-body class); also chasing jitter
- **Trigger:** cube stale (>1.0 s) or window empty; median off the arm's strict half-space; _target_safe False; left the latched sector past 10 deg hysteresis; side-home bearing jump > 45 deg; or move <= 1 cm deadband
- **Response:** returns None -> target stays FROZEN at the last validated point; only a fully validated move > DYN_RETRACK_M refreshes it
- **Constants:** DYN_LOST_GRACE_S=1.0, CUBE_LATCH_WINDOW_S=0.7, MIDLINE_MARGIN_M=0.03, DYN_RETRACK_M=0.01, BEARING_JUMP_DEG=45.0 (side-home), SECTOR_HYST_DEG=10.0 via _sector_left
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_inflight_retrack, t_inflight_sector_clamp (sector clamp); side/envelope reject paths untested
- **Incident:** 07-17: grabbed-off-centre class (balance sway drifts the body frame 1-2 cm mid-approach) + facet-2 cross-body basin slide; VIA-HOME rule user 07-18
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-011 · Dual-parallel secondary latch validation + GATE_CONSECUTIVE debounce

- **Location (legacy):** 1674-1730 (mode gates 1674-1688; per-cube checks 1689-1710; streak 1711-1721)
- **Protects against:** committing the second arm on a single stochastic gate verdict, to an on-midline / wrong-side / non-front / out-of-envelope cube, or while a joint staircase owns the wire
- **Trigger:** any of: hold_after_reach off; primary has a track or non-front sector (non-side-home); any held track active; arm busy; cube unconfirmed, |y|<=0.03, wrong side for the free arm, non-front, _target_safe False (live pos 1697 AND median 1708), _reachable_now False; or _sec_hits streak < 5 consecutive qualifying ticks
- **Response:** candidate skipped and/or streak reset (_sec_hits=None); _sec latches only after 5 consecutive qualifying evaluations of the same cube key
- **Constants:** MIDLINE_MARGIN_M=0.03, CUBE_LATCH_WINDOW_S=0.7, GATE_CONSECUTIVE=5
- **State:** self._sec, self._sec_hits
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_front (pass path), t_dual_parallel_side_gate (side-primary gate), t_dual_parallel_off (default-off); reject/debounce paths untested
- **Incident:** 07-17 audit: gate samples fresh random candidates per call so single verdicts chatter; legacy park mode would strand the mission in HOLD forever
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-012 · _dyn_reachable envelope-first + best_arm + directional GATE_CONSECUTIVE hysteresis

- **Location (legacy):** 1117-1145
- **Protects against:** hold follow flapping at the reachability boundary (stop-start arm chatter) or following a point outside the safety envelope / crowned to the other arm
- **Trigger:** per-tick HOLD follow of a held cube; _target_safe False short-circuits; else gate score/best_arm sampled
- **Response:** _target_safe False -> immediate False (no debounce). Gate path: h['best_arm']=best if score>threshold else None; servable requires best==arm; retracted arm needs 5 consecutive FRESH in-hits to re-reach, tracking arm needs 5 fresh out-misses to retract. --no-use-gate fallback: geometric hysteresis (re-reach only inside 0.28, retract beyond 0.32)
- **Constants:** GATE_CONSECUTIVE=5, GEOMETRIC_FLOOR_M=0.28, DYN_OUTREACH_M=0.32, gate.SCORE_THRESHOLD
- **State:** h['in_hits'], h['out_hits'], h['home'], h['best_arm']
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_gate_throttle (cached evals must not advance counters); envelope short-circuit untested
- **Incident:** gate samples fresh random candidates per call -> single verdicts chatter (sim ReachabilityGate semantics carried over)
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-013 · _reachable_now instantaneous check (envelope-first)

- **Location (legacy):** 1805-1814
- **Protects against:** a turnaround / secondary latch decided toward an unsafe or other-arm point
- **Trigger:** settled-station turnaround decisions and secondary-latch screening
- **Response:** _target_safe False -> False; else requires gate score>threshold AND best_arm==arm (geometric r<=0.32 fallback). Deliberately undebounced — callers add their own streaks
- **Constants:** DYN_OUTREACH_M=0.32, gate.SCORE_THRESHOLD
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_gate_throttle (throttle behavior via _reachable_now)
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-014 · _gate_eval throttle + fresh-sample flag

- **Location (legacy):** 1780-1803
- **Protects against:** unthrottled CPU-torch scoring starved the gaze loop (pre-reach camera shudder); counting cached repeats would let one lucky stochastic sample latch a debounce
- **Trigger:** any gate consumer; repeat query of the same 2 cm-quantized target within 0.1 s
- **Response:** returns the cached (score, best_arm) with fresh=False; debounce counters everywhere step ONLY on fresh=True samples
- **Constants:** GATE_EVAL_PERIOD_S=0.1, 0.02 m quantization key, cache cap 128
- **State:** self._gate_cache
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_gate_throttle
- **Incident:** 07-17 camera-shudder fix
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-015 · _cube_servable freshness + strict same-side + reachability

- **Location (legacy):** 1892-1901 (call site 1958 in the GO/REACH-phase follow)
- **Protects against:** a held arm following a stale, cross-midline, or unreachable cube while another arm reaches
- **Trigger:** cube unseen > 1.0 s, or y beyond -0.03 (left arm) / +0.03 (right arm), or _reachable_now False
- **Response:** (False, None) -> the follow is suppressed and the hold target stays put
- **Constants:** DYN_LOST_GRACE_S=1.0, MIDLINE_MARGIN_M=0.03
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none directly
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-016 · GO/REACH-phase held-arm sector-clamped follow (Cartesian-vs-sector gap)

- **Location (legacy):** 1936-1976 (clamp 1954-1968, VIA-HOME pass 1961-1965)
- **Protects against:** split-verdict torso-strike: following an out-of-sector cube slides h['target'] across a basin boundary un-staged; at staging-completion one Cartesian packet then commands a cross-sector move from a stale basin — a gap the joint-vs-Cartesian serialization guards do NOT cover
- **Trigger:** while another arm is in GO/REACH, the held cube drifts out of h['sector'] past hysteresis, jumps bearing > 45 deg (side-home), goes stale, fails _cube_servable, or the arm is grabbing/grasped/carried or mid-track
- **Response:** target NOT updated (frozen); grab send still allowed (gripper bus only, 1969-1975); joint retreats/releases strictly deferred to HOLD/SEARCH; out-of-sector cube handled later by the normal _sector_left release
- **Constants:** DYN_LOST_GRACE_S=1.0, DYN_RETRACK_M=0.01, BEARING_JUMP_DEG=45.0, SECTOR_HYST_DEG=10.0 (via _sector_left)
- **State:** h['sector'], h['target'], h['grasp'], h['track']
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (t_dual_serialized asserts only Guard-1 no-track, not this sector clamp)
- **Incident:** 07-15 adversarial audit torso-strike facet 2 (unclamped GO/REACH follow), companion to the 0519d3c ret_payload interleave fix
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-017 · HOLD_RELEASE_S persistent-unservable release (requeue to correct arm)

- **Location (legacy):** 2210-2278 timer, 2301-2307 release (const 193-195); wrong_t0 resets 2251, 2278, 2281; init 1649
- **Protects against:** (a) an arm dragged across the body midline chasing a moved cube (arm-vs-arm/torso collision); (b) transient jitter or a single gate verdict releasing a good hold
- **Trigger:** seen held cube continuously wrong_side (y beyond -0.03/+0.03 for left/right), OR left h['sector'] past hysteresis, OR h['best_arm'] not in (None, arm), for > 3.0 s on the wrong_t0 timer; any servable tick or unseen-gap resets the timer
- **Response:** hold released: cube.held_by=None (cube re-enters the pending queue per 1229-1230 for the correct arm), del self._held[arm]; if phase==HOLD rewind to SEARCH. Until release the arm retracts home (h['home']=True), never follows across
- **Constants:** HOLD_RELEASE_S=3.0, MIDLINE_MARGIN_M=0.03, SECTOR_HYST_DEG=10.0
- **State:** h['wrong_t0'], h['home'], h['best_arm'], cube.held_by
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** hard same-side rule for dual-cube holds; margin is a hysteresis band so detection jitter at y~0 cannot flap the verdict
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-018 · _lift_arc per-waypoint sector + envelope guard (truncate, never skip)

- **Location (legacy):** 959-986 (guard 983-984)
- **Protects against:** a lift-off waypoint leaving the hold's sector band or the safety envelope (cross-basin / torso-ward clearance sweep); conversely, an early-capped arc leaves the gripper over the cube's stand when the retract starts
- **Trigger:** any of the ~2 cm arc points has _sector(wp) != sector or _target_safe(wp) False
- **Response:** arc truncated at the first violating point (partial arc beats no arc); sweep is otherwise ALWAYS the full 8 cm, never capped at the station bearing
- **Constants:** GRASP_LIFT_OUT_M=0.08 full arc, ~0.02 m waypoint spacing, STATION_BEARING_DEG[sector]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (asserts the full-arc endpoint; truncation path untested)
- **Incident:** user 07-17: clearance beats alignment — overshooting the station is harmless, an early stop dragged the cube back through its stand
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-019 · Lift-off wp1 envelope check (skip lift, keep carry)

- **Location (legacy):** 2144-2180 (check 2147; skip branch 2177-2180)
- **Protects against:** the +5 cm vertical clearance point exceeding the z band (e.g. z > TARGET_Z_MAX_M=0.30) becoming an IK target
- **Trigger:** grasp succeeded but _target_safe(target + [0,0,0.05]) is False
- **Response:** lift phase skipped entirely — h['grasp'] set straight to 'grasped' and the carry home proceeds without the clearance move ('lift skipped: envelope')
- **Constants:** GRASP_LIFT_UP_M=0.05
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (pass path only in t_grasp_carry_side)
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-020 · VIA-HOME routing latch (side-home big bearing jump)

- **Location (legacy):** 2218-2241 (at_home 2224-2229, jump 2230-2235, unlatch 2238-2241), latch 2265-2269; mirrored guards 1624-1628 and 1961-1965
- **Protects against:** chasing a front->rear cube jump directly across the body instead of staging through the side home
- **Trigger:** side-home mode: standoff-point bearing differs from the current target bearing by > 45 deg
- **Response:** h['via_home'] latches; arm retracts to the SIDE home; re-extend permitted ONLY once the MEASURED (finite) EE is within 0.10 m of rest_ee — a mid-retract sighting cannot shortcut the detour; missing/non-finite EE keeps it blocked
- **Constants:** BEARING_JUMP_DEG=45.0, at_home tolerance 0.10 m vs rest_ee, isfinite on _ee_act
- **State:** h['via_home'], self._ee_act[arm]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_via_home_routing
- **Incident:** user 07-18 hard requirement: never across the body; measured-not-commanded (publishing the target is not the arm being there)
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-021 · Requeue path: unreachable staging hop -> front -> SEARCH

- **Location (legacy):** 1447-1449 (policy), 1467-1472
- **Protects against:** retrying an unreachable hop in place with the arm parked mid-basin
- **Trigger:** a reach-leg staging hop times out; the track's goal flips to front(0) and settles back at front with an empty seq
- **Response:** leg abandoned (_reach_tm=None), phase -> SEARCH re-acquire; no in-place retry (that is HOLD's job)
- **Constants:** track machinery timeouts (tm['timed_out'])
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none in test_grasp_carry.py
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-022 · Requeue path: unseeded secondary ABANDON (catapult guard)

- **Location (legacy):** 1760-1770 (mid-ramp variant 1771-1778)
- **Protects against:** registering a hold for an arm that never moved makes the next packet publish the full REST->cube target in ONE step — the audited 07-16 catapult class
- **Trigger:** secondary reach times out with s['stream'] still None (no EE seed ever arrived)
- **Response:** secondary abandoned (_sec=None), NO hold registered; cube remains pending for a classic serialized leg. If a stream exists, the hold is instead seeded with restream=s['stream'] so the classic machinery finishes the ramp gently
- **Constants:** REACH_TIMEOUT_S=8.0
- **State:** self._sec, h['restream']
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_unseeded_abandon (asserts no hold, no >2 cm packet jump)
- **Incident:** 07-17 audit; primary is immune because it stalls before its timeout check
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary

### SAFE-ENVELOPE-023 · Requeue path: GO lost-cube rescan

- **Location (legacy):** 1347-1349
- **Protects against:** walking toward a stale base-frame position after the cube is gone (positions go stale the moment the robot walks)
- **Trigger:** active cube unseen > 3.0 s during GO
- **Response:** phase -> SEARCH (camera claim kept, sighting must be re-confirmed); meanwhile any detection gap > 0.5 s already zeroes the walk command
- **Constants:** DET_LOST_RESCAN_S=3.0 (blind-stop uses DET_STALE_S=0.5 at 1159-1160)
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none in test_grasp_carry.py
- **New location (S6):** auto_operator/safety.py (target_safe/sector) — call sites unchanged across mission/arms/secondary


## Sector staging (classic)

### SAFE-STAGING-001 · Sector classification thresholds (front/side/rear)

- **Location (legacy):** 282-290, 938-947
- **Protects against:** mink IK is a LOCAL solver: front-rest to side/rear target crosses the torso basin and the arm strikes the body (observed on hardware; the reason rear reaches were banned)
- **Trigger:** _sector(pos): abs base-frame bearing atan2(y,x) in deg; <=60 -> front, <=120 -> side, >120 -> rear; applied to the latched reach target and to every held-cube position
- **Response:** side/rear targets are routed through the pre-defined transit-posture TRACK before any Cartesian reach; front keeps the direct-reach behavior unchanged
- **Constants:** SECTOR_FRONT_DEG=60.0, SECTOR_REAR_DEG=120.0
- **State:** _reach_sector, h["sector"]
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (side staircase leg), t_side_home_basic
- **Incident:** upstream's concern, observed on hardware: QP path from the front rest pose to a side/rear target crosses the torso basin
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-002 · Sector-exit hysteresis (_sector_left)

- **Location (legacy):** 291-292, 949-957
- **Protects against:** detection jitter at a sector boundary flapping a held arm between sectors (release/re-stage churn, un-staged target slides across the boundary)
- **Trigger:** bearing > band_hi + 10 deg or < band_lo - 10 deg of the hold's sector band {front:(0,60), side:(60,120), rear:(120,180)}
- **Response:** only past the hysteresis margin is the cube treated as having left its sector (drives the follow clamp, retrack refusal, and the debounced hold release)
- **Constants:** SECTOR_HYST_DEG=10.0
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_inflight_sector_clamp (drift to 80 deg > 60+10 refused)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-003 · STAGE_JOINTS audited transit postures

- **Location (legacy):** 293-322
- **Protects against:** a transit posture that clips the cube stand or torso, or that exits the validated configuration basin
- **Trigger:** every track hop streams one of these postures
- **Response:** only the audited fixed postures are ever streamed as joint targets; runtime never solves IK for postures (7-DOF redundancy let online IK escape 163 deg to another branch; the Cartesian-transit attempt audited -33.7 mm wrist-through-hip)
- **Constants:** STAGE_JOINTS side=[0.0,-0.9599,0,0,0,0,0] (pure shoulder_2 coronal abduction, raised 07-17 25->55 deg, EE z -0.183 -> +0.012); front=[-0.1943,-0.6844,-0.3430,1.6406,-0.4615,-0.6902,-0.5096] (raised home, EE z +0.18, max 0.16 rad/joint from power-on); rear=[0.0951,-0.8619,0.3088,-1.3323,1.1987,-0.8343,-0.3777] (re-solved 07-17, 0.1 mm residual, >=48.7 deg limit margins); right arm = exact negation
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (asserts arm lands at STAGE_JOINTS["front"] within 0.05 rad after carry home)
- **Incident:** 07-17: the SIDE transit hub used to sweep at tabletop height and clipped the cube stand's legs (hence the raise); chain hardware-validated 07-14 (zero contact 0.025-0.3 rad/s), offline audits >=40 deg limit margins / >=49.9 mm clearance, raised-chain re-audit >=28.9 mm — hardware re-validation of the raised chain still pending
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-004 · TRACK linear order — side is always the middle station

- **Location (legacy):** 326-332
- **Protects against:** a direct front<->rear motion sweeping the arm through the torso
- **Trigger:** any staged motion between stations (cur_idx, target_idx), direction = sign(target-cur)
- **Response:** single bidirectional track; every front<->rear transit physically passes through side; one shared primitive (_step_toward) walks both directions (replaced the old forward/reverse queue pair)
- **Constants:** TRACK=("front","side","rear"), SECTOR_IDX={front:0, side:1, rear:2}
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (staircase out and carry home both walk the track)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-005 · STATION_BEARING_DEG interior-bearing invariant

- **Location (legacy):** 333-339
- **Protects against:** grasp clearance arc sweeping out of the hold's sector across a basin boundary
- **Trigger:** _lift_arc sweeps toward the held cube's sector-station bearing
- **Response:** each bearing is interior to its sector band so the sweep cannot leave the sector by construction; MUST be re-derived if STAGE_JOINTS or the model changes
- **Constants:** STATION_BEARING_DEG={front:38.07, side:90.08, rear:142.00} (FK'd once offline on the deploy model, 07-17 re-derived for the raised stations)
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (expected arc endpoint computed from STATION_BEARING_DEG["side"])
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-006 · STAGE_RATE payload + rate field + frozen-other-arm snapshot

- **Location (legacy):** 340-341, 1410-1418, 1053-1054
- **Protects against:** joint stream slewing at uncapped speed, or the non-moving arm being dragged mid-staircase because its slot carries live values
- **Trigger:** every staged joint payload (_staged_joint_payload)
- **Response:** all-14 payload {"joint_pos": moving-arm posture + other arm pinned at its motion-start encoder snapshot, "rate": STAGE_RATE}; a Cartesian hold cannot ride the same packet — a holding arm freezes for the staircase's duration and resumes on the Cartesian return
- **Constants:** STAGE_RATE=0.2 rad/s (user-tuned; needs real_env per-packet rate knob ae1ca6c; robot-side hard cap 0.5)
- **State:** tm["frozen"]
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none (rig applies joint_pos but never asserts the rate field or the frozen values)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-007 · STAGE_TOL_RAD per-joint arrival tolerance

- **Location (legacy):** 342-343, 1072-1073
- **Protects against:** looser tolerances CUT CORNERS at speed and deviate from the audited clearance curve
- **Trigger:** max per-joint |encoder - posture| < 0.03 counts the hop as arrived
- **Response:** only then the dwell starts; otherwise the posture keeps streaming (arrival is encoder-verified, never assumed)
- **Constants:** STAGE_TOL_RAD=0.03 rad (1.7 deg)
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none (rig has perfect joint tracking)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-008 · STAGE_DWELL_S per-station dwell

- **Location (legacy):** 344-345, 1061-1071
- **Protects against:** next segment starting from a not-yet-stabilized configuration off the audited curve
- **Trigger:** encoder arrival at each hop station
- **Response:** freeze 0.6 s at the waypoint; the hop counts 'settled' only after the dwell completes (tm["cur"] advances, seq pops) — the identical arrive-and-stabilize event in BOTH directions
- **Constants:** STAGE_DWELL_S=0.6 s
- **State:** tm["dwell_until"], tm["cur"], tm["seq"]
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none directly (implicit in every staircase test)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-009 · Hop crawl-time budget + timeout->front flip

- **Location (legacy):** 1053-1057, 1075-1079, 1086-1089
- **Protects against:** an unreachable/jammed hop stalling the arm mid-staircase forever
- **Trigger:** (now - hop_t0) > hop_budget while err >= STAGE_TOL_RAD
- **Response:** tm["timed_out"] latches, target flipped to front(0) as a PLAIN direction change (not a special abort path), inward sequence rebuilt from cur; callers react on arrival (REACH abandons to SEARCH, carry warns 'jam/sag?' and keeps crawling with the cube)
- **Constants:** hop_budget = max|joint delta|/STAGE_RATE * 1.4 + 5.0 s (recomputed at motion start, each hop advance, and each flip)
- **State:** tm["hop_t0"], tm["hop_budget"], tm["timed_out"]
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-010 · `or [0]` first-hop-timeout re-plant guard

- **Location (legacy):** 1080-1085
- **Protects against:** first-hop timeout with cur==0: _track_seq(0,0) is empty, which would read as an instant (false) 'returned to front' while the arm is physically stranded mid-hop off-station, and the next leg assumes it starts at front (07-15 audit)
- **Trigger:** hop timeout while tm["cur"]==0 (outward first hop)
- **Response:** station 0 is re-planted explicitly so 'done' means the arm actually settled there
- **Constants:** tm["seq"] = _track_seq(tm["cur"], 0) or [0]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** 07-15 audit finding
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-011 · _track_seq inward re-plant semantics

- **Location (legacy):** 1006-1022
- **Protects against:** an inward (retreat) motion starts from a Cartesian cube-follow pose that has drifted off the exact station posture — skipping the re-plant would run the reverse chain from an unaudited start configuration
- **Trigger:** target < cur (inward)
- **Response:** inward sequence INCLUDES cur as the first hop to RE-PLANT it (rear(2)->front(0) = [2,1,0]); outward = [cur+1..target] (arm already firmly at cur); target==cur -> [] (no motion)
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side exercises the [1,0] carry-home sequence but never asserts the re-plant hop itself
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-012 · _retarget_track settled-only rule

- **Location (legacy):** 1092-1099
- **Protects against:** changing a track's goal mid-hop swaps joint postures with the arm between stations — an un-audited transition
- **Trigger:** direction reversal / new-sector goal wanted for an in-flight track
- **Response:** retarget is only legal at a settled station; sequence rebuilt from tm["cur"], dwell cleared, timed_out reset; both callers (lines 2032, 2050) sit inside status in ('settled','done') branches
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-013 · _step_toward telemetry-blip guard

- **Location (legacy):** 1046-1049, 1987-1988
- **Protects against:** commanding staged joints off a missing/short joint_pos snapshot (garbage frozen-arm values, wrong error computation)
- **Trigger:** jpos is None or len(jpos) < 31
- **Response:** returns (None, 'moving') — caller keeps the joint stream SILENT and holds state; _held_update `continue`s on payload None unless status=='done'
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (rig always supplies a full jpos)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-014 · Joint->Cartesian switch only at a settled target station

- **Location (legacy):** 1444-1449, 1473-1498
- **Protects against:** the robot re-seeds its IK from encoders on the joint->Cartesian switch (real_env 8335075); switching mid-hop seeds the solver in no validated basin
- **Trigger:** _step_toward status == 'done' at the sector's target station
- **Response:** only then _reach_tm clears and the Cartesian stream begins; the 8 s reach clock (_reach_t0) and the servo give-up clock start only AFTER the track settles — never mid-hop
- **State:** _reach_tm, _reach_t0
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_grasp_carry_side (side reach completes through the switch; not asserted mid-hop)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-015 · WAITING-at-station branch (missing EE seed)

- **Location (legacy):** 1475-1495
- **Protects against:** 07-16 review: old flow cleared _reach_tm then returned None BEFORE the 8 s timeout check — a persistent pos_actual gap became a silent forever-stall with the arm parked at side/rear and the stream dead
- **Trigger:** track 'done' but telemetry ee.<arm>.pos_actual is None this tick
- **Response:** keep STREAMING the settled station posture as a staged joint payload (frozen fallback = other arm's front-station posture when tm["frozen"] is None) and retry next tick; throttled 1 s '[reach] WAITING at the station' diagnostic
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (t_dual_parallel_unseeded_abandon covers the secondary Cartesian analog, not this joint branch)
- **Incident:** 07-16 review finding (silent forever-stall class)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-016 · Timed-out reach track abandons to SEARCH

- **Location (legacy):** 1467-1472
- **Protects against:** retrying an unreachable staged leg in place from an unknown intermediate state
- **Trigger:** tm["timed_out"] latched and seq exhausted (arm settled back at front)
- **Response:** leg abandoned: _reach_tm=None, phase=SEARCH re-acquires the cube fresh; explicitly NO in-place retry ('that is HOLD's job')
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-017 · SECTOR_REACH_ENABLED gating + legacy-park non-front forbid

- **Location (legacy):** 358-366, 1320-1333
- **Protects against:** reaching into a sector whose staged path is not hardware-validated; or a side/rear reach in legacy --no-hold-after-reach mode, which has no reverse staircase to come home on
- **Trigger:** per-cube eligibility check in _tick_search
- **Response:** ineligible cube skipped from the ready list with a once-per-cube warning (_sector_warned set)
- **Constants:** SECTOR_REACH_ENABLED={"front":True,"side":True,"rear":True}; enabled = side_home or (SECTOR_REACH_ENABLED[sector] and (hold_after_reach or sector=="front"))
- **State:** _sector_warned
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none for the disabled path (every side/rear test force-sets the dict to all-True first)
- **Incident:** side/rear disabled 2026-07-15 after a hardware cross-body jump (arm swept front->rear through the torso); re-enabled after both facets fixed (interleave serialization + sector-clamped follow) for HARDWARE VALIDATION, single-arm first
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-018 · Track routing at target latch

- **Location (legacy):** 1395-1408
- **Protects against:** front legs needlessly riding the joint stream, or side-home mode running staircases that its geometry assumptions do not cover
- **Trigger:** _latch_reach_target after median/half-space/envelope validation
- **Response:** front leg = zero-length track -> straight to Cartesian (behavior unchanged); side/rear = track from station 0 out to the sector station; side-home = EVERY leg direct Cartesian (offline mink study 07-18: continuous, no basin jumps)
- **Constants:** target_idx = 0 if side_home else SECTOR_IDX[sector]; _reach_tm = _new_track(arm, 0, target_idx) only when target_idx > 0
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_side_home_basic (asserts ZERO joint-staircase ticks), t_grasp_carry_side (side leg stages)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-019 · One-track-at-a-time token (tracks vs reaches vs dual-parallel)

- **Location (legacy):** 1981-1984, 2115-2119, 2296-2299
- **Protects against:** two joint tracks, or a track plus a staged/dual-parallel reach, interleaving in the global 14-joint stream — packets override each other and fling a mid-basin arm across the torso (the 07-15 hardware jump class)
- **Trigger:** a held arm's track step while ret_payload was already produced this tick or _reach_tm is not None; carry-staircase start requires h["track"] is None, _reach_tm is None AND _sec is None; retract-track start requires _sec is None (belt-and-suspenders — side/rear holds cannot coexist with a dual-parallel reach by construction)
- **Response:** the waiting arm's slot stays frozen inside the active track's packet; carries queue — first grasped walks home first
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized (a_track_during_reach==0; A carried before B's carry starts), t_dual_parallel_side_gate (no secondary during a side staircase)
- **Incident:** 07-15 cross-body jump; ret_payload interleave facet audited airtight in 0519d3c
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-020 · Re-extend only at the settled station matching the cube's sector

- **Location (legacy):** 2012-2028
- **Protects against:** re-extending in Cartesian from mid-hop or from the wrong station = an un-staged cross-sector Cartesian move
- **Trigger:** status in ('settled','done') AND _cube_servable ok AND sect == tm["cur"] AND live ee.pos_actual present
- **Response:** track cleared, h["sector"]/h["sector_idx"] set to the station, restream seeded from the MEASURED EE for the gentle 8 cm/s ramp; missing EE this tick -> stay settled and retry
- **State:** h["track"], h["sector_idx"], h["restream"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none directly
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-021 · In-place turnaround at a settled station

- **Location (legacy):** 2029-2034
- **Protects against:** releasing and re-queueing mid-track when the cube reappears in a different sector would start a fresh un-staged leg instead of the safe reversal
- **Trigger:** settled station, cube servable, sect != tm["target"]
- **Response:** _retarget_track(tm, sect) — reverse toward the cube's sector from THIS settled station, ownership kept, nothing re-queued
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-022 · Release only at the front station

- **Location (legacy):** 2035-2043
- **Protects against:** releasing a side/rear hold off-front strands the arm off-home with no owner (hold machinery forgets it, stream falls silent)
- **Trigger:** cube not servable AND tm["target"]==0 AND tm["cur"]==0 AND seq empty (settled at front)
- **Response:** 'the only place we release' — cube re-queued (held_by=None), arm freed, HOLD->SEARCH
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-023 · Absorbing-state escape (off-front settled, cube gone)

- **Location (legacy):** 2044-2052
- **Protects against:** 07-15 audit: without this branch the track is an ABSORBING state — no branch fires, the arm strands at side/rear and the stream stays silent forever
- **Trigger:** cube not servable AND status=='done' AND tm["target"] != 0 (a turnaround's goal reached with no cube)
- **Response:** _retarget_track(tm, 0) — walk home; re-enters the front-only release branch on arrival
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-15 audit finding
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-024 · Side/rear retraction is a TRACK motion (front stays in-place)

- **Location (legacy):** 2293-2300, 1916-1920
- **Protects against:** a side/rear arm pulled straight to REST in one Cartesian packet crosses the torso basin
- **Trigger:** hold state flips to retracted (h["home"]) with h["sector_idx"] > 0 and _sec is None
- **Response:** h["track"] = _new_track(arm, sector_idx, 0) — walks the transit chain in reverse, interruptible (re-decided only at each settled station); front holds (sector_idx 0) keep the classic in-place REST retract; side/rear holds then ALWAYS release at front (never re-extend in place)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none for the plain (non-grasp) retract track; t_grasp_carry_side covers the grasped carry track
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-025 · _lift_arc sector/envelope guard + full-sweep rule

- **Location (legacy):** 959-986, 279
- **Protects against:** clearance arc leaving the hold's sector or safety envelope; an early-stopped arc leaving the gripper still over the cube's stand when the retract starts (user 07-17: clearance beats alignment)
- **Trigger:** each arc waypoint checked: _sector(wp) != sector or not _target_safe(wp)
- **Response:** truncate at the first violating point (partial arc beats no arc); full sweeps stay in-band by construction (worst case ~19 deg vs 60-deg bands)
- **Constants:** GRASP_LIFT_OUT_M=0.08, ~2 cm point spacing, r<1e-6 -> [] guard; direction = sign(station_bearing - current_bearing)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (asserts the full-arc endpoint to 1e-6)
- **Incident:** 07-17 user rule after stand-catching: arc is ALWAYS the full 8 cm toward the sector station, never capped at the station bearing and never skipped; overshoot is harmless
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-026 · GO/REACH held-arm follow sector clamp

- **Location (legacy):** 1936-1957
- **Protects against:** the split-verdict torso-strike (07-15 adversarial audit): a follow slides h["target"] across the basin boundary UN-staged while the reaching arm holds this arm at a one-shot encoder snapshot; at staging completion one Cartesian packet commands a cross-sector move from a stale basin — a Cartesian-vs-sector gap the joint-vs-Cartesian serialization does NOT cover
- **Trigger:** phase in (GO, REACH) and a held arm's cube has drifted past _sector_left of h["sector"] (non side-home)
- **Response:** follow refused (target unchanged); the out-of-sector cube is handled safely after the reach ends via the normal _sector_left release re-queue
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none in test_grasp_carry.py (scratchpad/test_sector_clamp.py per comments; t_inflight_sector_clamp covers the reach-leg analog)
- **Incident:** 07-15 adversarial audit; offline-verified via scratchpad/test_sector_clamp.py
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-027 · In-flight retrack sector clamp

- **Location (legacy):** 1597-1632 (clamp at 1620-1622)
- **Protects against:** a cube drifting across a basin boundary mid-Cartesian-leg slides the reach target un-staged (facet-2 cross-body class)
- **Trigger:** _live_retrack: _sector_left(median, sector) true (non side-home), or half-space/envelope failure
- **Response:** returns None — target stays clamped at the last in-sector value; occlusion (empty window) also freezes
- **Constants:** DYN_RETRACK_M=0.01 deadband; CUBE_LATCH_WINDOW_S=0.7 median window
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_inflight_sector_clamp, t_inflight_retrack (the follow side)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-028 · Sector-left hold release debounce

- **Location (legacy):** 193, 2216-2217, 2270-2276, 2301-2307
- **Protects against:** an in-place slide of a held arm into a new sector ('a sector change needs a fresh staged leg, not an in-place slide'), or release churn from momentary boundary jitter
- **Trigger:** held cube seen beyond its sector band + hysteresis persistently > 3.0 s (h["wrong_t0"] timer; resets when back in-sector)
- **Response:** hold released, cube re-queued for a fresh forward-staged leg; HOLD phase rewinds to SEARCH
- **Constants:** HOLD_RELEASE_S=3.0 (with SECTOR_HYST_DEG=10.0 upstream)
- **State:** h["wrong_t0"], h["sector"]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-029 · Gentle staged->cube Cartesian ramp (post-switch)

- **Location (legacy):** 346-350, 1420-1435, 1509-1520
- **Protects against:** handing the IK the full staged->cube jump (~30 cm) in one packet read as 'too fast' on hardware; the direct front reach snapped at the robot-side 0.25 m/s cap — a catapult that could knock the cube away at contact with --reach-standoff 0 (07-16)
- **Trigger:** every final Cartesian approach after the track settles (and front legs from the rest pose)
- **Response:** published point ramps from the MEASURED hand position toward the target at 8 cm/s; no EE seed -> pub None -> caller stays SILENT that tick (never seed the ramp at the goal)
- **Constants:** STAGE_REACH_EE_RATE=0.08 m/s, step = rate*DT (DT=1/20 s)
- **State:** _reach_stream
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_startup_lift (per-tick step <= STAGE_REACH_EE_RATE*DT), t_side_home_basic (gentle glide)
- **Incident:** 07-16 hardware observation (user: 'too fast'); front legs included since 07-16
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-030 · Dual-home posture acceptance + fail-closed near-front check

- **Location (legacy):** 304-310, 323-325, 213-217, 2338-2354, 2498-2516
- **Protects against:** reaching from an unknown posture crosses configuration basins (no staircase protects it); a NaN posture would pass a plain comparison as 'home' (NaN > x is False — 07-15 fail-open class)
- **Trigger:** startup posture check (SystemExit refusal if neither home within tolerance) and _arm_near_front gating every idle-arm REST publish
- **Response:** accept ONLY the front station, the power-on default, or (side-home) SIDE_HOME_JOINTS within 0.35 rad; missing/short/non-finite telemetry returns False (fail closed); off-front idle arms are held IN PLACE, never dragged home in one packet
- **Constants:** FRONT_HOME_TOL_RAD=0.35 (PD-vs-gravity sag measured +/-14 deg 07-15; SIDE differs 85 deg, REAR 155 deg — discrimination intact); STAGE_JOINTS["front"] deliberately solved <=0.16 rad/joint from POWERON_JOINTS so the startup gate still passes
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_startup_lift and t_glide_no_mans_land (gate + glide interaction); no test for the NaN/short-telemetry fail-closed path
- **Incident:** 07-15 audit fail-open class; 07-17 raised-front redesign kept the power-on gate margin
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator

### SAFE-STAGING-031 · Lift-ramp pause while the joint stream owns the wire

- **Location (legacy):** 2084-2090
- **Protects against:** the lifting Cartesian ramp advancing while a staircase owns the 14-joint stream: the point is not being published, so on resume the arm would jump the accumulated distance in one packet
- **Trigger:** h["grasp"]=='lifting' while _reach_tm is not None or any other hold's track is not None
- **Response:** ramp paused (continue) — h["restream"] holds its point until the wire returns to Cartesian
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none directly (t_dual_serialized exercises lift/carry deferral broadly)
- **New location (S6):** auto_operator/staging.py (step_toward, staged_joint_payload) + operator (_track_posture/_track_seq/_retarget_track) + STAGE_JOINTS constants in operator


## Side-home profile

### SAFE-SIDEHOME-001 · SIDE_HOME_EE/JOINTS provenance (IK-validated side station)

- **Location (legacy):** 158-175
- **Protects against:** Homing to the raw side STATION EE (r=0.652) parks the arm on the workspace boundary where mink IK holds with 35 mm error under NEUTRAL_QUAT — a home that never settles and grinds at the joint limits; streaming SIDE_HOME_JOINTS directly would bypass the Cartesian glide machinery
- **Trigger:** side_home mode selects the home target
- **Response:** Home is fixed at [0, +/-0.55, 0] (1.6 mm IK hold, solves as a clean side-raise); SIDE_HOME_JOINTS is the converged posture used ONLY for the near-home posture check (_arm_near_front), never streamed as a command
- **Constants:** SIDE_HOME_EE={left:[0,0.55,0], right:[0,-0.55,0]}; SIDE_HOME_JOINTS=[-0.008,-0.482,-0.006,0.005,-0.008,-1.080,0.010]; NEUTRAL_QUAT=[1,0,0,0]
- **State:** rest_ee
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_side_home_basic (glide-endpoint assertions)
- **Incident:** 07-18: r=0.55 chosen with the robot's OWN mink IK offline after the raw side station proved boundary-bound; streamed continuity power-on-to-home and home-to-front/side/rear verified <=0.03 rad/solve, EE path r>=0.27 m, landing parity 34.6 vs 34.3 mm against the front-flow control group
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-002 · rest_ee single-binding rule

- **Location (legacy):** 526 (binding); consumers 1273, 1846, 2229, 2380
- **Protects against:** Split home frames — e.g. the carry parking at front REST while the via-home at_home gate waits at the side — would either command a cross-body Cartesian move or create a settle gate that can never pass
- **Trigger:** AutoOperator constructor (side_home=True)
- **Response:** self.rest_ee = SIDE_HOME_EE if side_home else REST_EE is the ONE binding consumed by the HOME-FIRST settle gate (1273), the carried park target (1846), the via-home measured at_home gate (2229), and the home glide target (2380) — all four mechanisms agree on the same home by construction
- **Constants:** SIDE_HOME_EE vs REST_EE={left:[0.3207,0.2512,0.18], right:[0.3207,-0.2493,0.18]}
- **State:** self.rest_ee
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_side_home_basic, t_via_home_routing
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-003 · HOME-FIRST settle gate (measured, both arms)

- **Location (legacy):** 1257-1292
- **Protects against:** Reaching from the power-on pose mid-glide starts the cube approach from an unvalidated IK seed (side-home: a ~40 cm un-audited power-on-to-cube path)
- **Trigger:** _tick_search before _home_settled: any non-held arm's MEASURED EE missing, non-finite, or > 0.06 m from rest_ee[arm] (side-home: SIDE_HOME_EE)
- **Response:** No reach may start (return). Held arms are exempt (1267-1268). Missing/non-finite pos_actual counts as not settled (fail closed). After 20 s (glide is ~5 s) the gate proceeds with a LOUD warning instead of stranding the mission forever — a deliberate bounded escape for EE-telemetry gaps/obstructed glides
- **Constants:** HOME_SETTLE_TOL_M=0.06; HOME_SETTLE_TIMEOUT_S=20.0
- **State:** _home_settled, _home_wait_t0, _ee_act
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (implicitly exercised by t_side_home_basic, no direct assertion on the gate)
- **Incident:** User 07-18 rule: home first, THEN grab — never reach from the power-on pose; measured-not-commanded because the glide publishing the target is not the arm being there
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-004 · Sector-machinery bypass in _tick_search — surviving checks

- **Location (legacy):** 1323-1333 (bypass); surviving 1300-1319 and 1373-1384
- **Protects against:** Bypassing sector gating must not also bypass envelope/midline — a torso-adjacent, face-height, or cross-midline cube would become a literal IK target (07-15 adversarial audit: confirmed hardware-damage findings, trunk keep-out + z-band were absent)
- **Trigger:** side_home=True cube selection for ANY sector
- **Response:** enabled = self.side_home or (...) skips SECTOR_REACH_ENABLED and the hold_after_reach front-only restriction ONLY; confirmed-sighting debounce (>=0.3 s), on-midline refusal (|y|<=0.03), free-arm half-space assignment, _target_safe envelope, and the latch-time median side/envelope re-checks (1373-1384) all remain active
- **Constants:** MIDLINE_MARGIN_M=0.03; TARGET_MIN_RADIUS_M=0.16; TARGET_MAX_RADIUS_M=0.45; TARGET_Z_MIN_M=-0.40; TARGET_Z_MAX_M=0.30; SECTOR_REACH_ENABLED (bypassed)
- **State:** _sector_warned
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_side_home_basic, t_side_home_dual_any_sector
- **Incident:** r>0.45 or z>0 for a table cube is the TORSO-TILT geometry-corruption signature (observed again 07-18 14:24)
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-005 · sector_idx=0 trackless rule (no staircase can ever start)

- **Location (legacy):** 1399-1404 (latch), 1651-1653 (hold), 2296 (retract branch inert), 2115-2126 (carry goes straight to carried)
- **Protects against:** A joint staircase launched in side-home would walk the audited front/side/rear chain from a side posture it was never validated from — configuration-basin jump / cross-body sweep (the 07-15 hardware jump class)
- **Trigger:** Any reach latch, hold registration, dyn-retract, or grasped carry while side_home
- **Response:** target_idx/sector_idx forced 0 at every site: _reach_tm stays None (no outbound track), the 'home and sector_idx>0' retract-track branch can never fire (retract = home glide instead), and a grasped hold enters carried directly (Cartesian ramp to the side home) — NO track machinery can ever start
- **Constants:** SECTOR_IDX forced to 0; STAGE_RATE/STAGE_JOINTS machinery never engaged
- **State:** _reach_tm, h['sector_idx'], h['track']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_side_home_basic ('ZERO joint-staircase ticks'), t_side_home_dual_any_sector
- **Incident:** Offline mink study 07-18: every side-home leg is continuous to front/side/rear-ish with no basin jumps, so staircases are unnecessary AND unvalidated from the side
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-006 · Lift-no-arc branch (straight ramp IS the clearance sweep)

- **Location (legacy):** 2144-2176 (no-arc at 2149-2154; envelope guard 2147; timeout 2078-2083)
- **Protects against:** Carrying straight from the grasp pose catches the gripper/cube on the cube STAND (07-17 incident); in side-home the classic sector-station arc is meaningless and could sweep the wrong way
- **Trigger:** Fresh grasp_detected=True on a side-home hold
- **Response:** wp1 = target + 5 cm straight up, envelope-guarded by _target_safe (unsafe wp1 => lift skipped entirely, carry from the grasp pose); _lift_arc is NOT called (no horizontal arc waypoints); the subsequent gentle ramp home to SIDE_HOME_EE is the clearance sweep; whole clearance move budgeted at 8 s then carries anyway (loud warning)
- **Constants:** GRASP_LIFT_UP_M=0.05; GRASP_LIFT_OUT_M=0.08 (arc SKIPPED in side-home); GRASP_LIFT_TIMEOUT_S=8.0
- **State:** h['grasp']='lifting', h['lift_wps'], h['lift_t0'], h['restream'], h['bias'] cleared (2141-2143)
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_side_home_basic (saw_lift + carried at side home; arc absence not directly asserted)
- **Incident:** User 07-18 spec: lift a little, then slowly back to the side — the 8 cm/s straight ramp to the side home provides the outward clearance the arc used to
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-007 · BEARING_JUMP via-home rule, site 1: _live_retrack (in-flight reach)

- **Location (legacy):** 1624-1628 (surviving checks 1611-1622)
- **Protects against:** An in-flight (extended) arm chasing a cube whose bearing teleported (front->rear) would slide the Cartesian target across the torso mid-reach — the cross-body strike class
- **Trigger:** Live retrack during REACH (primary or dual-parallel secondary): |bearing(new standoff) - bearing(current target)| > 45 deg while side_home
- **Response:** Return None — the target stays FROZEN at the old point; the leg finishes/times out there and the HOLD machinery afterwards routes via the side home. Note in side-home the _sector_left clamp at 1621 is bypassed and this bearing check is its replacement; freshness, midline half-space, and safety-envelope checks remain
- **Constants:** BEARING_JUMP_DEG=45.0; DYN_LOST_GRACE_S=1.0; CUBE_LATCH_WINDOW_S=0.7; DYN_RETRACK_M=0.01
- **State:** _reach_target / _sec['target'] (left unchanged)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (t_inflight_sector_clamp covers only the classic-mode sector clamp)
- **Incident:** User 07-18 hard requirement: the side home (bearing 90 deg) IS the stage; every big transition routes through it, nothing crosses the body directly
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-008 · BEARING_JUMP via-home rule, site 2: Guard-1 held-arm follow

- **Location (legacy):** 1956-1965 (sector-clamp bypass 1956-1957; bearing check 1961-1965)
- **Protects against:** While another arm's GO/REACH owns the tick (retreats/releases suspended per Guard 1), a held arm following a >45-deg bearing jump would slide its published target across the body with no recovery machinery running
- **Trigger:** Phase GO/REACH, side_home held-arm follow computes |bearing(live standoff) - bearing(h['target'])| > 45 deg
- **Response:** Follow is skipped (pass) — no target slide; the situation is handled safely in HOLD (site 3). Surviving checks on this path: grasp-state gate (only None/'failed' follow), track None, freshness, _cube_servable (midline + reachability), DYN_RETRACK_M deadband
- **Constants:** BEARING_JUMP_DEG=45.0; DYN_RETRACK_M=0.01
- **State:** h['target'] (left unchanged)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Descendant of the 07-15 split-verdict torso-strike (Cartesian-vs-sector gap); in side-home the _sector_left clamp is bypassed so the bearing check is the sole cross-body guard on this path
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-009 · BEARING_JUMP via-home rule, site 3: classic HOLD follow — via_home latch + measured at_home gate

- **Location (legacy):** 2218-2269 (at_home 2224-2229; jump 2230-2233; via_block 2234-2235; latch clear 2238-2241; latch set 2265-2269)
- **Protects against:** Cube teleports front->rear on a holding arm: a direct chase crosses the body; separately, a mid-retract sighting could shortcut the detour and re-extend from halfway (never physically at the stage)
- **Trigger:** HOLD-phase follow with side_home: |bearing(standoff(live)) - bearing(h['target'])| > 45 deg, OR h['via_home'] already latched, while measured EE is NOT within 0.10 m of rest_ee[arm]
- **Response:** via_block forces h['home']=True — the arm home-glides to the side home; h['via_home'] LATCHES on first jump detection so mid-retract sightings cannot shortcut; re-extend permitted only when at_home (MEASURED _ee_act present, finite, <0.10 m from rest_ee — missing/NaN EE keeps the block, fail closed); latch cleared at 2238-2241 on the from-the-stage re-extend, which re-seeds a gentle restream ramp (2242-2249)
- **Constants:** BEARING_JUMP_DEG=45.0; at_home radius 0.10 m (hardcoded, distinct from HOME_SETTLE_TOL_M=0.06)
- **State:** h['via_home'], h['home'], h['restream'], _ee_act
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_via_home_routing (asserts the arm physically passes through the side home, then re-extends rear)
- **Incident:** User 07-18 hard requirement: retract to the SIDE home FIRST and re-extend ONLY once the measured EE is physically AT home
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-010 · Home glide — gentle 8 cm/s ramp seeded from measured EE

- **Location (legacy):** 2370-2393 (via _gentle_approach 1420-1435)
- **Protects against:** Publishing rest_ee RAW lets the robot-side IK sweep the whole distance at its 0.25 m/s cap — an invisible 8 cm hop in front mode but a violent ~40 cm sweep to the side home
- **Trigger:** Any idle or dyn-retracted arm homing (idle REST and the via-home/unseen retract)
- **Response:** Home target is ramped from the arm's MEASURED EE at 8 cm/s; returns None until telemetry provides a seed — caller goes stream-silent so the robot holds pose (fail closed, never seeds at the goal); _home_stream[arm] resets whenever the slot is owned by anything other than home (2455-2457) so every fresh homing re-seeds from where the arm actually is
- **Constants:** STAGE_REACH_EE_RATE=0.08 m/s; robot-side IK cap 0.25 m/s (the hazard being avoided); announce threshold 0.03 m
- **State:** _home_stream[arm], _ee_act
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_startup_lift (front), t_side_home_basic (side, per-tick step <= rate*DT), t_glide_no_mans_land
- **Incident:** Hardware 07-18: 'arms shot out to the sides at startup' when the side home was published raw
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-011 · No-man's-land stream exemption — active glide keeps publishing

- **Location (legacy):** 2356-2368 (_idle_arms_home OR-clause), 2435-2449 (_arm_packet idle-slot logic), stranded-arm hold 2450-2454
- **Protects against:** Mid-glide the joints traverse the no-man's-land between the power-on and side-home postures where the near-home check is False for EVERY reference — gating the keep-alive on it freezes the glide short of home forever
- **Trigger:** Idle arm slot whose _arm_near_front is False but whose _home_stream[arm] is not None (glide in progress)
- **Response:** The ACTIVE glide stream keeps publishing to completion — the glide itself is the safety mechanism (measured-EE seed, 8 cm/s, straight line). A STRANDED arm (no stream) is still never pulled home: it is held in place at its measured EE, and a missing/non-finite EE makes the whole packet None (stream silence, robot failsafe holds) — fail closed
- **Constants:** FRONT_HOME_TOL_RAD=0.35 (the check being exempted); STAGE_REACH_EE_RATE=0.08
- **State:** _home_stream, _ee_act, _jpos
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_glide_no_mans_land
- **Incident:** Hardware 07-18: glide FROZE 8 cm out, pos_target stuck at [0.27, 0.29], arms never reached the side
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-012 · _arm_near_front accepts SIDE_HOME_JOINTS + fail-closed telemetry

- **Location (legacy):** 2338-2354 (side-home addition 2350-2352)
- **Protects against:** NaN > x is False, so a plain comparison would pass a garbage posture as 'home' (07-15 audit fail-open class) and authorize a Cartesian REST publish from an unknown posture; without SIDE_HOME_JOINTS in the list an arm resting at the side would never be recognized as home and never re-glide
- **Trigger:** Every idle-arm slot decision (_arm_packet) and _idle_arms_home
- **Response:** Returns False (fail CLOSED) on missing/short (<31) /non-finite jpos; posture accepted iff within 0.35 rad (max-abs, per joint, mirrored for right) of ANY known home — side-home adds SIDE_HOME_JOINTS as the third reference
- **Constants:** FRONT_HOME_TOL_RAD=0.35; homes = STAGE_JOINTS['front'], POWERON_JOINTS, + SIDE_HOME_JOINTS (side-home only)
- **State:** _jpos
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (exercised indirectly by t_glide_no_mans_land)
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-013 · Startup posture gate does NOT accept the side-home posture

- **Location (legacy):** 2498-2517 (references FRONT_HOME_TOL_RAD 213-217)
- **Protects against:** An operator restarted while an arm is parked away from the front/power-on basins would command a cross-basin Cartesian reach with no staircase protecting it
- **Trigger:** First finite arm telemetry after operator launch with either arm >0.35 rad from both the front station and the power-on pose
- **Response:** raise SystemExit('STARTUP REFUSED: arm(s) not at the power-on pose...'); until the check PASSES the mission publishes gaze only, nav zero, NO arm packet (2539-2551) — the robot's arm-silence failsafe holds/crawls
- **Constants:** FRONT_HOME_TOL_RAD=0.35; accepted homes at startup: STAGE_JOINTS['front'] + POWERON_JOINTS ONLY (SIDE_HOME_JOINTS deliberately absent)
- **State:** _startup_posture_ok
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none for the side-home-parked restart case (t_tilt_gate covers the adjacent tilt refusal)
- **Incident:** Interplay note: SIDE_HOME_JOINTS is a valid RUNTIME home (_arm_near_front) but a restart with arms parked at the side home is STARTUP REFUSED — home the arm first (power-cycle / failsafe crawl / humanoid_stage_walk_test.py)
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-014 · Dual-parallel any-sector enablement (trackless precondition)

- **Location (legacy):** 1679-1683 (front-only bypass), surviving checks 1674-1721; secondary hold sector 1757 and 1774-1777
- **Protects against:** A parallel secondary latched under a joint-staircase primary would put Cartesian and joint payloads in modal-wire conflict (the 07-15 wire contract); an un-checked secondary could also latch a torso/cross-midline cube
- **Trigger:** _try_latch_secondary with side_home: primary trackless requirement '(side_home or front)' is satisfied for ANY sector
- **Response:** Parallel reaches allowed for any sector mix because EVERY side-home leg is trackless (_reach_tm None by the sector_idx=0 rule). Surviving guards: hold_after_reach required (1674), no held-arm track owns the wire (1684), confirmed sighting, off-midline on the free arm's side, safety envelope, _reachable_now, median-of-window re-checks, GATE_CONSECUTIVE latch debounce (1711-1720). The secondary's hold registers its TRUE sector via _sector(tgt) (not 'front') with sector_idx still 0
- **Constants:** GATE_CONSECUTIVE debounce; MIDLINE_MARGIN_M=0.03; CUBE_LATCH_WINDOW_S=0.7
- **State:** _sec, _sec_hits, _reach_tm
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_side_home_dual_any_sector (SIDE primary + FRONT secondary, zero joint ticks)
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants

### SAFE-SIDEHOME-015 · side_home CLI wiring and gaze-only override

- **Location (legacy):** 2750-2756, 2778
- **Protects against:** Gaze-only missions must never engage arm homing/reaching machinery; operators must know which profile (side-home vs classic staging) is live before powering the arms
- **Trigger:** CLI parse at launch
- **Response:** side_home defaults ON (07-18 mission profile) but is forced OFF under --gaze-only; the active profile is printed loudly at startup (2786-2791); --no-side-home restores the classic three-sector staging with the front chest home
- **Constants:** Args.side_home default True; op side_home = args.side_home and not args.gaze_only
- **State:** op.side_home
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** auto_operator/arms.py (via-home, lift-no-arc) + arbitration.py (home glide) + mission.py (sector bypasses) + operator constants


## Command-wire arbitration

### SAFE-ARBITRATION-001 · Packet priority chain (ret_payload > staircase suppression > phase packet > keep-alive)

- **Location (legacy):** 2635-2664
- **Protects against:** Two writers on the single 9874 arm stream: a phase Cartesian packet overriding an in-flight joint staircase would flip real_env's wire mode and let re-seeded IK drag a far-basin arm across the torso (07-15 hardware cross-body jump class)
- **Trigger:** Every tick() after phase handlers: ret_payload from _held_update, arm_packet['arm_targets'] from the phase, staircase_active computed
- **Response:** Strict elif chain: (1) ret_payload (held-arm joint staircase) unconditionally replaces arm_targets; (2) else if staircase_active and packet is not joint (at is None or 'joint_pos' not in at) -> arm_packet.pop('arm_targets') = silence; (3) else if at is None -> keep-alive REST only when _idle_arms_home(jpos); note the elif ordering means keep-alive can NEVER fire during a staircase blip
- **Constants:** staircase_active = (_reach_tm is not None) or any(h['track'] is not None)
- **State:** self._reach_tm, self._held[arm]['track'], arm_packet['arm_targets']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized, t_grasp_carry_side (indirect via joint_ticks/cart_ticks accounting)
- **Incident:** 07-15 hardware: dual-arm interleave let one stream override the other mid-basin; real_env re-engages IK on the last target after 0.5 s of arm-silence
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-002 · staircase_active Cartesian suppression

- **Location (legacy):** 2643-2649
- **Protects against:** A Cartesian packet published mid-staircase flips the robot to Cartesian mode; IK re-seeds from encoders (real_env 8335075) and sweeps a side/rear-basin arm toward REST through the torso
- **Trigger:** Any tick where _reach_tm is not None OR any held h['track'] is not None, and the phase produced a non-joint (or no) payload
- **Response:** arm_packet.pop('arm_targets', None): stream stays in JOINT mode or goes silent; real_env holds its last target on silence
- **Constants:** STAGE_RATE=0.2 rad/s (the joint payload rate being protected)
- **State:** _reach_tm, _held[*]['track']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side ('carry used the joint staircase'), t_dual_serialized (indirect)
- **Incident:** 07-15 modal-wire contract audit: joint tracks and Cartesian reaches are mutually exclusive on the global 14-joint stream
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-003 · Silence-means-hold contract

- **Location (legacy):** 2400-2404, 2420, 2446-2447, 2452-2453, 2639-2642, 1987-1988
- **Protects against:** Publishing a fabricated/unsafe slot value on a telemetry blip; conversely, expecting a robot-side crawl-home failsafe that does not exist would strand assumptions
- **Trigger:** Any slot in _arm_packet with no safe value (no EE seed, non-finite _ee_act, off-front idle arm with no glide), or _step_toward returning payload None on a telemetry blip
- **Response:** Return None / continue -> tick() publishes NO arm_targets; real_env HOLDS ITS LAST TARGET (07-15 contract audit: with --use-ik the crawl-home-on-silence path is dead code) — position held, nothing moves
- **Constants:** none (contract, not a threshold)
- **State:** _ee_act, _home_stream
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (no test asserts hold-on-silence directly; mk_rig fakes perfect tracking)
- **Incident:** 07-15 contract audit found the --use-ik retract-on-silence failsafe is dead code; shutdown protocol relies on holds releasing before exit
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-004 · _arm_engaged semantics (never-go-silent latch)

- **Location (legacy):** 504-511, 2663-2664
- **Protects against:** Arm-silence after a staircase: real_env keeps the last Cartesian target forever and re-engages IK on it after 0.5 s of silence, snapping the arm back to a stale cube target
- **Trigger:** First tick where any arm_targets payload is published
- **Response:** Set self._arm_engaged = True. NOTE: currently WRITE-ONLY (set at 2664, never read) — the invariant it documented is now enforced by the 07-17 keep-alive-from-mission-start plus the silence-means-hold contract; a refactor must not resurrect silence-gating on this flag
- **Constants:** real_env arm-silence re-engage window ~0.5 s
- **State:** self._arm_engaged
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Hardware 07-14: silence after a staircase snapped the arm back to a stale cube target
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-005 · Keep-alive REST gated by _idle_arms_home (Guard against Cartesian drag-home)

- **Location (legacy):** 2650-2662, 2356-2368
- **Protects against:** An arm stranded in the side/rear basin (timed-out retreat) pulled home ACROSS the torso by a single un-staged Cartesian REST packet
- **Trigger:** Tick with no ret_payload, no staircase, no phase packet (at is None)
- **Response:** keep_alive = _arm_packet(None) ONLY if _idle_arms_home(jpos): every unheld arm must verifiably sit near a KNOWN home posture OR have an active home glide; the dual-parallel secondary's arm is exempt from the check (its slot is stream-owned). Otherwise stay silent -> real_env holds pose. Fires from MISSION START (07-17 user spec) so idle arms rise from power-on to RAISED REST (+0.18 z) right after startup checks
- **Constants:** REST_EE z=0.18 (raised), FRONT_HOME_TOL_RAD=0.35
- **State:** _held, _home_stream, _sec, _jpos
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_startup_lift
- **Incident:** 07-17 user spec moved keep-alive to mission start; original rule: side/rear arm must never be pulled home by a Cartesian REST
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-006 · Home-glide exemption in the idle gate (no-man's-land freeze fix)

- **Location (legacy):** 2358-2368, 2435-2449
- **Protects against:** Home glide freezing partway: mid-glide joints sit between power-on and side-home postures where _arm_near_front is False for every reference, so posture-gating the keep-alive stopped publication and froze the arm 8 cm out
- **Trigger:** Idle arm with _home_stream[arm] is not None (an ACTIVE glide) but posture not near any home reference
- **Response:** The active glide counts as 'home' in _idle_arms_home and keeps publishing to completion in _arm_packet; the glide ITSELF is the safety mechanism (seeded from measured EE, 8 cm/s straight line). A STRANDED arm (stream None) is still held in place at measured EE, never glided blind
- **Constants:** STAGE_REACH_EE_RATE=0.08 m/s
- **State:** _home_stream[arm]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_glide_no_mans_land
- **Incident:** Hardware 07-18: pos_target stuck at [0.27, 0.29], arms never reached the side home
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-007 · _arm_near_front fail-closed posture verification

- **Location (legacy):** 2338-2354
- **Protects against:** NaN/garbage joint telemetry passing as 'arm is home' (NaN > x is False, so a plain tolerance comparison FAILS OPEN) and authorizing a Cartesian REST at a stranded arm
- **Trigger:** Every _idle_arms_home / per-arm REST-gate evaluation
- **Response:** Return False (NOT home) when jpos is None, len < 31, or any non-finite value in the arm's 7-joint slice; else max-abs error against STAGE_JOINTS['front'], POWERON_JOINTS, and (side-home mode) SIDE_HOME_JOINTS must be <= FRONT_HOME_TOL_RAD
- **Constants:** FRONT_HOME_TOL_RAD=0.35 rad (gravity sag ~5 deg at 0.8 torque + 9.2 deg power-on->front offset)
- **State:** jpos slices 13:20 (left) / 20:27 (right, sign-mirrored)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (no NaN-telemetry test)
- **Incident:** 07-15 audit fail-open class: NaN comparisons silently pass guard checks
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-008 · _home_glide gentle ramp, seed-or-silence

- **Location (legacy):** 2370-2393
- **Protects against:** Publishing rest_ee RAW lets robot-side IK sweep the whole distance at its 0.25 m/s cap — a violent ~40 cm sweep to the side home
- **Trigger:** Any homing publication (idle keep-alive, held-arm home slot)
- **Response:** Ramp the published target from the MEASURED EE toward rest_ee at STAGE_REACH_EE_RATE via _gentle_approach; returns None (caller goes silent, robot holds pose) until telemetry provides an EE seed — never seeds the ramp at the goal
- **Constants:** STAGE_REACH_EE_RATE=0.08 m/s; per-tick step = 0.08*DT (DT=1/20 s -> 4 mm/tick); glide announcement threshold 0.03 m
- **State:** _home_stream[arm], _ee_act[arm]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_startup_lift (max_step <= STAGE_REACH_EE_RATE*DT), t_side_home_basic
- **Incident:** Hardware 07-18: 'arms shot out to the sides at startup' from raw REST publish
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-009 · Per-arm slot ownership in _arm_packet

- **Location (legacy):** 2395-2458
- **Protects against:** An idle arm caught off-front (stranded retreat/abort) dragged home in one un-staged Cartesian packet — 07-15 audit found phase-handler packets used to BYPASS the _idle_arms_home guard
- **Trigger:** Every _arm_packet build (reach, hold, park, keep-alive)
- **Response:** Slot precedence per arm: (1) holding non-home arm -> restream ramp point else bias-corrected h['target']; (2) holding home arm -> _home_glide (None seed -> whole packet None = silence); (3) reaching arm overlay reach_pos; (4) secondary stream injection; (5) idle arm near-front-or-gliding -> _home_glide; (6) else HOLD IN PLACE at measured _ee_act (finite-checked; non-finite -> whole packet None). No safe value for ANY slot silences the ENTIRE packet
- **Constants:** NEUTRAL_QUAT orientation (yaw-to-face shelved in af97f1f)
- **State:** _held[arm]['home'/'restream'/'target'/'bias'], _reach_arm, _sec['stream'], _ee_act, _home_stream
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_front ('packet carried BOTH live reach slots'), t_grasp_carry_side (park at REST)
- **Incident:** 07-15 audit: phase packets bypassed the idle-home guard; fix centralizes all slot safety here
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-010 · Secondary slot injection never overrides an owned slot; unseeded leaves slot idle

- **Location (legacy):** 2425-2431
- **Protects against:** The dual-parallel secondary stream clobbering a held/reaching arm's slot, or an unseeded secondary implying motion that never happened
- **Trigger:** _sec is not None and _sec['stream'] is not None during any packet build
- **Response:** Inject _sec['stream'] into the free arm's slot ONLY if ee_pos[j] is None (slot unowned); an unseeded secondary (stream None) leaves the slot to the idle logic — the arm has not moved yet. Paired: the unseeded-timeout ABANDON at 1760-1770 refuses to register a hold whose raw target would snap REST->cube in one packet (07-16 catapult class)
- **Constants:** REACH_TIMEOUT_S=8.0 (abandon clock)
- **State:** _sec['arm'/'stream']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_unseeded_abandon ('right slot never jumped', max_jump < 0.02 m)
- **Incident:** 07-16 catapult class: full REST-to-cube target in one packet; 07-17 audit added the unseeded-abandon
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-011 · Glided-reset of _home_stream (re-seed on ownership change)

- **Location (legacy):** 2455-2457
- **Protects against:** A stale glide ramp resuming from an old point after the arm's slot was owned by reach/hold/carry — the published target would jump from wherever the old ramp left off, not from where the arm now is
- **Trigger:** End of every _arm_packet build, for each arm NOT in the glided set this tick
- **Response:** self._home_stream[arm] = None so the next homing re-seeds from the MEASURED EE (fresh _gentle_approach seed)
- **Constants:** none
- **State:** _home_stream, glided set
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Documented in _home_glide docstring: ramp state resets whenever the slot is owned by anything other than home
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-012 · Guard-1: GO/REACH freeze of held arms (+ sector-clamped follow, via-home bearing clamp, grab exemption)

- **Location (legacy):** 1928-1976
- **Protects against:** A held arm starting/advancing a JOINT retreat while another arm reaches: the global 14-joint packet overrides the reaching arm's stream and flings a mid-basin arm across the torso (07-15 hardware jump); also the split-verdict torso-strike where an unclamped Cartesian follow slides h['target'] across the sector boundary un-staged while the arm is frozen at a stale snapshot
- **Trigger:** self.phase in (Phase.GO, Phase.REACH) with any arm in _held
- **Response:** continue before all track/release logic: NO retreat/turnaround/release starts (those run only in HOLD/SEARCH). Cartesian follow allowed ONLY if grasp in (None,'failed'), track is None, cube fresh within DYN_LOST_GRACE_S, and cube still inside h['sector'] (_sector_left clamp; side-home: bearing delta vs h['target'] <= BEARING_JUMP_DEG or the slide is deferred to HOLD's via-home). Unservable cube -> HOLD the last target (never retract to REST). _maybe_send_grab still fires (gripper rides its own serial bus, not the 14-joint stream — 07-17 dual-parallel audit)
- **Constants:** DYN_LOST_GRACE_S=1.0, DYN_RETRACK_M=0.01, BEARING_JUMP_DEG=45.0
- **State:** _held[arm]['grasp'/'track'/'sector'/'target'], self.phase
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized ('A never ran a track during B's GO/REACH (Guard 1)'); sector-clamp-of-held-follow facet: none (t_inflight_sector_clamp covers only the reach retrack)
- **Incident:** 07-15 hardware cross-body jump (facet 2: unclamped GO/REACH follow); 07-15 adversarial audit's split-verdict torso-strike is a Cartesian-vs-sector gap the joint-vs-Cartesian guards do NOT cover
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-013 · Guard-2: SEARCH gate on in-flight held tracks

- **Location (legacy):** 1234-1240
- **Protects against:** Starting a new reach leg while a held arm is mid-track: reaches and joint tracks both drive the global joint stream, so running them together lets one override the other (the 07-15 jump)
- **Trigger:** _tick_search with any(h['track'] is not None for h in _held.values())
- **Response:** return — no new leg starts; the in-flight track finishes, lands in HOLD, and SEARCH re-enters
- **Constants:** none
- **State:** _held[*]['track']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized (indirect — no direct assertion that SEARCH deferred)
- **Incident:** 07-15 crossbody-jump serialization: GUARD 2 in _tick_search pairs with GUARD 1 in _held_update
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-014 · One-track token + frozen-arm snapshot

- **Location (legacy):** 1981-1984, 1410-1418
- **Protects against:** Two joint staircases (or a staircase plus REACH staging) interleaving on the global 14-joint stream — the ret_payload interleave facet of the 07-15 dual-arm torso strike
- **Trigger:** _held_update iterating a second arm whose track is pending while ret_payload is already set this tick OR _reach_tm is not None (REACH staging has priority)
- **Response:** continue: ONE track at a time; the waiting arm's 7 joints stay frozen at the staircase-start snapshot (tm['frozen']) INSIDE the active track's _staged_joint_payload — the joint schema is global, so a Cartesian hold cannot ride the same packet; the holding arm freezes for the staircase's duration and resumes when the stream returns to Cartesian
- **Constants:** STAGE_RATE=0.2 rad/s
- **State:** ret_payload, _reach_tm, tm['frozen']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized ('first grasped carried home first' ordering)
- **Incident:** 0519d3c fixed the ret_payload interleave (audited airtight + lagged repro per crossbody-jump memo)
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-015 · Lifting-ramp pause while the joint stream owns the wire

- **Location (legacy):** 2084-2090
- **Protects against:** The lift's Cartesian ramp advancing while unpublished (another arm's staircase owns the wire): on resume the first published point would be far from the frozen arm — a resumed jump
- **Trigger:** h['grasp']=='lifting' while _reach_tm is not None or any other h['track'] is not None
- **Response:** continue without advancing h['restream'] — the ramp holds its point so it never runs ahead of the frozen arm
- **Constants:** GRASP_LIFT_TIMEOUT_S=8.0 (whole clearance budget still ticking)
- **State:** h['restream'], h['lift_t0']
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_serialized (indirect: A's lift/carry defer during B's reach)
- **Incident:** Comment: 'pause the ramp so it never runs ahead of the frozen arm (a resumed jump otherwise)'
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-016 · _sec block: SEARCH return (no new leg / no base motion under a latched secondary)

- **Location (legacy):** 1293-1296
- **Protects against:** Starting a third leg or walking the base while a dual-parallel Cartesian reach target is latched — base motion invalidates the body-frame target mid-flight
- **Trigger:** _tick_search with self._sec is not None
- **Response:** return — no new leg starts and (with walk enabled) absolutely no base motion until the secondary resolves (hold registered or abandoned)
- **Constants:** none
- **State:** _sec
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_front (indirect)
- **Incident:** 07-17 dual-parallel design rule
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-017 · _sec block: carry gate (joint staircase waits out the dual-parallel reach)

- **Location (legacy):** 2115-2121
- **Protects against:** A grasped arm's carry staircase (JOINT) starting while the secondary's Cartesian reach is in flight — the joint packet would freeze/override the flying arm's stream
- **Trigger:** h['grasp']=='grasped', h['sector_idx']>0, h['track'] is None, _reach_tm is None, but _sec is not None
- **Response:** Do NOT create the carry track (h stays frozen at its last target); track starts only once self._sec is None
- **Constants:** none
- **State:** _sec, h['sector_idx'], h['track'], _reach_tm
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_front (indirect: both end carried after sec resolves)
- **Incident:** Guard-1/one-track extension for dual-parallel (07-17)
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-018 · _sec block: retreat gate (belt-and-suspenders)

- **Location (legacy):** 2296-2299
- **Protects against:** A side/rear hold's retract staircase starting under a dual-parallel Cartesian reach — same joint-vs-Cartesian wire conflict
- **Trigger:** Hold transitions to 'retracted' (h['home'] True) with h['sector_idx']>0 while _sec is not None
- **Response:** Skip creating the retreat track this transition; comment notes side/rear holds cannot coexist with a dual-parallel reach BY CONSTRUCTION (secondary latches only under trackless front/side-home primaries), so the gate is belt-and-suspenders
- **Constants:** none
- **State:** _sec, h['sector_idx'], h['home']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-17 dual-parallel audit hardening
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-019 · Secondary latch wire-mode gates + GATE_CONSECUTIVE debounce

- **Location (legacy):** 1674-1685, 1711-1721
- **Protects against:** A secondary latched under a joint-mode primary (side/rear staircase) would put a Cartesian slot on a JOINT-mode wire; a single chattering learned-gate verdict would commit an arm to an unreachable cube
- **Trigger:** _try_latch_secondary each REACH tick with _sec None
- **Response:** Refuse latch unless: hold_after_reach (legacy park mode would strand HOLD forever, 07-17 audit); primary trackless (_reach_tm is None and sector=='front' unless side_home); no held arm mid-track (joint stream owns the wire); free arm unheld. Candidate must pass confirmed + off-midline (MIDLINE_MARGIN_M) + front-sector + _target_safe + _reachable_now on BOTH the live pos and the CUBE_LATCH_WINDOW_S median, then survive a GATE_CONSECUTIVE streak (_sec_hits debounce; broken streak resets)
- **Constants:** GATE_CONSECUTIVE=5, MIDLINE_MARGIN_M=0.03, CUBE_LATCH_WINDOW_S=0.7
- **State:** _sec_hits=[cube_key, streak], _sec, _reach_tm, _reach_sector
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_side_gate, t_dual_parallel_off
- **Incident:** 07-17 audit: learned ReachabilityGate samples fresh random candidates per call so single verdicts chatter; 07-15 modal-wire contract untouched (staircases never parallelize)
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-020 · ee_action side-channel: repeat-send with rotation, no arm-stream interference

- **Location (legacy):** 2666-2701
- **Protects against:** One-shot gripper commands silently eaten by a real_env stall or a mid-reconnect 9874 subscriber (recurring 'left gripper never opened'); or gripper traffic perturbing the joint/Cartesian arbitration
- **Trigger:** Any queued ee_action (zero_gripper, hand_grab, hand_open) on the latest-value-wins no-ack channel
- **Response:** Re-send each command for EE_ACTION_REPEAT_TICKS packets; after >=5 sends with a newer command waiting, YIELD but ROTATE the unsent remainder to the back of the queue (never drop; a queued same-side newer command supersedes the remainder). Repeats safe: service rejects a busy side, hand_open idempotent. ee_action does NOT refresh real_env's arm-stream clock, so it never interferes with the joint/Cartesian arbitration
- **Constants:** EE_ACTION_REPEAT_TICKS=30 (1.5 s at 20 Hz); yield threshold 5 sends (0.25 s burst)
- **State:** _ee_queue, _ee_current=[cmd, remaining, sent_count]
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_repeat_send (full budget both sides, >=2 s spread, no _ticks leak)
- **Incident:** 07-16 audit: a single real_env stall ate one-shots; 07-17: dropping the remainder starved the left startup zero to one 0.25 s burst missed by a reconnecting subscriber
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order

### SAFE-ARBITRATION-021 · Startup fail-closed branch: no arm packet until posture check PASSES

- **Location (legacy):** 2539-2551 (check itself 2479-2517)
- **Protects against:** An operator relaunched against a wrong --robot-ip, or before the arm homed, driving a blind cross-basin Cartesian reach with no staircase protection; NaN posture would fail-open without the isfinite gate (NaN > x is False)
- **Trigger:** Full-mission tick while _startup_posture_ok is False (needs telemetry to run: both arms within FRONT_HOME_TOL_RAD of STAGE_JOINTS['front'] or POWERON_JOINTS, tilt <= TILT_REFUSE_DEG)
- **Response:** Publish gaze only, nav ramped to zero, NO arm packet — the robot's arm-silence behavior holds; a tilted torso (> TILT_REFUSE_DEG) raises SystemExit outright (tilt corrupts every body-frame target: 07-16/18 incident class, RULE #0 legs straight)
- **Constants:** TILT_REFUSE_DEG=15.0, TILT_WARN_DEG=8.0, FRONT_HOME_TOL_RAD=0.35
- **State:** _startup_posture_ok, _gaze_seeded, _vx, _wz
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_tilt_gate (30-deg refused, 5-deg passes); posture-refusal branch: none
- **Incident:** 07-16/18: bent-leg hang -> torso tilt -> geometry corruption (3 failed sessions' root cause); startup refusal added with RULE #0
- **New location (S6):** auto_operator/arbitration.py (build_arm_packet/arbitrate/attach_ee_action/home_glide) + operator tick() dispatch order


## Held-arm dynamics

### SAFE-HELD-001 · Guard 1: held-arm serialization during GO/REACH

- **Location (legacy):** 1928-1976
- **Protects against:** A held arm starting/advancing a joint retreat while another arm reaches would put a global 14-joint packet on the wire, overriding the reaching arm's stream and flinging a mid-basin arm across the torso (07-15 hardware cross-body jump).
- **Trigger:** self.phase in (Phase.GO, Phase.REACH) while any arm is in self._held
- **Response:** Held arm may only Cartesian-FOLLOW its cube in-sector (in-basin, safe); it must never start or advance a joint track; retreats/turnarounds/releases run only in HOLD/SEARCH. If the cube is not servable right now, HOLD the last target (never retract to REST — that itself would command a side/rear arm toward front). _maybe_send_grab may still fire (gripper rides its own serial bus, not the 14-joint stream).
- **Constants:** DYN_LOST_GRACE_S=1.0 (freshness on the follow)
- **State:** self.phase, h["track"], h["grasp"], h["target"], h["sector"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized ("dual: A never ran a track during B's GO/REACH (Guard 1)")
- **Incident:** 07-15 hardware: dual-arm interleave swept an arm front-to-rear through the torso; fixed in 0519d3c and audited with lagged repro (40 override ticks pre-fix, 0 post).
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-002 · Sector-clamp on the held-arm follow during GO/REACH

- **Location (legacy):** 1937-1968 (predicate _sector_left 949-957)
- **Protects against:** Following a cube that drifted out of h["sector"] would slide h["target"] across the front/side/rear basin boundary UN-staged; because the reaching arm freezes this arm at a one-shot encoder snapshot for the whole staircase, the single Cartesian packet at staging-completion would command an un-staged cross-sector move from a stale basin — a torso strike.
- **Trigger:** Held cube drifts past its sector's bearing band by more than SECTOR_HYST_DEG while another arm is in GO/REACH
- **Response:** Follow refused — target stays frozen; the out-of-sector cube is handled the safe way after the reach ends (HOLD's _sector_left release re-queues it for a fresh staged leg). In side_home mode the sector test is bypassed but a >BEARING_JUMP_DEG bearing jump is refused instead (via-home handled in HOLD).
- **Constants:** SECTOR_HYST_DEG=10.0, SECTOR_FRONT_DEG=60.0, SECTOR_REAR_DEG=120.0, BEARING_JUMP_DEG=45.0
- **State:** h["sector"], h["target"], self.side_home
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (in-flight-reach analogue covered by t_inflight_sector_clamp; held-follow clamp itself only in scratchpad/test_sector_clamp.py, outside this suite)
- **Incident:** 07-15 adversarial audit: the split-verdict torso-strike is a Cartesian-vs-sector gap the joint-vs-Cartesian serialization guards do NOT cover.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-003 · Follow deadband + grasp-settle restart

- **Location (legacy):** 1966-1968, 2252-2255 (const 183-184)
- **Protects against:** Detection jitter continuously wiggling an extended arm at the cube, and a grab firing on a still-moving target.
- **Trigger:** Live standoff point vs h["target"] on every servable tick
- **Response:** Target updates only when the cube moved > DYN_RETRACK_M (1 cm); on a real move the grasp settle timer restarts (h["grasp_move_t"]=now) so _maybe_send_grab waits a fresh GRASP_SETTLE_S.
- **Constants:** DYN_RETRACK_M=0.01, GRASP_SETTLE_S=1.0
- **State:** h["target"], h["grasp_move_t"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_retry_then_fail ("failed-grasp hold still dyn-follows") — follow path only; the 1 cm threshold itself is untested
- **Incident:** Deadband comment: "so detection jitter cannot wiggle the arm".
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-004 · _cube_servable gate

- **Location (legacy):** 1892-1901
- **Protects against:** Arm chasing a stale, cross-midline, or unreachable cube — arm-vs-arm collision (the IK has no reliable arm-vs-arm avoidance) or an out-of-envelope IK target.
- **Trigger:** Every held-arm decision: GO/REACH follow, settled-station re-extend / turnaround / head-home / release
- **Response:** Servable only if cube.fresh within DYN_LOST_GRACE_S AND on the arm's half-space (not past the midline by > MIDLINE_MARGIN_M) AND _reachable_now (which itself fails closed through the _target_safe envelope). Returns the sector index used for track decisions.
- **Constants:** DYN_LOST_GRACE_S=1.0, MIDLINE_MARGIN_M=0.03, TARGET_MIN_RADIUS_M=0.16, TARGET_MAX_RADIUS_M=0.45, TARGET_Z_MIN_M=-0.40, TARGET_Z_MAX_M=0.30
- **State:** cube.pos, cube freshness
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none directly (exercised implicitly by every hold test)
- **Incident:** MIDLINE_MARGIN_M comment: arms never cross the midline; user places one cube per half-space; margin is a hysteresis band so jitter at y≈0 cannot flap the verdict.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-005 · _dyn_reachable debounce + hysteresis (retract/re-reach)

- **Location (legacy):** 1117-1145
- **Protects against:** The learned gate samples fresh random candidates per call, so single verdicts chatter — undebounced, the arm would flap extend/retract every tick at the reachability boundary; stepping counters on cached repeats would let one lucky sample latch.
- **Trigger:** Per-tick HOLD follow reachability of the held cube's live position
- **Response:** Asymmetric debounce in the hold entry: retracted (h["home"]) needs in_hits >= GATE_CONSECUTIVE fresh hits to re-reach; extended needs out_hits >= GATE_CONSECUTIVE fresh misses to retract. Geometric fallback (--no-use-gate) uses hysteresis GEOMETRIC_FLOOR_M (re-reach) vs DYN_OUTREACH_M (retract), no debounce (deterministic). _target_safe False short-circuits to unreachable.
- **Constants:** GATE_CONSECUTIVE=5, GATE_EVAL_PERIOD_S=0.1 (so the streak spans ~0.5 s wall-clock), GEOMETRIC_FLOOR_M=0.28, DYN_OUTREACH_M=0.32
- **State:** h["in_hits"], h["out_hits"], h["home"], h["best_arm"]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_gate_throttle ("cached evals do NOT advance the debounce counters")
- **Incident:** 07-17 gate-throttle campaign (GATE_EVAL_PERIOD_S): unthrottled scoring starved the gaze loop into the pre-reach camera shudder; debounce counters step on FRESH samples only.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-006 · Unseen-grace retract (freshness)

- **Location (legacy):** 2209, 2279-2282 (const 178-179)
- **Protects against:** Arm parked over a vanished/occluded cube forever; or an unseen spell counting toward release and giving away a cube that was merely occluded.
- **Trigger:** Held cube unseen > DYN_LOST_GRACE_S (HOLD/SEARCH phases only — Guard 1 defers it during GO/REACH)
- **Response:** h["home"]=True (front holds glide to REST via the _arm_packet home-glide; side/rear holds launch a staged track retreat), wrong_t0 cleared — unseen time NEVER counts toward release, ownership is kept — and restream cleared so a stale ramp point cannot republish.
- **Constants:** DYN_LOST_GRACE_S=1.0
- **State:** h["home"], h["wrong_t0"], h["restream"]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (tests occlude the cube only after grasp)
- **Incident:** DYN_LOST_GRACE_S comment: robot-side EE slew 0.25 m/s IS the "slowly" of the retract; reappearing resumes the reach.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-007 · Retract-reason state transition + prints

- **Location (legacy):** 2283-2292
- **Protects against:** Silent state flips — unexplained arm behavior on hardware is the failure class the 07-16 audit was commissioned against; the transition is also the ONLY launch point of side/rear track retreats, so losing it strands the retract logic.
- **Trigger:** h["home"] flips the derived state between "tracking" and "retracted"
- **Response:** One-shot "[dyn] {arm} -> {state}" print with the precise reason: (unseen) / (crossed midline) / (left {sector} sector) / (gate favors {arm} arm) / (out of reach); h["state"] updated; on entering retracted with sector_idx>0 and no dual-parallel secondary, a track retreat to front(0) starts.
- **Constants:** none
- **State:** h["state"], h["home"], h["sector_idx"], self._sec
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Print carries the exact retract reason for hardware debugging.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-008 · Release persistence: wrong_t0 bookkeeping + HOLD_RELEASE_S + reasons

- **Location (legacy):** 2270-2278, 2301-2307; resets at 2023, 2251, 2278, 2281
- **Protects against:** Transient noise (midline jitter, one stochastic gate verdict) releasing a legitimately held cube; conversely a wrong-arm/cross-midline cube held forever would deadlock the mission.
- **Trigger:** Cube VISIBLE but unservable for a release-eligible reason: crossed the midline (wrong_side), left the hold's sector past hysteresis (non-side-home), or the gate crowns the other arm (h["best_arm"] not in (None, arm)) — continuously for > HOLD_RELEASE_S
- **Response:** wrong_t0 starts on the first such tick; after 3.0 s continuous, release with printed reason ("crossed the midline" / "left the {sector} sector" / "gate favors the {best_arm} arm"), cube.held_by=None (re-queued), del self._held[arm], and terminal Phase.HOLD rewinds to SEARCH so the freed cube is re-pursued. wrong_t0 resets whenever the cube becomes servable (2251), the reason clears (2278), the cube goes unseen (2281), or a track re-extend re-arms the hold (2023). Plain out-of-reach (best_arm None) retracts but never releases.
- **Constants:** HOLD_RELEASE_S=3.0, MIDLINE_MARGIN_M=0.03, SECTOR_HYST_DEG=10.0
- **State:** h["wrong_t0"], h["best_arm"], cube.held_by, self.phase
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** HOLD_RELEASE_S comment: persistently-unservable time before a holding arm gives its cube back to the pending queue for the correct arm.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-009 · Re-extend gentle ramp (front, in-place)

- **Location (legacy):** 2242-2261 (primitive _gentle_approach 1420-1435; clear-on-retract 2264, 2282)
- **Protects against:** Publishing the full REST-to-cube jump in one packet when a retracted hold re-extends — the 07-16 "catapult"/knock-the-cube class.
- **Trigger:** Retracted front hold (h["home"]=True, restream None) whose cube passes _dyn_reachable again
- **Response:** restream is seeded from the MEASURED EE (telemetry pos_actual) and _gentle_approach walks the published point toward the (possibly moving) target at STAGE_REACH_EE_RATE (4 mm/tick at 20 Hz); cleared when within 1e-9. CAVEAT for the refactor: if pos_actual is absent on the exact re-extend tick, h["home"] still flips False with restream None and the next packet publishes the raw target — the backstop is only the robot-side 0.25 m/s EE slew. Retract/unseen paths clear restream so a stale ramp point can never republish.
- **Constants:** STAGE_REACH_EE_RATE=0.08, DT=0.05, DYN_RETRACK_M=0.01
- **State:** h["restream"], h["home"], h["target"]
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none for hold re-extend (t_startup_lift and t_side_home_basic verify the same 8 cm/s primitive for homing)
- **Incident:** Comment: "approach at the gentle rate instead of snapping the full REST->cube jump in one packet (same catapult/knock-the-cube issue as reach)".
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-010 · Track re-decisions only at settled stations (self-check rule)

- **Location (legacy):** 1977-1980, 2012, 2053-2055
- **Protects against:** Re-deciding mid-hop would swap the joint stream for Cartesian from an intermediate posture, deviating from the audited collision-free curves (corner cutting through the torso basin).
- **Trigger:** h["track"] is not None: an in-flight retreat or turnaround
- **Response:** Mid-hop: no new decision, payload forwarded as ret_payload, joint stream never swapped for Cartesian. Only when _step_toward reports "settled"/"done" are re-extend / reverse / release / head-home evaluated (all via _cube_servable).
- **Constants:** STAGE_TOL_RAD=0.03, STAGE_DWELL_S=0.6, STAGE_RATE=0.2
- **State:** h["track"] (tm cur/target/seq/dwell_until)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none directly (t_grasp_carry_side walks the carry staircase but never interrupts it)
- **Incident:** The four self-check rules of the staged-track design; joint->Cartesian switch happens ONLY at a settled station (real_env 8335075 re-seeds IK from encoders on the switch).
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-011 · Settled-station re-extend (side/rear, EE-seeded)

- **Location (legacy):** 2013-2028
- **Protects against:** Re-extending without a live EE seed or away from a station would make the joint->Cartesian switch a blind jump from an unknown posture.
- **Trigger:** Track settled with the cube servable AND its sector index equal to the current station (sect == tm["cur"])
- **Response:** Track dropped; hold re-armed AT that station: sector/sector_idx updated, home/wrong_t0/state reset to tracking, restream seeded from pos_actual (mandatory — no EE this tick means stay settled and retry next tick), target = standoff(cube.pos); Cartesian approach then runs at the gentle 8 cm/s ramp.
- **Constants:** STAGE_REACH_EE_RATE=0.08
- **State:** h["track"], h["sector"], h["sector_idx"], h["home"], h["wrong_t0"], h["restream"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Comment: re-extend needs the live EE to seed the ramp; if telemetry lacks it this tick, stay settled and retry.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-012 · Turnaround only at settled stations (_retarget_track)

- **Location (legacy):** 2029-2034 (primitive 1092-1099)
- **Protects against:** Reversing a track mid-hop would rebuild the hop sequence from a between-stations posture, leaving the audited chain.
- **Trigger:** Settled at a station; cube servable but its sector differs from both the current station and the track's goal
- **Response:** _retarget_track(tm, sect, now) turns around IN PLACE toward the cube's sector from the settled station; ownership kept, nothing re-queued; print "[hold] {arm} reverses toward {station}".
- **Constants:** none
- **State:** tm["target"], tm["seq"], tm["cur"], tm["timed_out"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** _retarget_track docstring: "Only legal at a settled station — the sequence is rebuilt from tm['cur']".
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-013 · Release only at the front station

- **Location (legacy):** 2035-2043
- **Protects against:** Releasing a side/rear-stranded arm would let the next reach leg start from a non-front basin — the local mink IK path from there crosses the torso.
- **Trigger:** Track settled with cur==0, target==0, seq empty (physically AT front) and the cube still not servable
- **Response:** The ONLY release point for a tracked hold: print "[hold] {arm} arm releases {key} (retreated to front) — cube re-queued", cube.held_by=None, del self._held[arm], Phase.HOLD -> SEARCH.
- **Constants:** none
- **State:** tm["cur"], tm["target"], tm["seq"], cube.held_by, self.phase
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Docstring 1916-1919: SIDE/REAR retraction walks the chain in reverse then ALWAYS releases; those holds never re-extend in place — a fresh staged leg is the only safe way back in.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-014 · Absorbing-state fix (off-front strand recovery)

- **Location (legacy):** 2044-2052
- **Protects against:** A turnaround that settles at side/rear with the cube gone matched NO branch — the track became an ABSORBING state: arm stranded at side/rear, joint stream silent forever.
- **Trigger:** status=="done" at a station with tm["target"] != 0 and the cube not servable
- **Response:** _retarget_track(tm, 0, now): walk home along the audited chain; arriving at front re-enters the release-at-front branch. Print "[hold] {arm} arm heads home — {key} gone at the {station} station".
- **Constants:** none
- **State:** tm["target"], tm["cur"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-15 audit finding, cited verbatim in the comment ("no branch fires, the arm strands at side/rear and the stream stays silent forever").
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-015 · One-track-at-a-time token

- **Location (legacy):** 1981-1984 (+ new-retreat gate 2296-2298, docstring: _sec belt-and-suspenders)
- **Protects against:** Two joint tracks interleaving in one 14-joint packet — the ret_payload interleave facet of the 07-15 cross-body jump.
- **Trigger:** A second held arm's track wants to advance while ret_payload is already set this tick or the primary reach staircase (self._reach_tm) is active; also a new side/rear retreat while a dual-parallel secondary (self._sec) flies
- **Response:** The waiting arm's track does not step — its joints stay frozen inside the active track's packet; the new retreat is simply not launched while _sec is alive (belt-and-suspenders: side/rear holds cannot coexist with a dual-parallel reach by construction).
- **Constants:** none
- **State:** ret_payload, self._reach_tm, self._sec, h["track"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized (partial — "first grasped carried home first" serializes the carries)
- **Incident:** 0519d3c serialized the two arms after the dual-arm torso strike; audited airtight with a lagged repro (scratchpad/repro_crossbody_jump.py).
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-016 · Front vs side/rear retract difference

- **Location (legacy):** 2293-2300 (+ _register_hold 1645-1658, esp. 1651-1653; docstring 1916-1921)
- **Protects against:** An in-place Cartesian retract from a side/rear hold would drag the arm across the torso basin un-staged; conversely forcing front holds through the track would forbid the validated in-place retract/re-extend.
- **Trigger:** Hold transitions tracking -> retracted
- **Response:** sector_idx > 0 (side/rear): retract is a JOINT TRACK to front(0) via _new_track — interruptible only at settled stations, terminal release only at front. sector_idx == 0 (front): classic in-place retract — h["home"] makes _arm_packet home-glide the arm, and re-extension happens in place. side_home mode pins sector_idx=0 at registration so the track machinery can NEVER start (all retreats/carries are direct Cartesian to the side home).
- **Constants:** SECTOR_IDX front=0/side=1/rear=2, TRACK=(front,side,rear)
- **State:** h["sector_idx"], h["home"], h["track"], self.side_home
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none for retract (t_grasp_carry_side covers the carry staircase; no test retracts a live hold)
- **Incident:** Cross-body class: only the staged joint chain is hardware-validated for front<->side<->rear motion.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-017 · VIA-HOME rule for bearing jumps (side-home mode)

- **Location (legacy):** 2218-2241 + 2265-2269 (HOLD), 1961-1965 (GO/REACH refusal); const 254-261
- **Protects against:** An extended arm chasing a target whose bearing jumped (cube teleported front->rear) would slide the Cartesian target across the body — exactly the cross-body strike the staging exists to prevent.
- **Trigger:** side_home hold whose live standoff bearing differs from h["target"] bearing by > BEARING_JUMP_DEG
- **Response:** via_home LATCHES (a mid-retract sighting cannot shortcut the detour); the arm retracts to the side home and re-extends ONLY once the measured EE (finite-checked) is within 0.10 m of rest_ee (at_home). During another arm's GO/REACH the jump-follow is simply refused (no slide). Prints announce both the routing and the re-extend from the stage.
- **Constants:** BEARING_JUMP_DEG=45.0, at-home tolerance 0.10 m, SIDE_HOME_EE=[0,±0.55,0]
- **State:** h["via_home"], h["home"], self._ee_act[arm]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_via_home_routing ("arm physically passed THROUGH the side home", "then re-extended to the rear")
- **Incident:** User 07-18 hard requirement: the side home (bearing 90 deg) IS the stage; every big transition routes through it, nothing crosses the body directly.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-018 · Lifting-ramp restream pause while the joint stream owns the wire

- **Location (legacy):** 2073-2107 (pause 2084-2090; timeout 2078-2083)
- **Protects against:** A ramp that keeps advancing while its arm is frozen inside another arm's joint-track packet produces a resumed Cartesian jump the moment the wire returns.
- **Trigger:** Hold in grasp=="lifting" while self._reach_tm is active or any held arm's track runs
- **Response:** The clearance ramp pauses — restream holds its point (it is not being published anyway); with no EE seed yet it holds and retries; the whole clearance move has a GRASP_LIFT_TIMEOUT_S budget after which it degrades to carrying home from wherever it is (loud WARNING). Waypoint arithmetic (1e-9 arrival, pop lift_wps) advances only while unpaused.
- **Constants:** GRASP_LIFT_TIMEOUT_S=8.0, GRASP_LIFT_UP_M=0.05, GRASP_LIFT_OUT_M=0.08, STAGE_REACH_EE_RATE=0.08
- **State:** h["restream"], h["lift_wps"], h["lift_t0"], self._reach_tm, h["track"]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_serialized (lift/carry defer per serialization), t_grasp_carry_side (lift endpoint exactness)
- **Incident:** Comment: "pause the ramp so it never runs ahead of the frozen arm (a resumed jump otherwise)".
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-019 · Follow suppression while grabbing/grasped/carried

- **Location (legacy):** 1950-1954 (GO/REACH), 2060-2208 (grasp branches continue before the follow), 2128-2130
- **Protects against:** Once the gripper closes, detections track the HAND — following them chases ourselves in a feedback loop; a mid-grab retract would yank the cube out of the closing fingers.
- **Trigger:** h["grasp"] not in (None, "failed")
- **Response:** No follow, no retract, and no cube-position release for the hold; during "sent" the arm holds perfectly still; "lifting"/"grasped"/"carried" run their own ramps and skip the follow/release block entirely (drop detection is the sole cube-position check that still runs).
- **Constants:** none
- **State:** h["grasp"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side, t_dual_parallel_front (rig withholds hand-riding detections; drop path in t_drop_detection)
- **Incident:** Comment: "the fingers occlude the tags, and a mid-grab retract would yank the cube".
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-020 · Carry-track cube-decision bypass + jam warning

- **Location (legacy):** 1989-2011
- **Protects against:** During a carry the cube rides in the hand, so re-extend/reverse/release decisions computed from "cube position" are meaningless and would abort a good carry or release a gripped cube.
- **Trigger:** h["grasp"]=="grasped" hold whose carry track is in flight
- **Response:** All settled-station cube decisions are skipped; a timed-out hop keeps retrying toward front at the crawl rate with a 5 s-throttled "[grasp] WARNING ... (jam/sag?) ... watch the arm" print; at "done" the transition to carried requires live pos_actual (no EE this tick: stay settled and retry), then _enter_carried starts the gentle park ramp to REST.
- **Constants:** carry_warn throttle 5.0 s, STAGE_RATE=0.2
- **State:** h["track"], tm["timed_out"], h["carry_warn_t"], h["grasp"]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_grasp_carry_side ("carry used the joint staircase", "parked at REST"), t_dual_serialized
- **Incident:** Comment: detections track the hand; carry-hop timeout means jam/sag — keep crawling, warn the operator.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-021 · Held-arm per-tick packet overlay + silence fallback

- **Location (legacy):** 2395-2458 (held slots 2408-2422; silence returns 2420, 2447, 2453)
- **Protects against:** A holding arm's slot regressing to REST or garbage for even one packet yanks the hand off the cube; publishing an unseeded/NaN point commands an unbounded jump.
- **Trigger:** Every arm_targets publish, every phase
- **Response:** Each holding arm's slot publishes: restream ramp point when mid re-extend/lift/park, else the bias-corrected held target (h["target"] - h["bias"]); retracted (h["home"]) arms publish the home-glide; any missing seed or non-finite EE makes _arm_packet return None — the caller goes silent and real_env holds the last target. Idle off-front arms are held IN PLACE at their measured EE, never dragged home in one un-staged packet.
- **Constants:** REST_EE, SIDE_HOME_EE, NEUTRAL_QUAT
- **State:** h["home"], h["restream"], h["bias"], h["target"], self._ee_act
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_front ("packet carried BOTH live reach slots"), t_retry_then_fail (follow published), t_grasp_carry_side (REST park)
- **Incident:** 07-15 contract audit: with --use-ik there is NO robot-side retract-on-silence — silence makes real_env HOLD its last target, which is the safe fallback the overlay relies on.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-022 · Timed-out-secondary restream handoff into the hold

- **Location (legacy):** 1634-1658 (docstring 1642-1644, seed 1655-1656), consumer 2256-2261
- **Protects against:** Registering a hold at the raw cube target after a secondary timed out mid-ramp would snap the remaining approach distance in one packet (07-16 catapult class).
- **Trigger:** _register_hold called with restream= the secondary's current ramp point (timeout short of the cube)
- **Response:** The hold starts mid-ramp; the classic HOLD machinery's _gentle_approach finishes the approach at 8 cm/s instead of snapping.
- **Constants:** STAGE_REACH_EE_RATE=0.08, REACH_TIMEOUT_S=8.0
- **State:** h["restream"]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_unseeded_abandon (abandon path + "right slot never jumped (no catapult packet)")
- **Incident:** 07-17 audit: the sibling never-seeded case is ABANDONED outright (1760-1770) — registering it would publish the full REST->cube target in ONE step.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-023 · Grab gated on a quiescent hold

- **Location (legacy):** 1553-1555 (within _maybe_send_grab)
- **Protects against:** Closing the gripper while the arm is still ramping, retracted, or mid-track pinches air or yanks the cube.
- **Trigger:** _maybe_send_grab per tick for each held arm
- **Response:** hand_grab may fire only when h["grasp"] is None AND h["track"] is None AND h["restream"] is None AND not h["home"] — i.e. a settled, extended, non-ramping hold; further gated on EE-chain alive, zero grace, GRASP_SETTLE_S unmoved target, live EE within GRASP_EE_OK_M, and cube fresh.
- **Constants:** GRASP_SETTLE_S=1.0, GRASP_EE_OK_M=0.06, GRASP_ZERO_GRACE_S=10.0, DYN_LOST_GRACE_S=1.0
- **State:** h["grasp"], h["track"], h["restream"], h["home"], h["grasp_move_t"]
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_grasp_carry_side ("hand_grab sent after settle"), t_dual_parallel_sec_first_grab
- **Incident:** 07-16 audit: a hold that never grabs must say WHY — every unmet precondition prints throttled diagnostics (1592-1595).
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-024 · HOLD-phase servo-bias mirroring (single chain, most recent leg)

- **Location (legacy):** 2310-2328
- **Protects against:** One servo chain toggling between two held hands would corrupt both bias estimates; a stale bias applied to a re-targeted hold offsets the published point by |bias|.
- **Trigger:** Phase.HOLD with visual_servo enabled and self._reach_arm still holding
- **Response:** The servo keeps refining ONLY the most recent leg's bias; the refined b-hat is mirrored live into that arm's h["bias"]; the dyn-followed h["target"] feeds the chain each tick. (Related fail-safe in the grasp path: h["bias"]=None on grasp success at 2141-2143 — a cube-aimed bias must not offset the REST park.)
- **Constants:** none
- **State:** h["bias"], self._servo_bias, self._reach_arm, self._reach_target
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (all rigs run visual_servo=False)
- **Incident:** Docstring: earlier holds keep the bias frozen at registration — one chain cannot measure two hands; shutdown protocol documented: after Ctrl+C the arms HOLD last targets (no robot-side retract), so remove cubes, let holds release, THEN exit.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)

### SAFE-HELD-025 · dynamic_track kill-switch (fail-static holds)

- **Location (legacy):** 1926-1927 (flag 533; docstring 1920-1921)
- **Protects against:** N/A by design — bench opt-out; with tracking off a moving/vanishing cube must not induce ANY held-arm motion.
- **Trigger:** self.dynamic_track False
- **Response:** The entire per-arm body of _held_update is skipped: no follow, no retract, no release, no tracks — the packet overlay keeps publishing the frozen latch targets forever.
- **Constants:** dynamic_track (CLI --no-dynamic-track)
- **State:** self.dynamic_track
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (all rigs pass dynamic_track=True)
- **Incident:** --no-dynamic-track: targets stay frozen at their latch values.
- **New location (S6):** auto_operator/arms.py (held_update + lifecycle helpers)


## Dual-parallel

### SAFE-DUAL-001 · Dual-parallel enable + REACH-phase-only latch window

- **Location (legacy):** 2609-2616, 471, 523, 2757-2762, 2777
- **Protects against:** Secondary-arm machinery engaging in classic serialized missions (or gaze-only runs), producing unaudited two-arm motion
- **Trigger:** tick(): _try_latch_secondary is called only when self.dual_parallel AND self._sec is None AND self.phase == Phase.REACH; _tick_secondary only while self._sec is not None
- **Response:** With dual_parallel False the secondary never latches (bit-preserved one-reach-at-a-time). Constructor default is False; CLI default True but forced off by --gaze-only (dual_parallel=args.dual_parallel and not args.gaze_only, line 2777)
- **Constants:** dual_parallel=False (ctor default) / True (CLI default)
- **State:** self.dual_parallel, self._sec, self.phase
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_off, t_dual_parallel_front
- **Incident:** 07-17 dual-parallel feature added on top of the 07-15 modal-wire contract; classic mode must remain bit-identical
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-002 · Latch precondition: hold_after_reach required

- **Location (legacy):** 1674-1678
- **Protects against:** Mission stranded in HOLD forever: legacy park-and-DONE mode has no hold machinery to consume a secondary hold
- **Trigger:** _try_latch_secondary entry: if not self.hold_after_reach → return
- **Response:** No secondary is ever latched in --no-hold-after-reach mode
- **Constants:** hold_after_reach (ctor flag, default True)
- **State:** self.hold_after_reach
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-17 audit: a secondary hold in legacy mode would strand the mission in HOLD forever
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-003 · Latch precondition: trackless primary only (front sector or side_home)

- **Location (legacy):** 1679-1683
- **Protects against:** Parallelizing a joint staircase: the global 14-joint stream and a Cartesian secondary would fight over the modal wire (07-15 cross-body-jump class)
- **Trigger:** _try_latch_secondary: return if self._reach_tm is not None OR not (self.side_home or self._reach_sector == 'front')
- **Response:** Secondary latches only while the primary is a pure-Cartesian (trackless) leg; side/rear staircase legs never parallelize. In side_home mode EVERY leg is trackless so any sector may parallelize
- **Constants:** SECTOR_FRONT_DEG=60.0 (via _reach_sector)
- **State:** self._reach_tm, self._reach_sector, self.side_home
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_side_gate, t_side_home_dual_any_sector
- **Incident:** Joint staircases stay strictly exclusive per the 07-15 modal-wire contract (hardware front→rear arm sweep through the torso)
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-004 · Latch precondition: no held-arm joint track in flight

- **Location (legacy):** 1684-1685
- **Protects against:** Latching a Cartesian secondary while a held arm walks a joint retreat/carry track — two owners of the single arm-targets wire
- **Trigger:** _try_latch_secondary: if any(h['track'] is not None for h in self._held.values()) → return
- **Response:** Latch deferred until the joint stream releases the wire
- **Constants:** none
- **State:** self._held[arm]['track']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (t_dual_serialized covers the reverse direction: no track during GO/REACH)
- **Incident:** "a joint stream owns the wire" — same 07-15 joint-vs-Cartesian exclusivity contract
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-005 · Latch precondition: free opposite arm only, one secondary max

- **Location (legacy):** 1686-1688, 2613
- **Protects against:** Commanding a secondary reach with an arm that is already holding/carrying a cube, or stacking multiple secondaries
- **Trigger:** _try_latch_secondary: arm = opposite of self._reach_arm; if arm in self._held → return; caller requires self._sec is None
- **Response:** At most one secondary, always on the arm not used by the primary and not engaged in a hold
- **Constants:** none
- **State:** self._reach_arm, self._held, self._sec
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_front
- **Incident:** Dual-parallel is strictly front+front, one cube per arm (07-17 design)
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-006 · Latch candidate filters: confirmed, off-midline, same-side, front sector, envelope, reachable

- **Location (legacy):** 1689-1699
- **Protects against:** Secondary latched onto a fluke sighting, a midline cube (arm choice = coin flip → cross-body reach), a torso/face-height IK target, or an unreachable cube
- **Trigger:** Per-cube skip: active/held/reached cubes; not c.confirmed(now) (≥0.3 s steady streak, fresh <1.0 s); abs(c.pos[1]) <= MIDLINE_MARGIN_M; half-space arm != free arm; sector != 'front' (unless side_home); not _target_safe (r_xy in [TARGET_MIN_RADIUS_M, TARGET_MAX_RADIUS_M], z in [TARGET_Z_MIN_M, TARGET_Z_MAX_M]); not _reachable_now (gate score > SCORE_THRESHOLD and best_arm==arm, or r_xy <= DYN_OUTREACH_M without gate)
- **Response:** Cube skipped; no latch this tick
- **Constants:** MIDLINE_MARGIN_M=0.03, TARGET_MIN_RADIUS_M=0.16, TARGET_MAX_RADIUS_M=0.45, TARGET_Z_MIN_M=-0.40, TARGET_Z_MAX_M=0.30, DYN_OUTREACH_M=0.32
- **State:** c.confirmed/first_seen/last_seen, c.held_by, c.reached, self.side_home
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_front (happy path); envelope/midline branches: none
- **Incident:** Envelope from the 07-15 adversarial audit (IK has no self-collision awareness; trunk keep-out + z-band were absent → confirmed hardware-damage findings); r>0.45/z>0 is the torso-tilt hang-too-low signature (07-18)
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-007 · Latch median-of-window re-validation

- **Location (legacy):** 1701-1710
- **Protects against:** A single 1-inlier PnP glitch frame poisoning the secondary target, or a lagged median disagreeing with the instantaneous frame on side/sector/envelope (07-15 rear-midline cross-back class)
- **Trigger:** window = sightings within CUBE_LATCH_WINDOW_S; empty window → skip cube; median must pass side_ok (median[1] beyond ±MIDLINE_MARGIN_M on the free arm's side), _target_safe, and front sector (unless side_home)
- **Response:** Cube skipped this call; target (when latched) is the per-axis median passed through _standoff_point
- **Constants:** CUBE_LATCH_WINDOW_S=0.7, MIDLINE_MARGIN_M=0.03, REACH_STANDOFF_M=0.05, AIM_UP_M=0.01 (0 until 07-18, then +2 cm, trimmed to +1 cm same day per user)
- **State:** c.hist (deque maxlen=5)
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (median validation tested only on the primary path via t_inflight_sector_clamp)
- **Incident:** Mirrors the primary's _latch_reach_target: 07-15 audit found the arrival frame alone chose the arm while the median the arm chases could disagree
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-008 · _sec_hits GATE_CONSECUTIVE latch debounce

- **Location (legacy):** 1711-1721, 1730, 123-132
- **Protects against:** Committing an arm to a cube on one lucky stochastic ReachabilityGate sample — the learned gate samples fresh random candidates per call, so a single _reachable_now verdict chatters
- **Trigger:** _sec_hits = [cube_key, streak]; same key increments streak, different key resets to 1; latch only when streak >= GATE_CONSECUTIVE; any call with no qualifying candidate resets _sec_hits = None (line 1730)
- **Response:** Latch deferred until 5 consecutive qualifying evaluations of the same cube; with GATE_EVAL_PERIOD_S=0.1 fresh-sample throttling this spans ~0.5 s wall-clock
- **Constants:** GATE_CONSECUTIVE=5, GATE_EVAL_PERIOD_S=0.1
- **State:** self._sec_hits
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none for _sec_hits itself (t_gate_throttle covers the shared fresh-sample debounce rule)
- **Incident:** 07-17 audit: mirrors the arrival gate's GATE_CONSECUTIVE streak; also 07-17 gate-throttle fix (unthrottled CPU-torch scoring blew the 50 ms tick budget → pre-reach camera shudder)
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-009 · Secondary in-flight retrack with latch-grade validation

- **Location (legacy):** 1741-1745 (call), 1597-1632 (_live_retrack)
- **Protects against:** Grabbing off-centre as balance sway drifts the body frame 1-2 cm during a multi-second approach; OR the facet-2 cross-body class — a drifting cube sliding the target un-staged across a sector boundary
- **Trigger:** Every _tick_secondary tick when dynamic_track: target follows the live median of the last CUBE_LATCH_WINDOW_S sightings; update returned ONLY if cube fresh (DYN_LOST_GRACE_S), window non-empty, median on the arm's half-space (MIDLINE_MARGIN_M), _target_safe, NOT _sector_left(median, sector) (hysteresis SECTOR_HYST_DEG), bearing jump <= BEARING_JUMP_DEG in side_home (VIA-HOME rule), and move > DYN_RETRACK_M deadband
- **Response:** Passing update replaces s['target']; any failed check returns None → target stays frozen at the last good point (occlusion → frozen behavior)
- **Constants:** CUBE_LATCH_WINDOW_S=0.7, DYN_LOST_GRACE_S=1.0, DYN_RETRACK_M=0.01, MIDLINE_MARGIN_M=0.03, SECTOR_HYST_DEG=10.0, BEARING_JUMP_DEG=45.0
- **State:** self._sec['target'], cube.hist, self.dynamic_track
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_inflight_retrack, t_inflight_sector_clamp (shared helper, exercised via the primary path)
- **Incident:** 07-17: frozen latch is a body-frame point; standing sway / reach-reaction tilt made it drift (grabbed-off-centre class); sector clamp closes the facet-2 cross-body gap
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-010 · Secondary gentle-approach ramp + unseeded silence (WAITING)

- **Location (legacy):** 1746-1752, 1420-1435 (_gentle_approach), 346-350
- **Protects against:** Handing the robot IK the full REST→cube jump in one packet — the arm sweeps at the robot-side 0.25 m/s cap ('too fast' on hardware)
- **Trigger:** _tick_secondary: s['stream'] ramped toward tgt at STAGE_REACH_EE_RATE*DT per tick; ramp seeds ONLY from telemetry ee.<arm>.pos_actual — with no seed, pub is None and the arm's slot stays silent (never seeded at the goal); WAITING diagnostic printed, throttled to 1.0 s via _sec_diag_t
- **Response:** Published point moves at most 4 mm/tick (0.08 m/s at 20 Hz); no telemetry → no publish (robot holds pose) + loud [reach2] WAITING print
- **Constants:** STAGE_REACH_EE_RATE=0.08, DT=1/OP_RATE_HZ (OP_RATE_HZ=20.0), _sec_diag throttle=1.0 s
- **State:** self._sec['stream'], self._sec_diag_t
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_front, t_dual_parallel_unseeded_abandon (max per-tick jump < 0.02 m asserted)
- **Incident:** 07-16: every final Cartesian approach is streamed (full 30 cm jump in one packet read as too fast); silent-waiting must self-report (07-16 audit failure class)
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-011 · Secondary arrival → immediate hold registration (no visual servo)

- **Location (legacy):** 1753-1759, 1634-1658 (_register_hold)
- **Protects against:** First-arrived arm idling on the cube waiting for the other arm (stale settle window, cube drift) instead of entering the audited hold/grasp machinery
- **Trigger:** actual within REACH_OK_M of target → _register_hold(arm, cube, tgt, 'front' (or _sector(tgt) if side_home), bias=None, now); self._sec = None
- **Response:** Normal per-arm hold/grasp machinery owns the arm immediately — first arrived starts grabbing with no cross-arm wait; secondary carries NO servo bias (bias stays None)
- **Constants:** REACH_OK_M=0.05
- **State:** self._sec, self._held[arm], cube.held_by
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_front, t_dual_parallel_sec_first_grab
- **Incident:** 07-17 design: arrival OR timeout registers the hold — same policy as the primary
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-012 · Unseeded secondary timeout → ABANDON (anti-catapult)

- **Location (legacy):** 1760-1770
- **Protects against:** The audited 07-16 catapult class: registering a hold for an arm that never moved makes the next packet publish the full REST→cube target in ONE step
- **Trigger:** (now - s['t0']) > REACH_TIMEOUT_S AND s['stream'] is None (EE telemetry never seeded the ramp)
- **Response:** Secondary ABANDONED loudly ([reach2] ... never got an EE seed ... ABANDONED); NO hold registered; self._sec = None; cube stays pending and re-queues for a classic serialized leg
- **Constants:** REACH_TIMEOUT_S=8.0
- **State:** self._sec['stream'], self._sec['t0']
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_unseeded_abandon
- **Incident:** 07-17 audit: the primary is immune because it stalls before its timeout check; the secondary needed this explicit branch
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-013 · Mid-ramp secondary timeout → restream handoff

- **Location (legacy):** 1771-1778, 1642-1644 + 1655-1656 (_register_hold restream), 2412-2413 (_arm_packet publishes restream)
- **Protects against:** A secondary that timed out short of the cube snapping the REMAINING distance in one packet when the hold takes over
- **Trigger:** (now - s['t0']) > REACH_TIMEOUT_S AND s['stream'] is not None
- **Response:** _register_hold(..., restream=s['stream']): the hold is seeded at the CURRENT ramp point and the classic re-extend machinery finishes the gentle approach; _arm_packet publishes h['restream'] instead of the raw target until it clears
- **Constants:** REACH_TIMEOUT_S=8.0, STAGE_REACH_EE_RATE=0.08
- **State:** self._sec['stream'], self._held[arm]['restream']
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (no test drives a seeded-but-timed-out secondary)
- **Incident:** 07-17 audit: same anti-snap principle as the abandon branch, applied to a partially-flown ramp
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-014 · Slot injection: secondary ramp owns the free arm's packet slot

- **Location (legacy):** 2423-2431 (in _arm_packet)
- **Protects against:** The idle-arm REST/home-glide logic publishing REST for the secondary's arm mid-flight — yanking a mid-reach arm back home; or an unseeded secondary's slot publishing the raw goal
- **Trigger:** _arm_packet: if self._sec is not None and self._sec['stream'] is not None, the ramp point fills the free arm's slot in EVERY phase — but only if the slot is not already owned by a hold or the primary reach (ee_pos[j] is None check); an unseeded secondary (stream None) leaves the slot to the idle logic since the arm has not moved yet
- **Response:** Packet always carries the audited ramp point for the flying secondary; slot-ownership priority: held/restream > primary reach_pos > secondary stream > idle REST/glide/hold-in-place
- **Constants:** none
- **State:** self._sec['stream'], self._sec['arm'], ee_pos slots
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_front (saw_both_slots), t_dual_parallel_unseeded_abandon (right slot never jumped)
- **Incident:** 07-17: the packet carries independent per-arm slots; robot IK solves both arms together, so Cartesian+Cartesian has no wire-mode conflict
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-015 · _idle_arms_home exemption for the in-flight secondary arm

- **Location (legacy):** 2356-2368 (exemption at 2368), 2650-2662 (keep-alive gate)
- **Protects against:** Keep-alive REST suppressed for the whole mission because the secondary's arm is (correctly) away from home — or conversely the off-home secondary arm being classified as 'stranded' and frozen
- **Trigger:** _idle_arms_home iterates arms not in self._held AND not (self._sec is not None and self._sec['arm'] == arm) — the secondary's arm is exempt from the near-home/glide requirement
- **Response:** Keep-alive packets keep flowing while a secondary flies; the exempted arm's slot is then filled by the _sec stream injection, never by REST
- **Constants:** FRONT_HOME_TOL_RAD=0.35 (near-home check the exempted arm skips)
- **State:** self._sec['arm'], self._held, self._home_stream
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (exercised implicitly by t_dual_parallel_front reaching REST assertions)
- **Incident:** 07-15 audit: an idle arm caught off-front is held in place, never dragged home in one un-staged Cartesian packet — the secondary arm needed an explicit carve-out from that rule
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-016 · SEARCH gate: no new leg / no base motion while a secondary flies

- **Location (legacy):** 1293-1296 (in _tick_search)
- **Protects against:** Walking the base (or starting a new GO leg) under a latched body-frame Cartesian reach target — the target frame moves out from under the flying arm
- **Trigger:** _tick_search: if self._sec is not None → return before any candidate selection
- **Response:** SEARCH idles until the secondary resolves (hold registered or abandoned); with walk enabled, absolutely no base motion under a latched Cartesian reach target
- **Constants:** none
- **State:** self._sec, self.phase
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (implicit in t_dual_parallel_front end-to-end)
- **Incident:** Comment: 'dual-parallel approach in flight: no new leg starts (and with walk enabled, absolutely no base motion under a latched Cartesian reach target)'
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-017 · Carry gate: carry-home staircase waits out the secondary

- **Location (legacy):** 2115-2121
- **Protects against:** A grasped arm starting its sector→front joint staircase while the other arm flies a Cartesian secondary — the global 14-joint packet would override the Cartesian stream and fling the flying arm (07-15 hardware jump class)
- **Trigger:** grasp == 'grasped' with sector_idx > 0: track starts only if h['track'] is None AND self._reach_tm is None AND self._sec is None
- **Response:** Carry deferred; h stays frozen at its last target until the wire is free
- **Constants:** none
- **State:** self._sec, self._held[arm]['track'], self._held[arm]['sector_idx'], self._reach_tm
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none explicit (t_dual_parallel_front is all-front so no staircase; t_dual_parallel_side_gate never has _sec during the staircase)
- **Incident:** Comment: 'a joint staircase must wait out the dual-parallel Cartesian reach' — extension of Guard 1 / one-track token
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-018 · Retreat gate: dyn-retract track blocked while a secondary flies

- **Location (legacy):** 2296-2300
- **Protects against:** A side/rear hold starting its retraction staircase (joint stream) concurrent with the secondary's Cartesian stream — same wire-mode conflict
- **Trigger:** Hold going home with sector_idx > 0: h['track'] = _new_track(...) only if self._sec is None
- **Response:** Retreat deferred one tick at a time until the secondary resolves; comment marks this belt-and-suspenders (side/rear holds cannot coexist with a dual-parallel reach by construction — latch requires a trackless front primary)
- **Constants:** none
- **State:** self._sec, self._held[arm]['sector_idx'], self._held[arm]['home']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-17 audit hardening: defense-in-depth on the 07-15 joint-vs-Cartesian exclusivity even where reachable-by-construction says it cannot happen
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)

### SAFE-DUAL-019 · First-arrival grab during GO/REACH (gripper bus exempt from Guard 1)

- **Location (legacy):** 1969-1975 (call in _held_update GO/REACH branch), 1545-1556 (_maybe_send_grab preconditions)
- **Protects against:** A hold registered by secondary first-arrival silently waiting to grab until the other arm's reach ends — cube drifts / settle window goes stale; conversely, an unguarded grab could have violated wire exclusivity
- **Trigger:** Phase GO/REACH branch of _held_update calls _maybe_send_grab every tick; grab still requires grasp enabled, EE chain alive (not _ee_chain_dead), h['track'] is None, h['restream'] is None, not h['home'], zero-grace elapsed (GRASP_ZERO_GRACE_S), settle (GRASP_SETTLE_S since last target move), cube fresh (DYN_LOST_GRACE_S), EE within GRASP_EE_OK_M of target
- **Response:** hand_grab dispatched IMMEDIATELY during the other arm's reach — safe because the gripper rides its own serial bus, never the 14-joint stream (Guard 1 modal-wire exclusivity untouched); lift/carry/result-processing keep their deferral (they need the wire); un-met preconditions are reported loudly every 3 s
- **Constants:** GRASP_ZERO_GRACE_S=10.0, DYN_LOST_GRACE_S=1.0, GRASP_SETTLE_S / GRASP_EE_OK_M (grasp block), diag throttle=3.0 s
- **State:** self._held[arm]['grasp'/'track'/'restream'/'home'/'grasp_move_t'], self._ee_chain_dead, self._zeros_t
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_sec_first_grab, t_dual_parallel_front (saw_grab_while_flying)
- **Incident:** 07-17 dual-parallel audit: 'a hold registered while the other arm still reaches (secondary first-arrival) must GRAB now, not after that reach ends'; silent waiting is the failure class the 07-16 audit was commissioned against
- **New location (S6):** auto_operator/secondary.py + arms.py (_sec gates) + arbitration.py (slot injection)


## Visual servo

### SAFE-SERVO-001 · Servo hard-disable when model lacks gripper bodies

- **Location (legacy):** 550-561 (constants 406-407)
- **Protects against:** Servo running against a model that cannot FK the gripper base would correct against garbage and walk the hand off the validated pose
- **Trigger:** At construction with visual_servo=True, mj_name2id finds neither SERVO_BODIES entry (L_base/R_base) in the MJCF
- **Response:** Prints '[servo] model has no ... bodies — visual servo DISABLED (open-loop reach)' and sets self.visual_servo=False for the whole mission; reach stays pure open-loop
- **Constants:** SERVO_BODIES={'left':'L_base','right':'R_base'}, SERVO_GRIPPER_KEYS={'left':'gripper_L_base','right':'gripper_R_base'}
- **State:** self._grip_body_id (empty dict = servo off), self.visual_servo
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (entire test rig runs visual_servo=False)
- **Incident:** Model predates the grippers (header comment line 550); MEMORY: monitor must publish --bodies gripper_L_base or servo is starved
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-002 · Gripper-sighting ingest gates: min inliers + NaN drop + measurement-only

- **Location (legacy):** 781-793 (SERVO_MIN_INLIERS def 409)
- **Protects against:** Single-16mm-tag PnP garbage poses or NaN detections feeding the control loop and poisoning the bias / every downstream target
- **Trigger:** MonitorDetectionPacket entry for a SERVO_GRIPPER_KEYS key with n_inliers < SERVO_MIN_INLIERS, or pos not shape (3,) finite
- **Response:** Sighting silently dropped (never enters self._grip); gripper bases are measurement-only — never tracked/claimed like cubes, never a gaze target
- **Constants:** SERVO_MIN_INLIERS=2
- **State:** self._grip[arm] = {pos, port, stamp, seen}
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 'single 16 mm-tag PnP poses are garbage-prone and this feeds a control loop' (line 783); NaN drop mirrors the cube-track NaN drop from the 07-15 audit fail-open class (lines 758-761, 789)
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-003 · Same-camera rule (no cross-camera servo)

- **Location (legacy):** 844-852 (port bound at 1393-1394, rationale 386-388, 1360-1362)
- **Protects against:** Correcting a cube fix from camera A with a gripper measurement from camera B — common-mode extrinsic errors only cancel within one camera chain, so the 'correction' injects the inter-camera disagreement into the hand position
- **Trigger:** grip['port'] != self._reach_port (the MODAL measurement port of the latch window, NOT the claim port — the monitor's targets dict is last-camera-wins)
- **Response:** Update skipped; after SERVO_GIVEUP_S a once-per-leg diagnostic explains the overwrite mechanism ('cam X may well see the gripper and be overwritten — not necessarily an aiming problem')
- **Constants:** SERVO_GIVEUP_S=3.0, CUBE_LATCH_WINDOW_S=0.7 (window whose modal port binds _reach_port)
- **State:** self._reach_port, self._grip[arm]['port'], self._servo_warned key 'port'
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** MEMORY visual-servo: per-leg b-hat with same-camera rule closes the reach loop; claim port ≠ measurement port because monitor publishes ONE entry per key, last camera wins
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-004 · Per-leg bias reset at every latch

- **Location (legacy):** 719-742 (call sites 549 construction, 1392 GO→REACH latch; rationale 389-391)
- **Protects against:** A bias learned at cube #1's bearing miscorrecting cube #2's reach — the underlying angular error sources rotate with gimbal aim and arm pose
- **Trigger:** Every GO→REACH latch (_latch calls _reset_leg_state(now) at 1392) and at construction
- **Response:** Wipes _servo_bias/_servo_seen/_servo_accepted/_servo_stamp/_servo_cmd_prev/_servo_ee_prev/_servo_gimbal_prev/_servo_accept_t/_servo_warned; b-hat re-learned from zero each leg. Deliberately NOT reset: _servo_log_t (print throttle) and _grip (measurement stream, staleness handled by SERVO_FRESH_S)
- **Constants:** none (structural rule; rejected alternative: persisting b-hat across legs, line 391)
- **State:** _servo_bias, _servo_seen, _servo_accepted, _servo_stamp, _servo_cmd_prev, _servo_ee_prev, _servo_gimbal_prev, _servo_accept_t, _servo_warned, _reach_port
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Header lines 389-391: bias is per-LEG because its angular error sources rotate with gimbal aim and arm pose
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-005 · Sighting freshness gate

- **Location (legacy):** 837-843 (SERVO_FRESH_S def 410)
- **Protects against:** Correcting from a stale gripper sighting taken at a different arm/gimbal pose — the hand would chase where the gripper used to be
- **Trigger:** self._grip[arm] is None or (now - grip['seen']) > SERVO_FRESH_S in _servo_step
- **Response:** No update this tick; after SERVO_GIVEUP_S since _reach_t0, once-per-leg 'unseen' warning that the reach stays open-loop (names the missing monitor --bodies key)
- **Constants:** SERVO_FRESH_S=0.5, SERVO_GIVEUP_S=3.0
- **State:** self._grip[arm]['seen'], self._reach_t0, self._servo_warned key 'unseen'
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Requires monitor to publish the gripper body (--bodies ... gripper_L_base), header line 405; MEMORY: monitor needs --bodies gripper_L_base
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-006 · Capture-stamp dedup (one attempt per new camera frame)

- **Location (legacy):** 853-856 (idempotency rationale 382-383)
- **Protects against:** Re-processing one camera frame across many 20 Hz control ticks would multiply its EMA weight; combined with measurement latency this could wind the estimator up
- **Trigger:** grip['stamp'] == self._servo_stamp (same capture_stamp already processed)
- **Response:** Return without an update; _servo_stamp advances only on a genuinely new frame. (Estimator form is also idempotent per frame, so latency cannot wind it up — belt and suspenders)
- **Constants:** none (uses capture_stamp from the detection packet, default now at ingest line 792)
- **State:** self._servo_stamp, self._grip[arm]['stamp']
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Header 382-383: 'idempotent per frame ... so measurement latency cannot wind it up'
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-007 · Executed gate (command settled before trusting a measurement)

- **Location (legacy):** 821-824, enforced 872; _servo_cmd_prev maintained at 910 and 2326 (SERVO_SETTLE_M def 411)
- **Protects against:** Updating b-hat while the arm is still flying to the last published target — the FK-vs-camera difference then contains tracking error, not hand bias
- **Trigger:** ee pos_actual missing, or ||pos_actual - _servo_cmd_prev|| > SERVO_SETTLE_M, evaluated every _servo_step
- **Response:** executed=False → measurement not accepted (line 872 requires executed AND ee_static AND gimbal_static); missing telemetry also yields False (fail closed). _servo_target() must run every tick (even in HOLD, line 2326) purely to keep this bookkeeping alive
- **Constants:** SERVO_SETTLE_M=0.03
- **State:** self._servo_cmd_prev (last PUBLISHED target), telemetry ee[arm].pos_actual
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Header 392-394: updates only when the previous command was EXECUTED; leg report names 'IK tracking error > 3 cm' as an open-loop cause (933-934)
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-008 · Arm quasi-static gate across camera frames

- **Location (legacy):** 857-862, enforced 872 (SERVO_STATIC_M def 412)
- **Protects against:** A frame captured 50-150 ms before arrival measured while the EE moves — motion during the latency window pollutes b_meas and steers the hand with blurred/lagged geometry
- **Trigger:** EE-FK moved ≥ SERVO_STATIC_M between consecutive processed camera frames (_servo_ee_prev vs current pos_actual), or either sample missing
- **Response:** ee_static=False → frame skipped; _servo_ee_prev refreshed each frame so a later still frame can pass
- **Constants:** SERVO_STATIC_M=0.005
- **State:** self._servo_ee_prev
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Comment 857-858: 'the frame was captured 50-150 ms ago, so trust it only if nothing (arm, gimbal) moved meaningfully across frames around that window'
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-009 · Gimbal quasi-static gate (measuring camera only)

- **Location (legacy):** 863-871, enforced 872 (SERVO_STATIC_GIMBAL_RAD def 413)
- **Protects against:** Camera extrinsics lagging the encoders mid-gimbal-motion — the measured gripper position would be expressed through a stale camera pose and corrupt b_meas
- **Trigger:** Per-frame motion of the MEASURING camera's own two gimbal joints (jpos[27+ys], jpos[27+ps] via GAZE_ORDER[self._reach_port]) ≥ SERVO_STATIC_GIMBAL_RAD
- **Response:** gimbal_static=False → frame skipped; the OTHER camera may legitimately sweep (its motion cannot blur this camera's frames, comment 863-864); missing jpos (< 31) passes this gate but is caught by the FK guard
- **Constants:** SERVO_STATIC_GIMBAL_RAD=0.01, GAZE_ORDER
- **State:** self._servo_gimbal_prev, self._reach_port
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Extrinsic-lag class: frame arrives 50-150 ms late so gimbal motion means camera-to-base transform is stale (comments 413, 857-858)
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-010 · FK-unavailable fail-closed in _grip_base_fk

- **Location (legacy):** 801-813 (guard 806-808), caller 874-879
- **Protects against:** Computing b_meas against an undefined/garbage FK prediction when telemetry joint_pos is missing or shorter than the model's joint count
- **Trigger:** self._grip_body_id has no entry for the arm, jpos is None, or len(jpos) < self.fk.n_joints
- **Response:** _grip_base_fk returns None → _servo_step warns once ('servo inactive, reach stays open-loop', naming n_joints) and skips the update; FK reuses the GimbalCameraFK model/data safely because every user re-writes qpos before reading (single tick thread, docstring 804-805)
- **Constants:** self.fk.n_joints
- **State:** self._grip_body_id, self._servo_warned key 'fk'
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Telemetry-schema drift class: joint_pos missing/short vs model — degrade to open-loop rather than correct against garbage (877-878)
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-011 · Single-measurement reject threshold (mis-ID guard)

- **Location (legacy):** 880-885 (SERVO_MEAS_REJECT_M def 414)
- **Protects against:** A mis-identified tag (e.g. the other gripper, a cube face) or a garbage PnP pose entering the EMA and dragging the hand toward a phantom bias
- **Trigger:** ||b_meas|| = ||camera-measured base - FK-predicted base|| > SERVO_MEAS_REJECT_M
- **Response:** Measurement dropped before touching b-hat; once-per-leg warning prints the offending offset in mm, further rejects dropped silently
- **Constants:** SERVO_MEAS_REJECT_M=0.15
- **State:** self._servo_warned key 'reject'
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Constant comment 414: 'single measurement offset beyond this = mis-ID/garbage, dropped'
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-012 · EMA gain + per-frame step clamp

- **Location (legacy):** 886-891 (SERVO_GAIN def 408, SERVO_STEP_M def 415)
- **Protects against:** One bad-but-under-reject-threshold frame yanking the hand: an unclamped EMA step could move the published target several cm in one 20 Hz tick against a possibly-contacting hand
- **Trigger:** Every accepted measurement; clamp engages when ||delta|| = ||SERVO_GAIN*(b_meas - b-hat)|| > SERVO_STEP_M
- **Response:** delta rescaled to exactly SERVO_STEP_M — the published target moves ≤ 1 cm per accepted frame ('one frame moves the hand ≤1 cm', 889)
- **Constants:** SERVO_GAIN=0.3, SERVO_STEP_M=0.01
- **State:** self._servo_bias
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Header 395-396: one bad frame moves the hand ≤1 cm; b-hat converges to ~66% within REACH's 3 accepted updates (2314-2315)
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-013 · Total bias-norm clamp (bounded deviation from validated open-loop pose)

- **Location (legacy):** 892-895, bound restated in _servo_target 904-911 (SERVO_BIAS_MAX_M def 416)
- **Protects against:** A run of correlated bad frames walking the hand arbitrarily far — e.g. into the torso or table — from the hardware-validated open-loop contact pose
- **Trigger:** ||b-hat|| > SERVO_BIAS_MAX_M after the EMA step
- **Response:** b-hat rescaled to norm SERVO_BIAS_MAX_M; published target = latched reach target - b-hat, so TOTAL deviation from the validated pose is ≤ 8 cm forever. Rejected alternative (397-398): an absolute workspace box, because it corrupts legitimate high/low targets by clipping the whole target rather than the correction
- **Constants:** SERVO_BIAS_MAX_M=0.08
- **State:** self._servo_bias, self._servo_cmd_prev
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Header 380-382: driving the tag origin onto the standoff point would walk the hand ~8.6 cm past the validated contact pose (tag origin sits 8.6 cm behind the EE site — MEMORY visual-servo); the bias-estimator form + this clamp preserve the open-loop geometry bit-for-bit at equilibrium
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-014 · REACH done-verdict gate: min accepted updates or provable inability

- **Location (legacy):** 913-922 enforcement 1522-1525 (SERVO_MIN_UPDATES def 417, SERVO_GIVEUP_S 418, REACH_OK_M 139, REACH_TIMEOUT_S 140)
- **Protects against:** A non-final cube exiting REACH at open-loop accuracy (multi-cm bias) while claiming closed-loop success — the grab would then close on air or knock the cube
- **Trigger:** REACH done check: EE within REACH_OK_M of target AND _servo_ready(now); _servo_ready is True only if servo off, _servo_accepted >= SERVO_MIN_UPDATES, or ((not _servo_seen) and now - _reach_t0 > SERVO_GIVEUP_S)
- **Response:** Success verdict withheld until the servo actually corrected or provably cannot on this leg; the REACH_TIMEOUT_S=8.0 clock still exits unconditionally (deliberate escape hatch so a stuck servo can never deadlock the mission). Both the 8 s clock and the give-up clock start only AFTER the staging track settles (_reach_t0 set at 1497)
- **Constants:** SERVO_MIN_UPDATES=3, SERVO_GIVEUP_S=3.0, REACH_OK_M=0.05, REACH_TIMEOUT_S=8.0
- **State:** self._servo_accepted, self._servo_seen, self._reach_t0
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Header 399-401: 'otherwise non-final cubes would exit at open-loop accuracy'; note _servo_seen=True requires only a same-port fresh sighting, so a seen-but-never-accepted leg rides to the 8 s timeout
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-015 · Blocked-and-blind bias decay in HOLD

- **Location (legacy):** 825-835 (SERVO_DECAY_MPS def 419; 5.0 s no-accept threshold hardcoded at 826)
- **Protects against:** The arm pressing forever against an obstacle with a possibly-poisoned correction: command stops being executable and no camera evidence arrives to fix b-hat
- **Trigger:** phase == HOLD, _servo_bias not None, executed == False (EE-FK > SERVO_SETTLE_M from published target), and now - _servo_accept_t > 5.0 s
- **Response:** ||b-hat|| shrinks by SERVO_DECAY_MPS*DT per tick toward 0 (floor 0), walking the hand back to the validated open-loop pose at 1 mm/s; once-per-leg 'decay' warning explains why
- **Constants:** SERVO_DECAY_MPS=0.001, DT, SERVO_SETTLE_M=0.03, hardcoded 5.0 s
- **State:** self._servo_bias, self._servo_accept_t, self._servo_warned key 'decay'
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** Header 402-404: blocked-and-blind self-heal — hand walks back instead of pressing forever
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-016 · Per-leg closing diagnostic (starvation must not look like success)

- **Location (legacy):** 924-935, called at 1526 on every REACH exit (done or timeout)
- **Protects against:** Operator misreading a starved open-loop leg as a closed-loop success and trusting the landing accuracy on the next hardware session
- **Trigger:** Every leg end: _servo_accepted == 0 vs > 0
- **Response:** One line naming the exact cause: 'gripper tags never usable (unseen or wrong camera)' vs 'no frame passed the executed/static gates (IK tracking error > 3 cm, or arm/gimbal never still)' — the diagnostic distinguishes ingest starvation from gate starvation
- **Constants:** SERVO_SETTLE_M (quoted in the message)
- **State:** self._servo_accepted, self._servo_seen, self._servo_bias
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Docstring 925: 'starvation must not look like success'
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-017 · HOLD refinement: single chain refines only the most recent leg; earlier holds frozen

- **Location (legacy):** 2310-2328 (refine gate 2322, mirror 2327); freeze at registration 1533-1537 and 1645-1647; secondary bias stays None 1673; consumer 2412-2415
- **Protects against:** One servo chain 'refining' a hand it is not measuring: applying live b-hat updates to an earlier hold's arm would correct arm A with camera evidence about arm B
- **Trigger:** HOLD tick with visual_servo and self._reach_arm in self._held — only that arm's hold gets _reach_target rebound, _servo_step, and h['bias'] mirrored from the live _servo_bias
- **Response:** b-hat (converged to ~66% during REACH's 3-update exit) finishes converging in HOLD and is mirrored live into that arm's hold entry; every EARLIER hold keeps the bias copied at _register_hold ('one chain cannot measure two hands'); the dual-parallel secondary never gets a bias at all. _held_update publishes target - bias per-arm (2415)
- **Constants:** SERVO_MIN_UPDATES=3 (why refinement is still needed in HOLD)
- **State:** self._reach_arm, self._held[arm]['bias'], self._servo_bias, self._reach_target (rebound to h['target'] so the dyn-followed target feeds the chain)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (t_dual_serialized/t_dual_parallel_front run the hold flow but with visual_servo=False, bias always None)
- **Incident:** Docstring 2312-2317; 07-15 contract audit note in same docstring: with --use-ik there is NO robot-side retract-on-silence, shutdown protocol is remove cubes → holds release → THEN exit
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-018 · Servo bias cleared before carry

- **Location (legacy):** 2140-2143 (consumer that would misuse it: 2412-2415)
- **Protects against:** Carrying home with a stale cube-aimed bias: the carry targets REST, so target - stale_bias would park the arm |bias| off REST and then step-jump when the bias is later dropped
- **Trigger:** Fresh grasp_detected == True arrives for a sent grab (the moment the carry lifecycle starts)
- **Response:** h['bias'] = None before the lift waypoints are built — 'servo bias aimed at the cube; the carry targets REST — a stale bias would park |bias| off and step-jump'
- **Constants:** none
- **State:** self._held[arm]['bias']
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (t_grasp_carry_side executes the line, but the rig runs visual_servo=False so bias is already None — the mechanism itself is unexercised)
- **Incident:** Comment 2141-2143 (step-jump class: same family as the catapult/snap incidents)
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)

### SAFE-SERVO-019 · Idle-camera PARK-at-neutral (servo starvation guard, gaze side)

- **Location (legacy):** 706-712
- **Protects against:** A sweeping idle camera grazing the arm/cube region: its single-tag detections OVERWRITE the tracking camera's entries in the monitor's one-entry-per-key targets dict (last camera wins), starving the visual servo of same-port gripper sightings
- **Trigger:** A camera has no claimed fresh cube and nothing unclaimed is pending
- **Response:** Camera parks at yaw=0, pitch=0 instead of running the scan serpentine — it never crosses the working region while the servo depends on the other camera's stream
- **Constants:** DET_LOST_RESCAN_S (fresh-claim window feeding the decision)
- **State:** cube.camera_port, gaze slots via GAZE_ORDER[port]
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Comment 707-711: 'a sweeping idle camera is not harmless'; same last-camera-wins mechanism the same-camera rule defends against downstream
- **New location (S6):** auto_operator/servo.py (full trust-rule chain)


## Gripper / EE chain

### SAFE-GRIPPER-001 · ee_alive-gated startup zero ritual

- **Location (legacy):** 2553-2584 (constants 232-237)
- **Protects against:** Fire-and-forget zero_gripper commands silently swallowed by a dead EE chain; mission then runs on fake pinches with never-calibrated grippers, or a closed gripper is rammed into a cube
- **Trigger:** grasp=True and not self._hands_opened at mission tick; zeros dispatched only once self._ee_alive(telemetry) returns True
- **Response:** Queue zero_gripper for both sides, set _zeros_t=now and _hands_opened=True; until ee_alive is True the operator waits (starting _ee_wait_t0 clock), sending nothing
- **Constants:** EE_ALIVE_WAIT_S=15.0
- **State:** _hands_opened (one-shot latch), _ee_wait_t0, _zeros_t, _ee_queue
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_grasp_carry_side (asserts zero_gripper pair emitted for both sides); t_chain_dead (asserts no zero ever sent when dead)
- **Incident:** 07-16 audit: EE command/status path is lossy (latest-value-wins, no acks) and liveness used to be invisible — zeros were silently swallowed and the mission ran on fake pinches
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-002 · latched grasp_detected NOTE — zero anyway

- **Location (legacy):** 2566-2581
- **Protects against:** Skipping the zero ritual on every operator restart after any successful grab, silently losing gripper calibration for the whole mission (grasp_detected latches robot-side; hand_open never clears it)
- **Trigger:** During startup zeroing, a side's telemetry grasp_detected reads truthy
- **Response:** Print loud NOTE that the flag is latched from an earlier grab and ZERO ANYWAY; the EMPTY-GRIPPERS-AT-START operator protocol is the real protection against calibrating a held object's width as closed
- **Constants:** none (behavioral rule)
- **State:** telemetry ee.<side>.grasp_detected
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-16 bench bug: earlier code skipped zeroing when grasp_detected was latched True and silently lost the ritual on every restart after a success
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-003 · blind-zero fallback (no ee_alive field)

- **Location (legacy):** 2585-2593
- **Protects against:** An un-patched real_env (telemetry lacking ee_alive) would otherwise permanently block zeroing or be silently mistaken for a live chain
- **Trigger:** (now - _ee_wait_t0) > EE_ALIVE_WAIT_S and _ee_alive(telemetry) is None (field absent)
- **Response:** Queue both zero_gripper commands BLIND with a loud WARNING that the chain cannot be verified; set _zeros_t and _hands_opened
- **Constants:** EE_ALIVE_WAIT_S=15.0
- **State:** _ee_wait_t0, _zeros_t, _hands_opened
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none
- **Incident:** 07-16 audit hardening: real_env liveness patch may be missing; caller must degrade loudly, never assume alive (docstring of _ee_alive, lines 1832-1838)
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-004 · EE chain-dead declaration + mission-wide grasp disable

- **Location (legacy):** 2585-2601, plus guards at 1246-1247 and 1553
- **Protects against:** Operator mislabels a DEAD EE chain as physical empty pinches and could hand_open a gripper that is actually holding the cube (dropping it); or first reach blocks forever waiting for zeros that can never happen
- **Trigger:** ee_alive stayed False for the full EE_ALIVE_WAIT_S startup window
- **Response:** Set _ee_chain_dead=True, print banner 'GRASPING DISABLED for this mission', continue as classic reach-and-hold with the gripper inert; _ee_chain_dead also SKIPS the zero-grace reach gate and every grab precondition
- **Constants:** EE_ALIVE_WAIT_S=15.0
- **State:** _ee_chain_dead (permanent for the mission), _ee_wait_t0
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_chain_dead (declared dead, zero/grab never sent, mission continues as classic hold)
- **Incident:** 07-16 grasp-chain audit: chain liveness used to be invisible; dead chain looked identical to empty pinches
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-005 · zero-open gate on the FIRST reach + GRASP_ZERO_GRACE

- **Location (legacy):** 1241-1256 (constant 268-269)
- **Protects against:** With --reach-standoff 0 the jaw centre is driven ONTO the cube centre — a still-closed (un-zeroed) gripper would be rammed into the cube; a hand_grab sent during ZEROING_IN_PROGRESS is rejected unseen by the service
- **Trigger:** grasp=True, chain not dead, and (_hands_opened is False OR (now - _zeros_t) < GRASP_ZERO_GRACE_S)
- **Response:** Hold the first reach entirely (return from phase logic), printing a throttled (3.0 s) diagnostic that reaches wait for the grippers to finish zeroing OPEN
- **Constants:** GRASP_ZERO_GRACE_S=10.0; diag throttle 3.0 s (_zero_wait_diag_t)
- **State:** _hands_opened, _zeros_t, _ee_chain_dead, _zero_wait_diag_t
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none directly asserted (implicitly exercised by every grasp test's pre-reach wait, e.g. t_grasp_carry_side)
- **Incident:** Grace exists because a grab during the service's ZEROING_IN_PROGRESS is rejected unseen (comment at 268-269)
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-006 · grab-dispatch precondition chain + WHY-diagnostics

- **Location (legacy):** 1545-1595 (constants 227-231)
- **Protects against:** Grabbing air (timeout-registered hold that never converged), grabbing during zero settling, grabbing a cube that moved or vanished, or a hold that never grabs while staying SILENT about why
- **Trigger:** _maybe_send_grab each tick for a registered hold with grasp/track/restream/home all clear; all preconditions must pass: ee_alive not False, zero grace elapsed, (now - grasp_move_t) > GRASP_SETTLE_S, EE pos_actual present, cube fresh within DYN_LOST_GRACE_S, EE-to-target distance < GRASP_EE_OK_M
- **Response:** All pass: queue hand_grab, set h.grasp='sent', h.grasp_t0=now, capture h.grasp_base baseline. Any failure: NO grab, print throttled (3.0 s) message naming the exact unmet precondition with full target/EE coordinates
- **Constants:** GRASP_SETTLE_S=1.0, GRASP_EE_OK_M=0.06, DYN_LOST_GRACE_S=1.0; diag throttle 3.0 s (grasp_diag_t)
- **State:** h.grasp, h.grasp_t0, h.grasp_base, h.grasp_move_t (restarts settle on retrack), h.grasp_diag_t
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_grasp_carry_side, t_retry_then_fail (settle-then-grab path); why-diagnostics themselves: none
- **Incident:** 07-16 audit was commissioned against silent waiting — a hold that never grabs must say WHY; GRASP_EE_OK_M exists because a timeout-registered hold never converged: never grab air
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-007 · grab may fire from GO/REACH (dual-parallel first arrival)

- **Location (legacy):** 1547-1552 docstring; call site 2207
- **Protects against:** A hold registered while the OTHER arm is still reaching would wait to grab, letting the cube be nudged/moved before closing
- **Trigger:** Hold registered during the other arm's GO/REACH (dual-parallel first-arrival)
- **Response:** Grab immediately — the gripper rides its own serial bus, never the 14-joint stream, so this cannot violate modal-wire exclusivity (Guard 1); only the carry-home track queues
- **Constants:** none
- **State:** h.grasp
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_dual_parallel_sec_first_grab, t_dual_parallel_front (grab sent while other arm still flew)
- **Incident:** 07-17 audit: first-arrived arm must grab while the other still flies
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-008 · hold-still during grab (state 'sent')

- **Location (legacy):** 2128-2131
- **Protects against:** A mid-grab follow/retract would yank the cube out of the closing fingers; the fingers occlude the cube's tags so any 'motion' seen is garbage
- **Trigger:** h.grasp == 'sent'
- **Response:** The arm holds still: no dynamic follow, no retract, until the result resolves
- **Constants:** none
- **State:** h.grasp
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (implicitly), t_lost_result
- **Incident:** Fingers occlude the tags; a mid-grab retract would yank the cube (comment 2129-2130)
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-009 · grasp result policy: FRESH-only via baseline compare

- **Location (legacy):** 2136-2139 (baseline captured 1587-1588)
- **Protects against:** Trusting a STALE latched grasp_detected (it latches robot-side and never clears on hand_open) as this grab's result — false success on a restart, or acting on a lost/duplicated status packet
- **Trigger:** h.grasp == 'sent': fresh_result = (cur is not None and cur != h.grasp_base) where grasp_base was snapshotted at dispatch
- **Response:** Only a value CHANGED from the pre-grab baseline is acted on; None/unchanged falls through to the timeout/chain-loss branches (which never reopen)
- **Constants:** none (comparison rule)
- **State:** h.grasp_base, telemetry grasp_detected
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_retry_then_fail (None-to-False counts as fresh), t_grasp_carry_side (True-to-lifting), t_lost_result (unchanged never resolves as success)
- **Incident:** 07-16 audit RESULT POLICY: a timeout/unchanged/None result means the packet was lost or the chain died — the gripper MAY be holding the cube
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-010 · empty-pinch retry loop + GRASP_MAX_TRIES

- **Location (legacy):** 2181-2192 (constant 231)
- **Protects against:** Endless pinching at a cube that is not there / not graspable, or giving up after a single slip
- **Trigger:** Fresh result with grasp_detected False (empty pinch)
- **Response:** Increment h.grasp_tries, queue hand_open; below GRASP_MAX_TRIES: h.grasp=None and grasp_move_t=now (full re-settle before retry); at GRASP_MAX_TRIES: h.grasp='failed' — degrade to plain hold, hold entry and cube ownership kept
- **Constants:** GRASP_MAX_TRIES=2
- **State:** h.grasp_tries, h.grasp, h.grasp_move_t
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_retry_then_fail (reopen after pinch 1, second grab, failed into plain hold that still dyn-follows)
- **Incident:** Plain-hold degradation keeps the arm on the cube instead of abandoning it
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-011 · mid-grab chain loss: NEVER reopen, disable grasping

- **Location (legacy):** 2193-2198
- **Protects against:** hand_open sent to a gripper whose state is unknown after the EE service died mid-grab — would DROP a successfully grasped cube
- **Trigger:** h.grasp == 'sent' and _ee_alive(telemetry) is False (no fresh result)
- **Response:** h.grasp='failed', _ee_chain_dead=True (grasping disabled for the rest of the mission), keep the hold, do NOT queue hand_open; loud print 'gripper state UNKNOWN'
- **Constants:** none
- **State:** _ee_chain_dead, h.grasp
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (t_chain_dead covers startup-dead only, not mid-grab loss)
- **Incident:** 07-16 audit: uncertainty used to be resolved by opening — which dropped a successfully grasped cube
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-012 · grasp result timeout: keep hold, NEVER open

- **Location (legacy):** 2199-2205 (constant 230)
- **Protects against:** A lost status packet / silent chain treated as failure-and-reopen would drop a cube that is actually gripped
- **Trigger:** h.grasp == 'sent' and (now - h.grasp_t0) > GRASP_RESULT_TIMEOUT_S with no fresh result
- **Response:** h.grasp='failed' (plain hold), KEEP the hold, do NOT hand_open; print instructing the operator to verify the gripper by eye
- **Constants:** GRASP_RESULT_TIMEOUT_S=20.0 (worst-case hand_grab ~12 s service-side + margin)
- **State:** h.grasp_t0, h.grasp
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_lost_result (failed WITHOUT reopening, cube ownership kept)
- **Incident:** 07-16 audit: never hand_open on uncertainty
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-013 · ee_action repeat-send (lossy channel hardening)

- **Location (legacy):** 2666-2701 (constant 238-244)
- **Protects against:** One-shot gripper commands silently eaten by a real_env stall — the 'gripper never opened/closed' class; channel is latest-value-wins with no ack at BOTH hops (operator→real_env→service)
- **Trigger:** Any queued ee_action payload being published in the 9874 packet
- **Response:** Each command re-sent for EE_ACTION_REPEAT_TICKS packets (1.5 s at 20 Hz). Duplicates are safe: service rejects a busy side (grab/zero) and hand_open is idempotent; ee_action does NOT refresh real_env's arm-stream clock so it never disturbs joint/Cartesian arbitration
- **Constants:** EE_ACTION_REPEAT_TICKS=30 (raised from 5/250 ms which did not cover real_env stalls)
- **State:** _ee_queue, _ee_current=[payload, remaining_repeats, sent_this_burst]
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_repeat_send (both sides send FULL repeat budget)
- **Incident:** 07-16: channel seen eating commands live on hardware; 5 repeats insufficient
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-014 · queue yield>=5 + rotation + same-side supersession + _ticks bookkeeping

- **Location (legacy):** 2674-2695
- **Protects against:** Back-to-back commands starving each other: dropping the yielded remainder gave the FIRST of a pair (left startup zero) a single 0.25 s burst that a 9874 subscriber mid-reconnect never saw — the recurring 'left gripper never opened'
- **Trigger:** _ee_current has sent >=5 packets this burst AND another command waits in _ee_queue
- **Response:** Yield the wire but ROTATE the unsent remainder to the back of the queue carrying '_ticks'=remaining (spreads both bursts across the whole ~3 s window); a queued newer command for the SAME side supersedes the remainder instead of resurrecting behind it; '_ticks' is stripped before the payload rides the packet
- **Constants:** yield threshold literal 5 (sent_this_burst >= 5); EE_ACTION_REPEAT_TICKS=30 default when _ticks absent
- **State:** _ee_current[2] sent_this_burst, _ee_queue entries' _ticks key
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_repeat_send (>=5 consecutive then yield, full budget both sides, spread past 2 s reconnect dead-window, _ticks never leaks into packet); same-side supersession: none
- **Incident:** 07-17: dropping the remainder starved the first of a back-to-back pair; rotation fix
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-015 · servo-bias clear on grasp success

- **Location (legacy):** 2141-2143
- **Protects against:** The visual-servo bias was aimed at the CUBE; the carry targets REST — a stale bias would park the arm |bias| off home and cause a step-jump
- **Trigger:** Fresh grasp_detected True received
- **Response:** h.bias = None before building the lift waypoints
- **Constants:** none
- **State:** h.bias
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none directly (t_* rigs run visual_servo=False)
- **Incident:** Comment inline at 2141-2143
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-016 · lift-off clearance waypoints + envelope/sector guards

- **Location (legacy):** 2144-2180 build, _lift_arc 959-986 (constants 272-280)
- **Protects against:** Walking the carry straight from the grasp pose catches the gripper/cube on the STAND the cubes sit on; an unguarded arc could sweep outside the sector basin or safety envelope (cross-body / collision class)
- **Trigger:** Fresh grasp_detected True
- **Response:** wp1 = target + GRASP_LIFT_UP_M vertical; then a constant-radius horizontal arc of the FULL GRASP_LIFT_OUT_M toward the sector station (never capped/skipped — clearance beats alignment; overshoot harmless), ~2 cm spacing, each point sector- AND _target_safe-guarded, truncating at first violation (partial arc beats no arc); wp1 itself unsafe → skip lift entirely, go straight to 'grasped' (carry from here); side-home mode: no arc, the straight gentle ramp home IS the sweep
- **Constants:** GRASP_LIFT_UP_M=0.05, GRASP_LIFT_OUT_M=0.08
- **State:** h.grasp='lifting', h.lift_wps, h.lift_t0, h.target, h.restream
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side (exact arc-endpoint assertion), t_side_home_basic (lift then straight ramp)
- **Incident:** 07-17 (user): gripper/cube caught on the cube stand; 07-17 user: early arc stop can leave the gripper still over the stand when retract starts
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-017 · lift timeout + no-EE-seed hold

- **Location (legacy):** 2078-2083 timeout, 2091-2096 seed (constant 280)
- **Protects against:** A jammed/obstructed clearance move stranding the mission in 'lifting' forever; or seeding the lift ramp from nothing and snapping the arm
- **Trigger:** (now - h.lift_t0) > GRASP_LIFT_TIMEOUT_S; or h.restream is None with no pos_actual this tick
- **Response:** Timeout: WARN and demote to 'grasped' (carry home from wherever it is), restream cleared; no seed: hold in place and retry next tick — the ramp only ever starts from a measured EE position
- **Constants:** GRASP_LIFT_TIMEOUT_S=8.0
- **State:** h.lift_t0, h.restream, h.grasp
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (lift success paths covered; timeout branch untested)
- **Incident:** Budget-then-carry-anyway prevents an absorbing lifting state
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-018 · lift wire-pause during other arm's joint traffic

- **Location (legacy):** 2084-2090
- **Protects against:** While another arm's staircase/reach owns the wire in JOINT mode, our Cartesian lift point is not being published — advancing the ramp anyway makes the arm JUMP the accumulated distance when publishing resumes
- **Trigger:** h.grasp=='lifting' and (self._reach_tm is not None or any held arm has an active track)
- **Response:** continue without advancing: the restream ramp pauses and just holds its point until the wire is free (paused automatically during the other arm's GO/REACH by branch ordering)
- **Constants:** none
- **State:** _reach_tm, h.track (all holds), h.restream
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_serialized (A's lift/carry defer under B's GO/REACH, Guard-1 track violations == 0)
- **Incident:** 'a resumed jump otherwise' (inline comment); same modal-wire exclusivity class as the 07-15 cross-body audit
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-019 · carry-home gating (state 'grasped')

- **Location (legacy):** 2109-2127
- **Protects against:** A carry staircase starting while another track/reach/dual-parallel Cartesian leg owns the wire — mode-flip mid-staircase lets re-seeded IK drag a far-basin arm across the torso
- **Trigger:** h.grasp=='grasped': carry staircase starts only when h.track is None AND self._reach_tm is None AND self._sec is None (dual-parallel secondary must finish); sector_idx==0 skips the staircase and enters carried directly (needs pos_actual to seed the park ramp)
- **Response:** Defer frozen at the last target until the wire frees (first grasped walks home first — carries are ordinary tracks under the one-staircase-at-a-time token and Guard-1/Guard-2 serialization); drop detection still runs while waiting
- **Constants:** none (token-based serialization)
- **State:** h.track, _reach_tm, _sec, h.sector_idx, h.grasp
- **Flags:** fail-closed, timing-insensitive
- **Tests:** t_dual_serialized (first grasped carried home first; A never tracked during B's GO/REACH), t_grasp_carry_side (carry used the joint staircase)
- **Incident:** Serialization inherited from the 07-15 dual-arm cross-body jump fix (0519d3c)
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-020 · carry staircase semantics: no cube-decisions, timed-out hop crawl-retry

- **Location (legacy):** 1989-2011
- **Protects against:** During carry the cube is IN the hand so detections track the hand — any re-extend/reverse/release decision on 'cube position' is meaningless and would mis-drive the arm; a jammed/sagging hop could strand the carry
- **Trigger:** h.track active with h.grasp=='grasped'
- **Response:** Walk the staircase to front ignoring all cube-position branches; on tm.timed_out print throttled (5.0 s) WARNING and keep retrying toward front at crawl rate with the cube in hand; at 'done' with pos_actual present → _enter_carried (no EE this tick: stay settled and retry)
- **Constants:** warn throttle 5.0 s (carry_warn_t)
- **State:** h.track, tm.timed_out, h.carry_warn_t, h.sector/sector_idx reset to front/0
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_grasp_carry_side (staircase walked home, arm at front posture); timed-out hop branch: none
- **Incident:** Jam/sag on hardware could time out a hop mid-carry — never release, watch the arm
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-021 · _enter_carried: terminal park + camera-claim release

- **Location (legacy):** 1840-1855 (+ carried upkeep 2060-2072)
- **Protects against:** A cube riding in the gripper keeping its camera claim STARVES the other cube's search when only one camera survives; a raw REST target would snap the arm instead of ramping
- **Trigger:** Carry staircase done at front (or sector_idx==0 direct) with measured pos_actual available
- **Response:** restream seeded from measured EE, target=rest_ee[arm], state='carrying', grasp='carried' (terminal per-arm state — no following, no releases, only drop detection can exit); cube.no_claim=True and cube.camera_port=None released; carried upkeep finishes the gentle park ramp to REST
- **Constants:** ramp rate STAGE_REACH_EE_RATE=0.08 m/s via _gentle_approach
- **State:** h.restream, h.target, h.state, h.grasp, cube.no_claim, cube.camera_port
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** t_grasp_carry_side / t_dual_parallel_front (carried at REST, hold kept); claim-release itself asserted: none
- **Incident:** 07-16 audit: keeping the claim starved the other cube's search with one surviving camera
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)

### SAFE-GRIPPER-022 · drop detection (grasped/carried holds)

- **Location (legacy):** 1857-1890 (constants 270-271); call sites 2064, 2113
- **Protects against:** A dropped cube staying 'held' forever — mission ends 'complete' with an empty gripper (the chain has no other feedback: grasp_detected never updates after the grab)
- **Trigger:** Hold in 'grasped'/'carried' with h.track None (mid-staircase SKIPPED — fingers occlude tags): cube fresh (DYN_LOST_GRACE_S) and seen > GRASP_DROP_DIST_M from pos_actual, persisting > GRASP_DROP_PERSIST_S (drop_t0 debounce resets the instant the far-sighting stops)
- **Response:** Print DROPPED, queue hand_open, cube.no_claim=False (needs an eye again), h.grasp=None, grasp_move_t=now, restream=None, state='tracking' — revert to a normal hold; existing dynamics re-reach and re-grab where it fell (or retract if unservable)
- **Constants:** GRASP_DROP_DIST_M=0.15, GRASP_DROP_PERSIST_S=2.0, DYN_LOST_GRACE_S=1.0
- **State:** h.drop_t0, h.grasp, h.grasp_move_t, h.restream, h.state, cube.no_claim
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_drop_detection (detected, reverted to live hold, exactly one hand_open, ownership kept); t_dual_parallel_front's det() harness notes stale table positions correctly trip it
- **Incident:** 07-16 audit: a dropped cube stayed held forever and the mission ended complete with an empty gripper
- **New location (S6):** auto_operator/arbitration.py (attach_ee_action burst) + arms.py (grasp lifecycle) + operator tick() (zero ritual/ee_alive)


## Shutdown & silence

### SAFE-SHUTDOWN-001 · main() finally nav-zero on exit

- **Location (legacy):** 2845-2849 (break path 2842-2843)
- **Protects against:** Robot keeps walking on the last nonzero nav command after operator death/Ctrl+C — the explicit zero beats the robot-side 1 s nav-silence failsafe window instead of relying on it
- **Trigger:** Any exit from the main loop: Ctrl+C, uncaught exception, or the Phase.DONE 'mission complete' break
- **Response:** finally: publish {"nav_cmd": [0.0, 0.0, 0.0]} on 9873 (skipped if gaze-only, nav_pub is None), time.sleep(0.1) to let the publish flush, then det_sub.stop(); tel_sub.stop()
- **Constants:** literal [0.0,0.0,0.0]; 0.1 s flush sleep; robot-side nav silence failsafe = 1 s (header line 53)
- **State:** nav_pub, det_sub, tel_sub
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (test_grasp_carry.py drives op.tick() directly; main() is never executed)
- **Incident:** Header safety contract (lines 52-54): stale/frozen detections zero the walk command; failsafes stay armed only because the operator publishes continuously
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-002 · Telemetry freshness gate in main (9870)

- **Location (legacy):** 2814-2822
- **Protects against:** A frozen/restarted 9870 stream would keep feeding yesterday's encoders to every posture guard (fail-open): NNGSubscriber.data holds the LAST packet forever, so stale joint_pos would pass the startup-posture, near-home, and EE-seed checks
- **Trigger:** tel_sub.data_id unchanged for > 0.5 s (literal, unnamed constant) since last_tel_arrival
- **Response:** telemetry = {} is passed to tick() — tick sees no telemetry at all, so every downstream guard fails CLOSED (startup posture gate blocks the mission, _arm_near_front returns False, _arm_packet returns None → arm-stream silence, robot holds last target)
- **Constants:** stale window = 0.5 s (unnamed literal); state clocked by time.monotonic()
- **State:** last_tel_id, last_tel_arrival
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (rig hands tick() fresh synthetic telemetry every step; the main-loop gate is never exercised)
- **Incident:** 07-15 audit: telemetry-freshness hole found during the fail-open class sweep — 'stale ⇒ tick() sees no telemetry at all and every guard fails CLOSED'
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-003 · Detection freshness gate in main (monitor dead/frozen)

- **Location (legacy):** 2806-2813
- **Protects against:** Monitor frozen/dead: retaining the last detection dict would keep re-stamping cube.last_seen every tick and the robot would walk forever on dead perception
- **Trigger:** det_sub.data_id unchanged for > 0.5 s (literal, unnamed constant) since last_det_arrival
- **Response:** det_targets = {} — cubes stop being re-stamped; inside tick, cube.fresh(now, DET_STALE_S) then fails and GO zeroes the walk command (header contract lines 52-53)
- **Constants:** arrival stale window = 0.5 s (unnamed literal); distinct from DET_STALE_S = 0.5 (line 142) which gates per-cube capture_stamp age inside tick
- **State:** last_det_id, last_det_arrival, det_targets
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (rig injects detections dicts directly into tick(); the main-loop arrival clock is bypassed)
- **Incident:** Inline comment: 'Monitor frozen/dead: retaining the last dict would keep re-stamping cube.last_seen and the robot would walk forever on dead perception'
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-004 · Hold-last-target contract — no crawl-home under --use-ik

- **Location (legacy):** 2635-2649 (arbitration), 2318-2321 (_tick_hold docstring), 2402-2404 (_arm_packet docstring)
- **Protects against:** Any handler designed on the assumption that arm-stream silence retracts/homes the arm — it does not: with --use-ik there is NO robot-side crawl-home/retract-on-silence failsafe (07-15 contract audit found that path is dead code); silence == freeze at last published target. Companion guard: publishing a Cartesian packet mid-staircase would flip the robot's wire mode and let re-seeded IK drag a far-basin arm toward REST
- **Trigger:** Any tick with no safe arm value (telemetry blip, unseeded home glide, off-front idle arm) → _arm_packet returns None → caller goes silent; or staircase_active with a non-joint payload → arm_targets popped from the packet
- **Response:** Deliberate stream silence: real_env HOLDS ITS LAST TARGET — position held, nothing moves; while any staircase is in flight (staircase_active = _reach_tm is not None or any held track) the stream stays joint-mode-only
- **Constants:** real_env arm-silence re-engage window = 0.5 s (header line 54)
- **State:** _reach_tm, _held[*]['track'], staircase_active, ret_payload
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_dual_parallel_unseeded_abandon covers the adjacent no-catapult-packet property; the hold-last-target silence semantics themselves: none (rig has no silence-timeout model)
- **Incident:** 07-15 contract audit: crawl-home path is dead code with --use-ik; after Ctrl+C the arms HOLD the last published targets
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-005 · Never-go-arm-silent after engagement / explicit REST keep-alive

- **Location (legacy):** 504-511 (_arm_engaged comment), 2650-2662 (keep-alive), 2663-2664 (latch)
- **Protects against:** real_env keeps the last Cartesian target forever and RE-ENGAGES its IK on it after 0.5 s of arm-silence — silence after a staircase snapped the arm back to a stale cube target (observed on hardware 07-14)
- **Trigger:** Tick produced no arm_targets and no staircase is in flight
- **Response:** Publish an explicit REST keep-alive ('explicit REST beats stream silence', 07-17 user spec: fires from MISSION START) — but ONLY when _idle_arms_home(jpos) verifies every unheld arm is near a home posture or mid-glide; otherwise stay silent so real_env holds the current pose (an arm stranded in the side/rear basin must never be pulled home across the torso)
- **Constants:** real_env arm-silence window = 0.5 s; OP_RATE_HZ = 20.0; FRONT_HOME_TOL_RAD (near-home check); STAGE_REACH_EE_RATE = 0.08 m/s glide
- **State:** _arm_engaged (set at 2664 but never read — vestigial latch; the enforcement is the keep-alive branch itself), _home_stream, _held, _sec
- **Flags:** fail-closed, timing-sensitive
- **Tests:** t_startup_lift and t_glide_no_mans_land (keep-alive publishes the gentle home glide from mission start; glide never freezes mid-posture-no-man's-land); the snap-back-on-silence hardware failure itself: none
- **Incident:** Hardware 07-14: silence after a staircase snapped the arm back to a stale cube target; 07-17 user spec extended the keep-alive to mission start
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-006 · Shutdown protocol for held cubes (HOLD is terminal)

- **Location (legacy):** 2310-2321 (_tick_hold docstring), 520 (hold_after_reach)
- **Protects against:** Operator killed while arms hold cubes and someone expects the arms to come home — they freeze extended, gripping, at the last published targets (no retract-on-silence with --use-ik)
- **Trigger:** Ctrl+C during HOLD (HOLD runs until Ctrl+C; default hold_after_reach=True makes it the mission-terminal phase)
- **Response:** Documented human shutdown protocol (07-15 contract audit note): remove the cubes from the hands, let the per-arm holds release and the arms come home under the operator's own logic, THEN exit the operator
- **Constants:** hold_after_reach default True (Args line 2725; --no-hold-after-reach restores legacy park-and-DONE)
- **State:** phase == Phase.HOLD, _held
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none (procedural/human protocol; tests end at carried/HOLD and never model operator exit)
- **Incident:** 07-15 contract audit: with --use-ik there is NO robot-side retract-on-silence — after Ctrl+C the arms HOLD the last published targets
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-007 · Continuous publish keeps robot-side silence failsafes armed

- **Location (legacy):** 52-54 (header), 85-86 (OP_RATE_HZ), 2844 (loop pacing)
- **Protects against:** Operator ticking slower than the robot-side windows would spuriously trip the failsafes mid-mission (arm 0.5 s IK re-engage on a stale target, nav 1 s stop, gaze 2 s hold) — and conversely, genuine operator death is only detected robot-side BECAUSE these timeouts stay armed
- **Trigger:** Every loop iteration: time.sleep(max(0.0, DT - elapsed)) paces the loop at OP_RATE_HZ
- **Response:** Publish nav/arm/gaze streams at 20 Hz, faster than every robot-side silence window; header contract: 'every silence failsafe on the robot side (nav 1 s, arm 0.5 s, gaze 2 s) stays armed because we publish continuously at OP_RATE_HZ'
- **Constants:** OP_RATE_HZ = 20.0 ('> all robot-side timeout rates'), DT = 0.05 s; robot-side windows nav 1 s / arm 0.5 s / gaze 2 s
- **State:** t0 loop clock
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (rig steps tick() with a synthetic clock; real-time pacing untested)
- **Incident:** GATE_EVAL_PERIOD_S comment (124-132): unthrottled gate scoring blew the 50 ms tick budget — tick-rate integrity is a known hazard
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-008 · DONE behavior: entry, gaze reference hold, mission-complete break

- **Location (legacy):** 1232 (entry), 2703-2706 (gaze stops), 2842-2843 (break), 1674-1678 (dual-parallel refusal), 2725-2726 (hold_after_reach arg)
- **Protects against:** At DONE the operator stops publishing gaze_targets — safe only because the reference holds robot-side (gaze 2 s failsafe); and in legacy park-and-DONE mode a dual-parallel secondary hold would strand the mission in HOLD forever with no exit (07-17 audit), so _try_latch_secondary refuses when hold_after_reach is False
- **Trigger:** SEARCH finds nothing pending AND no arm holds a cube → Phase.DONE (reachable only in legacy --no-hold-after-reach park mode; default missions terminate in HOLD until Ctrl+C)
- **Response:** tick: no phase handler runs (nav desires stay 0.0, ramped to zero; keep-alive REST may still publish for home arms), gaze_targets omitted from the packet ('then reference holds robot-side'); main: prints 'mission complete', breaks the loop → finally nav-zero fires
- **Constants:** hold_after_reach default True; PARK_HOLD_S = 3.0 (PARK streams rest before SEARCH resolves to DONE)
- **State:** phase, hold_after_reach, _held, _gaze_seeded
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (all tests run with hold_after_reach default True and end in HOLD/carried; Phase.DONE is never reached)
- **Incident:** 07-17 audit: legacy park-and-DONE has no hold machinery — a secondary hold would strand the mission in HOLD forever
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-009 · --no-walk hard-zero of nav at the publish point

- **Location (legacy):** 2826-2827 (zero), 2739-2743 (arg contract), 2792-2794 (banner)
- **Protects against:** A STANDING (balance-in-place) mission commanded to step by GO would fall; the zero is applied at the last possible point so the state machine (GO gating, arrival envelope) is bit-identical to the walking mission
- **Trigger:** args.walk is False and a nav packet is about to be published
- **Response:** nav overwritten to [0.0, 0.0, 0.0] immediately before nav_pub.publish — 'hard-zeroes every nav command'; caveat in the arg doc: cubes must sit inside the arrival gate's envelope or GO waits forever
- **Constants:** args.walk (default True); publish literal [0.0, 0.0, 0.0]
- **State:** args.walk
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (main()-level; the rig never publishes nav)
- **Incident:** Standing-0.8 mission profile validated 07-16 (grasp-carry memory: standing 0.8 validated same day as the hang-height incident class)
- **New location (S6):** operator main() (unchanged)

### SAFE-SHUTDOWN-010 · Gaze-only mode: 9873 never bound or published

- **Location (legacy):** 2781 (nav_pub=None), 2527-2534 (nav=None), 2825 (publish guard), 47-50 (header)
- **Protects against:** A gaze-only operator publishing even zero nav would fight/mask another 9873 commander (header: 'Nothing else may bind 9873 here'); with no publisher at all, the robot's 1 s nav-silence failsafe (or another commander) owns the legs
- **Trigger:** --gaze-only flag
- **Response:** nav_pub = None so 9873 is never bound; tick returns nav=None and main's 'nav is not None and nav_pub is not None' guard skips publishing; the finally nav-zero is likewise skipped (nothing was ever commanded)
- **Constants:** args.gaze_only; grasp/dual_parallel/side_home all force-disabled with it (2775-2778)
- **State:** nav_pub, args.gaze_only
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** Header prerequisite note lines 47-50: gaze-only needs neither --use-ik nor 9873
- **New location (S6):** operator main() (unchanged)


## Gap sweep (uncategorized)

### SAFE-GAPS-001 · Publisher import shadowing workaround (retired 2026-08-22)

- **Location (legacy):** 68-80
- **Protects against:** Silently importing legged_env_v2's shadowing `common` package instead of control/common — wrong/absent NNGPublisher/NNGSubscriber means the 9873/9874 command sockets never behave as validated (or the operator crashes at import).
- **Trigger:** Module import: sys.path.insert of <legged_env_v2> (+/mj_envs) at lines 68-69 puts the shadowing package FIRST on sys.path.
- **Response:** control/common/publisher.py is loaded by absolute file path via importlib.util.spec_from_file_location('_control_publisher', ...) relative to __file__, bypassing sys.path resolution entirely (same workaround as humanoid_monitor).
- **Constants:** paths='<legged_env_v2>', '<legged_env_v2>/mj_envs'; module='_control_publisher'
- **State:** _spec, _pub_mod (module-level)
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (test_grasp_carry.py's load() executes the workaround incidentally but asserts nothing about it)
- **Incident:** Comment: legged_env_v2 ships its own `common` package which shadows control/common on sys.path.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)
- **Retired (2026-08-22):** the control-side package was renamed `control/common` → `control/ipc`, a name no other `sys.path` entry provides, so the operator now does a plain `from ipc.publisher import NNGPublisher, NNGSubscriber` and `_spec` / `_pub_mod` no longer exist. The protection is structural (no bare `common` on the control side) instead of per-file. For the record, the shadow that actually bit was `perception/common`, put at sys.path[0] by humanoid_monitor's VISUAL_SERVOING_ROOT insert — not an upstream package, as the legacy comment claimed.

### SAFE-GAPS-002 · os.chdir for ReachabilityGate scorer load

- **Location (legacy):** 2766-2770
- **Protects against:** Gate constructed from the wrong CWD loads scorers from a wrong/missing relative cache path — arrival/reachability verdicts on garbage weights (walk too close or never stop); plus a permanent process-wide CWD side effect on any later relative-path I/O.
- **Trigger:** main() with use_gate=True and not gaze_only, before ReachabilityGate(device='cpu') import+construction.
- **Response:** os.chdir('<legged_env_v2>/mj_envs') so the gate's relative cache path resolves; gate imported UNCHANGED from sim (mission docstring contract).
- **Constants:** chdir path='<legged_env_v2>/mj_envs'
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none (all tests use gate=None or a FakeGate)
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-003 · _wrap principal-branch angle math

- **Location (legacy):** 676-678 (def); uses 596, 614, 629, 635, 1208
- **Protects against:** A ±2π branch flip at the ±π cut: the rear-mounted right camera tracking a FRONT cube sits exactly on the cut, where detection noise flips the raw yaw by 2π and sends the gimbal the long way around (yaw ROM is ±4.71 rad so off-principal branches are legal — cable wind-up / slew slam); line 1208's backward-walk bearing error is also wrong by 2π without it.
- **Trigger:** Every aim computation (_probe_cam_frame, _aim_angles closed form + FK refinement + line-635 nearest-branch unwrap onto the ACTUAL joint) and every backward GO leg.
- **Response:** (a + pi) % (2*pi) - pi; at line 635 the commanded yaw is re-based onto the branch nearest the encoder reading, so the published reference stays continuous with the physical joint.
- **Constants:** yaw ROM ±4.71 rad (comment, line 606)
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-004 · _ramp per-channel accel limiter (incl. fail-closed nav zeroing)

- **Location (legacy):** 1212-1214 (def); callers 2546-2547, 2709-2710; interplay 1351
- **Protects against:** Instantaneous walk-command steps kicking the balance controller; and (via the |vx|<0.02 check at 1351) latching a reach target while the base is still decelerating — the ramp IS the stop-before-latch sequencer.
- **Trigger:** Every tick output (vx, wz), including the pre-startup fail-closed branch which ramps nav to zero rather than snapping.
- **Response:** Output steps clipped to ACCEL_LIMIT*DT = 0.15 m/s (or rad/s — wz shares the linear-accel constant, a refactor trap) per tick, applied AFTER the hard clamps np.clip(vx, ±CRUISE_VX) / np.clip(wz, ±WZ_MAX).
- **Constants:** ACCEL_LIMIT=3.0, DT=0.05, CRUISE_VX=0.3, WZ_MAX=0.2, arrival-latch threshold |vx|<0.02
- **State:** self._vx, self._wz
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (test rig ignores nav entirely)
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-005 · _reset_leg_state deliberate non-resets

- **Location (legacy):** 719-742 (called at construction + GO→REACH latch, 1392)
- **Protects against:** Leg-state leakage: a servo bias learned at cube #1's bearing miscorrects cube #2 (angular error sources rotate with gimbal aim/arm pose — rejected alternative at 389-391); conversely over-resetting would drop the live gripper measurement stream (_grip) or permanently silence servo prints.
- **Trigger:** Every new REACH leg.
- **Response:** Resets _reach_port, _servo_bias, _servo_seen/_accepted/_stamp/_cmd_prev/_ee_prev/_gimbal_prev/_accept_t, _servo_warned (once-per-LEG warnings re-arm each leg), _reach_tm/_reach_stream/_reach_sector; deliberately does NOT reset _servo_log_t (cross-leg 1 s print throttle) and _grip (a measurement stream, not leg state).
- **State:** kept: _servo_log_t, _grip; reset: 11 per-leg vars
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-006 · Learned-gate inference throttle + quantized cache + fresh-flag

- **Location (legacy):** 1780-1803 (def); constants block 124-132
- **Protects against:** Tick-budget blowout: unthrottled CPU-torch gate.score() at 17-37 ms/call, 2-3 pending cubes per 20 Hz tick = 35-75 ms, blew the 50 ms budget — the resulting gaze publish jitter excited the gimbal trim integrator into the violent periodic pre-reach camera shudder; also a single lucky stochastic sample repeatedly latching a debounce counter.
- **Trigger:** Every gate consultation (GO arrival, _dyn_reachable, _reachable_now) within GATE_EVAL_PERIOD_S of a cached verdict for the same 2 cm-quantized target.
- **Response:** Return cached (score, best_arm) with fresh=False; debounce counters (_gate_hits, in_hits/out_hits) step ONLY on fresh samples, so GATE_CONSECUTIVE now spans ~0.5 s wall-clock; cache cleared when >128 entries.
- **Constants:** GATE_EVAL_PERIOD_S=0.1, quantization=0.02 m, cache cap=128, GATE_CONSECUTIVE=5
- **State:** _gate_cache
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** t_gate_throttle (throttle=1 call per window, refresh after period, cached evals do not advance debounce counters)
- **Incident:** 07-17: pre-reach camera shudder, calm again once REACH started and gate calls stopped.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-007 · Main-loop tick pacing / robot-side watchdog liveness contract

- **Location (legacy):** 85-86, 2804-2805, 2844
- **Protects against:** Chronically slow ticks starve the robot-side silence failsafes the operator relies on staying ARMED-but-fed (nav 1 s, arm 0.5 s, gaze 2 s — docstring 52-55): arm silence makes real_env re-engage IK on its last (possibly stale-cube) target; publish jitter feeds the trim-loop shudder (see gate throttle).
- **Trigger:** Every loop iteration.
- **Response:** time.sleep(max(0.0, DT - elapsed)) paces the loop at OP_RATE_HZ=20 Hz, chosen explicitly > all robot-side timeout rates; publishing is continuous in every phase.
- **Constants:** OP_RATE_HZ=20.0, DT=0.05; robot-side timeouts nav 1 s / arm 0.5 s / gaze 2 s
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none (tests call op.tick() directly, no real-time loop)
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-008 · Telemetry-stream freshness gate (main I/O loop)

- **Location (legacy):** 2814-2822
- **Protects against:** Fail-open posture guards: tel_sub.data holds the LAST packet forever, so a frozen/restarted 9870 stream would keep feeding yesterday's encoders to the startup posture check, tilt gate, _arm_near_front, and EE gates — all passing on dead data (07-15 audit finding).
- **Trigger:** Telemetry packet older than 0.5 s wall-clock (hardcoded horizon at line 2822).
- **Response:** tick() receives telemetry={} — every downstream guard sees 'no telemetry' and fails CLOSED (gaze unseeded, startup check unpassed, arm packet withheld, robot holds/crawls on its own failsafes).
- **Constants:** staleness horizon=0.5 s (literal, unnamed)
- **State:** last_tel_id, last_tel_arrival
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (tests inject telemetry straight into tick(), bypassing main())
- **Incident:** 07-15 audit: stale-telemetry fail-open class.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-009 · Write-only _arm_engaged latch (vestigial invariant marker)

- **Location (legacy):** 504-511 (init+comment), 2663-2664 (only set, never read)
- **Protects against:** None enforced by the variable itself — it is WRITE-ONLY. The invariant its comment documents is real and hardware-derived: after any arm packet, going arm-silent lets real_env re-engage its IK on the last Cartesian target after stream silence, snapping the arm back to a stale cube target (observed on hardware 07-14). Enforcement now lives entirely in the keep-alive/arbitration block (2650-2662).
- **Trigger:** First tick that emits any arm_targets payload.
- **Response:** Sets self._arm_engaged=True; nothing consumes it. Refactor note: do not mistake it for an active guard; either delete it or preserve the comment's invariant with the keep-alive logic.
- **State:** _arm_engaged
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-14 hardware: post-staircase silence snapped an arm back to a stale cube target.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-010 · Once-per-cube refusal latches (_sector_warned)

- **Location (legacy):** 494-495, 1309-1319 (envelope, key f'{key}:env'), 1327-1333 (disabled sector, key=key)
- **Protects against:** Operational blindness: the envelope-refusal print is the operator-facing DETECTOR for the torso-tilt geometry-corruption class (cubes reading z=+0.2 at r_xy≈0.5, observed again 07-18 14:24) — because the latch is once-per-cube-per-mission and never cleared, a recurring or later-recurring corruption prints exactly once and then refuses silently.
- **Trigger:** SEARCH evaluating a pending cube that fails _target_safe or sits in a disabled sector.
- **Response:** Cube skipped (never reached — fail-closed) with a single loud print naming TARGET_MIN/MAX_RADIUS_M, TARGET_Z_MIN/MAX_M and RULE #0 (legs straight, torso upright); subsequent ticks refuse silently.
- **Constants:** TARGET_MIN_RADIUS_M=0.16, TARGET_MAX_RADIUS_M=0.45, TARGET_Z_MIN_M=-0.40, TARGET_Z_MAX_M=0.30; separate latch keys c.key vs f'{c.key}:env'
- **State:** _sector_warned
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none
- **Incident:** 07-18 14:24: OOD-fooled gate reached at torso-tilt-corrupted targets; refusal message added to 'refuse loudly'.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-011 · Lazily-created throttled diagnostic clocks (anti-silent-stall)

- **Location (legacy):** 1251-1255 (3s), 1288-1291 (3s), 1453-1466 (1s), 1483-1487+1516-1519 (1s, SHARED key), 1592-1594 (3s per-hold), 1749-1752 (1s), 1994-1999 (5s), 2518-2524 (5s mid-mission tilt watch)
- **Protects against:** Silent stalls indistinguishable from hangs on hardware — 'silent waiting is the failure class the 07-16 audit was commissioned against'; the 2518-2524 clock is also the ONLY mid-mission detector for a degrading hang/stance (warning-only, no refusal action after startup).
- **Trigger:** Any persistent wait state: zeroing hold, home-settle hold, staging progress, missing ee.pos_actual (two DISTINCT wait states share _missing_ee_diag_t — one can mask the other for 1 s), un-met grasp preconditions, unseeded secondary, carry hop timeout, torso tilt > TILT_WARN_DEG.
- **Response:** Throttled WHY-print per wait state via `now - getattr(self, '_x_diag_t', -1e9) > T` timestamps — created lazily, NOT declared in __init__ (refactor hazard: easy to lose or double-create; grasp_diag_t/carry_warn_t live inside hold dicts via setdefault).
- **Constants:** throttles 1/3/5 s; TILT_WARN_DEG=8.0
- **State:** _zero_wait_diag_t, _home_wait_diag_t, _reach_diag_t, _missing_ee_diag_t, _sec_diag_t, _tilt_diag_t, h['grasp_diag_t'], h['carry_warn_t']
- **Flags:** not fail-closed, timing-sensitive
- **Tests:** none (no test asserts any diagnostic print)
- **Incident:** 07-16 audit: a hold that never grabs must say WHY; 07-16/18 tilt recurrences motivated the continuous watch.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-012 · CLI flag coupling silently disabling features

- **Location (legacy):** 2772-2778
- **Protects against:** Running the grasp lifecycle without the dynamic-track settle/follow machinery it depends on (hand_grab fired against a frozen-target hold that never re-settles), or walking/reaching/parallelizing in gaze-only mode.
- **Trigger:** main() constructing AutoOperator from CLI args.
- **Response:** grasp = args.grasp AND not gaze_only AND args.dynamic_track (so --no-dynamic-track silently drops grasping — no warning printed); dual_parallel and side_home each ANDed with not gaze_only.
- **Constants:** grasp default True, dynamic_track default True, side_home default True, dual_parallel default True
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (tests construct AutoOperator directly with explicit kwargs; t_dual_parallel_off only checks the constructor default)
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-013 · Shared MuJoCo FK buffer single-thread assumption

- **Location (legacy):** 801-813 (_grip_base_fk); shared with gaze FK 608/625 and probe 588; ids 551-557
- **Protects against:** A parallelized or caching refactor races the single self.fk._model/_data pair between the gaze-aim FK, the camera-frame probe, and the gripper-base FK — FK evaluated at another caller's qpos yields a wrong servo bias b_meas or a wrong camera aim.
- **Trigger:** Every _grip_base_fk call (servo step) and every _aim_angles call, all on the one tick thread.
- **Response:** Documented contract: 'single tick thread; every user re-writes qpos before reading' — _grip_base_fk zeroes d.qpos[7:] then writes the full joint vector before mj_kinematics; nothing is cached across calls. Also fails closed to None on jpos missing/short vs fk.n_joints.
- **State:** self.fk._model, self.fk._data, _grip_body_id
- **Flags:** fail-closed, timing-insensitive
- **Tests:** none (tests run visual_servo=False)
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-014 · Empty-latch-window crash guard (occlusion at arrival tick)

- **Location (legacy):** 1363-1371
- **Protects against:** np.median over an empty stack would CRASH the operator mid-mission with the robot walking/arms live (07-15 audit) when the cube is occluded exactly at the arrival tick and no sighting falls inside the latch window.
- **Trigger:** GO arrival latch with zero sightings newer than CUBE_LATCH_WINDOW_S in active.hist.
- **Response:** Print '[reach] latch window empty ... re-acquiring via SEARCH' and rewind to Phase.SEARCH instead of latching — no target, no arm motion.
- **Constants:** CUBE_LATCH_WINDOW_S=0.7; hist deque maxlen=5
- **Flags:** fail-closed, timing-sensitive
- **Tests:** none (tests always feed a detection at the arrival tick)
- **Incident:** 07-15 adversarial audit: mid-mission crash class.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

### SAFE-GAPS-015 · no_claim as a dynamic (non-dataclass) attribute

- **Location (legacy):** 768-770 (read via getattr default False), 1850 (set True), 1884 (set False); CubeTrack def 433-446 lacks the field
- **Protects against:** The carried-cube camera-claim suppression (a cube riding in the gripper must not re-claim an eye — it starves the other cube's search, 07-16 audit) hinges on a dynamically-added attribute read through getattr(cube, 'no_claim', False): any typo'd or refactored set silently defaults back to False and the carried cube re-claims a camera.
- **Trigger:** _ingest claim assignment for a cube entering CARRIED (_enter_carried) or reverting on a confirmed drop (_grasp_dropped).
- **Response:** getattr default-False read gates the claim branch; refactor should promote no_claim to a declared CubeTrack field so attribute errors surface.
- **State:** cube.no_claim (dynamic), cube.camera_port
- **Flags:** not fail-closed, timing-insensitive
- **Tests:** none (t_grasp_carry_side/t_drop_detection execute the set/clear paths but assert nothing about camera claims)
- **Incident:** 07-16 audit: carried cube's kept claim starved the other cube's search when only one camera survived.
- **New location (S6):** unchanged in operator unless listed above (import workaround, _ramp, _reset_leg_state non-resets)

---

# Addendum 2026-07-18 · `--stage-sectors` (ADR-8, INC-10)

Joint staging RE-ACTIVATES under side-home behind `op.stage_sectors`
(ctor default False — flag-off is replay-proven tick-identical). Deltas to
the mechanism inventory above; everything not listed re-activates unchanged.

**Home-index redefinition** (SAFE-STAGING-009/010/016/022/023/024,
SAFE-HELD-006/007/009/013/014/016/020 and every other "front(0) is home"
site): `op._home_idx` is the single authority — 0 classic, 1 (SIDE) under
stage-sectors. Timeout flips, `or [hidx]` re-plants, release-only-at-home,
carry/retreat destinations, retarget fallbacks, the tick_reach idle-arm pin
and the abort messages all key on it. Verified: classic scenarios in the
replay gate exercise these lines with home_idx=0, zero divergence.

**Station-1 re-bake** (SAFE-STAGING-003/004/005): under stage-sectors ONLY,
`_track_posture(arm, 1)` returns SIDE_HOME_JOINTS — home and station 1 are
the same physical posture, so tracks arrive where the arm rests and the
near-home checks fire. Classic chains keep STAGE_JOINTS["side"]. New chain
audited offline (S-A 07-18): home↔front and home↔rear equal-rate curves,
both arms — trunk clearance 28.9 mm (the static shoulder-torso bound),
inter-arm ≥240 mm, limit margins ≥31°, EE z min ≈0.00; STATION_BEARING_DEG
unchanged (90.0/38.1/142.0). Hardware re-validation of the re-baked chain:
watch the first staircase run (same discipline as the 07-17 raised chain).

**Routing authority** (`_staged_target_idx`, new): side + low-front
(z ≤ FRONT_STAGE_Z_M = 0.0, judged on the STANDOFF point) = DIRECT;
high-front + all rear = STAGED. Consumers: latch_reach_target,
register_hold, tick_search enable gate (SECTOR_REACH_ENABLED authority
restored for staged legs), secondary candidate filter.

**One arbiter per hold** (new rule): staged holds (sector_idx != home_idx)
answer to the sector machinery — sector clamp restored at Guard-1, the
live_retrack clamp and the sector-exit release; via-home is EXCLUDED for
them. Direct holds keep via-home and never start tracks. Without the split
the two retract mechanisms fight (station↔side-home ping-pong).

**Deferral hardening** (new, closes a recon gap): the staged-hold retract
track start is evaluated EVERY tick (not only on the state edge) because a
staged hold CAN now coexist with a dual-parallel secondary; while the track
has not started, build_arm_packet holds the arm IN PLACE at its measured EE
— the Cartesian home glide is forbidden for staged holds (cross-basin).

**Secondary containment** (SAFE-DUAL-*): secondaries remain pure Cartesian
and may only take DIRECT-classified legs, judged on the standoff point (the
same value register_hold will record). A staged primary blocks secondaries
until its staircase settles; its Cartesian hop then coexists safely.

**Startup gate extension** (SAFE-STARTUP-*): side_home profiles additionally
accept SIDE_HOME_JOINTS as a known-safe startup posture (restart-after-park
gap); classic profiles still refuse it (covered by test). DECLARED
INTENTIONAL FLAG-OFF DELTA (review finding #1): this fires on side_home
alone, not stage_sectors — a plain side-home restart parked at the side no
longer demands a power-cycle. It is the ONLY flag-off behavior change in
this feature; the post-gate flow is the hardware-validated side-home path.

**New behavioral coverage** (~24 asserts): staged rear staircase E2E
(track 1→2 → station posture → Cartesian → grab → arc lift → reverse
staircase → side-home park), front conditional routing, timeout-flips-to-
home, secondary gating (none-during-track / direct-after-settle / never-
rear), startup-at-side-home accept + classic refuse, flag-off preservation.
