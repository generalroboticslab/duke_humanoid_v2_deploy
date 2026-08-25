# Journey walk, aim and arrival — value ledger

Migrated verbatim from `humanoid_auto_operator.py` during the public-release
cleanup. These are not commentary: each entry is a value the robot actually
ran, the hardware run that changed it, and the measurement that justified the
change. Dates are 2026. The live constants keep a condensed rationale inline
and point here.

The walk law itself is `journey_align_error` (one law, two users: the walk's
steering and any alignment logic). In-place rotation is FORBIDDEN on this
robot (upstream ruling, 2026-08-08) — curved walking only.


## `JOURNEY_AIM_OFFSET_M`

Value at the time of migration: `0.05`

```
JOURNEY_AIM_OFFSET_M = 0.05  # 08-13 (user): 0 -> 0.05 — a small lateral
                           # parking offset returns after the 0-offset day:
                           # dead-on stops landed r_xy 0.365-0.40 against a
                           # HIGH cube (+0.086) and fed the elbow-fold /
                           # INFEASIBLE failures; 5 cm beside the shoulder
                           # buys arm room without recreating the 08-09
                           # table-strike geometry (that lesson was at 0.10+
                           # with the WALK doing collision avoidance).
                           # PRIOR 08-12 late night (user): 0.30 -> 0 — the aim
                           # point IS the cube centre; forward legs walk
                           # dead-on. ⚠️ This knowingly overrides the 08-09
                           # table-strike lesson below (0.10 crushed into the
                           # table face; the lateral offset WAS the collision
                           # avoidance). With aim = centre the body stops
                           # ~0.5 m from the CUBE and the table's front edge
                           # sits 30-40 cm nearer: table clearance now comes
                           # ENTIRELY from placement — keep the cube near the
                           # table's NEAR edge, and expect the tip guard /
                           # C_clearance gates to be the last line, not the
                           # walk. PRIOR 08-09 night: 0.25 -> 0.30 minutes
                           # later — back to the exact value of the two
                           # grasped legs earlier tonight (landed y 0.26 /
                           # crossed fine), maximum distance from the table
                           # face. 0.10 -> 0.25 was the same stroke: after the
                           # 0.10 run CRUSHED INTO THE TABLE: the walk law
                           # has no obstacle concept — the arrival cutoff
                           # measures distance to the CUBE, and the table's
                           # front edge sits 30-40 cm nearer. At 0.25+ the
                           # body lands BESIDE the table corner (every
                           # validated grasp run: offsets 0.22-0.47); at
                           # 0.10 it lands square on the table face and the
                           # coast momentum delivers the strike. The lateral
                           # offset is load-bearing collision avoidance, not
                           # just grasp-geometry comfort. Straight walking
                           # comes from PLACEMENT (cube 0.25-0.30 m off the
                           # walk axis on the serving side), not this dial.
                           # PRIOR 0.10 note (one run, table strike): walk
                           # as straight as possible, with floor 25 -> 15.
                           # PRIOR 0.30 note: 0.20 -> 0.30 — split the
                           # difference between 0.20 (lands ~0.14 after
                           # the ~0.06 settle loss, right AT the comfort
                           # line) and the 0.38 era (4-for-4 grasps but
                           # the far park hid the rear cube behind the
                           # carrying elbow). 0.30 - 0.06 lands ~0.24:
                           # clear of the 0.15 torso line with margin,
                           # short of the elbow-shadow zone.
                           # PRIOR 0.20 note: 0 -> 0.20 after the plain-
                           # pursuit experiment: offset 0 landed the cube
                           # at y=+0.069 (midline) and reproduced the wall a
                           # fourth time — 29/30 F_windup refusals, and the
                           # one marginal route that slipped through seeded
                           # the MPC in a contortion that threw T09 (0.78
                           # rad/tick at right_wrist_1). Score to date:
                           # midline lands 0-for-4, lateral lands (0.38 era,
                           # y 0.22-0.47) 4-for-4. 0.20 is the user's middle
                           # ground: on short legs the settle transient eats
                           # ~0.05-0.06, so expect landings near y ~0.14 —
                           # right at the 0.15 comfort line; placement off
                           # the walk axis buys the margin the steering
                           # cannot.
                           # PRIOR: steer at an aim point offset
                           # INTO the serving arm's half-space, so the pursuit
                           # equilibrium parks the cube at y ~= +/-this instead
                           # of ON the midline. Must stay >> MIDLINE_MARGIN_M
                           # + detection noise; grasp geometry also prefers the
                           # cube in front of the serving shoulder.
                           # 08-08: 0.12 -> 0.18 -> 0.22, both MEASURED.
                           # Run 1 (0.12): landed y=+0.119, every left-arm
                           # route swept wrist_2_L within 1-22 mm of the torso
                           # (C_clearance floor 25 mm, 30/30 refused, SKIP).
                           # The 08-05 windup finding already put the
                           # front-cube comfort line at |y| >= 0.15.
                           # Run 2 (0.18): the cube RODE THE AIM LINE the
                           # whole approach (wz +0.00) and still landed
                           # y=+0.124 — the STOP transient (coast + settle
                           # wobble, a few deg of yaw) costs ~0.05-0.06 of
                           # lateral offset AFTER steering ends, and no gain
                           # or offset can steer a transient that happens
                           # after the last command. So the aim must budget
                           # for it: 0.22 - 0.06 lands ~0.16, still past the
                           # 0.15 line. Placement helps too: put the cube
                           # ~0.25 m off the walk axis (serving-arm side) so
                           # even a short leg that never converges starts and
                           # ends outside the wall.
                           # 08-09: 0.38 -> 0.45 (user-directed, after watching
                           # the run): on the reverse leg the CARRYING arm's
                           # elbow sat between the rear camera and the cube —
                           # blind spells, ghost fixes (drift 375 mm), one
                           # EMPTY close on a stale anchor, then T07. A wider
                           # park moves the cube out from behind the loaded
                           # elbow's silhouette. Geometry is proven: leg 1
                           # grasped twice tonight at essentially (x 0.22,
                           # y 0.44) — the same shape this value now aims for.
                           # 08-08 late: 0.22 -> 0.38 (user-directed). Parks
                           # the cube far to the serving side; with arrival
                           # at 0.60 and ~0.17 coast the cube lands around
                           # (x~0.2-0.3, y~0.33-0.38) — lateral-heavy but
                           # inside the envelope (a standing run grasped at
                           # |y|=0.414, r=0.549). Watch the first runs for
                           # the opposite failure: a cube parked TOO lateral
                           # shortens x and can graze the shoulder envelope
                           # edge; if INFEASIBLE-far appears, come back down.
```

## `JOURNEY_AIM_OFFSET_BACK_M`

Value at the time of migration: `0.0`

```
JOURNEY_AIM_OFFSET_BACK_M = 0.0  # 08-10 (user): 0.35 -> 0 — reverse legs walk
                           # STRAIGHT at the cube; the only lateral term left
                           # is the 0.08 drift comp, so the aim equilibria are
                           # -0.08/-0.08 (both arms aim slightly right to eat
                           # the measured backward-gait left drift). Landings
                           # ride on the drift alone: expect |y| ~0.00-0.08.
                           # RISK, stated: the 0.10 and 0.13 runs both died
                           # near the midline (INFEASIBLE wall / empty blind
                           # pinch) — a dead-midline cube is at the envelope's
                           # hardest corner. The settled-arm re-decision and
                           # CROSS_MIDLINE_REACH_M own the arm choice there.
                           # PRIOR 0.35 note: parks the cube well inside the
                           # proven lateral band; equilibria +0.27/-0.43.
                           # PRIOR 0.13 note: video framing dial; landings
                           # spanned 0.05-0.28 — the no-drift left-arm case
                           # (+0.05) sat dead midline.
                           # PRIOR 0.20 note: for-the-camera park at the low
                           # edge of the proven lateral band (6-for-6 at
                           # |y| 0.22-0.47).
                           # PRIOR 0.35 note: the reverse grasp is FOR THE
                           # CAMERA; WATCH the carried-elbow shadow if the
                           # reverse leg ever runs FIRST (08-09 0.38 lesson).
                           # PRIOR 0.05 note: 0.10 -> 0.05 — reverse walks
                           # essentially straight at the cube. RISK, stated
                           # and accepted: the settled fix lands ~|y| 0.02-
                           # 0.05, dead midline — the 0.10 run already died
                           # there once (0.366/y=-0.11, F_windup 0.70 + 25x
                           # INFEASIBLE); the same evening a 0.371 landing
                           # at y=+0.21 planned FIRST TRY. Distance armor is
                           # the 0.62 reverse arrival (lands 0.37-0.44);
                           # if reverse legs exhaust rounds on windup, the
                           # offset is the suspect — raise it back first.
                           # Table rule still binds: 0.05 is safe only
                           # because the rear cube sits on a low stand, not
                           # a table face. Forward keeps 0.30.
```

## `JOURNEY_REVERSE_DRIFT_COMP_M`

Value at the time of migration: `0.0`

```
JOURNEY_REVERSE_DRIFT_COMP_M = 0.0  # 08-13 night (user): 0.05 -> 0 — the
                           # right-bias came back for half a day and pushed
                           # a left-side cube (y +0.37 at survey) to the
                           # MIDLINE (settled y -0.001) on a leg where the
                           # historical left drift never appeared; the
                           # left arm then had to cross the body for a low
                           # cube. Both 08-13 reverse legs showed NO drift,
                           # so the compensation only steers aim error in.
                           # PRIOR same day morning: 0 -> 0.05. PRIOR
                           # 08-12 late night (user): 0.08 -> 0 —
                           # the right-bias is CANCELLED; reverse legs aim
                           # dead at the cube line with no drift compensation.
                           # If the leftward settle drift documented below
                           # returns (misses of +0.15..0.21 to the LEFT on
                           # full-length legs), this constant is where it
                           # gets paid again.
                           # PRIOR 08-10 later (user): 0.15 -> 0.08 after
                           # the first rightward-steering reverse leg showed
                           # NO drift at all (settled y=-0.109 EXACTLY on the
                           # biased aim -0.10) and the full 0.15 bias itself
                           # pushed a well-placed cube across the midline —
                           # 30 INFEASIBLEs. The drift now looks steering-
                           # direction-dependent (4 leftward-correcting legs
                           # drifted +0.15..0.21; 1 rightward leg drifted 0);
                           # 0.08 splits the difference: worst case either
                           # way is ~±0.08 of the parking spot instead of
                           # -0.15 or +0.20.
                           # PRIOR 0.15 note: the backward gait carries a
                           # systematic LEFTWARD settle drift the steering law
                           # cannot see — every full-length reverse leg on
                           # record missed its aim by +0.15..+0.21 m to the
                           # LEFT, both arms, three different offsets (settled
                           # -0.154 vs -0.30, -0.093 vs -0.30, +0.261 vs
                           # +0.10, +0.099 vs -0.05), while forward legs land
                           # within 0.05. The law itself is direction-
                           # symmetric (differential-pinned); this constant
                           # shifts every REVERSE aim equilibrium this far to
                           # the RIGHT so the measured drift carries the cube
                           # onto the intended parking spot. Same pattern as
                           # JOURNEY_ARRIVE_M_BACK compensating the worse
                           # backward braking. Re-tune from the next run's
                           # settled-fix line; forward legs are untouched.
```

## `JOURNEY_ARRIVE_M_BACK`

Value at the time of migration: `0.50`

```
JOURNEY_ARRIVE_M_BACK = 0.50  # 08-13 (user, 7th pass): 0.53 -> 0.50 — stop
                           # closer on reverse legs so the rear-reach
                           # envelope bends lower: the low-table cube read
                           # z=-0.04..-0.05 at r_xy 0.497 and cuRobo went
                           # 30/30 INFEASIBLE; a smaller radius is the one
                           # stance-side lever against a low rear cube.
                           # PRIOR 08-12 (user, 6th pass): 0.50 -> 0.53.
                           # PRIOR 5th pass: 0.55 -> 0.50 — the rear
                           # leg's cube landed at r_xy 0.49 and sat 4 mm below
                           # the solver's envelope floor (30 INFEASIBLEs); a
                           # closer stop shrinks the radius and buys downward
                           # reach. FIRST TIME BELOW the fwd cutoff — reverse
                           # coast is LONGER (0.18-0.25 band), watch for
                           # wind-up-basin landings (<0.30).
                           # PRIOR 4th pass: 0.60 -> 0.55; 3rd: 0.57 -> 0.60.
                           # PRIOR same-day: 0.55 -> 0.57 — back off
                           # a touch from the morning's 0.55 cut.
                           # PRIOR 08-12 morning: 0.62 -> 0.55 — walk in closer
                           # on reverse legs too. PRIOR 08-10: 0.60 -> 0.62 after the 0.60 cut
                           # + 0.10 back-offset run landed 0.366/y=-0.11 —
                           # the wind-up basin (F_windup 0.70 rad + 25x
                           # INFEASIBLE, leg refused). The deep-coast end
                           # (0.25) now lands 0.37+; typical lands 0.40-0.48.
                           # PRIOR 0.60 note: 0.68 -> 0.60 after the a/b night's
                           # reverse leg landed 0.64 (coast only ~0.04 that
                           # run — the 0.18-0.25 band is not a floor) and the
                           # grasp fought the envelope line. WATCH: a
                           # 0.32-0.35 landing brushes the wind-up basin.
                           # PRIOR 0.68 note, 08-08 night: reverse legs coast FURTHER than
                           # forward ones, and both hardware reverse landings
                           # prove it — stop at 0.50 landed 0.321 (coast
                           # 0.179), stop at 0.60 landed 0.346 (coast 0.254),
                           # while the four forward landings coasted 0.09-0.23.
                           # Backward gait braking is simply worse. 0.68 minus
                           # the 0.18-0.25 measured band lands the cube at
                           # 0.43-0.50 — inside the validated grasp band
                           # (tonight's forward leg grasped first-attempt at
                           # 0.511). Forward keeps 0.60 below.
```

## `JOURNEY_ARRIVE_M`

Value at the time of migration: `0.63`

```
JOURNEY_ARRIVE_M = 0.63    # 08-13 (user, 7th pass): 0.58 -> 0.63 — the day's
                           # ledger split cleanly on landing radius: both 2/2
                           # missions landed r_xy 0.40-0.47, then five
                           # straight failures landed 0.36-0.41 (elbow-fold
                           # standoffs / INFEASIBLE against a +85 mm cube).
                           # Stop coast + settle refinement eats ~0.20 m off
                           # the live-distance cutoff, so 0.63 aims the
                           # landing at ~0.45, the middle of the proven band.
                           # PRIOR 08-12 (user, 6th pass): 0.52 -> 0.58 — after the
                           # aim-at-centre change landed a cube at r_xy 0.313
                           # (midline, high) and 30/30 solves came back
                           # F_windup: a farther stop keeps the landing out
                           # of the wind-up basin with the centre aim.
                           # PRIOR 5th pass: 0.50 -> 0.52; 4th: 0.55 -> 0.50.
                           # PRIOR same-day: 0.50 -> 0.53 — back off a
                           # touch from the morning's 0.50 cut.
                           # PRIOR 08-12 morning: 0.55 -> 0.50 — stop closer to the
                           # cube. PRIOR 08-10: 0.60 -> 0.55 in the same stroke as
                           # the reverse cut above — walk closer, stop the
                           # settled fix from riding the envelope line. With
                           # the measured 0.09-0.23 forward coast this lands
                           # ~0.32-0.46. WATCH the low end: 0.321 was the
                           # measured wind-up-basin landing (refused 3 rounds)
                           # — if F_windup refusals return, come back up.
                           # PRIOR 0.60 note, 08-08: 0.50 -> 0.60, forced by the SAME physics
                           # that set 0.50 — the walk floor rose again (0.3 ->
                           # 0.4) and the coast rose with the square of it.
                           # MEASURED on the first 0.4 m/s journey leg
                           # (reverse): command stopped at 0.50, the cube then
                           # read r_xy 0.321 — a 0.18 m coast, double the 0.098
                           # median at 0.3 m/s and right on the v^2 scaling
                           # (0.098 * (0.4/0.3)^2 = 0.174). 0.321 is deep in
                           # the wind-up basin: cuRobo refused all 3 rounds
                           # (F_windup 0.68-0.75 rad / INFEASIBLE) and the leg
                           # was SKIPPED. At 0.60 with a ~0.17-0.18 m coast
                           # both directions land the cube at ~0.42, the middle
                           # of the validated 0.40-0.48 grasp band.
                           # 08-07 (user): 0.34 -> 0.50, and it is the SAME
                           # decision as the NEAR_VX raise above, not a second
                           # one. A 0.2 m/s floor cannot be asked to stop within
                           # 0.34 m: the old value budgeted 5-8 cm of braking at
                           # 0.1 m/s, and the robot now arrives at twice that
                           # speed. Stopping the COMMAND at 0.50 m spends the
                           # extra braking distance instead of the safety
                           # margin.
                           # MEASURED 2026-08-07, forward, four runs at 0.30 m/s
                           # (landmark odometry off a stationary tag cube, since
                           # base_lin_vel is hard zeros): coast 0.071 / 0.124 /
                           # 0.142 / 0.056 m, median 0.098. So the cube lands at
                           # 0.36-0.44 m — mid-envelope, clear of the 0.16
                           # keep-out, inside the ~0.51 clean edge, and far from
                           # the 0.26 loaded-carry collision band. 0.50 STANDS
                           # for forward legs; it was an estimate and it
                           # survived measurement.
                           # Same runs: travel tracked 66-82% of commanded
                           # (median 72%), and the base went quiet 0.91-1.36 s
                           # after the stop — comfortably inside the 2.5 s blind
                           # stillness window.
                           # REVERSE, same day, three runs: coast 0.125 / 0.091
                           # / 0.099, median 0.099 — INDISTINGUISHABLE from
                           # forward, and a tighter spread. So this constant
                           # stays direction-blind, which was worth checking
                           # rather than assuming: two hand-paced 4 s runs had
                           # suggested reverse travelled 108% of commanded
                           # against forward's 83%, and the prediction drawn
                           # from that (reverse carries more momentum into the
                           # stop, so it needs its own larger arrival) is WRONG.
                           # Landmark odometry at 2 s puts both directions at
                           # 66-101% with overlapping ranges. Pacing error, not
                           # a direction effect. What IS asymmetric is start-up,
                           # not stopping — see JOURNEY_NEAR_VX above.
                           # 07-21 history (user: "stop closer"): the gate often
                           # says reachable at 0.40+, so the old value creeped at
                           # NEAR_VX to 0.34 for a ~0.28-0.33 m standoff. Its
                           # floor note still binds and is now far away: never
                           # let the FINAL standoff reach 0.26, where the
                           # loaded-carry collision band starts (07-20
                           # forensics).
```

## `JOURNEY_CUBE_Z_MIN`

Value at the time of migration: `-0.06`

```
JOURNEY_CUBE_Z_MIN = -0.06  # m, base frame. 08-13 (user): -0.01 -> -0.06 —
                           # the -0.01 fail-fast bar over-generalized the
                           # 08-08/09 evidence: BOTH low-pole failures
                           # (z=-0.021 20/20, z=-0.014@r0.49 30/30) were
                           # LARGE-RADIUS rear reaches, and the envelope
                           # floor is radius-dependent (today: z=-0.002 at
                           # r=0.42 grasped clean). Below -0.06 the user
                           # agrees nothing reaches; in between, the SOLVER
                           # judges — worst case is the ~90 s of rounds this
                           # gate existed to save, not a wrong grasp.
```
