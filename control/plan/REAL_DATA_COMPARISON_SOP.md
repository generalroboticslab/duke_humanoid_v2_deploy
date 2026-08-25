# Real Robot Deployment Recording — Comparison SOP

Standard operating procedure for the **compare → diagnose → improve** cycle using
real-robot `.pkl` recordings from `humanoid_real_env.py`.

Primary goal: surface behavioral differences between policy variants on hardware.
Unlike simulation logs, real recordings contain **no reward, no termination flags,
no curriculum state, and no training metrics**. All diagnostics are derived from
sensor telemetry alone.

---

## 1. What Is Recorded

Each `.pkl` is a pickled dict with per-step arrays and static config:

### Per-step arrays (shape = `(N_steps, N_joints)` or `(N_steps, N)`)

| Key | Shape | Description |
|---|---|---|
| `time` | `(N,)` | Wall-clock seconds since recording start |
| `q_pos` | `(N, 27)` | Encoder joint positions (rad) |
| `q_vel` | `(N, 27)` | Encoder joint velocities (rad/s) |
| `q_tau` | `(N, 27)` | Measured joint torques (Nm) |
| `tau_grav_raw` | `(N, 27)` | Raw gravity-comp torque before scaling |
| `tau_grav_applied` | `(N, 27)` | Gravity-comp torque actually applied |
| `cmd_pos` | `(N, 27)` | Commanded joint positions (rad) |
| `actions` | `(N, 27)` | Policy output actions (normalized) |
| `base_ang_vel` | `(N, 3)` | IMU angular velocity (rad/s, body frame) |
| `projected_gravity` | `(N, 3)` | Gravity vector in body frame |
| `imu_quat_wxyz` | `(N, 4)` | IMU quaternion (w, x, y, z) |
| `imu_timestamp` | `(N,)` | IMU hardware timestamp (int) |
| `command` | `(N, 3)` | Velocity command `[vx, vy, wz]` |
| `keyframe` | `(N,)` | bool — manually marked keyframes |

### Static config (same for all steps)

| Key | Description |
|---|---|
| `kp`, `kd` | Motor P/D gains |
| `action_scale` | Per-joint action scale |
| `default_joint_pos` | Default pose (rad) |
| `joint_names` | List of 27 joint names |
| `control_freq` | Recording rate (Hz; typically 50) |
| `grav_comp_scale` | Gravity comp scale factor |

### What is NOT available (vs. simulation log)

- Episode reward, episode length, termination flags
- Curriculum weights and stage info
- Training loss, entropy, value loss
- Base linear velocity (must be estimated or computed from IMU+FK)
- Gait phase clock (not published in telemetry)
- End-effector poses (not in these files; see `humanoid_real_env.py:ee_poses` on port 9875)

---

## 2. Naming Convention

Recording files are auto-generated:

```
{ExperimentClassName}_{YYYYMMDD}_{HHMMSS}[_{optional_suffix}].pkl
```

Example:
```
HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv6_20260404_175237.pkl
```

Recording directories: `control/recordings/`

---

## 3. Quick-Start: Compare Three Variants

```bash
# Inspect structure (all files must have identical keys)
python3 -c "
import pickle, sys
files = [
    'control/recordings/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv6_20260404_175237.pkl',
    'control/recordings/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv7_20260404_183449.pkl',
    'control/recordings/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv9_20260404_184352.pkl',
]
for f in files:
    with open(f,'rb') as fh:
        d=pickle.load(fh)
    n=len(d['time'])
    print(f'{f.split(\"/\")[-1]}: {n} steps, {d[\"time\"][-1]:.1f}s, freq={d[\"control_freq\"]}Hz')
"
```

---

## 4. Metrics to Compute

All metrics are derived post-hoc from sensor arrays. Use the reference script
`mj_envs/parse_real_recording.py` or inline analysis.

### 4.1 Core per-segment statistics

Segment recordings by **keyframe** markers (manual cuts) or by command changes.
Default: whole recording is one segment.

| Metric | Formula | What it reveals |
|---|---|---|
| `mean_q_vel` | `mean(|q_vel|)` per joint then mean | overall activity level |
| `max_q_tau` | `max(|q_tau|)` per joint then max | peak torque demand |
| `rms_action_rate` | `sqrt(mean(diff(actions)^2))` | smoothness / action chatter |
| `tracking_error_lin` | `rmse(q_pos - cmd_pos)` for leg joints | position tracking fidelity |
| `grav_comp_delta` | `mean(|tau_grav_applied - tau_grav_raw|)` | gravity comp modulation |
| `base_ang_vel_max` | `max(|base_ang_vel|)` | angular instability |
| `upright_estimate` | `mean(projected_gravity[:, 2])` | body tilt (Z component should be ~−1) |
| `commanded_speed` | `mean(|command[:, :2]|)` | commanded velocity intent |
| `frac_stationary` | `mean(sqrt(cmd_x²+cmd_y²) < 0.05)` | fraction of time stopped |
| `frac_high_cmd` | `mean(sqrt(cmd_x²+cmd_y²) > 0.3)` | fraction at high speed |

### 4.2 Per-joint torque breakdown

```python
# Torque per joint, averaged over segment
mean_tau = np.mean(np.abs(d['q_tau']), axis=0)   # (27,)
max_tau  = np.max(np.abs(d['q_tau']), axis=0)    # (27,)

# Gravity comp contribution (arm joints only; legs handled by policy)
arm_mask = [i for i, n in enumerate(d['joint_names'])
            if not re.match(r'^(waist|.*hip|.*knee|.*ankle)', n)]
mean_arm_tau    = np.mean(np.abs(d['q_tau'][:, arm_mask]), axis=0)
mean_grav_arm   = np.mean(np.abs(d['tau_grav_applied'][:, arm_mask]), axis=0)
```

### 4.2 Per-joint torque heatmap

```python
mean_tau_per_joint = np.mean(np.abs(d['q_tau']), axis=0)
plt.figure(figsize=(14, 4))
plt.bar(d['joint_names'], mean_tau_per_joint)
plt.ylabel('Mean |torque| (Nm)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### 4.3 High-value mining metrics (from existing data)

These metrics are not in the core per-segment set but have proven discriminative
power across the EEDRv6–v12 deployment series:

#### `cmd_ratio` — command tracking fidelity
```python
act_speed = np.sqrt(np.mean(d['q_vel'][:, leg_idx]**2, axis=1))   # per-step scalar
cmd_speed = np.sqrt(d['command'][:, 0]**2 + d['command'][:, 1]**2)
ratio = act_speed / np.where(cmd_speed > 0.01, cmd_speed, np.nan)
ratio = np.nanmean(ratio)   # nan avoids division by zero
```
| cmd_ratio | Interpretation |
|---|---|
| > 3.0 | Running fast relative to command — unstable stepping |
| 2.0–2.5 | Healthy walking (baseline = 2.30) |
| < 1.0 | Not following command — broken or standing |

#### `LR_asym` — L/R leg torque asymmetry
```python
left_idx  = [i for i,n in enumerate(d['joint_names']) if 'left'  in n and LEG_PATTERN.match(n)]
right_idx = [i for i,n in enumerate(d['joint_names']) if 'right' in n and LEG_PATTERN.match(n)]
lt = np.mean(np.abs(d['q_tau'][:, left_idx]))
rt = np.mean(np.abs(d['q_tau'][:, right_idx]))
asym = abs(lt - rt) / ((lt + rt) / 2) * 100   # percent
```
Operator note: "v6 leans towards stance leg side" → confirmed by LR_asym=13.1%.
v7 (better stepping) → 4.7%. v8 (most symmetric) → 1.0%.

#### `ankle_hip_ratio` — balance strategy
```python
ankle_idx = [i for i,n in enumerate(d['joint_names']) if 'ankle' in n]
hip_idx   = [i for i,n in enumerate(d['joint_names']) if 'hip'   in n]
ratio = np.mean(np.abs(d['q_tau'][:, ankle_idx])) / np.mean(np.abs(d['q_tau'][:, hip_idx]))
```
- v12 (standing) = 1.047 (ankle-dominant, no stepping)
- Walking variants: 0.61–0.80 (hip-dominant; lower = more hip compensation for instability)

#### `tilt_max` and `tilt_p95` — lateral instability
```python
tilt_xy = np.sqrt(d['projected_gravity'][:, 0]**2 + d['projected_gravity'][:, 1]**2)
tilt_max = np.max(tilt_xy)
tilt_p95 = np.percentile(tilt_xy, 95)
```
- v6 tilt_max = 0.459 (operator: "unstably lean/tilt towards stance leg side")
- baseline tilt_max = 0.195; v7 = 0.128 (better)
- Mean upright (gz) is nearly identical across variants — `tilt_max` is the discriminative signal, not the mean

#### `arm_energy_ratio` — arm coordination
```python
arm_vel  = np.mean(np.abs(d['q_vel'][:, arm_idx]))
leg_vel  = np.mean(np.abs(d['q_vel'][:, leg_idx]))
arm_tau  = np.mean(np.abs(d['q_tau'][:, arm_idx]))
leg_tau  = np.mean(np.abs(d['q_tau'][:, leg_idx]))
ratio = (arm_tau * arm_vel) / (leg_tau * leg_vel)
```
- baseline = 5.6% (highest; best push recovery per operator)
- EEDRv6–v11 = 1.9–3.1% (arm override reduces arm coordination)
- Higher arm energy correlates with better push recovery

#### `saturation` — action saturation fraction
```python
scale = d['action_scale']   # per-joint action scale
norm_actions = d['actions'] / np.where(scale > 0.01, scale, 1)
sat = np.mean(np.abs(norm_actions) > 0.9)   # fraction of steps near clip
```
- baseline = 17.3% (policy drives hard)
- v12 = 1.4% (not trying — broken)

#### `tau_spike_ratio` — angular instability torque correlation
```python
ang_mag = np.sqrt(d['base_ang_vel'][:, 0]**2 + d['base_ang_vel'][:, 1]**2 + d['base_ang_vel'][:, 2]**2)
thr = np.mean(ang_mag) + 2 * np.std(ang_mag)   # 2-sigma threshold
spike_mask = ang_mag > thr
rms_tau = np.sqrt(np.mean(d['q_tau']**2, axis=1))
ratio = rms_tau[spike_mask].mean() / rms_tau[~spike_mask].mean()
```
Captures how much torque spikes during angular velocity events.
Note: does NOT distinguish cause (push) from effect (correction).

#### Time-trend analysis — behavioral drift
```python
win = 200   # rolling window at 50Hz (4-second windows)
rolling = lambda x: np.array([np.mean(x[max(0,i-win):i+1]) for i in range(len(x))])

ratio_roll = rolling(act_speed / np.where(cmd_speed > 0.01, cmd_speed, np.nan))
ang_roll   = rolling(ang_mag)
tilt_roll  = rolling(tilt_xy)

# Fit linear trend (slope × 1000 steps)
x = np.arange(len(ratio_roll)); mask = np.isfinite(ratio_roll)
slope = np.polyfit(x[mask], ratio_roll[mask], 1)[0] * 1000
```
- v12: ang_vel trend −0.18 (robot slowing — battery drain or policy degradation)
- v7: ang_vel trend +0.11 (robot accelerating — instability buildup)
- baseline: stable (~0)

---

### 4.4 Segment comparison table

Group steps by `keyframe=True` markers into segments. Compute per-segment metrics,
then average across segments for a per-run summary.

---

## 5. Analysis Script Template

```python
"""
mj_envs/parse_real_recording.py
Usage:
    python mj_envs/parse_real_recording.py \\
        recordings/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv6_20260404_175237.pkl \\
        recordings/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv7_20260404_183449.pkl \\
        recordings/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv9_20260404_184352.pkl
"""
import pickle, argparse, re
import numpy as np

LEG_PATTERN = re.compile(r'^(waist|.*hip|.*knee|.*ankle)')
ARM_PATTERN = re.compile(r'^(?!waist|.*hip|.*knee|.*ankle).*')

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def segment_stats(d, start=0, end=None):
    end = end or len(d['time'])
    t   = d['time'][start:end]
    q_pos   = d['q_pos'][start:end]
    q_vel   = d['q_vel'][start:end]
    q_tau   = d['q_tau'][start:end]
    actions = d['actions'][start:end]
    grav_a  = d['tau_grav_applied'][start:end]
    grav_r  = d['tau_grav_raw'][start:end]
    b_av    = d['base_ang_vel'][start:end]
    grav    = d['projected_gravity'][start:end]
    cmd     = d['command'][start:end]
    cmd_pos = d['cmd_pos'][start:end]

    joint_names = d['joint_names']
    n_joints = len(joint_names)

    leg_idx = [i for i, n in enumerate(joint_names) if LEG_PATTERN.match(n)]
    arm_idx = [i for i, n in enumerate(joint_names) if ARM_PATTERN.match(n)]

    # Per-joint means
    mean_vel   = np.mean(np.abs(q_vel), axis=0)
    mean_tau   = np.mean(np.abs(q_tau), axis=0)
    max_tau    = np.max(np.abs(q_tau), axis=0)
    rms_action = np.sqrt(np.mean(np.diff(actions, axis=0)**2, axis=0))

    # Tracking error (legs only)
    track_err = np.sqrt(np.mean((q_pos[:, leg_idx] - cmd_pos[:, leg_idx])**2, axis=0))

    # Overall
    duration = t[-1] - t[0]
    cmd_lin = np.sqrt(cmd[:, 0]**2 + cmd[:, 1]**2)
    mean_cmd_lin = np.mean(cmd_lin)
    frac_stationary = np.mean(cmd_lin < 0.05)
    frac_high_cmd   = np.mean(cmd_lin > 0.3)
    mean_ang_vel = np.mean(np.sqrt(b_av[:, 0]**2 + b_av[:, 1]**2 + b_av[:, 2]**2))
    ang_vel_mag  = np.sqrt(b_av[:, 0]**2 + b_av[:, 1]**2 + b_av[:, 2]**2)
    threshold = np.mean(ang_vel_mag) + 3 * np.std(ang_vel_mag)
    n_ang_spikes = int(np.sum(ang_vel_mag > threshold))
    upright = np.mean(grav[:, 2])   # should be ≈ -1 when upright

    return dict(
        duration=duration,
        n_steps=end - start,
        mean_vel_legs=mean_vel[leg_idx].mean(),
        mean_vel_arms=mean_vel[arm_idx].mean(),
        mean_tau_legs=mean_tau[leg_idx].mean(),
        mean_tau_arms=mean_tau[arm_idx].mean(),
        max_tau=max_tau.max(),
        rms_action_legs=rms_action[leg_idx].mean(),
        rms_action_arms=rms_action[arm_idx].mean(),
        track_err_legs=track_err.mean(),
        mean_ang_vel=mean_ang_vel,
        n_ang_vel_spikes=n_ang_spikes,
        upright=upright,
        mean_cmd_lin=mean_cmd_lin,
        frac_stationary=frac_stationary,
        frac_high_cmd=frac_high_cmd,
    )

def _print_stats(s, frac_stationary_ref=None):
    print(f"  Duration:        {s.get('duration', 0):.1f}s")
    print(f"  Leg vel (mean):   {s.get('mean_vel_legs', 0):.4f} rad/s")
    print(f"  Arm vel (mean):   {s.get('mean_vel_arms', 0):.4f} rad/s")
    print(f"  Leg tau (mean):   {s.get('mean_tau_legs', 0):.3f} Nm")
    print(f"  Arm tau (mean):   {s.get('mean_tau_arms', 0):.3f} Nm")
    print(f"  Max tau:          {s.get('max_tau', 0):.3f} Nm")
    print(f"  Leg action RMS:   {s.get('rms_action_legs', 0):.4f}")
    print(f"  Arm action RMS:   {s.get('rms_action_arms', 0):.4f}")
    print(f"  Track err legs:   {s.get('track_err_legs', 0):.4f} rad")
    print(f"  Ang vel (mean):   {s.get('mean_ang_vel', 0):.4f} rad/s")
    print(f"  Ang vel spikes:   {s.get('n_ang_vel_spikes', 0)} events (>3σ)")
    print(f"  Upright (gz):     {s.get('upright', 0):.4f} (≈ -1 is upright)")
    print(f"  Cmd lin speed:    {s.get('mean_cmd_lin', 0):.3f} m/s")
    print(f"  Stationary frac:  {s.get('frac_stationary', 0):.1%}")
    print(f"  High-cmd frac:    {s.get('frac_high_cmd', 0):.1%}")
    # Comparability warning
    if frac_stationary_ref is not None:
        delta = abs(s.get('frac_stationary', 0) - frac_stationary_ref)
        if delta > 0.30:
            print(f"  ⚠ COMPARABILITY WARNING: stationary frac delta={delta:.1%} — torque/vel metrics may not be comparable")
        else:
            print(f"  Stationary frac delta={delta:.1%} — comparable")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parse_real_recording.py <recording1.pkl> [<recording2.pkl> ...]")
        sys.exit(1)

    all_stats = {}
    for path in sys.argv[1:]:
        d = load(path)
        name = path.split('/')[-1]

        kf = d['keyframe']
        kf_idx = [i for i, v in enumerate(kf) if v]
        if kf_idx:
            starts = [0] + [i + 1 for i in kf_idx[:-1]]
            ends   = kf_idx + [len(d['time'])]
        else:
            starts, ends = [0], [len(d['time'])]

        # IMU gap check
        dt_imu = np.diff(d['imu_timestamp'])
        expected_us = 1.0 / d['control_freq'] * 1e6
        gap_frac = np.mean(dt_imu > 2 * expected_us)
        if gap_frac > 0.05:
            print(f"⚠ {name}: IMU gap fraction {gap_frac:.1%} — results may be unreliable")

        if len(starts) <= 1:
            all_stats[name] = segment_stats(d)
            all_stats[name]['n_keyframes'] = len(kf_idx)
        else:
            segs = [segment_stats(d, s, e) for s, e in zip(starts, ends)]
            total_t = sum(s['duration'] for s in segs)
            avg = {k: sum(s[k] * s['duration'] for s in segs) / total_t
                   for k in segs[0] if k not in ('duration', 'n_steps')}
            avg['n_keyframes'] = len(kf_idx)
            all_stats[name] = avg

    # Comparability: frac_stationary reference = mean across all runs
    frac_ref = np.mean([s.get('frac_stationary', 0) for s in all_stats.values()])

    # Per-run print
    for name, stats in all_stats.items():
        short = name.replace('.pkl', '')
        print(f"\n=== {short} | {stats['duration']:.1f}s | {stats['n_steps']} steps | {stats['n_keyframes']} keyframes ===")
        _print_stats(stats)
        delta = abs(stats.get('frac_stationary', 0) - frac_ref)
        if delta > 0.30:
            print(f"  ⚠ Stationary frac delta={delta:.1%} vs mean — torque/vel NOT directly comparable")
        else:
            print(f"  Stationary frac delta={delta:.1%} vs mean — comparable")

    # Cross-run comparison table
    if len(all_stats) >= 2:
        print("\n### Summary table")
        keys = ['mean_vel_legs', 'mean_tau_legs', 'max_tau', 'rms_action_legs',
                'track_err_legs', 'mean_ang_vel', 'n_ang_vel_spikes', 'upright',
                'mean_cmd_lin', 'frac_stationary', 'frac_high_cmd']
        names = list(all_stats.keys())
        col_w = max(22, max(len(n) for n in names) + 2)
        header = f"{'':22}" + "".join(f"{n.replace('.pkl',''):>{col_w}}" for n in names)
        print(header)
        print("-" * (22 + col_w * len(names)))
        for k in keys:
            vals = [s.get(k, 0) for s in all_stats.values()]
            fmt_val = lambda v: f"{v:.4f}" if abs(v) < 100 else f"{v:.3f}"
            row = f"{k:<22}" + "".join(f"{fmt_val(v):>{col_w}}" for v in vals)
            print(row)
```

---

## 6. Visualization

```python
import matplotlib.pyplot as plt
import pickle, numpy as np

def plot_recording(path, save=None):
    with open(path, 'rb') as f:
        d = pickle.load(f)

    t = d['time'] - d['time'][0]
    kp = d['kp']; kd = d['kd']
    names = d['joint_names']
    leg_idx = [i for i, n in enumerate(names)
               if re.match(r'^(waist|.*hip|.*knee|.*ankle)', n)]
    arm_idx = [i for i, n in enumerate(names)
               if not re.match(r'^(waist|.*hip|.*knee|.*ankle)', n)]

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(14, 12))
    fig.suptitle(path.split('/')[-1])

    # 1. Command + estimated speed
    axes[0].plot(t, d['command'][:, 0], label='vx cmd', color='C0')
    axes[0].plot(t, d['command'][:, 1], label='vy cmd', color='C1')
    axes[0].plot(t, d['command'][:, 2], label='wz cmd', color='C2')
    axes[0].set_ylabel('Command')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Velocity Command [vx, vy, wz]')

    # 2. Base angular velocity
    for i, lbl in enumerate(['wx', 'wy', 'wz']):
        axes[1].plot(t, d['base_ang_vel'][:, i], label=lbl)
    axes[1].set_ylabel('rad/s')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Base Angular Velocity')

    # 3. Projected gravity (upright indicator)
    axes[2].plot(t, d['projected_gravity'], label=['gx', 'gy', 'gz'])
    axes[2].set_ylabel('g')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Projected Gravity (gz≈-1 is upright)')

    # 4. Mean leg torque over time (windowed)
    win = 20
    leg_tau = np.abs(d['q_tau'][:, leg_idx]).mean(axis=1)
    axes[3].plot(t, leg_tau, label='legs', color='C0')
    arm_tau = np.abs(d['q_tau'][:, arm_idx]).mean(axis=1)
    axes[3].plot(t, arm_tau, label='arms', color='C1')
    axes[3].set_ylabel('Nm')
    axes[3].legend(loc='upper right')
    axes[3].set_title(f'Mean |joint torque| (rolling win={win})')

    # 5. Tracking error (leg positions)
    err = np.sqrt(np.mean((d['q_pos'][:, leg_idx] - d['cmd_pos'][:, leg_idx])**2, axis=1))
    axes[4].plot(t, err, label='leg pos err', color='C0')
    axes[4].set_ylabel('rad')
    axes[4].set_xlabel('Time (s)')
    axes[4].legend(loc='upper right')
    axes[4].set_title('Leg Position Tracking RMSE')

    # Mark keyframes
    for ax in axes:
        for i, kf in enumerate(d['keyframe']):
            if kf:
                for a in ax.lines:
                    a.get_paths()[i].codes = None  # reset

    plt.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()
```

---

## 7. Comparison Template

```
### v6 vs v7 vs v9 (real robot deployment, {date})

Recording duration: v6=____s, v7=____s, v9=____s
Keyframes: v6=N, v7=N, v9=N

| Metric | v6 | v7 | v9 | Winner |
|---|---:|---:|---:|:---:|
| mean_vel_legs (rad/s) | | | | |
| mean_vel_arms (rad/s) | | | | |
| mean_tau_legs (Nm) | | | | |
| mean_tau_arms (Nm) | | | | |
| max_tau (Nm) | | | | |
| rms_action_legs | | | | |
| rms_action_arms | | | | |
| track_err_legs (rad) | | | | |
| mean_ang_vel (rad/s) | | | | |
| upright (gz) | | | | |
| mean_cmd_lin (m/s) | | | | |

Observations:
  [1-3 bullets on visible behavioral differences]
  [1-3 bullets on torque/activity differences]
  [1-3 bullets on what cannot be determined without simulation log]
```

---

## 8. Rules

1. **Always check `joint_names` match** across recordings before comparing per-joint
   arrays. Variant experiments may change the controlled joint set.
2. **Control frequency may differ** (typically 50 Hz real robot vs 50 Hz sim).
   Time-normalize before comparing if rates differ.
3. **Real data has no ground-truth base linear velocity.** If needed, estimate from
   IMU integration or FK, but flag as estimated.
4. **Keyframes are manual cuts** — they indicate operator-judged interesting segments,
   not episode boundaries. Do not treat them as proxy reward episodes.
5. **Torque values are hardware-sensor readings**, not feedforward. They include
   gravity-comp contribution on arms and policy output on legs.
6. **Compare same-command recordings only.** If one run has `command=0` and another
   has `command=[0.5,0,0]`, torque levels are not comparable.
7. **No statistical significance** from N=1 rollouts. Treat differences as
   qualitative indicators, not proof.

---

## 9. Workflow

```
Collect .pkl recordings (ongoing or post-deployment)
           ↓
Run mj_envs/parse_real_recording.py — compute per-run metrics
           ↓
Visualize with plot_recording.py — spot-check segments
           ↓
Fill comparison table with real numbers
           ↓
Identify behavioral differences (smoothness, torque, tracking)
           ↓
Cross-reference with simulation training.log if available
(e.g., did simulation predict the same instability?)
           ↓
Propose policy change → train in simulation → deploy → compare again
```

---

## 10. Failure Modes — What Can Go Wrong

| Failure | Symptom | Detection | Mitigation |
|---|---|---|---|
| **Script crash** | `AttributeError` on line 254 | Run script before presenting results | Fixed: use `import sys; sys.argv` |
| **Mismatched command distribution** | Variant A spent 80% at cmd=0; Variant B spent 80% at cmd=[0.5,0,0] → torque/vel numbers incomparable | Print `frac_stationary = mean(|command[:,:2]| < 0.05) / N` per recording; flag if delta > 0.3 | Segment by command regime; compare only matching segments |
| **IMU packet drops / gaps** | Time gaps between consecutive `imu_timestamp` > 2× expected dt | Check `np.diff(d['imu_timestamp'])` for gaps > 30 ms; flag and note fraction of dropped time | Exclude gap steps from statistics; note in report |
| **Recording only shows reset episode** | Whole recording is a single prepare→reset cycle, not a walk test | Check `duration`, `mean_vel`, `n_steps` — if all three are very low, recording is invalid | Discard from comparison |
| **Keyframes never set** | `keyframe` array is all `False` → whole recording treated as one segment | `np.sum(keyframe) == 0` | Correct behavior; whole-recording analysis is the fallback |
| **Operator interrupted recording** | Recording ends mid-step with no keyframe | Check if last 5% of recording has anomalously low activity | Note as truncation; exclude tail from analysis |
| **Recording concatenates post-reset segments** | Activity metrics spike after keyframe then go flat; two distinct regimes | Plot `cumsum(q_vel)` per segment; look for slope changes | Analyze segments independently |
| **Gravity vector ambiguity** | gz = −0.99 could mean upright standing OR robot lying on ground with same orientation | Combine with `base_ang_vel` and `q_vel` magnitude: if all three are near-zero simultaneously, robot is stationary (could be standing or fallen) | No reliable fall detection from gravity alone; flag as indeterminate |
| **Fall vs. recovery indistinguishable** | gz approaching 0 (robot tipped) followed by gz returning to −1 (reset) | Use `kf` (keyframe) as proxy; also scan for sharp `base_ang_vel` spikes > 3σ above mean | Note: no explicit fall-over metric available |
| **N=1 rollout** | All differences are qualitative; no statistical significance | Always label comparisons as **qualitative** | Never claim "variant X is better" — claim "variant X appears to" |
| **Comparing across control_freq** | If someone changes `control_freq` in the config, time-normalized metrics shift | Check `d['control_freq']` match across all files | Normalize time-based metrics to steps rather than seconds |

---

## 11. Additional Metrics to Compute

### 11.1 Stationary fraction (command comparability gate)

```python
cmd_lin = np.sqrt(d['command'][:, 0]**2 + d['command'][:, 1]**2)
frac_stationary = np.mean(cmd_lin < 0.05)   # fraction of time nearly stopped
frac_high_cmd   = np.mean(cmd_lin > 0.3)    # fraction at high speed
print(f"Stationary: {frac_stationary:.1%}  High-cmd: {frac_high_cmd:.1%}")
```

If two recordings have `frac_stationary` differing by > 30%, torque and velocity
metrics are **not directly comparable** — segment by command regime first.

### 11.2 Angular velocity spike detection (fall proxy)

```python
ang_vel_mag = np.sqrt(d['base_ang_vel'][:, 0]**2 + d['base_ang_vel'][:, 1]**2 + d['base_ang_vel'][:, 2]**2)
threshold = np.mean(ang_vel_mag) + 3 * np.std(ang_vel_mag)
spike_idx = np.where(ang_vel_mag > threshold)[0]
print(f"Ang-vel spikes: {len(spike_idx)} events, max={ang_vel_mag.max():.3f} rad/s")
```

### 11.3 Per-joint torque heatmap

```python
import matplotlib.pyplot as plt

mean_tau_per_joint = np.mean(np.abs(d['q_tau']), axis=0)
plt.figure(figsize=(12, 4))
plt.bar(d['joint_names'], mean_tau_per_joint)
plt.ylabel('Mean |torque| (Nm)')
plt.title(d['joint_names'][0].split('_')[0] + ' per-joint torque')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### 11.4 Base linear velocity estimation (FK + IMU integration)

No ground-truth linear velocity is recorded. Estimate with:
1. Integrate `base_ang_vel` for orientation
2. Use IMU accelerometer (projected_gravity changes) or FK COM velocity from `q_vel`

```python
# Simple: FK COM velocity in world frame from joint velocities
# Requires robot model — this is an approximation only
# Better: use policy's own base_lin_vel if deployed with estimator enabled
```

> **Note**: base linear velocity estimation from IMU integration drifts. Do not use
> for absolute speed comparison; use only for relative shape analysis (e.g., response
> to a push event).

### 11.5 IMU timestamp gap analysis

```python
dt_imu = np.diff(d['imu_timestamp'])
expected_dt = 1.0 / d['control_freq'] * 1e6  # microseconds
gaps = dt_imu > 2 * expected_dt
gap_frac = np.sum(gaps) / len(gaps)
print(f"IMU gaps >2×dt: {np.sum(gaps)} / {len(gaps)} steps ({gap_frac:.1%})")
if gap_frac > 0.05:
    print("WARNING: >5% packet drops — results may be unreliable")
```

---

## 12. File Inventory

Real robot recordings live in:
```
<control>/recordings/
```

Naming convention: `{TaskName}_{YYYYMMDD}_{HHMMSS}[_{suffix}].pkl`

Reference scripts:
```
mj_envs/parse_real_recording.py   # (to be created)
mj_envs/plot_real_recording.py    # (to be created, or inline)
```
