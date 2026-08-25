"""Record Real Robot Log with Chirp or Dwell Excitation.

# Chirp mode (frequency sweep):
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 1.36 --joint left_ankle_2_joint right_ankle_2_joint --no-fake
# Dwell mode (sample-and-hold random values):
python humanoid_record_log.py --mode dwell --duration 5.0 --dwell_time 0.5 --amp 0.1 --joint left_ankle_2_joint right_ankle_2_joint --no-fake 


# leg (amp is now in radians, converted to action internally)
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.8 --joint left_ankle_2_joint right_ankle_2_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.4 --joint left_ankle_1_joint right_ankle_1_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.8 --joint left_knee_joint right_knee_joint --no-fake --no-grav_comp
# hip 
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.5 --joint left_hip_3_joint right_hip_3_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp -0.4 0.4 --offset -0.6 0.6 --joint left_hip_2_joint right_hip_2_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.5 --joint left_hip_1_joint right_hip_1_joint --no-fake --no-grav_comp

# waist
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.5 --joint waist_joint --no-fake --no-grav_comp

# shoulder
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.8 --joint left_shoulder_1_joint right_shoulder_1_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.5 -0.5 --offset -0.5 0.5 --joint left_shoulder_2_joint right_shoulder_2_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp -0.3 0.3 --offset 0.3 -0.3 --joint left_shoulder_3_joint right_shoulder_3_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.8 --joint left_elbow_joint right_elbow_joint --no-fake --no-grav_comp
# wrist
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.4 --joint left_wrist_1_joint right_wrist_1_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.8 --offset -0.8 0.8 --joint left_wrist_2_joint right_wrist_2_joint --no-fake --no-grav_comp
python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.5 --joint left_wrist_3_joint right_wrist_3_joint --no-fake --no-grav_comp


python humanoid_record_log.py --duration 5.0 --min_freq 1.0 --max_freq 5.0 --amp 0.15 --no-fake --no-grav_comp
python humanoid_record_log.py --mode dwell --duration 5.0 --dwell_time 0.5 --amp 0.1 --no-fake --no-grav_comp

python humanoid_record_log.py --duration 5.0 --min_freq 0.5 --max_freq 6.0 --amp 0.05 -0.05 --joint left_shoulder_3_joint right_shoulder_3_joint --no-fake

"""
import time, pickle, tyro, sys, re, torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Local imports
sys.path.append(".")
from humanoid_real_env import HumanoidRealEnv, DEFAULT_CONFIG, URDF_JOINT_NAMES
from loop_rate_limiters import RateLimiter

# Joints that need sign reversal when mirroring left<->right (due to asymmetric limits)
MIRROR_SIGN_FLIP = {"hip_2", "shoulder_2", "shoulder_3"}

def get_symmetric_pairs() -> list[tuple[int, int, float]]:
    """Get (right_idx, left_idx, sign) pairs for symmetric motion.
    Sign is -1 for joints that need reversal, +1 otherwise."""
    pairs = []
    for i, name in enumerate(URDF_JOINT_NAMES):
        if name.startswith("right_"):
            left_name = "left_" + name[6:]  # replace "right_" with "left_"
            if left_name in URDF_JOINT_NAMES:
                left_idx = URDF_JOINT_NAMES.index(left_name)
                # Check if this joint type needs sign flip
                joint_type = name[6:].replace("_joint", "")  # e.g., "hip_2"
                sign = -1.0 if joint_type in MIRROR_SIGN_FLIP else 1.0
                pairs.append((i, left_idx, sign))
    return pairs

@dataclass
class Args:
    config: str = DEFAULT_CONFIG
    mode: str = "chirp"     # "chirp" or "dwell"
    duration: float = 5.0
    min_freq: float = 0.5   # Hz (chirp mode)
    max_freq: float = 3.0   # Hz (chirp mode)
    dwell_time: float = 0.5 # seconds to hold each sample (dwell mode)
    amp: tuple[float, ...] = (1.0,)  # position amplitude in rad, converted to action internally
    offset: tuple[float, ...] = (0.0,)  # position offset in rad, converted to action internally
    out: str | None = None  # Auto-generated from joint/mode/amp/freq if not set
    fake: bool = True
    tau_lim: float = 0.8
    joint: list[str] | None = None # Optional: Actuate only these joints (default: all)
    symmetric: bool = False  # Mirror right side motion to left side
    grav_comp: bool = True   # Enable gravity compensation

class Recorder:
    def __init__(self, N, nj, kp, kd, cur_kp, cur_ki, cur_filt, spd_filt, grav_comp_scale):
        self.max, self.cur = N, 0
        self.data = {
            'time': np.zeros(N),
            'frequency': np.zeros(N),
            'q_pos': np.zeros((N, nj)),
            'q_vel': np.zeros((N, nj)),
            'q_tau': np.zeros((N, nj)),
            'tau_grav_raw': np.zeros((N, nj)),      # raw gravity torque from MuJoCo
            'tau_grav_applied': np.zeros((N, nj)),  # applied torque (scaled + corrected)
            'cmd_pos': np.zeros((N, nj)),
            'actions': np.zeros((N, nj)),
            'base_ang_vel': np.zeros((N, 3)),
            'root_quat': np.zeros((N, 4)),
            'kp': kp.copy(),
            'kd': kd.copy(),
            'cur_kp': cur_kp.copy(),
            'cur_ki': cur_ki.copy(),
            'cur_filt': cur_filt.copy(),
            'spd_filt': spd_filt.copy(),
            'grav_comp_scale': grav_comp_scale,
        }
    
    def add(self, t, env, act, tgt, freq):
        if self.cur >= self.max: return
        i = self.cur
        self.data['time'][i] = t
        self.data['frequency'][i] = freq
        for k, v in [('q_pos', env.joint_pos), ('q_vel', env.joint_vel), ('q_tau', env.joint_torque),
                    ('tau_grav_raw', env.gravity_torque_raw), ('tau_grav_applied', env.gravity_torque_applied),
                    ('cmd_pos', tgt), ('base_ang_vel', env.base_ang_vel), ('actions', act)]:
            self.data[k][i] = v.cpu().numpy()
        
        q = env.base_quat_xyzw.cpu().numpy() # xyzw -> wxyz
        self.data['root_quat'][i] = [q[3], q[0], q[1], q[2]]
        self.cur += 1

    def save(self, name):
        print(f"Saving {self.cur} steps to {name}...")
        # Truncate time-series data, keep motor_kp/kd as-is (they're per-joint, not per-timestep)
        out = {}
        for k, v in self.data.items():
            if k in ('kp', 'kd', 'cur_kp', 'cur_ki', 'cur_filt', 'spd_filt', 'grav_comp_scale'):
                out[k] = v  # 1D per-joint arrays
            else:
                out[k] = v[:self.cur]  # time-series truncation
        with open(name, 'wb') as f: pickle.dump(out, f)

def gen_log_name(args) -> str:
    """Derive log filename from joints, mode, amp, and freq/dwell params."""
    # Joint group: strip left_/right_ and _joint, deduplicate
    if args.joint:
        parts = list(dict.fromkeys(re.sub(r"^(left|right)_|_joint$", "", j) for j in args.joint))
        joint_str = "_".join(parts)
    else:
        joint_str = "full"
    amp_str = "_".join(f"{a:g}" for a in args.amp)
    if args.mode == "chirp":
        param_str = f"f{args.min_freq:g}-{args.max_freq:g}"
    else:
        param_str = f"d{args.dwell_time:g}"
    # Gravity compensation: gc1.0 if enabled (scale=1.0), gc0 if disabled
    gc_str = "gc1.0" if args.grav_comp else "gc0"
    return f"logs/real_robot_log_{joint_str}_{args.mode}_a{amp_str}_{param_str}_{gc_str}.pkl"

def main():
    args = tyro.cli(Args)
    if args.out is None:
        args.out = gen_log_name(args)
    Path(args.out).parent.mkdir(exist_ok=True)
    if args.mode == "chirp":
        print(f"--- CHIRP: {args.duration}s, {args.min_freq}-{args.max_freq}Hz, A={args.amp} ---")
    else:
        print(f"--- DWELL: {args.duration}s, hold={args.dwell_time}s, A={args.amp} ---")

    # Pre-calculate joint indices
    target_idxs = None
    if args.joint:
        target_idxs = []
        for j in args.joint:
            if j not in URDF_JOINT_NAMES:
                raise ValueError(f"Joint '{j}' not found. Available: {URDF_JOINT_NAMES}")
            target_idxs.append(URDF_JOINT_NAMES.index(j))

    # Get symmetric pairs if enabled
    symmetric_pairs = get_symmetric_pairs() if args.symmetric else []
    if args.symmetric:
        print(f"Symmetric mode: mirroring {len(symmetric_pairs)} joint pairs")

    env = HumanoidRealEnv(args.config, torque_limit=args.tau_lim, use_fake=args.fake, gravity_compensation_enabled=args.grav_comp)

    env.prepare(duration=3.0)

    # Convert position-space amp/offset (rad) to action-space
    scales = env.action_scale[target_idxs].cpu().numpy() if target_idxs else env.action_scale.cpu().numpy()
    amps = np.array(args.amp) / scales
    offsets = np.array(args.offset) / scales

    # env.control_freq = 100.0  # Hz


    N = int(args.duration * env.control_freq)
    k = (args.max_freq - args.min_freq) / args.duration

    rec = Recorder(N + 100, env.n_joints, kp=env.motor_kp, kd=env.motor_kd,
                   cur_kp=env.motor.cur_kp, cur_ki=env.motor.cur_ki,
                   cur_filt=env.motor.cur_filt_gain, spd_filt=env.motor.spd_filt_gain,
                   grav_comp_scale=env.gravity_comp_scale)
    rate = RateLimiter(frequency=env.control_freq, warn=True)

    # Dwell state: sample random value, hold for dwell_time
    dwell_val = np.random.uniform(-amps, amps)
    dwell_next_sample_time = args.dwell_time

    print("\nStarting...")
    t0 = time.time()
    try:
        for _ in range(N):
            rate.sleep()
            t = time.time() - t0

            if args.mode == "chirp":
                # Chirp: f(t) = f0 + kt -> Phase = 2pi * (f0*t + k*t^2/2)
                val = amps * np.sin(2 * np.pi * (args.min_freq * t + k * t**2 / 2))
                freq = args.min_freq + k * t
            else:
                # Dwell: hold constant, resample after dwell_time
                if t >= dwell_next_sample_time:
                    dwell_val = np.random.uniform(-amps, amps)
                    dwell_next_sample_time += args.dwell_time
                val = dwell_val
                freq = 0.0

            val = val + offsets
            if target_idxs is not None:
                act = torch.zeros(env.n_joints, device=env.device)
                act[target_idxs] = torch.from_numpy(val).float().to(env.device)
            else:
                act = torch.from_numpy(val).float().to(env.device).expand(env.n_joints).clone()

            # Apply symmetric mirroring: copy right side to left side with sign flip
            if args.symmetric:
                for right_idx, left_idx, sign in symmetric_pairs:
                    act[left_idx] = act[right_idx] * sign

            env.step(act)
            rec.add(t, env, act, env.default_joint_pos + act * env.action_scale, freq)

    except KeyboardInterrupt: print("\nInterrupted.")
    finally: 
        rec.save(args.out)
        env.shutdown()
        time.sleep(0.1)
        env.motor.disable(False)

if __name__ == "__main__": main()
