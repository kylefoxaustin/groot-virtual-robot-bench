"""Headless MuJoCo sim for SONIC sim2sim, mirroring gear_sonic/scripts/run_sim_loop.py + BaseSimulator.start(),
with (a) commands on stdin (band_off / band_on / quit), (b) a 50 Hz state log, (c) a count of the sim's own
fall-resets (base_sim.check_fall resets the scene when pelvis z < 0.2 m).
Log: npz with t_mono_ns (CLOCK_MONOTONIC, same clock as C++ steady_clock), sim_time, qpos (full), qvel[:6],
band (0/1), resets (cumulative), overrun (cumulative sim steps that exceeded sim_dt wall budget).
Run with the upstream gear_sonic package importable (PYTHONPATH=<GR00T-WholeBodyControl checkout>).
usage: sim_driver.py --out sim_state.npz [--max_s 900]      env: SIM_POLL_CMDS=1, SIM_RELEASE_DELAY_MS=D (see below), SIM_DBG=1"""
import argparse, os, select, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--max_s', type=float, default=900)
a = ap.parse_args()

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

cfg = SimLoopConfig()
cfg.enable_onscreen = False
cfg.enable_offscreen = False
wbc = cfg.load_wbc_yaml()
wbc['ENV_NAME'] = cfg.env_name
init_channel(config=wbc)
sim = SimulatorFactory.create_simulator(config=wbc, env_name=cfg.env_name, onscreen=False, offscreen=False,
                                        enable_image_publish=False)
env = sim.sim_env
resets = [0]
_orig_reset = env.reset
def _counting_reset():
    resets[0] += 1
    _orig_reset()
env.reset = _counting_reset

# --- harness change (disclosed; used on the Jetsons, NOT on the RTX 5090): poll the command readers instead of the
# upstream DDS listener callbacks. The upstream listener runs a Python callback per sample from the DDS receive thread
# and needs the GIL held by this physics loop; on a slower CPU it can fall behind the deploy's 500 Hz LowCmd stream, so
# the sim applies commands late. Polling takes the LATEST sample of a KEEP_LAST(1) reader once per physics step
# (<= 5 ms age). Enabled with SIM_POLL_CMDS=1; unset = upstream behaviour.
POLL = os.environ.get('SIM_POLL_CMDS') == '1'
polls = []
if POLL:
    import gc
    from cyclonedds.sub import DataReader
    from cyclonedds.internal import InvalidSample
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, HandCmd_
    br = env.unitree_bridge
    for nm in ('low_cmd_suber', 'left_hand_cmd_suber', 'right_hand_cmd_suber'):
        getattr(br, nm).Close()
    gc.collect()
    def _find_reader(o, depth=0):
        if isinstance(o, DataReader): return o
        if depth > 3 or not hasattr(o, '__dict__'): return None
        for v in vars(o).values():
            r = _find_reader(v, depth + 1)
            if r is not None: return r
        return None
    for topic, T, h in (('rt/lowcmd', LowCmd_, br.LowCmdHandler), ('rt/dex3/left/cmd', HandCmd_, br.LeftHandCmdHandler),
                        ('rt/dex3/right/cmd', HandCmd_, br.RightHandCmdHandler)):
        s_ = ChannelSubscriber(topic, T); s_.Init(None); rd = _find_reader(s_)
        assert rd is not None, topic
        polls.append((s_, rd, h))
    print('[sim_driver] SIM_POLL_CMDS=1: command readers polled once per physics step', flush=True)
def poll_cmds():
    for s_, rd, h in polls:
        try:
            got = [x for x in rd.take(N=4) if not isinstance(x, InvalidSample)]
        except Exception:
            got = []
        if got: h(got[-1])

dt = sim.sim_dt
# --- harness change (disclosed; the controlled band-drop test only): controlled band release. With SIM_RELEASE_DELAY_MS=D set,
# stdin is polled EVERY physics step (unset: every 20 steps = 100 ms, as in the first campaign) and 'band_off [t_cmd_ns]' schedules
# the release at the first physics step that starts at or after t_cmd_ns + D ms (t_cmd_ns = the orchestrator's CLOCK_MONOTONIC at
# its stdin write; receipt time if absent). The release readback (command, receipt and release times, physics steps, pelvis tilt
# and horizontal speed at release) is printed as one JSON line and saved in the npz as release_json.
import json as _json
REL_D = os.environ.get('SIM_RELEASE_DELAY_MS', '')
REL_D = float(REL_D) if REL_D else None
POLL_EVERY = 1 if REL_D is not None else 20
rel = dict(D_ms=REL_D, dt_s=float(dt), poll_every_steps=POLL_EVERY)
rel_target_ns = None
def _tilt_vxy():
    q = np.asarray(env.mj_data.qpos[3:7], dtype=float); q = q / np.linalg.norm(q)
    cz = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])   # world-z component of the pelvis z axis (MuJoCo quat w,x,y,z)
    return float(np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))), float(np.hypot(env.mj_data.qvel[0], env.mj_data.qvel[1]))
log_every = int(round(0.02 / dt))
T, ST, QP, QV, BD, RS, OV = [], [], [], [], [], [], []
overruns = 0
cnt = 0
t_end = time.monotonic() + a.max_s
print(f'[sim_driver] ready dt={dt} log_every={log_every} nq={env.mj_model.nq}', flush=True)
quit_ = False
DBG = os.environ.get('SIM_DBG') == '1'
t0 = time.monotonic()
try:
    while not quit_ and time.monotonic() < t_end:
        # non-blocking stdin commands
        if cnt % POLL_EVERY == 0:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                line = sys.stdin.readline().strip()
                tok = line.split(); cmd0 = tok[0] if tok else ''
                if cmd0 == 'band_off' and env.elastic_band is not None and REL_D is not None:
                    t_rx = time.monotonic_ns(); t_cmd = int(tok[1]) if len(tok) > 1 else t_rx
                    rel_target_ns = t_cmd + int(round(REL_D * 1e6))
                    rel.update(t_cmd_ns=t_cmd, t_rx_ns=t_rx, step_rx=cnt, rx_latency_ms=(t_rx - t_cmd) / 1e6)
                    print('[sim_driver] band_off scheduled +%g ms' % REL_D, flush=True)
                elif cmd0 == 'band_off' and env.elastic_band is not None:
                    env.elastic_band.enable = False; print('[sim_driver] band_off', flush=True)
                elif cmd0 == 'band_on' and env.elastic_band is not None:
                    env.elastic_band.enable = True; print('[sim_driver] band_on', flush=True)
                elif line == 'quit' or line == '':
                    quit_ = True
        if POLL: poll_cmds()
        if rel_target_ns is not None and time.monotonic_ns() >= rel_target_ns:
            _tl, _vx = _tilt_vxy()
            env.elastic_band.enable = False; t_rel = time.monotonic_ns(); rel_target_ns = None
            rel.update(t_release_ns=t_rel, step_release=cnt, steps_rx_to_release=cnt - rel['step_rx'],
                       sim_time_release=float(env.mj_data.time), release_delay_ms=(t_rel - rel['t_cmd_ns']) / 1e6,
                       tilt_at_release_deg=_tl, vxy_at_release_mps=_vx, resets_at_release=resets[0], overruns_at_release=overruns)
            print('[sim_driver] RELEASE ' + _json.dumps(rel), flush=True)
        env.sim_step()
        if cnt % int(sim.reward_dt / dt) == 0:
            env.update_reward()
        if DBG and cnt % 200 == 0:
            br = sim.sim_env.unitree_bridge if hasattr(sim.sim_env, 'unitree_bridge') else None
            print(f'[dbg] t={time.monotonic()-t0:.1f} bridge={br is not None} recv={getattr(br,"low_cmd_received",None)} kp0={br.low_cmd.motor_cmd[0].kp if br else None}', flush=True)
        if cnt % log_every == 0:
            T.append(time.monotonic_ns()); ST.append(env.mj_data.time)
            QP.append(env.mj_data.qpos.copy()); QV.append(env.mj_data.qvel[:6].copy())
            BD.append(1 if (env.elastic_band is not None and env.elastic_band.enable) else 0)
            RS.append(resets[0]); OV.append(overruns)
        # Absolute-deadline pacing (upstream uses sleep(dt - elapsed), which accumulates sleep overshoot and
        # measured ~0.82x real time on the RTX 5090 host; the C++ controller runs on wall clock, so we pace to wall clock).
        cnt += 1
        next_t = t0 + cnt * dt
        now = time.monotonic()
        if now > next_t + dt:
            overruns += 1
        if now < next_t:
            time.sleep(next_t - now)
finally:
    np.savez(a.out, t_mono_ns=np.array(T, dtype=np.int64), sim_time=np.array(ST), qpos=np.array(QP),
             qvel6=np.array(QV), band=np.array(BD), resets=np.array(RS), overruns=np.array(OV),
             dt=dt, steps=cnt, release_json=_json.dumps(rel))
    print(f'[sim_driver] saved {len(T)} samples, steps={cnt}, resets={resets[0]}, overruns={overruns}', flush=True)
    os._exit(0)
