"""Headless MuJoCo sim for SONIC sim2sim, mirroring gear_sonic/scripts/run_sim_loop.py + BaseSimulator.start(),
with (a) commands on stdin (band_off / band_on / quit), (b) a 50 Hz state log, (c) a count of the sim's own
fall-resets (base_sim.check_fall resets the scene when pelvis z < 0.2 m).
Log: npz with t_mono_ns (CLOCK_MONOTONIC, same clock as C++ steady_clock), sim_time, qpos (full), qvel[:6],
band (0/1), resets (cumulative), overrun (cumulative sim steps that exceeded sim_dt wall budget).
Run with the upstream gear_sonic package importable (PYTHONPATH=<GR00T-WholeBodyControl checkout>).
usage: sim_driver.py --out sim_state.npz [--max_s 900]      env: SIM_POLL_CMDS=1 (see below), SIM_DBG=1"""
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
        if cnt % 20 == 0:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                line = sys.stdin.readline().strip()
                if line == 'band_off' and env.elastic_band is not None:
                    env.elastic_band.enable = False; print('[sim_driver] band_off', flush=True)
                elif line == 'band_on' and env.elastic_band is not None:
                    env.elastic_band.enable = True; print('[sim_driver] band_on', flush=True)
                elif line == 'quit' or line == '':
                    quit_ = True
        if POLL: poll_cmds()
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
             dt=dt, steps=cnt)
    print(f'[sim_driver] saved {len(T)} samples, steps={cnt}, resets={resets[0]}, overruns={overruns}', flush=True)
    os._exit(0)
