"""One SONIC sim2sim closed-loop run ON the IQ-9075: headless MuJoCo (sonic/harness/sim_driver.py with SIM_POLL_CMDS=1, on the
board's CPU) + upstream g1_deploy_onnx_ref cross-built with the QNN shim (iq9/sonic/deploy: encoder/decoder on NSP0, planner
blocks on NSP1, planner host stages on the CPU), driven by the SAME fixed keyboard script (SCRIPTS['v1']) as
sonic/harness/run_closed_loop_x86.py and run_closed_loop_jetson.py, on the deploy's stdin. Timestamps CLOCK_MONOTONIC.
Run as root, like the Jetson runs: upstream then gives its main thread SCHED_FIFO 99 on CPU0 (and the late-start settings need
root or CAP_SYS_NICE for SCHED_FIFO). Reference motions are recreated on tmpfs by sonic/harness/mkref.py so the deploy's
directory_iterator order matches the reference runs (the first motion loaded sets the stand pose; the dance clip is picked by index).
Graceful stop only (stdin 'o', then SIGINT, then SIGTERM; never SIGKILL - the deploy holds NSP sessions).
Options: --release_delay_ms D (controlled band drop: the simulator releases the band D ms after the band-off command, which carries
this process's CLOCK_MONOTONIC write time); --deploy_env KEY=VAL,... (the late-start settings: SONIC_CTRL_FIFO=90,
SONIC_QNN_PLANNER_THREADS=1,SONIC_MAIN_SLEEP=abs); --cpu_arm (encoder+decoder on the CPU with onnxruntime fp32: the control arm).
Every run also writes levers_readback.json: scheduling policy, priority, affinity and CPU time of every deploy and simulator thread
from /proc, plus the cpufreq governor, read after init, at script t+20 s and at the stop command.
usage (as root, from $IQ9_SONIC_BOARD_DIR with env.sh sourced): run_closed_loop_iq9.py --run_dir runs/cl_r1 --script v1 [...]"""
import argparse, json, os, subprocess, sys, threading, time
SCRIPTS = {
 'smoke': [(0.0, ']'), (2.0, 'sim:band_off'), (8.0, 'o')],
 'v1': [
  (0.0, ']'), (2.0, 'sim:band_off'), (7.0, '\n'),
  (10.0, '2'), (10.1, 'w'),
  (18.0, 'q'), (18.5, 'q'), (19.0, 'q'), (19.1, 'w'),
  (25.0, '3'), (25.1, 'w'), (31.0, 'r'),
  (36.0, '1'), (36.1, 'w'), (41.0, 'r'),
  (45.0, '\n'), (47.0, 'select:macarena_001__A545'), (49.0, 't'), (82.0, 'o'),
 ],
}
ap = argparse.ArgumentParser()
ap.add_argument('--run_dir', required=True); ap.add_argument('--script', default='v1')
ap.add_argument('--init_timeout', type=float, default=600)
ap.add_argument('--deploy_prefix', default='')   # diagnostic only: 'setpriv --bounding-set=-sys_nice --inh-caps=-sys_nice --'
ap.add_argument('--deploy_env', default='')      # extra KEY=VAL,KEY=VAL for the deploy
ap.add_argument('--cpu_arm', action='store_true')
ap.add_argument('--release_delay_ms', default='')
ap.add_argument('--readback_at', type=float, default=20.0)
a = ap.parse_args(); a.run_dir = os.path.abspath(a.run_dir); os.makedirs(a.run_dir, exist_ok=True)
B = os.environ.get('IQ9_SONIC_BOARD_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QNN = os.environ['IQ9_QNN_ROOT']; PY = os.path.join(os.environ['IQ9_PY_ROOT'], 'bin', 'python3')
sched = SCRIPTS[a.script]; events = []
def ev(kind, **kw): events.append(dict(t_mono_ns=time.monotonic_ns(), kind=kind, **kw))
def census(tag):
    c = {'t_mono_ns': time.monotonic_ns(), 'loadavg': open('/proc/loadavg').read().strip()}
    vm = lambda: {l.split()[0]: int(l.split()[1]) for l in open('/proc/vmstat') if l.startswith(('pswpin', 'pswpout'))}
    v0 = vm(); time.sleep(2); v1 = vm(); c['swap_pages_per_s'] = sum(v1[k] - v0[k] for k in v0) / 2
    c['mem_available_kB'] = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1])
    c['remoteproc'] = subprocess.run('cat /sys/class/remoteproc/*/state', shell=True, capture_output=True, text=True).stdout.split()
    c['top_cpu'] = subprocess.run(['ps', '-eo', 'pcpu,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout.splitlines()[1:7]
    json.dump(c, open(f'{a.run_dir}/census_{tag}.json', 'w'), indent=1)
census('pre')
mk = subprocess.run([PY, f'{B}/simsrc/mkref.py', f'{B}/work/reference/example'], capture_output=True, text=True).stdout.strip()
assert mk.startswith('ORDER_OK'), mk
ev('ref_dir', result=mk[:8])
senv = dict(os.environ, SIM_POLL_CMDS='1', PYTHONNOUSERSITE='1', PYTHONPATH=f'{B}/simpy:{B}/simsrc',
            CYCLONEDDS_HOME=f'{B}/simsrc/cyclone_home', MUJOCO_GL='disable')
if a.release_delay_ms: senv['SIM_RELEASE_DELAY_MS'] = str(a.release_delay_ms)
simlog = open(f'{a.run_dir}/sim_stdout.log', 'w')
sim = subprocess.Popen([PY, f'{B}/simsrc/sim_driver.py', '--out', f'{a.run_dir}/sim_state.npz', '--max_s', '900'],
                       stdin=subprocess.PIPE, stdout=simlog, stderr=subprocess.STDOUT, text=True, cwd='/tmp', env=senv)
ev('sim_started', pid=sim.pid); time.sleep(6)
denv = dict(os.environ, SONIC_PROBE_DIR=a.run_dir, LD_LIBRARY_PATH=f'{B}/lib:{QNN}/lib', ADSP_LIBRARY_PATH=f'{QNN}/skel',
            SONIC_QNN_BACKEND=f'{QNN}/lib/libQnnHtp.so', SONIC_QNN_SYSLIB=f'{QNN}/lib/libQnnSystem.so',
            SONIC_QNN_ENC=f'{B}/ctx/sonic_enc_core_aaf0_fp16.bin', SONIC_QNN_DEC=f'{B}/ctx/sonic_dec_aaf0_fp16.bin',
            SONIC_QNN_PLANNER_DIR=f'{B}/planner', SONIC_QNN_ENC_DEV='0', SONIC_QNN_DEC_DEV='0', SONIC_QNN_PLANNER_DEV='1')
if a.cpu_arm:
    denv.update(SONIC_CPU_ENC_ONNX=f'{B}/onnx/model_encoder.onnx', SONIC_CPU_DEC_ONNX=f'{B}/onnx/model_decoder.onnx')
for kv in [x for x in a.deploy_env.split(',') if x]:
    k_, v_ = kv.split('=', 1); denv[k_] = v_
cmd = [f'{B}/bin/g1_deploy_onnx_ref_qnn', 'lo', 'policy/release/model_decoder.onnx', '/dev/shm/sonic_ref/example/',
       '--obs-config', 'policy/release/observation_config.yaml', '--encoder-file', 'policy/release/model_encoder.onnx',
       '--planner-file', 'planner/target_vel/V2/planner_sonic.onnx', '--input-type', 'keyboard', '--output-type', 'zmq',
       '--zmq-host', 'localhost', '--disable-crc-check']
dlog = open(f'{a.run_dir}/deploy_stdout.log', 'w')
cmd = a.deploy_prefix.split() + cmd
dep = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                       cwd=f'{B}/work', env=denv)
ev('deploy_started', pid=dep.pid, deploy_env={k_: v_ for k_, v_ in denv.items() if k_.startswith('SONIC_') and k_ != 'SONIC_PROBE_DIR'})
ready = threading.Event(); load_order = []
def reader():
    for line in dep.stdout:
        dlog.write(f'{time.monotonic_ns()} {line}'); dlog.flush()
        if 'Init Done' in line: ready.set()
        if '✓ Loaded ' in line: load_order.append(line.split('✓ Loaded ')[1].split(' (')[0])
    ready.set()
threading.Thread(target=reader, daemon=True).start()
t_wait = time.monotonic()
while not ready.wait(2.0):
    if sim.poll() is not None: ev('sim_died_before_init', rc=sim.returncode); break
    if time.monotonic() - t_wait > a.init_timeout: break
ok = ready.is_set() and dep.poll() is None and sim.poll() is None
ev('init_done' if ok else 'init_timeout', init_s=time.monotonic() - t_wait)
HZ = os.sysconf('SC_CLK_TCK')
def readback(tag):   # every deploy + simulator thread: policy (0 OTHER, 1 FIFO), rt priority, nice, affinity, utime/stime (clock ticks)
    out = dict(tag=tag, t_mono_ns=time.monotonic_ns(), clk_tck=HZ, procs={}, cpufreq={})
    for nm, pid in (('deploy', dep.pid), ('sim', sim.pid)):
        th = []
        try: tids = sorted(os.listdir(f'/proc/{pid}/task'), key=int)
        except Exception: tids = []
        for t_ in tids:
            try:
                comm = open(f'/proc/{pid}/task/{t_}/comm').read().strip()
                st = open(f'/proc/{pid}/task/{t_}/stat').read(); f = st[st.rindex(')') + 2:].split()
                aff = [l.split(':', 1)[1].strip() for l in open(f'/proc/{pid}/task/{t_}/status') if l.startswith('Cpus_allowed_list')][0]
                th.append(dict(tid=int(t_), main=(int(t_) == pid), comm=comm, state=f[0], nice=int(f[16]), cpu=int(f[36]), rtprio=int(f[37]),
                               policy=int(f[38]), cpus_allowed=aff, utime=int(f[11]), stime=int(f[12])))
            except Exception as e: th.append(dict(tid=int(t_), err=str(e)))
        out['procs'][nm] = th
    for c in range(os.cpu_count()):
        b_ = f'/sys/devices/system/cpu/cpu{c}/cpufreq/'
        try: out['cpufreq'][c] = {k: open(b_ + k).read().strip() for k in ('scaling_governor', 'scaling_cur_freq', 'scaling_min_freq', 'scaling_max_freq')}
        except Exception as e: out['cpufreq'][c] = str(e)
    return out
rb = [readback('after_init')] if ok else []
if ok:
    time.sleep(2.0); t0 = time.monotonic()
    threading.Timer(a.readback_at, lambda: rb.append(readback(f'script_t+{a.readback_at:g}s'))).start()
    for t, act in sched:
        dt = t0 + t - time.monotonic()
        if dt > 0: time.sleep(dt)
        if dep.poll() is not None: ev('deploy_died_early', rc=dep.returncode); break
        if act.startswith('select:'):
            k = load_order.index(act[7:])
            for _ in range(k): dep.stdin.write('n'); dep.stdin.flush(); time.sleep(0.15)
            ev('action', t_sched=t, act=act, n_presses=k); continue
        if act == 'o': rb.append(readback('at_o'))
        if act == 'sim:band_off' and a.release_delay_ms:
            t_cmd = time.monotonic_ns(); sim.stdin.write(f'band_off {t_cmd}\n'); sim.stdin.flush(); ev('band_off_cmd', t_cmd_ns=t_cmd)
        elif act.startswith('sim:'): sim.stdin.write(act[4:] + '\n'); sim.stdin.flush()
        else: dep.stdin.write(act); dep.stdin.flush()
        if t == 0.0:   # thread snapshot 4 s after start: scheduling class and priority of every thread
            threading.Timer(4.0, lambda: open(f'{a.run_dir}/threads_snapshot.txt', 'w').write(subprocess.run(['ps', '-eLo', 'pid,tid,psr,cls,rtprio,ni,pcpu,comm'], capture_output=True, text=True).stdout)).start()
        ev('action', t_sched=t, act=act)
else:
    try: dep.stdin.write('o'); dep.stdin.flush()
    except Exception: pass
for sig, to in [(None, 60), (2, 30), (15, 60)]:
    try:
        if sig: dep.send_signal(sig); ev('deploy_signal', sig=sig)
        dep.wait(timeout=to); break
    except subprocess.TimeoutExpired: continue
ev('deploy_exited', rc=dep.returncode)
try: sim.stdin.write('quit\n'); sim.stdin.flush()
except Exception: pass
try: sim.wait(timeout=30)
except subprocess.TimeoutExpired: sim.terminate(); sim.wait(timeout=30)
ev('sim_exited', rc=sim.returncode)
census('post')
json.dump(dict(load_order=load_order, script=a.script, schedule=sched, events=events), open(f'{a.run_dir}/run_events.json', 'w'), indent=1)
json.dump(rb, open(f'{a.run_dir}/levers_readback.json', 'w'), indent=0)
print('done', a.run_dir, 'init_ok', ok, 'deploy_rc', dep.returncode)
