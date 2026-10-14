"""One SONIC sim2sim closed-loop run ON a Jetson (AGX Orin / AGX Thor): headless MuJoCo (sim_driver.py, SIM_POLL_CMDS=1,
board CPU) + upstream g1_deploy_onnx_ref built natively on the board (build_jetson.sh; engines built on the board by
upstream code at first launch, fp32 upstream default), driven by the SAME fixed keyboard script (SCRIPTS['v1']) as
run_closed_loop_x86.py on the deploy's stdin. Timestamps CLOCK_MONOTONIC.
Run as root (as in our Jetson runs) so the upstream SCHED_FIFO-99/CPU0 main-thread request takes effect; the reference
motions are recreated on tmpfs as root by mkref.py (systemd RemoveIPC wipes a user's /dev/shm files at logout).
Graceful stop only (stdin 'o', then SIGINT, then SIGTERM; never SIGKILL - the deploy holds a CUDA context).
usage (as root, env from sonic/env.sh): run_closed_loop_jetson.py --run_dir $SONIC_RUNS/cl_r1 --script v1 [--deploy_prefix '...']"""
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
ap.add_argument('--init_timeout', type=float, default=1800)
ap.add_argument('--deploy_prefix', default='')  # diagnostic only: 'setpriv --bounding-set=-sys_nice --inh-caps=-sys_nice --' makes the FIFO request fail
a = ap.parse_args(); a.run_dir = os.path.abspath(a.run_dir); os.makedirs(a.run_dir, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ['SONIC_REPO']; D = os.environ.get('SONIC_DEPLOY', f'{REPO}/gear_sonic_deploy'); PY = os.environ['SONIC_SIM_PY']
ORT = os.environ['SONIC_ORT']
sched = SCRIPTS[a.script]; events = []
def ev(kind, **kw): events.append(dict(t_mono_ns=time.monotonic_ns(), kind=kind, **kw))
def census(tag):
    c = {'t_mono_ns': time.monotonic_ns(), 'loadavg': open('/proc/loadavg').read().strip()}
    vm = lambda: {l.split()[0]: int(l.split()[1]) for l in open('/proc/vmstat') if l.startswith(('pswpin', 'pswpout'))}
    v0 = vm(); time.sleep(2); v1 = vm(); c['swap_pages_per_s'] = sum(v1[k] - v0[k] for k in v0) / 2
    c['mem_available_kB'] = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1])
    c['top_cpu'] = subprocess.run(['ps', '-eo', 'pcpu,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout.splitlines()[1:7]
    json.dump(c, open(f'{a.run_dir}/census_{tag}.json', 'w'), indent=1)
census('pre')
mk = subprocess.run([PY, f'{HERE}/mkref.py', f'{D}/reference/example'], capture_output=True, text=True).stdout.strip()
assert mk.startswith('ORDER_OK'), mk
ev('ref_dir', result=mk[:8])
senv = dict(os.environ, SIM_POLL_CMDS='1', PYTHONNOUSERSITE='1', PYTHONPATH=REPO, MUJOCO_GL='disable')
simlog = open(f'{a.run_dir}/sim_stdout.log', 'w')
sim = subprocess.Popen([PY, f'{HERE}/sim_driver.py', '--out', f'{a.run_dir}/sim_state.npz', '--max_s', '900'],
                       stdin=subprocess.PIPE, stdout=simlog, stderr=subprocess.STDOUT, text=True, cwd='/tmp', env=senv)
ev('sim_started', pid=sim.pid); time.sleep(6)
denv = dict(os.environ, SONIC_PROBE_DIR=a.run_dir, LD_LIBRARY_PATH=f'{ORT}/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')
cmd = [f'{D}/target/release/g1_deploy_onnx_ref', 'lo', 'policy/release/model_decoder.onnx', '/dev/shm/sonic_ref/example/',
       '--obs-config', 'policy/release/observation_config.yaml', '--encoder-file', 'policy/release/model_encoder.onnx',
       '--planner-file', 'planner/target_vel/V2/planner_sonic.onnx', '--input-type', 'keyboard', '--output-type', 'zmq',
       '--zmq-host', 'localhost', '--disable-crc-check']
dlog = open(f'{a.run_dir}/deploy_stdout.log', 'w')
cmd = a.deploy_prefix.split() + cmd
dep = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=D, env=denv)
ev('deploy_started', pid=dep.pid)
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
if ok:
    time.sleep(2.0); t0 = time.monotonic()
    for t, act in sched:
        dt = t0 + t - time.monotonic()
        if dt > 0: time.sleep(dt)
        if dep.poll() is not None: ev('deploy_died_early', rc=dep.returncode); break
        if act.startswith('select:'):
            k = load_order.index(act[7:])
            for _ in range(k): dep.stdin.write('n'); dep.stdin.flush(); time.sleep(0.15)
            ev('action', t_sched=t, act=act, n_presses=k); continue
        if act.startswith('sim:'): sim.stdin.write(act[4:] + '\n'); sim.stdin.flush()
        else: dep.stdin.write(act); dep.stdin.flush()
        if t == 0.0:   # thread snapshot 4 s after start: shows whether the main thread got SCHED_FIFO 99 (cls FF)
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
if os.environ.get('SUDO_UID'):   # hand the run dir back to the invoking user
    os.system(f"chown -R {os.environ['SUDO_UID']}:{os.environ.get('SUDO_GID', os.environ['SUDO_UID'])} {a.run_dir}")
print('done', a.run_dir, 'init_ok', ok, 'deploy_rc', dep.returncode)
