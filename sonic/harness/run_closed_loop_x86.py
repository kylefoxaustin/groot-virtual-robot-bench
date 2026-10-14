"""One SONIC sim2sim closed-loop run on an x86 + NVIDIA GPU host (our RTX 5090 runs v1_r1..v1_r3):
headless MuJoCo (sim_driver.py, upstream DDS listener) + upstream C++/TensorRT g1_deploy_onnx_ref in the container
from sonic/harness/Dockerfile (tag sonic-deploy:trt1013), driven by a FIXED keyboard script on the deploy's stdin.
All timestamps CLOCK_MONOTONIC (shared by host python and container C++ steady_clock).
The container runs with --user, so the deploy is non-root and its SCHED_FIFO-99 request fails silently (as in our runs).
usage: source sonic/env.sh; python3 sonic/harness/run_closed_loop_x86.py --run_dir $SONIC_RUNS/v1_r1 --script v1"""
import argparse, json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ['SONIC_REPO']; TRT = os.environ['SONIC_TRT']; SIMPY = os.environ['SONIC_SIM_PY']

# (seconds after control start ']', action). action = key string for deploy stdin, or 'sim:<cmd>'.
SCRIPTS = {
 'smoke': [(0.0, ']'), (2.0, 'sim:band_off'), (8.0, 'o')],
 'v1': [
  (0.0, ']'),            # start control (robot held by elastic band)
  (2.0, 'sim:band_off'), # drop robot to ground (upstream: press 9 in viewer)
  (7.0, '\n'),           # ENTER -> planner mode (Locomotion set, idle)
  (10.0, '2'), (10.1, 'w'),            # walk forward
  (18.0, 'q'), (18.5, 'q'), (19.0, 'q'), (19.1, 'w'),  # turn facing (3 x pi/6) while walking
  (25.0, '3'), (25.1, 'w'),            # run (mode default speed)
  (31.0, 'r'),                          # momentum reset -> idle
  (36.0, '1'), (36.1, 'w'),            # slow walk
  (41.0, 'r'),                          # stop
  (45.0, '\n'),                         # back to reference-motion (normal) mode
  (47.0, 'select:macarena_001__A545'),  # 'n' presses computed from the deploy's motion load order
  (49.0, 't'),                          # play reference clip (1375 frames = 27.5 s) to completion
  (82.0, 'o'),                          # stop control and exit
 ],
}

ap = argparse.ArgumentParser()
ap.add_argument('--run_dir', required=True)
ap.add_argument('--script', default='v1')
ap.add_argument('--init_timeout', type=float, default=900)
a = ap.parse_args()
os.makedirs(a.run_dir, exist_ok=True)
sched = SCRIPTS[a.script]
events = []
def ev(kind, **kw):
    events.append(dict(t_mono_ns=time.monotonic_ns(), kind=kind, **kw))

simlog = open(f'{a.run_dir}/sim_stdout.log', 'w')
sim = subprocess.Popen([SIMPY, f'{HERE}/sim_driver.py', '--out', f'{os.path.abspath(a.run_dir)}/sim_state.npz', '--max_s', '1200'],
                       stdin=subprocess.PIPE, stdout=simlog, stderr=subprocess.STDOUT, text=True, cwd='/tmp',
                       env=dict(os.environ, PYTHONPATH=REPO, MUJOCO_GL='disable'))
ev('sim_started', pid=sim.pid)
time.sleep(4)

uid, gid = os.getuid(), os.getgid()
cmd = ['docker', 'run', '--rm', '-i', '--name', f'sonic_{os.getpid()}', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
       '--user', f'{uid}:{gid}', '-v', '/etc/passwd:/etc/passwd:ro', '-v', '/etc/group:/etc/group:ro', '-e', 'HOME=/tmp', '-e', 'SONIC_PROBE_DIR=/probe',
       '-v', f'{REPO}:/workspace/repo', '-v', f'{TRT}:/opt/TensorRT:ro', '-v', f'{os.path.abspath(a.run_dir)}:/probe',
       '-w', '/workspace/repo/gear_sonic_deploy', 'sonic-deploy:trt1013',
       './target/release/g1_deploy_onnx_ref', 'lo', 'policy/release/model_decoder.onnx', 'reference/example/',
       '--obs-config', 'policy/release/observation_config.yaml',
       '--encoder-file', 'policy/release/model_encoder.onnx',
       '--planner-file', 'planner/target_vel/V2/planner_sonic.onnx',
       '--input-type', 'keyboard', '--output-type', 'zmq', '--zmq-host', 'localhost', '--disable-crc-check']
dlog = open(f'{a.run_dir}/deploy_stdout.log', 'w')
dep = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
ev('deploy_started')
ready = threading.Event()
load_order = []
def reader():
    for line in dep.stdout:
        dlog.write(f'{time.monotonic_ns()} {line}'); dlog.flush()
        if 'Init Done' in line: ready.set()
        if '✓ Loaded ' in line: load_order.append(line.split('✓ Loaded ')[1].split(' (')[0])
    ready.set()  # stdout closed (process exited) -> unblock waiter
threading.Thread(target=reader, daemon=True).start()
ok = ready.wait(a.init_timeout) and dep.poll() is None
ev('init_done' if ok else 'init_timeout')
if ok:
    time.sleep(2.0)
    t0 = time.monotonic()
    for t, act in sched:
        dt = t0 + t - time.monotonic()
        if dt > 0: time.sleep(dt)
        if act.startswith('select:'):
            k = load_order.index(act[7:])   # current index is 0 after start
            for _ in range(k):
                dep.stdin.write('n'); dep.stdin.flush(); time.sleep(0.15)
            ev('action', t_sched=t, act=act, n_presses=k)
            continue
        if act.startswith('sim:'):
            sim.stdin.write(act[4:] + '\n'); sim.stdin.flush()
        else:
            dep.stdin.write(act); dep.stdin.flush()
        ev('action', t_sched=t, act=act)
# wait for the deploy to exit after 'o' (graceful: docker stop sends SIGTERM; never SIGKILL a process holding a CUDA context)
try:
    dep.wait(timeout=60)
except subprocess.TimeoutExpired:
    ev('deploy_exit_timeout'); subprocess.run(['docker', 'stop', '-t', '20', f'sonic_{os.getpid()}'])
    dep.wait(timeout=60)
ev('deploy_exited', rc=dep.returncode)
sim.stdin.write('quit\n'); sim.stdin.flush()
try: sim.wait(timeout=30)
except subprocess.TimeoutExpired: sim.terminate(); sim.wait(timeout=30)
ev('sim_exited', rc=sim.returncode)
json.dump(dict(load_order=load_order, script=a.script, schedule=sched, events=events), open(f'{a.run_dir}/run_events.json', 'w'), indent=1)
print('done', a.run_dir, 'init_ok', ok, 'deploy_rc', dep.returncode)
