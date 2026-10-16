"""The drone simulator: a kinematic quadrotor in MuJoCo with a forward-facing FPV camera.

GR00T sits at the GUIDANCE layer: it emits body-frame velocity setpoints [vx, vy, vz, yaw_rate] (m/s, rad/s), which are
integrated kinematically every 20 ms (a real drone's flight controller would track them). The scene (sim/scene.xml) is a
checkered floor, a sky, and four coloured pillars; the task is "fly to the <colour> pillar", reached when the drone is
within 0.8 m of it horizontally. Rendering is 224 x 224 RGB from the 'fpv' camera.

Rendering backend: MuJoCo's EGL. For runs you want to compare episode by episode with ours, render with Mesa's software
rasteriser llvmpipe (bit-repeatable); NVIDIA's EGL is not (frames differ by 1 LSB from process to process).
"""
import os
import numpy as np
import mujoco

XML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.xml")).read()
TARGETS = {"red": (8, 0, 2.0), "green": (-6, 5, 2.0), "blue": (0, -8, 2.0), "yellow": (-7, -6, 2.0)}
IMG = 224; HORIZON = 16; DT = 0.02
REACH_M = 0.8


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(XML); self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, IMG, IMG); self.mid = self.m.body("drone").mocapid[0]

    def set(self, pos, yaw):
        self.d.mocap_pos[self.mid] = pos; cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        self.d.mocap_quat[self.mid] = [cy, 0, 0, sy]; mujoco.mj_forward(self.m, self.d)

    def img(self):
        self.r.update_scene(self.d, camera="fpv"); return self.r.render().copy()

    def close(self):   # free the GL context while it is still current (avoids EGL noise at interpreter exit)
        try: self.r.close()
        except Exception: pass


def expert(pos, yaw, tgt):
    """the scripted teacher: body-frame [vx, vy, vz, yaw_rate] proportional guidance toward tgt."""
    d = tgt - pos; dxy = d[:2]; dist = np.linalg.norm(dxy)
    des_yaw = np.arctan2(dxy[1], dxy[0])
    yaw_err = (des_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    fwd = np.clip(dist, 0, 3.0) * max(0.0, np.cos(yaw_err))   # forward speed gated by how well we point at the target
    vx = np.clip(fwd, 0, 3.0); vy = 0.0
    vz = np.clip(d[2] * 1.5, -1.5, 1.5); yr = np.clip(2.0 * yaw_err, -2.0, 2.0)
    return np.array([vx, vy, vz, yr], np.float32)


def step_kin(pos, yaw, a):
    """one 20 ms tick: body-frame velocity -> world, integrated kinematically; altitude clipped to [0.6, 7.0] m."""
    c, s = np.cos(yaw), np.sin(yaw); vx, vy, vz, yr = a
    pos = pos + DT * np.array([c * vx - s * vy, s * vx + c * vy, vz])
    pos[2] = np.clip(pos[2], 0.6, 7.0); yaw = (yaw + DT * yr + np.pi) % (2 * np.pi) - np.pi
    return pos, yaw


def episode_start(e):
    """held-out evaluation episode e (seed 7000 + e): colour, target, start position and yaw."""
    colors = list(TARGETS); color = colors[e % 4]; tgt = np.array(TARGETS[color], float)
    rng = np.random.default_rng(7000 + e)
    pos = np.array([rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(1.5, 3.0)]); yaw = rng.uniform(-np.pi, np.pi)
    return color, tgt, pos, yaw, f"fly to the {color} pillar"


def renderer_name():
    from OpenGL import GL
    return GL.glGetString(GL.GL_RENDERER).decode()
