"""Minimal Cassie + Izhikevich CPG integration experiment.

Run this file from the project root:

    python cassie_cpg_main.py

Data flow
---------
IzhikevichCPG
    -> left/right neural motor primitives
CassieGaitDecoder
    -> 10 desired joint positions q_des [rad]
CassieController
    -> 10 MuJoCo motor controls
MujocoEnv
    -> rail-constrained Cassie physics

This is intentionally a small bring-up script, not the project's final main.py.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import mujoco
import numpy as np

from CNS.spinalcord.izhikevich_cpg import IzhikevichCPG
from PNS.motor.cassie_gait_decoder import CassieGaitDecoder
from PNS.motor.cassie_controller import CassieController
from PNS.motor.cassie_standing import CassieStanding
from PNS.mujoco_env import MujocoEnv


# ---------------------------------------------------------------------------
# First-test settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_XML = PROJECT_ROOT / "configs" / "agility_cassie" / "scene_stand.xml"

SIM_DURATION_S = 1.0
CPG_DT_S = 0.002          # 500 Hz neural / gait update
RENDER_HZ = 60.0
PRINT_EVERY_S = 0.05

# Start gently.  This multiplies the controller output before MuJoCo ctrlrange
# clipping.  Increase later only after looking at tracking and saturation.
CTRL_SCALE = 1.00

# Start with the gait decoder at half amplitude.  This lets us first verify the
# sign and timing of hip/knee/foot motion without asking for a huge stride.
GAIT_AMPLITUDE_SCALE = 1.00

# Keep the visual simulation near wall-clock speed so the motion is watchable.
REALTIME = True

Joint_diagnostics_timer = 1


def resolve_joint_qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    """Return qpos address for a named 1-DoF joint."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"MuJoCo joint not found: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def main() -> None:
    joint_dump_done = False
    # 1) Load the rail-constrained Cassie scene and reset to the supplied home
    #    posture.  The XML, not this Python file, owns the rail constraint.
    env = MujocoEnv(str(SCENE_XML))
    env.reset(keyframe="home")

    physics_dt = float(env.model.opt.timestep)
    physics_hz = 1.0 / physics_dt

    # The CPG runs more slowly than MuJoCo physics.  For the current Cassie XML:
    #   physics_dt = 0.0005 s -> 2000 Hz
    #   CPG_DT_S   = 0.0020 s ->  500 Hz
    # so one CPG output is held for four physics/PD updates.
    exact_ratio = CPG_DT_S / physics_dt
    physics_steps_per_cpg = int(round(exact_ratio))
    if physics_steps_per_cpg < 1 or not math.isclose(
        exact_ratio, physics_steps_per_cpg, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            "CPG_DT_S must be an integer multiple of MuJoCo timestep. "
            f"Got CPG_DT_S={CPG_DT_S}, physics_dt={physics_dt}."
        )

    # 2) Build the three layers.  Notice that each layer only knows the output
    #    of the layer immediately before it.
    standing = CassieStanding(env.model)
    cpg = IzhikevichCPG(sim_dt=CPG_DT_S)
    decoder = CassieGaitDecoder(
        amplitude_scale=GAIT_AMPLITUDE_SCALE,
        dt=CPG_DT_S,
    )
    controller = CassieController(
        env.model,
        ctrl_scale=CTRL_SCALE,
    )

    cpg.reset()
    decoder.reset()
    standing.reset(env.data)

    rail_qpos_adr = resolve_joint_qpos_address(env.model, "pelvis-rail-x")
    pitch_qpos_adr = resolve_joint_qpos_address(env.model, "pelvis-rail-pitch")
    height_qpos_adr = resolve_joint_qpos_address(env.model, "pelvis-rail-z")

    print("=== Cassie Izhikevich CPG rail test ===")
    print(f"scene              : {SCENE_XML}")
    print(f"MuJoCo physics     : {physics_hz:.0f} Hz (dt={physics_dt:.6f} s)")
    print(f"CPG / decoder      : {1.0 / CPG_DT_S:.0f} Hz (dt={CPG_DT_S:.6f} s)")
    print(f"physics steps / CPG: {physics_steps_per_cpg}")
    print(f"gait amplitude     : {GAIT_AMPLITUDE_SCALE:.2f}")
    print(f"controller scale   : {CTRL_SCALE:.2f}")
    print("Close the MuJoCo viewer to stop early.\n")

    # Launch the passive viewer once.  render() subsequently only syncs it.
    env.render()

    next_render_time = 0.0
    next_print_time = 0.0
    wall_start = time.perf_counter()
    sim_start = float(env.data.time)

    last_control = None
    last_cpg = None

    try:
        while float(env.data.time) - sim_start < SIM_DURATION_S:
            if env.viewer is not None and not env.viewer.is_running():
                break

            # ---------------------------------------------------------------
            # A. Neural rhythm + pattern formation at 500 Hz
            # ---------------------------------------------------------------
            #last_cpg = cpg.step(tonic_scale=1.0)
            #q_des = decoder.decode(last_cpg)

            
            # ---------------------------------------------------------------
            # B. Low-level PD + MuJoCo physics at 2000 Hz
            #    q_des stays constant during these four small physics steps.
            # ---------------------------------------------------------------
           
            for _ in range(physics_steps_per_cpg):
                stand_out = standing.step(env.data,dt=physics_dt)
                last_control = controller.apply(env.data, stand_out.as_dict())
                env.step(None)

            sim_elapsed = float(env.data.time) - sim_start

            # ---------------------------------------------------------------
            # C. Human-speed rendering.  Rendering does not need 2000 Hz.
            # ---------------------------------------------------------------
            if sim_elapsed >= next_render_time:
                env.render()
                next_render_time += 1.0 / RENDER_HZ

            # ---------------------------------------------------------------
            # D. Small diagnostics: is it moving, tracking, or saturating?
            # ---------------------------------------------------------------
            if sim_elapsed >= next_print_time and last_control is not None:
                x = float(env.data.qpos[rail_qpos_adr])
                pitch = float(env.data.qpos[pitch_qpos_adr])
                z = float(env.data.qpos[height_qpos_adr])
                raw_max_error = float(np.max(np.abs(last_control.position_error)))
                pd_max_error = min(raw_max_error, 0.60)
                print(
                    f"t={sim_elapsed:6.2f}s  "
                    f"x={x:+7.3f}m  "
                    f"pitch={math.degrees(pitch):+7.3f}deg  "
                    f"pitch_rate={math.degrees(stand_out.pitch_rate):+.1f}deg/s  "
                    f"z={z:+7.3f}m  "
                    #f"CPG L/R={last_cpg.left:+.3f}/{last_cpg.right:+.3f}  "
                    f"qerr={math.degrees(raw_max_error):+.3f}deg  "
                    f"PDerr<={math.degrees(pd_max_error):+.3f}deg  "
                    f"sat={last_control.saturation_count}/10  "
                    f"zdot={stand_out.z_velocity:+.3f}  "
                    f"dz={stand_out.knee_height_delta:+.3f}  "
                    f"height_integral={standing._height_integral:+.3f}  "
                    f"dhip={math.degrees(stand_out.hip_pitch_delta):+.1f}deg  "
                    f"dfoot={math.degrees(stand_out.foot_pitch_delta):+.1f}deg  "
                )
                next_print_time += PRINT_EVERY_S

            if sim_elapsed >= Joint_diagnostics_timer and not joint_dump_done:
                print(f"\n=== Joint diagnostics at ~{Joint_diagnostics_timer} s ===")

                for name, q, qdes, err, tau, ctrl in zip(
                    controller.ACTUATOR_ORDER,
                    last_control.q,
                    last_control.q_des,
                    last_control.position_error,
                    last_control.joint_torque,
                    last_control.ctrl,
                ):
                    print(
                        f"{name:18s} "
                        f"q={q:+.3f}  "
                        f"qdes={qdes:+.3f}  "
                        f"err={err:+.3f}  "
                        f"tau={tau:+7.2f} Nm  "
                        f"ctrl={ctrl:+7.3f}"
                    )

                joint_dump_done = True

            # Pace in chunks rather than sleeping every 0.5 ms physics step.
            if REALTIME:
                target_wall = wall_start + sim_elapsed
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    finally:
        # Zero motor commands before closing the viewer.
        env.data.ctrl[:] = 0.0
        env.close()


    final_x = float(env.data.qpos[rail_qpos_adr])
    final_t = float(env.data.time) - sim_start
    print("\n=== Test finished ===")
    print(f"simulated time: {final_t:.3f} s")
    print(f"rail displacement x: {final_x:+.4f} m")


if __name__ == "__main__":
    main()