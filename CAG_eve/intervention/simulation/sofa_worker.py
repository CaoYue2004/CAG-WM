# sofa_worker.py
import traceback
import numpy as np


def pack_state(env):
    """
    Extract a minimal, pickle-safe state snapshot from SofaBeamAdapter.
    This MUST NOT contain any SOFA objects or C++ pointers.
    """
    return {
        "dof_positions": (
            None if env.dof_positions is None
            else np.asarray(env.dof_positions, dtype=np.float32)
        ),
        "inserted_lengths": (
            None if env.inserted_lengths is None
            else np.asarray(env.inserted_lengths, dtype=np.float32)
        ),
        "rotations": (
            None if env.rotations is None
            else np.asarray(env.rotations, dtype=np.float32)
        ),
        "simulation_error": bool(env.simulation_error),
    }

def sofa_worker_main(conn, env_ctor, env_kwargs):
    try:
        env = env_ctor(**env_kwargs)

        env.init_visual_nodes = False

        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")

            if cmd == "close":
                try:
                    env.close()
                finally:
                    conn.send({"ok": True})
                break

            elif cmd == "reset":
                args = msg["args"]
                devices_cfg = args.pop("devices_cfg")
                from ..device import Device
                devices = [Device.from_config_dict(cfg) for cfg in devices_cfg]
                args["devices"] = devices
                env.reset(**args)

                conn.send({
                    "ok": True,
                    "state": {
                        "dof_positions": env.dof_positions,
                        "inserted_lengths": np.asarray(env.inserted_lengths),
                        "rotations": np.asarray(env.rotations),
                        "simulation_error": bool(env.simulation_error),
                    }
                })

            elif cmd == "step":
                action = msg["action"]
                duration = float(msg["duration"])
                env.step(action, duration)

                conn.send({
                    "ok": True,
                    "state": {
                        "dof_positions": env.dof_positions,
                        "inserted_lengths": np.asarray(env.inserted_lengths),
                        "rotations": np.asarray(env.rotations),
                        "simulation_error": bool(env.simulation_error),
                    }
                })

            elif cmd == "reset_devices":
                env.reset_devices()
                conn.send({"ok": True, "state": pack_state(env)})

            elif cmd == "add_interim_targets":
                res = env.add_interim_targets(msg["positions"])
                conn.send({"ok": True, "result": None, "state": pack_state(env)})


            else:
                conn.send({"ok": False, "error": f"Unknown cmd: {cmd}"})

    except Exception as e:
        tb = traceback.format_exc()
        try:
            conn.send({"ok": False, "error": str(e), "traceback": tb})
        except Exception:
            pass

