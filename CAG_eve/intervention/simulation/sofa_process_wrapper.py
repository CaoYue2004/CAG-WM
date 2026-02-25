# sofa_process_wrapper.py
import multiprocessing as mp
import numpy as np
import time

from .sofa_worker import sofa_worker_main
from . import SofaBeamAdapter  

class SofaBeamAdapterProcess:

    def __init__(self, friction=0.1, dt_simulation=0.002, step_timeout_s=10.0):
        self.env_kwargs = dict(friction=friction, dt_simulation=dt_simulation)
        self.step_timeout_s = float(step_timeout_s)

        self._proc = None
        self._conn = None  # parent conn

        self.simulation_error = False
        self._dof_positions = None
        self._inserted_lengths = None
        self._rotations = None

        self._start_worker()

    @property
    def dof_positions(self):
        return self._dof_positions

    @property
    def inserted_lengths(self):
        return self._inserted_lengths

    @property
    def rotations(self):
        return self._rotations

    def _start_worker(self):
        ctx = mp.get_context("spawn")

        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sofa_worker_main,
            args=(child_conn, SofaBeamAdapter, self.env_kwargs),
            daemon=True
        )
        self._proc.start()

    def _kill_worker(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass

        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2.0)

        self._proc = None
        self._conn = None

    def _restart_worker(self):
        self._kill_worker()
        self._start_worker()

    def _rpc(self, cmd, timeout=None, **payload):
        if self._conn is None or (self._proc is None) or (not self._proc.is_alive()):
            self.simulation_error = True
            self._restart_worker()
            raise RuntimeError("SOFA worker not available (restarted).")

        self._conn.send({"cmd": cmd, **payload})
        if not self._conn.poll(timeout=timeout or self.step_timeout_s):
            self.simulation_error = True
            self._restart_worker()
            raise TimeoutError(f"SOFA worker {cmd} timeout (killed & restarted).")

        reply = self._conn.recv()
        if not reply.get("ok", False):
            self.simulation_error = True
            self._restart_worker()
            raise RuntimeError(f"SOFA worker {cmd} failed: {reply.get('error')}\n{reply.get('traceback', '')}")

        state = reply.get("state")
        if state is not None:
            self._update_from_state(state)
        return reply.get("result")

    def close(self):
        if self._conn is None:
            return
        try:
            self._conn.send({"cmd": "close"})
            if self._conn.poll(timeout=2.0):
                _ = self._conn.recv()
        except Exception:
            pass
        finally:
            self._kill_worker()

    def reset(self, **reset_kwargs):
        """
        Reset the SOFA simulation in the worker process.

        reset_kwargs should match SofaBeamAdapter.reset(...) signature:
          insertion_point, insertion_direction, mesh_path, devices,
          coords_high/low, vessel_visual_path, seed, ...

        Training mode typically passes coords_* / vessel_visual_path as None.
        """

        # 1) devices: List[Device]  -> devices_cfg: List[dict] (pickle-safe)
        if "devices" in reset_kwargs:
            devices = reset_kwargs.pop("devices")
            reset_kwargs["devices_cfg"] = [
                d.get_config_dict(eve_classes_only=False) for d in devices
            ]

        # 2) Delegate to RPC layer.
        # Worker expects payload under "args".
        self._rpc("reset", args=reset_kwargs, timeout=self.step_timeout_s)
        self.simulation_error = False

        # _rpc() already updated local cached state via _update_from_state()
        # Nothing else needed here.

    def step(self, action: np.ndarray, duration: float):
        """
        Step the SOFA simulation in the worker process.

        action: np.ndarray, shape (n_devices, 2) typically [insert_vel, rot_vel]
        duration: float (seconds)
        """

        payload = {
            "action": np.asarray(action, dtype=np.float32),
            "duration": float(duration),
        }
        try:
            self._rpc("step", timeout=self.step_timeout_s, **payload)
        except TimeoutError:
            self.simulation_error = True
            return

    def reset_devices(self):
        self._rpc("reset_devices")

    def add_interim_targets(self, positions):
        return []

    def remove_interim_target(self, interim_target):
        return None

    def _update_from_state(self, state: dict):
        self._dof_positions = state["dof_positions"]
        self._inserted_lengths = state["inserted_lengths"]
        self._rotations = state["rotations"]
        self.simulation_error = bool(state["simulation_error"])

