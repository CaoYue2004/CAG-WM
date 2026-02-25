import time
from .truncation import Truncation

class MaxStepTime(Truncation):
    def __init__(self, max_step_time: float) -> None:
        self.max_step_time = max_step_time
        self._step_start_time = None
        self._step_time = 0.0

    @property
    def truncated(self) -> bool:
        return self._step_time > self.max_step_time

    def step(self) -> None:
        """
        Called AFTER env.step() finished.
        Computes elapsed wall-clock time.
        """
        if self._step_start_time is None:
            return
        self._step_time = time.perf_counter() - self._step_start_time

    def reset(self, episode_nr: int = 0) -> None:
        self._step_start_time = None
        self._step_time = 0.0

    def before_step(self) -> None:
        """
        Call this RIGHT BEFORE env.step().
        """
        self._step_start_time = time.perf_counter()

