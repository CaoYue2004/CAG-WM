from . import Terminal
from ..intervention import Intervention

class ComputeFailed(Terminal):
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention

    @property
    def terminal(self) -> bool:
        return self.intervention.simulation.simulation_error        

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...


