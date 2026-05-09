import math

class SimulationResult:
    def __init__(self, finish_time: float, proportion_completed: float, energy_remaining: list[float], velocity_profile: list[float]) -> None:
        self.finish_time = finish_time
        self.energy_remaining = energy_remaining
        self.proportion_completed = proportion_completed
        self.velocity_profile = velocity_profile
        self.results: list[dict] = []

    def get_finish_time(self) -> float:
        return self.finish_time

    def get_proportion_completed(self) -> float:
        return self.proportion_completed

    def get_energy_remaining(self) -> list[float]:
        return self.energy_remaining

    def get_velocity_profile(self) -> list[float]:
        return self.velocity_profile

    def to_string(self) -> str:
        output: str = 'Simulation Result\n-----------------\n'
        if self.finish_time < math.inf:
            output = f'{output} Finish Time: {self.finish_time} seconds\n'
            for i in range(0, len(self.energy_remaining)):
                output = f'{output} Cyclist {i + 1} Energy Remaining: {self.energy_remaining[i]} joules\n'
        else:
            output = f'{output} Riders had insufficient energy for race completion\nProportion of race completed: {self.proportion_completed * 100}%\n'
        return output
