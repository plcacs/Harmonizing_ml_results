import math

class simulationresult:

    def __init__(self, finish_time: Union[int, float], proportion_completed: Union[bool, float], energy_remaining: Union[int, float, None], velocity_profile: Union[str, int, None]) -> None:
        self.finish_time = finish_time
        self.energy_remaining = energy_remaining
        self.proportion_completed = proportion_completed
        self.velocity_profile = velocity_profile
        self.results = []

    def get_finish_time(self):
        return self.finish_time

    def get_proportion_completed(self):
        return self.proportion_completed

    def get_energy_remaining(self):
        return self.energy_remaining

    def get_velocity_profile(self):
        return self.velocity_profile

    def to_string(self) -> typing.Text:
        output = 'Simulation Result\n-----------------\n'
        if self.finish_time < math.inf:
            output = f'{output} Finish Time: {self.finish_time} seconds\n'
            for i in range(0, len(self.energy_remaining)):
                output = f'{output} Cyclist {i + 1} Energy Remaining: {self.energy_remaining[i]} joules\n'
        else:
            output = f'{output} Riders had insufficient energy for race completion\nProportion of race completed: {self.proportion_completed * 100}%\n'
        return output