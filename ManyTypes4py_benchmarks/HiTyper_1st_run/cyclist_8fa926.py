from .teampursuit import teampursuit
import math

class Cyclist:
    max_power = 1200
    min_power = 200
    drag_coeffiecient = 0.65
    mechanical_efficiency = 0.977
    bike_mass = 7.7
    fatigue_level = 0

    def __init__(self, height: Union[int, list[int], float], weight: Union[int, dict[int, tuple[int]], float], mean_maximum_power: Union[int, float], event: Union[list[float], list['ValidatorRecord'], list[int]], start_position: Union[int, float], gender: Union[int, str]) -> None:
        self.height = height
        self.weight = weight
        self.mean_maximum_power = mean_maximum_power
        self.event = event
        self.start_position = start_position
        self.position = start_position
        self.gender = gender
        self.current_velocity = 0.0
        self.fatigue_level = 0
        self.update_cda()
        self.update_total_energy()
        self.remaining_energy = self.total_energy

    def set_pace(self, power: Union[float, int]) -> float:
        fatigue_factor = 1 - 0.01 * self.fatigue_level
        delta_ke = (power * self.mechanical_efficiency * fatigue_factor - self.coefficient_drag_area * 0.5 * self.event.air_density * math.pow(self.current_velocity, 3) - teampursuit.friction_coefficient * (self.weight + self.bike_mass) * teampursuit.gravitational_acceleration * self.current_velocity) * teampursuit.time_step
        new_velocity = math.pow(2 * delta_ke / (self.weight + self.bike_mass) + math.pow(self.current_velocity, 2), 0.5)
        acceleration = new_velocity - self.current_velocity
        distance = self.current_velocity * teampursuit.time_step + 0.5 * acceleration * math.pow(teampursuit.time_step, 2)
        self.current_velocity = new_velocity
        if self.remaining_energy > power * teampursuit.time_step:
            self.remaining_energy -= power * teampursuit.time_step
        else:
            self.remaining_energy = 0.0
        return distance

    def follow(self, distance: Union[float, int]) -> None:
        fatigue_factor = 1 - 0.01 * self.fatigue_level
        acceleration = 2 * (distance - self.current_velocity * teampursuit.time_step) / math.pow(teampursuit.time_step, 2)
        new_velocity = self.current_velocity + acceleration * teampursuit.time_step
        delta_ke = 0.5 * (self.weight + self.bike_mass) * (new_velocity - self.current_velocity)
        power = (self.coefficient_drag_area * teampursuit.drafting_coefficients[self.position - 2] * 0.5 * self.event.air_density * math.pow(self.current_velocity, 3) + teampursuit.friction_coefficient * (self.weight + self.bike_mass) * teampursuit.gravitational_acceleration * self.current_velocity + delta_ke / teampursuit.time_step) / (self.mechanical_efficiency * fatigue_factor)
        self.current_velocity = new_velocity
        if self.remaining_energy > power * teampursuit.time_step:
            self.remaining_energy -= power * teampursuit.time_step
        else:
            self.remaining_energy = 0.0

    def get_height(self):
        return self.height

    def get_weight(self):
        return self.weight

    def get_mean_maximum_power(self):
        return self.mean_maximum_power

    def get_remaining_energy(self):
        return self.remaining_energy

    def get_position(self):
        return self.position

    def set_weight(self, weight: Union[float, int]) -> None:
        self.weight = weight
        self.update_cda()
        self.update_total_energy()

    def set_height(self, height: Union[int, float]) -> None:
        self.height = height
        self.update_cda()

    def set_mean_maximum_power(self, mean_maximum_power: Union[float, tuple[float]]) -> None:
        self.mean_maximum_power = mean_maximum_power
        self.update_total_energy()

    def set_position(self, position: Union[int, str]) -> None:
        self.position = position

    def increase_fatigue(self) -> None:
        self.fatigue_level += 2

    def recover(self) -> None:
        if self.fatigue_level > 0:
            self.fatigue_level -= 1

    def reset(self) -> None:
        self.remaining_energy = self.total_energy
        self.position = self.start_position
        self.fatigue_level = 0
        self.current_velocity = 0

    def update_cda(self) -> None:
        self.coefficient_drag_area = self.drag_coeffiecient * (0.0293 * math.pow(self.height, 0.725) * math.pow(self.weight, 0.425) + 0.0604)

    def update_total_energy(self) -> None:
        coeff = 240 if self.gender == 'male' else 210
        self.total_energy = self.mean_maximum_power * self.weight * coeff