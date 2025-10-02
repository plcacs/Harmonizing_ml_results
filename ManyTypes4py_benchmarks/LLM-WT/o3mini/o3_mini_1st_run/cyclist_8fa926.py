from .teampursuit import teampursuit
import math
from typing import Any

class Cyclist:
    max_power: int = 1200
    min_power: int = 200
    drag_coeffiecient: float = 0.65
    mechanical_efficiency: float = 0.977
    bike_mass: float = 7.7
    fatigue_level: int = 0

    def __init__(self, height: float, weight: float, mean_maximum_power: float, event: Any, start_position: int, gender: str) -> None:
        self.height: float = height
        self.weight: float = weight
        self.mean_maximum_power: float = mean_maximum_power
        self.event: Any = event
        self.start_position: int = start_position
        self.position: int = start_position
        self.gender: str = gender
        self.current_velocity: float = 0.0
        self.fatigue_level: int = 0
        self.update_cda()
        self.update_total_energy()
        self.remaining_energy: float = self.total_energy

    def set_pace(self, power: float) -> float:
        fatigue_factor: float = 1 - 0.01 * self.fatigue_level
        delta_ke: float = (power * self.mechanical_efficiency * fatigue_factor - 
                             self.coefficient_drag_area * 0.5 * self.event.air_density * math.pow(self.current_velocity, 3) - 
                             teampursuit.friction_coefficient * (self.weight + self.bike_mass) * teampursuit.gravitational_acceleration * self.current_velocity) * teampursuit.time_step
        new_velocity: float = math.sqrt(2 * delta_ke / (self.weight + self.bike_mass) + math.pow(self.current_velocity, 2))
        acceleration: float = new_velocity - self.current_velocity
        distance: float = self.current_velocity * teampursuit.time_step + 0.5 * acceleration * math.pow(teampursuit.time_step, 2)
        self.current_velocity = new_velocity
        if self.remaining_energy > power * teampursuit.time_step:
            self.remaining_energy -= power * teampursuit.time_step
        else:
            self.remaining_energy = 0.0
        return distance

    def follow(self, distance: float) -> None:
        fatigue_factor: float = 1 - 0.01 * self.fatigue_level
        acceleration: float = 2 * (distance - self.current_velocity * teampursuit.time_step) / math.pow(teampursuit.time_step, 2)
        new_velocity: float = self.current_velocity + acceleration * teampursuit.time_step
        delta_ke: float = 0.5 * (self.weight + self.bike_mass) * (new_velocity - self.current_velocity)
        power: float = (self.coefficient_drag_area * teampursuit.drafting_coefficients[self.position - 2] * 0.5 * self.event.air_density * math.pow(self.current_velocity, 3) + 
                        teampursuit.friction_coefficient * (self.weight + self.bike_mass) * teampursuit.gravitational_acceleration * self.current_velocity + 
                        delta_ke / teampursuit.time_step) / (self.mechanical_efficiency * fatigue_factor)
        self.current_velocity = new_velocity
        if self.remaining_energy > power * teampursuit.time_step:
            self.remaining_energy -= power * teampursuit.time_step
        else:
            self.remaining_energy = 0.0

    def get_height(self) -> float:
        return self.height

    def get_weight(self) -> float:
        return self.weight

    def get_mean_maximum_power(self) -> float:
        return self.mean_maximum_power

    def get_remaining_energy(self) -> float:
        return self.remaining_energy

    def get_position(self) -> int:
        return self.position

    def set_weight(self, weight: float) -> None:
        self.weight = weight
        self.update_cda()
        self.update_total_energy()

    def set_height(self, height: float) -> None:
        self.height = height
        self.update_cda()

    def set_mean_maximum_power(self, mean_maximum_power: float) -> None:
        self.mean_maximum_power = mean_maximum_power
        self.update_total_energy()

    def set_position(self, position: int) -> None:
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
        self.coefficient_drag_area: float = self.drag_coeffiecient * (0.0293 * math.pow(self.height, 0.725) * math.pow(self.weight, 0.425) + 0.0604)

    def update_total_energy(self) -> None:
        coeff: int = 240 if self.gender == 'male' else 210
        self.total_energy: float = self.mean_maximum_power * self.weight * coeff