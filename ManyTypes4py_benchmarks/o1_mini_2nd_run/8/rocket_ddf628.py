"""
Approximate Rocket Simulation
Based on
https://raw.githubusercontent.com/purdue-orbital/rocket-simulation/master/Simulation2.py
"""
import math
import pyproj
import numpy as np
from nevergrad.parametrization import parameter
from typing import Tuple, List
from ..base import ArrayExperimentFunction

class Rocket(ArrayExperimentFunction):

    def __init__(self, symmetry: int = 0) -> None:
        super().__init__(rocket, parametrization=parameter.Array(shape=(24,)), symmetry=symmetry)

def rocket(thrust_bias: np.ndarray) -> float:
    assert len(thrust_bias) == 24, 'Bad guide length.'

    def rad(ang: float) -> float:
        return ang / 360 * 2 * math.pi

    def air_density(altitude: float) -> float:
        if altitude <= 11000:
            T: float = 15.04 - 0.00649 * altitude
            p: float = 101.29 * math.pow((T + 273.1) / 288.08, 5.256)
        elif altitude <= 25000:
            T = -56.46
            p = 22.65 * math.exp(1.73 - 0.000157 * altitude)
        else:
            T = -131.21 + 0.00299 * altitude
            p = 2.488 * ((T + 273.1) / 216.6) ** (-11.388)
        d: float = p / (0.2869 * (T + 273.1))
        return d

    def alt(Ex: float, Ey: float, Ez: float) -> float:
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        transformer = pyproj.Transformer.from_proj(ecef, lla)
        _, _, altitude = transformer.transform(Ex, Ey, Ez, radians=True)
        return altitude

    def grav_force(Ex: float, Ey: float, Ez: float, m: float) -> Tuple[float, float, float]:
        G: float = -6.67408e-11
        M: float = 5.972e24
        r: float = math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
        F: float = G * M * m / r ** 2
        F_z: float = F * Ez / r
        F_x: float = F * Ex / math.sqrt(Ex ** 2 + Ey ** 2)
        F_y: float = F * Ey / math.sqrt(Ex ** 2 + Ey ** 2)
        return (F_x, F_y, F_z)

    def drag_force(Ex: float, Ey: float, Ez: float, Evx: float, Evy: float, Evz: float) -> Tuple[float, float, float]:
        cd: float = 0.94
        a: float = 0.00487
        p: float = air_density(alt(Ex, Ey, Ez))
        v_sqrd: float = Evx ** 2 + Evy ** 2 + Evz ** 2
        if Evx == 0:
            Ex_drag = 0.0
        else:
            Ex_drag = 0.5 * p * v_sqrd * cd * a * (-Evx / math.sqrt(v_sqrd))
        if Evy == 0:
            Ey_drag = 0.0
        else:
            Ey_drag = 0.5 * p * v_sqrd * cd * a * (-Evy / math.sqrt(Evx ** 2 + Evy ** 2))
        if Evz == 0:
            Ez_drag = 0.0
        else:
            Ez_drag = 0.5 * p * v_sqrd * cd * a * (-Evz / math.sqrt(Evx ** 2 + Evy ** 2))
        return (Ex_drag, Ey_drag, Ez_drag)

    def net_force(Ex: float, Ey: float, Ez: float, Evx: float, Evy: float, Evz: float, m: float) -> Tuple[float, float, float]:
        Fx_drag, Fy_drag, Fz_drag = drag_force(Ex, Ey, Ez, Evx, Evy, Evz)
        Fx_grav, Fy_grav, Fz_grav = grav_force(Ex, Ey, Ez, m)
        Fx: float = Fx_drag + Fx_grav
        Fy: float = Fy_drag + Fy_grav
        Fz: float = Fz_drag + Fz_grav
        return (Fx, Fy, Fz)

    altitude_init: float = 0.0
    latitude: float = rad(28.5729)
    longitude: float = rad(80.659)
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(lla, ecef)
    Ex, Ey, Ez = transformer.transform(longitude, latitude, altitude_init, radians=True)
    Evx: float = 0.0
    Evy: float = 0.0
    Evz: float = 0.0
    r_initial: float = math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
    roc_mass: float = 0.0472
    theta: float = 45.0
    phi: float = 45.0
    eng_mass_initial: float = 0.024
    eng_mass_final: float = 0.0132
    total_mass: float = roc_mass + eng_mass_initial
    final_roc_mass: float = roc_mass + eng_mass_final
    thrust: np.ndarray = np.array([
        [0.0, 0.0],
        [0.946, 0.031],
        [4.826, 0.092],
        [9.936, 0.139],
        [14.09, 0.192],
        [11.446, 0.209],
        [7.381, 0.231],
        [6.151, 0.248],
        [5.489, 0.292],
        [4.921, 0.37],
        [4.448, 0.475],
        [4.258, 0.671],
        [4.542, 0.702],
        [4.164, 0.723],
        [4.448, 0.85],
        [4.353, 1.063],
        [4.353, 1.211],
        [4.069, 1.242],
        [4.258, 1.303],
        [4.353, 1.468],
        [4.448, 1.656],
        [4.448, 1.821],
        [2.933, 1.834],
        [1.325, 1.847],
        [0.0, 1.86]
    ], dtype=np.float64)
    thrust_list: np.ndarray = np.array([thrust[int(i)][0] for i in range(len(thrust) - 1)], dtype=np.float64)
    thrust_time_list: np.ndarray = np.array([thrust[i + 1][1] - thrust[i][1] for i in range(len(thrust) - 1)], dtype=np.float64)
    total_thrust: float = float(np.sum(thrust_list * thrust_time_list))
    thrust_list = thrust_list * np.exp(thrust_bias)
    thrust_list = thrust_list * total_thrust / float(np.sum(thrust_list * thrust_time_list))
    for i in range(len(thrust) - 1):
        thrust[i][0] = thrust_list[i]
    mass_time: List[List[float]] = []
    mass_loss: float = eng_mass_initial - eng_mass_final
    mass_reman: float = eng_mass_initial
    for row in thrust:
        percentage: float = row[0] / float(np.sum(thrust_list))
        assert percentage >= -1e-05, f'percentage is {percentage}'
        assert percentage <= 1.0
        mass_loss = mass_reman * percentage
        mass_reman -= mass_loss
        total_mass: float = roc_mass + mass_reman
        mass_time.append([total_mass, row[1]])
    mass_list: List[float] = [mass_time[i][0] for i in range(len(thrust))]
    Ex_list: np.ndarray = np.array([Ex], dtype=np.float64)
    Ey_list: np.ndarray = np.array([Ey], dtype=np.float64)
    Ez_list: np.ndarray = np.array([Ez], dtype=np.float64)
    Evx_list: np.ndarray = np.array([Evx], dtype=np.float64)
    Evy_list: np.ndarray = np.array([Evy], dtype=np.float64)
    Evz_list: np.ndarray = np.array([Evz], dtype=np.float64)
    time_list: np.ndarray = np.array([0.0], dtype=np.float64)
    r_list: np.ndarray = np.array([r_initial], dtype=np.float64)
    time: float = 0.0
    for i in range(len(thrust) - 2):
        r = math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
        dt: float = thrust[i][1]
        Efx, Efy, Efz = net_force(Ex, Ey, Ez, Evx, Evy, Evz, mass_list[i])
        Ex += Evx * dt
        Ey += Evy * dt
        Ez += Evz * dt
        dt = thrust[i + 1][1] - thrust[i][1]
        Evz += (thrust[i][0] * math.cos(theta) + Efz) * dt / mass_list[i]
        Evx += (thrust[i][0] * math.sin(theta) * math.cos(phi) + Efx) * dt / mass_list[i]
        Evy += (thrust[i][0] * math.sin(theta) * math.sin(phi) + Efy) * dt / mass_list[i]
        time += dt
        Ex_list = np.append(Ex_list, round(Ex, 6))
        Ey_list = np.append(Ey_list, round(Ey, 6))
        Ez_list = np.append(Ez_list, round(Ez, 6))
        Evx_list = np.append(Evx_list, round(Evx, 6))
        Evy_list = np.append(Evy_list, round(Evy, 6))
        Evz_list = np.append(Evz_list, round(Evz, 6))
        time_list = np.append(time_list, round(time, 6))
        r_list = np.append(r_list, round(r, 6))
    time_step: float = 0.05
    dt = time_step
    while r > r_initial:
        r = math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
        Efx, Efy, Efz = net_force(Ex, Ey, Ez, Evx, Evy, Evz, final_roc_mass)
        Ex += Evx * dt
        Ey += Evy * dt
        Ez += Evz * dt
        Evx += Efx * dt / final_roc_mass
        Evy += Efy * dt / final_roc_mass
        Evz += Efz * dt / final_roc_mass
        time += dt
        Ex_list = np.append(Ex_list, round(Ex, 6))
        Ey_list = np.append(Ey_list, round(Ey, 6))
        Ez_list = np.append(Ez_list, round(Ez, 6))
        Evx_list = np.append(Evx_list, round(Evx, 6))
        Evy_list = np.append(Evy_list, round(Evy, 6))
        Evz_list = np.append(Evz_list, round(Evz, 6))
        time_list = np.append(time_list, round(time, 6))
        r_list = np.append(r_list, round(r, 6))
    return 1.0 - max(Ez_list) / 3032708.353202
