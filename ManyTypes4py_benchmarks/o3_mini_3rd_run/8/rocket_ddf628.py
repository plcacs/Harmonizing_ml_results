import math
import pyproj
import numpy as np
from nevergrad.parametrization import parameter
from typing import Tuple
from ..base import ArrayExperimentFunction

class Rocket(ArrayExperimentFunction):
    def __init__(self, symmetry: int = 0) -> None:
        super().__init__(rocket, parametrization=parameter.Array(shape=(24,)), symmetry=symmetry)

def rocket(thrust_bias: np.ndarray) -> float:
    assert len(thrust_bias) == 24, 'Bad guide length.'

    def rad(ang: float) -> float:
        return ang / 360 * 2 * 3.1415926

    def air_density(alt: float) -> float:
        if alt <= 11000:
            T: float = 15.04 - 0.00649 * alt
            p: float = 101.29 * math.pow((T + 273.1) / 288.08, 5.256)
        elif alt <= 25000:
            T = -56.46
            p = float(22.65 * math.exp(1.73 - 0.000157 * alt)).real
        else:
            T = -131.21 + 0.00299 * alt
            p = 2.488 * (((T + 273.1) / 216.6) ** (-11.388)).real
        d: float = p / (0.2869 * (T + 273.1))
        return d

    def alt(Ex: float, Ey: float, Ez: float) -> float:
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        transformer = pyproj.Transformer.from_proj(ecef, lla)
        _, _, altitude_local = transformer.transform(Ex, Ey, Ez, radians=True)
        return altitude_local

    def grav_force(Ex: float, Ey: float, Ez: float, m: float) -> Tuple[float, float, float]:
        G: float = -6.67408 * (1 / 10 ** 11)
        M: float = 5.972 * 10 ** 24
        r: float = math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
        F: float = G * M * m / (r ** 2)
        F_z: float = F * Ez / r
        # To avoid division by zero in calculating F_x and F_y, we ensure that the denominator is non-zero.
        horizontal_norm: float = math.sqrt(Ex ** 2 + Ey ** 2)
        if horizontal_norm == 0:
            F_x: float = 0.0
            F_y: float = 0.0
        else:
            F_x = F * (Ex / horizontal_norm)
            F_y = F * (Ey / horizontal_norm)
        return (F_x, F_y, F_z)

    def drag_force(Ex: float, Ey: float, Ez: float, Evx: float, Evy: float, Evz: float) -> Tuple[float, float, float]:
        cd: float = 0.94
        a: float = 0.00487
        p: float = air_density(alt(Ex, Ey, Ez))
        v_sqrd: float = Evx ** 2 + Evy ** 2 + Evz ** 2
        Ex_drag: float = 0.0
        Ey_drag: float = 0.0
        Ez_drag: float = 0.0
        if v_sqrd != 0:
            v_total: float = math.sqrt(v_sqrd)
            Ex_drag = 0.5 * p * v_sqrd * cd * a * (-Evx / v_total)
            # For Ey_drag and Ez_drag, using the norm in the horizontal plane if applicable
            horizontal_norm: float = math.sqrt(Evx ** 2 + Evy ** 2)
            if horizontal_norm != 0:
                Ey_drag = 0.5 * p * v_sqrd * cd * a * (-Evy / horizontal_norm)
            if horizontal_norm != 0:
                Ez_drag = 0.5 * p * v_sqrd * cd * a * (-Evz / horizontal_norm)
        return (Ex_drag, Ey_drag, Ez_drag)

    def net_force(Ex: float, Ey: float, Ez: float, Evx: float, Evy: float, Evz: float, m: float) -> Tuple[float, float, float]:
        Fx_drag, Fy_drag, Fz_drag = drag_force(Ex, Ey, Ez, Evx, Evy, Evz)
        Fx_grav, Fy_grav, Fz_grav = grav_force(Ex, Ey, Ez, m)
        Fx: float = Fx_drag + Fx_grav
        Fy: float = Fy_drag + Fy_grav
        Fz: float = Fz_drag + Fz_grav
        return (Fx, Fy, Fz)

    altitude: float = 0.0
    latitude: float = rad(28.5729)
    longitude: float = rad(80.659)
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(lla, ecef)
    Ex, Ey, Ez = transformer.transform(longitude, latitude, altitude, radians=True)
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
    thrust = np.asarray([
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
    ])
    thrust_list: np.ndarray = np.asarray([thrust[int(i)][0] for i in range(len(thrust) - 1)])
    thrust_time_list: np.ndarray = np.asarray([thrust[i + 1][1] - thrust[i][1] for i in range(0, len(thrust) - 1)])
    total_thrust: float = np.sum(np.multiply(thrust_list, thrust_time_list))
    thrust_list = np.multiply(thrust_list, np.exp(thrust_bias))
    thrust_list = thrust_list * total_thrust / np.sum(np.multiply(thrust_list, thrust_time_list))
    for i in range(len(thrust) - 1):
        thrust[i][0] = thrust_list[i]
    mass_time = []
    mass_loss: float = eng_mass_initial - eng_mass_final
    mass_reman: float = eng_mass_initial
    for row in thrust:
        percentage: float = row[0] / np.sum(thrust_list)
        assert percentage >= -1e-05, f'percentage is {percentage}'
        assert percentage <= 1.0
        mass_loss = mass_reman * percentage
        mass_reman -= mass_loss
        total_mass = roc_mass + mass_reman
        mass_time.append([total_mass, row[1]])
    mass_list = [mass_time[i][0] for i in range(0, len(thrust))]
    Ex_list: np.ndarray = np.asarray([Ex])
    Ey_list: np.ndarray = np.asarray([Ey])
    Ez_list: np.ndarray = np.asarray([Ez])
    Evx_list: np.ndarray = np.asarray([Evx])
    Evy_list: np.ndarray = np.asarray([Evy])
    Evz_list: np.ndarray = np.asarray([Evz])
    time_list: np.ndarray = np.asarray([0.0])
    r_list: np.ndarray = np.asarray([math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)])
    time: float = 0.0
    for i in range(len(thrust) - 2):
        r: float = math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
        dt: float = thrust[i][1]
        Efx, Efy, Efz = net_force(Ex, Ey, Ez, Evx, Evy, Evz, mass_list[i])
        Ex += Evx * dt
        Ey += Evy * dt
        Ez += Evz * dt
        dt = thrust[i + 1][1] - thrust[i][1]
        Evz += (thrust[i][0] * math.cos(math.radians(theta)) + Efz) * dt / mass_list[i]
        Evx += (thrust[i][0] * math.sin(math.radians(theta)) * math.cos(math.radians(phi)) + Efx) * dt / mass_list[i]
        Evy += (thrust[i][0] * math.sin(math.radians(theta)) * math.sin(math.radians(phi)) + Efy) * dt / mass_list[i]
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
    dt: float = time_step
    r: float = math.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
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