"""Test Home Assistant unit conversion utility functions."""

from __future__ import annotations

import inspect
from itertools import chain
from typing import Dict, List, Tuple, Type, Union

import pytest

from homeassistant.const import (
    CONCENTRATION_PARTS_PER_BILLION,
    CONCENTRATION_PARTS_PER_MILLION,
    PERCENTAGE,
    UnitOfArea,
    UnitOfBloodGlucoseConcentration,
    UnitOfConductivity,
    UnitOfDataRate,
    UnitOfElectricCurrent,
    UnitOfElectricPotential,
    UnitOfEnergy,
    UnitOfEnergyDistance,
    UnitOfInformation,
    UnitOfLength,
    UnitOfMass,
    UnitOfPower,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
    UnitOfTime,
    UnitOfVolume,
    UnitOfVolumeFlowRate,
    UnitOfVolumetricFlux,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import unit_conversion
from homeassistant.util.unit_conversion import (
    AreaConverter,
    BaseUnitConverter,
    BloodGlucoseConcentrationConverter,
    ConductivityConverter,
    DataRateConverter,
    DistanceConverter,
    DurationConverter,
    ElectricCurrentConverter,
    ElectricPotentialConverter,
    EnergyConverter,
    EnergyDistanceConverter,
    InformationConverter,
    MassConverter,
    PowerConverter,
    PressureConverter,
    SpeedConverter,
    TemperatureConverter,
    UnitlessRatioConverter,
    VolumeConverter,
    VolumeFlowRateConverter,
)

INVALID_SYMBOL: str = "bob"

# Dict containing all converters that need to be tested.
# The VALID_UNITS are sorted to ensure that pytest runs are consistent
# and avoid `different tests were collected between gw0 and gw1`
_ALL_CONVERTERS: Dict[Type[BaseUnitConverter], List[Union[str, None]]] = {
    converter: sorted(converter.VALID_UNITS, key=lambda x: (x is None, x))
    for converter in (
        AreaConverter,
        BloodGlucoseConcentrationConverter,
        ConductivityConverter,
        DataRateConverter,
        DistanceConverter,
        DurationConverter,
        ElectricCurrentConverter,
        ElectricPotentialConverter,
        EnergyConverter,
        InformationConverter,
        MassConverter,
        PowerConverter,
        PressureConverter,
        SpeedConverter,
        TemperatureConverter,
        UnitlessRatioConverter,
        EnergyDistanceConverter,
        VolumeConverter,
        VolumeFlowRateConverter,
    )
}

# Dict containing all converters with a corresponding unit ratio.
_GET_UNIT_RATIO: Dict[
    Type[BaseUnitConverter], Tuple[Union[str, None], Union[str, None], float]
] = {
    AreaConverter: (UnitOfArea.SQUARE_KILOMETERS, UnitOfArea.SQUARE_METERS, 0.000001),
    BloodGlucoseConcentrationConverter: (
        UnitOfBloodGlucoseConcentration.MILLIGRAMS_PER_DECILITER,
        UnitOfBloodGlucoseConcentration.MILLIMOLE_PER_LITER,
        18,
    ),
    ConductivityConverter: (
        UnitOfConductivity.MICROSIEMENS_PER_CM,
        UnitOfConductivity.MILLISIEMENS_PER_CM,
        1000,
    ),
    DataRateConverter: (
        UnitOfDataRate.BITS_PER_SECOND,
        UnitOfDataRate.BYTES_PER_SECOND,
        8,
    ),
    DistanceConverter: (UnitOfLength.KILOMETERS, UnitOfLength.METERS, 0.001),
    DurationConverter: (UnitOfTime.MINUTES, UnitOfTime.SECONDS, 1 / 60),
    ElectricCurrentConverter: (
        UnitOfElectricCurrent.AMPERE,
        UnitOfElectricCurrent.MILLIAMPERE,
        0.001,
    ),
    ElectricPotentialConverter: (
        UnitOfElectricPotential.MILLIVOLT,
        UnitOfElectricPotential.VOLT,
        1000,
    ),
    EnergyConverter: (UnitOfEnergy.WATT_HOUR, UnitOfEnergy.KILO_WATT_HOUR, 1000),
    EnergyDistanceConverter: (
        UnitOfEnergyDistance.MILES_PER_KILO_WATT_HOUR,
        UnitOfEnergyDistance.KM_PER_KILO_WATT_HOUR,
        0.621371,
    ),
    InformationConverter: (UnitOfInformation.BITS, UnitOfInformation.BYTES, 8),
    MassConverter: (UnitOfMass.STONES, UnitOfMass.KILOGRAMS, 0.157473),
    PowerConverter: (UnitOfPower.WATT, UnitOfPower.KILO_WATT, 1000),
    PressureConverter: (UnitOfPressure.HPA, UnitOfPressure.INHG, 33.86389),
    SpeedConverter: (
        UnitOfSpeed.KILOMETERS_PER_HOUR,
        UnitOfSpeed.MILES_PER_HOUR,
        1.609343,
    ),
    TemperatureConverter: (
        UnitOfTemperature.CELSIUS,
        UnitOfTemperature.FAHRENHEIT,
        0.555556,
    ),
    UnitlessRatioConverter: (PERCENTAGE, None, 100),
    VolumeConverter: (UnitOfVolume.GALLONS, UnitOfVolume.LITERS, 0.264172),
    VolumeFlowRateConverter: (
        UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR,
        UnitOfVolumeFlowRate.LITERS_PER_MINUTE,
        0.06,
    ),
}

# Dict containing a conversion test for every known unit.
_CONVERTED_VALUE: Dict[
    Type[BaseUnitConverter], List[Tuple[float, Union[str, None], float, Union[str, None]]]
] = {
    AreaConverter: [
        # Square Meters to other units
        (5, UnitOfArea.SQUARE_METERS, 50000, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.SQUARE_METERS, 5000000, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.SQUARE_METERS, 0.000005, UnitOfArea.SQUARE_KILOMETERS),
        (5, UnitOfArea.SQUARE_METERS, 7750.015500031001, UnitOfArea.SQUARE_INCHES),
        (5, UnitOfArea.SQUARE_METERS, 53.81955, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.SQUARE_METERS, 5.979950231505403, UnitOfArea.SQUARE_YARDS),
        (5, UnitOfArea.SQUARE_METERS, 1.9305107927122295e-06, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.SQUARE_METERS, 0.0012355269073358272, UnitOfArea.ACRES),
        (5, UnitOfArea.SQUARE_METERS, 0.0005, UnitOfArea.HECTARES),
        # Square Kilometers to other units
        (1, UnitOfArea.SQUARE_KILOMETERS, 1000000, UnitOfArea.SQUARE_METERS),
        (1, UnitOfArea.SQUARE_KILOMETERS, 1e10, UnitOfArea.SQUARE_CENTIMETERS),
        (1, UnitOfArea.SQUARE_KILOMETERS, 1e12, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.SQUARE_KILOMETERS, 1.9305107927122296, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.SQUARE_KILOMETERS, 1235.5269073358272, UnitOfArea.ACRES),
        (5, UnitOfArea.SQUARE_KILOMETERS, 500, UnitOfArea.HECTARES),
        # Acres to other units
        (5, UnitOfArea.ACRES, 20234.3, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.ACRES, 202342821.11999995, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.ACRES, 20234282111.999992, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.ACRES, 0.0202343, UnitOfArea.SQUARE_KILOMETERS),
        (5, UnitOfArea.ACRES, 217800, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.ACRES, 24200.0, UnitOfArea.SQUARE_YARDS),
        (5, UnitOfArea.ACRES, 0.0078125, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.ACRES, 2.02343, UnitOfArea.HECTARES),
        # Hectares to other units
        (5, UnitOfArea.HECTARES, 50000, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.HECTARES, 500000000, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.HECTARES, 50000000000.0, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.HECTARES, 0.019305107927122298, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.HECTARES, 538195.5, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.HECTARES, 59799.50231505403, UnitOfArea.SQUARE_YARDS),
        (5, UnitOfArea.HECTARES, 12.355269073358272, UnitOfArea.ACRES),
        # Square Miles to other units
        (5, UnitOfArea.SQUARE_MILES, 12949940.551679997, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.SQUARE_MILES, 129499405516.79997, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.SQUARE_MILES, 12949940551679.996, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.SQUARE_MILES, 1294.9940551679997, UnitOfArea.HECTARES),
        (5, UnitOfArea.SQUARE_MILES, 3200, UnitOfArea.ACRES),
        # Square Yards to other units
        (5, UnitOfArea.SQUARE_YARDS, 4.1806367999999985, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.SQUARE_YARDS, 41806.4, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.SQUARE_YARDS, 4180636.7999999984, UnitOfArea.SQUARE_MILLIMETERS),
        (
            5,
            UnitOfArea.SQUARE_YARDS,
            4.180636799999998e-06,
            UnitOfArea.SQUARE_KILOMETERS,
        ),
        (5, UnitOfArea.SQUARE_YARDS, 45.0, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.SQUARE_YARDS, 6479.999999999998, UnitOfArea.SQUARE_INCHES),
        (5, UnitOfArea.SQUARE_YARDS, 1.6141528925619832e-06, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.SQUARE_YARDS, 0.0010330578512396695, UnitOfArea.ACRES),
    ],
    BloodGlucoseConcentrationConverter: [
        (
            90,
            UnitOfBloodGlucoseConcentration.MILLIGRAMS_PER_DECILITER,
            5,
            UnitOfBloodGlucoseConcentration.MILLIMOLE_PER_LITER,
        ),
        (
            1,
            UnitOfBloodGlucoseConcentration.MILLIMOLE_PER_LITER,
            18,
            UnitOfBloodGlucoseConcentration.MILLIGRAMS_PER_DECILITER,
        ),
    ],
    ConductivityConverter: [
        # Deprecated to deprecated
        (5, UnitOfConductivity.SIEMENS, 5e3, UnitOfConductivity.MILLISIEMENS),
        (5, UnitOfConductivity.SIEMENS, 5e6, UnitOfConductivity.MICROSIEMENS),
        (5, UnitOfConductivity.MILLISIEMENS, 5e3, UnitOfConductivity.MICROSIEMENS),
        (5, UnitOfConductivity.MILLISIEMENS, 5e-3, UnitOfConductivity.SIEMENS),
        (5e6, UnitOfConductivity.MICROSIEMENS, 5e3, UnitOfConductivity.MILLISIEMENS),
        (5e6, UnitOfConductivity.MICROSIEMENS, 5, UnitOfConductivity.SIEMENS),
        # Deprecated to new
        (5, UnitOfConductivity.SIEMENS, 5e3, UnitOfConductivity.MILLISIEMENS_PER_CM),
        (5, UnitOfConductivity.SIEMENS, 5e6, UnitOfConductivity.MICROSIEMENS_PER_CM),
        (
            5,
            UnitOfConductivity.MILLISIEMENS,
            5e3,
            UnitOfConductivity.MICROSIEMENS_PER_CM,
        ),
        (5, UnitOfConductivity.MILLISIEMENS, 5e-3, UnitOfConductivity.SIEMENS_PER_CM),
        (
            5e6,
            UnitOfConductivity.MICROSIEMENS,
            5e3,
            UnitOfConductivity.MILLISIEMENS_PER_CM,
        ),
        (5e6, UnitOfConductivity.MICROSIEMENS, 5, UnitOfConductivity.SIEMENS_PER_CM),
        # New to deprecated
        (5, UnitOfConductivity.SIEMENS_PER_CM, 5e3, UnitOfConductivity.MILLISIEMENS),
        (5, UnitOfConductivity.SIEMENS_PER_CM, 5e6, UnitOfConductivity.MICROSIEMENS),
        (
            5,
            UnitOfConductivity.MILLISIEMENS_PER_CM,
            5e3,
            UnitOfConductivity.MICROSIEMENS,
        ),
        (5, UnitOfConductivity.MILLISIEMENS_PER_CM, 5e-3, UnitOfConductivity.SIEMENS),
        (
            5e6,
            UnitOfConductivity.MICROSIEMENS_PER_CM,
            5e3,
            UnitOfConductivity.MILLISIEMENS,
        ),
        (5e6, UnitOfConductivity.MICROSIEMENS_PER_CM, 5, UnitOfConductivity.SIEMENS),
        # New to new
        (
            5,
            UnitOfConductivity.SIEMENS_PER_CM,
            5e3,
            UnitOfConductivity.MILLISIEMENS_PER_CM,
        ),
        (
            5,
            UnitOfConductivity.SIEMENS_PER_CM,
            5e6,
            UnitOfConductivity.MICROSIEMENS_PER_CM,
        ),
        (
            5,
            UnitOfConductivity.MILLISIEMENS_PER_CM,
            5e3,
            UnitOfConductivity.MICROSIEMENS_PER_CM,
        ),
        (
            5,
            UnitOfConductivity.MILLISIEMENS_PER_CM,
            5e-3,
            UnitOfConductivity.SIEMENS_PER_CM,
        ),
        (
            5e6,
            UnitOfConductivity.MICROSIEMENS_PER_CM,
            5e3,
            UnitOfConductivity.MILLISIEMENS_PER_CM,
        ),
        (
            5e6,
            UnitOfConductivity.MICROSIEMENS_PER_CM,
            5,
            UnitOfConductivity.SIEMENS_PER_CM,
        ),
    ],
    DataRateConverter: [
        (8e3, UnitOfDataRate.BITS_PER_SECOND, 8, UnitOfDataRate.KILOBITS_PER_SECOND),
        (8e6, UnitOfDataRate.BITS_PER_SECOND, 8, UnitOfDataRate.MEGABITS_PER_SECOND),
        (8e9, UnitOfDataRate.BITS_PER_SECOND, 8, UnitOfDataRate.GIGABITS_PER_SECOND),
        (8, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.BYTES_PER_SECOND),
        (8e3, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.KILOBYTES_PER_SECOND),
        (8e6, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.MEGABYTES_PER_SECOND),
        (8e9, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.GIGABYTES_PER_SECOND),
        (
            8 * 2**10,
            UnitOfDataRate.BITS_PER_SECOND,
            1,
            UnitOfDataRate.KIBIBYTES_PER_SECOND,
        ),
        (
            8 * 2**20,
            UnitOfDataRate.BITS_PER_SECOND,
            1,
            UnitOfDataRate.MEBIBYTES_PER_SECOND,
        ),
        (
            8 * 2**30,
            UnitOfDataRate.BITS_PER_SECOND,
            1,
            UnitOfDataRate.GIBIBYTES_PER_SECOND,
        ),
    ],
    DistanceConverter: [
        (5, UnitOfLength.MILES, 8.04672, UnitOfLength.KILOMETERS),
        (5, UnitOfLength.MILES, 8046.72, UnitOfLength.METERS),
        (5, UnitOfLength.MILES, 804672.0, UnitOfLength.CENTIMETERS),
        (5, UnitOfLength.MILES, 8046720.0, UnitOfLength.MILLIMETERS),
        (5, UnitOfLength.MILES, 8800.0, UnitOfLength.YARDS),
        (5, UnitOfLength.MILES, 26400.0008448, UnitOfLength.FEET),
        (5, UnitOfLength.MILES, 316800.171072, UnitOfLength.INCHES),
        (5, UnitOfLength