from __future__ import annotations
import inspect
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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

_ALL_CONVERTERS: Dict[Type[BaseUnitConverter], List[Optional[str]]] = {
    converter: sorted(
        converter.VALID_UNITS, key=lambda x: (x is None, x)
    )
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

_GET_UNIT_RATIO: Dict[
    Type[BaseUnitConverter], Tuple[Optional[str], Optional[str], float]
] = {
    AreaConverter: (
        UnitOfArea.SQUARE_KILOMETERS,
        UnitOfArea.SQUARE_METERS,
        1e-06,
    ),
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
    DistanceConverter: (
        UnitOfLength.KILOMETERS,
        UnitOfLength.METERS,
        0.001,
    ),
    DurationConverter: (
        UnitOfTime.MINUTES,
        UnitOfTime.SECONDS,
        1 / 60,
    ),
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
    EnergyConverter: (
        UnitOfEnergy.WATT_HOUR,
        UnitOfEnergy.KILO_WATT_HOUR,
        1000,
    ),
    EnergyDistanceConverter: (
        UnitOfEnergyDistance.MILES_PER_KILO_WATT_HOUR,
        UnitOfEnergyDistance.KM_PER_KILO_WATT_HOUR,
        0.621371,
    ),
    InformationConverter: (
        UnitOfInformation.BITS,
        UnitOfInformation.BYTES,
        8,
    ),
    MassConverter: (
        UnitOfMass.STONES,
        UnitOfMass.KILOGRAMS,
        0.157473,
    ),
    PowerConverter: (
        UnitOfPower.WATT,
        UnitOfPower.KILO_WATT,
        1000,
    ),
    PressureConverter: (
        UnitOfPressure.HPA,
        UnitOfPressure.INHG,
        33.86389,
    ),
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
    UnitlessRatioConverter: (
        PERCENTAGE,
        None,
        100,
    ),
    VolumeConverter: (
        UnitOfVolume.GALLONS,
        UnitOfVolume.LITERS,
        0.264172,
    ),
    VolumeFlowRateConverter: (
        UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR,
        UnitOfVolumeFlowRate.LITERS_PER_MINUTE,
        0.06,
    ),
}

_CONVERTED_VALUE: Dict[
    Type[BaseUnitConverter],
    List[Tuple[float, Optional[str], float, Optional[str]]],
] = {
    AreaConverter: [
        (5, UnitOfArea.SQUARE_METERS, 50000, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.SQUARE_METERS, 5000000, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.SQUARE_METERS, 5e-06, UnitOfArea.SQUARE_KILOMETERS),
        (5, UnitOfArea.SQUARE_METERS, 7750.015500031001, UnitOfArea.SQUARE_INCHES),
        (5, UnitOfArea.SQUARE_METERS, 53.81955, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.SQUARE_METERS, 5.979950231505403, UnitOfArea.SQUARE_YARDS),
        (5, UnitOfArea.SQUARE_METERS, 1.9305107927122295e-06, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.SQUARE_METERS, 0.0012355269073358272, UnitOfArea.ACRES),
        (5, UnitOfArea.SQUARE_METERS, 0.0005, UnitOfArea.HECTARES),
        (1, UnitOfArea.SQUARE_KILOMETERS, 1000000, UnitOfArea.SQUARE_METERS),
        (1, UnitOfArea.SQUARE_KILOMETERS, 10000000000.0, UnitOfArea.SQUARE_CENTIMETERS),
        (1, UnitOfArea.SQUARE_KILOMETERS, 1000000000000.0, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.SQUARE_KILOMETERS, 1.9305107927122296, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.SQUARE_KILOMETERS, 1235.5269073358272, UnitOfArea.ACRES),
        (5, UnitOfArea.SQUARE_KILOMETERS, 500, UnitOfArea.HECTARES),
        (5, UnitOfArea.ACRES, 20234.3, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.ACRES, 202342821.11999995, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.ACRES, 20234282111.999992, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.ACRES, 0.0202343, UnitOfArea.SQUARE_KILOMETERS),
        (5, UnitOfArea.ACRES, 217800, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.ACRES, 24200.0, UnitOfArea.SQUARE_YARDS),
        (5, UnitOfArea.ACRES, 0.0078125, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.ACRES, 2.02343, UnitOfArea.HECTARES),
        (5, UnitOfArea.HECTARES, 50000, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.HECTARES, 500000000, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.HECTARES, 50000000000.0, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.HECTARES, 0.019305107927122298, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.HECTARES, 538195.5, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.HECTARES, 59799.50231505403, UnitOfArea.SQUARE_YARDS),
        (5, UnitOfArea.HECTARES, 12.355269073358272, UnitOfArea.ACRES),
        (5, UnitOfArea.SQUARE_MILES, 12949940.551679997, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.SQUARE_MILES, 129499405516.79997, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.SQUARE_MILES, 12949940551679.996, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.SQUARE_MILES, 1294.9940551679997, UnitOfArea.HECTARES),
        (5, UnitOfArea.SQUARE_MILES, 3200, UnitOfArea.ACRES),
        (5, UnitOfArea.SQUARE_YARDS, 4.1806367999999985, UnitOfArea.SQUARE_METERS),
        (5, UnitOfArea.SQUARE_YARDS, 41806.4, UnitOfArea.SQUARE_CENTIMETERS),
        (5, UnitOfArea.SQUARE_YARDS, 4180636.7999999984, UnitOfArea.SQUARE_MILLIMETERS),
        (5, UnitOfArea.SQUARE_YARDS, 4.180636799999998e-06, UnitOfArea.SQUARE_KILOMETERS),
        (5, UnitOfArea.SQUARE_YARDS, 45.0, UnitOfArea.SQUARE_FEET),
        (5, UnitOfArea.SQUARE_YARDS, 6479.999999999998, UnitOfArea.SQUARE_INCHES),
        (5, UnitOfArea.SQUARE_YARDS, 1.6141528925619832e-06, UnitOfArea.SQUARE_MILES),
        (5, UnitOfArea.SQUARE_YARDS, 0.0010330578512396695, UnitOfArea.ACRES),
    ],
    BloodGlucoseConcentrationConverter: [
        (90, UnitOfBloodGlucoseConcentration.MILLIGRAMS_PER_DECILITER, 5, UnitOfBloodGlucoseConcentration.MILLIMOLE_PER_LITER),
        (1, UnitOfBloodGlucoseConcentration.MILLIMOLE_PER_LITER, 18, UnitOfBloodGlucoseConcentration.MILLIGRAMS_PER_DECILITER),
    ],
    ConductivityConverter: [
        (5, UnitOfConductivity.SIEMENS, 5000.0, UnitOfConductivity.MILLISIEMENS),
        (5, UnitOfConductivity.SIEMENS, 5000000.0, UnitOfConductivity.MICROSIEMENS),
        (5, UnitOfConductivity.MILLISIEMENS, 5000.0, UnitOfConductivity.MICROSIEMENS),
        (5, UnitOfConductivity.MILLISIEMENS, 0.005, UnitOfConductivity.SIEMENS),
        (5000000.0, UnitOfConductivity.MICROSIEMENS, 5000.0, UnitOfConductivity.MILLISIEMENS),
        (5000000.0, UnitOfConductivity.MICROSIEMENS, 5, UnitOfConductivity.SIEMENS),
        (5, UnitOfConductivity.SIEMENS, 5000.0, UnitOfConductivity.MILLISIEMENS_PER_CM),
        (5, UnitOfConductivity.SIEMENS, 5000000.0, UnitOfConductivity.MICROSIEMENS_PER_CM),
        (5, UnitOfConductivity.MILLISIEMENS, 5000.0, UnitOfConductivity.MICROSIEMENS_PER_CM),
        (5, UnitOfConductivity.MILLISIEMENS, 0.005, UnitOfConductivity.SIEMENS_PER_CM),
        (5000000.0, UnitOfConductivity.MICROSIEMENS, 5000.0, UnitOfConductivity.MILLISIEMENS_PER_CM),
        (5000000.0, UnitOfConductivity.MICROSIEMENS, 5, UnitOfConductivity.SIEMENS_PER_CM),
        (5, UnitOfConductivity.SIEMENS_PER_CM, 5000.0, UnitOfConductivity.MILLISIEMENS),
        (5, UnitOfConductivity.SIEMENS_PER_CM, 5000000.0, UnitOfConductivity.MICROSIEMENS),
        (5, UnitOfConductivity.MILLISIEMENS_PER_CM, 5000.0, UnitOfConductivity.MICROSIEMENS),
        (5, UnitOfConductivity.MILLISIEMENS_PER_CM, 0.005, UnitOfConductivity.SIEMENS),
        (5000000.0, UnitOfConductivity.MICROSIEMENS_PER_CM, 5000.0, UnitOfConductivity.MILLISIEMENS),
        (5000000.0, UnitOfConductivity.MICROSIEMENS_PER_CM, 5, UnitOfConductivity.SIEMENS),
        (5, UnitOfConductivity.SIEMENS_PER_CM, 5000.0, UnitOfConductivity.MILLISIEMENS_PER_CM),
        (5, UnitOfConductivity.SIEMENS_PER_CM, 5000000.0, UnitOfConductivity.MICROSIEMENS_PER_CM),
        (5, UnitOfConductivity.MILLISIEMENS_PER_CM, 5000.0, UnitOfConductivity.MICROSIEMENS_PER_CM),
        (5, UnitOfConductivity.MILLISIEMENS_PER_CM, 0.005, UnitOfConductivity.SIEMENS_PER_CM),
        (5000000.0, UnitOfConductivity.MICROSIEMENS_PER_CM, 5000.0, UnitOfConductivity.MILLISIEMENS_PER_CM),
        (5000000.0, UnitOfConductivity.MICROSIEMENS_PER_CM, 5, UnitOfConductivity.SIEMENS_PER_CM),
    ],
    DataRateConverter: [
        (8000.0, UnitOfDataRate.BITS_PER_SECOND, 8, UnitOfDataRate.KILOBITS_PER_SECOND),
        (8000000.0, UnitOfDataRate.BITS_PER_SECOND, 8, UnitOfDataRate.MEGABITS_PER_SECOND),
        (8000000000.0, UnitOfDataRate.BITS_PER_SECOND, 8, UnitOfDataRate.GIGABITS_PER_SECOND),
        (8, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.BYTES_PER_SECOND),
        (8000.0, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.KILOBYTES_PER_SECOND),
        (8000000.0, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.MEGABYTES_PER_SECOND),
        (8000000000.0, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.GIGABYTES_PER_SECOND),
        (8 * 2 ** 10, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.KIBIBYTES_PER_SECOND),
        (8 * 2 ** 20, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.MEBIBYTES_PER_SECOND),
        (8 * 2 ** 30, UnitOfDataRate.BITS_PER_SECOND, 1, UnitOfDataRate.GIBIBYTES_PER_SECOND),
    ],
    DistanceConverter: [
        (5, UnitOfLength.MILES, 8.04672, UnitOfLength.KILOMETERS),
        (5, UnitOfLength.MILES, 8046.72, UnitOfLength.METERS),
        (5, UnitOfLength.MILES, 804672.0, UnitOfLength.CENTIMETERS),
        (5, UnitOfLength.MILES, 8046720.0, UnitOfLength.MILLIMETERS),
        (5, UnitOfLength.MILES, 8800.0, UnitOfLength.YARDS),
        (5, UnitOfLength.MILES, 26400.0008448, UnitOfLength.FEET),
        (5, UnitOfLength.MILES, 316800.171072, UnitOfLength.INCHES),
        (5, UnitOfLength.NAUTICAL_MILES, 9.26, UnitOfLength.KILOMETERS),
        (5, UnitOfLength.NAUTICAL_MILES, 9260.0, UnitOfLength.METERS),
        (5, UnitOfLength.NAUTICAL_MILES, 926000.0, UnitOfLength.CENTIMETERS),
        (5, UnitOfLength.NAUTICAL_MILES, 9260000.0, UnitOfLength.MILLIMETERS),
        (5, UnitOfLength.NAUTICAL_MILES, 10126.859142607176, UnitOfLength.YARDS),
        (5, UnitOfLength.NAUTICAL_MILES, 30380.57742782153, UnitOfLength.FEET),
        (5, UnitOfLength.NAUTICAL_MILES, 364566.9291338583, UnitOfLength.INCHES),
        (5, UnitOfLength.YARDS, 0.004572, UnitOfLength.KILOMETERS),
        (5, UnitOfLength.YARDS, 4.572, UnitOfLength.METERS),
        (5, UnitOfLength.YARDS, 457.2, UnitOfLength.CENTIMETERS),
        (5, UnitOfLength.YARDS, 4572, UnitOfLength.MILLIMETERS),
        (5, UnitOfLength.YARDS, 0.002840908212, UnitOfLength.MILES),
        (5, UnitOfLength.YARDS, 15.00000048, UnitOfLength.FEET),
        (5, UnitOfLength.YARDS, 180.0000972, UnitOfLength.INCHES),
        (5000, UnitOfLength.FEET, 1.524, UnitOfLength.KILOMETERS),
        (5000, UnitOfLength.FEET, 1524, UnitOfLength.METERS),
        (5000, UnitOfLength.FEET, 152400.0, UnitOfLength.CENTIMETERS),
        (5000, UnitOfLength.FEET, 1524000.0, UnitOfLength.MILLIMETERS),
        (5000, UnitOfLength.FEET, 0.946969404, UnitOfLength.MILES),
        (5000, UnitOfLength.FEET, 1666.66667, UnitOfLength.YARDS),
        (5000, UnitOfLength.FEET, 60000.0324, UnitOfLength.INCHES),
        (5000, UnitOfLength.INCHES, 0.127, UnitOfLength.KILOMETERS),
        (5000, UnitOfLength.INCHES, 127.0, UnitOfLength.METERS),
        (5000, UnitOfLength.INCHES, 12700.0, UnitOfLength.CENTIMETERS),
        (5000, UnitOfLength.INCHES, 127000.0, UnitOfLength.MILLIMETERS),
        (5000, UnitOfLength.INCHES, 0.078914117, UnitOfLength.MILES),
        (5000, UnitOfLength.INCHES, 138.88889, UnitOfLength.YARDS),
        (5000, UnitOfLength.INCHES, 416.66668, UnitOfLength.FEET),
        (5, UnitOfLength.KILOMETERS, 5000, UnitOfLength.METERS),
        (5, UnitOfLength.KILOMETERS, 500000, UnitOfLength.CENTIMETERS),
        (5, UnitOfLength.KILOMETERS, 5000000, UnitOfLength.MILLIMETERS),
        (5, UnitOfLength.KILOMETERS, 3.106855, UnitOfLength.MILES),
        (5, UnitOfLength.KILOMETERS, 5468.066, UnitOfLength.YARDS),
        (5, UnitOfLength.KILOMETERS, 16404.2, UnitOfLength.FEET),
        (5, UnitOfLength.KILOMETERS, 196850.5, UnitOfLength.INCHES),
        (5000, UnitOfLength.METERS, 5, UnitOfLength.KILOMETERS),
        (5000, UnitOfLength.METERS, 500000, UnitOfLength.CENTIMETERS),
        (5000, UnitOfLength.METERS, 5000000, UnitOfLength.MILLIMETERS),
        (5000, UnitOfLength.METERS, 3.106855, UnitOfLength.MILES),
        (5000, UnitOfLength.METERS, 5468.066, UnitOfLength.YARDS),
        (5000, UnitOfLength.METERS, 16404.2, UnitOfLength.FEET),
        (5000, UnitOfLength.METERS, 196850.5, UnitOfLength.INCHES),
        (500000, UnitOfLength.CENTIMETERS, 5, UnitOfLength.KILOMETERS),
        (500000, UnitOfLength.CENTIMETERS, 5000, UnitOfLength.METERS),
        (500000, UnitOfLength.CENTIMETERS, 5000000, UnitOfLength.MILLIMETERS),
        (500000, UnitOfLength.CENTIMETERS, 3.106855, UnitOfLength.MILES),
        (500000, UnitOfLength.CENTIMETERS, 5468.066, UnitOfLength.YARDS),
        (500000, UnitOfLength.CENTIMETERS, 16404.2, UnitOfLength.FEET),
        (500000, UnitOfLength.CENTIMETERS, 196850.5, UnitOfLength.INCHES),
        (5000000, UnitOfLength.MILLIMETERS, 5, UnitOfLength.KILOMETERS),
        (5000000, UnitOfLength.MILLIMETERS, 5000, UnitOfLength.METERS),
        (5000000, UnitOfLength.MILLIMETERS, 500000, UnitOfLength.CENTIMETERS),
        (5000000, UnitOfLength.MILLIMETERS, 3.106855, UnitOfLength.MILES),
        (5000000, UnitOfLength.MILLIMETERS, 5468.066, UnitOfLength.YARDS),
        (5000000, UnitOfLength.MILLIMETERS, 16404.2, UnitOfLength.FEET),
        (5000000, UnitOfLength.MILLIMETERS, 196850.5, UnitOfLength.INCHES),
    ],
    DurationConverter: [
        (5, UnitOfTime.MICROSECONDS, 0.005, UnitOfTime.MILLISECONDS),
        (5, UnitOfTime.MICROSECONDS, 5e-06, UnitOfTime.SECONDS),
        (5, UnitOfTime.MICROSECONDS, 8.333333333333333e-08, UnitOfTime.MINUTES),
        (5, UnitOfTime.MICROSECONDS, 1.388888888888889e-09, UnitOfTime.HOURS),
        (5, UnitOfTime.MICROSECONDS, 5.787e-11, UnitOfTime.DAYS),
        (5, UnitOfTime.MICROSECONDS, 8.267195767195768e-12, UnitOfTime.WEEKS),
        (5, UnitOfTime.MILLISECONDS, 5000, UnitOfTime.MICROSECONDS),
        (5, UnitOfTime.MILLISECONDS, 0.005, UnitOfTime.SECONDS),
        (5, UnitOfTime.MILLISECONDS, 8.333333333333333e-05, UnitOfTime.MINUTES),
        (5, UnitOfTime.MILLISECONDS, 1.388888888888889e-06, UnitOfTime.HOURS),
        (5, UnitOfTime.MILLISECONDS, 5.787e-08, UnitOfTime.DAYS),
        (5, UnitOfTime.MILLISECONDS, 8.267195767195767e-09, UnitOfTime.WEEKS),
        (5, UnitOfTime.SECONDS, 5000000.0, UnitOfTime.MICROSECONDS),
        (5, UnitOfTime.SECONDS, 5000, UnitOfTime.MILLISECONDS),
        (5, UnitOfTime.SECONDS, 0.0833333, UnitOfTime.MINUTES),
        (5, UnitOfTime.SECONDS, 0.00138889, UnitOfTime.HOURS),
        (5, UnitOfTime.SECONDS, 5.787037037037037e-05, UnitOfTime.DAYS),
        (5, UnitOfTime.SECONDS, 8.267195767195768e-06, UnitOfTime.WEEKS),
        (5, UnitOfTime.MINUTES, 300000000.0, UnitOfTime.MICROSECONDS),
        (5, UnitOfTime.MINUTES, 300000, UnitOfTime.MILLISECONDS),
        (5, UnitOfTime.MINUTES, 300, UnitOfTime.SECONDS),
        (5, UnitOfTime.MINUTES, 0.0833333, UnitOfTime.HOURS),
        (5, UnitOfTime.MINUTES, 0.00347222, UnitOfTime.DAYS),
        (5, UnitOfTime.MINUTES, 0.000496031746031746, UnitOfTime.WEEKS),
        (5, UnitOfTime.HOURS, 18000000000, UnitOfTime.MICROSECONDS),
        (5, UnitOfTime.HOURS, 18000000, UnitOfTime.MILLISECONDS),
        (5, UnitOfTime.HOURS, 18000, UnitOfTime.SECONDS),
        (5, UnitOfTime.HOURS, 300, UnitOfTime.MINUTES),
        (5, UnitOfTime.HOURS, 0.208333333, UnitOfTime.DAYS),
        (5, UnitOfTime.HOURS, 0.02976190476190476, UnitOfTime.WEEKS),
        (5, UnitOfTime.DAYS, 432000000000.0, UnitOfTime.MICROSECONDS),
        (5, UnitOfTime.DAYS, 432000000.0, UnitOfTime.MILLISECONDS),
        (5, UnitOfTime.DAYS, 432000, UnitOfTime.SECONDS),
        (5, UnitOfTime.DAYS, 7200, UnitOfTime.MINUTES),
        (5, UnitOfTime.DAYS, 120, UnitOfTime.HOURS),
        (5, UnitOfTime.DAYS, 0.7142857142857143, UnitOfTime.WEEKS),
        (5, UnitOfTime.WEEKS, 3024000000000.0, UnitOfTime.MICROSECONDS),
        (5, UnitOfTime.WEEKS, 3024000000.0, UnitOfTime.MILLISECONDS),
        (5, UnitOfTime.WEEKS, 3024000, UnitOfTime.SECONDS),
        (5, UnitOfTime.WEEKS, 50400, UnitOfTime.MINUTES),
        (5, UnitOfTime.WEEKS, 840, UnitOfTime.HOURS),
        (5, UnitOfTime.WEEKS, 35, UnitOfTime.DAYS),
    ],
    ElectricCurrentConverter: [
        (5, UnitOfElectricCurrent.AMPERE, 5000, UnitOfElectricCurrent.MILLIAMPERE),
        (5, UnitOfElectricCurrent.MILLIAMPERE, 0.005, UnitOfElectricCurrent.AMPERE),
    ],
    ElectricPotentialConverter: [
        (5, UnitOfElectricPotential.VOLT, 5000, UnitOfElectricPotential.MILLIVOLT),
        (5, UnitOfElectricPotential.VOLT, 5000000.0, UnitOfElectricPotential.MICROVOLT),
        (5, UnitOfElectricPotential.VOLT, 0.005, UnitOfElectricPotential.KILOVOLT),
        (5, UnitOfElectricPotential.VOLT, 5e-06, UnitOfElectricPotential.MEGAVOLT),
        (5, UnitOfElectricPotential.MILLIVOLT, 0.005, UnitOfElectricPotential.VOLT),
        (5, UnitOfElectricPotential.MILLIVOLT, 5000.0, UnitOfElectricPotential.MICROVOLT),
        (5, UnitOfElectricPotential.MILLIVOLT, 5e-06, UnitOfElectricPotential.KILOVOLT),
        (5, UnitOfElectricPotential.MILLIVOLT, 5e-09, UnitOfElectricPotential.MEGAVOLT),
        (5, UnitOfElectricPotential.MICROVOLT, 0.005, UnitOfElectricPotential.MILLIVOLT),
        (5, UnitOfElectricPotential.MICROVOLT, 5e-06, UnitOfElectricPotential.VOLT),
        (5, UnitOfElectricPotential.MICROVOLT, 5e-09, UnitOfElectricPotential.KILOVOLT),
        (5, UnitOfElectricPotential.MICROVOLT, 5e-12, UnitOfElectricPotential.MEGAVOLT),
        (5, UnitOfElectricPotential.KILOVOLT, 5000000000.0, UnitOfElectricPotential.MICROVOLT),
        (5, UnitOfElectricPotential.KILOVOLT, 5000000.0, UnitOfElectricPotential.MILLIVOLT),
        (5, UnitOfElectricPotential.KILOVOLT, 5000.0, UnitOfElectricPotential.VOLT),
        (5, UnitOfElectricPotential.KILOVOLT, 0.005, UnitOfElectricPotential.MEGAVOLT),
        (5, UnitOfElectricPotential.MEGAVOLT, 5000000000000.0, UnitOfElectricPotential.MICROVOLT),
        (5, UnitOfElectricPotential.MEGAVOLT, 5000000000.0, UnitOfElectricPotential.MILLIVOLT),
        (5, UnitOfElectricPotential.MEGAVOLT, 5000000.0, UnitOfElectricPotential.VOLT),
        (5, UnitOfElectricPotential.MEGAVOLT, 5000.0, UnitOfElectricPotential.KILOVOLT),
    ],
    EnergyConverter: [
        (10, UnitOfEnergy.MILLIWATT_HOUR, 1e-05, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.WATT_HOUR, 10000, UnitOfEnergy.MILLIWATT_HOUR),
        (10, UnitOfEnergy.WATT_HOUR, 0.01, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.WATT_HOUR, 1e-05, UnitOfEnergy.MEGA_WATT_HOUR),
        (10, UnitOfEnergy.WATT_HOUR, 1e-08, UnitOfEnergy.GIGA_WATT_HOUR),
        (10, UnitOfEnergy.WATT_HOUR, 1e-11, UnitOfEnergy.TERA_WATT_HOUR),
        (10, UnitOfEnergy.KILO_WATT_HOUR, 10000, UnitOfEnergy.WATT_HOUR),
        (10, UnitOfEnergy.KILO_WATT_HOUR, 0.01, UnitOfEnergy.MEGA_WATT_HOUR),
        (10, UnitOfEnergy.MEGA_WATT_HOUR, 10000000, UnitOfEnergy.WATT_HOUR),
        (10, UnitOfEnergy.MEGA_WATT_HOUR, 10000, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.GIGA_WATT_HOUR, 10000000.0, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.GIGA_WATT_HOUR, 10000000000.0, UnitOfEnergy.WATT_HOUR),
        (10, UnitOfEnergy.TERA_WATT_HOUR, 10000000000.0, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.TERA_WATT_HOUR, 10000000000000.0, UnitOfEnergy.WATT_HOUR),
        (10, UnitOfEnergy.GIGA_JOULE, 2777.78, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.GIGA_JOULE, 2.77778, UnitOfEnergy.MEGA_WATT_HOUR),
        (10, UnitOfEnergy.MEGA_JOULE, 2.77778, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.MEGA_JOULE, 0.00277778, UnitOfEnergy.MEGA_WATT_HOUR),
        (10, UnitOfEnergy.KILO_JOULE, 2.77778, UnitOfEnergy.WATT_HOUR),
        (10, UnitOfEnergy.KILO_JOULE, 2.77778e-06, UnitOfEnergy.MEGA_WATT_HOUR),
        (10, UnitOfEnergy.JOULE, 0.00277778, UnitOfEnergy.WATT_HOUR),
        (10, UnitOfEnergy.JOULE, 2.390057, UnitOfEnergy.CALORIE),
        (10, UnitOfEnergy.CALORIE, 0.01, UnitOfEnergy.KILO_CALORIE),
        (10, UnitOfEnergy.CALORIE, 0.011622222, UnitOfEnergy.WATT_HOUR),
        (10, UnitOfEnergy.KILO_CALORIE, 0.01, UnitOfEnergy.MEGA_CALORIE),
        (10, UnitOfEnergy.KILO_CALORIE, 0.011622222, UnitOfEnergy.KILO_WATT_HOUR),
        (10, UnitOfEnergy.MEGA_CALORIE, 0.01, UnitOfEnergy.GIGA_CALORIE),
        (10, UnitOfEnergy.MEGA_CALORIE, 0.011622222, UnitOfEnergy.MEGA_WATT_HOUR),
        (10, UnitOfEnergy.GIGA_CALORIE, 10000, UnitOfEnergy.MEGA_CALORIE),
        (10, UnitOfEnergy.GIGA_CALORIE, 11.622222, UnitOfEnergy.MEGA_WATT_HOUR),
    ],
    EnergyDistanceConverter: [
        (10, UnitOfEnergyDistance.KILO_WATT_HOUR_PER_100_KM, 6.213712, UnitOfEnergyDistance.MILES_PER_KILO_WATT_HOUR),
        (25, UnitOfEnergyDistance.KILO_WATT_HOUR_PER_100_KM, 4, UnitOfEnergyDistance.KM_PER_KILO_WATT_HOUR),
        (20, UnitOfEnergyDistance.MILES_PER_KILO_WATT_HOUR, 3.106856, UnitOfEnergyDistance.KILO_WATT_HOUR_PER_100_KM),
        (10, UnitOfEnergyDistance.MILES_PER_KILO_WATT_HOUR, 16.09344, UnitOfEnergyDistance.KM_PER_KILO_WATT_HOUR),
        (16.09344, UnitOfEnergyDistance.KM_PER_KILO_WATT_HOUR, 10, UnitOfEnergyDistance.MILES_PER_KILO_WATT_HOUR),
    ],
    InformationConverter: [
        (8000.0, UnitOfInformation.BITS, 8, UnitOfInformation.KILOBITS),
        (8000000.0, UnitOfInformation.BITS, 8, UnitOfInformation.MEGABITS),
        (8000000000.0, UnitOfInformation.BITS, 8, UnitOfInformation.GIGABITS),
        (8, UnitOfInformation.BITS, 1, UnitOfInformation.BYTES),
        (8000.0, UnitOfInformation.BITS, 1, UnitOfInformation.KILOBYTES),
        (8000000.0, UnitOfInformation.BITS, 1, UnitOfInformation.MEGABYTES),
        (8000000000.0, UnitOfInformation.BITS, 1, UnitOfInformation.GIGABYTES),
        (8000000000000.0, UnitOfInformation.BITS, 1, UnitOfInformation.TERABYTES),
        (8000000000000000.0, UnitOfInformation.BITS, 1, UnitOfInformation.PETABYTES),
        (8e+18, UnitOfInformation.BITS, 1, UnitOfInformation.EXABYTES),
        (8e+21, UnitOfInformation.BITS, 1, UnitOfInformation.ZETTABYTES),
        (8e+24, UnitOfInformation.BITS, 1, UnitOfInformation.YOTTABYTES),
        (8 * 2 ** 10, UnitOfInformation.BITS, 1, UnitOfInformation.KIBIBYTES),
        (8 * 2 ** 20, UnitOfInformation.BITS, 1, UnitOfInformation.MEBIBYTES),
        (8 * 2 ** 30, UnitOfInformation.BITS, 1, UnitOfInformation.GIBIBYTES),
        (8 * 2 ** 40, UnitOfInformation.BITS, 1, UnitOfInformation.TEBIBYTES),
        (8 * 2 ** 50, UnitOfInformation.BITS, 1, UnitOfInformation.PEBIBYTES),
        (8 * 2 ** 60, UnitOfInformation.BITS, 1, UnitOfInformation.EXBIBYTES),
        (8 * 2 ** 70, UnitOfInformation.BITS, 1, UnitOfInformation.ZEBIBYTES),
        (8 * 2 ** 80, UnitOfInformation.BITS, 1, UnitOfInformation.YOBIBYTES),
    ],
    MassConverter: [
        (10, UnitOfMass.KILOGRAMS, 10000, UnitOfMass.GRAMS),
        (10, UnitOfMass.KILOGRAMS, 10000000, UnitOfMass.MILLIGRAMS),
        (10, UnitOfMass.KILOGRAMS, 10000000000, UnitOfMass.MICROGRAMS),
        (10, UnitOfMass.KILOGRAMS, 352.73961, UnitOfMass.OUNCES),
        (10, UnitOfMass.KILOGRAMS, 22.046226, UnitOfMass.POUNDS),
        (10, UnitOfMass.GRAMS, 0.01, UnitOfMass.KILOGRAMS),
        (10, UnitOfMass.GRAMS, 10000, UnitOfMass.MILLIGRAMS),
        (10, UnitOfMass.GRAMS, 10000000, UnitOfMass.MICROGRAMS),
        (10, UnitOfMass.GRAMS, 0.35273961, UnitOfMass.OUNCES),
        (10, UnitOfMass.GRAMS, 0.022046226, UnitOfMass.POUNDS),
        (10, UnitOfMass.MILLIGRAMS, 1e-05, UnitOfMass.KILOGRAMS),
        (10, UnitOfMass.MILLIGRAMS, 0.01, UnitOfMass.GRAMS),
        (10, UnitOfMass.MILLIGRAMS, 10000, UnitOfMass.MICROGRAMS),
        (10, UnitOfMass.MILLIGRAMS, 0.00035273961, UnitOfMass.OUNCES),
        (10, UnitOfMass.MILLIGRAMS, 2.2046226e-05, UnitOfMass.POUNDS),
        (10000, UnitOfMass.MICROGRAMS, 1e-05, UnitOfMass.KILOGRAMS),
        (10000, UnitOfMass.MICROGRAMS, 0.01, UnitOfMass.GRAMS),
        (10000, UnitOfMass.MICROGRAMS, 10, UnitOfMass.MILLIGRAMS),
        (10000, UnitOfMass.MICROGRAMS, 0.00035273961, UnitOfMass.OUNCES),
        (10000, UnitOfMass.MICROGRAMS, 2.2046226e-05, UnitOfMass.POUNDS),
        (1, UnitOfMass.POUNDS, 0.45359237, UnitOfMass.KILOGRAMS),
        (1, UnitOfMass.POUNDS, 453.59237, UnitOfMass.GRAMS),
        (1, UnitOfMass.POUNDS, 453592.37, UnitOfMass.MILLIGRAMS),
        (1, UnitOfMass.POUNDS, 453592370, UnitOfMass.MICROGRAMS),
        (1, UnitOfMass.POUNDS, 16, UnitOfMass.OUNCES),
        (16, UnitOfMass.OUNCES, 0.45359237, UnitOfMass.KILOGRAMS),
        (16, UnitOfMass.OUNCES, 453.59237, UnitOfMass.GRAMS),
        (16, UnitOfMass.OUNCES, 453592.37, UnitOfMass.MILLIGRAMS),
        (16, UnitOfMass.OUNCES, 453592370, UnitOfMass.MICROGRAMS),
        (16, UnitOfMass.OUNCES, 1, UnitOfMass.POUNDS),
        (1, UnitOfMass.STONES, 6.350293, UnitOfMass.KILOGRAMS),
        (1, UnitOfMass.STONES, 6350.293, UnitOfMass.GRAMS),
        (1, UnitOfMass.STONES, 6350293, UnitOfMass.MILLIGRAMS),
        (1, UnitOfMass.STONES, 14, UnitOfMass.POUNDS),
        (1, UnitOfMass.STONES, 224, UnitOfMass.OUNCES),
    ],
    PowerConverter: [
        (10, UnitOfPower.KILO_WATT, 10000, UnitOfPower.WATT),
        (10, UnitOfPower.MEGA_WATT, 10000000.0, UnitOfPower.WATT),
        (10, UnitOfPower.GIGA_WATT, 10000000000.0, UnitOfPower.WATT),
        (10, UnitOfPower.TERA_WATT, 10000000000000.0, UnitOfPower.WATT),
        (10, UnitOfPower.WATT, 0.01, UnitOfPower.KILO_WATT),
        (10, UnitOfPower.MILLIWATT, 0.01, UnitOfPower.WATT),
    ],
    PressureConverter: [
        (1000, UnitOfPressure.HPA, 14.5037743897, UnitOfPressure.PSI),
        (1000, UnitOfPressure.HPA, 29.5299801647, UnitOfPressure.INHG),
        (1000, UnitOfPressure.HPA, 100000, UnitOfPressure.PA),
        (1000, UnitOfPressure.HPA, 100, UnitOfPressure.KPA),
        (1000, UnitOfPressure.HPA, 1000, UnitOfPressure.MBAR),
        (1000, UnitOfPressure.HPA, 100, UnitOfPressure.CBAR),
        (100, UnitOfPressure.KPA, 14.5037743897, UnitOfPressure.PSI),
        (100, UnitOfPressure.KPA, 29.5299801647, UnitOfPressure.INHG),
        (100, UnitOfPressure.KPA, 100000, UnitOfPressure.PA),
        (100, UnitOfPressure.KPA, 1000, UnitOfPressure.HPA),
        (100, UnitOfPressure.KPA, 1000, UnitOfPressure.MBAR),
        (100, UnitOfPressure.KPA, 100, UnitOfPressure.CBAR),
        (30, UnitOfPressure.INHG, 14.7346266155, UnitOfPressure.PSI),
        (30, UnitOfPressure.INHG, 101.59167, UnitOfPressure.KPA),
        (30, UnitOfPressure.INHG, 1015.9167, UnitOfPressure.HPA),
        (30, UnitOfPressure.INHG, 101591.67, UnitOfPressure.PA),
        (30, UnitOfPressure.INHG, 1015.9167, UnitOfPressure.MBAR),
        (30, UnitOfPressure.INHG, 101.59167, UnitOfPressure.CBAR),
        (30, UnitOfPressure.INHG, 762, UnitOfPressure.MMHG),
        (30, UnitOfPressure.MMHG, 0.580103, UnitOfPressure.PSI),
        (30, UnitOfPressure.MMHG, 3.99967, UnitOfPressure.KPA),
        (30, UnitOfPressure.MMHG, 39.9967, UnitOfPressure.HPA),
        (30, UnitOfPressure.MMHG, 3999.67, UnitOfPressure.PA),
        (30, UnitOfPressure.MMHG, 39.9967, UnitOfPressure.MBAR),
        (30, UnitOfPressure.MMHG, 3.99967, UnitOfPressure.CBAR),
        (30, UnitOfPressure.MMHG, 1.181102, UnitOfPressure.INHG),
        (5, UnitOfPressure.BAR, 72.51887, UnitOfPressure.PSI),
    ],
    SpeedConverter: [
        (5, UnitOfSpeed.KILOMETERS_PER_HOUR, 3.106856, UnitOfSpeed.MILES_PER_HOUR),
        (5, UnitOfSpeed.MILES_PER_HOUR, 8.04672, UnitOfSpeed.KILOMETERS_PER_HOUR),
        (5, UnitOfVolumetricFlux.INCHES_PER_DAY, 127, UnitOfVolumetricFlux.MILLIMETERS_PER_DAY),
        (5, UnitOfVolumetricFlux.MILLIMETERS_PER_DAY, 0.1968504, UnitOfVolumetricFlux.INCHES_PER_DAY),
        (48, UnitOfVolumetricFlux.MILLIMETERS_PER_DAY, 2, UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR),
        (5, UnitOfVolumetricFlux.INCHES_PER_HOUR, 3048, UnitOfVolumetricFlux.MILLIMETERS_PER_DAY),
        (5, UnitOfSpeed.METERS_PER_SECOND, 708661.42, UnitOfVolumetricFlux.INCHES_PER_HOUR),
        (5, UnitOfSpeed.METERS_PER_SECOND, 5000, UnitOfSpeed.MILLIMETERS_PER_SECOND),
        (5, UnitOfSpeed.METERS_PER_SECOND, 5 / 0.0254, UnitOfSpeed.INCHES_PER_SECOND),
        (5000, UnitOfVolumetricFlux.INCHES_PER_HOUR, 0.0352778, UnitOfSpeed.METERS_PER_SECOND),
        (5, UnitOfSpeed.KNOTS, 2.57222, UnitOfSpeed.METERS_PER_SECOND),
        (5, UnitOfSpeed.FEET_PER_SECOND, 1.524, UnitOfSpeed.METERS_PER_SECOND),
        (20.7, UnitOfSpeed.METERS_PER_SECOND, 8.0, UnitOfSpeed.BEAUFORT),
    ],
    TemperatureConverter: [
        (100, UnitOfTemperature.CELSIUS, 212, UnitOfTemperature.FAHRENHEIT),
        (100, UnitOfTemperature.CELSIUS, 373.15, UnitOfTemperature.KELVIN),
        (100, UnitOfTemperature.FAHRENHEIT, 37.7778, UnitOfTemperature.CELSIUS),
        (100, UnitOfTemperature.FAHRENHEIT, 310.9277, UnitOfTemperature.KELVIN),
        (100, UnitOfTemperature.KELVIN, -173.15, UnitOfTemperature.CELSIUS),
        (100, UnitOfTemperature.KELVIN, -279.6699, UnitOfTemperature.FAHRENHEIT),
    ],
    UnitlessRatioConverter: [
        (5, None, 500, PERCENTAGE),
        (5, None, 5000000000, CONCENTRATION_PARTS_PER_BILLION),
        (5, None, 5000000, CONCENTRATION_PARTS_PER_MILLION),
        (5, PERCENTAGE, 0.05, None),
    ],
    VolumeConverter: [
        (5, UnitOfVolume.LITERS, 1.32086, UnitOfVolume.GALLONS),
        (5, UnitOfVolume.GALLONS, 18.92706, UnitOfVolume.LITERS),
        (5, UnitOfVolume.CUBIC_METERS, 176.5733335, UnitOfVolume.CUBIC_FEET),
        (500, UnitOfVolume.CUBIC_FEET, 14.1584233, UnitOfVolume.CUBIC_METERS),
        (500, UnitOfVolume.CUBIC_FEET, 14.1584233, UnitOfVolume.CUBIC_METERS),
        (500, UnitOfVolume.CUBIC_FEET, 478753.2467, UnitOfVolume.FLUID_OUNCES),
        (500, UnitOfVolume.CUBIC_FEET, 3740.25974, UnitOfVolume.GALLONS),
        (500, UnitOfVolume.CUBIC_FEET, 14158.42329599, UnitOfVolume.LITERS),
        (500, UnitOfVolume.CUBIC_FEET, 14158423.29599, UnitOfVolume.MILLILITERS),
        (500, UnitOfVolume.CUBIC_METERS, 500, UnitOfVolume.CUBIC_METERS),
        (500, UnitOfVolume.CUBIC_METERS, 16907011.35, UnitOfVolume.FLUID_OUNCES),
        (500, UnitOfVolume.CUBIC_METERS, 132086.02617, UnitOfVolume.GALLONS),
        (500, UnitOfVolume.CUBIC_METERS, 500000, UnitOfVolume.LITERS),
        (500, UnitOfVolume.CUBIC_METERS, 500000000, UnitOfVolume.MILLILITERS),
        (500, UnitOfVolume.FLUID_OUNCES, 0.52218967, UnitOfVolume.CUBIC_FEET),
        (500, UnitOfVolume.FLUID_OUNCES, 0.014786764, UnitOfVolume.CUBIC_METERS),
        (500, UnitOfVolume.FLUID_OUNCES, 3.90625, UnitOfVolume.GALLONS),
        (500, UnitOfVolume.FLUID_OUNCES, 14.786764, UnitOfVolume.LITERS),
        (500, UnitOfVolume.FLUID_OUNCES, 14786.764, UnitOfVolume.MILLILITERS),
        (500, UnitOfVolume.GALLONS, 66.84027, UnitOfVolume.CUBIC_FEET),
        (500, UnitOfVolume.GALLONS, 1.892706, UnitOfVolume.CUBIC_METERS),
        (500, UnitOfVolume.GALLONS, 64000, UnitOfVolume.FLUID_OUNCES),
        (500, UnitOfVolume.GALLONS, 1892.70589, UnitOfVolume.LITERS),
        (500, UnitOfVolume.GALLONS, 1892705.89, UnitOfVolume.MILLILITERS),
        (500, UnitOfVolume.LITERS, 17.65733, UnitOfVolume.CUBIC_FEET),
        (500, UnitOfVolume.LITERS, 0.5, UnitOfVolume.CUBIC_METERS),
        (500, UnitOfVolume.LITERS, 16907.011, UnitOfVolume.FLUID_OUNCES),
        (500, UnitOfVolume.LITERS, 132.086, UnitOfVolume.GALLONS),
        (500, UnitOfVolume.LITERS, 500000, UnitOfVolume.MILLILITERS),
        (500, UnitOfVolume.MILLILITERS, 0.01765733, UnitOfVolume.CUBIC_FEET),
        (500, UnitOfVolume.MILLILITERS, 0.0005, UnitOfVolume.CUBIC_METERS),
        (500, UnitOfVolume.MILLILITERS, 16.907, UnitOfVolume.FLUID_OUNCES),
        (500, UnitOfVolume.MILLILITERS, 0.132086, UnitOfVolume.GALLONS),
        (500, UnitOfVolume.MILLILITERS, 0.5, UnitOfVolume.LITERS),
        (5, UnitOfVolume.CENTUM_CUBIC_FEET, 500, UnitOfVolume.CUBIC_FEET),
        (5, UnitOfVolume.CENTUM_CUBIC_FEET, 14.15842, UnitOfVolume.CUBIC_METERS),
        (5, UnitOfVolume.CENTUM_CUBIC_FEET, 478753.24, UnitOfVolume.FLUID_OUNCES),
        (5, UnitOfVolume.CENTUM_CUBIC_FEET, 3740.26, UnitOfVolume.GALLONS),
        (5, UnitOfVolume.CENTUM_CUBIC_FEET, 14158.42, UnitOfVolume.LITERS),
    ],
    VolumeFlowRateConverter: [
        (1, UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR, 16.6666667, UnitOfVolumeFlowRate.LITERS_PER_MINUTE),
        (1, UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR, 0.58857777, UnitOfVolumeFlowRate.CUBIC_FEET_PER_MINUTE),
        (1, UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR, 4.40286754, UnitOfVolumeFlowRate.GALLONS_PER_MINUTE),
        (1, UnitOfVolumeFlowRate.LITERS_PER_MINUTE, 0.06, UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR),
        (1, UnitOfVolumeFlowRate.LITERS_PER_MINUTE, 0.03531466, UnitOfVolumeFlowRate.CUBIC_FEET_PER_MINUTE),
        (1, UnitOfVolumeFlowRate.LITERS_PER_MINUTE, 0.264172052, UnitOfVolumeFlowRate.GALLONS_PER_MINUTE),
        (1, UnitOfVolumeFlowRate.CUBIC_FEET_PER_MINUTE, 1.69901079, UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR),
        (1, UnitOfVolumeFlowRate.CUBIC_FEET_PER_MINUTE, 28.3168465, UnitOfVolumeFlowRate.LITERS_PER_MINUTE),
        (1, UnitOfVolumeFlowRate.CUBIC_FEET_PER_MINUTE, 7.48051948, UnitOfVolumeFlowRate.GALLONS_PER_MINUTE),
        (9, UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR, 2500, UnitOfVolumeFlowRate.MILLILITERS_PER_SECOND),
        (3, UnitOfVolumeFlowRate.LITERS_PER_MINUTE, 50, UnitOfVolumeFlowRate.MILLILITERS_PER_SECOND),
    ],
}


@pytest.mark.parametrize(
    "converter",
    [
        obj
        for _, obj in inspect.getmembers(unit_conversion)
        if inspect.isclass(obj)
        and issubclass(obj, BaseUnitConverter)
        and (obj != BaseUnitConverter)
    ],
)
def test_all_converters(converter: Type[BaseUnitConverter]) -> None:
    """Ensure all unit converters are tested."""
    assert converter in _ALL_CONVERTERS, "converter is not present in _ALL_CONVERTERS"
    assert converter in _GET_UNIT_RATIO, "converter is not present in _GET_UNIT_RATIO"
    unit_ratio_item = _GET_UNIT_RATIO[converter]
    assert unit_ratio_item[0] != unit_ratio_item[1], "ratio units should be different"
    assert converter in _CONVERTED_VALUE, "converter is not present in _CONVERTED_VALUE"
    converted_value_items = _CONVERTED_VALUE[converter]
    for valid_unit in converter.VALID_UNITS:
        assert any((item for item in converted_value_items if valid_unit in {item[1], item[3]})), f"Unit `{valid_unit}` is not tested in _CONVERTED_VALUE"


@pytest.mark.parametrize(
    ("converter", "valid_unit"),
    [
        (converter, valid_unit)
        for converter, valid_units in _ALL_CONVERTERS.items()
        for valid_unit in valid_units
    ],
)
def test_convert_same_unit(converter: Type[BaseUnitConverter], valid_unit: Optional[str]) -> None:
    """Test conversion from any valid unit to same unit."""
    assert converter.convert(2, valid_unit, valid_unit) == 2


@pytest.mark.parametrize(
    ("converter", "valid_unit"),
    [
        (converter, valid_unit)
        for converter, valid_units in _ALL_CONVERTERS.items()
        for valid_unit in valid_units
    ],
)
def test_convert_invalid_unit(converter: Type[BaseUnitConverter], valid_unit: Optional[str]) -> None:
    """Test exception is thrown for invalid units."""
    with pytest.raises(HomeAssistantError, match="is not a recognized .* unit"):
        converter.convert(5, INVALID_SYMBOL, valid_unit)
    with pytest.raises(HomeAssistantError, match="is not a recognized .* unit"):
        converter.convert(5, valid_unit, INVALID_SYMBOL)


@pytest.mark.parametrize(
    ("converter", "from_unit", "to_unit"),
    [
        (converter, valid_units[0], valid_units[1])
        for converter, valid_units in _ALL_CONVERTERS.items()
    ],
)
def test_convert_nonnumeric_value(
    converter: Type[BaseUnitConverter], from_unit: Optional[str], to_unit: Optional[str]
) -> None:
    """Test exception is thrown for nonnumeric type."""
    with pytest.raises(TypeError):
        converter.convert("a", from_unit, to_unit)


@pytest.mark.parametrize(
    ("converter", "from_unit", "to_unit", "expected"),
    [
        (converter, item[0], item[1], item[2])
        for converter, item in _GET_UNIT_RATIO.items()
    ],
)
def test_get_unit_ratio(
    converter: Type[BaseUnitConverter],
    from_unit: Optional[str],
    to_unit: Optional[str],
    expected: float,
) -> None:
    """Test unit ratio."""
    ratio = converter.get_unit_ratio(from_unit, to_unit)
    assert ratio == pytest.approx(expected)
    assert converter.get_unit_ratio(to_unit, from_unit) == pytest.approx(1 / ratio)


@pytest.mark.parametrize(
    ("converter", "value", "from_unit", "expected", "to_unit"),
    [
        (converter, list_item[0], list_item[1], list_item[2], list_item[3])
        for converter, item in _CONVERTED_VALUE.items()
        for list_item in item
    ],
)
def test_unit_conversion(
    converter: Type[BaseUnitConverter],
    value: float,
    from_unit: Optional[str],
    expected: float,
    to_unit: Optional[str],
) -> None:
    """Test conversion to other units."""
    assert converter.convert(value, from_unit, to_unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("converter", "value", "from_unit", "expected", "to_unit"),
    [
        (converter, value, from_unit, expected, to_unit)
        for converter, item in _CONVERTED_VALUE.items()
        for value, from_unit, expected, to_unit in item
    ],
)
def test_unit_conversion_factory(
    converter: Type[BaseUnitConverter],
    value: float,
    from_unit: Optional[str],
    expected: float,
    to_unit: Optional[str],
) -> None:
    """Test conversion to other units."""
    assert converter.converter_factory(from_unit, to_unit)(value) == pytest.approx(expected)


def test_unit_conversion_factory_allow_none_with_none() -> None:
    """Test test_unit_conversion_factory_allow_none with None."""
    assert SpeedConverter.converter_factory_allow_none(
        UnitOfSpeed.FEET_PER_SECOND, UnitOfSpeed.FEET_PER_SECOND
    )(1) == 1
    assert SpeedConverter.converter_factory_allow_none(
        UnitOfSpeed.FEET_PER_SECOND, UnitOfSpeed.FEET_PER_SECOND
    )(None) is None
    assert TemperatureConverter.converter_factory_allow_none(
        UnitOfTemperature.CELSIUS, UnitOfTemperature.CELSIUS
    )(1) == 1
    assert TemperatureConverter.converter_factory_allow_none(
        UnitOfTemperature.CELSIUS, UnitOfTemperature.CELSIUS
    )(None) is None


@pytest.mark.parametrize(
    ("converter", "value", "from_unit", "expected", "to_unit"),
    list(
        chain(
            [
                (converter, value, from_unit, expected, to_unit)
                for converter, item in _CONVERTED_VALUE.items()
                for value, from_unit, expected, to_unit in item
            ],
            [
                (converter, None, from_unit, None, to_unit)
                for converter, item in _CONVERTED_VALUE.items()
                for value, from_unit, expected, to_unit in item
            ],
        )
    ),
)
def test_unit_conversion_factory_allow_none(
    converter: Type[BaseUnitConverter],
    value: Optional[float],
    from_unit: Optional[str],
    expected: Optional[float],
    to_unit: Optional[str],
) -> None:
    """Test conversion to other units."""
    assert converter.converter_factory_allow_none(from_unit, to_unit)(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "from_unit", "expected", "to_unit"),
    [
        (100, UnitOfTemperature.CELSIUS, 180, UnitOfTemperature.FAHRENHEIT),
        (100, UnitOfTemperature.CELSIUS, 100, UnitOfTemperature.KELVIN),
        (100, UnitOfTemperature.FAHRENHEIT, 55.5556, UnitOfTemperature.CELSIUS),
        (100, UnitOfTemperature.FAHRENHEIT, 55.5556, UnitOfTemperature.KELVIN),
        (100, UnitOfTemperature.KELVIN, 100, UnitOfTemperature.CELSIUS),
        (100, UnitOfTemperature.KELVIN, 180, UnitOfTemperature.FAHRENHEIT),
    ],
)
def test_temperature_convert_with_interval(
    value: float, from_unit: Optional[str], expected: float, to_unit: Optional[str]
) -> None:
    """Test conversion to other units."""
    expected = pytest.approx(expected)
    assert TemperatureConverter.convert_interval(value, from_unit, to_unit) == expected
