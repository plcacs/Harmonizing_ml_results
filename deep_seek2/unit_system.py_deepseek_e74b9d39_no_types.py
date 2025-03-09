"""Unit system helper class and methods."""
from __future__ import annotations
from numbers import Number
from typing import TYPE_CHECKING, Final, Dict, Set, Tuple, Optional
import voluptuous as vol
from homeassistant.const import ACCUMULATED_PRECIPITATION, AREA, LENGTH, MASS, PRESSURE, TEMPERATURE, UNIT_NOT_RECOGNIZED_TEMPLATE, VOLUME, WIND_SPEED, UnitOfArea, UnitOfLength, UnitOfMass, UnitOfPrecipitationDepth, UnitOfPressure, UnitOfSpeed, UnitOfTemperature, UnitOfVolume, UnitOfVolumetricFlux
from .unit_conversion import AreaConverter, DistanceConverter, PressureConverter, SpeedConverter, TemperatureConverter, VolumeConverter
if TYPE_CHECKING:
    from homeassistant.components.sensor import SensorDeviceClass
_CONF_UNIT_SYSTEM_IMPERIAL: Final[str] = 'imperial'
_CONF_UNIT_SYSTEM_METRIC: Final[str] = 'metric'
_CONF_UNIT_SYSTEM_US_CUSTOMARY: Final[str] = 'us_customary'
AREA_UNITS: Set[str] = AreaConverter.VALID_UNITS
LENGTH_UNITS: Set[str] = DistanceConverter.VALID_UNITS
MASS_UNITS: Set[str] = {UnitOfMass.POUNDS, UnitOfMass.OUNCES, UnitOfMass.KILOGRAMS, UnitOfMass.GRAMS}
PRESSURE_UNITS: Set[str] = PressureConverter.VALID_UNITS
VOLUME_UNITS: Set[str] = VolumeConverter.VALID_UNITS
WIND_SPEED_UNITS: Set[str] = SpeedConverter.VALID_UNITS
TEMPERATURE_UNITS: Set[str] = {UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.CELSIUS}
_VALID_BY_TYPE: Dict[str, Set[str] | Set[Optional[str]]] = {LENGTH: LENGTH_UNITS, ACCUMULATED_PRECIPITATION: LENGTH_UNITS, WIND_SPEED: WIND_SPEED_UNITS, TEMPERATURE: TEMPERATURE_UNITS, MASS: MASS_UNITS, VOLUME: VOLUME_UNITS, PRESSURE: PRESSURE_UNITS, AREA: AREA_UNITS}

def _is_valid_unit(unit, unit_type):
    """Check if the unit is valid for it's type."""
    if (units := _VALID_BY_TYPE.get(unit_type)):
        return unit in units
    return False

class UnitSystem:
    """A container for units of measure."""

    def __init__(self, name, *, accumulated_precipitation: UnitOfPrecipitationDepth, area: UnitOfArea, conversions: Dict[Tuple[Optional[SensorDeviceClass | str], Optional[str]], str], length: UnitOfLength, mass: UnitOfMass, pressure: UnitOfPressure, temperature: UnitOfTemperature, volume: UnitOfVolume, wind_speed: UnitOfSpeed):
        """Initialize the unit system object."""
        errors: str = ', '.join((UNIT_NOT_RECOGNIZED_TEMPLATE.format(unit, unit_type) for unit, unit_type in ((accumulated_precipitation, ACCUMULATED_PRECIPITATION), (area, AREA), (temperature, TEMPERATURE), (length, LENGTH), (wind_speed, WIND_SPEED), (volume, VOLUME), (mass, MASS), (pressure, PRESSURE)) if not _is_valid_unit(unit, unit_type)))
        if errors:
            raise ValueError(errors)
        self._name = name
        self.accumulated_precipitation_unit = accumulated_precipitation
        self.area_unit = area
        self.length_unit = length
        self.mass_unit = mass
        self.pressure_unit = pressure
        self.temperature_unit = temperature
        self.volume_unit = volume
        self.wind_speed_unit = wind_speed
        self._conversions = conversions

    def temperature(self, temperature, from_unit):
        """Convert the given temperature to this unit system."""
        if not isinstance(temperature, Number):
            raise TypeError(f'{temperature!s} is not a numeric value.')
        return TemperatureConverter.convert(temperature, from_unit, self.temperature_unit)

    def length(self, length, from_unit):
        """Convert the given length to this unit system."""
        if not isinstance(length, Number):
            raise TypeError(f'{length!s} is not a numeric value.')
        return DistanceConverter.convert(length, from_unit, self.length_unit)

    def accumulated_precipitation(self, precip, from_unit):
        """Convert the given length to this unit system."""
        if not isinstance(precip, Number):
            raise TypeError(f'{precip!s} is not a numeric value.')
        return DistanceConverter.convert(precip, from_unit, self.accumulated_precipitation_unit)

    def area(self, area, from_unit):
        """Convert the given area to this unit system."""
        if not isinstance(area, Number):
            raise TypeError(f'{area!s} is not a numeric value.')
        return AreaConverter.convert(area, from_unit, self.area_unit)

    def pressure(self, pressure, from_unit):
        """Convert the given pressure to this unit system."""
        if not isinstance(pressure, Number):
            raise TypeError(f'{pressure!s} is not a numeric value.')
        return PressureConverter.convert(pressure, from_unit, self.pressure_unit)

    def wind_speed(self, wind_speed, from_unit):
        """Convert the given wind_speed to this unit system."""
        if not isinstance(wind_speed, Number):
            raise TypeError(f'{wind_speed!s} is not a numeric value.')
        return SpeedConverter.convert(wind_speed, from_unit, self.wind_speed_unit)

    def volume(self, volume, from_unit):
        """Convert the given volume to this unit system."""
        if not isinstance(volume, Number):
            raise TypeError(f'{volume!s} is not a numeric value.')
        return VolumeConverter.convert(volume, from_unit, self.volume_unit)

    def as_dict(self):
        """Convert the unit system to a dictionary."""
        return {LENGTH: self.length_unit, ACCUMULATED_PRECIPITATION: self.accumulated_precipitation_unit, AREA: self.area_unit, MASS: self.mass_unit, PRESSURE: self.pressure_unit, TEMPERATURE: self.temperature_unit, VOLUME: self.volume_unit, WIND_SPEED: self.wind_speed_unit}

    def get_converted_unit(self, device_class, original_unit):
        """Return converted unit given a device class or an original unit."""
        return self._conversions.get((device_class, original_unit))

def get_unit_system(key):
    """Get unit system based on key."""
    if key == _CONF_UNIT_SYSTEM_US_CUSTOMARY:
        return US_CUSTOMARY_SYSTEM
    if key == _CONF_UNIT_SYSTEM_METRIC:
        return METRIC_SYSTEM
    raise ValueError(f'`{key}` is not a valid unit system key')

def _deprecated_unit_system(value):
    """Convert deprecated unit system."""
    if value == _CONF_UNIT_SYSTEM_IMPERIAL:
        return _CONF_UNIT_SYSTEM_US_CUSTOMARY
    return value
validate_unit_system = vol.All(vol.Lower, _deprecated_unit_system, vol.Any(_CONF_UNIT_SYSTEM_METRIC, _CONF_UNIT_SYSTEM_US_CUSTOMARY))
METRIC_SYSTEM = UnitSystem(_CONF_UNIT_SYSTEM_METRIC, accumulated_precipitation=UnitOfPrecipitationDepth.MILLIMETERS, conversions={**{('atmospheric_pressure', unit): UnitOfPressure.HPA for unit in UnitOfPressure if unit != UnitOfPressure.HPA}, ('area', UnitOfArea.SQUARE_INCHES): UnitOfArea.SQUARE_CENTIMETERS, ('area', UnitOfArea.SQUARE_FEET): UnitOfArea.SQUARE_METERS, ('area', UnitOfArea.SQUARE_MILES): UnitOfArea.SQUARE_KILOMETERS, ('area', UnitOfArea.SQUARE_YARDS): UnitOfArea.SQUARE_METERS, ('area', UnitOfArea.ACRES): UnitOfArea.HECTARES, ('distance', UnitOfLength.FEET): UnitOfLength.METERS, ('distance', UnitOfLength.INCHES): UnitOfLength.MILLIMETERS, ('distance', UnitOfLength.MILES): UnitOfLength.KILOMETERS, ('distance', UnitOfLength.NAUTICAL_MILES): UnitOfLength.KILOMETERS, ('distance', UnitOfLength.YARDS): UnitOfLength.METERS, ('gas', UnitOfVolume.CENTUM_CUBIC_FEET): UnitOfVolume.CUBIC_METERS, ('gas', UnitOfVolume.CUBIC_FEET): UnitOfVolume.CUBIC_METERS, ('precipitation', UnitOfLength.INCHES): UnitOfLength.MILLIMETERS, ('precipitation_intensity', UnitOfVolumetricFlux.INCHES_PER_DAY): UnitOfVolumetricFlux.MILLIMETERS_PER_DAY, ('precipitation_intensity', UnitOfVolumetricFlux.INCHES_PER_HOUR): UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR, ('pressure', UnitOfPressure.PSI): UnitOfPressure.KPA, ('pressure', UnitOfPressure.INHG): UnitOfPressure.HPA, ('speed', UnitOfSpeed.FEET_PER_SECOND): UnitOfSpeed.KILOMETERS_PER_HOUR, ('speed', UnitOfSpeed.INCHES_PER_SECOND): UnitOfSpeed.MILLIMETERS_PER_SECOND, ('speed', UnitOfSpeed.MILES_PER_HOUR): UnitOfSpeed.KILOMETERS_PER_HOUR, ('speed', UnitOfVolumetricFlux.INCHES_PER_DAY): UnitOfVolumetricFlux.MILLIMETERS_PER_DAY, ('speed', UnitOfVolumetricFlux.INCHES_PER_HOUR): UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR, ('volume', UnitOfVolume.CENTUM_CUBIC_FEET): UnitOfVolume.CUBIC_METERS, ('volume', UnitOfVolume.CUBIC_FEET): UnitOfVolume.CUBIC_METERS, ('volume', UnitOfVolume.FLUID_OUNCES): UnitOfVolume.MILLILITERS, ('volume', UnitOfVolume.GALLONS): UnitOfVolume.LITERS, ('water', UnitOfVolume.CENTUM_CUBIC_FEET): UnitOfVolume.CUBIC_METERS, ('water', UnitOfVolume.CUBIC_FEET): UnitOfVolume.CUBIC_METERS, ('water', UnitOfVolume.GALLONS): UnitOfVolume.LITERS, **{('wind_speed', unit): UnitOfSpeed.KILOMETERS_PER_HOUR for unit in UnitOfSpeed if unit not in (UnitOfSpeed.KILOMETERS_PER_HOUR, UnitOfSpeed.KNOTS)}}, area=UnitOfArea.SQUARE_METERS, length=UnitOfLength.KILOMETERS, mass=UnitOfMass.GRAMS, pressure=UnitOfPressure.PA, temperature=UnitOfTemperature.CELSIUS, volume=UnitOfVolume.LITERS, wind_speed=UnitOfSpeed.METERS_PER_SECOND)
US_CUSTOMARY_SYSTEM = UnitSystem(_CONF_UNIT_SYSTEM_US_CUSTOMARY, accumulated_precipitation=UnitOfPrecipitationDepth.INCHES, conversions={**{('atmospheric_pressure', unit): UnitOfPressure.INHG for unit in UnitOfPressure if unit != UnitOfPressure.INHG}, ('area', UnitOfArea.SQUARE_METERS): UnitOfArea.SQUARE_FEET, ('area', UnitOfArea.SQUARE_CENTIMETERS): UnitOfArea.SQUARE_INCHES, ('area', UnitOfArea.SQUARE_MILLIMETERS): UnitOfArea.SQUARE_INCHES, ('area', UnitOfArea.SQUARE_KILOMETERS): UnitOfArea.SQUARE_MILES, ('area', UnitOfArea.HECTARES): UnitOfArea.ACRES, ('distance', UnitOfLength.CENTIMETERS): UnitOfLength.INCHES, ('distance', UnitOfLength.KILOMETERS): UnitOfLength.MILES, ('distance', UnitOfLength.METERS): UnitOfLength.FEET, ('distance', UnitOfLength.MILLIMETERS): UnitOfLength.INCHES, ('gas', UnitOfVolume.CUBIC_METERS): UnitOfVolume.CUBIC_FEET, ('precipitation', UnitOfLength.CENTIMETERS): UnitOfLength.INCHES, ('precipitation', UnitOfLength.MILLIMETERS): UnitOfLength.INCHES, ('precipitation_intensity', UnitOfVolumetricFlux.MILLIMETERS_PER_DAY): UnitOfVolumetricFlux.INCHES_PER_DAY, ('precipitation_intensity', UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR): UnitOfVolumetricFlux.INCHES_PER_HOUR, ('pressure', UnitOfPressure.MBAR): UnitOfPressure.PSI, ('pressure', UnitOfPressure.CBAR): UnitOfPressure.PSI, ('pressure', UnitOfPressure.BAR): UnitOfPressure.PSI, ('pressure', UnitOfPressure.PA): UnitOfPressure.PSI, ('pressure', UnitOfPressure.HPA): UnitOfPressure.PSI, ('pressure', UnitOfPressure.KPA): UnitOfPressure.PSI, ('pressure', UnitOfPressure.MMHG): UnitOfPressure.INHG, ('speed', UnitOfSpeed.METERS_PER_SECOND): UnitOfSpeed.MILES_PER_HOUR, ('speed', UnitOfSpeed.MILLIMETERS_PER_SECOND): UnitOfSpeed.INCHES_PER_SECOND, ('speed', UnitOfSpeed.KILOMETERS_PER_HOUR): UnitOfSpeed.MILES_PER_HOUR, ('speed', UnitOfVolumetricFlux.MILLIMETERS_PER_DAY): UnitOfVolumetricFlux.INCHES_PER_DAY, ('speed', UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR): UnitOfVolumetricFlux.INCHES_PER_HOUR, ('volume', UnitOfVolume.CUBIC_METERS): UnitOfVolume.CUBIC_FEET, ('volume', UnitOfVolume.LITERS): UnitOfVolume.GALLONS, ('volume', UnitOfVolume.MILLILITERS): UnitOfVolume.FLUID_OUNCES, ('water', UnitOfVolume.CUBIC_METERS): UnitOfVolume.CUBIC_FEET, ('water', UnitOfVolume.LITERS): UnitOfVolume.GALLONS, **{('wind_speed', unit): UnitOfSpeed.MILES_PER_HOUR for unit in UnitOfSpeed if unit not in (UnitOfSpeed.KNOTS, UnitOfSpeed.MILES_PER_HOUR)}}, area=UnitOfArea.SQUARE_FEET, length=UnitOfLength.MILES, mass=UnitOfMass.POUNDS, pressure=UnitOfPressure.PSI, temperature=UnitOfTemperature.FAHRENHEIT, volume=UnitOfVolume.GALLONS, wind_speed=UnitOfSpeed.MILES_PER_HOUR)
IMPERIAL_SYSTEM = US_CUSTOMARY_SYSTEM
'IMPERIAL_SYSTEM is deprecated. Please use US_CUSTOMARY_SYSTEM instead.'