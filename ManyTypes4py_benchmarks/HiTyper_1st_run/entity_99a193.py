"""Tuya Home Assistant Base Device Model."""
from __future__ import annotations
import base64
from dataclasses import dataclass
import json
import struct
from typing import Any, Literal, Self, overload
from tuya_sharing import CustomerDevice, Manager
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from .const import DOMAIN, LOGGER, TUYA_HA_SIGNAL_UPDATE_ENTITY, DPCode, DPType
from .util import remap_value
_DPTYPE_MAPPING = {'Bitmap': DPType.RAW, 'bitmap': DPType.RAW, 'bool': DPType.BOOLEAN, 'enum': DPType.ENUM, 'json': DPType.JSON, 'raw': DPType.RAW, 'string': DPType.STRING, 'value': DPType.INTEGER}

@dataclass
class IntegerTypeData:
    """Integer Type Data."""
    unit = None
    type = None

    @property
    def max_scaled(self):
        """Return the max scaled."""
        return self.scale_value(self.max)

    @property
    def min_scaled(self):
        """Return the min scaled."""
        return self.scale_value(self.min)

    @property
    def step_scaled(self) -> int:
        """Return the step scaled."""
        return self.step / 10 ** self.scale

    def scale_value(self, value: Union[float, int]) -> float:
        """Scale a value."""
        return value / 10 ** self.scale

    def scale_value_back(self, value: Any) -> int:
        """Return raw value for scaled."""
        return int(value * 10 ** self.scale)

    def remap_value_to(self, value: Union[float, int], to_min: int=0, to_max: int=255, reverse: bool=False):
        """Remap a value from this range to a new range."""
        return remap_value(value, self.min, self.max, to_min, to_max, reverse)

    def remap_value_from(self, value: Union[int, float], from_min: int=0, from_max: int=255, reverse: bool=False):
        """Remap a value from its current range to this range."""
        return remap_value(value, from_min, from_max, self.min, self.max, reverse)

    @classmethod
    def from_json(cls: Union[dict[str, typing.Any], dict, list[dict[str, typing.Any]]], dpcode: Union[dict[str, typing.Any], dict, list[dict[str, typing.Any]]], data: Union[str, bytes, typing.Any, None]) -> None:
        """Load JSON string and return a IntegerTypeData object."""
        if not (parsed := json.loads(data)):
            return None
        return cls(dpcode, min=int(parsed['min']), max=int(parsed['max']), scale=float(parsed['scale']), step=max(float(parsed['step']), 1), unit=parsed.get('unit'), type=parsed.get('type'))

@dataclass
class EnumTypeData:
    """Enum Type Data."""

    @classmethod
    def from_json(cls: Union[dict[str, typing.Any], dict, list[dict[str, typing.Any]]], dpcode: Union[dict[str, typing.Any], dict, list[dict[str, typing.Any]]], data: Union[str, bytes, typing.Any, None]) -> None:
        """Load JSON string and return a EnumTypeData object."""
        if not (parsed := json.loads(data)):
            return None
        return cls(dpcode, **parsed)

@dataclass
class ElectricityTypeData:
    """Electricity Type Data."""
    electriccurrent = None
    power = None
    voltage = None

    @classmethod
    def from_json(cls: Union[dict[str, typing.Any], dict, list[dict[str, typing.Any]]], data: Union[str, bytes, typing.Any, None]) -> None:
        """Load JSON string and return a ElectricityTypeData object."""
        return cls(**json.loads(data.lower()))

    @classmethod
    def from_raw(cls: Union[dict, str, list[str]], data: Union[bytes, str, int]) -> Union[str, bool, None]:
        """Decode base64 string and return a ElectricityTypeData object."""
        raw = base64.b64decode(data)
        voltage = struct.unpack('>H', raw[0:2])[0] / 10.0
        electriccurrent = struct.unpack('>L', b'\x00' + raw[2:5])[0] / 1000.0
        power = struct.unpack('>L', b'\x00' + raw[5:8])[0] / 1000.0
        return cls(electriccurrent=str(electriccurrent), power=str(power), voltage=str(voltage))

class TuyaEntity(Entity):
    """Tuya base device."""
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, device: str, device_manager: str) -> None:
        """Init TuyaHaEntity."""
        self._attr_unique_id = f'tuya.{device.id}'
        device.set_up = True
        self.device = device
        self.device_manager = device_manager

    @property
    def device_info(self) -> DeviceInfo:
        """Return a device description for device registry."""
        return DeviceInfo(identifiers={(DOMAIN, self.device.id)}, manufacturer='Tuya', name=self.device.name, model=self.device.product_name, model_id=self.device.product_id)

    @property
    def available(self):
        """Return if the device is available."""
        return self.device.online

    @overload
    def find_dpcode(self, dpcodes: Union[typing.Iterable[typing.Callable], typing.Iterable[str], str], *, prefer_function: bool=False, dptype: Union[str, typing.Callable]) -> None:
        ...

    @overload
    def find_dpcode(self, dpcodes: Union[typing.Iterable[typing.Callable], typing.Iterable[str], str], *, prefer_function: bool=False, dptype: Union[str, typing.Callable]) -> None:
        ...

    @overload
    def find_dpcode(self, dpcodes: Union[typing.Iterable[typing.Callable], typing.Iterable[str], str], *, prefer_function: bool=False) -> None:
        ...

    def find_dpcode(self, dpcodes: Union[typing.Iterable[typing.Callable], typing.Iterable[str], str], *, prefer_function: bool=False, dptype: Union[str, typing.Callable]=None) -> None:
        """Find a matching DP code available on for this device."""
        if dpcodes is None:
            return None
        if isinstance(dpcodes, str):
            dpcodes = (DPCode(dpcodes),)
        elif not isinstance(dpcodes, tuple):
            dpcodes = (dpcodes,)
        order = ['status_range', 'function']
        if prefer_function:
            order = ['function', 'status_range']
        if not dptype:
            order.append('status')
        for dpcode in dpcodes:
            for key in order:
                if dpcode not in getattr(self.device, key):
                    continue
                if dptype == DPType.ENUM and getattr(self.device, key)[dpcode].type == DPType.ENUM:
                    if not (enum_type := EnumTypeData.from_json(dpcode, getattr(self.device, key)[dpcode].values)):
                        continue
                    return enum_type
                if dptype == DPType.INTEGER and getattr(self.device, key)[dpcode].type == DPType.INTEGER:
                    if not (integer_type := IntegerTypeData.from_json(dpcode, getattr(self.device, key)[dpcode].values)):
                        continue
                    return integer_type
                if dptype not in (DPType.ENUM, DPType.INTEGER):
                    return dpcode
        return None

    def get_dptype(self, dpcode: Union[str, None, dict[str, typing.Any]], prefer_function: bool=False) -> Union[None, DPType, str]:
        """Find a matching DPCode data type available on for this device."""
        if dpcode is None:
            return None
        order = ['status_range', 'function']
        if prefer_function:
            order = ['function', 'status_range']
        for key in order:
            if dpcode in getattr(self.device, key):
                current_type = getattr(self.device, key)[dpcode].type
                try:
                    return DPType(current_type)
                except ValueError:
                    return _DPTYPE_MAPPING.get(current_type)
        return None

    async def async_added_to_hass(self):
        """Call when entity is added to hass."""
        self.async_on_remove(async_dispatcher_connect(self.hass, f'{TUYA_HA_SIGNAL_UPDATE_ENTITY}_{self.device.id}', self._handle_state_update))

    async def _handle_state_update(self, updated_status_properties):
        self.async_write_ha_state()

    def _send_command(self, commands: Union[str, typing.Sequence, tuple[str]]) -> None:
        """Send command to the device."""
        LOGGER.debug('Sending commands for device %s: %s', self.device.id, commands)
        self.device_manager.send_commands(self.device.id, commands)