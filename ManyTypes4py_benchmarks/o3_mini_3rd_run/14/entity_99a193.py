from __future__ import annotations
import base64
from dataclasses import dataclass
import json
import struct
from typing import Any, Optional, Union, Literal, overload
from tuya_sharing import CustomerDevice, Manager
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from .const import DOMAIN, LOGGER, TUYA_HA_SIGNAL_UPDATE_ENTITY, DPCode, DPType
from .util import remap_value

_DPTYPE_MAPPING: dict[str, DPType] = {
    'Bitmap': DPType.RAW,
    'bitmap': DPType.RAW,
    'bool': DPType.BOOLEAN,
    'enum': DPType.ENUM,
    'json': DPType.JSON,
    'raw': DPType.RAW,
    'string': DPType.STRING,
    'value': DPType.INTEGER,
}


@dataclass
class IntegerTypeData:
    dpcode: DPCode
    min: int
    max: int
    scale: float
    step: float
    unit: Optional[str] = None
    type: Optional[Any] = None

    @property
    def max_scaled(self) -> float:
        return self.scale_value(self.max)

    @property
    def min_scaled(self) -> float:
        return self.scale_value(self.min)

    @property
    def step_scaled(self) -> float:
        return self.step / 10 ** self.scale

    def scale_value(self, value: Union[int, float]) -> float:
        return value / 10 ** self.scale

    def scale_value_back(self, value: float) -> int:
        return int(value * 10 ** self.scale)

    def remap_value_to(
        self, value: float, to_min: float = 0, to_max: float = 255, reverse: bool = False
    ) -> float:
        return remap_value(value, self.min, self.max, to_min, to_max, reverse)

    def remap_value_from(
        self, value: float, from_min: float = 0, from_max: float = 255, reverse: bool = False
    ) -> float:
        return remap_value(value, from_min, from_max, self.min, self.max, reverse)

    @classmethod
    def from_json(cls, dpcode: DPCode, data: str) -> Optional[IntegerTypeData]:
        parsed: Any = json.loads(data)
        if not parsed:
            return None
        return cls(
            dpcode,
            min=int(parsed['min']),
            max=int(parsed['max']),
            scale=float(parsed['scale']),
            step=max(float(parsed['step']), 1),
            unit=parsed.get('unit'),
            type=parsed.get('type'),
        )


@dataclass
class EnumTypeData:
    dpcode: DPCode
    data: dict[str, Any]

    @classmethod
    def from_json(cls, dpcode: DPCode, data: str) -> Optional[EnumTypeData]:
        parsed: Any = json.loads(data)
        if not parsed:
            return None
        return cls(dpcode, data=parsed)


@dataclass
class ElectricityTypeData:
    electriccurrent: Optional[str] = None
    power: Optional[str] = None
    voltage: Optional[str] = None

    @classmethod
    def from_json(cls, data: str) -> ElectricityTypeData:
        return cls(**json.loads(data.lower()))

    @classmethod
    def from_raw(cls, data: str) -> ElectricityTypeData:
        raw: bytes = base64.b64decode(data)
        voltage_val: float = struct.unpack('>H', raw[0:2])[0] / 10.0
        electriccurrent_val: float = struct.unpack('>L', b'\x00' + raw[2:5])[0] / 1000.0
        power_val: float = struct.unpack('>L', b'\x00' + raw[5:8])[0] / 1000.0
        return cls(
            electriccurrent=str(electriccurrent_val),
            power=str(power_val),
            voltage=str(voltage_val),
        )


class TuyaEntity(Entity):
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, device: CustomerDevice, device_manager: Manager) -> None:
        self._attr_unique_id: str = f'tuya.{device.id}'
        device.set_up = True
        self.device: CustomerDevice = device
        self.device_manager: Manager = device_manager

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self.device.id)},
            manufacturer='Tuya',
            name=self.device.name,
            model=self.device.product_name,
            model_id=self.device.product_id,
        )

    @property
    def available(self) -> bool:
        return self.device.online

    @overload
    def find_dpcode(
        self, dpcodes: Union[str, DPCode], *, prefer_function: bool = False, dptype: Literal[DPType.ENUM]
    ) -> Optional[EnumTypeData]:
        ...

    @overload
    def find_dpcode(
        self, dpcodes: Union[str, DPCode], *, prefer_function: bool = False, dptype: Literal[DPType.INTEGER]
    ) -> Optional[IntegerTypeData]:
        ...

    @overload
    def find_dpcode(
        self, dpcodes: Union[str, DPCode, tuple[Union[str, DPCode], ...]], *, prefer_function: bool = False
    ) -> Optional[Union[EnumTypeData, IntegerTypeData, DPCode]]:
        ...

    def find_dpcode(
        self,
        dpcodes: Union[str, DPCode, tuple[Union[str, DPCode], ...], None],
        *,
        prefer_function: bool = False,
        dptype: Optional[DPType] = None,
    ) -> Optional[Union[EnumTypeData, IntegerTypeData, DPCode]]:
        if dpcodes is None:
            return None
        if isinstance(dpcodes, str):
            dpcodes = (DPCode(dpcodes),)
        elif not isinstance(dpcodes, tuple):
            dpcodes = (dpcodes,)
        order: list[str] = ['status_range', 'function']
        if prefer_function:
            order = ['function', 'status_range']
        if not dptype:
            order.append('status')
        for dpcode in dpcodes:
            for key in order:
                if dpcode not in getattr(self.device, key):
                    continue
                if dptype == DPType.ENUM and getattr(self.device, key)[dpcode].type == DPType.ENUM:
                    enum_type: Optional[EnumTypeData] = EnumTypeData.from_json(dpcode, getattr(self.device, key)[dpcode].values)
                    if not enum_type:
                        continue
                    return enum_type
                if dptype == DPType.INTEGER and getattr(self.device, key)[dpcode].type == DPType.INTEGER:
                    integer_type: Optional[IntegerTypeData] = IntegerTypeData.from_json(dpcode, getattr(self.device, key)[dpcode].values)
                    if not integer_type:
                        continue
                    return integer_type
                if dptype not in (DPType.ENUM, DPType.INTEGER):
                    return dpcode
        return None

    def get_dptype(self, dpcode: DPCode, prefer_function: bool = False) -> Optional[DPType]:
        if dpcode is None:
            return None
        order: list[str] = ['status_range', 'function']
        if prefer_function:
            order = ['function', 'status_range']
        for key in order:
            if dpcode in getattr(self.device, key):
                current_type: Any = getattr(self.device, key)[dpcode].type
                try:
                    return DPType(current_type)
                except ValueError:
                    return _DPTYPE_MAPPING.get(current_type)
        return None

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, f'{TUYA_HA_SIGNAL_UPDATE_ENTITY}_{self.device.id}', self._handle_state_update
            )
        )

    async def _handle_state_update(self, updated_status_properties: Any) -> None:
        self.async_write_ha_state()

    def _send_command(self, commands: list[dict[str, Any]]) -> None:
        LOGGER.debug('Sending commands for device %s: %s', self.device.id, commands)
        self.device_manager.send_commands(self.device.id, commands)