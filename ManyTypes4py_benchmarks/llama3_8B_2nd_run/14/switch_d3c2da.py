from __future__ import annotations
import logging
from typing import Any

class FritzBoxBaseCoordinatorSwitch(CoordinatorEntity[AvmWrapper], SwitchEntity):
    """Fritz switch coordinator base class."""
    _attr_has_entity_name: bool = True

    def __init__(self, avm_wrapper: AvmWrapper, device_name: str, description: SwitchEntityDescription):
        """Init device info class."""
        super().__init__(avm_wrapper)
        self.entity_description = description
        self._device_name: str = device_name
        self._attr_unique_id: str = f'{avm_wrapper.unique_id}-{description.key}'

    # ...

class FritzBoxPortSwitch(FritzBoxBaseSwitch):
    """Defines a FRITZ!Box Tools PortForward switch."""
    def __init__(self, avm_wrapper: AvmWrapper, device_friendly_name: str, port_mapping: Any, port_name: str, idx: int, connection_type: str):
        """Init Fritzbox port switch."""
        self._avm_wrapper: AvmWrapper = avm_wrapper
        self._attributes: dict = {}
        self.connection_type: str = connection_type
        self.port_mapping: Any = port_mapping
        self._idx: int = idx
        # ...

class FritzBoxDeflectionSwitch(FritzBoxBaseCoordinatorSwitch):
    """Defines a FRITZ!Box Tools PortForward switch."""
    _attr_entity_category: EntityCategory = EntityCategory.CONFIG

    def __init__(self, avm_wrapper: AvmWrapper, device_friendly_name: str, deflection_id: int):
        """Init Fritxbox Deflection class."""
        self.deflection_id: int = deflection_id
        description: SwitchEntityDescription = SwitchEntityDescription(key=f'call_deflection_{self.deflection_id}', name=f'Call deflection {self.deflection_id}', icon='mdi:phone-forward')
        super().__init__(avm_wrapper, device_friendly_name, description)

    # ...

class FritzBoxProfileSwitch(FritzDeviceBase, SwitchEntity):
    """Defines a FRITZ!Box Tools DeviceProfile switch."""
    _attr_icon: str = 'mdi:router-wireless-settings'

    def __init__(self, avm_wrapper: AvmWrapper, device: FritzDevice):
        """Init Fritz profile."""
        super().__init__(avm_wrapper, device)
        self._attr_is_on: bool = False
        self._name: str = f'{device.hostname} Internet Access'
        self._attr_unique_id: str = f'{self._mac}_internet_access'
        self._attr_entity_category: EntityCategory = EntityCategory.CONFIG
        # ...

class FritzBoxWifiSwitch(FritzBoxBaseSwitch):
    """Defines a FRITZ!Box Tools Wifi switch."""

    def __init__(self, avm_wrapper: AvmWrapper, device_friendly_name: str, network_num: int, network_data: Any):
        """Init Fritz Wifi switch."""
        self._avm_wrapper: AvmWrapper = avm_wrapper
        self._attributes: dict = {}
        self._attr_entity_category: EntityCategory = EntityCategory.CONFIG
        self._network_num: int = network_num
        switch_info: SwitchInfo = SwitchInfo(description=f'Wi-Fi {network_data["switch_name"]}', friendly_name=device_friendly_name, icon='mdi:wifi', type=SWITCH_TYPE_WIFINETWORK, callback_update=self._async_fetch_update, callback_switch=self._async_switch_on_off_executor, init_state=network_data['enabled'])
        super().__init__(self._avm_wrapper, device_friendly_name, switch_info)

    # ...
