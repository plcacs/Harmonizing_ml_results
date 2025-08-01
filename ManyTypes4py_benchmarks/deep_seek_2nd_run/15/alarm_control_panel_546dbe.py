"""Interfaces with Egardia/Woonveilig alarm control panel."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Set, List
import requests
from homeassistant.components.alarm_control_panel import AlarmControlPanelEntity, AlarmControlPanelEntityFeature, AlarmControlPanelState
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import CONF_REPORT_SERVER_CODES, CONF_REPORT_SERVER_ENABLED, CONF_REPORT_SERVER_PORT, EGARDIA_DEVICE, EGARDIA_SERVER, REPORT_SERVER_CODES_IGNORE

_LOGGER = logging.getLogger(__name__)
STATES: Dict[str, AlarmControlPanelState] = {
    'ARM': AlarmControlPanelState.ARMED_AWAY,
    'DAY HOME': AlarmControlPanelState.ARMED_HOME,
    'DISARM': AlarmControlPanelState.DISARMED,
    'ARMHOME': AlarmControlPanelState.ARMED_HOME,
    'HOME': AlarmControlPanelState.ARMED_HOME,
    'NIGHT HOME': AlarmControlPanelState.ARMED_NIGHT,
    'TRIGGERED': AlarmControlPanelState.TRIGGERED
}

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Egardia Alarm Control Panel platform."""
    if discovery_info is None:
        return
    device = EgardiaAlarm(
        discovery_info['name'],
        hass.data[EGARDIA_DEVICE],
        discovery_info[CONF_REPORT_SERVER_ENABLED],
        discovery_info.get(CONF_REPORT_SERVER_CODES),
        discovery_info[CONF_REPORT_SERVER_PORT]
    )
    add_entities([device], True)

class EgardiaAlarm(AlarmControlPanelEntity):
    """Representation of a Egardia alarm."""
    _attr_code_arm_required: bool = False
    _attr_supported_features: AlarmControlPanelEntityFeature = (
        AlarmControlPanelEntityFeature.ARM_HOME | AlarmControlPanelEntityFeature.ARM_AWAY
    )

    def __init__(
        self,
        name: str,
        egardiasystem: Any,
        rs_enabled: bool = False,
        rs_codes: Optional[Dict[str, List[str]]] = None,
        rs_port: int = 52010
    ) -> None:
        """Initialize the Egardia alarm."""
        self._attr_name: str = name
        self._egardiasystem: Any = egardiasystem
        self._rs_enabled: bool = rs_enabled
        self._rs_codes: Optional[Dict[str, List[str]]] = rs_codes
        self._rs_port: int = rs_port
        self._attr_alarm_state: Optional[AlarmControlPanelState] = None

    async def async_added_to_hass(self) -> None:
        """Add Egardiaserver callback if enabled."""
        if self._rs_enabled:
            _LOGGER.debug('Registering callback to Egardiaserver')
            self.hass.data[EGARDIA_SERVER].register_callback(self.handle_status_event)

    @property
    def should_poll(self) -> bool:
        """Poll if no report server is enabled."""
        if not self._rs_enabled:
            return True
        return False

    def handle_status_event(self, event: Dict[str, Any]) -> None:
        """Handle the Egardia system status event."""
        if (statuscode := event.get('status')) is not None:
            status = self.lookupstatusfromcode(statuscode)
            self.parsestatus(status)
            self.schedule_update_ha_state()

    def lookupstatusfromcode(self, statuscode: str) -> str:
        """Look at the rs_codes and returns the status from the code."""
        if self._rs_codes is None:
            return 'UNKNOWN'
        return next(
            (status_group.upper() 
             for status_group, codes in self._rs_codes.items() 
             for code in codes if statuscode == code),
            'UNKNOWN'
        )

    def parsestatus(self, status: str) -> None:
        """Parse the status."""
        _LOGGER.debug('Parsing status %s', status)
        if status.lower().strip() != REPORT_SERVER_CODES_IGNORE:
            _LOGGER.debug('Not ignoring status %s', status)
            newstatus = STATES.get(status.upper())
            _LOGGER.debug('newstatus %s', newstatus)
            self._attr_alarm_state = newstatus
        else:
            _LOGGER.error('Ignoring status')

    def update(self) -> None:
        """Update the alarm status."""
        status = self._egardiasystem.getstate()
        self.parsestatus(status)

    def alarm_disarm(self, code: Optional[str] = None) -> None:
        """Send disarm command."""
        try:
            self._egardiasystem.alarm_disarm()
        except requests.exceptions.RequestException as err:
            _LOGGER.error('Egardia device exception occurred when sending disarm command: %s', err)

    def alarm_arm_home(self, code: Optional[str] = None) -> None:
        """Send arm home command."""
        try:
            self._egardiasystem.alarm_arm_home()
        except requests.exceptions.RequestException as err:
            _LOGGER.error('Egardia device exception occurred when sending arm home command: %s', err)

    def alarm_arm_away(self, code: Optional[str] = None) -> None:
        """Send arm away command."""
        try:
            self._egardiasystem.alarm_arm_away()
        except requests.exceptions.RequestException as err:
            _LOGGER.error('Egardia device exception occurred when sending arm away command: %s', err)
