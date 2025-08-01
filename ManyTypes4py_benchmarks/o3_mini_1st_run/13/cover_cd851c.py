from __future__ import annotations
import logging
from typing import Any
from scsgate.tasks import HaltRollerShutterTask, LowerRollerShutterTask, RaiseRollerShutterTask
import voluptuous as vol
from homeassistant.components.cover import PLATFORM_SCHEMA as COVER_PLATFORM_SCHEMA, CoverEntity
from homeassistant.const import CONF_DEVICES, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import CONF_SCS_ID, DOMAIN, SCSGATE_SCHEMA

PLATFORM_SCHEMA = COVER_PLATFORM_SCHEMA.extend({vol.Required(CONF_DEVICES): cv.schema_with_slug_keys(SCSGATE_SCHEMA)})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the SCSGate cover."""
    devices: dict[str, dict[str, Any]] = config.get(CONF_DEVICES)  # type: ignore
    covers: list[SCSGateCover] = []
    logger: logging.Logger = logging.getLogger(__name__)
    scsgate = hass.data[DOMAIN]
    if devices:
        for entity_info in devices.values():
            if entity_info[CONF_SCS_ID] in scsgate.devices:
                continue
            name: str = entity_info[CONF_NAME]
            scs_id: Any = entity_info[CONF_SCS_ID]
            logger.info('Adding %s scsgate.cover', name)
            cover = SCSGateCover(name=name, scs_id=scs_id, logger=logger, scsgate=scsgate)
            scsgate.add_device(cover)
            covers.append(cover)
    add_entities(covers)

class SCSGateCover(CoverEntity):
    """Representation of SCSGate cover."""
    _attr_should_poll: bool = False

    def __init__(self, scs_id: Any, name: str, logger: logging.Logger, scsgate: Any) -> None:
        """Initialize the cover."""
        self._scs_id: Any = scs_id
        self._name: str = name
        self._logger: logging.Logger = logger
        self._scsgate: Any = scsgate

    @property
    def scs_id(self) -> Any:
        """Return the SCSGate ID."""
        return self._scs_id

    @property
    def name(self) -> str:
        """Return the name of the cover."""
        return self._name

    @property
    def is_closed(self) -> bool | None:
        """Return if the cover is closed."""
        return None

    def open_cover(self, **kwargs: Any) -> None:
        """Move the cover."""
        self._scsgate.append_task(RaiseRollerShutterTask(target=self._scs_id))

    def close_cover(self, **kwargs: Any) -> None:
        """Move the cover down."""
        self._scsgate.append_task(LowerRollerShutterTask(target=self._scs_id))

    def stop_cover(self, **kwargs: Any) -> None:
        """Stop the cover."""
        self._scsgate.append_task(HaltRollerShutterTask(target=self._scs_id))

    def process_event(self, message: Any) -> None:
        """Handle a SCSGate message related with this cover."""
        self._logger.debug('Cover %s, got message %s', self._scs_id, message.toggled)