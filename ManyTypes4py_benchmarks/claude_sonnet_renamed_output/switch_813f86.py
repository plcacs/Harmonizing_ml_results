"""Support for switches which integrates with other components."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import voluptuous as vol
from homeassistant.components.switch import (
    ENTITY_ID_FORMAT,
    PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA,
    SwitchEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_FRIENDLY_NAME,
    CONF_DEVICE_ID,
    CONF_NAME,
    CONF_SWITCHES,
    CONF_UNIQUE_ID,
    CONF_VALUE_TEMPLATE,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.device import async_device_info_to_link_from_device_id
from homeassistant.helpers.entity import async_generate_entity_id
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
    AddEntitiesCallback,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.script import Script
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CONF_TURN_OFF, CONF_TURN_ON, DOMAIN
from .template_entity import (
    TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY,
    TemplateEntity,
    rewrite_common_legacy_to_modern_conf,
)

_VALID_STATES: List[str] = [STATE_ON, STATE_OFF, "true", "false"]

SWITCH_SCHEMA = vol.All(
    cv.deprecated(ATTR_ENTITY_ID),
    vol.Schema(
        {
            vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
            vol.Required(CONF_TURN_ON): cv.SCRIPT_SCHEMA,
            vol.Required(CONF_TURN_OFF): cv.SCRIPT_SCHEMA,
            vol.Optional(ATTR_FRIENDLY_NAME): cv.string,
            vol.Optional(ATTR_ENTITY_ID): cv.entity_ids,
            vol.Optional(CONF_UNIQUE_ID): cv.string,
        }
    ).extend(TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY.schema),
)

PLATFORM_SCHEMA = SWITCH_PLATFORM_SCHEMA.extend(
    {vol.Required(CONF_SWITCHES): cv.schema_with_slug_keys(SWITCH_SCHEMA)}
)

SWITCH_CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME): cv.template,
        vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_TURN_ON): cv.SCRIPT_SCHEMA,
        vol.Optional(CONF_TURN_OFF): cv.SCRIPT_SCHEMA,
        vol.Optional(CONF_DEVICE_ID): selector.DeviceSelector(),
    }
)


async def func_9tifpxx2(
    hass: HomeAssistant, config: ConfigType
) -> List[SwitchTemplate]:
    """Create the Template switches."""
    switches: List[SwitchTemplate] = []
    for object_id, entity_config in config[CONF_SWITCHES].items():
        entity_config = rewrite_common_legacy_to_modern_conf(hass, entity_config)
        unique_id: Optional[str] = entity_config.get(CONF_UNIQUE_ID)
        switches.append(SwitchTemplate(hass, object_id, entity_config, unique_id))
    return switches


async def func_ii2xrofa(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the template switches."""
    switches: List[SwitchTemplate] = await func_9tifpxx2(hass, config)
    async_add_entities(switches)


async def func_eixt12i8(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Initialize config entry."""
    _options: Dict[str, Any] = dict(config_entry.options)
    _options.pop("template_type", None)
    validated_config: Dict[str, Any] = SWITCH_CONFIG_SCHEMA(_options)
    switch = SwitchTemplate(
        hass, None, validated_config, config_entry.entry_id
    )
    async_add_entities([switch])


@callback
def func_tzulejuf(
    hass: HomeAssistant, name: str, config: Dict[str, Any]
) -> SwitchTemplate:
    """Create a preview switch."""
    validated_config: Dict[str, Any] = SWITCH_CONFIG_SCHEMA(config | {CONF_NAME: name})
    return SwitchTemplate(hass, None, validated_config, None)


class SwitchTemplate(TemplateEntity, SwitchEntity, RestoreEntity):
    """Representation of a Template switch."""

    _attr_should_poll: bool = False

    _template: Optional[Any]
    _on_script: Optional[Script]
    _off_script: Optional[Script]
    _state: Optional[bool]
    _attr_assumed_state: bool
    _attr_device_info: Any

    def __init__(
        self,
        hass: HomeAssistant,
        object_id: Optional[str],
        config: Dict[str, Any],
        unique_id: Optional[str],
    ) -> None:
        """Initialize the Template switch."""
        super().__init__(
            hass, config=config, fallback_name=object_id, unique_id=unique_id
        )
        if object_id is not None:
            self.entity_id = async_generate_entity_id(
                ENTITY_ID_FORMAT, object_id, hass=hass
            )
        friendly_name: str = self._attr_name
        self._template: Optional[Any] = config.get(CONF_VALUE_TEMPLATE)
        self._on_script: Optional[Script] = (
            Script(
                hass,
                config.get(CONF_TURN_ON),
                friendly_name,
                DOMAIN,
            )
            if config.get(CONF_TURN_ON) is not None
            else None
        )
        self._off_script: Optional[Script] = (
            Script(
                hass,
                config.get(CONF_TURN_OFF),
                friendly_name,
                DOMAIN,
            )
            if config.get(CONF_TURN_OFF) is not None
            else None
        )
        self._state: Optional[bool] = False
        self._attr_assumed_state = self._template is None
        self._attr_device_info = async_device_info_to_link_from_device_id(
            hass, config.get(CONF_DEVICE_ID)
        )

    @callback
    def func_38t43lf8(
        self, result: Union[Any, TemplateError]
    ) -> None:
        super()._update_state(result)
        if isinstance(result, TemplateError):
            self._state = None
            return
        if isinstance(result, bool):
            self._state = result
            return
        if isinstance(result, str):
            self._state = result.lower() in ("true", STATE_ON)
            return
        self._state = False

    async def func_im8dooj3(self) -> None:
        """Register callbacks."""
        if self._template is None:
            await super().async_added_to_hass()
            state = await self.async_get_last_state()
            if state:
                self._state = state.state == STATE_ON
        await super().async_added_to_hass()

    @callback
    def func_izcu8f5g(self) -> None:
        """Set up templates."""
        if self._template is not None:
            self.add_template_attribute(
                "_state", self._template, None, self._update_state
            )
        super()._async_setup_templates()

    @property
    def func_1fdvq74o(self) -> Optional[bool]:
        """Return true if device is on."""
        return self._state

    async def func_ppfd4cd3(self, **kwargs: Any) -> None:
        """Fire the on action."""
        if self._on_script:
            await self.async_run_script(
                self._on_script, context=self._context
            )
        if self._template is None:
            self._state = True
            self.async_write_ha_state()

    async def func_q0ayqqjg(self, **kwargs: Any) -> None:
        """Fire the off action."""
        if self._off_script:
            await self.async_run_script(
                self._off_script, context=self._context
            )
        if self._template is None:
            self._state = False
            self.async_write_ha_state()
