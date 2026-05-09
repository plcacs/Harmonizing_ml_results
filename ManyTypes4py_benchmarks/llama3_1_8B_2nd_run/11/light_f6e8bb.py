"""Support for the Philips Hue lights."""
from __future__ import annotations
import asyncio
from datetime import timedelta
from functools import partial
import logging
import random
import aiohue
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_EFFECT,
    ATTR_FLASH,
    ATTR_HS_COLOR,
    ATTR_TRANSITION,
    DEFAULT_MAX_KELVIN,
    DEFAULT_MIN_KELVIN,
    EFFECT_COLORLOOP,
    EFFECT_RANDOM,
    FLASH_LONG,
    FLASH_SHORT,
    ColorMode,
    LightEntity,
    LightEntityFeature,
    filter_supported_color_modes,
)
from homeassistant.core import callback
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import color as color_util
from ..bridge import HueBridge
from ..const import (
    CONF_ALLOW_HUE_GROUPS,
    CONF_ALLOW_UNREACHABLE,
    DEFAULT_ALLOW_HUE_GROUPS,
    DEFAULT_ALLOW_UNREACHABLE,
    DOMAIN as HUE_DOMAIN,
    GROUP_TYPE_ENTERTAINMENT,
    GROUP_TYPE_LIGHT_GROUP,
    GROUP_TYPE_LIGHT_SOURCE,
    GROUP_TYPE_LUMINAIRE,
    GROUP_TYPE_ROOM,
    GROUP_TYPE_ZONE,
    REQUEST_REFRESH_DELAY,
)
from .helpers import remove_devices

SCAN_INTERVAL: timedelta = timedelta(seconds=5)
LOGGER: logging.Logger = logging.getLogger(__name__)

COLOR_MODES_HUE_ON_OFF: set[ColorMode] = {ColorMode.ONOFF}
COLOR_MODES_HUE_DIMMABLE: set[ColorMode] = {ColorMode.BRIGHTNESS}
COLOR_MODES_HUE_COLOR_TEMP: set[ColorMode] = {ColorMode.COLOR_TEMP}
COLOR_MODES_HUE_COLOR: set[ColorMode] = {ColorMode.HS}
COLOR_MODES_HUE_EXTENDED: set[ColorMode] = {ColorMode.COLOR_TEMP, ColorMode.HS}
COLOR_MODES_HUE: dict[str, set[ColorMode]] = {
    'Extended color light': COLOR_MODES_HUE_EXTENDED,
    'Color light': COLOR_MODES_HUE_COLOR,
    'Dimmable light': COLOR_MODES_HUE_DIMMABLE,
    'On/Off plug-in unit': COLOR_MODES_HUE_ON_OFF,
    'Color temperature light': COLOR_MODES_HUE_COLOR_TEMP,
}
SUPPORT_HUE_ON_OFF: LightEntityFeature = LightEntityFeature.FLASH | LightEntityFeature.TRANSITION
SUPPORT_HUE_DIMMABLE: LightEntityFeature = SUPPORT_HUE_ON_OFF
SUPPORT_HUE_COLOR_TEMP: LightEntityFeature = SUPPORT_HUE_DIMMABLE
SUPPORT_HUE_COLOR: LightEntityFeature = SUPPORT_HUE_DIMMABLE | LightEntityFeature.EFFECT
SUPPORT_HUE_EXTENDED: LightEntityFeature = SUPPORT_HUE_COLOR_TEMP | SUPPORT_HUE_COLOR
SUPPORT_HUE: dict[str, LightEntityFeature] = {
    'Extended color light': SUPPORT_HUE_EXTENDED,
    'Color light': SUPPORT_HUE_COLOR,
    'Dimmable light': SUPPORT_HUE_DIMMABLE,
    'On/Off plug-in unit': SUPPORT_HUE_ON_OFF,
    'Color temperature light': SUPPORT_HUE_COLOR_TEMP,
}
ATTR_IS_HUE_GROUP: str = 'is_hue_group'
GAMUT_TYPE_UNAVAILABLE: str = 'None'
GROUP_MIN_API_VERSION: tuple[int, int, int] = (1, 13, 0)


async def async_setup_platform(
    hass: homeassistant.core.HomeAssistant,
    config: dict[str, dict],
    async_add_entities: Callable[[list[LightEntity]], None],
    discovery_info: dict[str, str] | None = None,
) -> None:
    """Old way of setting up Hue lights."""

async def create_light(
    item_class: type[HueLight],
    coordinator: DataUpdateCoordinator,
    bridge: HueBridge,
    is_group: bool,
    rooms: dict[str, str],
    api: aiohue.API,
    item_id: str,
) -> HueLight:
    """Create the light."""
    api_item = api[item_id]
    if is_group:
        supported_color_modes: set[ColorMode] = set()
        supported_features: LightEntityFeature = LightEntityFeature(0)
        for light_id in api_item.lights:
            if light_id not in bridge.api.lights:
                continue
            light = bridge.api.lights[light_id]
            supported_features |= SUPPORT_HUE.get(light.type, SUPPORT_HUE_EXTENDED)
            supported_color_modes.update(COLOR_MODES_HUE.get(light.type, COLOR_MODES_HUE_EXTENDED))
        supported_features = supported_features or SUPPORT_HUE_EXTENDED
        supported_color_modes = supported_color_modes or COLOR_MODES_HUE_EXTENDED
        supported_color_modes = filter_supported_color_modes(supported_color_modes)
    else:
        supported_color_modes: set[ColorMode] = COLOR_MODES_HUE.get(api_item.type, COLOR_MODES_HUE_EXTENDED)
        supported_features: LightEntityFeature = SUPPORT_HUE.get(api_item.type, SUPPORT_HUE_EXTENDED)
    return item_class(
        coordinator=coordinator,
        bridge=bridge,
        is_group=is_group,
        light=api_item,
        supported_color_modes=supported_color_modes,
        supported_features=supported_features,
        rooms=rooms,
    )


async def async_setup_entry(
    hass: homeassistant.core.HomeAssistant,
    config_entry: homeassistant.config_entries.ConfigEntry,
    async_add_entities: Callable[[list[LightEntity]], None],
) -> None:
    """Set up the Hue lights from a config entry."""
    bridge: HueBridge = hass.data[HUE_DOMAIN][config_entry.entry_id]
    api_version: tuple[int, int, int] = tuple(
        int(v) for v in bridge.api.config.apiversion.split(".")
    )
    rooms: dict[str, str] = {}
    allow_groups: bool = config_entry.options.get(
        CONF_ALLOW_HUE_GROUPS, DEFAULT_ALLOW_HUE_GROUPS
    )
    supports_groups: bool = api_version >= GROUP_MIN_API_VERSION
    if allow_groups and (not supports_groups):
        LOGGER.warning("Please update your Hue bridge to support groups")
    light_coordinator: DataUpdateCoordinator = DataUpdateCoordinator(
        hass=hass,
        logger=LOGGER,
        name="light",
        update_method=partial(async_safe_fetch, bridge, bridge.api.lights.update),
        update_interval=SCAN_INTERVAL,
        request_refresh_debouncer=Debouncer(
            bridge.hass,
            LOGGER,
            cooldown=REQUEST_REFRESH_DELAY,
            immediate=True,
        ),
    )
    await light_coordinator.async_refresh()
    if not light_coordinator.last_update_success:
        raise PlatformNotReady
    if not supports_groups:
        update_lights_without_group_support = partial(
            async_update_items,
            bridge,
            bridge.api.lights,
            {},
            async_add_entities,
            partial(
                create_light,
                HueLight,
                light_coordinator,
                False,
                rooms,
            ),
            None,
        )
        bridge.reset_jobs.append(
            light_coordinator.async_add_listener(update_lights_without_group_support)
        )
        return
    group_coordinator: DataUpdateCoordinator = DataUpdateCoordinator(
        hass=hass,
        logger=LOGGER,
        name="group",
        update_method=partial(
            async_safe_fetch, bridge, bridge.api.groups.update
        ),
        update_interval=SCAN_INTERVAL,
        request_refresh_debouncer=Debouncer(
            bridge.hass,
            LOGGER,
            cooldown=REQUEST_REFRESH_DELAY,
            immediate=True,
        ),
    )
    if allow_groups:
        update_groups = partial(
            async_update_items,
            bridge,
            bridge.api.groups,
            {},
            async_add_entities,
            partial(
                create_light,
                HueLight,
                group_coordinator,
                True,
                None,
            ),
            None,
        )
        bridge.reset_jobs.append(group_coordinator.async_add_listener(update_groups))
    cancel_update_rooms_listener: Callable[[], None] | None = None

    @callback
    def _async_update_rooms() -> None:
        """Update rooms."""
        nonlocal cancel_update_rooms_listener
        rooms.clear()
        for item_id in bridge.api.groups:
            group = bridge.api.groups[item_id]
            if group.type not in [GROUP_TYPE_ROOM, GROUP_TYPE_ZONE]:
                continue
            for light_id in group.lights:
                rooms[light_id] = group.name
        bridge.reset_jobs.remove(cancel_update_rooms_listener)
        cancel_update_rooms_listener()
        cancel_update_rooms_listener = None

    @callback
    def _setup_rooms_listener() -> None:
        nonlocal cancel_update_rooms_listener
        if cancel_update_rooms_listener is not None:
            return
        cancel_update_rooms_listener = group_coordinator.async_add_listener(
            _async_update_rooms
        )
        bridge.reset_jobs.append(cancel_update_rooms_listener)
    _setup_rooms_listener()
    await group_coordinator.async_refresh()
    update_lights_with_group_support = partial(
        async_update_items,
        bridge,
        bridge.api.lights,
        {},
        async_add_entities,
        partial(
            create_light,
            HueLight,
            light_coordinator,
            False,
            rooms,
        ),
        _setup_rooms_listener,
    )
    bridge.reset_jobs.append(
        light_coordinator.async_add_listener(update_lights_with_group_support)
    )
    update_lights_with_group_support()


async def async_safe_fetch(
    bridge: HueBridge, fetch_method: Callable[[], Awaitable[aiohue.API]]
) -> aiohue.API:
    """Safely fetch data."""
    try:
        async with asyncio.timeout(4):
            return await bridge.async_request_call(fetch_method)
    except aiohue.Unauthorized as err:
        await bridge.handle_unauthorized_error()
        raise UpdateFailed("Unauthorized") from err
    except aiohue.AiohueException as err:
        raise UpdateFailed(f"Hue error: {err}") from err


@callback
def async_update_items(
    bridge: HueBridge,
    api: aiohue.API,
    current: dict[str, HueLight],
    async_add_entities: Callable[[list[LightEntity]], None],
    create_item: Callable[[aiohue.API, str], HueLight],
    new_items_callback: Callable[[], None] | None,
) -> None:
    """Update items."""
    new_items: list[HueLight] = []
    for item_id in api:
        if item_id in current:
            continue
        current[item_id] = create_item(api, item_id)
        new_items.append(current[item_id])
    bridge.hass.async_create_task(remove_devices(bridge, api, current))
    if new_items:
        if new_items_callback:
            new_items_callback()
        async_add_entities(new_items)


def hue_brightness_to_hass(value: int) -> int:
    """Convert hue brightness 1..254 to hass format 0..255."""
    return min(255, round(value / 254 * 255))


def hass_to_hue_brightness(value: int) -> int:
    """Convert hass brightness 0..255 to hue 1..254 scale."""
    return max(1, round(value / 255 * 254))


class HueLight(CoordinatorEntity, LightEntity):
    """Representation of a Hue light."""

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        bridge: HueBridge,
        is_group: bool,
        light: aiohue.Light,
        supported_color_modes: set[ColorMode],
        supported_features: LightEntityFeature,
        rooms: dict[str, str],
    ) -> None:
        """Initialize the light."""
        super().__init__(coordinator)
        self._attr_supported_color_modes: set[ColorMode] = supported_color_modes
        self._attr_supported_features: LightEntityFeature = supported_features
        self.light = light
        self.bridge = bridge
        self.is_group = is_group
        self._rooms: dict[str, str] = rooms
        self.allow_unreachable: bool = self.bridge.config_entry.options.get(
            CONF_ALLOW_UNREACHABLE, DEFAULT_ALLOW_UNREACHABLE
        )
        self._fixed_color_mode: ColorMode | None = None
        if len(supported_color_modes) == 1:
            self._fixed_color_mode = next(iter(supported_color_modes))
        else:
            assert supported_color_modes == {ColorMode.COLOR_TEMP, ColorMode.HS}
        if is_group:
            self.is_osram: bool = False
            self.is_philips: bool = False
            self.is_innr: bool = False
            self.is_ewelink: bool = False
            self.is_livarno: bool = False
            self.is_s31litezb: bool = False
            self.gamut_typ: str = GAMUT_TYPE_UNAVAILABLE
            self.gamut: aiohue.ColorGamut | None = None
        else:
            self.is_osram: bool = light.manufacturername == "OSRAM"
            self.is_philips: bool = light.manufacturername == "Philips"
            self.is_innr: bool = light.manufacturername == "innr"
            self.is_ewelink: bool = light.manufacturername == "eWeLink"
            self.is_livarno: bool = light.modelid.startswith("_TZ3000_")
            self.is_s31litezb: bool = light.modelid == "S31 Lite zb"
            self.gamut_typ: str = light.colorgamuttype
            self.gamut: aiohue.ColorGamut | None = light.colorgamut
            LOGGER.debug("Color gamut of %s: %s", self.name, str(self.gamut))
            if light.swupdatestate == "readytoinstall":
                err = "Please check for software updates of the %s bulb in the Philips Hue App."
                LOGGER.warning(err, self.name)
            if self.gamut and (not color_util.check_valid_gamut(self.gamut)):
                err = "Color gamut of %s: %s, not valid, setting gamut to None."
                LOGGER.debug(err, self.name, str(self.gamut))
                self.gamut_typ = GAMUT_TYPE_UNAVAILABLE
                self.gamut = None

    @property
    def unique_id(self) -> str:
        """Return the unique ID of this Hue light."""
        unique_id = self.light.uniqueid
        if not unique_id and self.is_group:
            unique_id = self.light.id
        return unique_id

    @property
    def device_id(self) -> str:
        """Return the ID of this Hue light."""
        return self.unique_id

    @property
    def name(self) -> str:
        """Return the name of the Hue light."""
        return self.light.name

    @property
    def brightness(self) -> int | None:
        """Return the brightness of this light between 0..255."""
        if self.is_group:
            bri = self.light.action.get("bri")
        else:
            bri = self.light.state.get("bri")
        if bri is None:
            return bri
        return hue_brightness_to_hass(bri)

    @property
    def color_mode(self) -> ColorMode:
        """Return the color mode of the light."""
        if self._fixed_color_mode:
            return self._fixed_color_mode
        mode = self._color_mode
        if mode in ("xy", "hs"):
            return ColorMode.HS
        return ColorMode.COLOR_TEMP

    @property
    def _color_mode(self) -> str:
        """Return the hue color mode."""
        if self.is_group:
            return self.light.action.get("colormode")
        return self.light.state.get("colormode")

    @property
    def hs_color(self) -> tuple[float, float] | None:
        """Return the hs color value."""
        mode = self._color_mode
        source = self.light.action if self.is_group else self.light.state
        if mode in ("xy", "hs") and "xy" in source:
            return color_util.color_xy_to_hs(*source["xy"], self.gamut)
        return None

    @property
    def color_temp_kelvin(self) -> int | None:
        """Return the color temperature value in Kelvin."""
        if self._color_mode != "ct":
            return None
        ct = self.light.action.get("ct") if self.is_group else self.light.state.get("ct")
        return color_util.color_temperature_mired_to_kelvin(ct) if ct else None

    @property
    def max_color_temp_kelvin(self) -> int:
        """Return the coldest color_temp_kelvin that this light supports."""
        if self.is_group:
            return DEFAULT_MAX_KELVIN
        min_mireds = self.light.controlcapabilities.get("ct", {}).get("min")
        if not min_mireds:
            return DEFAULT_MAX_KELVIN
        return color_util.color_temperature_mired_to_kelvin(min_mireds)

    @property
    def min_color_temp_kelvin(self) -> int:
        """Return the warmest color_temp_kelvin that this light supports."""
        if self.is_group:
            return DEFAULT_MIN_KELVIN
        if self.is_livarno:
            return 2000
        max_mireds = self.light.controlcapabilities.get("ct", {}).get("max")
        if not max_mireds:
            return DEFAULT_MIN_KELVIN
        return color_util.color_temperature_mired_to_kelvin(max_mireds)

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        if self.is_group:
            return self.light.state["any_on"]
        return self.light.state["on"]

    @property
    def available(self) -> bool:
        """Return if light is available."""
        return self.coordinator.last_update_success and (
            self.is_group or self.allow_unreachable or self.light.state["reachable"]
        )

    @property
    def effect(self) -> str | None:
        """Return the current effect."""
        return self.light.state.get("effect", None)

    @property
    def effect_list(self) -> list[str]:
        """Return the list of supported effects."""
        if self.is_osram:
            return [EFFECT_RANDOM]
        return [EFFECT_COLORLOOP, EFFECT_RANDOM]

    @property
    def device_info(self) -> DeviceInfo | None:
        """Return the device info."""
        if self.light.type in (
            GROUP_TYPE_ENTERTAINMENT,
            GROUP_TYPE_LIGHT_GROUP,
            GROUP_TYPE_ROOM,
            GROUP_TYPE_LUMINAIRE,
            GROUP_TYPE_LIGHT_SOURCE,
            GROUP_TYPE_ZONE,
        ):
            return None
        suggested_area: str | None = None
        if self._rooms and self.light.id in self._rooms:
            suggested_area = self._rooms[self.light.id]
        return DeviceInfo(
            identifiers={(HUE_DOMAIN, self.device_id)},
            manufacturer=self.light.manufacturername,
            model=self.light.productname or self.light.modelid,
            name=self.name,
            sw_version=self.light.swversion,
            suggested_area=suggested_area,
            via_device=(HUE_DOMAIN, self.bridge.api.config.bridgeid),
        )

    async def async_turn_on(self, **kwargs: dict[str, Any]) -> None:
        """Turn the specified or all lights on."""
        command: dict[str, Any] = {"on": True}
        if ATTR_TRANSITION in kwargs:
            command["transitiontime"] = int(kwargs[ATTR_TRANSITION] * 10)
        if ATTR_HS_COLOR in kwargs:
            if self.is_osram:
                command["hue"] = int(kwargs[ATTR_HS_COLOR][0] / 360 * 65535)
                command["sat"] = int(kwargs[ATTR_HS_COLOR][1] / 100 * 255)
            else:
                xy_color = color_util.color_hs_to_xy(*kwargs[ATTR_HS_COLOR], self.gamut)
                command["xy"] = xy_color
        elif ATTR_COLOR_TEMP_KELVIN in kwargs:
            temp_k = max(
                self.min_color_temp_kelvin, min(self.max_color_temp_kelvin, kwargs[ATTR_COLOR_TEMP_KELVIN])
            )
            command["ct"] = color_util.color_temperature_kelvin_to_mired(temp_k)
        if ATTR_BRIGHTNESS in kwargs:
            command["bri"] = hass_to_hue_brightness(kwargs[ATTR_BRIGHTNESS])
        flash = kwargs.get(ATTR_FLASH)
        if flash == FLASH_LONG:
            command["alert"] = "lselect"
            del command["on"]
        elif flash == FLASH_SHORT:
            command["alert"] = "select"
            del command["on"]
        elif not self.is_innr and (not self.is_ewelink) and (not self.is_livarno) and (not self.is_s31litezb):
            command["alert"] = "none"
        if ATTR_EFFECT in kwargs:
            effect = kwargs[ATTR_EFFECT]
            if effect == EFFECT_COLORLOOP:
                command["effect"] = "colorloop"
            elif effect == EFFECT_RANDOM:
                command["hue"] = random.randrange(0, 65535)
                command["sat"] = random.randrange(150, 254)
            else:
                command["effect"] = "none"
        if self.is_group:
            await self.bridge.async_request_call(self.light.set_action, **command)
        else:
            await self.bridge.async_request_call(self.light.set_state, **command)
        await self.coordinator.async_request_refresh()

    async def async_turn_off(self, **kwargs: dict[str, Any]) -> None:
        """Turn the specified or all lights off."""
        command: dict[str, Any] = {"on": False}
        if ATTR_TRANSITION in kwargs:
            command["transitiontime"] = int(kwargs[ATTR_TRANSITION] * 10)
        flash = kwargs.get(ATTR_FLASH)
        if flash == FLASH_LONG:
            command["alert"] = "lselect"
            del command["on"]
        elif flash == FLASH_SHORT:
            command["alert"] = "select"
            del command["on"]
        elif not self.is_innr and (not self.is_livarno):
            command["alert"] = "none"
        if self.is_group:
            await self.bridge.async_request_call(self.light.set_action, **command)
        else:
            await self.bridge.async_request_call(self.light.set_state, **command)
        await self.coordinator.async_request_refresh()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the device state attributes."""
        if not self.is_group:
            return {}
        return {ATTR_IS_HUE_GROUP: self.is_group}
