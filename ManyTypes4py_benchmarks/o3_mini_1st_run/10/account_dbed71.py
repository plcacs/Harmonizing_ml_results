from __future__ import annotations
from datetime import timedelta
import logging
import operator
from typing import Any, Dict, List, Optional
from pyicloud import PyiCloudService
from pyicloud.exceptions import PyiCloudFailedLoginException, PyiCloudNoDevicesException, PyiCloudServiceNotActivatedException
from pyicloud.services.findmyiphone import AppleDevice
from homeassistant.components.zone import async_active_zone
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_USERNAME
from homeassistant.core import CALLBACK_TYPE, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.dispatcher import dispatcher_send
from homeassistant.helpers.event import track_point_in_utc_time
from homeassistant.helpers.storage import Store
from homeassistant.util import slugify
from homeassistant.util.async_ import run_callback_threadsafe
from homeassistant.util.dt import utcnow
from homeassistant.util.location import distance
from .const import DEVICE_BATTERY_LEVEL, DEVICE_BATTERY_STATUS, DEVICE_CLASS, DEVICE_DISPLAY_NAME, DEVICE_ID, DEVICE_LOCATION, DEVICE_LOCATION_HORIZONTAL_ACCURACY, DEVICE_LOCATION_LATITUDE, DEVICE_LOCATION_LONGITUDE, DEVICE_LOST_MODE_CAPABLE, DEVICE_LOW_POWER_MODE, DEVICE_NAME, DEVICE_PERSON_ID, DEVICE_RAW_DEVICE_MODEL, DEVICE_STATUS, DEVICE_STATUS_CODES, DEVICE_STATUS_SET, DOMAIN

ATTR_ACCOUNT_FETCH_INTERVAL = 'account_fetch_interval'
ATTR_BATTERY = 'battery'
ATTR_BATTERY_STATUS = 'battery_status'
ATTR_DEVICE_NAME = 'device_name'
ATTR_DEVICE_STATUS = 'device_status'
ATTR_LOW_POWER_MODE = 'low_power_mode'
ATTR_OWNER_NAME = 'owner_fullname'
SERVICE_ICLOUD_PLAY_SOUND = 'play_sound'
SERVICE_ICLOUD_DISPLAY_MESSAGE = 'display_message'
SERVICE_ICLOUD_LOST_DEVICE = 'lost_device'
SERVICE_ICLOUD_UPDATE = 'update'
ATTR_ACCOUNT = 'account'
ATTR_LOST_DEVICE_MESSAGE = 'message'
ATTR_LOST_DEVICE_NUMBER = 'number'
ATTR_LOST_DEVICE_SOUND = 'sound'
_LOGGER = logging.getLogger(__name__)


class IcloudAccount:
    """Representation of an iCloud account."""

    def __init__(
        self,
        hass: HomeAssistant,
        username: str,
        password: str,
        icloud_dir: Store,
        with_family: bool,
        max_interval: int,
        gps_accuracy_threshold: float,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize an iCloud account."""
        self.hass: HomeAssistant = hass
        self._username: str = username
        self._password: str = password
        self._with_family: bool = with_family
        self._fetch_interval: int = max_interval
        self._max_interval: int = max_interval
        self._gps_accuracy_threshold: float = gps_accuracy_threshold
        self._icloud_dir: Store = icloud_dir
        self.api: Optional[PyiCloudService] = None
        self._owner_fullname: Optional[str] = None
        self._family_members_fullname: Dict[str, str] = {}
        self._devices: Dict[str, IcloudDevice] = {}
        self._retried_fetch: bool = False
        self._config_entry: ConfigEntry = config_entry
        self.listeners: List[CALLBACK_TYPE] = []

    def setup(self) -> None:
        """Set up an iCloud account."""
        try:
            self.api = PyiCloudService(
                self._username, self._password, self._icloud_dir.path, with_family=self._with_family
            )
            if self.api.requires_2fa:
                raise PyiCloudFailedLoginException
        except PyiCloudFailedLoginException:
            self.api = None
            _LOGGER.error(
                "Your password for '%s' is no longer working; Go to the Integrations menu and click on Configure on the discovered Apple iCloud card to login again",
                self._config_entry.data[CONF_USERNAME],
            )
            self._require_reauth()
            return
        try:
            api_devices = self.api.devices
            user_info: Dict[str, Any] = api_devices.response["userInfo"]
        except (PyiCloudServiceNotActivatedException, PyiCloudNoDevicesException) as err:
            _LOGGER.error("No iCloud device found")
            raise ConfigEntryNotReady from err
        self._owner_fullname = f'{user_info["firstName"]} {user_info["lastName"]}'
        self._family_members_fullname = {}
        if user_info.get("membersInfo") is not None:
            for prs_id, member in user_info["membersInfo"].items():
                self._family_members_fullname[prs_id] = f'{member["firstName"]} {member["lastName"]}'
        self._devices = {}
        self.update_devices()

    def update_devices(self) -> None:
        """Update iCloud devices."""
        if self.api is None:
            return
        _LOGGER.debug("Updating devices")
        if self.api.requires_2fa:
            self._require_reauth()
            return
        try:
            api_devices = self.api.devices
        except Exception as err:
            _LOGGER.error("Unknown iCloud error: %s", err)
            self._fetch_interval = 2
            dispatcher_send(self.hass, self.signal_device_update)
            self._schedule_next_fetch()
            return
        new_device: bool = False
        for device in api_devices:
            status: Dict[str, Any] = device.status(DEVICE_STATUS_SET)
            device_id: str = status[DEVICE_ID]
            device_name: str = status[DEVICE_NAME]
            if status[DEVICE_BATTERY_STATUS] == "Unknown" or status.get(DEVICE_BATTERY_LEVEL) is None:
                continue
            if self._devices.get(device_id) is not None:
                _LOGGER.debug("Updating iCloud device: %s", device_name)
                self._devices[device_id].update(status)
            else:
                _LOGGER.debug("Adding iCloud device: %s [model: %s]", device_name, status[DEVICE_RAW_DEVICE_MODEL])
                self._devices[device_id] = IcloudDevice(self, device, status)
                self._devices[device_id].update(status)
                new_device = True
        # Checking status of the first device in api_devices; assuming api_devices is indexable
        first_device_status: Dict[str, Any] = list(api_devices)[0].status(DEVICE_STATUS_SET)
        if DEVICE_STATUS_CODES.get(first_device_status[DEVICE_STATUS]) == "pending" and (not self._retried_fetch):
            _LOGGER.debug("Pending devices, trying again in 15s")
            self._fetch_interval = 0.25
            self._retried_fetch = True
        else:
            self._fetch_interval = self._determine_interval()
            self._retried_fetch = False
        dispatcher_send(self.hass, self.signal_device_update)
        if new_device:
            dispatcher_send(self.hass, self.signal_device_new)
        self._schedule_next_fetch()

    def _require_reauth(self) -> None:
        """Require the user to log in again."""
        self.hass.add_job(self._config_entry.async_start_reauth, self.hass)

    def _determine_interval(self) -> int:
        """Calculate new interval between two API fetch (in minutes)."""
        intervals: Dict[str, int] = {"default": self._max_interval}
        for device in self._devices.values():
            if device.location is None:
                continue
            current_zone = run_callback_threadsafe(
                self.hass.loop,
                async_active_zone,
                self.hass,
                device.location[DEVICE_LOCATION_LATITUDE],
                device.location[DEVICE_LOCATION_LONGITUDE],
                device.location[DEVICE_LOCATION_HORIZONTAL_ACCURACY],
            ).result()
            if current_zone is not None:
                continue
            zones = (self.hass.states.get(entity_id) for entity_id in sorted(self.hass.states.entity_ids("zone")))
            distances: List[float] = []
            for zone_state in zones:
                if zone_state is None:
                    continue
                zone_state_lat = zone_state.attributes[DEVICE_LOCATION_LATITUDE]
                zone_state_long = zone_state.attributes[DEVICE_LOCATION_LONGITUDE]
                zone_distance = distance(
                    device.location[DEVICE_LOCATION_LATITUDE],
                    device.location[DEVICE_LOCATION_LONGITUDE],
                    zone_state_lat,
                    zone_state_long,
                )
                if zone_distance is not None:
                    distances.append(round(zone_distance / 1000, 1))
            if not distances:
                continue
            mindistance: float = min(distances)
            interval: int = round(mindistance / 2)
            interval = max(interval, 1)
            if interval > 180:
                interval = self._max_interval
            if device.battery_level is not None and device.battery_level <= 33 and (mindistance > 3):
                interval = interval * 2
            intervals[device.name] = interval
        return max(int(min(intervals.items(), key=operator.itemgetter(1))[1]), self._max_interval)

    def _schedule_next_fetch(self) -> None:
        if not self._config_entry.pref_disable_polling:
            track_point_in_utc_time(self.hass, self.keep_alive, utcnow() + timedelta(minutes=self._fetch_interval))

    def keep_alive(self, now: Optional[Any] = None) -> None:
        """Keep the API alive."""
        if self.api is None:
            self.setup()
        if self.api is None:
            return
        self.api.authenticate()
        self.update_devices()

    def get_devices_with_name(self, name: str) -> List[IcloudDevice]:
        """Get devices by name."""
        name_slug: str = slugify(name.replace(" ", "", 99))
        result: List[IcloudDevice] = [
            device for device in self.devices.values() if slugify(device.name.replace(" ", "", 99)) == name_slug
        ]
        if not result:
            raise ValueError(f'No device with name {name}')
        return result

    @property
    def username(self) -> str:
        """Return the account username."""
        return self._username

    @property
    def owner_fullname(self) -> Optional[str]:
        """Return the account owner fullname."""
        return self._owner_fullname

    @property
    def family_members_fullname(self) -> Dict[str, str]:
        """Return the account family members fullname."""
        return self._family_members_fullname

    @property
    def fetch_interval(self) -> int:
        """Return the account fetch interval."""
        return self._fetch_interval

    @property
    def devices(self) -> Dict[str, IcloudDevice]:
        """Return the account devices."""
        return self._devices

    @property
    def signal_device_new(self) -> str:
        """Event specific per Freebox entry to signal new device."""
        return f"{DOMAIN}-{self._username}-device-new"

    @property
    def signal_device_update(self) -> str:
        """Event specific per Freebox entry to signal updates in devices."""
        return f"{DOMAIN}-{self._username}-device-update"


class IcloudDevice:
    """Representation of a iCloud device."""
    _attr_attribution: str = "Data provided by Apple iCloud"

    def __init__(self, account: IcloudAccount, device: AppleDevice, status: Dict[str, Any]) -> None:
        """Initialize the iCloud device."""
        self._account: IcloudAccount = account
        self._device: AppleDevice = device
        self._status: Dict[str, Any] = status
        self._name: str = self._status[DEVICE_NAME]
        self._device_id: str = self._status[DEVICE_ID]
        self._device_class: str = self._status[DEVICE_CLASS]
        self._device_model: str = self._status[DEVICE_DISPLAY_NAME]
        self._battery_level: Optional[int] = None
        self._battery_status: Optional[str] = None
        self._location: Optional[Dict[str, Any]] = None
        self._attrs: Dict[str, Any] = {
            ATTR_ACCOUNT_FETCH_INTERVAL: self._account.fetch_interval,
            ATTR_DEVICE_NAME: self._device_model,
            ATTR_DEVICE_STATUS: None,
        }
        if self._status[DEVICE_PERSON_ID]:
            self._attrs[ATTR_OWNER_NAME] = account.family_members_fullname[self._status[DEVICE_PERSON_ID]]
        elif account.owner_fullname is not None:
            self._attrs[ATTR_OWNER_NAME] = account.owner_fullname

    def update(self, status: Dict[str, Any]) -> None:
        """Update the iCloud device."""
        self._status = status
        self._status[ATTR_ACCOUNT_FETCH_INTERVAL] = self._account.fetch_interval
        device_status: str = DEVICE_STATUS_CODES.get(self._status[DEVICE_STATUS], "error")
        self._attrs[ATTR_DEVICE_STATUS] = device_status
        self._battery_status = self._status[DEVICE_BATTERY_STATUS]
        self._attrs[ATTR_BATTERY_STATUS] = self._battery_status
        device_battery_level: Optional[float] = self._status.get(DEVICE_BATTERY_LEVEL, 0)
        if self._battery_status != "Unknown" and device_battery_level is not None:
            self._battery_level = int(device_battery_level * 100)
            self._attrs[ATTR_BATTERY] = self._battery_level
            self._attrs[ATTR_LOW_POWER_MODE] = self._status[DEVICE_LOW_POWER_MODE]
            if self._status[DEVICE_LOCATION] and self._status[DEVICE_LOCATION][DEVICE_LOCATION_LATITUDE]:
                location: Dict[str, Any] = self._status[DEVICE_LOCATION]
                if self._location is None:
                    dispatcher_send(self._account.hass, self._account.signal_device_new)
                self._location = location

    def play_sound(self) -> None:
        """Play sound on the device."""
        if self._account.api is None:
            return
        self._account.api.authenticate()
        _LOGGER.debug("Playing sound for %s", self.name)
        self.device.play_sound()

    def display_message(self, message: str, sound: bool = False) -> None:
        """Display a message on the device."""
        if self._account.api is None:
            return
        self._account.api.authenticate()
        _LOGGER.debug("Displaying message for %s", self.name)
        self.device.display_message("Subject not working", message, sound)

    def lost_device(self, number: str, message: str) -> None:
        """Make the device in lost state."""
        if self._account.api is None:
            return
        self._account.api.authenticate()
        if self._status[DEVICE_LOST_MODE_CAPABLE]:
            _LOGGER.debug("Make device lost for %s", self.name)
            self.device.lost_device(number, message, None)
        else:
            _LOGGER.error("Cannot make device lost for %s", self.name)

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return self._device_id

    @property
    def name(self) -> str:
        """Return the Apple device name."""
        return self._name

    @property
    def device(self) -> AppleDevice:
        """Return the Apple device."""
        return self._device

    @property
    def device_class(self) -> str:
        """Return the Apple device class."""
        return self._device_class

    @property
    def device_model(self) -> str:
        """Return the Apple device model."""
        return self._device_model

    @property
    def battery_level(self) -> Optional[int]:
        """Return the Apple device battery level."""
        return self._battery_level

    @property
    def battery_status(self) -> Optional[str]:
        """Return the Apple device battery status."""
        return self._battery_status

    @property
    def location(self) -> Optional[Dict[str, Any]]:
        """Return the Apple device location."""
        return self._location

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the attributes."""
        return self._attrs
