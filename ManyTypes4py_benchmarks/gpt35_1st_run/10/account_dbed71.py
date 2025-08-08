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

ATTR_ACCOUNT_FETCH_INTERVAL: str = 'account_fetch_interval'
ATTR_BATTERY: str = 'battery'
ATTR_BATTERY_STATUS: str = 'battery_status'
ATTR_DEVICE_NAME: str = 'device_name'
ATTR_DEVICE_STATUS: str = 'device_status'
ATTR_LOW_POWER_MODE: str = 'low_power_mode'
ATTR_OWNER_NAME: str = 'owner_fullname'
SERVICE_ICLOUD_PLAY_SOUND: str = 'play_sound'
SERVICE_ICLOUD_DISPLAY_MESSAGE: str = 'display_message'
SERVICE_ICLOUD_LOST_DEVICE: str = 'lost_device'
SERVICE_ICLOUD_UPDATE: str = 'update'
ATTR_ACCOUNT: str = 'account'
ATTR_LOST_DEVICE_MESSAGE: str = 'message'
ATTR_LOST_DEVICE_NUMBER: str = 'number'
ATTR_LOST_DEVICE_SOUND: str = 'sound'
_LOGGER: logging.Logger = logging.getLogger(__name__)

class IcloudAccount:
    def __init__(self, hass: HomeAssistant, username: str, password: str, icloud_dir: Any, with_family: bool, max_interval: int, gps_accuracy_threshold: int, config_entry: ConfigEntry) -> None:
    def setup(self) -> None:
    def update_devices(self) -> None:
    def _require_reauth(self) -> None:
    def _determine_interval(self) -> int:
    def _schedule_next_fetch(self) -> None:
    def keep_alive(self, now: Optional[timedelta] = None) -> None:
    def get_devices_with_name(self, name: str) -> List[IcloudDevice]:
    @property
    def username(self) -> str:
    @property
    def owner_fullname(self) -> str:
    @property
    def family_members_fullname(self) -> Dict[str, str]:
    @property
    def fetch_interval(self) -> int:
    @property
    def devices(self) -> Dict[str, IcloudDevice]:
    @property
    def signal_device_new(self) -> str:
    @property
    def signal_device_update(self) -> str:

class IcloudDevice:
    def __init__(self, account: IcloudAccount, device: AppleDevice, status: Dict[str, Any]) -> None:
    def update(self, status: Dict[str, Any]) -> None:
    def play_sound(self) -> None:
    def display_message(self, message: str, sound: bool = False) -> None:
    def lost_device(self, number: str, message: str) -> None:
    @property
    def unique_id(self) -> str:
    @property
    def name(self) -> str:
    @property
    def device(self) -> AppleDevice:
    @property
    def device_class(self) -> str:
    @property
    def device_model(self) -> str:
    @property
    def battery_level(self) -> Optional[int]:
    @property
    def battery_status(self) -> Optional[str]:
    @property
    def location(self) -> Optional[Dict[str, Any]]:
    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
