from collections.abc import Generator
from unittest.mock import AsyncMock, patch
from pyseventeentrack.package import Package
import pytest
from homeassistant.components.seventeentrack.const import CONF_SHOW_ARCHIVED, CONF_SHOW_DELIVERED, DEFAULT_SHOW_ARCHIVED, DEFAULT_SHOW_DELIVERED
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from tests.common import MockConfigEntry
from typing import Dict, Any

DEFAULT_SUMMARY: Dict[str, int] = {'Not Found': 0, 'In Transit': 0, 'Expired': 0, 'Ready to be Picked Up': 0, 'Undelivered': 0, 'Delivered': 0, 'Returned': 0}
DEFAULT_SUMMARY_LENGTH: int = len(DEFAULT_SUMMARY)
ACCOUNT_ID: str = '1234'
NEW_SUMMARY_DATA: Dict[str, int] = {'Not Found': 1, 'In Transit': 1, 'Expired': 1, 'Ready to be Picked Up': 1, 'Undelivered': 1, 'Delivered': 1, 'Returned': 1}
ARCHIVE_PACKAGE_NUMBER: str = '123'
CONFIG_ENTRY_ID_KEY: str = 'config_entry_id'
PACKAGE_TRACKING_NUMBER_KEY: str = 'package_tracking_number'
PACKAGE_STATE_KEY: str = 'package_state'
VALID_CONFIG: Dict[str, str] = {CONF_USERNAME: 'test', CONF_PASSWORD: 'test'}
INVALID_CONFIG: Dict[str, str] = {'notusername': 'seventeentrack', 'notpassword': 'test'}
VALID_OPTIONS: Dict[str, bool] = {CONF_SHOW_ARCHIVED: True, CONF_SHOW_DELIVERED: True}
NO_DELIVERED_OPTIONS: Dict[str, bool] = {CONF_SHOW_ARCHIVED: False, CONF_SHOW_DELIVERED: False}
VALID_PLATFORM_CONFIG_FULL: Dict[str, Dict[str, Any]] = {'sensor': {'platform': 'seventeentrack', CONF_USERNAME: 'test', CONF_PASSWORD: 'test', CONF_SHOW_ARCHIVED: True, CONF_SHOW_DELIVERED: True}}

def get_package(tracking_number: str = '456', destination_country: int = 206, friendly_name: str = 'friendly name 1', info_text: str = 'info text 1', location: str = 'location 1', timestamp: str = '2020-08-10 10:32', origin_country: int = 206, package_type: int = 2, status: int = 0, tz: str = 'UTC') -> Package:
    """Build a Package of the 17Track API."""
    return Package(tracking_number=tracking_number, destination_country=destination_country, friendly_name=friendly_name, info_text=info_text, location=location, timestamp=timestamp, origin_country=origin_country, package_type=package_type, status=status, tz=tz)
