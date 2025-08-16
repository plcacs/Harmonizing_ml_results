from __future__ import annotations
from datetime import timedelta
from http import HTTPStatus
import logging
import boto3
import requests
import voluptuous as vol
from homeassistant.const import CONF_DOMAIN, CONF_TTL, CONF_ZONE
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import track_time_interval
from homeassistant.helpers.typing import ConfigType
from typing import Any, Dict, List, Union

_LOGGER: logging.Logger

CONF_ACCESS_KEY_ID: str
CONF_SECRET_ACCESS_KEY: str
CONF_RECORDS: str
DOMAIN: str
INTERVAL: timedelta
DEFAULT_TTL: int
CONFIG_SCHEMA: vol.Schema

def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    domain: str
    records: List[str]
    zone: str
    aws_access_key_id: str
    aws_secret_access_key: str
    ttl: int

    def update_records_interval(now: Any) -> None:
        pass

    def update_records_service(call: ServiceCall) -> None:
        pass

    def _get_fqdn(record: str, domain: str) -> str:
        pass

    def _update_route53(aws_access_key_id: str, aws_secret_access_key: str, zone: str, domain: str, records: List[str], ttl: int) -> None:
        pass
