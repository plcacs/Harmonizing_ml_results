from __future__ import annotations
from datetime import date
import decimal
import logging
from typing import Any, List, Dict, Optional
import sqlalchemy
from sqlalchemy.engine import Result
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.sql.lambdas import StatementLambdaElement
from sqlalchemy.util import LRUCache
from homeassistant.components.recorder import CONF_DB_URL, SupportedDialect, get_instance
from homeassistant.components.sensor import CONF_STATE_CLASS
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_DEVICE_CLASS, CONF_ICON, CONF_NAME, CONF_UNIQUE_ID, CONF_UNIT_OF_MEASUREMENT, CONF_VALUE_TEMPLATE, EVENT_HOMEASSISTANT_STOP, MATCH_ALL
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.template import Template
from homeassistant.helpers.trigger_template_entity import CONF_AVAILABILITY, CONF_PICTURE, ManualTriggerSensorEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import CONF_COLUMN_NAME, CONF_QUERY, DOMAIN
from .models import SQLData
from .util import redact_credentials, resolve_db_url

_LOGGER: logging.Logger = logging.getLogger(__name__)
_SQL_LAMBDA_CACHE: LRUCache = LRUCache(1000)
TRIGGER_ENTITY_OPTIONS: Tuple[str, ...] = (CONF_AVAILABILITY, CONF_DEVICE_CLASS, CONF_ICON, CONF_PICTURE, CONF_UNIQUE_ID, CONF_STATE_CLASS, CONF_UNIT_OF_MEASUREMENT)

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    ...

@callback
def _async_get_or_init_domain_data(hass: HomeAssistant) -> SQLData:
    ...

async def async_setup_sensor(hass: HomeAssistant, trigger_entity_config: Dict[str, Any], query_str: str, column_name: str, value_template: Optional[Template], unique_id: str, db_url: str, yaml: bool, async_add_entities: AddEntitiesCallback) -> None:
    ...

def _validate_and_get_session_maker_for_db_url(db_url: str) -> Optional[scoped_session]:
    ...

def _generate_lambda_stmt(query: str) -> StatementLambdaElement:
    ...

class SQLSensor(ManualTriggerSensorEntity):
    ...

    def __init__(self, trigger_entity_config: Dict[str, Any], sessmaker: scoped_session, query: str, column: str, value_template: Optional[Template], yaml: bool, use_database_executor: bool) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_update(self) -> None:
        ...

    def _update(self) -> Any:
        ...
