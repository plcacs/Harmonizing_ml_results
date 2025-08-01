"""Sensor from an SQL Query."""
from __future__ import annotations
from datetime import date
import decimal
import logging
from typing import Any, Optional, Dict
import sqlalchemy
from sqlalchemy import lambda_stmt
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
TRIGGER_ENTITY_OPTIONS = (CONF_AVAILABILITY, CONF_DEVICE_CLASS, CONF_ICON, CONF_PICTURE, CONF_UNIQUE_ID, CONF_STATE_CLASS, CONF_UNIT_OF_MEASUREMENT)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the SQL sensor from yaml."""
    if (conf := discovery_info) is None:
        return
    name: Any = conf[CONF_NAME]
    query_str: str = conf[CONF_QUERY]
    value_template: Optional[Template] = conf.get(CONF_VALUE_TEMPLATE)
    column_name: str = conf[CONF_COLUMN_NAME]
    unique_id: Optional[Any] = conf.get(CONF_UNIQUE_ID)
    db_url: str = resolve_db_url(hass, conf.get(CONF_DB_URL))
    trigger_entity_config: Dict[str, Any] = {CONF_NAME: name}
    for key in TRIGGER_ENTITY_OPTIONS:
        if key not in conf:
            continue
        trigger_entity_config[key] = conf[key]
    await async_setup_sensor(
        hass,
        trigger_entity_config,
        query_str,
        column_name,
        value_template,
        unique_id,
        db_url,
        True,
        async_add_entities,
    )


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the SQL sensor from config entry."""
    db_url: str = resolve_db_url(hass, entry.options.get(CONF_DB_URL))
    name: str = entry.options[CONF_NAME]
    query_str: str = entry.options[CONF_QUERY]
    template: Optional[str] = entry.options.get(CONF_VALUE_TEMPLATE)
    column_name: str = entry.options[CONF_COLUMN_NAME]
    value_template: Optional[Template] = None
    if template is not None:
        try:
            value_template = Template(template, hass)
            value_template.ensure_valid()
        except TemplateError:
            value_template = None
    name_template: Template = Template(name, hass)
    trigger_entity_config: Dict[str, Any] = {CONF_NAME: name_template, CONF_UNIQUE_ID: entry.entry_id}
    for key in TRIGGER_ENTITY_OPTIONS:
        if key not in entry.options:
            continue
        trigger_entity_config[key] = entry.options[key]
    await async_setup_sensor(
        hass,
        trigger_entity_config,
        query_str,
        column_name,
        value_template,
        entry.entry_id,
        db_url,
        False,
        async_add_entities,
    )


@callback
def _async_get_or_init_domain_data(hass: HomeAssistant) -> SQLData:
    """Get or initialize domain data."""
    if DOMAIN in hass.data:
        sql_data: SQLData = hass.data[DOMAIN]
        return sql_data
    session_makers_by_db_url: Dict[str, scoped_session] = {}

    def _shutdown_db_engines(event: Event) -> None:
        """Shutdown all database engines."""
        for sessmaker in session_makers_by_db_url.values():
            sessmaker.connection().engine.dispose()

    cancel_shutdown = hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _shutdown_db_engines)
    sql_data = SQLData(cancel_shutdown, session_makers_by_db_url)
    hass.data[DOMAIN] = sql_data
    return sql_data


async def async_setup_sensor(
    hass: HomeAssistant,
    trigger_entity_config: Dict[str, Any],
    query_str: str,
    column_name: str,
    value_template: Optional[Template],
    unique_id: Any,
    db_url: str,
    yaml: bool,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the SQL sensor."""
    try:
        instance = get_instance(hass)
    except KeyError:
        uses_recorder_db: bool = False
    else:
        uses_recorder_db = db_url == instance.db_url
    sql_data: SQLData = _async_get_or_init_domain_data(hass)
    use_database_executor: bool = False
    if uses_recorder_db and instance.dialect_name == SupportedDialect.SQLITE:
        use_database_executor = True
        assert instance.engine is not None
        sessmaker: scoped_session = scoped_session(sessionmaker(bind=instance.engine, future=True))
    elif db_url in sql_data.session_makers_by_db_url:
        sessmaker = sql_data.session_makers_by_db_url[db_url]
    elif (sessmaker := (await hass.async_add_executor_job(_validate_and_get_session_maker_for_db_url, db_url))):
        sql_data.session_makers_by_db_url[db_url] = sessmaker
    else:
        return
    upper_query: str = query_str.upper()
    if uses_recorder_db:
        redacted_query: str = redact_credentials(query_str)
        issue_key: Any = unique_id if unique_id else redacted_query
        if ('ENTITY_ID,' in upper_query or 'ENTITY_ID ' in upper_query) and 'STATES_META' not in upper_query:
            _LOGGER.error(
                "The query `%s` contains the keyword `entity_id` but does not reference the `states_meta` table. "
                "This will cause a full table scan and database instability. Please check the documentation and use "
                "`states_meta.entity_id` instead", redacted_query
            )
            ir.async_create_issue(
                hass,
                DOMAIN,
                f'entity_id_query_does_full_table_scan_{issue_key}',
                translation_key='entity_id_query_does_full_table_scan',
                translation_placeholders={'query': redacted_query},
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
            )
            raise ValueError('Query contains entity_id but does not reference states_meta')
        ir.async_delete_issue(hass, DOMAIN, f'entity_id_query_does_full_table_scan_{issue_key}')
    if not ('LIMIT' in upper_query or 'SELECT TOP' in upper_query):
        if 'mssql' in db_url:
            query_str = upper_query.replace('SELECT', 'SELECT TOP 1')
        else:
            query_str = query_str.replace(';', '') + ' LIMIT 1;'
    async_add_entities([SQLSensor(trigger_entity_config, sessmaker, query_str, column_name, value_template, yaml, use_database_executor)])


def _validate_and_get_session_maker_for_db_url(db_url: str) -> Optional[scoped_session]:
    """Validate the db_url and return a session maker.

    This does I/O and should be run in the executor.
    """
    sess: Optional[Session] = None
    try:
        engine = sqlalchemy.create_engine(db_url, future=True)
        sessmaker: scoped_session = scoped_session(sessionmaker(bind=engine, future=True))
        sess = sessmaker()
        sess.execute(sqlalchemy.text('SELECT 1;'))
    except SQLAlchemyError as err:
        _LOGGER.error("Couldn't connect using %s DB_URL: %s", redact_credentials(db_url), redact_credentials(str(err)))
        return None
    else:
        return sessmaker
    finally:
        if sess:
            sess.close()


def _generate_lambda_stmt(query: str) -> StatementLambdaElement:
    """Generate the lambda statement."""
    text_stmt = sqlalchemy.text(query)
    return lambda_stmt(lambda: text_stmt, lambda_cache=_SQL_LAMBDA_CACHE)


class SQLSensor(ManualTriggerSensorEntity):
    """Representation of an SQL sensor."""

    _unrecorded_attributes = frozenset({MATCH_ALL})

    def __init__(
        self,
        trigger_entity_config: Dict[str, Any],
        sessmaker: scoped_session,
        query: str,
        column: str,
        value_template: Optional[Template],
        yaml: bool,
        use_database_executor: bool,
    ) -> None:
        """Initialize the SQL sensor."""
        super().__init__(self.hass, trigger_entity_config)
        self._query: str = query
        self._template: Optional[Template] = value_template
        self._column_name: str = column
        self.sessionmaker: scoped_session = sessmaker
        self._attr_extra_state_attributes: Dict[str, Any] = {}
        self._use_database_executor: bool = use_database_executor
        self._lambda_stmt: StatementLambdaElement = _generate_lambda_stmt(query)
        if not yaml and (unique_id := trigger_entity_config.get(CONF_UNIQUE_ID)):
            self._attr_name: Optional[str] = None
            self._attr_has_entity_name: bool = True
            self._attr_device_info: DeviceInfo = DeviceInfo(
                entry_type=DeviceEntryType.SERVICE,
                identifiers={(DOMAIN, unique_id)},
                manufacturer='SQL',
                name=self._rendered.get(CONF_NAME),
            )

    @property
    def name(self) -> str:
        """Name of the entity."""
        if self.has_entity_name:
            # type: ignore[union-attr]
            return self._attr_name  # type: ignore[return-value]
        return self._rendered.get(CONF_NAME)

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added to hass."""
        await super().async_added_to_hass()
        await self.async_update()

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra attributes."""
        return dict(self._attr_extra_state_attributes)

    async def async_update(self) -> None:
        """Retrieve sensor data from the query using the right executor."""
        if self._use_database_executor:
            data: Any = await get_instance(self.hass).async_add_executor_job(self._update)
        else:
            data = await self.hass.async_add_executor_job(self._update)
        self._process_manual_data(data)

    def _update(self) -> Any:
        """Retrieve sensor data from the query."""
        data: Any = None
        self._attr_extra_state_attributes = {}
        sess: Session = self.sessionmaker()
        try:
            result: Result = sess.execute(self._lambda_stmt)
        except SQLAlchemyError as err:
            _LOGGER.error('Error executing query %s: %s', self._query, redact_credentials(str(err)))
            sess.rollback()
            sess.close()
            return None
        for res in result.mappings():
            _LOGGER.debug('Query %s result in %s', self._query, res.items())
            data = res[self._column_name]
            for key, value in res.items():
                if isinstance(value, decimal.Decimal):
                    value = float(value)
                elif isinstance(value, date):
                    value = value.isoformat()
                elif isinstance(value, (bytes, bytearray)):
                    value = f'0x{value.hex()}'
                self._attr_extra_state_attributes[key] = value
        if data is not None and isinstance(data, (bytes, bytearray)):
            data = f'0x{data.hex()}'
        if data is not None and self._template is not None:
            self._attr_native_value = self._template.async_render_with_possible_json_value(data, None)
        else:
            self._attr_native_value = data
        if data is None:
            _LOGGER.warning('%s returned no results', self._query)
        sess.close()
        return data