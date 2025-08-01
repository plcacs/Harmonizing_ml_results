"""Support for Tibber sensors."""
from __future__ import annotations
from collections.abc import Callable
import datetime
from datetime import timedelta
import logging
from random import randrange
from typing import Any, cast, TypedDict, NotRequired
import aiohttp
import tibber
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, PERCENTAGE, SIGNAL_STRENGTH_DECIBELS, EntityCategory, UnitOfElectricCurrent, UnitOfElectricPotential, UnitOfEnergy, UnitOfPower
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator
from homeassistant.util import Throttle, dt as dt_util
from .const import DOMAIN as TIBBER_DOMAIN, MANUFACTURER
from .coordinator import TibberDataCoordinator

_LOGGER = logging.getLogger(__name__)

ICON = 'mdi:currency-usd'
SCAN_INTERVAL = timedelta(minutes=1)
MIN_TIME_BETWEEN_UPDATES = timedelta(minutes=5)
PARALLEL_UPDATES = 0
TWENTY_MINUTES = 20 * 60

RT_SENSORS_UNIQUE_ID_MIGRATION: dict[str, str] = {
    'accumulated_consumption_last_hour': 'accumulated consumption current hour',
    'accumulated_production_last_hour': 'accumulated production current hour',
    'current_l1': 'current L1',
    'current_l2': 'current L2',
    'current_l3': 'current L3',
    'estimated_hour_consumption': 'Estimated consumption current hour'
}

RT_SENSORS_UNIQUE_ID_MIGRATION_SIMPLE: set[str] = {
    'accumulated_consumption', 'accumulated_cost', 'accumulated_production',
    'accumulated_reward', 'average_power', 'last_meter_consumption',
    'last_meter_production', 'max_power', 'min_power', 'power_factor',
    'power_production', 'signal_strength', 'voltage_phase1',
    'voltage_phase2', 'voltage_phase3'
}

RT_SENSORS: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key='averagePower',
        translation_key='average_power',
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.WATT
    ),
    # ... (rest of the RT_SENSORS tuple remains the same)
)

SENSORS: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key='month_cost',
        translation_key='month_cost',
        device_class=SensorDeviceClass.MONETARY
    ),
    # ... (rest of the SENSORS tuple remains the same)
)

class LiveMeasurementData(TypedDict):
    """TypedDict for live measurement data."""
    timestamp: str
    power: float | None
    powerProduction: float | None
    # ... (add all other possible fields from live measurement)

class TibberData(TypedDict):
    """TypedDict for Tibber data."""
    data: NotRequired[dict[str, LiveMeasurementData]]
    errors: NotRequired[list[str]]

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Tibber sensor."""
    tibber_connection = hass.data[TIBBER_DOMAIN]
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    coordinator: TibberDataCoordinator | None = None
    entities: list[TibberSensor] = []
    
    for home in tibber_connection.get_homes(only_active=False):
        try:
            await home.update_info()
        except TimeoutError as err:
            _LOGGER.error('Timeout connecting to Tibber home: %s ', err)
            raise PlatformNotReady from err
        except aiohttp.ClientError as err:
            _LOGGER.error('Error connecting to Tibber home: %s ', err)
            raise PlatformNotReady from err
            
        if home.has_active_subscription:
            entities.append(TibberSensorElPrice(home))
            if coordinator is None:
                coordinator = TibberDataCoordinator(hass, entry, tibber_connection)
            entities.extend(
                TibberDataSensor(home, coordinator, entity_description)
                for entity_description in SENSORS
            )
            
        if home.has_real_time_consumption:
            entity_creator = TibberRtEntityCreator(async_add_entities, home, entity_registry)
            await home.rt_subscribe(
                TibberRtDataCoordinator(entity_creator.add_sensors, home, hass).async_set_updated_data
            )
            
        old_id = home.info['viewer']['home']['meteringPointData']['consumptionEan']
        if old_id is None:
            continue
            
        old_entity_id = entity_registry.async_get_entity_id('sensor', TIBBER_DOMAIN, old_id)
        if old_entity_id is not None:
            entity_registry.async_update_entity(old_entity_id, new_unique_id=home.home_id)
            
        device_entry = device_registry.async_get_device(identifiers={(TIBBER_DOMAIN, old_id)})
        if device_entry and entry.entry_id in device_entry.config_entries:
            device_registry.async_update_device(device_entry.id, new_identifiers={(TIBBER_DOMAIN, home.home_id)})
            
    async_add_entities(entities, True)

class TibberSensor(SensorEntity):
    """Representation of a generic Tibber sensor."""
    _attr_has_entity_name: bool = True
    _tibber_home: tibber.TibberHome
    _home_name: str
    _device_name: str | None
    _model: str | None

    def __init__(self, *args: Any, tibber_home: tibber.TibberHome, **kwargs: Any) -> None:
        """Initialize the sensor."""
        super().__init__(*args, **kwargs)
        self._tibber_home = tibber_home
        self._home_name = tibber_home.info['viewer']['home']['appNickname']
        if self._home_name is None:
            self._home_name = tibber_home.info['viewer']['home']['address'].get('address1', '')
        self._device_name = None
        self._model = None

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device_info of the device."""
        device_info = DeviceInfo(
            identifiers={(TIBBER_DOMAIN, self._tibber_home.home_id)},
            name=self._device_name,
            manufacturer=MANUFACTURER
        )
        if self._model is not None:
            device_info['model'] = self._model
        return device_info

class TibberSensorElPrice(TibberSensor):
    """Representation of a Tibber sensor for el price."""
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT
    _attr_translation_key: str = 'electricity_price'
    _last_updated: datetime.datetime | None
    _spread_load_constant: int
    _attr_available: bool
    _attr_extra_state_attributes: dict[str, Any]
    _attr_icon: str
    _attr_unique_id: str
    _model: str
    _device_name: str

    def __init__(self, tibber_home: tibber.TibberHome) -> None:
        """Initialize the sensor."""
        super().__init__(tibber_home=tibber_home)
        self._last_updated = None
        self._spread_load_constant = randrange(TWENTY_MINUTES)
        self._attr_available = False
        self._attr_extra_state_attributes = {
            'app_nickname': None,
            'grid_company': None,
            'estimated_annual_consumption': None,
            'price_level': None,
            'max_price': None,
            'avg_price': None,
            'min_price': None,
            'off_peak_1': None,
            'peak': None,
            'off_peak_2': None,
            'intraday_price_ranking': None
        }
        self._attr_icon = ICON
        self._attr_unique_id = self._tibber_home.home_id
        self._model = 'Price Sensor'
        self._device_name = self._home_name

    async def async_update(self) -> None:
        """Get the latest data and updates the states."""
        now = dt_util.now()
        if not self._tibber_home.last_data_timestamp or \
           (self._tibber_home.last_data_timestamp - now).total_seconds() < 10 * 3600 - self._spread_load_constant or \
           (not self.available):
            _LOGGER.debug('Asking for new data')
            await self._fetch_data()
        elif self._tibber_home.current_price_total and \
             self._last_updated and \
             (self._last_updated.hour == now.hour) and \
             self._tibber_home.last_data_timestamp:
            return
            
        res = self._tibber_home.current_price_data()
        self._attr_native_value, price_level, self._last_updated, price_rank = res
        self._attr_extra_state_attributes['price_level'] = price_level
        self._attr_extra_state_attributes['intraday_price_ranking'] = price_rank
        attrs = self._tibber_home.current_attributes()
        self._attr_extra_state_attributes.update(attrs)
        self._attr_available = self._attr_native_value is not None
        self._attr_native_unit_of_measurement = self._tibber_home.price_unit

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    async def _fetch_data(self) -> None:
        _LOGGER.debug('Fetching data')
        try:
            await self._tibber_home.update_info_and_price_info()
        except (TimeoutError, aiohttp.ClientError):
            return
        data = self._tibber_home.info['viewer']['home']
        self._attr_extra_state_attributes['app_nickname'] = data['appNickname']
        self._attr_extra_state_attributes['grid_company'] = data['meteringPointData']['gridCompany']
        self._attr_extra_state_attributes['estimated_annual_consumption'] = data['meteringPointData']['estimatedAnnualConsumption']

class TibberDataSensor(TibberSensor, CoordinatorEntity[TibberDataCoordinator]):
    """Representation of a Tibber sensor."""

    def __init__(
        self,
        tibber_home: tibber.TibberHome,
        coordinator: TibberDataCoordinator,
        entity_description: SensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator=coordinator, tibber_home=tibber_home)
        self.entity_description = entity_description
        self._attr_unique_id = f'{self._tibber_home.home_id}_{self.entity_description.key}'
        if entity_description.key == 'month_cost':
            self._attr_native_unit_of_measurement = self._tibber_home.currency
        self._device_name = self._home_name

    @property
    def native_value(self) -> StateType:
        """Return the value of the sensor."""
        return getattr(self._tibber_home, self.entity_description.key)

class TibberSensorRT(TibberSensor, CoordinatorEntity['TibberRtDataCoordinator']):
    """Representation of a Tibber sensor for real time consumption."""

    def __init__(
        self,
        tibber_home: tibber.TibberHome,
        description: SensorEntityDescription,
        initial_state: StateType,
        coordinator: TibberRtDataCoordinator,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator=coordinator, tibber_home=tibber_home)
        self.entity_description = description
        self._model = 'Tibber Pulse'
        self._device_name = f'{self._model} {self._home_name}'
        self._attr_native_value = initial_state
        self._attr_unique_id = f'{self._tibber_home.home_id}_rt_{description.key}'
        if description.key in ('accumulatedCost', 'accumulatedReward'):
            self._attr_native_unit_of_measurement = tibber_home.currency

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._tibber_home.rt_subscription_running

    @callback
    def _handle_coordinator_update(self) -> None:
        if not (live_measurement := self.coordinator.get_live_measurement()):
            return
        state = live_measurement.get(self.entity_description.key)
        if state is None:
            return
        if self.entity_description.key in ('accumulatedConsumption', 'accumulatedProduction'):
            ts_local = dt_util.parse_datetime(live_measurement['timestamp'])
            if ts_local is not None:
                if self.last_reset is None or \
                   (state < 0.5 * self.native_value and 
                    (ts_local.hour == 0 or ts_local - self.last_reset > timedelta(hours=24))):
                    self._attr_last_reset = dt_util.as_utc(
                        ts_local.replace(hour=0, minute=0, second=0, microsecond=0)
                    )
        if self.entity_description.key == 'powerFactor':
            state *= 100.0
        self._attr_native_value = state
        self.async_write_ha_state()

class TibberRtEntityCreator:
    """Create realtime Tibber entities."""

    def __init__(
        self,
        async_add_entities: AddEntitiesCallback,
        tibber_home: tibber.TibberHome,
        entity_registry: er.EntityRegistry,
    ) -> None:
        """Initialize the data handler."""
        self._async_add_entities = async_add_entities
        self._tibber_home = tibber_home
        self._added_sensors: set[str] = set()
        self._entity_registry = entity_registry

    @callback
    def _migrate_unique_id(self, sensor_description: SensorEntityDescription) -> None:
        """Migrate unique id if needed."""
        home_id = self._tibber_home.home_id
        translation_key = sensor_description.translation_key
        description_key = sensor_description.key
        entity_id = None
        
        if translation_key in RT_SENSORS_UNIQUE_ID_MIGRATION_SIMPLE:
            entity_id = self._entity_registry.async_get_entity_id(
                'sensor',
                TIBBER_DOMAIN,
                f'{home_id}_rt_{translation_key.replace("_", " ")}'
            )
        elif translation_key in RT_SENSORS_UNIQUE_ID_MIGRATION:
            entity_id = self._entity_registry.async_get_entity_id(
                'sensor',
                TIBBER_DOMAIN,
                f'{home_id}_rt_{RT_SENSORS_UNIQUE_ID_MIGRATION[translation_key]}'
            )
        elif translation_key != description_key:
            entity_id = self._entity_registry.async_get_entity_id(
                'sensor',
                TIBBER_DOMAIN,
                f'{home_id}_rt_{translation_key}'
            )
                
        if entity_id is None:
            return
            
        new_unique_id = f'{home_id}_rt_{description_key}'
        _LOGGER.debug('Migrating unique id for %s to %s', entity_id, new_unique_id)
        try:
            self._entity_registry.async_update_entity(entity_id, new_unique_id=new_unique_id)
        except ValueError as err:
            _LOGGER.error(err)

    @callback
    def add_sensors(
        self,
        coordinator: TibberRtDataCoordinator,
        live_measurement: LiveMeasurementData,
    ) -> None:
        """Add sensor."""
        new_entities: list[TibberSensorRT] = []
        for sensor_description in RT_SENSORS:
            if sensor_description.key in self._added_sensors:
                continue
            state = live_measurement.get(sensor_description.key)
            if state is None:
                continue
            self._migrate_unique_id(sensor_description)
            entity = TibberSensorRT(self._tibber_home, sensor_description, state, coordinator)
            new_entities.append(entity)
            self._added_sensors.add(sensor_description.key)
        if new_entities:
            self._async_add_entities(new_entities)

class TibberRtDataCoordinator(DataUpdateCoordinator[TibberData]):
    """Handle Tibber realtime data."""

    def __init__(
        self,
        add_sensor_callback: Callable[[TibberRtDataCoordinator, LiveMeasurementData], None],
        tibber_home: tibber.TibberHome,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the data handler."""
        self._add_sensor_callback = add_sensor_callback
        super().__init__(
            hass,
            _LOGGER,
            name=tibber_home.info['viewer']['home']['address'].get('address1', 'Tibber')
        )
        self._async_remove_device_updates_handler = self.async_add_listener(self._data_updated)
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, self._handle_ha_stop)

    @callback
    def _handle_ha_stop(self, _event: Event) -> None:
        """Handle Home Assistant stopping."""
        self._async_remove_device_updates_handler()

    @callback
    def _data_updated(self) -> None:
        """Triggered when data