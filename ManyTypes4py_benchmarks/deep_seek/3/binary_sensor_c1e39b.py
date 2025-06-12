"""Use Bayesian Inference to trigger a binary sensor."""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Callable, Iterable
import logging
import math
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, Dict, List, Set, Tuple, cast
from uuid import UUID
import voluptuous as vol
from homeassistant.components.binary_sensor import PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA, BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.const import CONF_ABOVE, CONF_BELOW, CONF_DEVICE_CLASS, CONF_ENTITY_ID, CONF_NAME, CONF_PLATFORM, CONF_STATE, CONF_UNIQUE_ID, CONF_VALUE_TEMPLATE, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.exceptions import ConditionError, TemplateError
from homeassistant.helpers import condition, config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, TrackTemplateResultInfo, async_track_state_change_event, async_track_template_result
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.template import Template, result_as_boolean
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import DOMAIN, PLATFORMS
from .const import ATTR_OBSERVATIONS, ATTR_OCCURRED_OBSERVATION_ENTITIES, ATTR_PROBABILITY, ATTR_PROBABILITY_THRESHOLD, CONF_NUMERIC_STATE, CONF_OBSERVATIONS, CONF_P_GIVEN_F, CONF_P_GIVEN_T, CONF_PRIOR, CONF_PROBABILITY_THRESHOLD, CONF_TEMPLATE, CONF_TO_STATE, DEFAULT_NAME, DEFAULT_PROBABILITY_THRESHOLD
from .helpers import Observation
from .issues import raise_mirrored_entries, raise_no_prob_given_false

_LOGGER: logging.Logger = logging.getLogger(__name__)

def _above_greater_than_below(config: Dict[str, Any]) -> Dict[str, Any]:
    if config[CONF_PLATFORM] == CONF_NUMERIC_STATE:
        above: Optional[float] = config.get(CONF_ABOVE)
        below: Optional[float] = config.get(CONF_BELOW)
        if above is None and below is None:
            _LOGGER.error("For bayesian numeric state for entity: %s at least one of 'above' or 'below' must be specified", config[CONF_ENTITY_ID])
            raise vol.Invalid("For bayesian numeric state at least one of 'above' or 'below' must be specified.")
        if above is not None and below is not None:
            if above > below:
                _LOGGER.error("For bayesian numeric state 'above' (%s) must be less than 'below' (%s)", above, below)
                raise vol.Invalid("'above' is greater than 'below'")
    return config

NUMERIC_STATE_SCHEMA: vol.Schema = vol.All(
    vol.Schema({
        CONF_PLATFORM: CONF_NUMERIC_STATE,
        vol.Required(CONF_ENTITY_ID): cv.entity_id,
        vol.Optional(CONF_ABOVE): vol.Coerce(float),
        vol.Optional(CONF_BELOW): vol.Coerce(float),
        vol.Required(CONF_P_GIVEN_T): vol.Coerce(float),
        vol.Optional(CONF_P_GIVEN_F): vol.Coerce(float)
    }, required=True),
    _above_greater_than_below
)

class NumericConfig(NamedTuple):
    above: float
    below: float

def _no_overlapping(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric_configs: List[Dict[str, Any]] = [config for config in configs if config[CONF_PLATFORM] == CONF_NUMERIC_STATE]
    if len(numeric_configs) < 2:
        return configs

    d: Dict[str, List[NumericConfig]] = {}
    for _, config in enumerate(numeric_configs):
        above: float = config.get(CONF_ABOVE, -math.inf)
        below: float = config.get(CONF_BELOW, math.inf)
        entity_id: str = str(config[CONF_ENTITY_ID])
        d.setdefault(entity_id, []).append(NumericConfig(above, below))
    for ent_id, intervals in d.items():
        intervals = sorted(intervals, key=lambda tup: tup.above)
        for i, tup in enumerate(intervals):
            if len(intervals) > i + 1 and tup.below > intervals[i + 1].above:
                raise vol.Invalid(f'Ranges for bayesian numeric state entities must not overlap, but {ent_id} has overlapping ranges, above:{tup.above}, below:{tup.below} overlaps with above:{intervals[i + 1].above}, below:{intervals[i + 1].below}.')
    return configs

STATE_SCHEMA: vol.Schema = vol.Schema({
    CONF_PLATFORM: CONF_STATE,
    vol.Required(CONF_ENTITY_ID): cv.entity_id,
    vol.Required(CONF_TO_STATE): cv.string,
    vol.Required(CONF_P_GIVEN_T): vol.Coerce(float),
    vol.Optional(CONF_P_GIVEN_F): vol.Coerce(float)
}, required=True)

TEMPLATE_SCHEMA: vol.Schema = vol.Schema({
    CONF_PLATFORM: CONF_TEMPLATE,
    vol.Required(CONF_VALUE_TEMPLATE): cv.template,
    vol.Required(CONF_P_GIVEN_T): vol.Coerce(float),
    vol.Optional(CONF_P_GIVEN_F): vol.Coerce(float)
}, required=True)

PLATFORM_SCHEMA: vol.Schema = BINARY_SENSOR_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_UNIQUE_ID): cv.string,
    vol.Optional(CONF_DEVICE_CLASS): cv.string,
    vol.Required(CONF_OBSERVATIONS): vol.Schema(vol.All(
        cv.ensure_list,
        [vol.Any(TEMPLATE_SCHEMA, STATE_SCHEMA, NUMERIC_STATE_SCHEMA)],
        _no_overlapping
    )),
    vol.Required(CONF_PRIOR): vol.Coerce(float),
    vol.Optional(CONF_PROBABILITY_THRESHOLD, default=DEFAULT_PROBABILITY_THRESHOLD): vol.Coerce(float)
})

def update_probability(prior: float, prob_given_true: float, prob_given_false: float) -> float:
    """Update probability using Bayes' rule."""
    numerator: float = prob_given_true * prior
    denominator: float = numerator + prob_given_false * (1 - prior)
    return numerator / denominator

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Bayesian Binary sensor."""
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    name: str = config[CONF_NAME]
    unique_id: Optional[str] = config.get(CONF_UNIQUE_ID)
    observations: List[Dict[str, Any]] = config[CONF_OBSERVATIONS]
    prior: float = config[CONF_PRIOR]
    probability_threshold: float = config[CONF_PROBABILITY_THRESHOLD]
    device_class: Optional[str] = config.get(CONF_DEVICE_CLASS)
    broken_observations: List[Dict[str, Any]] = []
    
    for observation in observations:
        if CONF_P_GIVEN_F not in observation:
            text: str = f'{name}/{observation.get(CONF_ENTITY_ID, "")}{observation.get(CONF_VALUE_TEMPLATE, "")}'
            raise_no_prob_given_false(hass, text)
            _LOGGER.error('Missing prob_given_false YAML entry for %s', text)
            broken_observations.append(observation)
    
    observations = [x for x in observations if x not in broken_observations]
    async_add_entities([
        BayesianBinarySensor(name, unique_id, prior, observations, probability_threshold, device_class)
    ])

class BayesianBinarySensor(BinarySensorEntity):
    """Representation of a Bayesian sensor."""
    _attr_should_poll: bool = False

    def __init__(
        self,
        name: str,
        unique_id: Optional[str],
        prior: float,
        observations: List[Dict[str, Any]],
        probability_threshold: float,
        device_class: Optional[str]
    ) -> None:
        """Initialize the Bayesian sensor."""
        self._attr_name: str = name
        self._attr_unique_id: Optional[str] = unique_id and f'bayesian-{unique_id}'
        self._observations: List[Observation] = [
            Observation(
                entity_id=observation.get(CONF_ENTITY_ID),
                platform=observation[CONF_PLATFORM],
                prob_given_false=observation[CONF_P_GIVEN_F],
                prob_given_true=observation[CONF_P_GIVEN_T],
                observed=None,
                to_state=observation.get(CONF_TO_STATE),
                above=observation.get(CONF_ABOVE),
                below=observation.get(CONF_BELOW),
                value_template=observation.get(CONF_VALUE_TEMPLATE)
            ) for observation in observations
        ]
        self._probability_threshold: float = probability_threshold
        self._attr_device_class: Optional[str] = device_class
        self._attr_is_on: bool = False
        self._callbacks: List[TrackTemplateResult] = []
        self.prior: float = prior
        self.probability: float = prior
        self.current_observations: OrderedDict[UUID, Observation] = OrderedDict()
        self.observations_by_entity: Dict[str, List[Observation]] = self._build_observations_by_entity()
        self.observations_by_template: Dict[Template, List[Observation]] = self._build_observations_by_template()
        self.observation_handlers: Dict[str, Callable[..., Optional[bool]]] = {
            'numeric_state': self._process_numeric_state,
            'state': self._process_state
        }

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added."""
        @callback
        def async_threshold_sensor_state_listener(event: Event[EventStateChangedData]) -> None:
            """Handle sensor state changes."""
            entity_id: str = event.data['entity_id']
            self.current_observations.update(self._record_entity_observations(entity_id))
            self.async_set_context(event.context)
            self._recalculate_and_write_state()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                list(self.observations_by_entity),
                async_threshold_sensor_state_listener
            )
        )

        @callback
        def _async_template_result_changed(
            event: Optional[Event[EventStateChangedData]],
            updates: List[TrackTemplateResultInfo]
        ) -> None:
            track_template_result: TrackTemplateResult = updates.pop()
            template: Template = track_template_result.template
            result: Union[bool, TemplateError] = track_template_result.result
            entity_id: Optional[str] = None if event is None else event.data['entity_id']
            
            if isinstance(result, TemplateError):
                _LOGGER.error(
                    "TemplateError('%s') while processing template '%s' in entity '%s'",
                    result, template, self.entity_id
                )
                observed: Optional[bool] = None
            else:
                observed = result_as_boolean(result)
            
            for observation in self.observations_by_template[template]:
                observation.observed = observed
                if entity_id is not None:
                    observation.entity_id = entity_id
                self.current_observations[observation.id] = observation
            
            if event:
                self.async_set_context(event.context)
            self._recalculate_and_write_state()

        for template in self.observations_by_template:
            info: TrackTemplateResult = async_track_template_result(
                self.hass,
                [TrackTemplate(template, None)],
                _async_template_result_changed
            )
            self._callbacks.append(info)
            self.async_on_remove(info.async_remove)
            info.async_refresh()
        
        self.current_observations.update(self._initialize_current_observations())
        self.probability = self._calculate_new_probability()
        self._attr_is_on = self.probability >= self._probability_threshold
        
        for entity, observations in self.observations_by_entity.items():
            raise_mirrored_entries(self.hass, observations, text=f'{self._attr_name}/{entity}')
        
        all_template_observations: List[Observation] = [
            observations[0] for observations in self.observations_by_template.values()
        ]
        if len(all_template_observations) == 2:
            raise_mirrored_entries(
                self.hass,
                all_template_observations,
                text=f'{self._attr_name}/{all_template_observations[0].value_template}'
            )

    @callback
    def _recalculate_and_write_state(self) -> None:
        self.probability = self._calculate_new_probability()
        self._attr_is_on = bool(self.probability >= self._probability_threshold)
        self.async_write_ha_state()

    def _initialize_current_observations(self) -> OrderedDict[UUID, Observation]:
        local_observations: OrderedDict[UUID, Observation] = OrderedDict()
        for entity in self.observations_by_entity:
            local_observations.update(self._record_entity_observations(entity))
        return local_observations

    def _record_entity_observations(self, entity: str) -> OrderedDict[UUID, Observation]:
        local_observations: OrderedDict[UUID, Observation] = OrderedDict()
        for observation in self.observations_by_entity[entity]:
            platform: str = observation.platform
            observation.observed = self.observation_handlers[platform](observation, observation.multi)
            local_observations[observation.id] = observation
        return local_observations

    def _calculate_new_probability(self) -> float:
        prior: float = self.prior
        for observation in self.current_observations.values():
            if observation.observed is True:
                prior = update_probability(
                    prior,
                    observation.prob_given_true,
                    observation.prob_given_false
                )
                continue
            if observation.observed is False:
                prior = update_probability(
                    prior,
                    1 - observation.prob_given_true,
                    1 - observation.prob_given_false
                )
                continue
            if observation.entity_id is not None:
                _LOGGER.debug(
                    "Observation for entity '%s' returned None, it will not be used for Bayesian updating",
                    observation.entity_id
                )
                continue
            _LOGGER.debug(
                'Observation for template entity returned None rather than a valid boolean, '
                'it will not be used for Bayesian updating'
            )
        return prior

    def _build_observations_by_entity(self) -> Dict[str, List[Observation]]:
        """Build and return observations by entity."""
        observations_by_entity: Dict[str, List[Observation]] = {}
        for observation in self._observations:
            if (key := observation.entity_id) is None:
                continue
            observations_by_entity.setdefault(key, []).append(observation)
        
        for entity_observations in observations_by_entity.values():
            if len(entity_observations) == 1:
                continue
            for observation in entity_observations:
                observation.multi = True
        return observations_by_entity

    def _build_observations_by_template(self) -> Dict[Template, List[Observation]]:
        """Build and return observations by template."""
        observations_by_template: Dict[Template, List[Observation]] = {}
        for observation in self._observations:
            if observation.value_template is None:
                continue
            template: Template = observation.value_template
            observations_by_template.setdefault(template, []).append(observation)
        return observations_by_template

    def _process_numeric_state(
        self,
        entity_observation: Observation,
        multi: bool = False
    ) -> Optional[bool]:
        """Return True if numeric condition is met, return False if not, return None otherwise."""
        entity_id: str = entity_observation.entity_id
        if TYPE_CHECKING:
            assert entity_id is not None
        entity = self.hass.states.get(entity_id)
        if entity is None:
            return None
        try:
            if condition.state(self.hass, entity, [STATE_UNKNOWN, STATE_UNAVAILABLE]):
                return None
            result: bool = condition.async_numeric_state(
                self.hass,
                entity,
                entity_observation.below,
                entity_observation.above,
                None,
                entity_observation.to_dict()
            )
            if result:
                return True
            if multi:
                state: float = float(entity.state)
                if entity_observation.below is not None and state == entity_observation.below:
                    return True
                return None
        except ConditionError:
            return None
        else:
            return False

    def _process_state(
        self,
        entity_observation: Observation,
        multi: bool = False
    ) -> Optional[bool]:
        """Return True if state conditions are met."""
        entity: str = entity_observation.entity_id
        try:
            if condition.state(self.hass, entity, [STATE_UNKNOWN, STATE_UNAVAILABLE]):
                return None
            result: bool = condition.state(self.hass, entity, entity_observation.to_state)
            if multi and (not result):
                return None
        except ConditionError:
            return None
        else:
            return result

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the sensor."""
        occurred_entities: Set[str] = {
            observation.entity_id for observation in self.current