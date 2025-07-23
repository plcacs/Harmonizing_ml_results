"""Standard conversation implementation for Home Assistant."""
from __future__ import annotations
import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
import functools
import logging
from pathlib import Path
import re
import time
from typing import IO, Any, Optional, Union, cast, Dict, List, Set, Tuple, TypeVar, Type

from hassil.expression import Expression, ListReference, Sequence, TextChunk
from hassil.intents import Intents, SlotList, TextSlotList, TextSlotValue, WildcardSlotList
from hassil.recognize import MISSING_ENTITY, RecognizeResult, recognize_all, recognize_best
from hassil.string_matcher import UnmatchedRangeEntity, UnmatchedTextEntity
from hassil.trie import Trie
from hassil.util import merge_dict
from home_assistant_intents import ErrorKey, get_intents, get_languages
import yaml
from homeassistant import core
from homeassistant.components.homeassistant.exposed_entities import async_listen_entity_updates, async_should_expose
from homeassistant.const import EVENT_STATE_CHANGED, MATCH_ALL
from homeassistant.helpers import (
    area_registry as ar,
    chat_session,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
    intent,
    start as ha_start,
    template,
    translation,
)
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.event import async_track_state_added_domain
from homeassistant.util.json import JsonObjectType, json_loads_object
from .chat_log import AssistantContent, async_get_chat_log
from .const import DATA_DEFAULT_ENTITY, DEFAULT_EXPOSED_ATTRIBUTES, DOMAIN, ConversationEntityFeature
from .entity import ConversationEntity
from .models import ConversationInput, ConversationResult
from .trace import ConversationTraceEventType, async_conversation_trace_append

_LOGGER = logging.getLogger(__name__)
_DEFAULT_ERROR_TEXT = "Sorry, I couldn't understand that"
_ENTITY_REGISTRY_UPDATE_FIELDS = ['aliases', 'name', 'original_name']
REGEX_TYPE = type(re.compile(''))
TRIGGER_CALLBACK_TYPE = Callable[[ConversationInput, RecognizeResult], Awaitable[Optional[str]]]
METADATA_CUSTOM_SENTENCE = 'hass_custom_sentence'
METADATA_CUSTOM_FILE = 'hass_custom_file'
ERROR_SENTINEL = object()

T = TypeVar('T')

def json_load(fp: IO) -> JsonObjectType:
    """Wrap json_loads for get_intents."""
    return json_loads_object(fp.read())

@dataclass(slots=True)
class LanguageIntents:
    """Loaded intents for a language."""
    intents: Intents
    intents_dict: Dict[str, Any]
    intent_responses: Dict[str, Any]
    error_responses: Dict[str, Any]
    language_variant: str

@dataclass(slots=True)
class TriggerData:
    """List of sentences and the callback for a trigger."""
    sentences: List[str]
    callback: TRIGGER_CALLBACK_TYPE

@dataclass(slots=True)
class SentenceTriggerResult:
    """Result when matching a sentence trigger in an automation."""
    input_text: str
    matched_template: Optional[str]
    matched_triggers: Dict[int, RecognizeResult]

class IntentMatchingStage(Enum):
    """Stages of intent matching."""
    EXPOSED_ENTITIES_ONLY = auto()
    UNEXPOSED_ENTITIES = auto()
    FUZZY = auto()

@dataclass(frozen=True)
class IntentCacheKey:
    """Key for IntentCache."""
    text: str
    language: str
    device_id: Optional[str]

@dataclass(frozen=True)
class IntentCacheValue:
    """Value for IntentCache."""
    result: Optional[RecognizeResult]
    stage: IntentMatchingStage

class IntentCache:
    """LRU cache for intent recognition results."""

    def __init__(self, capacity: int) -> None:
        """Initialize cache."""
        self.cache: OrderedDict[IntentCacheKey, IntentCacheValue] = OrderedDict()
        self.capacity = capacity

    def get(self, key: IntentCacheKey) -> Optional[IntentCacheValue]:
        """Get value for cache or None."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: IntentCacheKey, value: IntentCacheValue) -> None:
        """Put a value in the cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

def _get_language_variations(language: str) -> Iterable[str]:
    """Generate language codes with and without region."""
    yield language
    parts = re.split('([-_])', language)
    if len(parts) == 3:
        lang, sep, region = parts
        if sep == '_':
            yield f'{lang}-{region}'
        yield lang

async def async_setup_default_agent(
    hass: core.HomeAssistant,
    entity_component: EntityComponent[ConversationEntity],
    config_intents: Optional[Dict[str, Any]]
) -> None:
    """Set up entity registry listener for the default agent."""
    entity = DefaultAgent(hass, config_intents)
    await entity_component.async_add_entities([entity])
    hass.data[DATA_DEFAULT_ENTITY] = entity

    @core.callback
    def async_entity_state_listener(event: core.Event) -> None:
        """Set expose flag on new entities."""
        async_should_expose(hass, DOMAIN, event.data['entity_id'])

    @core.callback
    def async_hass_started(hass: core.HomeAssistant) -> None:
        """Set expose flag on all entities."""
        for state in hass.states.async_all():
            async_should_expose(hass, DOMAIN, state.entity_id)
        async_track_state_added_domain(hass, MATCH_ALL, async_entity_state_listener)
    ha_start.async_at_started(hass, async_hass_started)

class DefaultAgent(ConversationEntity):
    """Default agent for conversation agent."""
    _attr_name = 'Home Assistant'
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, hass: core.HomeAssistant, config_intents: Optional[Dict[str, Any]]) -> None:
        """Initialize the default agent."""
        self.hass = hass
        self._lang_intents: Dict[str, Union[LanguageIntents, object]] = {}
        self._config_intents = config_intents
        self._slot_lists: Optional[Dict[str, SlotList]] = None
        self._exposed_names_trie: Optional[Trie] = None
        self._unexposed_names_trie: Optional[Trie] = None
        self.trigger_sentences: List[TriggerData] = []
        self._trigger_intents: Optional[Intents] = None
        self._unsub_clear_slot_list: Optional[List[core.CALLBACK_TYPE]] = None
        self._load_intents_lock = asyncio.Lock()
        self._intent_cache = IntentCache(capacity=128)

    @property
    def supported_languages(self) -> List[str]:
        """Return a list of supported languages."""
        return get_languages()

    @core.callback
    def _filter_entity_registry_changes(self, event_data: Dict[str, Any]) -> bool:
        """Filter entity registry changed events."""
        return event_data['action'] == 'update' and any(
            field in event_data['changes'] for field in _ENTITY_REGISTRY_UPDATE_FIELDS
        )

    @core.callback
    def _filter_state_changes(self, event_data: Dict[str, Any]) -> bool:
        """Filter state changed events."""
        return not event_data['old_state'] or not event_data['new_state']

    @core.callback
    def _listen_clear_slot_list(self) -> None:
        """Listen for changes that can invalidate slot list."""
        assert self._unsub_clear_slot_list is None
        self._unsub_clear_slot_list = [
            self.hass.bus.async_listen(ar.EVENT_AREA_REGISTRY_UPDATED, self._async_clear_slot_list),
            self.hass.bus.async_listen(fr.EVENT_FLOOR_REGISTRY_UPDATED, self._async_clear_slot_list),
            self.hass.bus.async_listen(
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                self._async_clear_slot_list,
                event_filter=self._filter_entity_registry_changes
            ),
            self.hass.bus.async_listen(
                EVENT_STATE_CHANGED,
                self._async_clear_slot_list,
                event_filter=self._filter_state_changes
            ),
            async_listen_entity_updates(self.hass, DOMAIN, self._async_clear_slot_list)
        ]

    async def async_recognize_intent(
        self,
        user_input: ConversationInput,
        strict_intents_only: bool = False
    ) -> Optional[RecognizeResult]:
        """Recognize intent from user input."""
        language = user_input.language or self.hass.config.language
        lang_intents = await self.async_get_or_load_intents(language)
        if lang_intents is None:
            _LOGGER.warning('No intents were loaded for language: %s', language)
            return None
        slot_lists = self._make_slot_lists()
        intent_context = self._make_intent_context(user_input)
        if self._exposed_names_trie is not None:
            text_lower = user_input.text.strip().lower()
            slot_lists['name'] = TextSlotList(
                name='name',
                values=[result[2] for result in self._exposed_names_trie.find(text_lower)]
            )
        start = time.monotonic()
        result = await self.hass.async_add_executor_job(
            self._recognize,
            user_input,
            lang_intents,
            slot_lists,
            intent_context,
            language,
            strict_intents_only
        )
        _LOGGER.debug('Recognize done in %.2f seconds', time.monotonic() - start)
        return result

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        response = None
        with chat_session.async_get_chat_session(
            self.hass, user_input.conversation_id
        ) as session, async_get_chat_log(self.hass, session, user_input) as chat_log:
            if (trigger_result := (await self.async_recognize_sentence_trigger(user_input))):
                response_text = await self._handle_trigger_result(trigger_result, user_input)
                response = intent.IntentResponse(language=user_input.language or self.hass.config.language)
                response.response_type = intent.IntentResponseType.ACTION_DONE
                response.async_set_speech(response_text)
            if response is None:
                intent_result = await self.async_recognize_intent(user_input)
                response = await self._async_process_intent_result(intent_result, user_input)
            speech = response.speech.get('plain', {}).get('speech', '')
            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=user_input.agent_id, content=speech)
            )
            return ConversationResult(response=response, conversation_id=session.conversation_id)

    async def _async_process_intent_result(
        self,
        result: Optional[RecognizeResult],
        user_input: ConversationInput
    ) -> intent.IntentResponse:
        """Process user input with intents."""
        language = user_input.language or self.hass.config.language
        lang_intents = await self.async_get_or_load_intents(language)
        if result is None:
            _LOGGER.debug("No intent was matched for '%s'", user_input.text)
            return _make_error_result(
                language,
                intent.IntentResponseErrorCode.NO_INTENT_MATCH,
                self._get_error_text(ErrorKey.NO_INTENT, lang_intents)
            )
        if result.unmatched_entities:
            _LOGGER.debug(
                "Recognized intent '%s' for template '%s' but had unmatched: %s",
                result.intent.name,
                result.intent_sentence.text if result.intent_sentence is not None else '',
                result.unmatched_entities_list
            )
            error_response_type, error_response_args = _get_unmatched_response(result)
            return _make_error_result(
                language,
                intent.IntentResponseErrorCode.NO_VALID_TARGETS,
                self._get_error_text(error_response_type, lang_intents, **error_response_args)
            )
        assert lang_intents is not None
        slots = {
            entity.name: {'value': entity.value, 'text': entity.text or entity.value}
            for entity in result.entities_list
        }
        device_area = self._get_device_area(user_input.device_id)
        if device_area:
            slots['preferred_area_id'] = {'value': device_area.id}
        async_conversation_trace_append(
            ConversationTraceEventType.TOOL_CALL,
            {
                'intent_name': result.intent.name,
                'slots': {entity.name: entity.value or entity.text for entity in result.entities_list}
            }
        )
        try:
            intent_response = await intent.async_handle(
                self.hass,
                DOMAIN,
                result.intent.name,
                slots,
                user_input.text,
                user_input.context,
                language,
                assistant=DOMAIN,
                device_id=user_input.device_id,
                conversation_agent_id=user_input.agent_id
            )
        except intent.MatchFailedError as match_error:
            error_response_type, error_response_args = _get_match_error_response(self.hass, match_error)
            return _make_error_result(
                language,
                intent.IntentResponseErrorCode.NO_VALID_TARGETS,
                self._get_error_text(error_response_type, lang_intents, **error_response_args)
            )
        except intent.IntentHandleError as err:
            _LOGGER.exception('Intent handling error')
            return _make_error_result(
                language,
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                self._get_error_text(err.response_key or ErrorKey.HANDLE_ERROR, lang_intents)
            )
        except intent.IntentUnexpectedError:
            _LOGGER.exception('Unexpected intent error')
            return _make_error_result(
                language,
                intent.IntentResponseErrorCode.UNKNOWN,
                self._get_error_text(ErrorKey.HANDLE_ERROR, lang_intents)
            )
        if not intent_response.speech and intent_response.intent is not None and (response_key := result.response):
            response_template_str = lang_intents.intent_responses.get(result.intent.name, {}).get(response_key)
            if response_template_str:
                response_template = template.Template(response_template_str, self.hass)
                speech = await self._build_speech(language, response_template, intent_response, result)
                intent_response.async_set_speech(speech)
        return intent_response

    def _recognize(
        self,
        user_input: ConversationInput,
        lang_intents: LanguageIntents,
        slot_lists: Dict[str, SlotList],
        intent_context: Optional[Dict[str, Any]],
        language: str,
        strict_intents_only: bool
    ) -> Optional[RecognizeResult]:
        """Search intents for a match to user input."""
        skip_exposed_match = False
        cache_key = IntentCacheKey(text=user_input.text, language=language, device_id=user_input.device_id)
        cache_value = self._intent_cache.get(cache_key)
        if cache_value is not None:
            if cache_value.result is not None and cache_value.stage == IntentMatchingStage.EXPOSED_ENTITIES_ONLY:
                _LOGGER.debug('Got cached result for exposed entities')
                return cache_value.result
            skip_exposed_match = True
        if not skip_exposed_match:
            start_time = time.monotonic()
            strict_result = self._recognize_strict(user_input, lang_intents, slot_lists, intent_context, language)
            _LOGGER.debug('Checked exposed entities in %s second(s)', time.monotonic() - start_time)
            self._intent_cache.put(cache_key, IntentCacheValue(result=strict_result, stage=IntentMatchingStage.EXPOSED_ENTITIES_ONLY))
            if strict_result is not None:
                return strict_result
        if strict_intents_only:
            return None
        skip_unexposed_entities_match = False
        if cache_value is not None:
            if cache_value.result is not None and cache_value.stage == IntentMatchingStage.UNEXPOSED_ENTITIES:
                _LOGGER.debug('Got cached result for all entities')
                return cache_value.result
            skip_unexposed_entities_match = True
        if not skip_unexposed_entities_match:
            unexposed_entities_slot_lists = {**slot_lists, 'name': self._get_unexposed_entity_names(user_input.text)}
            start_time = time.monotonic()
            strict_result = self._recognize_strict(user_input, lang_intents, unexposed_entities_slot_lists, intent_context, language)
            _LOGGER.debug('Checked all entities in %s second(s)', time.monotonic() - start_time)
            self._intent_cache.put(cache_key, IntentCacheValue(result=strict_result, stage=IntentMatchingStage.UNEXPOSED_ENTITIES))
            if strict_result is not None:
                return strict_result
        skip_fuzzy_match = False
        if cache_value is not None:
            if cache_value.result is not None and cache_value.stage == IntentMatchingStage.FUZZY:
                _LOGGER.debug('Got cached result for fuzzy match')
                return cache_value.result
            skip_fuzzy_match = True
        maybe_result = None
        if not skip_fuzzy_match:
            start_time = time