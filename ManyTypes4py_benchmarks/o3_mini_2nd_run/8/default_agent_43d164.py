from __future__ import annotations
import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto
import functools
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable as TypingCallable, Dict, Iterator, List, Optional, Set, Tuple, Union, cast, Generator
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
from homeassistant.helpers import area_registry as ar, chat_session, device_registry as dr, entity_registry as er, floor_registry as fr, intent, start as ha_start, template, translation
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.event import async_track_state_added_domain
from homeassistant.util.json import JsonObjectType, json_loads_object
from .chat_log import AssistantContent, async_get_chat_log
from .const import DATA_DEFAULT_ENTITY, DEFAULT_EXPOSED_ATTRIBUTES, DOMAIN, ConversationEntityFeature
from .entity import ConversationEntity
from .models import ConversationInput, ConversationResult
from .trace import ConversationTraceEventType, async_conversation_trace_append

_LOGGER: logging.Logger = logging.getLogger(__name__)
_DEFAULT_ERROR_TEXT = "Sorry, I couldn't understand that"
_ENTITY_REGISTRY_UPDATE_FIELDS: List[str] = ['aliases', 'name', 'original_name']
REGEX_TYPE = type(re.compile(''))
TRIGGER_CALLBACK_TYPE = Callable[[ConversationInput, RecognizeResult], Awaitable[Optional[str]]]
METADATA_CUSTOM_SENTENCE = 'hass_custom_sentence'
METADATA_CUSTOM_FILE = 'hass_custom_file'
ERROR_SENTINEL = object()

def json_load(fp: Any) -> JsonObjectType:
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
    text: str
    template: Optional[str]
    matched_triggers: Dict[int, RecognizeResult]

class IntentMatchingStage(Enum):
    """Stages of intent matching."""
    EXPOSED_ENTITIES_ONLY = auto()
    'Match against exposed entities only.'
    UNEXPOSED_ENTITIES = auto()
    'Match against unexposed entities in Home Assistant.'
    FUZZY = auto()
    'Capture names that are not known to Home Assistant.'

@dataclass(frozen=True)
class IntentCacheKey:
    """Key for IntentCache."""
    text: str  # User input text.
    language: str  # Language of text.
    device_id: str  # Device id from user input.

@dataclass(frozen=True)
class IntentCacheValue:
    """Value for IntentCache."""
    result: Optional[RecognizeResult]  # Result of intent recognition.
    stage: IntentMatchingStage  # Stage where result was found.

class IntentCache:
    """LRU cache for intent recognition results."""
    cache: OrderedDict[IntentCacheKey, IntentCacheValue]
    capacity: int

    def __init__(self, capacity: int) -> None:
        """Initialize cache."""
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: IntentCacheKey) -> Optional[IntentCacheValue]:
        """Get value for cache or None."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: IntentCacheKey, value: IntentCacheValue) -> None:
        """Put a value in the cache, evicting the least recently used item if necessary."""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

def _get_language_variations(language: str) -> Generator[str, None, None]:
    """Generate language codes with and without region."""
    yield language
    parts = re.split('([-_])', language)
    if len(parts) == 3:
        lang, sep, region = parts
        if sep == '_':
            yield f'{lang}-{region}'
        yield lang

async def async_setup_default_agent(hass: core.HomeAssistant, entity_component: EntityComponent, config_intents: Any) -> None:
    """Set up entity registry listener for the default agent."""
    entity = DefaultAgent(hass, config_intents)
    await entity_component.async_add_entities([entity])
    hass.data[DATA_DEFAULT_ENTITY] = entity

    @core.callback
    def async_entity_state_listener(event: Dict[str, Any]) -> None:
        """Set expose flag on new entities."""
        async_should_expose(hass, DOMAIN, event.data['entity_id'])

    @core.callback
    def async_hass_started(hass_inner: core.HomeAssistant) -> None:
        """Set expose flag on all entities."""
        for state in hass_inner.states.async_all():
            async_should_expose(hass_inner, DOMAIN, state.entity_id)
        async_track_state_added_domain(hass_inner, MATCH_ALL, async_entity_state_listener)
    ha_start.async_at_started(hass, async_hass_started)

class DefaultAgent(ConversationEntity):
    """Default agent for conversation agent."""
    _attr_name: str = 'Home Assistant'
    _attr_supported_features: int = ConversationEntityFeature.CONTROL

    def __init__(self, hass: core.HomeAssistant, config_intents: Any) -> None:
        """Initialize the default agent."""
        self.hass: core.HomeAssistant = hass
        self._lang_intents: Dict[str, Any] = {}
        self._config_intents: Any = config_intents
        self._slot_lists: Optional[Dict[str, TextSlotList]] = None
        self._exposed_names_trie: Optional[Trie] = None
        self._unexposed_names_trie: Optional[Trie] = None
        self.trigger_sentences: List[TriggerData] = []
        self._trigger_intents: Optional[Intents] = None
        self._unsub_clear_slot_list: Optional[List[Callable[[], Any]]] = None
        self._load_intents_lock: asyncio.Lock = asyncio.Lock()
        self._intent_cache: IntentCache = IntentCache(capacity=128)

    @property
    def supported_languages(self) -> List[str]:
        """Return a list of supported languages."""
        return get_languages()

    @core.callback
    def _filter_entity_registry_changes(self, event_data: Dict[str, Any]) -> bool:
        """Filter entity registry changed events."""
        return event_data['action'] == 'update' and any((field in event_data['changes'] for field in _ENTITY_REGISTRY_UPDATE_FIELDS))

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
            self.hass.bus.async_listen(er.EVENT_ENTITY_REGISTRY_UPDATED, self._async_clear_slot_list, event_filter=self._filter_entity_registry_changes), 
            self.hass.bus.async_listen(EVENT_STATE_CHANGED, self._async_clear_slot_list, event_filter=self._filter_state_changes), 
            async_listen_entity_updates(self.hass, DOMAIN, self._async_clear_slot_list)
        ]

    async def async_recognize_intent(self, user_input: ConversationInput, strict_intents_only: bool = False) -> Optional[RecognizeResult]:
        """Recognize intent from user input."""
        language: str = user_input.language or self.hass.config.language
        lang_intents: Optional[LanguageIntents] = await self.async_get_or_load_intents(language)
        if lang_intents is None:
            _LOGGER.warning('No intents were loaded for language: %s', language)
            return None
        slot_lists: Dict[str, Any] = self._make_slot_lists()
        intent_context: Optional[Dict[str, Any]] = self._make_intent_context(user_input)
        if self._exposed_names_trie is not None:
            text_lower: str = user_input.text.strip().lower()
            slot_lists['name'] = TextSlotList(name='name', values=[result[2] for result in self._exposed_names_trie.find(text_lower)])
        start: float = time.monotonic()
        result: Optional[RecognizeResult] = await self.hass.async_add_executor_job(
            self._recognize, user_input, lang_intents, slot_lists, intent_context, language, strict_intents_only
        )
        _LOGGER.debug('Recognize done in %.2f seconds', time.monotonic() - start)
        return result

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        response: Optional[intent.IntentResponse] = None
        with chat_session.async_get_chat_session(self.hass, user_input.conversation_id) as session, async_get_chat_log(self.hass, session, user_input) as chat_log:
            if (trigger_result := (await self.async_recognize_sentence_trigger(user_input))):
                response_text: str = await self._handle_trigger_result(trigger_result, user_input)
                response = intent.IntentResponse(language=user_input.language or self.hass.config.language)
                response.response_type = intent.IntentResponseType.ACTION_DONE
                response.async_set_speech(response_text)
            if response is None:
                intent_result: Optional[RecognizeResult] = await self.async_recognize_intent(user_input)
                response = await self._async_process_intent_result(intent_result, user_input)
            speech: str = response.speech.get('plain', {}).get('speech', '')
            chat_log.async_add_assistant_content_without_tools(AssistantContent(agent_id=user_input.agent_id, content=speech))
            return ConversationResult(response=response, conversation_id=session.conversation_id)

    async def _async_process_intent_result(self, result: Optional[RecognizeResult], user_input: ConversationInput) -> intent.IntentResponse:
        """Process user input with intents."""
        language: str = user_input.language or self.hass.config.language
        lang_intents: Optional[LanguageIntents] = await self.async_get_or_load_intents(language)
        if result is None:
            _LOGGER.debug("No intent was matched for '%s'", user_input.text)
            return _make_error_result(language, intent.IntentResponseErrorCode.NO_INTENT_MATCH, self._get_error_text(ErrorKey.NO_INTENT, lang_intents))
        if result.unmatched_entities:
            _LOGGER.debug("Recognized intent '%s' for template '%s' but had unmatched: %s", result.intent.name, result.intent_sentence.text if result.intent_sentence is not None else '', result.unmatched_entities_list)
            error_response_type, error_response_args = _get_unmatched_response(result)
            return _make_error_result(language, intent.IntentResponseErrorCode.NO_VALID_TARGETS, self._get_error_text(error_response_type, lang_intents, **error_response_args))
        assert lang_intents is not None
        slots: Dict[str, Any] = {entity.name: {'value': entity.value, 'text': entity.text or entity.value} for entity in result.entities_list}
        device_area = self._get_device_area(user_input.device_id)
        if device_area:
            slots['preferred_area_id'] = {'value': device_area.id}
        async_conversation_trace_append(ConversationTraceEventType.TOOL_CALL, {'intent_name': result.intent.name, 'slots': {entity.name: entity.value or entity.text for entity in result.entities_list}})
        try:
            intent_response: intent.IntentResponse = await intent.async_handle(
                self.hass, DOMAIN, result.intent.name, slots, user_input.text, user_input.context, language,
                assistant=DOMAIN, device_id=user_input.device_id, conversation_agent_id=user_input.agent_id
            )
        except intent.MatchFailedError as match_error:
            error_response_type, error_response_args = _get_match_error_response(self.hass, match_error)
            return _make_error_result(language, intent.IntentResponseErrorCode.NO_VALID_TARGETS, self._get_error_text(error_response_type, lang_intents, **error_response_args))
        except intent.IntentHandleError as err:
            _LOGGER.exception('Intent handling error')
            return _make_error_result(language, intent.IntentResponseErrorCode.FAILED_TO_HANDLE, self._get_error_text(err.response_key or ErrorKey.HANDLE_ERROR, lang_intents))
        except intent.IntentUnexpectedError:
            _LOGGER.exception('Unexpected intent error')
            return _make_error_result(language, intent.IntentResponseErrorCode.UNKNOWN, self._get_error_text(ErrorKey.HANDLE_ERROR, lang_intents))
        if not intent_response.speech and intent_response.intent is not None and (response_key := result.response):
            response_template_str: Optional[str] = lang_intents.intent_responses.get(result.intent.name, {}).get(response_key)
            if response_template_str:
                response_template: template.Template = template.Template(response_template_str, self.hass)
                speech: str = await self._build_speech(language, response_template, intent_response, result)
                intent_response.async_set_speech(speech)
        return intent_response

    def _recognize(self, user_input: ConversationInput, lang_intents: LanguageIntents, slot_lists: Dict[str, Any], intent_context: Optional[Dict[str, Any]], language: str, strict_intents_only: bool) -> Optional[RecognizeResult]:
        """Search intents for a match to user input."""
        skip_exposed_match: bool = False
        cache_key = IntentCacheKey(text=user_input.text, language=language, device_id=user_input.device_id)
        cache_value: Optional[IntentCacheValue] = self._intent_cache.get(cache_key)
        if cache_value is not None:
            if cache_value.result is not None and cache_value.stage == IntentMatchingStage.EXPOSED_ENTITIES_ONLY:
                _LOGGER.debug('Got cached result for exposed entities')
                return cache_value.result
            skip_exposed_match = True
        if not skip_exposed_match:
            start_time: float = time.monotonic()
            strict_result: Optional[RecognizeResult] = self._recognize_strict(user_input, lang_intents, slot_lists, intent_context, language)
            _LOGGER.debug('Checked exposed entities in %s second(s)', time.monotonic() - start_time)
            self._intent_cache.put(cache_key, IntentCacheValue(result=strict_result, stage=IntentMatchingStage.EXPOSED_ENTITIES_ONLY))
            if strict_result is not None:
                return strict_result
        if strict_intents_only:
            return None
        skip_unexposed_entities_match: bool = False
        if cache_value is not None:
            if cache_value.result is not None and cache_value.stage == IntentMatchingStage.UNEXPOSED_ENTITIES:
                _LOGGER.debug('Got cached result for all entities')
                return cache_value.result
            skip_unexposed_entities_match = True
        if not skip_unexposed_entities_match:
            unexposed_entities_slot_lists: Dict[str, Any] = {**slot_lists, 'name': self._get_unexposed_entity_names(user_input.text)}
            start_time = time.monotonic()
            strict_result = self._recognize_strict(user_input, lang_intents, unexposed_entities_slot_lists, intent_context, language)
            _LOGGER.debug('Checked all entities in %s second(s)', time.monotonic() - start_time)
            self._intent_cache.put(cache_key, IntentCacheValue(result=strict_result, stage=IntentMatchingStage.UNEXPOSED_ENTITIES))
            if strict_result is not None:
                return strict_result
        skip_fuzzy_match: bool = False
        if cache_value is not None:
            if cache_value.result is not None and cache_value.stage == IntentMatchingStage.FUZZY:
                _LOGGER.debug('Got cached result for fuzzy match')
                return cache_value.result
            skip_fuzzy_match = True
        maybe_result: Optional[RecognizeResult] = None
        if not skip_fuzzy_match:
            start_time = time.monotonic()
            best_num_matched_entities: int = 0
            best_num_unmatched_entities: int = 0
            best_num_unmatched_ranges: int = 0
            for result in recognize_all(user_input.text, lang_intents.intents, slot_lists=slot_lists, intent_context=intent_context, allow_unmatched_entities=True):
                if result.text_chunks_matched < 1:
                    continue
                num_matched_entities: int = 0
                for matched_entity in result.entities_list:
                    if matched_entity.name not in result.unmatched_entities:
                        num_matched_entities += 1
                num_unmatched_entities: int = 0
                num_unmatched_ranges: int = 0
                for unmatched_entity in result.unmatched_entities_list:
                    if isinstance(unmatched_entity, UnmatchedTextEntity):
                        if unmatched_entity.text != MISSING_ENTITY:
                            num_unmatched_entities += 1
                    elif isinstance(unmatched_entity, UnmatchedRangeEntity):
                        num_unmatched_ranges += 1
                        num_unmatched_entities += 1
                    else:
                        num_unmatched_entities += 1
                if maybe_result is None or num_matched_entities > best_num_matched_entities or (num_matched_entities == best_num_matched_entities and num_unmatched_entities < best_num_unmatched_entities) or (num_matched_entities == best_num_matched_entities and num_unmatched_entities == best_num_unmatched_entities and (num_unmatched_ranges > best_num_unmatched_ranges)) or (num_matched_entities == best_num_matched_entities and num_unmatched_entities == best_num_unmatched_entities and (num_unmatched_ranges == best_num_unmatched_ranges) and (result.text_chunks_matched > maybe_result.text_chunks_matched)) or (result.text_chunks_matched == maybe_result.text_chunks_matched and num_unmatched_entities == best_num_unmatched_entities and (num_unmatched_ranges == best_num_unmatched_ranges) and ('name' in result.entities or 'name' in result.unmatched_entities)):
                    maybe_result = result
                    best_num_matched_entities = num_matched_entities
                    best_num_unmatched_entities = num_unmatched_entities
                    best_num_unmatched_ranges = num_unmatched_ranges
            self._intent_cache.put(cache_key, IntentCacheValue(result=maybe_result, stage=IntentMatchingStage.FUZZY))
            _LOGGER.debug('Did fuzzy match in %s second(s)', time.monotonic() - start_time)
        return maybe_result

    def _get_unexposed_entity_names(self, text: str) -> TextSlotList:
        """Get filtered slot list with unexposed entity names in Home Assistant."""
        if self._unexposed_names_trie is None:
            self._unexposed_names_trie = Trie()
            for name_tuple in self._get_entity_name_tuples(exposed=False):
                self._unexposed_names_trie.insert(name_tuple[0].lower(), TextSlotValue.from_tuple(name_tuple, allow_template=False))
        text_lower: str = text.strip().lower()
        return TextSlotList(name='name', values=[result[2] for result in self._unexposed_names_trie.find(text_lower)])

    def _get_entity_name_tuples(self, exposed: bool) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Yield (input name, output name, context) tuples for entities."""
        entity_registry = er.async_get(self.hass)
        for state in self.hass.states.async_all():
            entity_exposed: bool = async_should_expose(self.hass, DOMAIN, state.entity_id)
            if exposed and (not entity_exposed):
                continue
            if not exposed and entity_exposed:
                continue
            context: Dict[str, Any] = {'domain': state.domain}
            if state.attributes:
                for attr in DEFAULT_EXPOSED_ATTRIBUTES:
                    if attr not in state.attributes:
                        continue
                    context[attr] = state.attributes[attr]
            if (entity := entity_registry.async_get(state.entity_id)) and entity.aliases:
                for alias in entity.aliases:
                    alias = alias.strip()
                    if not alias:
                        continue
                    yield (alias, alias, context)
            yield (state.name, state.name, context)

    def _recognize_strict(self, user_input: ConversationInput, lang_intents: LanguageIntents, slot_lists: Dict[str, Any], intent_context: Optional[Dict[str, Any]], language: str) -> Optional[RecognizeResult]:
        """Search intents for a strict match to user input."""
        return recognize_best(user_input.text, lang_intents.intents, slot_lists=slot_lists, intent_context=intent_context, language=language, best_metadata_key=METADATA_CUSTOM_SENTENCE, best_slot_name='name')

    async def _build_speech(self, language: str, response_template: template.Template, intent_response: intent.IntentResponse, recognize_result: RecognizeResult) -> str:
        state1: Optional[Any] = None
        if intent_response.matched_states:
            state1 = intent_response.matched_states[0]
        elif intent_response.unmatched_states:
            state1 = intent_response.unmatched_states[0]
        speech_slots: Dict[str, Any] = {entity_name: entity_value.text or entity_value.value for entity_name, entity_value in recognize_result.entities.items()}
        speech_slots.update(intent_response.speech_slots)
        speech: Optional[Any] = response_template.async_render({
            'slots': speech_slots,
            'state': template.TemplateState(self.hass, state1) if state1 is not None else None,
            'query': {
                'matched': [template.TemplateState(self.hass, state) for state in intent_response.matched_states],
                'unmatched': [template.TemplateState(self.hass, state) for state in intent_response.unmatched_states]
            }
        })
        if speech is not None:
            speech = str(speech)
            speech = ' '.join(speech.strip().split())
        return speech

    async def async_reload(self, language: Optional[str] = None) -> None:
        """Clear cached intents for a language."""
        if language is None:
            self._lang_intents.clear()
            _LOGGER.debug('Cleared intents for all languages')
        else:
            self._lang_intents.pop(language, None)
            _LOGGER.debug('Cleared intents for language: %s', language)
        self._intent_cache.clear()

    async def async_prepare(self, language: Optional[str] = None) -> None:
        """Load intents for a language."""
        if language is None:
            language = self.hass.config.language
        lang_intents: Optional[LanguageIntents] = await self.async_get_or_load_intents(language)
        if lang_intents is None:
            return
        self._make_slot_lists()

    async def async_get_or_load_intents(self, language: str) -> Optional[LanguageIntents]:
        """Load all intents of a language with lock."""
        if (lang_intents := self._lang_intents.get(language)):
            return None if lang_intents is ERROR_SENTINEL else cast(LanguageIntents, lang_intents)
        async with self._load_intents_lock:
            if (lang_intents := self._lang_intents.get(language)):
                return None if lang_intents is ERROR_SENTINEL else cast(LanguageIntents, lang_intents)
            start: float = time.monotonic()
            result: Optional[LanguageIntents] = await self.hass.async_add_executor_job(self._load_intents, language)
            if result is None:
                self._lang_intents[language] = ERROR_SENTINEL
            else:
                self._lang_intents[language] = result
            _LOGGER.debug('Full intents load completed for language=%s in %.2f seconds', language, time.monotonic() - start)
            return result

    def _load_intents(self, language: str) -> Optional[LanguageIntents]:
        """Load all intents for language (run inside executor)."""
        intents_dict: Dict[str, Any] = {}
        language_variant: Optional[str] = None
        supported_langs: Set[str] = set(get_languages())
        all_language_variants: Dict[str, str] = {lang.lower(): lang for lang in supported_langs}
        for maybe_variant in _get_language_variations(language):
            matching_variant: Optional[str] = all_language_variants.get(maybe_variant.lower())
            if matching_variant:
                language_variant = matching_variant
                break
        if not language_variant:
            _LOGGER.warning('Unable to find supported language variant for %s', language)
            return None
        lang_variant_intents: Any = get_intents(language_variant, json_load=json_load)
        if lang_variant_intents:
            intents_dict = lang_variant_intents
            _LOGGER.debug('Loaded built-in intents for language=%s (%s)', language, language_variant)
        custom_sentences_dir: Path = Path(self.hass.config.path('custom_sentences', language_variant))
        if custom_sentences_dir.is_dir():
            for custom_sentences_path in custom_sentences_dir.rglob('*.yaml'):
                with custom_sentences_path.open(encoding='utf-8') as custom_sentences_file:
                    custom_sentences_yaml: Any = yaml.safe_load(custom_sentences_file)
                    if not isinstance(custom_sentences_yaml, dict):
                        _LOGGER.warning('Custom sentences file does not match expected format path=%s', custom_sentences_file.name)
                        continue
                    custom_intents_dict: Dict[str, Any] = custom_sentences_yaml.get('intents', {})
                    for intent_dict in custom_intents_dict.values():
                        intent_data_list: List[Any] = intent_dict.get('data', [])
                        for intent_data in intent_data_list:
                            sentence_metadata: Dict[str, Any] = intent_data.get('metadata', {})
                            sentence_metadata[METADATA_CUSTOM_SENTENCE] = True
                            sentence_metadata[METADATA_CUSTOM_FILE] = str(custom_sentences_path.relative_to(custom_sentences_dir.parent))
                            intent_data['metadata'] = sentence_metadata
                    merge_dict(intents_dict, custom_sentences_yaml)
                _LOGGER.debug('Loaded custom sentences language=%s (%s), path=%s', language, language_variant, custom_sentences_path)
        if self._config_intents and self.hass.config.language in (language, language_variant):
            hass_config_path: str = self.hass.config.path()
            merge_dict(intents_dict, {'intents': {intent_name: {'data': [{'sentences': sentences, 'metadata': {METADATA_CUSTOM_SENTENCE: True, METADATA_CUSTOM_FILE: hass_config_path}}]} for intent_name, sentences in self._config_intents.items()}})
            _LOGGER.debug('Loaded intents from configuration.yaml')
        if not intents_dict:
            return None
        intents: Intents = Intents.from_dict(intents_dict)
        responses_dict: Dict[str, Any] = intents_dict.get('responses', {})
        intent_responses: Dict[str, Any] = responses_dict.get('intents', {})
        error_responses: Dict[str, Any] = responses_dict.get('errors', {})
        return LanguageIntents(intents, intents_dict, intent_responses, error_responses, language_variant)

    @core.callback
    def _async_clear_slot_list(self, event: Optional[Any] = None) -> None:
        """Clear slot lists when a registry has changed."""
        _LOGGER.debug('Clearing slot lists')
        if self._unsub_clear_slot_list is None:
            return
        self._slot_lists = None
        self._exposed_names_trie = None
        self._unexposed_names_trie = None
        for unsub in self._unsub_clear_slot_list:
            unsub()
        self._unsub_clear_slot_list = None
        self._intent_cache.clear()

    @core.callback
    def _make_slot_lists(self) -> Dict[str, TextSlotList]:
        """Create slot lists with areas and entity names/aliases."""
        if self._slot_lists is not None:
            return self._slot_lists
        start: float = time.monotonic()
        exposed_entity_names: List[Tuple[str, str, Dict[str, Any]]] = list(self._get_entity_name_tuples(exposed=True))
        _LOGGER.debug('Exposed entities: %s', exposed_entity_names)
        areas = ar.async_get(self.hass)
        area_names: List[Tuple[str, str]] = []
        for area in areas.async_list_areas():
            area_names.append((area.name, area.name))
            if not area.aliases:
                continue
            for alias in area.aliases:
                alias = alias.strip()
                if not alias:
                    continue
                area_names.append((alias, alias))
        floors = fr.async_get(self.hass)
        floor_names: List[Tuple[str, str]] = []
        for floor in floors.async_list_floors():
            floor_names.append((floor.name, floor.name))
            if not floor.aliases:
                continue
            for alias in floor.aliases:
                alias = alias.strip()
                if not alias:
                    continue
                floor_names.append((alias, floor.name))
        self._exposed_names_trie = Trie()
        name_list: TextSlotList = TextSlotList.from_tuples(exposed_entity_names, allow_template=False)
        for name_value in name_list.values:
            assert isinstance(name_value.text_in, TextChunk)
            name_text: str = name_value.text_in.text.strip().lower()
            self._exposed_names_trie.insert(name_text, name_value)
        self._slot_lists = {
            'area': TextSlotList.from_tuples(area_names, allow_template=False),
            'name': name_list,
            'floor': TextSlotList.from_tuples(floor_names, allow_template=False)
        }
        self._listen_clear_slot_list()
        _LOGGER.debug('Created slot lists in %.2f seconds', time.monotonic() - start)
        return self._slot_lists

    def _make_intent_context(self, user_input: ConversationInput) -> Optional[Dict[str, Any]]:
        """Return intent recognition context for user input."""
        if not user_input.device_id:
            return None
        device_area = self._get_device_area(user_input.device_id)
        if device_area is None:
            return None
        return {'area': {'value': device_area.name, 'text': device_area.name}}

    def _get_device_area(self, device_id: Optional[str]) -> Optional[Any]:
        """Return area object for given device identifier."""
        if device_id is None:
            return None
        devices = dr.async_get(self.hass)
        device = devices.async_get(device_id)
        if device is None or device.area_id is None:
            return None
        areas = ar.async_get(self.hass)
        return areas.async_get_area(device.area_id)

    def _get_error_text(self, error_key: Any, lang_intents: Optional[LanguageIntents], **response_args: Any) -> str:
        """Get response error text by type."""
        if lang_intents is None:
            return _DEFAULT_ERROR_TEXT
        if isinstance(error_key, ErrorKey):
            response_key = error_key.value
        else:
            response_key = error_key
        response_str: str = lang_intents.error_responses.get(response_key) or _DEFAULT_ERROR_TEXT
        response_template: template.Template = template.Template(response_str, self.hass)
        return response_template.async_render(response_args)

    @core.callback
    def register_trigger(self, sentences: List[str], callback: TRIGGER_CALLBACK_TYPE) -> Callable[[], None]:
        """Register a list of sentences that will trigger a callback when recognized."""
        trigger_data: TriggerData = TriggerData(sentences=sentences, callback=callback)
        self.trigger_sentences.append(trigger_data)
        self._trigger_intents = None
        return functools.partial(self._unregister_trigger, trigger_data)

    @core.callback
    def _rebuild_trigger_intents(self) -> None:
        """Rebuild the HassIL intents object from the current trigger sentences."""
        intents_dict: Dict[str, Any] = {
            'language': self.hass.config.language,
            'intents': {str(trigger_id): {'data': [{'sentences': trigger_data.sentences}]} for trigger_id, trigger_data in enumerate(self.trigger_sentences)}
        }
        trigger_intents: Intents = Intents.from_dict(intents_dict)
        wildcard_names: Set[str] = set()
        for trigger_intent in trigger_intents.intents.values():
            for intent_data in trigger_intent.data:
                for sentence in intent_data.sentences:
                    _collect_list_references(sentence, wildcard_names)
        for wildcard_name in wildcard_names:
            trigger_intents.slot_lists[wildcard_name] = WildcardSlotList(wildcard_name)
        self._trigger_intents = trigger_intents
        _LOGGER.debug('Rebuilt trigger intents: %s', intents_dict)

    @core.callback
    def _unregister_trigger(self, trigger_data: TriggerData) -> None:
        """Unregister a set of trigger sentences."""
        self.trigger_sentences.remove(trigger_data)
        self._trigger_intents = None

    async def async_recognize_sentence_trigger(self, user_input: ConversationInput) -> Optional[SentenceTriggerResult]:
        """Try to match sentence against registered trigger sentences.
        
        Calls the registered callbacks if there's a match and returns a sentence
        trigger result.
        """
        if not self.trigger_sentences:
            return None
        if self._trigger_intents is None:
            self._rebuild_trigger_intents()
        assert self._trigger_intents is not None
        matched_triggers: Dict[int, RecognizeResult] = {}
        matched_template: Optional[str] = None
        for result in recognize_all(user_input.text, self._trigger_intents):
            if result.intent_sentence is not None:
                matched_template = result.intent_sentence.text
            trigger_id: int = int(result.intent.name)
            if trigger_id in matched_triggers:
                break
            matched_triggers[trigger_id] = result
        if not matched_triggers:
            return None
        _LOGGER.debug("'%s' matched %s trigger(s): %s", user_input.text, len(matched_triggers), list(matched_triggers))
        return SentenceTriggerResult(user_input.text, matched_template, matched_triggers)

    async def _handle_trigger_result(self, result: SentenceTriggerResult, user_input: ConversationInput) -> str:
        """Run sentence trigger callbacks and return response text."""
        trigger_callbacks: List[Awaitable[Optional[str]]] = [
            self.trigger_sentences[trigger_id].callback(user_input, trigger_result)
            for trigger_id, trigger_result in result.matched_triggers.items()
        ]
        response_text: str = ''
        response_set_by_trigger: bool = False
        for trigger_future in asyncio.as_completed(trigger_callbacks):
            trigger_response: Optional[str] = await trigger_future
            if trigger_response is None:
                continue
            response_text = trigger_response
            response_set_by_trigger = True
            break
        if response_set_by_trigger:
            response_text = response_text or ''
        elif not response_text:
            language: str = user_input.language or self.hass.config.language
            translations: Dict[str, Any] = await translation.async_get_translations(self.hass, language, DOMAIN, [DOMAIN])
            response_text = translations.get(f'component.{DOMAIN}.conversation.agent.done', 'Done')
        return response_text

    async def async_handle_sentence_triggers(self, user_input: ConversationInput) -> Optional[str]:
        """Try to input sentence against sentence triggers and return response text.
        
        Returns None if no match occurred.
        """
        if (trigger_result := (await self.async_recognize_sentence_trigger(user_input))):
            return await self._handle_trigger_result(trigger_result, user_input)
        return None

    async def async_handle_intents(self, user_input: ConversationInput) -> Optional[intent.IntentResponse]:
        """Try to match sentence against registered intents and return response.
        
        Only performs strict matching with exposed entities and exact wording.
        Returns None if no match or a matching error occurred.
        """
        result: Optional[RecognizeResult] = await self.async_recognize_intent(user_input, strict_intents_only=True)
        if not isinstance(result, RecognizeResult):
            return None
        response: intent.IntentResponse = await self._async_process_intent_result(result, user_input)
        if response.response_type == intent.IntentResponseType.ERROR and response.error_code not in (intent.IntentResponseErrorCode.FAILED_TO_HANDLE, intent.IntentResponseErrorCode.UNKNOWN):
            return None
        return response

def _make_error_result(language: str, error_code: intent.IntentResponseErrorCode, response_text: str) -> intent.IntentResponse:
    """Create conversation result with error code and text."""
    response: intent.IntentResponse = intent.IntentResponse(language=language)
    response.async_set_error(error_code, response_text)
    return response

def _get_unmatched_response(result: RecognizeResult) -> Tuple[Any, Dict[str, Any]]:
    """Get key and template arguments for error when there are unmatched intent entities/slots."""
    unmatched_text: Dict[str, str] = {key: entity.text.strip() for key, entity in result.unmatched_entities.items() if isinstance(entity, UnmatchedTextEntity) and entity.text != MISSING_ENTITY}
    if (unmatched_area := unmatched_text.get('area')):
        return (ErrorKey.NO_AREA, {'area': unmatched_area})
    if (unmatched_floor := unmatched_text.get('floor')):
        return (ErrorKey.NO_FLOOR, {'floor': unmatched_floor})
    matched_area: Optional[str] = None
    if (matched_area_entity := result.entities.get('area')):
        matched_area = matched_area_entity.text.strip()
    matched_floor: Optional[str] = None
    if (matched_floor_entity := result.entities.get('floor')):
        matched_floor = matched_floor_entity.text.strip()
    if (unmatched_name := unmatched_text.get('name')):
        if matched_area:
            return (ErrorKey.NO_ENTITY_IN_AREA, {'entity': unmatched_name, 'area': matched_area})
        if matched_floor:
            return (ErrorKey.NO_ENTITY_IN_FLOOR, {'entity': unmatched_name, 'floor': matched_floor})
        return (ErrorKey.NO_ENTITY, {'entity': unmatched_name})
    return (ErrorKey.NO_INTENT, {})

def _get_match_error_response(hass: core.HomeAssistant, match_error: intent.MatchFailedError) -> Tuple[Any, Dict[str, Any]]:
    """Return key and template arguments for error when target matching fails."""
    constraints, result = (match_error.constraints, match_error.result)
    reason = result.no_match_reason
    if reason in (intent.MatchFailedReason.DEVICE_CLASS, intent.MatchFailedReason.DOMAIN) and constraints.device_classes:
        device_class = next(iter(constraints.device_classes))
        if constraints.area_name:
            return (ErrorKey.NO_DEVICE_CLASS_IN_AREA, {'device_class': device_class, 'area': constraints.area_name})
        return (ErrorKey.NO_DEVICE_CLASS, {'device_class': device_class})
    if reason == intent.MatchFailedReason.DOMAIN and constraints.domains:
        domain = next(iter(constraints.domains))
        if constraints.area_name:
            return (ErrorKey.NO_DOMAIN_IN_AREA, {'domain': domain, 'area': constraints.area_name})
        if constraints.floor_name:
            return (ErrorKey.NO_DOMAIN_IN_FLOOR, {'domain': domain, 'floor': constraints.floor_name})
        return (ErrorKey.NO_DOMAIN, {'domain': domain})
    if reason == intent.MatchFailedReason.DUPLICATE_NAME:
        if constraints.floor_name:
            return (ErrorKey.DUPLICATE_ENTITIES_IN_FLOOR, {'entity': result.no_match_name, 'floor': constraints.floor_name})
        if constraints.area_name:
            return (ErrorKey.DUPLICATE_ENTITIES_IN_AREA, {'entity': result.no_match_name, 'area': constraints.area_name})
        return (ErrorKey.DUPLICATE_ENTITIES, {'entity': result.no_match_name})
    if reason == intent.MatchFailedReason.INVALID_AREA:
        return (ErrorKey.NO_AREA, {'area': result.no_match_name})
    if reason == intent.MatchFailedReason.INVALID_FLOOR:
        return (ErrorKey.NO_FLOOR, {'floor': result.no_match_name})
    if reason == intent.MatchFailedReason.FEATURE:
        return (ErrorKey.FEATURE_NOT_SUPPORTED, {})
    if reason == intent.MatchFailedReason.STATE:
        assert constraints.states
        state = next(iter(constraints.states))
        return (ErrorKey.ENTITY_WRONG_STATE, {'state': state})
    if reason == intent.MatchFailedReason.ASSISTANT:
        if constraints.name:
            if constraints.area_name:
                return (ErrorKey.NO_ENTITY_IN_AREA_EXPOSED, {'entity': constraints.name, 'area': constraints.area_name})
            if constraints.floor_name:
                return (ErrorKey.NO_ENTITY_IN_FLOOR_EXPOSED, {'entity': constraints.name, 'floor': constraints.floor_name})
            return (ErrorKey.NO_ENTITY_EXPOSED, {'entity': constraints.name})
        if constraints.device_classes:
            device_class = next(iter(constraints.device_classes))
            if constraints.area_name:
                return (ErrorKey.NO_DEVICE_CLASS_IN_AREA_EXPOSED, {'device_class': device_class, 'area': constraints.area_name})
            if constraints.floor_name:
                return (ErrorKey.NO_DEVICE_CLASS_IN_FLOOR_EXPOSED, {'device_class': device_class, 'floor': constraints.floor_name})
            return (ErrorKey.NO_DEVICE_CLASS_EXPOSED, {'device_class': device_class})
        if constraints.domains:
            domain = next(iter(constraints.domains))
            if constraints.area_name:
                return (ErrorKey.NO_DOMAIN_IN_AREA_EXPOSED, {'domain': domain, 'area': constraints.area_name})
            if constraints.floor_name:
                return (ErrorKey.NO_DOMAIN_IN_FLOOR_EXPOSED, {'domain': domain, 'floor': constraints.floor_name})
            return (ErrorKey.NO_DOMAIN_EXPOSED, {'domain': domain})
    return (ErrorKey.NO_INTENT, {})

def _collect_list_references(expression: Expression, list_names: Set[str]) -> None:
    """Collect list reference names recursively."""
    if isinstance(expression, Sequence):
        seq: Sequence = expression
        for item in seq.items:
            _collect_list_references(item, list_names)
    elif isinstance(expression, ListReference):
        list_ref: ListReference = expression
        list_names.add(list_ref.slot_name)