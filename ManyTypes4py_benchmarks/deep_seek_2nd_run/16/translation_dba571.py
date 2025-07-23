"""Translation string lookup helpers."""
from __future__ import annotations
import asyncio
from collections.abc import Iterable, Mapping, Set
from contextlib import suppress
from dataclasses import dataclass
import logging
import pathlib
import string
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from homeassistant.const import EVENT_CORE_CONFIG_UPDATE, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant, async_get_hass, callback
from homeassistant.loader import Integration, async_get_config_flows, async_get_integrations, bind_hass
from homeassistant.util.json import load_json
from . import singleton

_LOGGER = logging.getLogger(__name__)
TRANSLATION_FLATTEN_CACHE = 'translation_flatten_cache'
LOCALE_EN = 'en'

def recursive_flatten(prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a flattened representation of dict data."""
    output: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            output.update(recursive_flatten(f'{prefix}{key}.', value))
        else:
            output[f'{prefix}{key}'] = value
    return output

def _load_translations_files_by_language(translation_files: Dict[str, Dict[str, pathlib.Path]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load and parse translation.json files."""
    loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for language, component_translation_file in translation_files.items():
        loaded_for_language: Dict[str, Dict[str, Any]] = {}
        loaded[language] = loaded_for_language
        for component, translation_file in component_translation_file.items():
            loaded_json = load_json(translation_file)
            if not isinstance(loaded_json, dict):
                _LOGGER.warning('Translation file is unexpected type %s. Expected dict for %s', type(loaded_json), translation_file)
                continue
            loaded_for_language[component] = loaded_json
    return loaded

def build_resources(translation_strings: Dict[str, Dict[str, Any]], components: Set[str], category: str) -> Dict[str, Any]:
    """Build the resources response for the given components."""
    return {component: category_strings for component in components if (component_strings := translation_strings.get(component)) and (category_strings := component_strings.get(category))}

async def _async_get_component_strings(hass: HomeAssistant, languages: List[str], components: Set[str], integrations: Dict[str, Integration]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load translations."""
    translations_by_language: Dict[str, Dict[str, Dict[str, Any]]] = {}
    files_to_load_by_language: Dict[str, Dict[str, pathlib.Path]] = {}
    loaded_translations_by_language: Dict[str, Dict[str, Dict[str, Any]]] = {}
    has_files_to_load = False
    for language in languages:
        file_name = f'{language}.json'
        files_to_load = {domain: integration.file_path / 'translations' / file_name for domain in components if (integration := integrations.get(domain)) and integration.has_translations}
        files_to_load_by_language[language] = files_to_load
        has_files_to_load |= bool(files_to_load)
    if has_files_to_load:
        loaded_translations_by_language = await hass.async_add_executor_job(_load_translations_files_by_language, files_to_load_by_language)
    for language in languages:
        loaded_translations = loaded_translations_by_language.setdefault(language, {})
        for domain in components:
            component_translations = loaded_translations.setdefault(domain, {})
            if 'title' not in component_translations and (integration := integrations.get(domain)):
                component_translations['title'] = integration.name
        translations_by_language.setdefault(language, {}).update(loaded_translations)
    return translations_by_language

@dataclass(slots=True)
class _TranslationsCacheData:
    """Data for the translation cache."""
    cache: Dict[str, Dict[str, Dict[str, Dict[str, str]]]]
    loaded: Dict[str, Set[str]]

class _TranslationCache:
    """Cache for flattened translations."""
    __slots__ = ('cache_data', 'hass', 'lock')

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the cache."""
        self.hass = hass
        self.cache_data = _TranslationsCacheData({}, {})
        self.lock = asyncio.Lock()

    @callback
    def async_is_loaded(self, language: str, components: Set[str]) -> bool:
        """Return if the given components are loaded for the language."""
        return components.issubset(self.cache_data.loaded.get(language, set()))

    async def async_load(self, language: str, components: Set[str]) -> None:
        """Load resources into the cache."""
        loaded = self.cache_data.loaded.setdefault(language, set())
        if (components_to_load := (components - loaded)):
            async with self.lock:
                if (components_to_load := (components - loaded)):
                    await self._async_load(language, components_to_load)

    async def async_fetch(self, language: str, category: str, components: Set[str]) -> Dict[str, Any]:
        """Load resources into the cache and return them."""
        await self.async_load(language, components)
        return self.get_cached(language, category, components)

    def get_cached(self, language: str, category: str, components: Set[str]) -> Dict[str, Any]:
        """Read resources from the cache."""
        category_cache = self.cache_data.cache.get(language, {}).get(category, {})
        if len(components) == 1 and (component := next(iter(components))):
            return category_cache.get(component, {})
        result: Dict[str, Any] = {}
        for component in components.intersection(category_cache):
            result.update(category_cache[component])
        return result

    async def _async_load(self, language: str, components: Set[str]) -> None:
        """Populate the cache for a given set of components."""
        loaded = self.cache_data.loaded
        _LOGGER.debug('Cache miss for %s: %s', language, components)
        languages = [LOCALE_EN] if language == LOCALE_EN else [LOCALE_EN, language]
        integrations: Dict[str, Integration] = {}
        ints_or_excs = await async_get_integrations(self.hass, components)
        for domain, int_or_exc in ints_or_excs.items():
            if isinstance(int_or_exc, Exception):
                _LOGGER.warning('Failed to load integration for translation: %s', int_or_exc)
                continue
            integrations[domain] = int_or_exc
        translation_by_language_strings = await _async_get_component_strings(self.hass, languages, components, integrations)
        self._build_category_cache(language, components, translation_by_language_strings[LOCALE_EN])
        if language != LOCALE_EN:
            self._build_category_cache(language, components, translation_by_language_strings[language])
            loaded_english_components = loaded.setdefault(LOCALE_EN, set())
            if loaded_english_components.isdisjoint(components):
                self._build_category_cache(LOCALE_EN, components, translation_by_language_strings[LOCALE_EN])
                loaded_english_components.update(components)
        loaded[language].update(components)

    def _validate_placeholders(self, language: str, updated_resources: Dict[str, str], cached_resources: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Validate if updated resources have same placeholders as cached resources."""
        if cached_resources is None:
            return updated_resources
        mismatches: Set[str] = set()
        for key, value in updated_resources.items():
            if key not in cached_resources:
                continue
            try:
                tuples = list(string.Formatter().parse(value))
            except ValueError:
                _LOGGER.error('Error while parsing localized (%s) string %s', language, key)
                continue
            updated_placeholders = {tup[1] for tup in tuples if tup[1] is not None}
            tuples = list(string.Formatter().parse(cached_resources[key]))
            cached_placeholders = {tup[1] for tup in tuples if tup[1] is not None}
            if updated_placeholders != cached_placeholders:
                _LOGGER.error('Validation of translation placeholders for localized (%s) string %s failed: (%s != %s)', language, key, updated_placeholders, cached_placeholders)
                mismatches.add(key)
        for mismatch in mismatches:
            del updated_resources[mismatch]
        return updated_resources

    @callback
    def _build_category_cache(self, language: str, components: Set[str], translation_strings: Dict[str, Dict[str, Any]]) -> None:
        """Extract resources into the cache."""
        cached = self.cache_data.cache.setdefault(language, {})
        categories = {category for component in translation_strings.values() for category in component}
        for category in categories:
            new_resources = build_resources(translation_strings, components, category)
            category_cache = cached.setdefault(category, {})
            for component, resource in new_resources.items():
                component_cache = category_cache.setdefault(component, {})
                if not isinstance(resource, dict):
                    component_cache[f'component.{component}.{category}'] = resource
                    continue
                prefix = f'component.{component}.{category}.'
                flat = recursive_flatten(prefix, resource)
                flat = self._validate_placeholders(language, flat, component_cache)
                component_cache.update(flat)

@bind_hass
async def async_get_translations(hass: HomeAssistant, language: str, category: str, integrations: Optional[Set[str]] = None, config_flow: Optional[bool] = None) -> Dict[str, Any]:
    """Return all backend translations."""
    if integrations is None and config_flow:
        components = await async_get_config_flows(hass) - hass.config.components
    elif integrations is not None:
        components = set(integrations)
    else:
        components = hass.config.top_level_components
    return await _async_get_translations_cache(hass).async_fetch(language, category, components)

@callback
def async_get_cached_translations(hass: HomeAssistant, language: str, category: str, integration: Optional[str] = None) -> Dict[str, Any]:
    """Return all cached backend translations."""
    components = {integration} if integration else hass.config.top_level_components
    return _async_get_translations_cache(hass).get_cached(language, category, components)

@singleton.singleton(TRANSLATION_FLATTEN_CACHE)
def _async_get_translations_cache(hass: HomeAssistant) -> _TranslationCache:
    """Return the translation cache."""
    return _TranslationCache(hass)

@callback
def async_setup(hass: HomeAssistant) -> None:
    """Create translation cache and register listeners for translation loaders."""
    cache = _TranslationCache(hass)
    current_language = hass.config.language
    _async_get_translations_cache(hass)

    @callback
    def _async_load_translations_filter(event_data: Dict[str, Any]) -> bool:
        """Filter out unwanted events."""
        nonlocal current_language
        if (new_language := event_data.get('language')) and new_language != current_language:
            current_language = new_language
            return True
        return False

    async def _async_load_translations(event: Event) -> None:
        new_language = event.data['language']
        _LOGGER.debug('Loading translations for language: %s', new_language)
        await cache.async_load(new_language, hass.config.components)
    hass.bus.async_listen(EVENT_CORE_CONFIG_UPDATE, _async_load_translations, event_filter=_async_load_translations_filter)

async def async_load_integrations(hass: HomeAssistant, integrations: Set[str]) -> None:
    """Load translations for integrations."""
    await _async_get_translations_cache(hass).async_load(hass.config.language, integrations)

@callback
def async_translations_loaded(hass: HomeAssistant, components: Set[str]) -> bool:
    """Return if the given components are loaded for the language."""
    return _async_get_translations_cache(hass).async_is_loaded(hass.config.language, components)

@callback
def async_get_exception_message(translation_domain: str, translation_key: str, translation_placeholders: Optional[Dict[str, Any]] = None) -> str:
    """Return a translated exception message."""
    language = 'en'
    hass = async_get_hass()
    localize_key = f'component.{translation_domain}.exceptions.{translation_key}.message'
    translations = async_get_cached_translations(hass, language, 'exceptions')
    if localize_key in translations:
        if (message := translations[localize_key]):
            message = message.rstrip('.')
        if not translation_placeholders:
            return message
        with suppress(KeyError):
            message = message.format(**translation_placeholders)
        return message
    return translation_key

@callback
def async_translate_state(hass: HomeAssistant, state: str, domain: str, platform: Optional[str], translation_key: Optional[str], device_class: Optional[str]) -> str:
    """Translate provided state using cached translations."""
    if state in [STATE_UNAVAILABLE, STATE_UNKNOWN]:
        return state
    language = hass.config.language
    if platform is not None and translation_key is not None:
        localize_key = f'component.{platform}.entity.{domain}.{translation_key}.state.{state}'
        translations = async_get_cached_translations(hass, language, 'entity')
        if localize_key in translations:
            return translations[localize_key]
    translations = async_get_cached_translations(hass, language, 'entity_component')
    if device_class is not None:
        localize_key = f'component.{domain}.entity_component.{device_class}.state.{state}'
        if localize_key in translations:
            return translations[localize_key]
    localize_key = f'component.{domain}.entity_component._.state.{state}'
    if localize_key in translations:
        return translations[localize_key]
    return state
