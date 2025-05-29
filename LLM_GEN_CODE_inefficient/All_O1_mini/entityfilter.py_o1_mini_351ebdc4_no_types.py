"""Helper class to implement include/exclude of entities and domains."""
from __future__ import annotations
from collections.abc import Callable
import fnmatch
from functools import lru_cache, partial
import operator
import re
from typing import Optional, Pattern, List, Dict, Any
import voluptuous as vol
from homeassistant.const import CONF_DOMAINS, CONF_ENTITIES, CONF_EXCLUDE, CONF_INCLUDE, MAX_EXPECTED_ENTITY_IDS
from homeassistant.core import split_entity_id
from . import config_validation as cv
CONF_INCLUDE_DOMAINS: str = 'include_domains'
CONF_INCLUDE_ENTITY_GLOBS: str = 'include_entity_globs'
CONF_INCLUDE_ENTITIES: str = 'include_entities'
CONF_EXCLUDE_DOMAINS: str = 'exclude_domains'
CONF_EXCLUDE_ENTITY_GLOBS: str = 'exclude_entity_globs'
CONF_EXCLUDE_ENTITIES: str = 'exclude_entities'
CONF_ENTITY_GLOBS: str = 'entity_globs'

class EntityFilter:
    """A entity filter."""

    def __init__(self, config):
        """Init the filter."""
        self.empty_filter: bool = sum((len(val) for val in config.values())) == 0
        self.config: Dict[str, List[str]] = config
        self._include_e: set[str] = set(config[CONF_INCLUDE_ENTITIES])
        self._exclude_e: set[str] = set(config[CONF_EXCLUDE_ENTITIES])
        self._include_d: set[str] = set(config[CONF_INCLUDE_DOMAINS])
        self._exclude_d: set[str] = set(config[CONF_EXCLUDE_DOMAINS])
        self._include_eg: Optional[Pattern[str]] = _convert_globs_to_pattern(config[CONF_INCLUDE_ENTITY_GLOBS])
        self._exclude_eg: Optional[Pattern[str]] = _convert_globs_to_pattern(config[CONF_EXCLUDE_ENTITY_GLOBS])
        self._filter: Callable[[str], bool] = _generate_filter_from_sets_and_pattern_lists(self._include_d, self._include_e, self._exclude_d, self._exclude_e, self._include_eg, self._exclude_eg)

    def explicitly_included(self, entity_id):
        """Check if an entity is explicitly included."""
        return entity_id in self._include_e or bool(self._include_eg and self._include_eg.match(entity_id))

    def explicitly_excluded(self, entity_id):
        """Check if an entity is explicitly excluded."""
        return entity_id in self._exclude_e or bool(self._exclude_eg and self._exclude_eg.match(entity_id))

    def get_filter(self):
        """Return the filter function."""
        return self._filter

    def __call__(self, entity_id):
        """Run the filter."""
        return self._filter(entity_id)

def convert_filter(config):
    """Convert the filter schema into a filter."""
    return EntityFilter(config)
BASE_FILTER_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_EXCLUDE_DOMAINS, default=[]): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_EXCLUDE_ENTITY_GLOBS, default=[]): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_EXCLUDE_ENTITIES, default=[]): cv.entity_ids, vol.Optional(CONF_INCLUDE_DOMAINS, default=[]): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_INCLUDE_ENTITY_GLOBS, default=[]): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_INCLUDE_ENTITIES, default=[]): cv.entity_ids})
FILTER_SCHEMA: Callable[[Any], EntityFilter] = vol.All(BASE_FILTER_SCHEMA, convert_filter)

def convert_include_exclude_filter(config):
    """Convert the include exclude filter schema into a filter."""
    include: Dict[str, List[str]] = config[CONF_INCLUDE]
    exclude: Dict[str, List[str]] = config[CONF_EXCLUDE]
    return convert_filter({CONF_INCLUDE_DOMAINS: include[CONF_DOMAINS], CONF_INCLUDE_ENTITY_GLOBS: include[CONF_ENTITY_GLOBS], CONF_INCLUDE_ENTITIES: include[CONF_ENTITIES], CONF_EXCLUDE_DOMAINS: exclude[CONF_DOMAINS], CONF_EXCLUDE_ENTITY_GLOBS: exclude[CONF_ENTITY_GLOBS], CONF_EXCLUDE_ENTITIES: exclude[CONF_ENTITIES]})
INCLUDE_EXCLUDE_FILTER_SCHEMA_INNER: vol.Schema = vol.Schema({vol.Optional(CONF_DOMAINS, default=[]): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_ENTITY_GLOBS, default=[]): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_ENTITIES, default=[]): cv.entity_ids})
INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA: vol.Schema = vol.Schema({vol.Optional(CONF_INCLUDE, default=INCLUDE_EXCLUDE_FILTER_SCHEMA_INNER({})): INCLUDE_EXCLUDE_FILTER_SCHEMA_INNER, vol.Optional(CONF_EXCLUDE, default=INCLUDE_EXCLUDE_FILTER_SCHEMA_INNER({})): INCLUDE_EXCLUDE_FILTER_SCHEMA_INNER})
INCLUDE_EXCLUDE_FILTER_SCHEMA: Callable[[Any], EntityFilter] = vol.All(INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA, convert_include_exclude_filter)

def _convert_globs_to_pattern(globs):
    """Convert a list of globs to a re pattern list."""
    if globs is None:
        return None
    translated_patterns: List[str] = [pattern for glob in set(globs) if (pattern := fnmatch.translate(glob))]
    if not translated_patterns:
        return None
    inner: str = '|'.join(translated_patterns)
    combined: str = f'(?:{inner})'
    return re.compile(combined)

def generate_filter(include_domains, include_entities, exclude_domains, exclude_entities, include_entity_globs=None, exclude_entity_globs=None):
    """Return a function that will filter entities based on the args."""
    return _generate_filter_from_sets_and_pattern_lists(set(include_domains), set(include_entities), set(exclude_domains), set(exclude_entities), _convert_globs_to_pattern(include_entity_globs), _convert_globs_to_pattern(exclude_entity_globs))

def _generate_filter_from_sets_and_pattern_lists(include_d, include_e, exclude_d, exclude_e, include_eg, exclude_eg):
    """Generate a filter from pre-computed sets and pattern lists."""
    have_exclude: bool = bool(exclude_e or exclude_d or exclude_eg)
    have_include: bool = bool(include_e or include_d or include_eg)
    if not have_include and (not have_exclude):
        return bool
    if have_include and (not have_exclude):

        @lru_cache(maxsize=MAX_EXPECTED_ENTITY_IDS)
        def entity_included(entity_id):
            """Return true if entity matches inclusion filters."""
            return entity_id in include_e or split_entity_id(entity_id)[0] in include_d or bool(include_eg and include_eg.match(entity_id))
        return entity_included
    if not have_include and have_exclude:

        @lru_cache(maxsize=MAX_EXPECTED_ENTITY_IDS)
        def entity_not_excluded(entity_id):
            """Return true if entity matches exclusion filters."""
            return not (entity_id in exclude_e or split_entity_id(entity_id)[0] in exclude_d or (exclude_eg and exclude_eg.match(entity_id)))
        return entity_not_excluded
    if include_d or include_eg:

        @lru_cache(maxsize=MAX_EXPECTED_ENTITY_IDS)
        def entity_filter_4a(entity_id):
            """Return filter function for case 4a."""
            return entity_id in include_e or (entity_id not in exclude_e and (bool(include_eg and include_eg.match(entity_id)) or (split_entity_id(entity_id)[0] in include_d and (not (exclude_eg and exclude_eg.match(entity_id))))))
        return entity_filter_4a
    if exclude_d or exclude_eg:

        @lru_cache(maxsize=MAX_EXPECTED_ENTITY_IDS)
        def entity_filter_4b(entity_id):
            """Return filter function for case 4b."""
            domain: str = split_entity_id(entity_id)[0]
            if domain in exclude_d or bool(exclude_eg and exclude_eg.match(entity_id)):
                return entity_id in include_e
            return entity_id not in exclude_e
        return entity_filter_4b
    return partial(operator.contains, include_e)