"""Provide pre-made queries on top of the recorder component."""
from __future__ import annotations
from collections.abc import Callable, Collection, Iterable
from typing import Any, Dict, List, Optional, Set, Union
from sqlalchemy import Column, Text, cast, not_, or_
from sqlalchemy.sql.elements import ColumnElement
from homeassistant.const import CONF_DOMAINS, CONF_ENTITIES, CONF_EXCLUDE, CONF_INCLUDE
from homeassistant.helpers.entityfilter import CONF_ENTITY_GLOBS
from homeassistant.helpers.json import json_dumps
from homeassistant.helpers.typing import ConfigType
from .db_schema import ENTITY_ID_IN_EVENT, OLD_ENTITY_ID_IN_EVENT, States, StatesMeta

DOMAIN: str = 'history'
HISTORY_FILTERS: str = 'history_filters'
JSON_NULL: str = json_dumps(None)
GLOB_TO_SQL_CHARS: Dict[int, str] = {ord('*'): '%', ord('?'): '_', ord('%'): '\\%', ord('_'): '\\_', ord('\\'): '\\\\'}
FILTER_TYPES: tuple[str, str] = (CONF_EXCLUDE, CONF_INCLUDE)
FITLER_MATCHERS: tuple[str, str, str] = (CONF_ENTITIES, CONF_DOMAINS, CONF_ENTITY_GLOBS)

def extract_include_exclude_filter_conf(conf: ConfigType) -> Dict[str, Dict[str, Set[str]]]:
    """Extract an include exclude filter from configuration.

    This makes a copy so we do not alter the original data.
    """
    return {filter_type: {matcher: set(conf.get(filter_type, {}).get(matcher) or []) for matcher in FITLER_MATCHERS} for filter_type in FILTER_TYPES}

def merge_include_exclude_filters(base_filter: Dict[str, Dict[str, Set[str]]], add_filter: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Dict[str, Set[str]]]:
    """Merge two filters.

    This makes a copy so we do not alter the original data.
    """
    return {filter_type: {matcher: base_filter[filter_type][matcher] | add_filter[filter_type][matcher] for matcher in FITLER_MATCHERS} for filter_type in FILTER_TYPES}

def sqlalchemy_filter_from_include_exclude_conf(conf: ConfigType) -> Optional[Filters]:
    """Build a sql filter from config."""
    exclude: Dict[str, Any] = conf.get(CONF_EXCLUDE, {})
    include: Dict[str, Any] = conf.get(CONF_INCLUDE, {})
    filters: Filters = Filters(
        excluded_entities=exclude.get(CONF_ENTITIES, []),
        excluded_domains=exclude.get(CONF_DOMAINS, []),
        excluded_entity_globs=exclude.get(CONF_ENTITY_GLOBS, []),
        included_entities=include.get(CONF_ENTITIES, []),
        included_domains=include.get(CONF_DOMAINS, []),
        included_entity_globs=include.get(CONF_ENTITY_GLOBS, [])
    )
    return filters if filters.has_config else None

class Filters:
    """Container for the configured include and exclude filters.

    A filter must never change after it is created since it is used in a
    cache key.
    """

    def __init__(
        self,
        excluded_entities: Optional[List[str]] = None,
        excluded_domains: Optional[List[str]] = None,
        excluded_entity_globs: Optional[List[str]] = None,
        included_entities: Optional[List[str]] = None,
        included_domains: Optional[List[str]] = None,
        included_entity_globs: Optional[List[str]] = None,
    ) -> None:
        """Initialise the include and exclude filters."""
        self._excluded_entities: List[str] = excluded_entities or []
        self._excluded_domains: List[str] = excluded_domains or []
        self._excluded_entity_globs: List[str] = excluded_entity_globs or []
        self._included_entities: List[str] = included_entities or []
        self._included_domains: List[str] = included_domains or []
        self._included_entity_globs: List[str] = included_entity_globs or []

    def __repr__(self) -> str:
        """Return human readable excludes/includes."""
        return f'<Filters excluded_entities={self._excluded_entities} excluded_domains={self._excluded_domains} excluded_entity_globs={self._excluded_entity_globs} included_entities={self._included_entities} included_domains={self._included_domains} included_entity_globs={self._included_entity_globs}>'

    @property
    def has_config(self) -> bool:
        """Determine if there is any filter configuration."""
        return bool(self._have_exclude or self._have_include)

    @property
    def _have_exclude(self) -> bool:
        return bool(self._excluded_entities or self._excluded_domains or self._excluded_entity_globs)

    @property
    def _have_include(self) -> bool:
        return bool(self._included_entities or self._included_domains or self._included_entity_globs)

    def _generate_filter_for_columns(
        self, columns: tuple[Column, ...], encoder: Callable[[str], str]
    ) -> ColumnElement:
        """Generate a filter from pre-computed sets and pattern lists.

        This must match exactly how homeassistant.helpers.entityfilter works.
        """
        i_domains: ColumnElement = _domain_matcher(self._included_domains, columns, encoder)
        i_entities: ColumnElement = _entity_matcher(self._included_entities, columns, encoder)
        i_entity_globs: ColumnElement = _globs_to_like(self._included_entity_globs, columns, encoder)
        includes: List[ColumnElement] = [i_domains, i_entities, i_entity_globs]
        e_domains: ColumnElement = _domain_matcher(self._excluded_domains, columns, encoder)
        e_entities: ColumnElement = _entity_matcher(self._excluded_entities, columns, encoder)
        e_entity_globs: ColumnElement = _globs_to_like(self._excluded_entity_globs, columns, encoder)
        excludes: List[ColumnElement] = [e_domains, e_entities, e_entity_globs]
        have_exclude: bool = self._have_exclude
        have_include: bool = self._have_include
        if not have_include and (not have_exclude):
            raise RuntimeError('No filter configuration provided, check has_config before calling this method.')
        if have_include and (not have_exclude):
            return or_(*includes).self_group()
        if not have_include and have_exclude:
            return not_(or_(*excludes).self_group())
        if self._included_domains or self._included_entity_globs:
            return or_(i_entities, ~e_entities & (i_entity_globs | ~e_entity_globs & i_domains)).self_group()
        if self._excluded_domains or self._excluded_entity_globs:
            return (not_(or_(*excludes)) | i_entities).self_group()
        return i_entities

    def states_entity_filter(self) -> ColumnElement:
        """Generate the States.entity_id filter query.

        This is no longer used except by the legacy queries.
        """

        def _encoder(data: str) -> str:
            """Nothing to encode for states since there is no json."""
            return data
        return self._generate_filter_for_columns((States.entity_id,), _encoder)

    def states_metadata_entity_filter(self) -> ColumnElement:
        """Generate the StatesMeta.entity_id filter query."""

        def _encoder(data: str) -> str:
            """Nothing to encode for states since there is no json."""
            return data
        return self._generate_filter_for_columns((StatesMeta.entity_id,), _encoder)

    def events_entity_filter(self) -> ColumnElement:
        """Generate the entity filter query."""
        _encoder = json_dumps
        return or_(
            ((ENTITY_ID_IN_EVENT == JSON_NULL) | ENTITY_ID_IN_EVENT.is_(None)) & 
            ((OLD_ENTITY_ID_IN_EVENT == JSON_NULL) | OLD_ENTITY_ID_IN_EVENT.is_(None)),
            self._generate_filter_for_columns((ENTITY_ID_IN_EVENT, OLD_ENTITY_ID_IN_EVENT), _encoder).self_group()
        )

def _globs_to_like(
    glob_strs: List[str], columns: tuple[Column, ...], encoder: Callable[[str], str]
) -> ColumnElement:
    """Translate glob to sql."""
    matchers: List[ColumnElement] = [
        column.is_not(None) & cast(column, Text()).like(encoder(glob_str).translate(GLOB_TO_SQL_CHARS), escape='\\')
        for glob_str in glob_strs for column in columns
    ]
    return or_(*matchers) if matchers else or_(False)

def _entity_matcher(
    entity_ids: List[str], columns: tuple[Column, ...], encoder: Callable[[str], str]
) -> ColumnElement:
    matchers: List[ColumnElement] = [
        column.is_not(None) & cast(column, Text()).in_([encoder(entity_id) for entity_id in entity_ids])
        for column in columns
    ]
    return or_(*matchers) if matchers else or_(False)

def _domain_matcher(
    domains: List[str], columns: tuple[Column, ...], encoder: Callable[[str], str]
) -> ColumnElement:
    matchers: List[ColumnElement] = [
        column.is_not(None) & cast(column, Text()).like(encoder(domain_matcher))
        for domain_matcher in like_domain_matchers(domains) for column in columns
    ]
    return or_(*matchers) if matchers else or_(False)

def like_domain_matchers(domains: List[str]) -> List[str]:
    """Convert a list of domains to sql LIKE matchers."""
    return [f'{domain}.%' for domain in domains]
