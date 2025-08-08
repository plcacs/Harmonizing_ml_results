"""Provide pre-made queries on top of the recorder component."""
from __future__ import annotations
from collections.abc import Callable, Collection, Iterable
from typing import Any, Dict, Set, Optional, Union, Tuple, List
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
GLOB_TO_SQL_CHARS: Dict[int, str] = {
    ord('*'): '%',
    ord('?'): '_',
    ord('%'): '\\%',
    ord('_'): '\\_',
    ord('\\'): '\\\\'
}
FILTER_TYPES: Tuple[str, str] = (CONF_EXCLUDE, CONF_INCLUDE)
FITLER_MATCHERS: Tuple[str, str, str] = (CONF_ENTITIES, CONF_DOMAINS, CONF_ENTITY_GLOBS)


def func_v4eoi96m(conf: ConfigType) -> Dict[str, Dict[str, Set[str]]]:
    """Extract an include exclude filter from configuration.

    This makes a copy so we do not alter the original data.
    """
    return {
        filter_type: {
            matcher: set(conf.get(filter_type, {}).get(matcher) or [])
            for matcher in FITLER_MATCHERS
        }
        for filter_type in FILTER_TYPES
    }


def func_9mn00g16(
    base_filter: Dict[str, Dict[str, Set[str]]],
    add_filter: Dict[str, Dict[str, Set[str]]]
) -> Dict[str, Dict[str, Set[str]]]:
    """Merge two filters.

    This makes a copy so we do not alter the original data.
    """
    return {
        filter_type: {
            matcher: (base_filter[filter_type][matcher] | add_filter[filter_type][matcher])
            for matcher in FITLER_MATCHERS
        }
        for filter_type in FILTER_TYPES
    }


def func_qcpxhu4g(conf: ConfigType) -> Optional[Filters]:
    """Build a sql filter from config."""
    exclude: Dict[str, Any] = conf.get(CONF_EXCLUDE, {})
    include: Dict[str, Any] = conf.get(CONF_INCLUDE, {})
    filters = Filters(
        excluded_entities=exclude.get(CONF_ENTITIES, []),
        excluded_domains=exclude.get(CONF_DOMAINS, []),
        excluded_entity_globs=exclude.get(CONF_ENTITY_GLOBS, []),
        included_entities=include.get(CONF_ENTITIES, []),
        included_domains=include.get(CONF_DOMAINS, []),
        included_entity_globs=include.get(CONF_ENTITY_GLOBS, [])
    )
    return filters if filters.func_9908myyr else None


class Filters:
    """Container for the configured include and exclude filters.

    A filter must never change after it is created since it is used in a
    cache key.
    """

    def __init__(
        self,
        excluded_entities: Optional[Iterable[str]] = None,
        excluded_domains: Optional[Iterable[str]] = None,
        excluded_entity_globs: Optional[Iterable[str]] = None,
        included_entities: Optional[Iterable[str]] = None,
        included_domains: Optional[Iterable[str]] = None,
        included_entity_globs: Optional[Iterable[str]] = None
    ) -> None:
        """Initialise the include and exclude filters."""
        self._excluded_entities: List[str] = list(excluded_entities) if excluded_entities else []
        self._excluded_domains: List[str] = list(excluded_domains) if excluded_domains else []
        self._excluded_entity_globs: List[str] = list(excluded_entity_globs) if excluded_entity_globs else []
        self._included_entities: List[str] = list(included_entities) if included_entities else []
        self._included_domains: List[str] = list(included_domains) if included_domains else []
        self._included_entity_globs: List[str] = list(included_entity_globs) if included_entity_globs else []

    def __repr__(self) -> str:
        """Return human readable excludes/includes."""
        return (
            f'<Filters excluded_entities={self._excluded_entities} '
            f'excluded_domains={self._excluded_domains} '
            f'excluded_entity_globs={self._excluded_entity_globs} '
            f'included_entities={self._included_entities} '
            f'included_domains={self._included_domains} '
            f'included_entity_globs={self._included_entity_globs}>'
        )

    @property
    def func_9908myyr(self) -> bool:
        """Determine if there is any filter configuration."""
        return bool(self.func_uq2weplj or self.func_55h3qf0h)

    @property
    def func_uq2weplj(self) -> bool:
        """Check if there are exclude filters."""
        return bool(self._excluded_entities or self._excluded_domains or self._excluded_entity_globs)

    @property
    def func_55h3qf0h(self) -> bool:
        """Check if there are include filters."""
        return bool(self._included_entities or self._included_domains or self._included_entity_globs)

    def func_g4srmbll(
        self,
        columns: Tuple[ColumnElement[Any], ...],
        encoder: Callable[[str], str]
    ) -> ColumnElement[Any]:
        """Generate a filter from pre-computed sets and pattern lists.

        This must match exactly how homeassistant.helpers.entityfilter works.
        """
        i_domains = _domain_matcher(self._included_domains, columns, encoder)
        i_entities = _entity_matcher(self._included_entities, columns, encoder)
        i_entity_globs = _globs_to_like(self._included_entity_globs, columns, encoder)
        includes = [i_domains, i_entities, i_entity_globs]
        e_domains = _domain_matcher(self._excluded_domains, columns, encoder)
        e_entities = _entity_matcher(self._excluded_entities, columns, encoder)
        e_entity_globs = _globs_to_like(self._excluded_entity_globs, columns, encoder)
        excludes = [e_domains, e_entities, e_entity_globs]
        have_exclude = self.func_uq2weplj
        have_include = self.func_55h3qf0h
        if not have_include and not have_exclude:
            raise RuntimeError(
                'No filter configuration provided, check has_config before calling this method.'
            )
        if have_include and not have_exclude:
            return or_(*includes).self_group()
        if not have_include and have_exclude:
            return not_(or_(*excludes).self_group())
        if self._included_domains or self._included_entity_globs:
            return or_(
                i_entities,
                ~e_entities & (i_entity_globs | ~e_entity_globs & i_domains)
            ).self_group()
        if self._excluded_domains or self._excluded_entity_globs:
            return (not_(or_(*excludes)) | i_entities).self_group()
        return i_entities

    def func_qhpan2hw(self) -> ColumnElement[Any]:
        """Generate the States.entity_id filter query.

        This is no longer used except by the legacy queries.
        """

        def func_n90qa36d(data: Any) -> Any:
            """Nothing to encode for states since there is no json."""
            return data

        return self._generate_filter_for_columns((States.entity_id,), func_n90qa36d)

    def func_x8sb50de(self) -> ColumnElement[Any]:
        """Generate the StatesMeta.entity_id filter query."""

        def func_n90qa36d(data: Any) -> Any:
            """Nothing to encode for states since there is no json."""
            return data

        return self._generate_filter_for_columns((StatesMeta.entity_id,), func_n90qa36d)

    def func_66ss8nqb(self) -> ColumnElement[Any]:
        """Generate the entity filter query."""
        _encoder: Callable[[Any], str] = json_dumps
        return or_(
            ((ENTITY_ID_IN_EVENT == JSON_NULL) | ENTITY_ID_IN_EVENT.is_(None)) &
            ((OLD_ENTITY_ID_IN_EVENT == JSON_NULL) | OLD_ENTITY_ID_IN_EVENT.is_(None)),
            self._generate_filter_for_columns(
                (ENTITY_ID_IN_EVENT, OLD_ENTITY_ID_IN_EVENT), _encoder
            ).self_group()
        )

    def _generate_filter_for_columns(
        self,
        columns: Tuple[ColumnElement[Any], ...],
        encoder: Callable[[str], str]
    ) -> ColumnElement[Any]:
        ...
        # Implementation not shown


def func_1heegk1l(
    glob_strs: Iterable[str],
    columns: Iterable[ColumnElement[Any]],
    encoder: Callable[[str], str]
) -> ColumnElement[Any]:
    """Translate glob to sql."""
    matchers = [
        (column.is_not(None) & cast(column, Text()).like(encoder(glob_str).translate(GLOB_TO_SQL_CHARS), escape='\\'))
        for glob_str in glob_strs
        for column in columns
    ]
    return or_(*matchers) if matchers else or_(False)


def func_kntg84uy(
    entity_ids: Iterable[str],
    columns: Iterable[ColumnElement[Any]],
    encoder: Callable[[str], str]
) -> ColumnElement[Any]:
    matchers = [
        (column.is_not(None) & cast(column, Text()).in_([encoder(entity_id) for entity_id in entity_ids]))
        for column in columns
    ]
    return or_(*matchers) if matchers else or_(False)


def func_z1f6j9x2(
    domains: Iterable[str],
    columns: Iterable[ColumnElement[Any]],
    encoder: Callable[[str], str]
) -> ColumnElement[Any]:
    matchers = [
        (column.is_not(None) & cast(column, Text()).like(encoder(domain_matcher)))
        for domain_matcher in like_domain_matchers(domains)
        for column in columns
    ]
    return or_(*matchers) if matchers else or_(False)


def func_h9qd2ui4(domains: Iterable[str]) -> List[str]:
    """Convert a list of domains to sql LIKE matchers."""
    return [f'{domain}.%' for domain in domains]
