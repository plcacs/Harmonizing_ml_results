from __future__ import annotations
import calendar
import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache
from time import struct_time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast
import pandas as pd
import parsedatetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from flask_babel import lazy_gettext as _
from holidays import country_holidays
from pyparsing import (
    CaselessKeyword,
    Forward,
    Group,
    Optional as ppOptional,
    ParseException,
    ParserElement,
    ParseResults,
    pyparsing_common,
    quotedString,
    Suppress,
)
from superset.commands.chart.exceptions import TimeDeltaAmbiguousError, TimeRangeAmbiguousError, TimeRangeParseFailError
from superset.constants import InstantTimeComparison, LRU_CACHE_MAX_SIZE, NO_TIME_RANGE

ParserElement.enable_packrat()
logger = logging.getLogger(__name__)

def parse_human_datetime(human_readable: str) -> datetime:
    """Returns ``datetime.datetime`` from human readable strings"""
    x_periods = '^\\s*([0-9]+)\\s+(second|minute|hour|day|week|month|quarter|year)s?\\s*$'
    if re.search(x_periods, human_readable, re.IGNORECASE):
        raise TimeRangeAmbiguousError(human_readable)
    try:
        default = datetime(year=datetime.now().year, month=1, day=1)
        dttm = parse(human_readable, default=default)
    except (ValueError, OverflowError) as ex:
        cal = parsedatetime.Calendar()
        parsed_dttm, parsed_flags = cal.parseDT(human_readable)
        if parsed_flags == 0:
            logger.debug(ex)
            raise TimeRangeParseFailError(human_readable) from ex
        if parsed_flags & 2 == 0:
            parsed_dttm = parsed_dttm.replace(hour=0, minute=0, second=0)
        dttm = dttm_from_timetuple(parsed_dttm.utctimetuple())
    return dttm

def normalize_time_delta(human_readable: str) -> Dict[str, int]:
    x_unit = '^\\s*([0-9]+)\\s+(second|minute|hour|day|week|month|quarter|year)s?\\s+(ago|later)*$'
    matched = re.match(x_unit, human_readable, re.IGNORECASE)
    if not matched:
        raise TimeDeltaAmbiguousError(human_readable)
    key = matched[2] + 's'
    value = int(matched[1])
    value = -value if matched[3] == 'ago' else value
    return {key: value}

def dttm_from_timetuple(date_: struct_time) -> datetime:
    return datetime(date_.tm_year, date_.tm_mon, date_.tm_mday, date_.tm_hour, date_.tm_min, date_.tm_sec)

def get_past_or_future(human_readable: str, source_time: Optional[datetime] = None) -> datetime:
    cal = parsedatetime.Calendar()
    source_dttm = dttm_from_timetuple(source_time.timetuple() if source_time else datetime.now().timetuple())
    return dttm_from_timetuple(cal.parse(human_readable or '', source_dttm)[0])

def parse_human_timedelta(human_readable: str, source_time: Optional[datetime] = None) -> timedelta:
    """
    Returns ``datetime.timedelta`` from natural language time deltas

    >>> parse_human_timedelta('1 day') == timedelta(days=1)
    True
    """
    source_dttm = dttm_from_timetuple(source_time.timetuple() if source_time else datetime.now().timetuple())
    return get_past_or_future(human_readable, source_time) - source_dttm

def parse_past_timedelta(delta_str: str, source_time: Optional[datetime] = None) -> timedelta:
    """
    Takes a delta like '1 year' and finds the timedelta for that period in
    the past, then represents that past timedelta in positive terms.

    parse_human_timedelta('1 year') find the timedelta 1 year in the future.
    parse_past_timedelta('1 year') returns -datetime.timedelta(-365)
    or datetime.timedelta(365).
    """
    return -parse_human_timedelta(delta_str if delta_str.startswith('-') else f'-{delta_str}', source_time)

def get_relative_base(unit: str, relative_start: Optional[datetime] = None) -> str:
    """
    Determines the relative base (`now` or `today`) based on the granularity of the unit
    and an optional user-provided base expression. This is used as the base for all
    queries parsed from `time_range_lookup`.

    Args:
        unit (str): The time unit (e.g., "second", "minute", "hour", "day", etc.).
        relative_start (datetime | None): Optional user-provided base time.

    Returns:
        datetime: The base time (`now`, `today`, or user-provided).
    """
    if relative_start is not None:
        return relative_start.strftime('%Y-%m-%dT%H:%M:%S')
    granular_units = {'second', 'minute', 'hour'}
    broad_units = {'day', 'week', 'month', 'quarter', 'year'}
    if unit.lower() in granular_units:
        return 'now'
    elif unit.lower() in broad_units:
        return 'today'
    raise ValueError(f'Unknown unit: {unit}')

def handle_start_of(base_expression: str, unit: str) -> str:
    """
    Generates a datetime expression for the start of a given unit (e.g., start of month,
     start of year).
    This function is used to handle queries matching the first regex in
    `time_range_lookup`.

    Args:
        base_expression (str): The base datetime expression (e.g., "DATETIME('now')"),
            provided by `get_relative_base`.
        unit (str): The granularity to calculate the start for (e.g., "year",
        "month", "week"),
            extracted from the regex.

    Returns:
        str: The resulting expression for the start of the specified unit.

    Raises:
        ValueError: If the unit is not one of the valid options.

    Relation to `time_range_lookup`:
        - Handles the "start of" or "beginning of" modifiers in the first regex pattern.
        - Example: "start of this month" → `DATETRUNC(DATETIME('today'), month)`.
    """
    valid_units = {'year', 'quarter', 'month', 'week', 'day'}
    if unit in valid_units:
        return f'DATETRUNC({base_expression}, {unit})'
    raise ValueError(f"Invalid unit for 'start of': {unit}")

def handle_end_of(base_expression: str, unit: str) -> str:
    """
    Generates a datetime expression for the end of a given unit (e.g., end of month,
      end of year).
    This function is used to handle queries matching the first regex in
    `time_range_lookup`.

    Args:
        base_expression (str): The base datetime expression (e.g., "DATETIME('now')"),
            provided by `get_relative_base`.
        unit (str): The granularity to calculate the end for (e.g., "year", "month",
          "week"), extracted from the regex.

    Returns:
        str: The resulting expression for the end of the specified unit.

    Raises:
        ValueError: If the unit is not one of the valid options.

    Relation to `time_range_lookup`:
        - Handles the "end of" modifier in the first regex pattern.
        - Example: "end of last month" → `LASTDAY(DATETIME('today'), month)`.
    """
    valid_units = {'year', 'quarter', 'month', 'week', 'day'}
    if unit in valid_units:
        return f'LASTDAY({base_expression}, {unit})'
    raise ValueError(f"Invalid unit for 'end of': {unit}")

def handle_modifier_and_unit(modifier: str, scope: str, delta: str, unit: str, relative_base: str) -> str:
    """
    Generates a datetime expression based on a modifier, scope, delta, unit,
    and relative base.
    This function handles queries matching the first regex pattern in
    `time_range_lookup`.

    Args:
        modifier (str): Specifies the operation (e.g., "start of", "end of").
            Extracted from the regex to determine whether to calculate the start or end.
        scope (str): The time scope (e.g., "this", "last", "next", "prior"),
            extracted from the regex.
        delta (str): The numeric delta value (e.g., "1", "2"), extracted from the regex.
        unit (str): The granularity (e.g., "day", "month", "year"), extracted from
                    the regex.
        relative_base (str): The base datetime expression (e.g., "now" or "today"),
            determined by `get_relative_base`.

    Returns:
        str: The resulting datetime expression.

    Raises:
        ValueError: If the modifier is invalid.

    Relation to `time_range_lookup`:
        - Processes queries like "start of this month" or "end of prior 2 years".
        - Example: "start of this month" → `DATETRUNC(DATETIME('today'), month)`.

    Example:
        >>> handle_modifier_and_unit("start of", "this", "", "month", "today")
        "DATETRUNC(DATETIME('today'), month)"

        >>> handle_modifier_and_unit("end of", "last", "1", "year", "today")
        "LASTDAY(DATEADD(DATETIME('today'), -1, year), year)"
    """
    base_expression = handle_scope_and_unit(scope, delta, unit, relative_base)
    if modifier.lower() in ['start of', 'beginning of']:
        return handle_start_of(base_expression, unit.lower())
    elif modifier.lower() == 'end of':
        return handle_end_of(base_expression, unit.lower())
    else:
        raise ValueError(f'Unknown modifier: {modifier}')

def handle_scope_and_unit(scope: str, delta: str, unit: str, relative_base: str) -> str:
    """
    Generates a datetime expression based on the scope, delta, unit, and relative base.
    This function handles queries matching the second regex pattern in
    `time_range_lookup`.

    Args:
        scope (str): The time scope (e.g., "this", "last", "next", "prior"),
            extracted from the regex.
        delta (str): The numeric delta value (e.g., "1", "2"), extracted from the regex.
        unit (str): The granularity (e.g., "second", "minute", "hour", "day"),
            extracted from the regex.
        relative_base (str): The base datetime expression (e.g., "now" or "today"),
            determined by `get_relative_base`.

    Returns:
        str: The resulting datetime expression.

    Raises:
        ValueError: If the scope is invalid.

    Relation to `time_range_lookup`:
        - Processes queries like "last 2 weeks" or "this month".
        - Example: "last 2 weeks" → `DATEADD(DATETIME('today'), -2, week)`.
    """
    _delta = int(delta) if delta else 1
    if scope.lower() == 'this':
        return f"DATETIME('{relative_base}')"
    elif scope.lower() in ['last', 'prior']:
        return f"DATEADD(DATETIME('{relative_base}'), -{_delta}, {unit})"
    elif scope.lower() == 'next':
        return f"DATEADD(DATETIME('{relative_base}'), {_delta}, {unit})"
    else:
        raise ValueError(f'Invalid scope: {scope}')

def get_since_until(
    time_range: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    time_shift: Optional[str] = None,
    relative_start: Optional[datetime] = None,
    relative_end: Optional[datetime] = None,
    instant_time_comparison_range: Optional[str] = None,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return `since` and `until` date time tuple from string representations of
    time_range, since, until and time_shift.

    This function supports both reading the keys separately (from `since` and
    `until`), as well as the new `time_range` key. Valid formats are:

        - ISO 8601
        - X days/years/hours/day/year/weeks
        - X days/years/hours/day/year/weeks ago
        - X days/years/hours/day/year/weeks from now
        - freeform

    Additionally, for `time_range` (these specify both `since` and `until`):

        - Last day
        - Last week
        - Last month
        - Last quarter
        - Last year
        - No filter
        - Last X seconds/minutes/hours/days/weeks/months/years
        - Next X seconds/minutes/hours/days/weeks/months/years

    """
    separator = ' : '
    _relative_start = relative_start.strftime('%Y-%m-%dT%H:%M:%S') if relative_start else 'today'
    _relative_end = relative_end.strftime('%Y-%m-%dT%H:%M:%S') if relative_end else 'today'
    if time_range == NO_TIME_RANGE or time_range == _(NO_TIME_RANGE):
        return (None, None)
    if time_range and time_range.startswith('Last') and (separator not in time_range):
        time_range = time_range + separator + _relative_end
    if time_range and time_range.startswith('Next') and (separator not in time_range):
        time_range = _relative_start + separator + time_range
    if time_range and time_range.startswith('previous calendar week') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), -1, WEEK), WEEK) : DATETRUNC(DATETIME('today'), WEEK)"
    if time_range and time_range.startswith('previous calendar month') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), -1, MONTH), MONTH) : DATETRUNC(DATETIME('today'), MONTH)"
    if time_range and time_range.startswith('previous calendar quarter') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), -1, QUARTER), QUARTER) : DATETRUNC(DATETIME('today'), QUARTER)"
    if time_range and time_range.startswith('previous calendar year') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), -1, YEAR), YEAR) : DATETRUNC(DATETIME('today'), YEAR)"
    if time_range and time_range.startswith('Current day') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), 0, DAY), DAY) : DATETRUNC(DATEADD(DATETIME('today'), 1, DAY), DAY)"
    if time_range and time_range.startswith('Current week') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), 0, WEEK), WEEK) : DATETRUNC(DATEADD(DATETIME('today'), 1, WEEK), WEEK)"
    if time_range and time_range.startswith('Current month') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), 0, MONTH), MONTH) : DATETRUNC(DATEADD(DATETIME('today'), 1, MONTH), MONTH)"
    if time_range and time_range.startswith('Current quarter') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), 0, QUARTER), QUARTER) : DATETRUNC(DATEADD(DATETIME('today'), 1, QUARTER), QUARTER)"
    if time_range and time_range.startswith('Current year') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), 0, YEAR), YEAR) : DATETRUNC(DATEADD(DATETIME('today'), 1, YEAR), YEAR)"
    if time_range and separator in time_range:
        time_range_lookup: List[Tuple[str, Callable[..., str]]] = [
            ('^(start of|beginning of|end of)\\s+(this|last|next|prior)\\s+([0-9]+)?\\s*(day|week|month|quarter|year)s?$', lambda modifier, scope, delta, unit: handle_modifier_and_unit(modifier, scope, delta, unit, get_relative_base(unit, relative_start))),
            ('^(this|last|next|prior)\\s+([0-9]+)?\\s*(second|minute|day|week|month|quarter|year)s?$', lambda scope, delta, unit: handle_scope_and_unit(scope, delta, unit, get_relative_base(unit, relative_start))),
            ('^(DATETIME.*|DATEADD.*|DATETRUNC.*|LASTDAY.*|HOLIDAY.*)$', lambda text: text),
        ]
        since_and_until_partition = [_.strip() for _ in time_range.split(separator, 1)]
        since_and_until = []
        for part in since_and_until_partition:
            if not part