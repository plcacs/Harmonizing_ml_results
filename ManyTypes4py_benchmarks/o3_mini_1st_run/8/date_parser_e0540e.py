from __future__ import annotations
import calendar
import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache
from time import struct_time
from typing import Optional, Union, Tuple, Dict, Any
import pandas as pd
import parsedatetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from flask_babel import lazy_gettext as _
from holidays import country_holidays
from pyparsing import CaselessKeyword, Forward, Group, Optional as ppOptional, ParseException, ParserElement, ParseResults, pyparsing_common, quotedString, Suppress
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
    source_dttm = dttm_from_timetuple(
        source_time.timetuple() if source_time else datetime.now().timetuple()
    )
    return dttm_from_timetuple(cal.parse(human_readable or '', source_dttm)[0])

def parse_human_timedelta(human_readable: str, source_time: Optional[datetime] = None) -> timedelta:
    """
    Returns ``datetime.timedelta`` from natural language time deltas

    >>> parse_human_timedelta('1 day') == timedelta(days=1)
    True
    """
    source_dttm = dttm_from_timetuple(
        source_time.timetuple() if source_time else datetime.now().timetuple()
    )
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

def get_relative_base(unit: str, relative_start: Optional[datetime] = None) -> Union[str, datetime]:
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
        return relative_start
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
    """
    valid_units = {'year', 'quarter', 'month', 'week', 'day'}
    if unit in valid_units:
        return f'DATETRUNC({base_expression}, {unit})'
    raise ValueError(f"Invalid unit for 'start of': {unit}")

def handle_end_of(base_expression: str, unit: str) -> str:
    """
    Generates a datetime expression for the end of a given unit (e.g., end of month,
      end of year).
    """
    valid_units = {'year', 'quarter', 'month', 'week', 'day'}
    if unit in valid_units:
        return f'LASTDAY({base_expression}, {unit})'
    raise ValueError(f"Invalid unit for 'end of': {unit}")

def handle_modifier_and_unit(modifier: str, scope: str, delta: str, unit: str, relative_base: Union[str, datetime]) -> str:
    """
    Generates a datetime expression based on a modifier, scope, delta, unit,
    and relative base.
    """
    base_expression = handle_scope_and_unit(scope, delta, unit, relative_base)
    if modifier.lower() in ['start of', 'beginning of']:
        return handle_start_of(base_expression, unit.lower())
    elif modifier.lower() == 'end of':
        return handle_end_of(base_expression, unit.lower())
    else:
        raise ValueError(f'Unknown modifier: {modifier}')

def handle_scope_and_unit(scope: str, delta: str, unit: str, relative_base: Union[str, datetime]) -> str:
    """
    Generates a datetime expression based on the scope, delta, unit, and relative base.
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

def get_since_until(time_range: Optional[str] = None,
                    since: Optional[str] = None,
                    until: Optional[str] = None,
                    time_shift: Optional[str] = None,
                    relative_start: Optional[datetime] = None,
                    relative_end: Optional[datetime] = None,
                    instant_time_comparison_range: Optional[InstantTimeComparison] = None
                   ) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return `since` and `until` date time tuple from string representations of
    time_range, since, until and time_shift.
    """
    separator = ' : '
    _relative_start: Union[str, datetime] = relative_start if relative_start else 'today'
    _relative_end: Union[str, datetime] = relative_end if relative_end else 'today'
    if time_range == NO_TIME_RANGE or time_range == _(NO_TIME_RANGE):
        return (None, None)
    if time_range and time_range.startswith('Last') and (separator not in time_range):
        time_range = time_range + separator + _relative_end  # type: ignore
    if time_range and time_range.startswith('Next') and (separator not in time_range):
        time_range = _relative_start + separator + time_range  # type: ignore
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
        time_range_lookup: list[Tuple[str, Any]] = [
            (
                '^(start of|beginning of|end of)\\s+(this|last|next|prior)\\s+([0-9]+)?\\s*(day|week|month|quarter|year)s?$',
                lambda modifier, scope, delta, unit: handle_modifier_and_unit(modifier, scope, delta, unit, get_relative_base(unit, relative_start))
            ),
            (
                '^(this|last|next|prior)\\s+([0-9]+)?\\s*(second|minute|day|week|month|quarter|year)s?$',
                lambda scope, delta, unit: handle_scope_and_unit(scope, delta, unit, get_relative_base(unit, relative_start))
            ),
            (
                '^(DATETIME.*|DATEADD.*|DATETRUNC.*|LASTDAY.*|HOLIDAY.*)$',
                lambda text: text
            )
        ]
        since_and_until_partition = [_.strip() for _ in time_range.split(separator, 1)]
        since_and_until: list[Any] = []
        for part in since_and_until_partition:
            if not part:
                since_and_until.append(None)
                continue
            matched = False
            for pattern, fn in time_range_lookup:
                result = re.search(pattern, part, re.IGNORECASE)
                if result:
                    matched = True
                    since_and_until.append(fn(*result.groups()))
            if not matched:
                since_and_until.append(f"DATETIME('{part}')")
        _since, _until = map(datetime_eval, since_and_until)
    else:
        since = since or ''
        if since:
            since = add_ago_to_since(since)
        _since = parse_human_datetime(since) if since else None
        _until = parse_human_datetime(until) if until else parse_human_datetime(_relative_end if isinstance(_relative_end, str) else str(_relative_end))
    if time_shift:
        time_delta_since = parse_past_timedelta(time_shift, _since)
        time_delta_until = parse_past_timedelta(time_shift, _until)
        _since = _since if _since is None else _since - time_delta_since
        _until = _until if _until is None else _until - time_delta_until
    if instant_time_comparison_range:
        from superset import feature_flag_manager
        if feature_flag_manager.is_feature_enabled('CHART_PLUGINS_EXPERIMENTAL'):
            time_unit = ''
            delta_in_days: Optional[int] = None
            if instant_time_comparison_range == InstantTimeComparison.YEAR:
                time_unit = 'YEAR'
            elif instant_time_comparison_range == InstantTimeComparison.MONTH:
                time_unit = 'MONTH'
            elif instant_time_comparison_range == InstantTimeComparison.WEEK:
                time_unit = 'WEEK'
            elif instant_time_comparison_range == InstantTimeComparison.INHERITED:
                delta_in_days = (_until - _since).days if _since and _until else None
                time_unit = 'DAY'
            if time_unit:
                strtfime_since = _since.strftime('%Y-%m-%dT%H:%M:%S') if _since else str(_relative_start)
                strtfime_until = _until.strftime('%Y-%m-%dT%H:%M:%S') if _until else str(_relative_end)
                since_and_until = [
                    f"DATEADD(DATETIME('{strtfime_since}'), -{delta_in_days or 1}, {time_unit})",
                    f"DATEADD(DATETIME('{strtfime_until}'), -{delta_in_days or 1}, {time_unit})"
                ]
                _since, _until = map(datetime_eval, since_and_until)
    if _since and _until and (_since > _until):
        raise ValueError(_('From date cannot be larger than to date'))
    return (_since, _until)

def add_ago_to_since(since: str) -> str:
    """
    Backwards compatibility hack. Without this slices with since: 7 days will
    be treated as 7 days in the future.
    """
    since_words = since.split(' ')
    grains = ['days', 'years', 'hours', 'day', 'year', 'weeks']
    if len(since_words) == 2 and since_words[1] in grains:
        since += ' ago'
    return since

class EvalText:
    def __init__(self, tokens: ParseResults) -> None:
        self.value: str = tokens[0]

    def eval(self) -> str:
        return self.value[1:-1]

class EvalDateTimeFunc:
    def __init__(self, tokens: ParseResults) -> None:
        self.value: Any = tokens[1]

    def eval(self) -> datetime:
        return parse_human_datetime(self.value.eval())

class EvalDateAddFunc:
    def __init__(self, tokens: ParseResults) -> None:
        self.value: Any = tokens[1]

    def eval(self) -> datetime:
        dttm_expression, delta, unit = self.value
        dttm = dttm_expression.eval()
        delta = delta.eval() if hasattr(delta, 'eval') else delta
        if unit.lower() == 'quarter':
            delta = delta * 3
            unit = 'month'
        return dttm + parse_human_timedelta(f'{delta} {unit}s', dttm)

class EvalDateDiffFunc:
    def __init__(self, tokens: ParseResults) -> None:
        self.value: Any = tokens[1]

    def eval(self) -> int:
        if len(self.value) == 2:
            _dttm_from, _dttm_to = self.value
            return (_dttm_to.eval() - _dttm_from.eval()).days
        if len(self.value) == 3:
            _dttm_from, _dttm_to, _unit = self.value
            if _unit == 'year':
                return _dttm_to.eval().year - _dttm_from.eval().year
            if _unit == 'day':
                return (_dttm_to.eval() - _dttm_from.eval()).days
        raise ValueError(_('Unable to calculate such a date delta'))

class EvalDateTruncFunc:
    def __init__(self, tokens: ParseResults) -> None:
        self.value: Any = tokens[1]

    def eval(self) -> datetime:
        dttm_expression, unit = self.value
        dttm = dttm_expression.eval()
        if unit == 'year':
            dttm = dttm.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit == 'quarter':
            dttm = pd.Period(pd.Timestamp(dttm), freq='Q').to_timestamp().to_pydatetime()
        elif unit == 'month':
            dttm = dttm.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif unit == 'week':
            dttm -= relativedelta(days=dttm.weekday())
            dttm = dttm.replace(hour=0, minute=0, second=0, microsecond=0)
        elif unit == 'day':
            dttm = dttm.replace(hour=0, minute=0, second=0, microsecond=0)
        elif unit == 'hour':
            dttm = dttm.replace(minute=0, second=0, microsecond=0)
        elif unit == 'minute':
            dttm = dttm.replace(second=0, microsecond=0)
        else:
            dttm = dttm.replace(microsecond=0)
        return dttm

class EvalLastDayFunc:
    def __init__(self, tokens: ParseResults) -> None:
        self.value: Any = tokens[1]

    def eval(self) -> datetime:
        dttm_expression, unit = self.value
        dttm = dttm_expression.eval()
        if unit == 'year':
            return dttm.replace(month=12, day=31, hour=0, minute=0, second=0, microsecond=0)
        if unit == 'month':
            return dttm.replace(day=calendar.monthrange(dttm.year, dttm.month)[1], hour=0, minute=0, second=0, microsecond=0)
        mon = dttm - relativedelta(days=dttm.weekday())
        mon = mon.replace(hour=0, minute=0, second=0, microsecond=0)
        return mon + relativedelta(days=6)

class EvalHolidayFunc:
    def __init__(self, tokens: ParseResults) -> None:
        self.value: Any = tokens[1]

    def eval(self) -> datetime:
        holiday = self.value[0].eval()
        dttm, country = [None, None]  # type: Tuple[Optional[datetime], Optional[Any]]
        if len(self.value) >= 2:
            dttm = self.value[1].eval()
        if len(self.value) == 3:
            country = self.value[2]
        holiday_year = dttm.year if dttm else parse_human_datetime('today').year
        country = country.eval() if country else 'US'
        holiday_lookup = country_holidays(country, years=[holiday_year], observed=False)
        searched_result = holiday_lookup.get_named(holiday, lookup='istartswith')
        if len(searched_result) > 0:
            return dttm_from_timetuple(searched_result[0].timetuple())
        raise ValueError(_('Unable to find such a holiday: [%(holiday)s]', holiday=holiday))

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def datetime_parser() -> ParserElement:
    DATETIME, DATEADD, DATEDIFF, DATETRUNC, LASTDAY, HOLIDAY, YEAR, QUARTER, MONTH, WEEK, DAY, HOUR, MINUTE, SECOND = map(
        CaselessKeyword, 'datetime dateadd datediff datetrunc lastday holiday year quarter month week day hour minute second'.split()
    )
    lparen, rparen, comma = map(Suppress, '(),')
    text_operand = quotedString.setName('text_operand').setParseAction(EvalText)
    datetime_func: Forward = Forward().setName('datetime')
    dateadd_func: Forward = Forward().setName('dateadd')
    datetrunc_func: Forward = Forward().setName('datetrunc')
    lastday_func: Forward = Forward().setName('lastday')
    holiday_func: Forward = Forward().setName('holiday')
    date_expr = datetime_func | dateadd_func | datetrunc_func | lastday_func | holiday_func
    datediff_func: Forward = Forward().setName('datediff')
    int_operand = pyparsing_common.signed_integer().setName('int_operand') | datediff_func
    datetime_func <<= (DATETIME + lparen + text_operand + rparen).setParseAction(EvalDateTimeFunc)
    dateadd_func <<= (DATEADD + lparen + Group(date_expr + comma + int_operand + comma +
                                                (YEAR | QUARTER | MONTH | WEEK | DAY | HOUR | MINUTE | SECOND) +
                                                ppOptional(comma)) + rparen).setParseAction(EvalDateAddFunc)
    datetrunc_func <<= (DATETRUNC + lparen + Group(date_expr + comma +
                                                   (YEAR | QUARTER | MONTH | WEEK | DAY | HOUR | MINUTE | SECOND) +
                                                   ppOptional(comma)) + rparen).setParseAction(EvalDateTruncFunc)
    lastday_func <<= (LASTDAY + lparen + Group(date_expr + comma + (YEAR | MONTH | WEEK) + ppOptional(comma)) + rparen).setParseAction(EvalLastDayFunc)
    holiday_func <<= (HOLIDAY + lparen + Group(text_operand +
                                                ppOptional(comma) +
                                                ppOptional(date_expr) +
                                                ppOptional(comma) +
                                                ppOptional(text_operand) +
                                                ppOptional(comma)) + rparen).setParseAction(EvalHolidayFunc)
    datediff_func <<= (DATEDIFF + lparen + Group(date_expr + comma + date_expr + ppOptional(comma + (YEAR | DAY) + ppOptional(comma))) + rparen).setParseAction(EvalDateDiffFunc)
    return date_expr | datediff_func

def datetime_eval(datetime_expression: Optional[str] = None) -> Optional[datetime]:
    if datetime_expression:
        try:
            return datetime_parser().parseString(datetime_expression)[0].eval()
        except ParseException as ex:
            raise ValueError(ex) from ex
    return None

class DateRangeMigration:
    x_dateunit_in_since: str = '"time_range":\\s*"\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*\\s:\\s'
    x_dateunit_in_until: str = '"time_range":\\s*".*\\s:\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*"'
    x_dateunit: str = '^\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*$'