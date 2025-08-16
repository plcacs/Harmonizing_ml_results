from __future__ import annotations
import calendar
import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache
from time import struct_time
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
    ...

def normalize_time_delta(human_readable: str) -> dict:
    ...

def dttm_from_timetuple(date_: struct_time) -> datetime:
    ...

def get_past_or_future(human_readable: str, source_time: datetime = None) -> datetime:
    ...

def parse_human_timedelta(human_readable: str, source_time: datetime = None) -> timedelta:
    ...

def parse_past_timedelta(delta_str: str, source_time: datetime = None) -> timedelta:
    ...

def get_relative_base(unit: str, relative_start: datetime = None) -> datetime:
    ...

def handle_start_of(base_expression: str, unit: str) -> str:
    ...

def handle_end_of(base_expression: str, unit: str) -> str:
    ...

def handle_modifier_and_unit(modifier: str, scope: str, delta: str, unit: str, relative_base: str) -> str:
    ...

def handle_scope_and_unit(scope: str, delta: str, unit: str, relative_base: str) -> str:
    ...

def get_since_until(time_range: str = None, since: str = None, until: str = None, time_shift: str = None, relative_start: str = None, relative_end: str = None, instant_time_comparison_range: InstantTimeComparison = None) -> tuple[datetime, datetime]:
    ...

def add_ago_to_since(since: str) -> str:
    ...

class EvalText:
    def __init__(self, tokens: ParseResults):
        ...
    def eval(self) -> str:
        ...

class EvalDateTimeFunc:
    def __init__(self, tokens: ParseResults):
        ...
    def eval(self) -> datetime:
        ...

class EvalDateAddFunc:
    def __init__(self, tokens: ParseResults):
        ...
    def eval(self) -> datetime:
        ...

class EvalDateDiffFunc:
    def __init__(self, tokens: ParseResults):
        ...
    def eval(self) -> int:
        ...

class EvalDateTruncFunc:
    def __init__(self, tokens: ParseResults):
        ...
    def eval(self) -> datetime:
        ...

class EvalLastDayFunc:
    def __init__(self, tokens: ParseResults):
        ...
    def eval(self) -> datetime:
        ...

class EvalHolidayFunc:
    def __init__(self, tokens: ParseResults):
        ...
    def eval(self) -> datetime:
        ...

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def datetime_parser() -> ParserElement:
    ...

def datetime_eval(datetime_expression: str = None) -> datetime:
    ...

class DateRangeMigration:
    x_dateunit_in_since: str = '"time_range":\\s*"\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*\\s:\\s'
    x_dateunit_in_until: str = '"time_range":\\s*".*\\s:\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*"'
    x_dateunit: str = '^\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*$'
