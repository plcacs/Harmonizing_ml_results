"""
Utility to format Raiden json logs into HTML.
Colorizes log key-values according to their md5 hash to make debugging easier.
Allows to filter records by `event`.
"""
import hashlib
import json
import webbrowser
from collections import Counter, namedtuple
from copy import copy
from datetime import datetime
from html import escape
from itertools import chain
from json import JSONDecodeError
from math import log10
from pathlib import Path
from typing import Any, Counter as CounterType, Dict, Generator, Iterable, List, Optional, Set, TextIO, Tuple, Union, cast
import click
from cachetools import LRUCache, cached
from click import UsageError
from click._compat import _default_text_stderr
from colour import Color
from eth_utils import is_address, to_canonical_address
from raiden.utils.formatting import pex

TIME_PAST: datetime = datetime(1970, 1, 1)
TIME_FUTURE: datetime = datetime(9999, 1, 1)
COLORMAP: List[str] = ['#440154', '#440256', '#450457', '#450559', '#46075a', '#46085c', '#460a5d', '#460b5e', '#470d60', '#470e61', '#471063', '#471164', '#471365', '#481467', '#481668', '#481769', '#48186a', '#481a6c', '#481b6d', '#481c6e', '#481d6f', '#481f70', '#482071', '#482173', '#482374', '#482475', '#482576', '#482677', '#482878', '#482979', '#472a7a', '#472c7a', '#472d7b', '#472e7c', '#472f7d', '#46307e', '#46327e', '#46337f', '#463480', '#453581', '#453781', '#453882', '#443983', '#443a83', '#443b84', '#433d84', '#433e85', '#423f85', '#424086', '#424186', '#414287', '#414487', '#404588', '#404688', '#3f4788', '#3f4889', '#3e4989', '#3e4a89', '#3e4c8a', '#3d4d8a', '#3d4e8a', '#3c4f8a', '#3c508b', '#3b518b', '#3b528b', '#3a538b', '#3a548c', '#39558c', '#39568c', '#38588c', '#38598c', '#375a8c', '#375b8d', '#365c8d', '#365d8d', '#355e8d', '#355f8d', '#34608d', '#34618d', '#33628d', '#33638d', '#32648e', '#32658e', '#31668e', '#31678e', '#31688e', '#30698e', '#306a8e', '#2f6b8e', '#2f6c8e', '#2e6d8e', '#2e6e8e', '#2e6f8e', '#2d708e', '#2d718e', '#2c718e', '#2c728e', '#2c738e', '#2b748e', '#2b758e', '#2a768e', '#2a778e', '#2a788e', '#29798e', '#297a8e', '#297b8e', '#287c8e', '#287d8e', '#277e8e', '#277f8e', '#27808e', '#26818e', '#26828e', '#26828e', '#25838e', '#25848e', '#25858e', '#24868e', '#24878e', '#23888e', '#23898e', '#238a8d', '#228b8d', '#228c8d', '#228d8d', '#218e8d', '#218f8d', '#21908d', '#21918c', '#20928c', '#20928c', '#20938c', '#1f948c', '#1f958b', '#1f968b', '#1f978b', '#1f988b', '#1f998a', '#1f9a8a', '#1e9b8a', '#1e9c89', '#1e9d89', '#1f9e89', '#1f9f88', '#1fa088', '#1fa188', '#1fa187', '#1fa287', '#20a386', '#20a486', '#21a585', '#21a685', '#22a785', '#22a884', '#23a983', '#24aa83', '#25ab82', '#25ac82', '#26ad81', '#27ad81', '#28ae80', '#29af7f', '#2ab07f', '#2cb17e', '#2db27d', '#2eb37c', '#2fb47c', '#31b57b', '#32b67a', '#34b679', '#35b779', '#37b878', '#38b977', '#3aba76', '#3bbb75', '#3dbc74', '#3fbc73', '#40bd72', '#42be71', '#44bf70', '#46c06f', '#48c16e', '#4ac16d', '#4cc26c', '#4ec36b', '#50c46a', '#52c569', '#54c568', '#56c667', '#58c765', '#5ac864', '#5cc863', '#5ec962', '#60ca60', '#63cb5f', '#65cb5e', '#67cc5c', '#69cd5b', '#6ccd5a', '#6ece58', '#70cf57', '#73d056', '#75d054', '#77d153', '#7ad151', '#7cd250', '#7fd34e', '#81d34d', '#84d44b', '#86d549', '#89d548', '#8bd646', '#8ed645', '#90d743', '#93d741', '#95d840', '#98d83e', '#9bd93c', '#9dd93b', '#a0da39', '#a2da37', '#a5db36', '#a8db34', '#aadc32', '#addc30', '#b0dd2f', '#b2dd2d', '#b5de2b', '#b8de29', '#bade28', '#bddf26', '#c0df25', '#c2df23', '#c5e021', '#c8e020', '#cae11f', '#cde11d', '#d0e11c', '#d2e21b', '#d5e21a', '#d8e219', '#dae319', '#dde318', '#dfe318', '#e2e418', '#e5e419', '#e7e419', '#eae51a', '#ece51b', '#efe51c', '#f1e51d', '#f4e61e', '#f6e620', '#f8e621', '#fbe723', '#fde725']
PAGE_BEGIN: str = '<!doctype html>\n<html>\n<head>\n<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n<style>\n* {{\n    font-family: "Fira Code", "Anonymous Pro", "Inconsolata", Menlo, "Source Code Pro",\n        "Envy Code R", Hack, "Ubuntu Mono", "Droid Sans Mono", "Deja Vu Sans Mono", "Courier New",\n        Courier;\n    font-size: small;\n}}\nbody {{\n    background: #202020;\n    color: white;\n}}\ntable {{\n    border: none;\n}}\ntable tr.head {{\n    position: sticky;\n}}\ntable tr:nth-child(2) {{\n    padding-top: 15px;\n}}\ntable tr:nth-child(odd) {{\n    background-color: #303030;\n}}\ntable tr:nth-child(even) {{\n    background-color: #202020;\n}}\ntable tr td:first-child {{\n    background-color: inherit;\n    position: sticky;\n    position: -webkit-sticky;\n    left: 8px;\n}}\ntable td {{\n    padding-right: 5px;\n    vertical-align: top;\n}}\ntable tr:hover {{\n    outline: 1px solid #d02020;\n}}\ntable tr.highlight {{\n    outline: 1px solid #20d020;\n}}\ntd.no, td.time * {{\n    white-space: pre;\n    font-family: courier;\n}}\ntd.no {{\n    text-align: right;\n}}\ntd.event {{\n    min-width: 20em;\n    max-width: 35em;\n}}\ntd.fields {{\n    white-space: nowrap;\n}}\n.lvl-debug {{\n    color: #20d0d0;\n}}\n.lvl-info {{\n    color: #20d020;\n}}\n.lvl-warning {{\n    color: #d0d020;\n}}\n.lvl-error {{\n    color: #d04020;\n}}\n.lvl-critical {{\n    color: #f1053c;\n}}\n.fn {{\n    color: #f040f0;\n}}\n</style>\n<body>\n<h1>{name}</h1>\n<h2>Generated on: {date:%Y-%m-%d %H:%M}</h2>\n<table>\n<tr class="head">\n   <td>No</td>\n   <td>Event</td>\n   <td>Timestamp</td>\n   <td>Logger</td>\n   <td>Level</td>\n   <td>Fields</td>\n</tr>\n'
PAGE_END: str = '</table>\n</body>\n</html>\n'
ROW_TEMPLATE: str = '\n<tr class="lvl-{record.level} {additional_row_class}">\n    <td class="no">{index}</td>\n    <td class="event"><b style="color: {event_color}">{record.event}</b></td>\n    <td class="time"><span title="{time_absolute}" style="{time_color}">{time_display}</span></td>\n    <td><span title="{record.logger}">{logger_name}</span></td>\n    <td>{record.level}</td>\n    <td class="fields">{fields}</td>\n</tr>\n'
Record = namedtuple('Record', ('event', 'timestamp', 'logger', 'truncated_logger', 'level', 'fields'))

@cached(LRUCache(maxsize=1000))
def truncate_logger_name(logger: str) -> str:
    """Truncate dotted logger path names.

    Keeps the last component unchanged.

    >>> truncate_logger_name("some.logger.name")
    s.l.name

    >>> truncate_logger_name("name")
    name
    """
    if '.' not in logger:
        return logger
    logger_path, _, logger_module = logger.rpartition('.')
    return '.'.join(chain((part[0] for part in logger_path.split('.')), [logger_module]))

def _colorize_cache_key(value: Any, min_luminance: Optional[float]) -> Tuple[Any, Optional[float]]:
    if isinstance(value, (list, dict)):
        return (repr(value), min_luminance)
    return (value, min_luminance)

@cached(LRUCache(maxsize=2 ** 24))
def rgb_color_picker(obj: Any, min_luminance: Optional[float] = None, max_luminance: Optional[float] = None) -> Color:
    """Modified version of colour.RGB_color_picker"""
    color_value = int.from_bytes(hashlib.md5(str(obj).encode('utf-8')).digest(), 'little') % 16777215
    color = Color(f'#{color_value:06x}')
    if min_luminance and color.get_luminance() < min_luminance:
        color.set_luminance(min_luminance)
    elif max_luminance and color.get_luminance() > max_luminance:
        color.set_luminance(max_luminance)
    return color

def nice_time_diff(time_base: datetime, time_now: datetime) -> Tuple[str, float]:
    delta = time_now - time_base
    total_seconds = delta.total_seconds()
    if total_seconds < 0.001:
        return (f'+ {delta.microseconds: 10.0f} µs', total_seconds)
    if total_seconds < 1:
        return (f'+ {delta.microseconds / 1000: 10.3f} ms', total_seconds)
    if total_seconds < 10:
        formatted_seconds = f'{total_seconds: 9.6f}'
        formatted_seconds = f'{formatted_seconds[:6]} {formatted_seconds[6:]}'
        return (f'+ {formatted_seconds} s', total_seconds)
    return (time_now.isoformat(), total_seconds)

def get_time_display(prev_record: Optional[Record], record: Record) -> Tuple[str, str, str]:
    time_absolute = record.timestamp.isoformat()
    time_color = ''
    if prev_record:
        time_display, delta_seconds = nice_time_diff(prev_record.timestamp, record.timestamp)
        if delta_seconds <= 10:
            if delta_seconds < 0.0001:
                time_color_value = COLORMAP[0]
            elif delta_seconds < 1:
                duration_value = delta_seconds * 1000000 / 100
                time_color_value = COLORMAP[int(log10(duration_value) / 4 * 255)]
            else:
                time_color_value = COLORMAP[-1]
            time_color = f'color: {time_color_value}'
    else:
        time_display = time_absolute
    return (time_absolute, time_color, time_display)

@cached(LRUCache(maxsize=2 ** 24), key=_colorize_cache_key)
def colorize_value(value: Any, min_luminance: float) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)((colorize_value(inner, min_luminance) for inner in value))
    elif isinstance(value, dict):
        return {colorize_value(k, min_luminance): colorize_value(v, min_luminance) for k, v in value.items()}
    str_value = str(value)
    color = rgb_color_picker(str_value, min_luminance=min_luminance)
    return f'<span style="color: {color.web}">{escape(str_value)}</span>'

def render_fields(record: Record, sorted_known_fields: List[str]) -> List[str]:
    rendered_fields = []
    for field_name in sorted_known_fields:
        if field_name not in record.fields:
            continue
        field_value = record.fields[field_name]
        colorized_value = str(colorize_value(field_value, min_luminance=0.6))
        rendered_fields.append(f'<span class="fn">{field_name}</span> = {colorized_value}')
    return rendered_fields

def parse_log(log_file: TextIO) -> Tuple[List[Record], CounterType[str]]:
    known_fields: CounterType[str] = Counter()
    log_records: List[Record] = []
    last_ts = TIME_PAST
    for i, line in enumerate(log_file, start=1):
        try:
            line_dict = json.loads(line.strip())
        except JSONDecodeError as ex:
            click.secho(f'Error parsing line {i}: {ex}')
            continue
        timestamp_str = line_dict.pop('timestamp', None)
        if timestamp_str:
            timestamp = last_ts = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = last_ts
        logger_name = line_dict.pop('logger', 'MISSING')
        log_records.append(Record(line_dict.pop('event'), timestamp, logger_name, truncate_logger_name(logger_name), line_dict.pop('level', 'MISSING'), line_dict))
        for field_name in line_dict.keys():
            known_fields[field_name] += 1
    return (log_records, known_fields)

def filter_records(log_records: Iterable[Record], *, drop_events: Optional[Set[str]], keep_events: Optional[Set[str]], drop_loggers: Set[str], time_range: Tuple[datetime, datetime]) -> Generator[Optional[Record], None, None]:
    time_from, time_to = time_range
    for record in log_records:
        event_name = record.event.lower()
        drop = (drop_events and event_name in drop_events or (keep_events and event_name not in keep_events)) or record.logger in drop_loggers or record.timestamp < time_from or (record.timestamp > time_to)
        if drop:
            yield None
        else:
            yield record

def transform_records(log_records: Iterable[Optional[Record]], replacements: Dict[str, Any]) -> Generator[Optional[Record], None, None]:

    def replace(value: Any) -> Any:
        if isinstance(value, tuple) and hasattr(value, '_fields'):
            return type(value)(*[replace(inner) for inner in value])
        if isinstance(value, (list, tuple)):
            return type(value)((replace(inner) for inner in value))
        elif isinstance(value, dict):
            return {replace(k): replace(v) for k, v in value.items()}
        str_value = str(value).casefold()
        if isinstance(value, str):
            keys_in_value = [key for key in replacement_keys if key in str_value]
            for key in keys_in_value:
                try:
                    repl_start = str_value.index(key)
                except ValueError:
                    continue
                value = f'{value[:repl_start]}{replacements[key]}{value[repl_start + len(key):]}'
                str_value = value.case