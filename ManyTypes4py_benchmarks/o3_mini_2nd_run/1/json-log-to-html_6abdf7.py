#!/usr/bin/env python3
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
from typing import Any, Counter as CounterType, Dict, Generator, Iterable, List, Optional, Set, TextIO, Tuple, Union

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
PAGE_BEGIN: str = (
    '<!doctype html>\n<html>\n<head>\n'
    '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n'
    '<style>\n'
    '* {{\n    font-family: "Fira Code", "Anonymous Pro", "Inconsolata", Menlo, "Source Code Pro",\n'
    '        "Envy Code R", Hack, "Ubuntu Mono", "Droid Sans Mono", "Deja Vu Sans Mono", "Courier New",\n'
    '        Courier;\n    font-size: small;\n}}\n'
    'body {{\n    background: #202020;\n    color: white;\n}}\n'
    'table {{\n    border: none;\n}}\n'
    'table tr.head {{\n    position: sticky;\n}}\n'
    'table tr:nth-child(2) {{\n    padding-top: 15px;\n}}\n'
    'table tr:nth-child(odd) {{\n    background-color: #303030;\n}}\n'
    'table tr:nth-child(even) {{\n    background-color: #202020;\n}}\n'
    'table tr td:first-child {{\n    background-color: inherit;\n    position: sticky;\n    position: -webkit-sticky;\n    left: 8px;\n}}\n'
    'table td {{\n    padding-right: 5px;\n    vertical-align: top;\n}}\n'
    'table tr:hover {{\n    outline: 1px solid #d02020;\n}}\n'
    'table tr.highlight {{\n    outline: 1px solid #20d020;\n}}\n'
    'td.no, td.time * {{\n    white-space: pre;\n    font-family: courier;\n}}\n'
    'td.no {{\n    text-align: right;\n}}\n'
    'td.event {{\n    min-width: 20em;\n    max-width: 35em;\n}}\n'
    'td.fields {{\n    white-space: nowrap;\n}}\n'
    '.lvl-debug {{\n    color: #20d0d0;\n}}\n'
    '.lvl-info {{\n    color: #20d020;\n}}\n'
    '.lvl-warning {{\n    color: #d0d020;\n}}\n'
    '.lvl-error {{\n    color: #d04020;\n}}\n'
    '.lvl-critical {{\n    color: #f1053c;\n}}\n'
    '.fn {{\n    color: #f040f0;\n}}\n'
    '</style>\n'
    '</head>\n<body>\n'
    '<h1>{name}</h1>\n'
    '<h2>Generated on: {date:%Y-%m-%d %H:%M}</h2>\n'
    '<table>\n'
    '<tr class="head">\n   <td>No</td>\n   <td>Event</td>\n   <td>Timestamp</td>\n   <td>Logger</td>\n   <td>Level</td>\n   <td>Fields</td>\n</tr>\n'
)
PAGE_END: str = '</table>\n</body>\n</html>\n'
ROW_TEMPLATE: str = (
    '\n<tr class="lvl-{record.level} {additional_row_class}">\n'
    '    <td class="no">{index}</td>\n'
    '    <td class="event"><b style="color: {event_color}">{record.event}</b></td>\n'
    '    <td class="time"><span title="{time_absolute}" style="{time_color}">{time_display}</span></td>\n'
    '    <td><span title="{record.logger}">{logger_name}</span></td>\n'
    '    <td>{record.level}</td>\n'
    '    <td class="fields">{fields}</td>\n'
    '</tr>\n'
)

Record = namedtuple('Record', ('event', 'timestamp', 'logger', 'truncated_logger', 'level', 'fields'))
# For type annotations
RecordType = Record

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
    color_value: int = int.from_bytes(hashlib.md5(str(obj).encode('utf-8')).digest(), 'little') % 16777215
    color: Color = Color(f'#{color_value:06x}')
    if min_luminance is not None and color.get_luminance() < min_luminance:
        color.set_luminance(min_luminance)
    elif max_luminance is not None and color.get_luminance() > max_luminance:
        color.set_luminance(max_luminance)
    return color

def nice_time_diff(time_base: datetime, time_now: datetime) -> Tuple[str, float]:
    delta = time_now - time_base
    total_seconds: float = delta.total_seconds()
    if total_seconds < 0.001:
        return (f'+ {delta.microseconds: 10.0f} µs', total_seconds)
    if total_seconds < 1:
        return (f'+ {delta.microseconds / 1000: 10.3f} ms', total_seconds)
    if total_seconds < 10:
        formatted_seconds: str = f'{total_seconds: 9.6f}'
        formatted_seconds = f'{formatted_seconds[:6]} {formatted_seconds[6:]}'
        return (f'+ {formatted_seconds} s', total_seconds)
    return (time_now.isoformat(), total_seconds)

def get_time_display(prev_record: Optional[RecordType], record: RecordType) -> Tuple[str, str, str]:
    time_absolute: str = record.timestamp.isoformat()
    time_color: str = ''
    if prev_record:
        time_display, delta_seconds = nice_time_diff(prev_record.timestamp, record.timestamp)
        if delta_seconds <= 10:
            if delta_seconds < 0.0001:
                time_color_value: str = COLORMAP[0]
            elif delta_seconds < 1:
                duration_value: float = delta_seconds * 1000000 / 100
                time_color_value = COLORMAP[int(log10(duration_value) / 4 * 255)]
            else:
                time_color_value = COLORMAP[-1]
            time_color = f'color: {time_color_value}'
    else:
        time_display = time_absolute
    return (time_absolute, time_color, time_display)

@cached(LRUCache(maxsize=2 ** 24), key=_colorize_cache_key)
def colorize_value(value: Any, min_luminance: Optional[float]) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)(colorize_value(inner, min_luminance) for inner in value)
    elif isinstance(value, dict):
        return {colorize_value(k, min_luminance): colorize_value(v, min_luminance) for k, v in value.items()}
    str_value: str = str(value)
    color: Color = rgb_color_picker(str_value, min_luminance=min_luminance)
    return f'<span style="color: {color.web}">{escape(str_value)}</span>'

def render_fields(record: RecordType, sorted_known_fields: List[str]) -> List[str]:
    rendered_fields: List[str] = []
    for field_name in sorted_known_fields:
        if field_name not in record.fields:
            continue
        field_value: Any = record.fields[field_name]
        colorized_value: str = str(colorize_value(field_value, min_luminance=0.6))
        rendered_fields.append(f'<span class="fn">{field_name}</span> = {colorized_value}')
    return rendered_fields

def parse_log(log_file: TextIO) -> Tuple[List[RecordType], CounterType[str]]:
    known_fields: CounterType[str] = Counter()
    log_records: List[RecordType] = []
    last_ts: datetime = TIME_PAST
    for i, line in enumerate(log_file, start=1):
        try:
            line_dict: Dict[str, Any] = json.loads(line.strip())
        except JSONDecodeError as ex:
            click.secho(f'Error parsing line {i}: {ex}')
            continue
        timestamp_str: Optional[str] = line_dict.pop('timestamp', None)
        if timestamp_str:
            timestamp: datetime = datetime.fromisoformat(timestamp_str)
            last_ts = timestamp
        else:
            timestamp = last_ts
        logger_name: str = line_dict.pop('logger', 'MISSING')
        record = Record(
            event=line_dict.pop('event'),
            timestamp=timestamp,
            logger=logger_name,
            truncated_logger=truncate_logger_name(logger_name),
            level=line_dict.pop('level', 'MISSING'),
            fields=line_dict
        )
        log_records.append(record)
        for field_name in line_dict.keys():
            known_fields[field_name] += 1
    return (log_records, known_fields)

def filter_records(
    log_records: Iterable[RecordType],
    *,
    drop_events: Set[str],
    keep_events: Set[str],
    drop_loggers: Set[str],
    time_range: Tuple[datetime, datetime]
) -> Generator[Optional[RecordType], None, None]:
    time_from, time_to = time_range
    for record in log_records:
        event_name: str = record.event.lower()
        drop: bool = (
            (drop_events and event_name in drop_events or (keep_events and event_name not in keep_events))
            or record.logger.lower() in drop_loggers
            or record.timestamp < time_from
            or record.timestamp > time_to
        )
        if drop:
            yield None
        else:
            yield record

def transform_records(
    log_records: Iterable[RecordType],
    replacements: Dict[Any, Any]
) -> Generator[RecordType, None, None]:
    def replace(value: Any) -> Any:
        if isinstance(value, tuple) and hasattr(value, '_fields'):
            return type(value)(*(replace(inner) for inner in value))
        if isinstance(value, (list, tuple)):
            return type(value)(replace(inner) for inner in value)
        elif isinstance(value, dict):
            return {replace(k): replace(v) for k, v in value.items()}
        str_value: str = str(value).casefold()
        if isinstance(value, str):
            keys_in_value = [key for key in replacement_keys if key in str_value]
            for key in keys_in_value:
                try:
                    repl_start = str_value.index(key)
                except ValueError:
                    continue
                value = f'{value[:repl_start]}{replacements[key]}{value[repl_start + len(key):]}'
                str_value = value.casefold()
        return replacements.get(str_value, value)
    replacements = {str(k).casefold(): v for k, v in replacements.items()}
    for k, v in list(replacements.items()):
        if isinstance(k, str) and k.startswith('0x') and is_address(k):
            bytes_address = to_canonical_address(k)
            replacements[pex(bytes_address)] = v
            replacements[repr(bytes_address).casefold()] = v
    replacement_keys = replacements.keys()
    for record in log_records:
        yield replace(record)

def render(
    name: str,
    log_records: Iterable[RecordType],
    known_fields: CounterType[str],
    output: TextIO,
    wrap: bool = False,
    show_time_diff: bool = True,
    truncate_logger: bool = False,
    highlight_records: Iterable[int] = ()
) -> None:
    sorted_known_fields: List[str] = [name for name, count in known_fields.most_common()]
    highlight_records_set: Set[int] = set(highlight_records) if highlight_records else set()
    prev_record: Optional[RecordType] = None
    output.write(PAGE_BEGIN.format(name=name, date=datetime.now()))
    field_joiner: str = '<br/>' if wrap else ' '
    i: int = 0
    for record in log_records:
        if record is None:
            continue
        time_absolute, time_color, time_display = get_time_display(prev_record, record)
        event_color: Color = rgb_color_picker(record.event, min_luminance=0.6)
        rendered_fields: List[str] = render_fields(record, sorted_known_fields)
        output.write(ROW_TEMPLATE.format(
            index=i,
            record=record,
            logger_name=record.truncated_logger if truncate_logger else record.logger,
            time_absolute=time_absolute,
            time_display=time_display,
            time_color=time_color,
            event_color=event_color,
            fields=field_joiner.join(rendered_fields),
            additional_row_class='highlight' if i in highlight_records_set else ''
        ))
        if show_time_diff:
            prev_record = record
        i += 1
    output.write(PAGE_END)

@click.command(help=__doc__)
@click.argument('log-file', type=click.File('rt'))
@click.option('-o', '--output', type=click.File('wt'), default='-', show_default=True)
@click.option('-e', '--drop-event', 'drop_events', multiple=True, help='Filter out log records with the given event. Case insensitive. Can be given multiple times.')
@click.option('--keep-event', 'keep_events', multiple=True, help='Only keep log records with the given event. Case insensitive. Can be given multiple times. Cannot be used together with with --drop-event.')
@click.option('-l', '--drop-logger', 'drop_loggers', multiple=True, help='Filter out log records with the given logger name. Case insensitive. Can be given multiple times.')
@click.option('-r', '--replacements', help='Replace values before rendering. Input must be a JSON object. Keys are transformed to lowercase strings before matching. Partial substring matches will also be replaced. Eth-Addresses will also be replaced in pex()ed and binary format.')
@click.option('-f', '--replacements-from-file', type=click.File('rt'), help='Behaves as -r / --replacements but reads the JSON object from the given file.')
@click.option('--time-range', default='^', help='Specify a time range of log messages to process. Format: "[<from>]^[<to>]", both in ISO8601')
@click.option('--time-diff/--no-time-diff', default=True, help='Display log record timestamps relative to previous lines (absolute on hover)', show_default=True)
@click.option('-w', '--wrap', is_flag=True, help='Wrap event details into multiple lines.', show_default=True)
@click.option('-t', '--truncate-logger', is_flag=True, show_default=True, help="Shorten logger module paths ('some.module.logger' -> 's.m.logger').")
@click.option('-h', '--highlight-record', 'highlight_records', multiple=True, type=int, help='Highlight record with given number. Can be given multiple times.')
@click.option('-b', '--open-browser', is_flag=True, help='Open output file in default browser after rendering', show_default=True)
def main(
    log_file: TextIO,
    drop_events: Tuple[str, ...],
    keep_events: Tuple[str, ...],
    drop_loggers: Tuple[str, ...],
    replacements: Optional[str],
    replacements_from_file: Optional[TextIO],
    time_range: str,
    wrap: bool,
    time_diff: bool,
    truncate_logger: bool,
    highlight_records: Tuple[int, ...],
    open_browser: bool,
    output: TextIO
) -> None:
    if replacements_from_file:
        replacements = replacements_from_file.read()
    if not replacements:
        replacements = '{}'
    try:
        replacements_dict: Dict[Any, Any] = json.loads(replacements)
    except (JSONDecodeError, UnicodeDecodeError) as ex:
        raise UsageError(f"Option '--replacements' contains invalid JSON: {ex}") from ex
    if drop_events and keep_events:
        raise UsageError("Options '--keep-event' and '--drop-event' cannot be used together.")
    time_from_str, _, time_to_str = time_range.partition('^')
    time_range_dt: Tuple[datetime, datetime] = (
        datetime.fromisoformat(time_from_str) if time_from_str else TIME_PAST,
        datetime.fromisoformat(time_to_str) if time_to_str else TIME_FUTURE
    )
    click.secho(f'Processing {click.style(log_file.name, fg="yellow")}', fg='green')
    log_records, known_fields = parse_log(log_file)
    prog_bar = click.progressbar(log_records, label=click.style('Rendering', fg='green'), file=_default_text_stderr())
    with prog_bar as log_records_progr:
        render(
            log_file.name,
            transform_records(
                filter_records(
                    log_records_progr,
                    drop_events=set((d.lower() for d in drop_events)),
                    keep_events=set((k.lower() for k in keep_events)),
                    drop_loggers=set((logger.lower() for logger in drop_loggers)),
                    time_range=time_range_dt
                ),
                replacements=replacements_dict
            ),
            known_fields=known_fields,
            output=output,
            wrap=wrap,
            show_time_diff=time_diff,
            truncate_logger=truncate_logger,
            highlight_records=highlight_records
        )
    click.secho(f'Output written to {click.style(output.name, fg="yellow")}', fg='green')
    if open_browser:
        if output.name == '<stdout>':
            click.secho("Can't open output when writing to stdout.", fg='red')
        else:
            webbrowser.open(f'file://{Path(output.name).resolve()}')

if __name__ == '__main__':
    main()