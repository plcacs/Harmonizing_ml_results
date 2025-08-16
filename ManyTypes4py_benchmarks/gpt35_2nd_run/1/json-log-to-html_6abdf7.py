import hashlib
import json
import webbrowser
from collections import Counter
from copy import copy
from datetime import datetime
from html import escape
from itertools import chain
from json import JSONDecodeError
from math import log10
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, TextIO, Tuple, Union

import click
from cachetools import LRUCache, cached
from click import UsageError
from click._compat import _default_text_stderr
from colour import Color
from eth_utils import is_address, to_canonical_address
from raiden.utils.formatting import pex

TIME_PAST = datetime(1970, 1, 1)
TIME_FUTURE = datetime(9999, 1, 1)
COLORMAP = ['#440154', '#440256', '#450457', '#450559', '#46075a', '#46085c', '#460a5d', '#460b5e', '#470d60', '#470e61', '#471063', '#471164', '#471365', '#481467', '#481668', '#481769', '#48186a', '#481a6c', '#481b6d', '#481c6e', '#481d6f', '#481f70', '#482071', '#482173', '#482374', '#482475', '#482576', '#482677', '#482878', '#482979', '#472a7a', '#472c7a', '#472d7b', '#472e7c', '#472f7d', '#46307e', '#46327e', '#46337f', '#463480', '#453581', '#453781', '#453882', '#443983', '#443a83', '#443b84', '#433d84', '#433e85', '#423f85', '#424086', '#424186', '#414287', '#414487', '#404588', '#404688', '#3f4788', '#3f4889', '#3e4989', '#3e4a89', '#3e4c8a', '#3d4d8a', '#3d4e8a', '#3c4f8a', '#3c508b', '#3b518b', '#3b528b', '#3a538b', '#3a548c', '#39558c', '#39568c', '#38588c', '#38598c', '#375a8c', '#375b8d', '#365c8d', '#365d8d', '#355e8d', '#355f8d', '#34608d', '#34618d', '#33628d', '#33638d', '#32648e', '#32658e', '#31668e', '#31678e', '#31688e', '#30698e', '#306a8e', '#2f6b8e', '#2f6c8e', '#2e6d8e', '#2e6e8e', '#2e6f8e', '#2d708e', '#2d718e', '#2c718e', '#2c728e', '#2c738e', '#2b748e', '#2b758e', '#2a768e', '#2a778e', '#2a788e', '#29798e', '#297a8e', '#297b8e', '#287c8e', '#287d8e', '#277e8e', '#277f8e', '#27808e', '#26818e', '#26828e', '#26828e', '#25838e', '#25848e', '#25858e', '#24868e', '#24878e', '#23888e', '#23898e', '#238a8d', '#228b8d', '#228c8d', '#228d8d', '#218e8d', '#218f8d', '#21908d', '#21918c', '#20928c', '#20928c', '#20938c', '#1f948c', '#1f958b', '#1f968b', '#1f978b', '#1f988b', '#1f998a', '#1f9a8a', '#1e9b8a', '#1e9c89', '#1e9d89', '#1f9e89', '#1f9f88', '#1fa088', '#1fa188', '#1fa187', '#1fa287', '#20a386', '#20a486', '#21a585', '#21a685', '#22a785', '#22a884', '#23a983', '#24aa83', '#25ab82', '#25ac82', '#26ad81', '#27ad81', '#28ae80', '#29af7f', '#2ab07f', '#2cb17e', '#2db27d', '#2eb37c', '#2fb47c', '#31b57b', '#32b67a', '#34b679', '#35b779', '#37b878', '#38b977', '#3aba76', '#3bbb75', '#3dbc74', '#3fbc73', '#40bd72', '#42be71', '#44bf70', '#46c06f', '#48c16e', '#4ac16d', '#4cc26c', '#4ec36b', '#50c46a', '#52c569', '#54c568', '#56c667', '#58c765', '#5ac864', '#5cc863', '#5ec962', '#60ca60', '#63cb5f', '#65cb5e', '#67cc5c', '#69cd5b', '#6ccd5a', '#6ece58', '#70cf57', '#73d056', '#75d054', '#77d153', '#7ad151', '#7cd250', '#7fd34e', '#81d34d', '#84d44b', '#86d549', '#89d548', '#8bd646', '#8ed645', '#90d743', '#93d741', '#95d840', '#98d83e', '#9bd93c', '#9dd93b', '#a0da39', '#a2da37', '#a5db36', '#a8db34', '#aadc32', '#addc30', '#b0dd2f', '#b2dd2d', '#b5de2b', '#b8de29', '#bade28', '#bddf26', '#c0df25', '#c2df23', '#c5e021', '#c8e020', '#cae11f', '#cde11d', '#d0e11c', '#d2e21b', '#d5e21a', '#d8e219', '#dae319', '#dde318', '#dfe318', '#e2e418', '#e5e419', '#e7e419', '#eae51a', '#ece51b', '#efe51c', '#f1e51d', '#f4e61e', '#f6e620', '#f8e621', '#fbe723', '#fde725']
PAGE_BEGIN = '<!doctype html>\n<html>\n<head>\n<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n<style>\n* {{\n    font-family: "Fira Code", "Anonymous Pro", "Inconsolata", Menlo, "Source Code Pro",\n        "Envy Code R", Hack, "Ubuntu Mono", "Droid Sans Mono", "Deja Vu Sans Mono", "Courier New",\n        Courier;\n    font-size: small;\n}}\nbody {{\n    background: #202020;\n    color: white;\n}}\ntable {{\n    border: none;\n}}\ntable tr.head {{\n    position: sticky;\n}}\ntable tr:nth-child(2) {{\n    padding-top: 15px;\n}}\ntable tr:nth-child(odd) {{\n    background-color: #303030;\n}}\ntable tr:nth-child(even) {{\n    background-color: #202020;\n}}\ntable tr td:first-child {{\n    background-color: inherit;\n    position: sticky;\n    position: -webkit-sticky;\n    left: 8px;\n}}\ntable td {{\n    padding-right: 5px;\n    vertical-align: top;\n}}\ntable tr:hover {{\n    outline: 1px solid #d02020;\n}}\ntable tr.highlight {{\n    outline: 1px solid #20d020;\n}}\ntd.no, td.time * {{\n    white-space: pre;\n    font-family: courier;\n}}\ntd.no {{\n    text-align: right;\n}}\ntd.event {{\n    min-width: 20em;\n    max-width: 35em;\n}}\ntd.fields {{\n    white-space: nowrap;\n}}\n.lvl-debug {{\n    color: #20d0d0;\n}}\n.lvl-info {{\n    color: #20d020;\n}}\n.lvl-warning {{\n    color: #d0d020;\n}}\n.lvl-error {{\n    color: #d04020;\n}}\n.lvl-critical {{\n    color: #f1053c;\n}}\n.fn {{\n    color: #f040f0;\n}}\n</style>\n<body>\n<h1>{name}</h1>\n<h2>Generated on: {date:%Y-%m-%d %H:%M}</h2>\n<table>\n<tr class="head">\n   <td>No</td>\n   <td>Event</td>\n   <td>Timestamp</td>\n   <td>Logger</td>\n   <td>Level</td>\n   <td>Fields</td>\n</tr>\n'
PAGE_END = '</table>\n</body>\n</html>\n'
ROW_TEMPLATE = '\n<tr class="lvl-{record.level} {additional_row_class}">\n    <td class="no">{index}</td>\n    <td class="event"><b style="color: {event_color}">{record.event}</b></td>\n    <td class="time"><span title="{time_absolute}" style="{time_color}">{time_display}</span></td>\n    <td><span title="{record.logger}">{logger_name}</span></td>\n    <td>{record.level}</td>\n    <td class="fields">{fields}</td>\n</tr>\n'
Record = namedtuple('Record', ('event', 'timestamp', 'logger', 'truncated_logger', 'level', 'fields'))

@cached(LRUCache(maxsize=1000))
def truncate_logger_name(logger: str) -> str:
    ...

def _colorize_cache_key(value: Any, min_luminance: Optional[float]) -> Tuple[Any, Optional[float]]:
    ...

@cached(LRUCache(maxsize=2 ** 24))
def rgb_color_picker(obj: Any, min_luminance: Optional[float] = None, max_luminance: Optional[float] = None) -> Color:
    ...

def nice_time_diff(time_base: datetime, time_now: datetime) -> Tuple[str, float]:
    ...

def get_time_display(prev_record: Optional[Record], record: Record) -> Tuple[str, str, str]:
    ...

@cached(LRUCache(maxsize=2 ** 24), key=_colorize_cache_key)
def colorize_value(value: Any, min_luminance: float) -> str:
    ...

def render_fields(record: Record, sorted_known_fields: List[str]) -> List[str]:
    ...

def parse_log(log_file: TextIO) -> Tuple[List[Record], Counter]:
    ...

def filter_records(log_records: List[Record], drop_events: Set[str], keep_events: Set[str], drop_loggers: Set[str], time_range: Tuple[datetime, datetime]) -> Generator[Optional[Record], None, None]:
    ...

def transform_records(log_records: List[Record], replacements: Dict[str, Any]) -> Generator[Record, None, None]:
    ...

def render(name: str, log_records: Iterable[Record], known_fields: Counter, output: TextIO, wrap: bool = False, show_time_diff: bool = True, truncate_logger: bool = False, highlight_records: List[int] = []) -> None:
    ...

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
def main(log_file: TextIO, drop_events: Set[str], keep_events: Set[str], drop_loggers: Set[str], replacements: str, replacements_from_file: TextIO, time_range: str, wrap: bool, time_diff: bool, truncate_logger: bool, highlight_records: List[int], open_browser: bool, output: TextIO) -> None:
    ...

if __name__ == '__main__':
    main()
