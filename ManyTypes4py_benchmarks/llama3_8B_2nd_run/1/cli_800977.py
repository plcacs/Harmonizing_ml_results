import errno
import json
import os
import re
from abc import ABCMeta, abstractmethod
from enum import EnumMeta
from itertools import groupby
from json.decoder import JSONDecodeError
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Set

import click
import requests
import structlog
from click import Choice
from click._compat import term_len
from click.core import ParameterSource, augment_usage_errors
from click.formatting import iter_rows, measure_table, wrap_text
from toml import TomlDecodeError, load
from web3.gas_strategies.time_based import fast_gas_price_strategy
from raiden.constants import ServerListType
from raiden.exceptions import ConfigurationError, InvalidChecksummedAddress
from raiden.network.rpc.middleware import faster_gas_price_strategy
from raiden.utils.formatting import address_checksum_and_decode
from raiden_contracts.constants import CHAINNAME_TO_ID
log: structlog.Logger = structlog.get_logger(__name__)
CONTEXT_KEY_DEFAULT_OPTIONS: str = 'raiden.options_using_default'
LOG_CONFIG_OPTION_NAME: str = 'log_config'

class HelpFormatter(click.HelpFormatter):
    ...

class Context(click.Context):
    formatter_class: type = HelpFormatter

class GroupableOption(click.Option):
    ...

class GroupableOptionCommand(click.Command):
    context_class: type = Context

class GroupableOptionCommandGroup(click.Group):
    ...

def command(name: Optional[str] = None, cls: type = GroupableOptionCommand, **attrs) -> click.Command:
    ...

def group(name: Optional[str] = None, **attrs) -> click.Group:
    ...

def option(*args: Any, **kwargs) -> click.Option:
    ...

def option_group(name: str, *options: Any) -> Any:
    ...

class AddressType(click.ParamType):
    name: str = 'address'

    def convert(self, value: str, param: click.Option, ctx: click.Context) -> str:
        ...

class LogLevelConfigType(click.ParamType):
    name: str = 'log-config'
    _validate_re: re.Pattern = re.compile('^(?:(?P<logger_name>[a-zA-Z0-9._]+)?:(?P<logger_level>debug|info|warn(?:ing)?|error|critical|fatal),?)*$', re.IGNORECASE)

    def convert(self, value: str, param: click.Option, ctx: click.Context) -> Dict[str, str]:
        ...

class ChainChoiceType(click.Choice):
    ...

class EnumChoiceType(Choice):
    def __init__(self, enum_type: type, case_sensitive: bool = True):
        ...

class GasPriceChoiceType(click.Choice):
    ...

class MatrixServerType(click.Choice):
    ...

class HypenTemplate(Template):
    idpattern: str = '(?-i:[_a-zA-Z-][_a-zA-Z0-9-]*)'

class ExpandablePath(click.Path):
    ...

class ExpandableFile(click.File):
    ...

class PathRelativePath(click.Path):
    ...

class SkipParsing(Exception):
    pass

class Parser(metaclass=ABCMeta):
    def __init__(self, param_name: str, priority: Optional[int] = None):
        ...

class ConfigParser(Parser):
    default_priority: int = 99

    def parse(self, ctx: click.Context, value: str, source: ParameterSource) -> Dict[str, Any]:
        ...
