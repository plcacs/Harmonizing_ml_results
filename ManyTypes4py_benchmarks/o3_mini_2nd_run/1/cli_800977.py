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
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Set, Tuple, Iterable, Mapping, Type

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

log = structlog.get_logger(__name__)
CONTEXT_KEY_DEFAULT_OPTIONS: str = 'raiden.options_using_default'
LOG_CONFIG_OPTION_NAME: str = 'log_config'


class HelpFormatter(click.HelpFormatter):
    """
    Subclass that allows multiple (option) sections to be formatted with pre-determined
    widths.
    """

    def write_dl(
        self,
        rows: List[Tuple[str, str]],
        col_max: int = 30,
        col_spacing: int = 2,
        widths: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Writes a definition list into the buffer. This is how options
        and commands are usually formatted.

        :param rows: a list of two item tuples for the terms and values.
        :param col_max: the maximum width of the first column.
        :param col_spacing: the number of spaces between the first and second column.
        :param widths: optional pre-calculated line widths
        """
        rows = list(rows)
        if widths is None:
            widths = measure_table(rows)
        if len(widths) != 2:
            raise TypeError('Expected two columns for definition list')
        first_col: int = min(widths[0], col_max) + col_spacing
        for first, second in iter_rows(rows, len(widths)):
            self.write('%*s%s' % (self.current_indent, '', first))
            if not second:
                self.write('\n')
                continue
            if term_len(first) <= first_col - col_spacing:
                self.write(' ' * (first_col - term_len(first)))
            else:
                self.write('\n')
                self.write(' ' * (first_col + self.current_indent))
            text_width: int = max(self.width - first_col - 2, 10)
            lines: Iterable[str] = iter(wrap_text(second, text_width).splitlines())
            if lines:
                self.write(next(lines) + '\n')
                for line in lines:
                    self.write('%*s%s\n' % (first_col + self.current_indent, '', line))
            else:
                self.write('\n')


class Context(click.Context):
    formatter_class = HelpFormatter


class GroupableOption(click.Option):
    def __init__(
        self,
        param_decls: Optional[List[str]] = None,
        show_default: bool = False,
        prompt: bool = False,
        confirmation_prompt: bool = False,
        prompt_required: bool = True,
        hide_input: bool = False,
        is_flag: Optional[bool] = None,
        flag_value: Any = None,
        multiple: bool = False,
        count: bool = False,
        allow_from_autoenv: bool = True,
        type: Any = None,
        help: Optional[str] = None,
        hidden: bool = False,
        show_choices: bool = True,
        show_envvar: bool = False,
        option_group: Optional[str] = None,
        option_parser_cls: Optional[Callable[..., Any]] = None,
        option_parser_priority: Optional[int] = None,
        **attrs: Any
    ) -> None:
        super().__init__(
            param_decls=param_decls,
            show_default=show_default,
            prompt=prompt,
            confirmation_prompt=confirmation_prompt,
            prompt_required=prompt_required,
            hide_input=hide_input,
            is_flag=is_flag,
            flag_value=flag_value,
            multiple=multiple,
            count=count,
            allow_from_autoenv=allow_from_autoenv,
            type=type,
            help=help,
            hidden=hidden,
            show_choices=show_choices,
            show_envvar=show_envvar,
            **attrs
        )
        self.option_group: Optional[str] = option_group
        self.option_parser: Optional[Any] = None
        if option_parser_cls is not None:
            self.option_parser = option_parser_cls(self.name, priority=option_parser_priority)


class GroupableOptionCommand(click.Command):
    context_class = Context

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        def keyfunc(o: Any) -> str:
            value: Optional[str] = getattr(o, 'option_group', None)
            return value if value is not None else ''

        grouped_options = groupby(sorted(self.get_params(ctx), key=keyfunc), key=keyfunc)
        options: Dict[Optional[str], List[Tuple[str, str]]] = {}
        for option_group, params in grouped_options:
            for param in params:
                rv: Optional[Tuple[str, str]] = param.get_help_record(ctx)
                if rv is not None:
                    options.setdefault(option_group, []).append(rv)
        if options:
            widths_pairs: List[Tuple[int, int]] = [measure_table(group_options) for group_options in options.values()]
            widths_a: Tuple[int, ...] = tuple(pair[0] for pair in widths_pairs)
            widths_b: Tuple[int, ...] = tuple(pair[1] for pair in widths_pairs)
            widths: Tuple[int, int] = (max(widths_a), max(widths_b))
            for option_group, group_options in options.items():
                with formatter.section(option_group if option_group else 'Options'):
                    formatter.write_dl(group_options, widths=widths)


class GroupableOptionCommandGroup(click.Group):
    context_class = Context

    def __init__(self, use_option_parsers: bool = True, **attrs: Any) -> None:
        super().__init__(**attrs)
        self._extra_parsers: List[Parser] = []
        self.internal_to_external_names: Dict[str, Any] = {}
        self.opt_name_to_param: Dict[str, click.Parameter] = {}
        self.use_option_parsers: bool = use_option_parsers
        for param in self.params:
            self.opt_name_to_param[param.name] = param
            parser = getattr(param, 'option_parser', None)
            if parser is not None:
                self._extra_parsers.append(parser)
                for param2 in self.params:
                    parser.register_param(param2)
        self._extra_parsers.sort()

    @staticmethod
    def _process_parse_result(
        ctx: click.Context,
        param_name: str,
        source: ParameterSource,
        value: Any,
        parser_value: Any
    ) -> None:
        can_be_overwritten: bool = value is None or source in (ParameterSource.DEFAULT_MAP, ParameterSource.DEFAULT)
        if param_name == LOG_CONFIG_OPTION_NAME and source is ParameterSource.COMMANDLINE:
            parser_value = {**parser_value, **value}
            can_be_overwritten = True
        if can_be_overwritten:
            ctx.params[param_name] = parser_value
            ctx.set_parameter_source(param_name, ParameterSource.DEFAULT_MAP)

    def invoke(self, ctx: click.Context) -> Any:
        if self.use_option_parsers is True:
            for parser in self._extra_parsers:
                parser_value: Any = ctx.params[parser.name]
                if parser_value is not None:
                    parser_source: ParameterSource = ctx.get_parameter_source(parser.name)
                    try:
                        parse_result: Dict[str, Any] = parser.parse(ctx, parser_value, parser_source)
                        for param_name, value in ctx.params.items():
                            source: ParameterSource = ctx.get_parameter_source(param_name)
                            parser_val: Any = parse_result.get(param_name)
                            if parser_val is not None:
                                param: click.Parameter = self.opt_name_to_param[param_name]
                                with augment_usage_errors(ctx, param=param):
                                    try:
                                        parsed_value: Any = param.process_value(ctx, parser_val)
                                        if param.callback is not None:
                                            value = param.callback(ctx, param, parsed_value)
                                    except Exception:
                                        if not ctx.resilient_parsing:
                                            raise
                                        parsed_value = None
                                if param.expose_value:
                                    self._process_parse_result(ctx, param_name, source, value, parsed_value)
                    except SkipParsing:
                        ctx.params[parser.name] = None
                        ctx.set_parameter_source(parser.name, ParameterSource.DEFAULT_MAP)
        return super().invoke(ctx)

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        GroupableOptionCommand.format_options(self, ctx, formatter)
        self.format_commands(ctx, formatter)

    def command(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        return super().command(*args, **{'cls': GroupableOptionCommand, **kwargs})

    def group(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        return super().group(*args, **{'cls': self.__class__, **kwargs})


def command(name: Optional[str] = None, cls: Any = GroupableOptionCommand, **attrs: Any) -> Callable[..., Any]:
    return click.command(name, cls, **attrs)


def group(name: Optional[str] = None, **attrs: Any) -> Callable[..., Any]:
    return click.group(name, cls=GroupableOptionCommandGroup, **attrs)


def option(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    return click.option(*args, cls=GroupableOption, **kwargs)


def option_group(name: str, *options: Callable[..., Any]) -> Callable[..., Any]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        for option_ in reversed(options):
            if option_.__closure__ is not None:
                for closure_cell in option_.__closure__:
                    if isinstance(closure_cell.cell_contents, dict):
                        closure_cell.cell_contents['option_group'] = name
                        break
            option_(f)
        return f
    return decorator


class AddressType(click.ParamType):
    name = 'address'

    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        try:
            return address_checksum_and_decode(value)
        except InvalidChecksummedAddress as e:
            self.fail(str(e), param, ctx)


class LogLevelConfigType(click.ParamType):
    name = 'log-config'
    _validate_re = re.compile(
        '^(?:(?P<logger_name>[a-zA-Z0-9._]+)?:(?P<logger_level>debug|info|warn(?:ing)?|error|critical|fatal),?)*$', 
        re.IGNORECASE
    )

    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        if isinstance(value, dict):
            for key, val in value.items():
                recombined_str: str = f'{key}:{val}'
                if not self._validate_re.match(recombined_str):
                    self.fail(f'`{recombined_str}` is not a valid logging format.', param, ctx)
            return value
        if not self._validate_re.match(value):
            self.fail(f'`{value}` is not a valid logging format.', param, ctx)
        level_config: Dict[str, str] = {}
        if value.strip(' ') == '':
            return None
        for logger_config in value.split(','):
            logger_name, logger_level = logger_config.split(':')
            level_config[logger_name] = logger_level.upper()
        return level_config


class ChainChoiceType(click.Choice):

    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        if isinstance(value, int):
            return value
        elif isinstance(value, str) and value.isnumeric():
            try:
                return int(value)
            except ValueError:
                self.fail(f'invalid numeric network id: {value}', param, ctx)
        else:
            network_name: str = super().convert(value, param, ctx)
            return CHAINNAME_TO_ID[network_name]


class EnumChoiceType(Choice):
    def __init__(self, enum_type: Type[Any], case_sensitive: bool = True) -> None:
        self._enum_type: Type[Any] = enum_type
        super().__init__([choice.value for choice in enum_type], case_sensitive=case_sensitive)

    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        try:
            return self._enum_type(value)
        except ValueError:
            self.fail(f"'{value}' is not a valid {self._enum_type.__name__.lower()}", param, ctx)


class GasPriceChoiceType(click.Choice):
    """Returns a GasPriceStrategy for the choice"""

    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        if isinstance(value, str) and value.isnumeric():
            try:
                gas_price: int = int(value)

                def fixed_gas_price_strategy(_web3: Any, _transaction_params: Any) -> int:
                    return gas_price
                return fixed_gas_price_strategy
            except ValueError:
                self.fail(f'invalid numeric gas price: {value}', param, ctx)
        else:
            if callable(value):
                return value
            gas_price_string: str = super().convert(value, param, ctx)
            if gas_price_string == 'fast':
                return faster_gas_price_strategy
            else:
                return fast_gas_price_strategy


class MatrixServerType(click.Choice):
    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        if isinstance(value, str) and value.startswith('http'):
            return value
        return super().convert(value, param, ctx)


class HypenTemplate(Template):
    idpattern = '(?-i:[_a-zA-Z-][_a-zA-Z0-9-]*)'


class ExpandablePath(click.Path):
    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        value = os.path.expanduser(value)
        return super().convert(value, param, ctx)


class ExpandableFile(click.File):
    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        value = os.path.expanduser(value)
        return super().convert(value, param, ctx)


class PathRelativePath(click.Path):
    """
    `click.Path` subclass that can default to a value depending on
    another option of type `click.Path`.

    Uses :ref:`string.Template` to expand the parameters default value.

    Example::

        @click.option('--some-dir', type=click.Path())
        @click.option('--some-file', type=PathRelativePath(), default='${some-dir}/file.txt')
    """
    def convert(self, value: Any, param: Optional[click.Parameter] = None, ctx: Optional[click.Context] = None) -> Any:
        if value == param.default:
            try:
                value = self.expand_default(value, ctx.params)
            except KeyError as ex:
                raise RuntimeError(
                    "Substitution parameter not found in context. Make sure it's defined with `is_eager=True`."
                ) from ex
        return super().convert(value, param, ctx)

    @staticmethod
    def expand_default(default: str, params: Mapping[str, Any]) -> str:
        return HypenTemplate(default).substitute(params)


class SkipParsing(Exception):
    pass


class Parser(metaclass=ABCMeta):
    def __init__(self, param_name: str, priority: Optional[int] = None) -> None:
        self.name: str = param_name
        self.priority: int = priority if priority is not None else self.default_priority
        self.name_map: Dict[str, str] = {}
        self._internal_names: Set[str] = set()

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Parser):
            return NotImplemented
        return self.priority > other.priority

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Parser):
            return NotImplemented
        return self.priority == other.priority

    @property
    @abstractmethod
    def default_priority(self) -> int:
        pass

    def get_internal_name(self, name: str) -> Optional[str]:
        """Convert parser-specific parameter name to click-internal name

        Get the click-internal name (name of call argument of wrapped function)
        from the names used in the parser.
        This method has to be "idempotent", so that if the click-internal name is provided as
        argument, it will simply pass through this name
        """
        if name in self._internal_names:
            return name
        return self.name_map.get(name)

    def register_param(self, param: click.Parameter) -> None:
        """Registers a click.Parameter so that the mapping from "external" name to "internal"
        name can be memorized

        Overwrite this method only when the parser uses
        different parameter names than the parameter names as defined by the `option()`
        """
        for opt_name in param.opts:
            if opt_name.startswith('--'):
                opt_name = opt_name[2:]
            self.name_map[opt_name] = param.name
            self._internal_names.add(param.name)

    @abstractmethod
    def parse(
        self, ctx: click.Context, value: Any, source: ParameterSource
    ) -> Dict[str, Any]:
        """
        Parses more values to provide to clicks 'ctx.params' based on the value of a parameter.

        value - the value of the parameter for `self.name` that was parsed by click
        source - the ParameterSource where the value of `self.name` was set by click.
        ctx - The current click `Context`

        This method should return a dictionary that maps the "internal" parameter name
        to the parameter values as read in by the parser.

        In order to get the internal name from the parameter name as defined by the parser,
        the `self.get_internal_name()` method should be called.
        """
        pass


class ConfigParser(Parser):
    default_priority: int = 99

    def parse(self, ctx: click.Context, value: Any, source: ParameterSource) -> Dict[str, Any]:
        config_path: Path = Path(value)
        try:
            with config_path.open() as config_file:
                parsed_config_dict: Dict[str, Any] = load(config_file)
                config_dict: Dict[str, Any] = {}
                for ext_name, config_value in parsed_config_dict.items():
                    int_name: Optional[str] = self.get_internal_name(ext_name)
                    if int_name is None:
                        raise ConfigurationError(f"Config file option '{ext_name}' not known.")
                    config_dict[int_name] = config_value
                return config_dict
        except OSError as ex:
            default_config_missing: bool = ex.errno == errno.ENOENT and source == ParameterSource.DEFAULT
            if default_config_missing:
                msg: str = f'Default configuration file {value} not found. Skipping parsing.'
                raise SkipParsing(msg)
            else:
                raise ConfigurationError(f'Error opening config file: {ex}')
        except TomlDecodeError as ex:
            raise ConfigurationError(f'Error loading config file: {ex}') from ex


def get_matrix_servers(
    url: str, server_list_type: ServerListType = ServerListType.ACTIVE_SERVERS
) -> List[str]:
    """
    Fetch a list of matrix servers from a URL

    The URL is expected to point to a JSON document of the following format::

        {
            "active_servers": [
                "url1",
                "url2",
                ...
            ],
            "all_servers": [
                "url1",
                "url2",
                ...
            ]
        }

    Which of the two lists is returned is controlled by the ``server_list_type`` argument.
    """
    try:
        response: requests.Response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as ex:
        raise RuntimeError(f'Could not fetch matrix servers list: {url!r} => {ex!r}') from ex
    try:
        known_servers: Dict[str, Any] = json.loads(response.text)
        msg: str = f'Unexpected format of known server list at {url}'
        assert {type_.value for type_ in ServerListType} == set(known_servers.keys()), msg
        active_servers: List[str] = known_servers[server_list_type.value]
    except (JSONDecodeError, AssertionError) as ex:
        raise RuntimeError(f'Could not process list of known matrix servers: {url!r} => {ex!r}') from ex
    return [f'https://{server}' if not server.startswith('http') else server for server in active_servers]


ADDRESS_TYPE: AddressType = AddressType()
LOG_LEVEL_CONFIG_TYPE: LogLevelConfigType = LogLevelConfigType()