import io
import json
import platform
import re
import sys
import tokenize
import traceback
from collections.abc import Collection, Generator, Iterator, MutableMapping, Sequence, Sized
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from json.decoder import JSONDecodeError
from pathlib import Path
from re import Pattern
from typing import Any, Optional, Union, Tuple, List, Set, Dict, ContextManager
import click
from click.core import ParameterSource
from mypy_extensions import mypyc_attr
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from _black_version import version as __version__
from black.cache import Cache
from black.comments import normalize_fmt_off
from black.const import DEFAULT_EXCLUDES, DEFAULT_INCLUDES, DEFAULT_LINE_LENGTH, STDIN_PLACEHOLDER
from black.files import best_effort_relative_path, find_project_root, find_pyproject_toml, find_user_pyproject_toml, gen_python_files, get_gitignore, parse_pyproject_toml, path_is_excluded, resolves_outside_root_or_cannot_stat, wrap_stream_for_windows
from black.handle_ipynb_magics import PYTHON_CELL_MAGICS, jupyter_dependencies_are_installed, mask_cell, put_trailing_semicolon_back, remove_trailing_semicolon, unmask_cell, validate_cell
from black.linegen import LN, LineGenerator, transform_line
from black.lines import EmptyLineTracker, LinesBlock
from black.mode import FUTURE_FLAG_TO_FEATURE, VERSION_TO_FEATURES, Feature
from black.mode import Mode as Mode
from black.mode import Preview, TargetVersion, supports_feature
from black.nodes import STARS, is_number_token, is_simple_decorator_expression, syms
from black.output import color_diff, diff, dump_to_file, err, ipynb_diff, out
from black.parsing import ASTSafetyError, InvalidInput, lib2to3_parse, parse_ast, stringify_ast
from black.ranges import adjusted_lines, convert_unchanged_lines, parse_line_ranges, sanitized_lines
from black.report import Changed, NothingChanged, Report
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
COMPILED: bool = Path(__file__).suffix in ('.pyd', '.so')
FileContent = str
Encoding = str
NewLine = str

class WriteBack(Enum):
    NO = 0
    YES = 1
    DIFF = 2
    CHECK = 3
    COLOR_DIFF = 4

    @classmethod
    def from_configuration(cls, *, check: bool, diff: bool, color: bool = False) -> 'WriteBack':
        if check and (not diff):
            return cls.CHECK
        if diff and color:
            return cls.COLOR_DIFF
        return cls.DIFF if diff else cls.YES
FileMode = Mode

def read_pyproject_toml(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[str]:
    """Inject Black configuration from "pyproject.toml" into defaults in `ctx`.

    Returns the path to a successfully found and read configuration file, None
    otherwise.
    """
    if not value:
        value = find_pyproject_toml(ctx.params.get('src', ()), ctx.params.get('stdin_filename', None))
        if value is None:
            return None
    try:
        config = parse_pyproject_toml(value)
    except (OSError, ValueError) as e:
        raise click.FileError(filename=value, hint=f'Error reading configuration file: {e}') from None
    if not config:
        return None
    else:
        spellcheck_pyproject_toml_keys(ctx, list(config), value)
        config = {k: str(v) if not isinstance(v, (list, dict)) else v for (k, v) in config.items()}
    target_version = config.get('target_version')
    if target_version is not None and (not isinstance(target_version, list)):
        raise click.BadOptionUsage('target-version', 'Config key target-version must be a list')
    exclude = config.get('exclude')
    if exclude is not None and (not isinstance(exclude, str)):
        raise click.BadOptionUsage('exclude', 'Config key exclude must be a string')
    extend_exclude = config.get('extend_exclude')
    if extend_exclude is not None and (not isinstance(extend_exclude, str)):
        raise click.BadOptionUsage('extend-exclude', 'Config key extend-exclude must be a string')
    line_ranges = config.get('line_ranges')
    if line_ranges is not None:
        raise click.BadOptionUsage('line-ranges', 'Cannot use line-ranges in the pyproject.toml file.')
    default_map: dict[str, Any] = {}
    if ctx.default_map:
        default_map.update(ctx.default_map)
    default_map.update(config)
    ctx.default_map = default_map
    return value

def spellcheck_pyproject_toml_keys(ctx: click.Context, config_keys: List[str], config_file_path: str) -> None:
    invalid_keys: list[str] = []
    available_config_options = {param.name for param in ctx.command.params}
    for key in config_keys:
        if key not in available_config_options:
            invalid_keys.append(key)
    if invalid_keys:
        keys_str = ', '.join(map(repr, invalid_keys))
        out(f'Invalid config keys detected: {keys_str} (in {config_file_path})', fg='red')

def target_version_option_callback(c: click.Context, p: click.Parameter, v: Tuple[str, ...]) -> List[TargetVersion]:
    """Compute the target versions from a --target-version flag.

    This is its own function because mypy couldn't infer the type correctly
    when it was a lambda, causing mypyc trouble.
    """
    return [TargetVersion[val.upper()] for val in v]

def enable_unstable_feature_callback(c: click.Context, p: click.Parameter, v: Tuple[str, ...]) -> List[Preview]:
    """Compute the features from an --enable-unstable-feature flag."""
    return [Preview[val] for val in v]

def re_compile_maybe_verbose(regex: str) -> Pattern[str]:
    """Compile a regular expression string in `regex`.

    If it contains newlines, use verbose mode.
    """
    if '\n' in regex:
        regex = '(?x)' + regex
    compiled: Pattern[str] = re.compile(regex)
    return compiled

def validate_regex(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[Pattern[str]]:
    try:
        return re_compile_maybe_verbose(value) if value is not None else None
    except re.error as e:
        raise click.BadParameter(f'Not a valid regular expression: {e}') from None

@click.command(context_settings={'help_option_names': ['-h', '--help']}, help='The uncompromising code formatter.')
@click.option('-c', '--code', type=str, help='Format the code passed in as a string.')
@click.option('-l', '--line-length', type=int, default=DEFAULT_LINE_LENGTH, help='How many characters per line to allow.', show_default=True)
@click.option('-t', '--target-version', type=click.Choice([v.name.lower() for v in TargetVersion]), callback=target_version_option_callback, multiple=True, help="Python versions that should be supported by Black's output. You should include all versions that your code supports. By default, Black will infer target versions from the project metadata in pyproject.toml. If this does not yield conclusive results, Black will use per-file auto-detection.")
@click.option('--pyi', is_flag=True, help='Format all input files like typing stubs regardless of file extension. This is useful when piping source on standard input.')
@click.option('--ipynb', is_flag=True, help='Format all input files like Jupyter Notebooks regardless of file extension. This is useful when piping source on standard input.')
@click.option('--python-cell-magics', multiple=True, help=f"When processing Jupyter Notebooks, add the given magic to the list of known python-magics ({', '.join(sorted(PYTHON_CELL_MAGICS))}). Useful for formatting cells with custom python magics.", default=[])
@click.option('-x', '--skip-source-first-line', is_flag=True, help='Skip the first line of the source code.')
@click.option('-S', '--skip-string-normalization', is_flag=True, help="Don't normalize string quotes or prefixes.")
@click.option('-C', '--skip-magic-trailing-comma', is_flag=True, help="Don't use trailing commas as a reason to split lines.")
@click.option('--preview', is_flag=True, help="Enable potentially disruptive style changes that may be added to Black's main functionality in the next major release.")
@click.option('--unstable', is_flag=True, help="Enable potentially disruptive style changes that have known bugs or are not currently expected to make it into the stable style Black's next major release. Implies --preview.")
@click.option('--enable-unstable-feature', type=click.Choice([v.name for v in Preview]), callback=enable_unstable_feature_callback, multiple=True, help='Enable specific features included in the `--unstable` style. Requires `--preview`. No compatibility guarantees are provided on the behavior or existence of any unstable features.')
@click.option('--check', is_flag=True, help="Don't write the files back, just return the status. Return code 0 means nothing would change. Return code 1 means some files would be reformatted. Return code 123 means there was an internal error.")
@click.option('--diff', is_flag=True, help="Don't write the files back, just output a diff to indicate what changes Black would've made. They are printed to stdout so capturing them is simple.")
@click.option('--color/--no-color', is_flag=True, help='Show (or do not show) colored diff. Only applies when --diff is given.')
@click.option('--line-ranges', multiple=True, metavar='START-END', help='When specified, Black will try its best to only format these lines. This option can be specified multiple times, and a union of the lines will be formatted. Each range must be specified as two integers connected by a `-`: `<START>-<END>`. The `<START>` and `<END>` integer indices are 1-based and inclusive on both ends.', default=())
@click.option('--fast/--safe', is_flag=True, help='By default, Black performs an AST safety check after formatting your code. The --fast flag turns off this check and the --safe flag explicitly enables it. [default: --safe]')
@click.option('--required-version', type=str, help='Require a specific version of Black to be running. This is useful for ensuring that all contributors to your project are using the same version, because different versions of Black may format code a little differently. This option can be set in a configuration file for consistent results across environments.')
@click.option('--exclude', type=str, callback=validate_regex, help=f'A regular expression that matches files and directories that should be excluded on recursive searches. An empty value means no paths are excluded. Use forward slashes for directories on all platforms (Windows, too). By default, Black also ignores all paths listed in .gitignore. Changing this value will override all default exclusions. [default: {DEFAULT_EXCLUDES}]', show_default=False)
@click.option('--extend-exclude', type=str, callback=validate_regex, help='Like --exclude, but adds additional files and directories on top of the default values instead of overriding them.')
@click.option('--force-exclude', type=str, callback=validate_regex, help='Like --exclude, but files and directories matching this regex will be excluded even when they are passed explicitly as arguments. This is useful when invoking Black programmatically on changed files, such as in a pre-commit hook or editor plugin.')
@click.option('--stdin-filename', type=str, is_eager=True, help='The name of the file when passing it through stdin. Useful to make sure Black will respect the --force-exclude option on some editors that rely on using stdin.')
@click.option('--include', type=str, default=DEFAULT_INCLUDES, callback=validate_regex, help='A regular expression that matches files and directories that should be included on recursive searches. An empty value means all files are included regardless of the name. Use forward slashes for directories on all platforms (Windows, too). Overrides all exclusions, including from .gitignore and command line options.', show_default=True)
@click.option('-W', '--workers', type=click.IntRange(min=1), default=None, help='When Black formats multiple files, it may use a process pool to speed up formatting. This option controls the number of parallel workers. This can also be specified via the BLACK_NUM_WORKERS environment variable. Defaults to the number of CPUs in the system.')
@click.option('-q', '--quiet', is_flag=True, help='Stop emitting all non-critical output. Error messages will still be emitted (which can silenced by 2>/dev/null).')
@click.option('-v', '--verbose', is_flag=True, help='Emit messages about files that were not changed or were ignored due to exclusion patterns. If Black is using a configuration file, a message detailing which one it is using will be emitted.')
@click.version_option(version=__version__, message=f"%(prog)s, %(version)s (compiled: {('yes' if COMPILED else 'no')})\nPython ({platform.python_implementation()}) {platform.python_version()}")
@click.argument('src', nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, allow_dash=True), is_eager=True, metavar='SRC ...')
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, allow_dash=False, path_type=str), is_eager=True, callback=read_pyproject_toml, help='Read configuration options from a configuration file.')
@click.pass_context
def main(ctx: click.Context, code: Optional[str], line_length: int, target_version: List[TargetVersion], check: bool, diff: bool, line_ranges: Tuple[str, ...], color: bool, fast: bool, pyi: bool, ipynb: bool, python_cell_magics: Tuple[str, ...], skip_source_first_line: bool, skip_string_normalization: bool, skip_magic_trailing_comma: bool, preview: bool, unstable: bool, enable_unstable_feature: List[Preview], quiet: bool, verbose: bool, required_version: Optional[str], include: Pattern[str], exclude: Optional[Pattern[str]], extend_exclude: Optional[Pattern[str]], force_exclude: Optional[Pattern[str]], stdin_filename: Optional[str], workers: Optional[int], src: Tuple[str, ...], config: Optional[str]) -> None:
    """The uncompromising code formatter."""
    ctx.ensure_object(dict)
    assert sys.version_info >= (3, 9), 'Black requires Python 3.9+'
    if sys.version_info[:3] == (3, 12, 5):
        out("Python 3.12.5 has a memory safety issue that can cause Black's AST safety checks to fail. Please upgrade to Python 3.12.6 or downgrade to Python 3.12.4")
        ctx.exit(1)
    if src and code is not None:
        out(main.get_usage(ctx) + "\n\n'SRC' and 'code' cannot be passed simultaneously.")
        ctx.exit(1)
    if not src and code is None:
        out(main.get_usage(ctx) + "\n\nOne of 'SRC' or 'code' is required.")
        ctx.exit(1)
    if enable_unstable_feature and (not (preview or unstable)):
        out(main.get_usage(ctx) + "\n\n'--enable-unstable-feature' requires '--preview'.")
        ctx.exit(1)
    (root, method) = find_project_root(src, stdin_filename) if code is None else (None, None)
    ctx.obj['root'] = root
    if verbose:
        if root:
            out(f'Identified `{root}` as project root containing a {method}.', fg='blue')
        if config:
            config_source = ctx.get_parameter_source('config')
            user_level_config = str(find_user_pyproject_toml())
            if config == user_level_config:
                out(f"Using configuration from user-level config at '{user_level_config}'.", fg='blue')
            elif config_source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP):
                out('Using configuration from project root.', fg='blue')
            else:
                out(f"Using configuration in '{config}'.", fg='blue')
            if ctx.default_map:
                for (param, value) in ctx.default_map.items():
                    out(f'{param}: {value}')
    error_msg = 'Oh no! 💥 💔 💥'
    if required_version and required_version != __version__ and (required_version != __version__.split('.')[0]):
        err(f'{error_msg} The required version `{required_version}` does not match the running version `{__version__}`!')
        ctx.exit(1)
    if ipynb and pyi:
        err('Cannot pass both `pyi` and `ipynb` flags!')
        ctx.exit(1)
    write_back = WriteBack.from_configuration(check=check, diff=diff, color=color)
    if target_version:
        versions = set(target_version)
    else:
        versions = set()
    mode = Mode(target_versions=versions, line_length=line_length, is_pyi=pyi, is_ipynb=ipynb, skip_source_first_line=skip_source_first_line, string_normalization=not skip_string_normalization, magic_trailing_comma=not skip_m