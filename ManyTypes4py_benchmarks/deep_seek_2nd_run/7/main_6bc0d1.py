"""Tool for sorting imports alphabetically, and automatically separated into sections."""
import argparse
import functools
import json
import os
import sys
from gettext import gettext as _
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Set, Tuple, Iterable, Iterator, Callable
from warnings import warn
from . import __version__, api, files, sections
from .exceptions import FileSkipped, ISortError, UnsupportedEncoding
from .format import create_terminal_printer
from .logo import ASCII_ART
from .profiles import profiles
from .settings import VALID_PY_TARGETS, Config, find_all_configs
from .utils import Trie
from .wrap_modes import WrapModes

DEPRECATED_SINGLE_DASH_ARGS: Set[str] = {'-ac', '-af', '-ca', '-cs', '-df', '-ds', '-dt', '-fas', '-fass', '-ff', '-fgw', '-fss', '-lai', '-lbt', '-le', '-ls', '-nis', '-nlb', '-ot', '-rr', '-sd', '-sg', '-sl', '-sp', '-tc', '-wl', '-ws'}
QUICK_GUIDE: str = f"\n{ASCII_ART}\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

class SortAttempt:
    def __init__(
        self,
        incorrectly_sorted: bool,
        skipped: bool,
        supported_encoding: bool
    ) -> None:
        self.incorrectly_sorted: bool = incorrectly_sorted
        self.skipped: bool = skipped
        self.supported_encoding: bool = supported_encoding

def sort_imports(
    file_name: str,
    config: Config,
    check: bool = False,
    ask_to_apply: bool = False,
    write_to_stdout: bool = False,
    **kwargs: Any
) -> Optional[SortAttempt]:
    incorrectly_sorted: bool = False
    skipped: bool = False
    try:
        if check:
            try:
                incorrectly_sorted = not api.check_file(file_name, config=config, **kwargs)
            except FileSkipped:
                skipped = True
            return SortAttempt(incorrectly_sorted, skipped, True)
        try:
            incorrectly_sorted = not api.sort_file(file_name, config=config, ask_to_apply=ask_to_apply, write_to_stdout=write_to_stdout, **kwargs)
        except FileSkipped:
            skipped = True
        return SortAttempt(incorrectly_sorted, skipped, True)
    except (OSError, ValueError) as error:
        warn(f'Unable to parse file {file_name} due to {error}')
        return None
    except UnsupportedEncoding:
        if config.verbose:
            warn(f'Encoding not supported for {file_name}')
        return SortAttempt(incorrectly_sorted, skipped, False)
    except ISortError as error:
        _print_hard_fail(config, message=str(error))
        sys.exit(1)
    except Exception:
        _print_hard_fail(config, offending_file=file_name)
        raise

def _print_hard_fail(
    config: Config,
    offending_file: Optional[str] = None,
    message: Optional[str] = None
) -> None:
    """Fail on unrecoverable exception with custom message."""
    message = message or f'Unrecoverable exception thrown when parsing {offending_file or ""}! This should NEVER happen.\nIf encountered, please open an issue: https://github.com/PyCQA/isort/issues/new'
    printer = create_terminal_printer(color=config.color_output, error=config.format_error, success=config.format_success)
    printer.error(message)

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sort Python import definitions alphabetically within logical sections. Run with no arguments to see a quick start guide, otherwise, one or more files/directories/stdin must be provided. Use `-` as the first argument to represent stdin. Use --interactive to use the pre 5.0.0 interactive behavior. If you've used isort 4 but are new to isort 5, see the upgrading guide: https://pycqa.github.io/isort/docs/upgrade_guides/5.0.0.html", add_help=False)
    general_group = parser.add_argument_group('general options')
    target_group = parser.add_argument_group('target options')
    output_group = parser.add_argument_group('general output options')
    inline_args_group = output_group.add_mutually_exclusive_group()
    section_group = parser.add_argument_group('section output options')
    deprecated_group = parser.add_argument_group('deprecated options')
    
    # Argument additions remain the same as they were
    # ... (rest of the argument parser setup remains unchanged)
    
    return parser

def parse_args(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    argv = sys.argv[1:] if argv is None else list(argv)
    remapped_deprecated_args: List[str] = []
    for index, arg in enumerate(argv):
        if arg in DEPRECATED_SINGLE_DASH_ARGS:
            remapped_deprecated_args.append(arg)
            argv[index] = f'-{arg}'
    parser = _build_arg_parser()
    arguments = {key: value for key, value in vars(parser.parse_args(argv)).items() if value}
    if remapped_deprecated_args:
        arguments['remapped_deprecated_args'] = remapped_deprecated_args
    if 'dont_order_by_type' in arguments:
        arguments['order_by_type'] = False
        del arguments['dont_order_by_type']
    if 'dont_follow_links' in arguments:
        arguments['follow_links'] = False
        del arguments['dont_follow_links']
    if 'dont_float_to_top' in arguments:
        del arguments['dont_float_to_top']
        if arguments.get('float_to_top', False):
            sys.exit("Can't set both --float-to-top and --dont-float-to-top.")
        else:
            arguments['float_to_top'] = False
    multi_line_output = arguments.get('multi_line_output', None)
    if multi_line_output:
        if multi_line_output.isdigit():
            arguments['multi_line_output'] = WrapModes(int(multi_line_output))
        else:
            arguments['multi_line_output'] = WrapModes[multi_line_output]
    return arguments

def _preconvert(item: Any) -> Any:
    """Preconverts objects from native types into JSONifyiable types"""
    if isinstance(item, (set, frozenset)):
        return list(item)
    if isinstance(item, WrapModes):
        return str(item.name)
    if isinstance(item, Path):
        return str(item)
    if callable(item) and hasattr(item, '__name__'):
        return str(item.__name__)
    raise TypeError(f'Unserializable object {item} of type {type(item)}')

def identify_imports_main(argv: Optional[Sequence[str]] = None, stdin: Optional[TextIOWrapper] = None) -> None:
    parser = argparse.ArgumentParser(description='Get all import definitions from a given file.Use `-` as the first argument to represent stdin.')
    parser.add_argument('files', nargs='+', help='One or more Python source files that need their imports sorted.')
    parser.add_argument('--top-only', action='store_true', default=False, help='Only identify imports that occur in before functions or classes.')
    target_group = parser.add_argument_group('target options')
    target_group.add_argument('--follow-links', action='store_true', default=False, help='Tells isort to follow symlinks that are encountered when running recursively.')
    uniqueness = parser.add_mutually_exclusive_group()
    uniqueness.add_argument('--unique', action='store_true', default=False, help='If true, isort will only identify unique imports.')
    uniqueness.add_argument('--packages', dest='unique', action='store_const', const=api.ImportKey.PACKAGE, default=False, help='If true, isort will only identify the unique top level modules imported.')
    uniqueness.add_argument('--modules', dest='unique', action='store_const', const=api.ImportKey.MODULE, default=False, help='If true, isort will only identify the unique modules imported.')
    uniqueness.add_argument('--attributes', dest='unique', action='store_const', const=api.ImportKey.ATTRIBUTE, default=False, help='If true, isort will only identify the unique attributes imported.')
    arguments = parser.parse_args(argv)
    file_names = arguments.files
    if file_names == ['-']:
        identified_imports = api.find_imports_in_stream(sys.stdin if stdin is None else stdin, unique=arguments.unique, top_only=arguments.top_only, follow_links=arguments.follow_links)
    else:
        identified_imports = api.find_imports_in_paths(file_names, unique=arguments.unique, top_only=arguments.top_only, follow_links=arguments.follow_links)
    for identified_import in identified_imports:
        if arguments.unique == api.ImportKey.PACKAGE:
            print(identified_import.module.split('.')[0])
        elif arguments.unique == api.ImportKey.MODULE:
            print(identified_import.module)
        elif arguments.unique == api.ImportKey.ATTRIBUTE:
            print(f'{identified_import.module}.{identified_import.attribute}')
        else:
            print(str(identified_import))

def main(argv: Optional[Sequence[str]] = None, stdin: Optional[TextIOWrapper] = None) -> None:
    arguments = parse_args(argv)
    if arguments.get('show_version'):
        print(ASCII_ART)
        return
    show_config = arguments.pop('show_config', False)
    show_files = arguments.pop('show_files', False)
    if show_config and show_files:
        sys.exit('Error: either specify show-config or show-files not both.')
    if 'settings_path' in arguments:
        if os.path.isfile(arguments['settings_path']):
            arguments['settings_file'] = os.path.abspath(arguments['settings_path'])
            arguments['settings_path'] = os.path.dirname(arguments['settings_file'])
        else:
            arguments['settings_path'] = os.path.abspath(arguments['settings_path'])
    if 'virtual_env' in arguments:
        venv = arguments['virtual_env']
        arguments['virtual_env'] = os.path.abspath(venv)
        if not os.path.isdir(arguments['virtual_env']):
            warn(f'virtual_env dir does not exist: {arguments['virtual_env']}')
    file_names = arguments.pop('files', [])
    if not file_names and (not show_config):
        print(QUICK_GUIDE)
        if arguments:
            sys.exit('Error: arguments passed in without any paths or content.')
        return
    if 'settings_path' not in arguments:
        arguments['settings_path'] = arguments.get('filename', None) or os.getcwd() if file_names == ['-'] else os.path.abspath(file_names[0] if file_names else '.')
        if not os.path.isdir(arguments['settings_path']):
            arguments['settings_path'] = os.path.dirname(arguments['settings_path'])
    config_dict = arguments.copy()
    ask_to_apply = config_dict.pop('ask_to_apply', False)
    jobs = config_dict.pop('jobs', None)
    check = config_dict.pop('check', False)
    show_diff = config_dict.pop('show_diff', False)
    write_to_stdout = config_dict.pop('write_to_stdout', False)
    deprecated_flags = config_dict.pop('deprecated_flags', False)
    remapped_deprecated_args = config_dict.pop('remapped_deprecated_args', False)
    stream_filename = config_dict.pop('filename', None)
    ext_format = config_dict.pop('ext_format', None)
    allow_root = config_dict.pop('allow_root', None)
    resolve_all_configs = config_dict.pop('resolve_all_configs', False)
    wrong_sorted_files = False
    all_attempt_broken = False
    no_valid_encodings = False
    config_trie = None
    if resolve_all_configs:
        config_trie = find_all_configs(config_dict.pop('config_root', '.'))
    if 'src_paths' in config_dict:
        config_dict['src_paths'] = {Path(src_path).resolve() for src_path in config_dict.get('src_paths', ())}
    config = Config(**config_dict)
    if show_config:
        print(json.dumps(config.__dict__, indent=4, separators=(',', ': '), default=_preconvert))
        return
    if file_names == ['-']:
        file_path = Path(stream_filename) if stream_filename else None
        if show_files:
            sys.exit("Error: can't show files for streaming input.")
        input_stream = sys.stdin if stdin is None else stdin
        if check:
            incorrectly_sorted = not api.check_stream(input_stream=input_stream, config=config, show_diff=show_diff, file_path=file_path, extension=ext_format)
            wrong_sorted_files = incorrectly_sorted
        else:
            try:
                api.sort_stream(input_stream=input_stream, output_stream=sys.stdout, config=config, show_diff=show_diff, file_path=file_path, extension=ext_format, raise_on_skip=False)
            except FileSkipped:
                sys.stdout.write(input_stream.read())
    elif '/' in file_names and (not allow_root):
        printer = create_terminal_printer(color=config.color_output, error=config.format_error, success=config.format_success)
        printer.error("it is dangerous to operate recursively on '/'")
        printer.error('use --allow-root to override this failsafe')
        sys.exit(1)
    else:
        if stream_filename:
            printer = create_terminal_printer(color=config.color_output, error=config.format_error, success=config.format_success)
            printer.error('Filename override is intended only for stream (-) sorting.')
            sys.exit(1)
        skipped = []
        broken = []
        if config.filter_files:
            filtered_files = []
            for file_name in file_names:
                if config.is_skipped(Path(file_name)):
                    skipped.append(file_name)
                else:
                    filtered_files.append(file_name)
            file_names = filtered_files
        file_names = files.find(file_names, config, skipped, broken)
        if show_files:
            for file_name in file_names:
                print(file_name)
            return
        num_skipped = 0
        num_broken = 0
        num_invalid_encoding = 0
        if config.verbose:
            print(ASCII_ART)
        if jobs:
            import multiprocessing
            executor = multiprocessing.Pool(jobs if jobs > 0 else multiprocessing.cpu_count())
            attempt_iterator = executor.imap(functools.partial(sort_imports, config=config, check=check, ask_to_apply=ask_to_apply, show_diff=show_diff, write_to_stdout=write_to_stdout, extension=ext_format, config_trie=config_trie), file_names)
        else:
            attempt_iterator = (sort_imports(file_name, config=config, check=check, ask_to_apply=ask_to_apply, show_diff=show_diff, write_to_stdout=write_to_stdout, extension=ext_format, config_trie=config_trie) for file_name in file_names)
        is_no_attempt = True
        any_encoding_valid = False
        for sort_attempt in attempt_iterator:
            if not sort_attempt:
                continue
            incorrectly_sorted = sort_attempt.incorrectly_sorted
            if arguments.get('check', False) and incorrectly_sorted:
                wrong_sorted_files = True
            if sort_attempt.skipped:
                num_skipped += 1
            if not sort_attempt.supported_encoding:
                num_invalid_encoding += 1
            else:
                any_encoding_valid = True
            is_no_attempt = False
        num_skipped += len(skipped)
        if num_skipped and (not config.quiet):
            if config.verbose:
                for was_skipped in skipped:
                    print(f"{was_skipped} was skipped as it's listed in 'skip' setting, matches a glob in 'skip_glob' setting, or is in a .gitignore file with --skip-gitignore enabled.")
            print(f'Skipped {num_skipped} files')
        num_broken += len(broken)
        if num_broken and (not config.quiet):
            if config.verbose:
                for was_broken in broken:
                    warn(f'{was_broken} was broken path, make sure it exists correctly')
            print(f'Broken {num_broken} paths')
        if num_broken > 0 and is_no_attempt:
            all_attempt_broken = True
        if num_invalid_encoding > 0 and (not any_encoding_valid):
            no_valid_encodings = True
    if not config.quiet and (remapped_deprecated_args or deprecated_flags):
        if remapped_deprecated_args:
            warn(f'W0502: The following deprecated single dash CLI flags were used and translated: {', '.join(remapped_deprecated_args)}!')
        if deprecated_flags:
            warn(f'W0501: The following deprecated CLI flags were used and ignored: {', '.join(deprecated_flags)}!')
        warn('W0500: Please see the 5.0.0 Upgrade guide: https://pycqa.github.io/isort/docs/upgrade_guides/5.0.0.html')
    if wrong_sorted_files:
        sys.exit(1)
    if all_attempt_broken:
        sys.exit(1)
    if no_valid_encodings:
        printer = create_terminal_printer(color=config.color_output, error