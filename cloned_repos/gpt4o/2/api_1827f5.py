__all__ = (
    'ImportKey', 'check_code_string', 'check_file', 'check_stream', 'find_imports_in_code',
    'find_imports_in_file', 'find_imports_in_paths', 'find_imports_in_stream', 'place_module',
    'place_module_with_reason', 'sort_code_string', 'sort_file', 'sort_stream'
)

import contextlib
import shutil
import sys
from enum import Enum
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Optional, Set, TextIO, Union, cast
from warnings import warn
from isort import core
from . import files, identify, io
from .exceptions import ExistingSyntaxErrors, FileSkipComment, FileSkipSetting, IntroducedSyntaxErrors
from .format import ask_whether_to_apply_changes_to_file, create_terminal_printer, show_unified_diff
from .io import Empty, File
from .place import module as place_module
from .place import module_with_reason as place_module_with_reason
from .settings import CYTHON_EXTENSIONS, DEFAULT_CONFIG, Config

class ImportKey(Enum):
    PACKAGE = 1
    MODULE = 2
    ATTRIBUTE = 3
    ALIAS = 4

def sort_code_string(
    code: str,
    extension: Optional[str] = None,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    disregard_skip: bool = False,
    show_diff: Union[bool, TextIO] = False,
    **config_kwargs: Any
) -> str:
    input_stream = StringIO(code)
    output_stream = StringIO()
    config = _config(path=file_path, config=config, **config_kwargs)
    sort_stream(
        input_stream, output_stream, extension=extension, config=config,
        file_path=file_path, disregard_skip=disregard_skip, show_diff=show_diff
    )
    output_stream.seek(0)
    return output_stream.read()

def check_code_string(
    code: str,
    show_diff: Union[bool, TextIO] = False,
    extension: Optional[str] = None,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    disregard_skip: bool = False,
    **config_kwargs: Any
) -> bool:
    config = _config(path=file_path, config=config, **config_kwargs)
    return check_stream(
        StringIO(code), show_diff=show_diff, extension=extension, config=config,
        file_path=file_path, disregard_skip=disregard_skip
    )

def sort_stream(
    input_stream: TextIO,
    output_stream: TextIO,
    extension: Optional[str] = None,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    disregard_skip: bool = False,
    show_diff: Union[bool, TextIO] = False,
    raise_on_skip: bool = True,
    **config_kwargs: Any
) -> bool:
    extension = extension or (file_path and file_path.suffix.lstrip('.')) or 'py'
    if show_diff:
        _output_stream = StringIO()
        _input_stream = StringIO(input_stream.read())
        changed = sort_stream(
            input_stream=_input_stream, output_stream=_output_stream, extension=extension,
            config=config, file_path=file_path, disregard_skip=disregard_skip,
            raise_on_skip=raise_on_skip, **config_kwargs
        )
        _output_stream.seek(0)
        _input_stream.seek(0)
        show_unified_diff(
            file_input=_input_stream.read(), file_output=_output_stream.read(),
            file_path=file_path, output=output_stream if show_diff is True else show_diff,
            color_output=config.color_output
        )
        return changed
    config = _config(path=file_path, config=config, **config_kwargs)
    content_source = str(file_path or 'Passed in content')
    if not disregard_skip and file_path and config.is_skipped(file_path):
        raise FileSkipSetting(content_source)
    _internal_output = output_stream
    if config.atomic:
        try:
            file_content = input_stream.read()
            compile(file_content, content_source, 'exec', flags=0, dont_inherit=True)
        except SyntaxError:
            if extension not in CYTHON_EXTENSIONS:
                raise ExistingSyntaxErrors(content_source)
            if config.verbose:
                warn(f'{content_source} Python AST errors found but ignored due to Cython extension')
        input_stream = StringIO(file_content)
        if not output_stream.readable():
            _internal_output = StringIO()
    try:
        changed = core.process(
            input_stream, _internal_output, extension=extension, config=config,
            raise_on_skip=raise_on_skip
        )
    except FileSkipComment:
        raise FileSkipComment(content_source)
    if config.atomic:
        _internal_output.seek(0)
        try:
            compile(_internal_output.read(), content_source, 'exec', flags=0, dont_inherit=True)
            _internal_output.seek(0)
        except SyntaxError:
            if extension not in CYTHON_EXTENSIONS:
                raise IntroducedSyntaxErrors(content_source)
            if config.verbose:
                warn(f'{content_source} Python AST errors found but ignored due to Cython extension')
        if _internal_output != output_stream:
            output_stream.write(_internal_output.read())
    return changed

def check_stream(
    input_stream: TextIO,
    show_diff: Union[bool, TextIO] = False,
    extension: Optional[str] = None,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    disregard_skip: bool = False,
    **config_kwargs: Any
) -> bool:
    config = _config(path=file_path, config=config, **config_kwargs)
    if show_diff:
        input_stream = StringIO(input_stream.read())
    changed = sort_stream(
        input_stream=input_stream, output_stream=Empty, extension=extension,
        config=config, file_path=file_path, disregard_skip=disregard_skip
    )
    printer = create_terminal_printer(
        color=config.color_output, error=config.format_error, success=config.format_success
    )
    if not changed:
        if config.verbose and (not config.only_modified):
            printer.success(f'{file_path or ""} Everything Looks Good!')
        return True
    printer.error(f'{file_path or ""} Imports are incorrectly sorted and/or formatted.')
    if show_diff:
        output_stream = StringIO()
        input_stream.seek(0)
        file_contents = input_stream.read()
        sort_stream(
            input_stream=StringIO(file_contents), output_stream=output_stream,
            extension=extension, config=config, file_path=file_path, disregard_skip=disregard_skip
        )
        output_stream.seek(0)
        show_unified_diff(
            file_input=file_contents, file_output=output_stream.read(), file_path=file_path,
            output=None if show_diff is True else show_diff, color_output=config.color_output
        )
    return False

def check_file(
    filename: Union[str, Path],
    show_diff: Union[bool, TextIO] = False,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    disregard_skip: bool = True,
    extension: Optional[str] = None,
    **config_kwargs: Any
) -> bool:
    file_config = config
    if 'config_trie' in config_kwargs:
        config_trie = config_kwargs.pop('config_trie', None)
        if config_trie:
            config_info = config_trie.search(filename)
            if config.verbose:
                print(f'{config_info[0]} used for file {filename}')
            file_config = Config(**config_info[1])
    with io.File.read(filename) as source_file:
        return check_stream(
            source_file.stream, show_diff=show_diff, extension=extension, config=file_config,
            file_path=file_path or source_file.path, disregard_skip=disregard_skip, **config_kwargs
        )

def _tmp_file(source_file: File) -> Path:
    return source_file.path.with_suffix(source_file.path.suffix + '.isorted')

@contextlib.contextmanager
def _in_memory_output_stream_context() -> Iterator[StringIO]:
    yield StringIO(newline=None)

@contextlib.contextmanager
def _file_output_stream_context(filename: Union[str, Path], source_file: File) -> Iterator[TextIO]:
    tmp_file = _tmp_file(source_file)
    with tmp_file.open('w+', encoding=source_file.encoding, newline='') as output_stream:
        shutil.copymode(filename, tmp_file)
        yield output_stream

def sort_file(
    filename: Union[str, Path],
    extension: Optional[str] = None,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    disregard_skip: bool = True,
    ask_to_apply: bool = False,
    show_diff: Union[bool, TextIO] = False,
    write_to_stdout: bool = False,
    output: Optional[TextIO] = None,
    **config_kwargs: Any
) -> bool:
    file_config = config
    if 'config_trie' in config_kwargs:
        config_trie = config_kwargs.pop('config_trie', None)
        if config_trie:
            config_info = config_trie.search(filename)
            if config.verbose:
                print(f'{config_info[0]} used for file {filename}')
            file_config = Config(**config_info[1])
    with io.File.read(filename) as source_file:
        actual_file_path = file_path or source_file.path
        config = _config(path=actual_file_path, config=file_config, **config_kwargs)
        changed = False
        try:
            if write_to_stdout:
                changed = sort_stream(
                    input_stream=source_file.stream, output_stream=sys.stdout, config=config,
                    file_path=actual_file_path, disregard_skip=disregard_skip, extension=extension
                )
            elif output is None:
                try:
                    if config.overwrite_in_place:
                        output_stream_context = _in_memory_output_stream_context()
                    else:
                        output_stream_context = _file_output_stream_context(filename, source_file)
                    with output_stream_context as output_stream:
                        changed = sort_stream(
                            input_stream=source_file.stream, output_stream=output_stream,
                            config=config, file_path=actual_file_path, disregard_skip=disregard_skip,
                            extension=extension
                        )
                        output_stream.seek(0)
                        if changed:
                            if show_diff or ask_to_apply:
                                source_file.stream.seek(0)
                                show_unified_diff(
                                    file_input=source_file.stream.read(), file_output=output_stream.read(),
                                    file_path=actual_file_path, output=None if show_diff is True else cast(TextIO, show_diff),
                                    color_output=config.color_output
                                )
                                if show_diff or (ask_to_apply and (not ask_whether_to_apply_changes_to_file(str(source_file.path)))):
                                    return False
                            source_file.stream.close()
                            if config.overwrite_in_place:
                                output_stream.seek(0)
                                with source_file.path.open('w') as fs:
                                    shutil.copyfileobj(output_stream, fs)
                    if changed:
                        if not config.overwrite_in_place:
                            tmp_file = _tmp_file(source_file)
                            tmp_file.replace(source_file.path)
                        if not config.quiet:
                            print(f'Fixing {source_file.path}')
                finally:
                    if not config.overwrite_in_place:
                        tmp_file = _tmp_file(source_file)
                        tmp_file.unlink(missing_ok=True)
            else:
                changed = sort_stream(
                    input_stream=source_file.stream, output_stream=output, config=config,
                    file_path=actual_file_path, disregard_skip=disregard_skip, extension=extension
                )
                if changed and show_diff:
                    source_file.stream.seek(0)
                    output.seek(0)
                    show_unified_diff(
                        file_input=source_file.stream.read(), file_output=output.read(),
                        file_path=actual_file_path, output=None if show_diff is True else show_diff,
                        color_output=config.color_output
                    )
                source_file.stream.close()
        except ExistingSyntaxErrors:
            warn(f'{actual_file_path} unable to sort due to existing syntax errors')
        except IntroducedSyntaxErrors:
            warn(f'{actual_file_path} unable to sort as isort introduces new syntax errors')
        return changed

def find_imports_in_code(
    code: str,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    unique: Union[bool, ImportKey] = False,
    top_only: bool = False,
    **config_kwargs: Any
) -> Iterator[Any]:
    yield from find_imports_in_stream(
        input_stream=StringIO(code), config=config, file_path=file_path,
        unique=unique, top_only=top_only, **config_kwargs
    )

def find_imports_in_stream(
    input_stream: TextIO,
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    unique: Union[bool, ImportKey] = False,
    top_only: bool = False,
    _seen: Optional[Set[str]] = None,
    **config_kwargs: Any
) -> Iterator[Any]:
    config = _config(config=config, **config_kwargs)
    identified_imports = identify.imports(input_stream, config=config, file_path=file_path, top_only=top_only)
    if not unique:
        yield from identified_imports
    seen = set() if _seen is None else _seen
    for identified_import in identified_imports:
        if unique in (True, ImportKey.ALIAS):
            key = identified_import.statement()
        elif unique == ImportKey.ATTRIBUTE:
            key = f'{identified_import.module}.{identified_import.attribute}'
        elif unique == ImportKey.MODULE:
            key = identified_import.module
        elif unique == ImportKey.PACKAGE:
            key = identified_import.module.split('.')[0]
        if key and key not in seen:
            seen.add(key)
            yield identified_import

def find_imports_in_file(
    filename: Union[str, Path],
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    unique: Union[bool, ImportKey] = False,
    top_only: bool = False,
    **config_kwargs: Any
) -> Iterator[Any]:
    with io.File.read(filename) as source_file:
        yield from find_imports_in_stream(
            input_stream=source_file.stream, config=config, file_path=file_path or source_file.path,
            unique=unique, top_only=top_only, **config_kwargs
        )

def find_imports_in_paths(
    paths: Union[str, Path],
    config: Config = DEFAULT_CONFIG,
    file_path: Optional[Path] = None,
    unique: Union[bool, ImportKey] = False,
    top_only: bool = False,
    **config_kwargs: Any
) -> Iterator[Any]:
    config = _config(config=config, **config_kwargs)
    seen = set() if unique else None
    yield from chain(
        *(find_imports_in_file(
            file_name, unique=unique, config=config, top_only=top_only, _seen=seen
        ) for file_name in files.find(map(str, paths), config, [], []))
    )

def _config(
    path: Optional[Path] = None,
    config: Config = DEFAULT_CONFIG,
    **config_kwargs: Any
) -> Config:
    if path and (config is DEFAULT_CONFIG and 'settings_path' not in config_kwargs and ('settings_file' not in config_kwargs)):
        config_kwargs['settings_path'] = path
    if config_kwargs:
        if config is not DEFAULT_CONFIG:
            raise ValueError('You can either specify custom configuration options using kwargs or passing in a Config object. Not Both!')
        config = Config(**config_kwargs)
    return config
