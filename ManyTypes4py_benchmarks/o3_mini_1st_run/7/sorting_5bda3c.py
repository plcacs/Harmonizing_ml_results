import re
from typing import Any, Callable, List, Optional, Pattern, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import Config
else:
    Config = Any

_import_line_intro_re: Pattern[str] = re.compile('^(?:from|import) ')
_import_line_midline_import_re: Pattern[str] = re.compile(' import ')


def module_key(module_name: str, config: Config, sub_imports: bool = False, ignore_case: bool = False, section_name: Optional[str] = None, straight_import: bool = False) -> str:
    match = re.match(r'^(\.+)\s*(.*)', module_name)
    if match:
        sep = ' ' if config.reverse_relative else '_'
        module_name = sep.join(match.groups())
    prefix = ''
    if ignore_case:
        module_name = str(module_name).lower()
    else:
        module_name = str(module_name)
    if sub_imports and config.order_by_type:
        if module_name in config.constants:
            prefix = 'A'
        elif module_name in config.classes:
            prefix = 'B'
        elif module_name in config.variables:
            prefix = 'C'
        elif module_name.isupper() and len(module_name) > 1:
            prefix = 'A'
        elif module_name in config.classes or module_name[0:1].isupper():
            prefix = 'B'
        else:
            prefix = 'C'
    if not config.case_sensitive:
        module_name = module_name.lower()
    length_sort = config.length_sort or (config.length_sort_straight and straight_import) or str(section_name).lower() in config.length_sort_sections
    _length_sort_maybe = f"{len(module_name)}:{module_name}" if length_sort else module_name
    return f'{"A" if module_name in config.force_to_top else "B"}{prefix}{_length_sort_maybe}'


def section_key(line: str, config: Config) -> str:
    section = 'B'
    if not config.sort_relative_in_force_sorted_sections and config.reverse_relative and line.startswith('from .'):
        match = re.match(r'^from (\.+)\s*(.*)', line)
        if match:
            line = f"from {' '.join(match.groups())}"
    if config.group_by_package and line.strip().startswith('from'):
        line = line.split(' import ', 1)[0]
    if config.lexicographical:
        line = _import_line_intro_re.sub('', _import_line_midline_import_re.sub('.', line))
    else:
        line = re.sub('^from ', '', line)
        line = re.sub('^import ', '', line)
    if config.sort_relative_in_force_sorted_sections:
        sep = ' ' if config.reverse_relative else '_'
        line = re.sub(r'^(\.+)', rf'\1{sep}', line)
    if line.split(' ')[0] in config.force_to_top:
        section = 'A'
    if config.honor_case_in_force_sorted_sections and config.case_sensitive != config.order_by_type:
        split_module = line.split(' import ', 1)
        if len(split_module) > 1:
            module_name, names = split_module
            if not config.case_sensitive:
                module_name = module_name.lower()
            if not config.order_by_type:
                names = names.lower()
            line = ' import '.join([module_name, names])
        elif not config.case_sensitive:
            line = line.lower()
    elif not config.order_by_type:
        line = line.lower()
    return f"{section}{(len(line) if config.length_sort else '')}{line}"


def sort(config: Config, to_sort: List[Any], key: Optional[Callable[[Any], Any]] = None, reverse: bool = False) -> List[Any]:
    return config.sorting_function(to_sort, key=key, reverse=reverse)


def naturally(to_sort: List[str], key: Optional[Callable[[str], str]] = None, reverse: bool = False) -> List[str]:
    """Returns a naturally sorted list"""
    if key is None:
        key_callback: Callable[[str], List[Union[int, str]]] = _natural_keys
    else:
        def key_callback(text: str) -> List[Union[int, str]]:
            return _natural_keys(key(text))
    return sorted(to_sort, key=key_callback, reverse=reverse)


def _atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def _natural_keys(text: str) -> List[Union[int, str]]:
    return [_atoi(c) for c in re.split(r'(\d+)', text)]