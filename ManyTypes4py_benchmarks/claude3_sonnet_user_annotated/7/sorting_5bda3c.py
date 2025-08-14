import re
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Union, Pattern, Match, TypeVar

if TYPE_CHECKING:
    from .settings import Config
else:
    Config = Any

_import_line_intro_re: Pattern[str] = re.compile("^(?:from|import) ")
_import_line_midline_import_re: Pattern[str] = re.compile(" import ")

T = TypeVar('T')

def module_key(
    module_name: str,
    config: Config,
    sub_imports: bool = False,
    ignore_case: bool = False,
    section_name: Optional[Any] = None,
    straight_import: Optional[bool] = False,
) -> str:
    match: Optional[Match[str]] = re.match(r"^(\.+)\s*(.*)", module_name)
    if match:
        sep: str = " " if config.reverse_relative else "_"
        module_name = sep.join(match.groups())

    prefix: str = ""
    if ignore_case:
        module_name = str(module_name).lower()
    else:
        module_name = str(module_name)

    if sub_imports and config.order_by_type:
        if module_name in config.constants:
            prefix = "A"
        elif module_name in config.classes:
            prefix = "B"
        elif module_name in config.variables:
            prefix = "C"
        elif module_name.isupper() and len(module_name) > 1:  # see issue #376
            prefix = "A"
        elif module_name in config.classes or module_name[0:1].isupper():
            prefix = "B"
        else:
            prefix = "C"
    if not config.case_sensitive:
        module_name = module_name.lower()

    length_sort: bool = (
        config.length_sort
        or (config.length_sort_straight and straight_import)
        or str(section_name).lower() in config.length_sort_sections
    )
    _length_sort_maybe: str = (str(len(module_name)) + ":" + module_name) if length_sort else module_name
    return f"{module_name in config.force_to_top and 'A' or 'B'}{prefix}{_length_sort_maybe}"


def section_key(line: str, config: Config) -> str:
    section: str = "B"

    if (
        not config.sort_relative_in_force_sorted_sections
        and config.reverse_relative
        and line.startswith("from .")
    ):
        match: Optional[Match[str]] = re.match(r"^from (\.+)\s*(.*)", line)
        if match:  # pragma: no cover - regex always matches if line starts with "from ."
            line = f"from {' '.join(match.groups())}"
    if config.group_by_package and line.strip().startswith("from"):
        line = line.split(" import ", 1)[0]

    if config.lexicographical:
        line = _import_line_intro_re.sub("", _import_line_midline_import_re.sub(".", line))
    else:
        line = re.sub("^from ", "", line)
        line = re.sub("^import ", "", line)
    if config.sort_relative_in_force_sorted_sections:
        sep: str = " " if config.reverse_relative else "_"
        line = re.sub(r"^(\.+)", rf"\1{sep}", line)
    if line.split(" ")[0] in config.force_to_top:
        section = "A"
    # * If honor_case_in_force_sorted_sections is true, and case_sensitive and
    #   order_by_type are different, only ignore case in part of the line.
    # * Otherwise, let order_by_type decide the sorting of the whole line. This
    #   is only "correct" if case_sensitive and order_by_type have the same value.
    if config.honor_case_in_force_sorted_sections and config.case_sensitive != config.order_by_type:
        split_module: List[str] = line.split(" import ", 1)
        if len(split_module) > 1:
            module_name: str = split_module[0]
            names: str = split_module[1]
            if not config.case_sensitive:
                module_name = module_name.lower()
            if not config.order_by_type:
                names = names.lower()
            line = " import ".join([module_name, names])
        elif not config.case_sensitive:
            line = line.lower()
    elif not config.order_by_type:
        line = line.lower()

    return f"{section}{len(line) if config.length_sort else ''}{line}"


def sort(
    config: Config,
    to_sort: Iterable[str],
    key: Optional[Callable[[str], Any]] = None,
    reverse: bool = False,
) -> List[str]:
    return config.sorting_function(to_sort, key=key, reverse=reverse)


def naturally(
    to_sort: Iterable[T], key: Optional[Callable[[T], Any]] = None, reverse: bool = False
) -> List[T]:
    """Returns a naturally sorted list"""
    if key is None:
        key_callback: Callable[[T], List[Any]] = _natural_keys  # type: ignore
    else:
        def key_callback(text: T) -> List[Any]:
            return _natural_keys(key(text))

    return sorted(to_sort, key=key_callback, reverse=reverse)


def _atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def _natural_keys(text: str) -> List[Any]:
    return [_atoi(c) for c in re.split(r"(\d+)", text)]
