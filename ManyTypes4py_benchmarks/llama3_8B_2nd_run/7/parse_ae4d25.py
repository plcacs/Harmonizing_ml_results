from collections import OrderedDict, defaultdict
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Set, Tuple

def _infer_line_separator(contents: str) -> str:
    if '\r\n' in contents:
        return '\r\n'
    if '\r' in contents:
        return '\r'
    return '\n'

def normalize_line(raw_line: str) -> Tuple[str, str]:
    """Normalizes import related statements in the provided line.

    Returns (normalized_line: str, raw_line: str)
    """
    line = re.sub('from(\\.+)cimport ', 'from \\g<1> cimport ', raw_line)
    line = re.sub('from(\\.+)import ', 'from \\g<1> import ', line)
    line = line.replace('import*', 'import *')
    line = re.sub(' (\\.+)import ', ' \\g<1> import ', line)
    line = re.sub(' (\\.+)cimport ', ' \\g<1> cimport ', line)
    line = line.replace('\t', ' ')
    return (line, raw_line)

def import_type(line: str, config: Config = DEFAULT_CONFIG) -> Optional[str]:
    """If the current line is an import line it will return its type (from or straight)"""
    if config.honor_noqa and line.lower().rstrip().endswith('noqa'):
        return None
    if 'isort:skip' in line or 'isort: skip' in line or 'isort: split' in line:
        return None
    if line.startswith(('import ', 'cimport ')):
        return 'straight'
    if line.startswith('from '):
        return 'from'
    return None

def strip_syntax(import_string: str) -> str:
    import_string = import_string.replace('_import', '[[i]]')
    import_string = import_string.replace('_cimport', '[[ci]]')
    for remove_syntax in ['\\', '(', ')', ',']:
        import_string = import_string.replace(remove_syntax, ' ')
    import_list = import_string.split()
    for key in ('from', 'import', 'cimport'):
        if key in import_list:
            import_list.remove(key)
    import_string = ' '.join(import_list)
    import_string = import_string.replace('[[i]]', '_import')
    import_string = import_string.replace('[[ci]]', '_cimport')
    return import_string.replace('{ ', '{|').replace(' }', '|}')

def skip_line(line: str, in_quote: str, index: int, section_comments: Dict[str, Any], needs_import: bool = True) -> Tuple[bool, str]:
    """Determine if a given line should be skipped.

    Returns back a tuple containing:

    (skip_line: bool,
     in_quote: str,)
    """
    should_skip = bool(in_quote)
    if '"' in line or "'" in line:
        char_index = 0
        while char_index < len(line):
            if line[char_index] == '\\':
                char_index += 1
            elif in_quote:
                if line[char_index:char_index + len(in_quote)] == in_quote:
                    in_quote = ''
            elif line[char_index] in ("'", '"'):
                long_quote = line[char_index:char_index + 3]
                if long_quote in ('"""', "'''"):
                    in_quote = long_quote
                    char_index += 2
                else:
                    in_quote = line[char_index]
            elif line[char_index] == '#':
                break
            char_index += 1
    if ';' in line.split('#')[0] and needs_import:
        for part in (part.strip() for part in line.split(';')):
            if part and (not part.startswith('from ')) and (not part.startswith(('import ', 'cimport '))):
                should_skip = True
    return (bool(should_skip or in_quote), in_quote)

class ParsedContent(NamedTuple):
    in_lines: List[str]
    lines_without_imports: List[str]
    import_index: int
    place_imports: Dict[str, List[str]]
    import_placements: Dict[str, str]
    as_map: Dict[str, Dict[str, List[str]]]
    imports: Dict[str, Dict[str, OrderedDict]]
    categorized_comments: Dict[str, Dict[str, List[str]]]
    change_count: int
    original_line_count: int
    line_separator: str
    sections: List[str]
    verbose_output: List[str]
    trailing_commas: Set[str]

def file_contents(contents: str, config: Config = DEFAULT_CONFIG) -> ParsedContent:
    """Parses a python file taking out and categorizing imports."""
    # ... (rest of the code remains the same)
