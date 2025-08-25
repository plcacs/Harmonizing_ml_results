from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import itertools
from .patcomp import PatternCompiler
from . import pygram
from .fixer_util import does_tree_import

class BaseFix(object):
    PATTERN: Optional[str] = None
    pattern: Any = None
    pattern_tree: Any = None
    options: Dict[str, Any]
    filename: Optional[str] = None
    numbers: Iterator[int] = itertools.count(1)
    used_names: Set[str] = set()
    order: str = 'post'
    explicit: bool = False
    run_order: int = 5
    _accept_type: Any = None
    keep_line_order: bool = False
    BM_compatible: bool = False
    syms: Any = pygram.python_symbols

    def __init__(self, options: Dict[str, Any], log: List[str]) -> None:
        self.options = options
        self.log: List[str] = log
        self.compile_pattern()

    def compile_pattern(self) -> None:
        if self.PATTERN is not None:
            PC: PatternCompiler = PatternCompiler()
            self.pattern, self.pattern_tree = PC.compile_pattern(self.PATTERN, with_tree=True)

    def set_filename(self, filename: str) -> None:
        self.filename = filename

    def match(self, node: Any) -> Optional[Dict[str, Any]]:
        results: Dict[str, Any] = {'node': node}
        if self.pattern.match(node, results):
            return results
        return None

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        raise NotImplementedError()

    def new_name(self, template: str = 'xxx_todo_changeme') -> str:
        name: str = template
        while name in self.used_names:
            name = template + str(next(self.numbers))
        self.used_names.add(name)
        return name

    def log_message(self, message: str) -> None:
        if getattr(self, 'first_log', True):
            self.first_log = False
            if self.filename is not None:
                self.log.append('### In file %s ###' % self.filename)
        self.log.append(message)

    def cannot_convert(self, node: Any, reason: Optional[str] = None) -> None:
        lineno: int = node.get_lineno()
        for_output: Any = node.clone()
        for_output.prefix = ''
        msg: str = 'Line %d: could not convert: %s' % (lineno, for_output)
        self.log_message(msg)
        if reason:
            self.log_message(reason)

    def warning(self, node: Any, reason: str) -> None:
        lineno: int = node.get_lineno()
        self.log_message('Line %d: %s' % (lineno, reason))

    def start_tree(self, tree: Any, filename: str) -> None:
        self.used_names = tree.used_names  # type: Set[str]
        self.set_filename(filename)
        self.numbers = itertools.count(1)
        self.first_log = True

    def finish_tree(self, tree: Any, filename: str) -> None:
        pass

class ConditionalFix(BaseFix):
    skip_on: Optional[str] = None

    def start_tree(self, *args: Any, **kwargs: Any) -> None:
        super(ConditionalFix, self).start_tree(*args, **kwargs)
        self._should_skip: Optional[bool] = None

    def should_skip(self, node: Any) -> bool:
        if self._should_skip is not None:
            return self._should_skip
        assert self.skip_on is not None
        pkg_parts = self.skip_on.split('.')
        name: str = pkg_parts[-1]
        pkg: str = '.'.join(pkg_parts[:-1])
        self._should_skip = does_tree_import(pkg, name, node)
        return self._should_skip