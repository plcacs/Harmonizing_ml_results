import itertools
from .patcomp import PatternCompiler
from . import pygram
from .fixer_util import does_tree_import
from typing import Dict, Any, Union, List

class BaseFix:
    PATTERN: Any = None
    pattern: Any = None
    pattern_tree: Any = None
    options: Dict[str, Any] = None
    filename: str = None
    numbers: itertools.count = itertools.count(1)
    used_names: set = set()
    order: str = 'post'
    explicit: bool = False
    run_order: int = 5
    _accept_type: Any = None
    keep_line_order: bool = False
    BM_compatible: bool = False
    syms: Any = pygram.python_symbols

    def __init__(self, options: Dict[str, Any], log: List[str]) -> None:
        self.options = options
        self.log = log
        self.compile_pattern()

    def compile_pattern(self) -> None:
        if self.PATTERN is not None:
            PC = PatternCompiler()
            self.pattern, self.pattern_tree = PC.compile_pattern(self.PATTERN, with_tree=True)

    def set_filename(self, filename: str) -> None:
        self.filename = filename

    def match(self, node: Any) -> Union[bool, Dict[str, Any]]:
        results: Dict[str, Any] = {'node': node}
        return self.pattern.match(node, results) and results

    def transform(self, node: Any, results: Dict[str, Any]) -> Any:
        raise NotImplementedError()

    def new_name(self, template: str = 'xxx_todo_changeme') -> str:
        name = template
        while name in self.used_names:
            name = template + str(next(self.numbers))
        self.used_names.add(name)
        return name

    def log_message(self, message: str) -> None:
        if self.first_log:
            self.first_log = False
            self.log.append('### In file %s ###' % self.filename)
        self.log.append(message)

    def cannot_convert(self, node: Any, reason: str = None) -> None:
        lineno = node.get_lineno()
        for_output = node.clone()
        for_output.prefix = ''
        msg = 'Line %d: could not convert: %s'
        self.log_message(msg % (lineno, for_output))
        if reason:
            self.log_message(reason)

    def warning(self, node: Any, reason: str) -> None:
        lineno = node.get_lineno()
        self.log_message('Line %d: %s' % (lineno, reason))

    def start_tree(self, tree: Any, filename: str) -> None:
        self.used_names = tree.used_names
        self.set_filename(filename)
        self.numbers = itertools.count(1)
        self.first_log = True

    def finish_tree(self, tree: Any, filename: str) -> None:
        pass

class ConditionalFix(BaseFix):
    skip_on: str = None

    def start_tree(self, *args: Any) -> None:
        super().start_tree(*args)
        self._should_skip = None

    def should_skip(self, node: Any) -> bool:
        if self._should_skip is not None:
            return self._should_skip
        pkg = self.skip_on.split('.')
        name = pkg[-1]
        pkg = '.'.join(pkg[:-1])
        self._should_skip = does_tree_import(pkg, name, node)
        return self._should_skip
