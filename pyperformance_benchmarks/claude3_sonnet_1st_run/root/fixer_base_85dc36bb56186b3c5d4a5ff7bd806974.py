'Base class for fixers (optional, but recommended).'
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from .patcomp import PatternCompiler
from . import pygram
from .fixer_util import does_tree_import

class BaseFix(object):
    "Optional base class for fixers.\n\n    The subclass name must be FixFooBar where FooBar is the result of\n    removing underscores and capitalizing the words of the fix name.\n    For example, the class name for a fixer named 'has_key' should be\n    FixHasKey.\n    "
    PATTERN: Optional[str] = None
    pattern: Any = None
    pattern_tree: Any = None
    options: Optional[Dict[str, Any]] = None
    filename: Optional[str] = None
    numbers = itertools.count(1)
    used_names: Set[str] = set()
    order: str = 'post'
    explicit: bool = False
    run_order: int = 5
    _accept_type: Optional[Any] = None
    keep_line_order: bool = False
    BM_compatible: bool = False
    syms = pygram.python_symbols

    def __init__(self, options: Dict[str, Any], log: List[str]) -> None:
        'Initializer.  Subclass may override.\n\n        Args:\n            options: a dict containing the options passed to RefactoringTool\n            that could be used to customize the fixer through the command line.\n            log: a list to append warnings and other messages to.\n        '
        self.options = options
        self.log = log
        self.compile_pattern()

    def compile_pattern(self) -> None:
        "Compiles self.PATTERN into self.pattern.\n\n        Subclass may override if it doesn't want to use\n        self.{pattern,PATTERN} in .match().\n        "
        if (self.PATTERN is not None):
            PC = PatternCompiler()
            (self.pattern, self.pattern_tree) = PC.compile_pattern(self.PATTERN, with_tree=True)

    def set_filename(self, filename: str) -> None:
        'Set the filename.\n\n        The main refactoring tool should call this.\n        '
        self.filename = filename

    def match(self, node: Any) -> Union[Dict[str, Any], bool]:
        'Returns match for a given parse tree node.\n\n        Should return a true or false object (not necessarily a bool).\n        It may return a non-empty dict of matching sub-nodes as\n        returned by a matching pattern.\n\n        Subclass may override.\n        '
        results: Dict[str, Any] = {'node': node}
        return (self.pattern.match(node, results) and results)

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        'Returns the transformation for a given parse tree node.\n\n        Args:\n          node: the root of the parse tree that matched the fixer.\n          results: a dict mapping symbolic names to part of the match.\n\n        Returns:\n          None, or a node that is a modified copy of the\n          argument node.  The node argument may also be modified in-place to\n          effect the same change.\n\n        Subclass *must* override.\n        '
        raise NotImplementedError()

    def new_name(self, template: str = 'xxx_todo_changeme') -> str:
        'Return a string suitable for use as an identifier\n\n        The new name is guaranteed not to conflict with other identifiers.\n        '
        name = template
        while (name in self.used_names):
            name = (template + str(next(self.numbers)))
        self.used_names.add(name)
        return name

    def log_message(self, message: str) -> None:
        if self.first_log:
            self.first_log = False
            self.log.append(('### In file %s ###' % self.filename))
        self.log.append(message)

    def cannot_convert(self, node: Any, reason: Optional[str] = None) -> None:
        "Warn the user that a given chunk of code is not valid Python 3,\n        but that it cannot be converted automatically.\n\n        First argument is the top-level node for the code in question.\n        Optional second argument is why it can't be converted.\n        "
        lineno = node.get_lineno()
        for_output = node.clone()
        for_output.prefix = ''
        msg = 'Line %d: could not convert: %s'
        self.log_message((msg % (lineno, for_output)))
        if reason:
            self.log_message(reason)

    def warning(self, node: Any, reason: str) -> None:
        "Used for warning the user about possible uncertainty in the\n        translation.\n\n        First argument is the top-level node for the code in question.\n        Optional second argument is why it can't be converted.\n        "
        lineno = node.get_lineno()
        self.log_message(('Line %d: %s' % (lineno, reason)))

    def start_tree(self, tree: Any, filename: str) -> None:
        'Some fixers need to maintain tree-wide state.\n        This method is called once, at the start of tree fix-up.\n\n        tree - the root node of the tree to be processed.\n        filename - the name of the file the tree came from.\n        '
        self.used_names = tree.used_names
        self.set_filename(filename)
        self.numbers = itertools.count(1)
        self.first_log: bool = True

    def finish_tree(self, tree: Any, filename: str) -> None:
        'Some fixers need to maintain tree-wide state.\n        This method is called once, at the conclusion of tree fix-up.\n\n        tree - the root node of the tree to be processed.\n        filename - the name of the file the tree came from.\n        '
        pass

class ConditionalFix(BaseFix):
    ' Base class for fixers which not execute if an import is found. '
    skip_on: Optional[str] = None

    def start_tree(self, *args: Any) -> None:
        super(ConditionalFix, self).start_tree(*args)
        self._should_skip: Optional[bool] = None

    def should_skip(self, node: Any) -> bool:
        if (self._should_skip is not None):
            return self._should_skip
        pkg = self.skip_on.split('.')
        name = pkg[(- 1)]
        pkg = '.'.join(pkg[:(- 1)])
        self._should_skip = does_tree_import(pkg, name, node)
        return self._should_skip
