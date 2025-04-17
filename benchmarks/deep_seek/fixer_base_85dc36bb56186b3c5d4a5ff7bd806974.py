"""Base class for fixers (optional, but recommended)."""
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
from .patcomp import PatternCompiler
from . import pygram
from .fixer_util import does_tree_import

T = TypeVar('T')

class BaseFix(object):
    """Optional base class for fixers.

    The subclass name must be FixFooBar where FooBar is the result of
    removing underscores and capitalizing the words of the fix name.
    For example, the class name for a fixer named 'has_key' should be
    FixHasKey.
    """
    PATTERN: Optional[str] = None
    pattern: Any = None
    pattern_tree: Any = None
    options: Dict[str, Any]
    filename: Optional[str] = None
    numbers: itertools.count = itertools.count(1)
    used_names: Set[str] = set()
    order: str = 'post'
    explicit: bool = False
    run_order: int = 5
    _accept_type: Optional[Any] = None
    keep_line_order: bool = False
    BM_compatible: bool = False
    syms: Any = pygram.python_symbols
    log: List[str]
    first_log: bool = True

    def __init__(self, options: Dict[str, Any], log: List[str]) -> None:
        """Initializer.  Subclass may override.

        Args:
            options: a dict containing the options passed to RefactoringTool
            that could be used to customize the fixer through the command line.
            log: a list to append warnings and other messages to.
        """
        self.options = options
        self.log = log
        self.compile_pattern()

    def compile_pattern(self) -> None:
        """Compiles self.PATTERN into self.pattern.

        Subclass may override if it doesn't want to use
        self.{pattern,PATTERN} in .match().
        """
        if self.PATTERN is not None:
            PC = PatternCompiler()
            self.pattern, self.pattern_tree = PC.compile_pattern(self.PATTERN, with_tree=True)

    def set_filename(self, filename: str) -> None:
        """Set the filename.

        The main refactoring tool should call this.
        """
        self.filename = filename

    def match(self, node: Any) -> Union[bool, Dict[str, Any]]:
        """Returns match for a given parse tree node.

        Should return a true or false object (not necessarily a bool).
        It may return a non-empty dict of matching sub-nodes as
        returned by a matching pattern.

        Subclass may override.
        """
        results = {'node': node}
        return self.pattern.match(node, results) and results

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        """Returns the transformation for a given parse tree node.

        Args:
          node: the root of the parse tree that matched the fixer.
          results: a dict mapping symbolic names to part of the match.

        Returns:
          None, or a node that is a modified copy of the
          argument node.  The node argument may also be modified in-place to
          effect the same change.

        Subclass *must* override.
        """
        raise NotImplementedError()

    def new_name(self, template: str = 'xxx_todo_changeme') -> str:
        """Return a string suitable for use as an identifier

        The new name is guaranteed not to conflict with other identifiers.
        """
        name = template
        while name in self.used_names:
            name = template + str(next(self.numbers))
        self.used_names.add(name)
        return name

    def log_message(self, message: str) -> None:
        if self.first_log:
            self.first_log = False
            self.log.append(f'### In file {self.filename} ###')
        self.log.append(message)

    def cannot_convert(self, node: Any, reason: Optional[str] = None) -> None:
        """Warn the user that a given chunk of code is not valid Python 3,
        but that it cannot be converted automatically.

        First argument is the top-level node for the code in question.
        Optional second argument is why it can't be converted.
        """
        lineno = node.get_lineno()
        for_output = node.clone()
        for_output.prefix = ''
        msg = 'Line %d: could not convert: %s'
        self.log_message(msg % (lineno, for_output))
        if reason:
            self.log_message(reason)

    def warning(self, node: Any, reason: str) -> None:
        """Used for warning the user about possible uncertainty in the
        translation.

        First argument is the top-level node for the code in question.
        Optional second argument is why it can't be converted.
        """
        lineno = node.get_lineno()
        self.log_message(f'Line {lineno}: {reason}')

    def start_tree(self, tree: Any, filename: str) -> None:
        """Some fixers need to maintain tree-wide state.
        This method is called once, at the start of tree fix-up.

        tree - the root node of the tree to be processed.
        filename - the name of the file the tree came from.
        """
        self.used_names = tree.used_names
        self.set_filename(filename)
        self.numbers = itertools.count(1)
        self.first_log = True

    def finish_tree(self, tree: Any, filename: str) -> None:
        """Some fixers need to maintain tree-wide state.
        This method is called once, at the conclusion of tree fix-up.

        tree - the root node of the tree to be processed.
        filename - the name of the file the tree came from.
        """
        pass

class ConditionalFix(BaseFix):
    """Base class for fixers which not execute if an import is found."""
    skip_on: Optional[str] = None
    _should_skip: Optional[bool] = None

    def start_tree(self, *args: Any) -> None:
        super(ConditionalFix, self).start_tree(*args)
        self._should_skip = None

    def should_skip(self, node: Any) -> bool:
        if self._should_skip is not None:
            return self._should_skip
        if self.skip_on is None:
            return False
        pkg = self.skip_on.split('.')
        name = pkg[-1]
        pkg = '.'.join(pkg[:-1])
        self._should_skip = does_tree_import(pkg, name, node)
        return self._should_skip
