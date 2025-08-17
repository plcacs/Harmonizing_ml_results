"""
Main module.

Implement the central Checker class.
Also, it models the Bindings and Scopes.
"""
import doctest
import os
import sys
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, TypeVar, Union

PY2 = sys.version_info < (3, 0)
PY32 = sys.version_info < (3, 3)    # Python 2.5 to 3.2
PY33 = sys.version_info < (3, 4)    # Python 2.5 to 3.3
builtin_vars = dir(__import__('__builtin__' if PY2 else 'builtins'))

try:
    import ast
except ImportError:     # Python 2.5
    import _ast as ast

    if 'decorator_list' not in ast.ClassDef._fields:
        # Patch the missing attribute 'decorator_list'
        ast.ClassDef.decorator_list = ()
        ast.FunctionDef.decorator_list = property(lambda s: s.decorators)

from pyflakes import messages


if PY2:
    def getNodeType(node_class: Type[ast.AST]) -> str:
        # workaround str.upper() which is locale-dependent
        return str(unicode(node_class.__name__).upper())
else:
    def getNodeType(node_class: Type[ast.AST]) -> str:
        return node_class.__name__.upper()

# Python >= 3.3 uses ast.Try instead of (ast.TryExcept + ast.TryFinally)
if PY32:
    def getAlternatives(n: ast.AST) -> List[List[ast.AST]]:
        if isinstance(n, (ast.If, ast.TryFinally)):
            return [n.body]
        if isinstance(n, ast.TryExcept):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
else:
    def getAlternatives(n: ast.AST) -> List[List[ast.AST]]:
        if isinstance(n, ast.If):
            return [n.body]
        if isinstance(n, ast.Try):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]


class _FieldsOrder(dict):
    """Fix order of AST node fields."""

    def _get_fields(self, node_class: Type[ast.AST]) -> Tuple[str, ...]:
        # handle iter before target, and generators before element
        fields = node_class._fields
        if 'iter' in fields:
            key_first = 'iter'.find
        elif 'generators' in fields:
            key_first = 'generators'.find
        else:
            key_first = 'value'.find
        return tuple(sorted(fields, key=key_first, reverse=True))

    def __missing__(self, node_class: Type[ast.AST]) -> Tuple[str, ...]:
        self[node_class] = fields = self._get_fields(node_class)
        return fields


def iter_child_nodes(node: ast.AST, omit: Optional[str] = None, _fields_order: _FieldsOrder = _FieldsOrder()) -> Generator[ast.AST, None, None]:
    """
    Yield all direct child nodes of *node*, that is, all fields that
    are nodes and all items of fields that are lists of nodes.
    """
    for name in _fields_order[node.__class__]:
        if name == omit:
            continue
        field = getattr(node, name, None)
        if isinstance(field, ast.AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                yield item


class Binding(object):
    """
    Represents the binding of a value to a name.

    The checker uses this to keep track of which names have been bound and
    which names have not. See L{Assignment} for a special type of binding that
    is checked with stricter rules.

    @ivar used: pair of (L{Scope}, line-number) indicating the scope and
                line number that this binding was last used
    """

    def __init__(self, name: str, source: ast.AST) -> None:
        self.name = name
        self.source = source
        self.used: Union[bool, Tuple['Scope', ast.AST]] = False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return '<%s object %r from line %r at 0x%x>' % (self.__class__.__name__,
                                                        self.name,
                                                        self.source.lineno,
                                                        id(self))

    def redefines(self, other: 'Binding') -> bool:
        return isinstance(other, Definition) and self.name == other.name


class Definition(Binding):
    """
    A binding that defines a function or a class.
    """


class Importation(Definition):
    """
    A binding created by an import statement.

    @ivar fullName: The complete name given to the import statement,
        possibly including multiple dotted components.
    @type fullName: C{str}
    """

    def __init__(self, name: str, source: ast.AST) -> None:
        self.fullName = name
        self.redefined: List[ast.AST] = []
        name = name.split('.')[0]
        super(Importation, self).__init__(name, source)

    def redefines(self, other: Binding) -> bool:
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        return isinstance(other, Definition) and self.name == other.name


class Argument(Binding):
    """
    Represents binding a name as an argument.
    """


class Assignment(Binding):
    """
    Represents binding a name with an explicit assignment.

    The checker will raise warnings for any Assignment that isn't used. Also,
    the checker does not consider assignments in tuple/list unpacking to be
    Assignments, rather it treats them as simple Bindings.
    """


class FunctionDefinition(Definition):
    pass


class ClassDefinition(Definition):
    pass


class ExportBinding(Binding):
    """
    A binding created by an C{__all__} assignment.  If the names in the list
    can be determined statically, they will be treated as names for export and
    additional checking applied to them.

    The only C{__all__} assignment that can be recognized is one which takes
    the value of a literal list containing literal strings.  For example::

        __all__ = ["foo", "bar"]

    Names which are imported and not otherwise used but appear in the value of
    C{__all__} will not have an unused import warning reported for them.
    """

    def __init__(self, name: str, source: ast.AST, scope: 'Scope') -> None:
        if '__all__' in scope and isinstance(source, ast.AugAssign):
            self.names = list(scope['__all__'].names)
        else:
            self.names = []
        if isinstance(source.value, (ast.List, ast.Tuple)):
            for node in source.value.elts:
                if isinstance(node, ast.Str):
                    self.names.append(node.s)
        super(ExportBinding, self).__init__(name, source)


class Scope(dict):
    importStarred: bool = False       # set to True when import * is found

    def __repr__(self) -> str:
        scope_cls = self.__class__.__name__
        return '<%s at 0x%x %s>' % (scope_cls, id(self), dict.__repr__(self))


class ClassScope(Scope):
    pass


class FunctionScope(Scope):
    """
    I represent a name scope for a function.

    @ivar globals: Names declared 'global' in this function.
    """
    usesLocals: bool = False
    alwaysUsed: Set[str] = set(['__tracebackhide__',
                              '__traceback_info__', '__traceback_supplement__'])

    def __init__(self) -> None:
        super(FunctionScope, self).__init__()
        # Simplify: manage the special locals as globals
        self.globals: Set[str] = self.alwaysUsed.copy()
        self.returnValue: Optional[ast.AST] = None     # First non-empty return
        self.isGenerator: bool = False    # Detect a generator

    def unusedAssignments(self) -> Generator[Tuple[str, 'Assignment'], None, None]:
        """
        Return a generator for the assignments which have not been used.
        """
        for name, binding in self.items():
            if (not binding.used and name not in self.globals
                    and not self.usesLocals
                    and isinstance(binding, Assignment)):
                yield name, binding


class GeneratorScope(Scope):
    pass


class ModuleScope(Scope):
    pass


# Globally defined names which are not attributes of the builtins module, or
# are only present on some platforms.
_MAGIC_GLOBALS = ['__file__', '__builtins__', 'WindowsError']


def getNodeName(node: ast.AST) -> Optional[str]:
    # Returns node.id, or node.name, or None
    if hasattr(node, 'id'):     # One of the many nodes with an id
        return node.id  # type: ignore
    if hasattr(node, 'name'):   # a ExceptHandler node
        return node.name  # type: ignore
    return None


class Checker(object):
    """
    I check the cleanliness and sanity of Python code.

    @ivar _deferredFunctions: Tracking list used by L{deferFunction}.  Elements
        of the list are two-tuples.  The first element is the callable passed
        to L{deferFunction}.  The second element is a copy of the scope stack
        at the time L{deferFunction} was called.

    @ivar _deferredAssignments: Similar to C{_deferredFunctions}, but for
        callables which are deferred assignment checks.
    """

    nodeDepth: int = 0
    offset: Optional[Tuple[int, int]] = None
    traceTree: bool = False

    builtIns: Set[str] = set(builtin_vars).union(_MAGIC_GLOBALS)
    _customBuiltIns = os.environ.get('PYFLAKES_BUILTINS')
    if _customBuiltIns:
        builtIns.update(_customBuiltIns.split(','))
    del _customBuiltIns

    def __init__(self, tree: ast.AST, filename: str = '(none)', builtins: Optional[Set[str]] = None,
                 withDoctest: bool = 'PYFLAKES_DOCTEST' in os.environ) -> None:
        self._nodeHandlers: Dict[Type[ast.AST], Callable[[ast.AST], None]] = {}
        self._deferredFunctions: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]] = []
        self._deferredAssignments: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]] = []
        self.deadScopes: List[Scope] = []
        self.messages: List[messages.Message] = []
        self.filename: str = filename
        if builtins:
            self.builtIns = self.builtIns.union(builtins)
        self.withDoctest: bool = withDoctest
        self.scopeStack: List[Scope] = [ModuleScope()]
        self.exceptHandlers: List[List[str]] = [()]
        self.futuresAllowed: bool = True
        self.root: ast.AST = tree
        self.handleChildren(tree)
        self.runDeferred(self._deferredFunctions)
        # Set _deferredFunctions to None so that deferFunction will fail
        # noisily if called after we've run through the deferred functions.
        self._deferredFunctions = None
        self.runDeferred(self._deferredAssignments)
        # Set _deferredAssignments to None so that deferAssignment will fail
        # noisily if called after we've run through the deferred assignments.
        self._deferredAssignments = None
        del self.scopeStack[1:]
        self.popScope()
        self.checkDeadScopes()

    def deferFunction(self, callable: Callable[[], None]) -> None:
        """
        Schedule a function handler to be called just before completion.

        This is used for handling function bodies, which must be deferred
        because code later in the file might modify the global scope. When
        `callable` is called, the scope at the time this is called will be
        restored, however it will contain any new bindings added to it.
        """
        self._deferredFunctions.append((callable, self.scopeStack[:], self.offset))

    def deferAssignment(self, callable: Callable[[], None]) -> None:
        """
        Schedule an assignment handler to be called just after deferred
        function handlers.
        """
        self._deferredAssignments.append((callable, self.scopeStack[:], self.offset))

    def runDeferred(self, deferred: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]]) -> None:
        """
        Run the callables in C{deferred} using their associated scope stack.
        """
        for handler, scope, offset in deferred:
            self.scopeStack = scope
            self.offset = offset
            handler()

    @property
    def scope(self) -> Scope:
        return self.scopeStack[-1]

    def popScope(self) -> None:
        self.deadScopes.append(self.scopeStack.pop())

    def checkDeadScopes(self) -> None:
        """
        Look at scopes which have been fully examined and report names in them
        which were imported but unused.
        """
        for scope in self.deadScopes:
            if isinstance(scope.get('__all__'), ExportBinding):
                all_names = set(scope['__all__'].names)
                if not scope.importStarred and \
                   os.path.basename(self.filename) != '__init__.py':
                    # Look for possible mistakes in the export list
                    undefined = all_names.difference(scope)
                    for name in undefined:
                        self.report(messages.UndefinedExport,
                                    scope['__all__'].source, name)
            else:
                all_names = []

            # Look for imported names that aren't used.
            for value in scope.values():
                if isinstance(value, Importation):
                    used = value.used or value.name in all_names
                    if not used:
                        messg = messages.UnusedImport
                        self.report(messg, value.source, value.name)
                    for node in value.redefined:
                        if isinstance(self.getParent(node), ast.For):
                            messg = messages.ImportShadowedByLoopVar
                        elif used:
                            continue
                        else:
                            messg = messages.RedefinedWhileUnused
                        self.report(messg, node, value.name, value.source)

    def pushScope(self, scopeClass: Type[Scope] = FunctionScope) -> None:
        self.scopeStack.append(scopeClass())

    def report(self, messageClass: Type[messages.Message], *args: Any, **kwargs: Any) -> None:
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def getParent(self, node: ast.AST) -> ast.AST:
        # Lookup the first parent which is not Tuple, List or Starred
        while True:
            node = node.parent  # type: ignore
            if not hasattr(node, 'elts') and not hasattr(node, 'ctx'):
                return node

    def getCommonAncestor(self, lnode: ast.AST, rnode: ast.AST, stop: ast.AST) -> Optional[ast.AST]:
        if stop in (lnode, rnode) or not (hasattr(lnode, 'parent') and
                                          hasattr(rnode, 'parent')):
            return None
        if lnode is rnode:
            return lnode

        if (lnode.depth > rnode.depth):  # type: ignore
            return self.getCommonAncestor(lnode.parent, rnode, stop)  # type: ignore
        if (lnode.depth < rnode.depth):  # type: ignore
            return self.getCommonAncestor(lnode, rnode.parent, stop)  # type: ignore
        return self.getCommonAncestor(lnode.parent, rnode.parent, stop)  # type: ignore

    def descendantOf(self, node: ast.AST, ancestors: List[ast.AST], stop: ast.AST) -> bool:
        for a in ancestors:
            if self.getCommonAncestor(node, a, stop):
                return True
        return False

    def differentForks(self, lnode: ast.AST, rnode: ast.AST) -> bool:
        """True, if lnode and rnode are located on different forks of IF/TRY"""
        ancestor = self.getCommonAncestor(lnode, rnode, self.root)
        parts = getAlternatives(ancestor) if ancestor else []
        if parts:
            for items in parts:
                if self.descendantOf(lnode, items, ancestor) ^ \
                   self.descendantOf(rnode, items, ancestor):
                    return True
        return False

    def addBinding(self, node: ast.AST, value: Binding) -> None:
        """
        Called when a binding is altered.

        - `node` is the statement responsible for the change
        - `value` is the new value, a Binding instance
        """
        # assert value.source in (node, node.parent):
        for scope in self.scopeStack[::-1]:
            if value.name in scope:
                break
        existing = scope.get(value.name)

        if existing and not self.differentForks(node, existing.source):

            parent_stmt = self.getParent(value.source)
            if isinstance(existing, Importation) and isinstance(parent_stmt, ast.For):
                self.report(messages.ImportShadowedByLoopVar,
                            node, value.name, existing.source)

            elif scope is self.scope:
                if (isinstance(parent_stmt, ast.comprehension) and
                        not isinstance(self.getParent(existing.source),
                                       (ast.For, ast.comprehension))):
                    self.report(messages.RedefinedInListComp,
                                node, value.name, existing.source)
                elif not existing.used and value.redefines(existing):
                    self.report(messages.RedefinedWhileUnused,
                                node, value.name, existing.source)

            elif isinstance(existing, Importation) and value.redefines(ex