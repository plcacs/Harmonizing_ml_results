"""
Main module.

Implement the central Checker class.
Also, it models the Bindings and Scopes.
"""
from __future__ import annotations
import ast
import bisect
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import tokenize
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from pyflakes import messages

PY2: bool = sys.version_info < (3, 0)
PY35_PLUS: bool = sys.version_info >= (3, 5)
PY36_PLUS: bool = sys.version_info >= (3, 6)
PY38_PLUS: bool = sys.version_info >= (3, 8)

try:
    sys.pypy_version_info
    PYPY = True
except AttributeError:
    PYPY = False

builtin_vars: List[str] = dir(__import__('__builtin__' if PY2 else 'builtins'))
parse_format_string = string.Formatter().parse
if PY2:

    def tokenize_tokenize(readline: Callable[[], bytes]) -> Iterator[tokenize.TokenInfo]:
        return tokenize.generate_tokens(readline)
else:
    tokenize_tokenize = tokenize.tokenize

if PY2:

    def getNodeType(node_class: type) -> str:
        return str(unicode(node_class.__name__).upper())

    def get_raise_argument(node: ast.Raise) -> Optional[ast.AST]:
        return node.type
else:

    def getNodeType(node_class: type) -> str:
        return node_class.__name__.upper()

    def get_raise_argument(node: ast.Raise) -> Optional[ast.AST]:
        return node.exc

    unicode = str

if PY2:

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

if PY35_PLUS:
    FOR_TYPES: Tuple[type, ...] = (ast.For, ast.AsyncFor)
    LOOP_TYPES: Tuple[type, ...] = (ast.While, ast.For, ast.AsyncFor)
    FUNCTION_TYPES: Tuple[type, ...] = (ast.FunctionDef, ast.AsyncFunctionDef)
else:
    FOR_TYPES = (ast.For,)
    LOOP_TYPES = (ast.While, ast.For)
    FUNCTION_TYPES = (ast.FunctionDef,)

if PY36_PLUS:
    ANNASSIGN_TYPES: Tuple[type, ...] = (ast.AnnAssign,)
else:
    ANNASSIGN_TYPES = ()

if PY38_PLUS:

    def _is_singleton(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, (bool, type(Ellipsis), type(None)))
elif not PY2:

    def _is_singleton(node: ast.AST) -> bool:
        return isinstance(node, (ast.NameConstant, ast.Ellipsis))
else:

    def _is_singleton(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in {'True', 'False', 'Ellipsis', 'None'}


def _is_tuple_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Tuple) and all((_is_constant(elt) for elt in node.elts))


if PY38_PLUS:

    def _is_constant(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) or _is_tuple_constant(node)
else:
    _const_tps: Tuple[type, ...] = (ast.Str, ast.Num)
    if not PY2:
        _const_tps += (ast.Bytes,)

    def _is_constant(node: ast.AST) -> bool:
        return isinstance(node, _const_tps) or _is_singleton(node) or _is_tuple_constant(node)


def _is_const_non_singleton(node: ast.AST) -> bool:
    return _is_constant(node) and (not _is_singleton(node))


def _is_name_or_attr(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (
        isinstance(node, ast.Attribute) and node.attr == name
    )


TYPE_COMMENT_RE: re.Pattern = re.compile('^#\\s*type:\\s*')
ASCII_NON_ALNUM: str = ''.join([chr(i) for i in range(128) if not chr(i).isalnum()])
TYPE_IGNORE_RE: re.Pattern = re.compile(TYPE_COMMENT_RE.pattern + 'ignore([{}]|$)'.format(ASCII_NON_ALNUM))
TYPE_FUNC_RE: re.Pattern = re.compile('^(\\(.*?\\))\\s*->\\s*(.*)$')
MAPPING_KEY_RE: re.Pattern = re.compile('\\(([^()]*)\\)')
CONVERSION_FLAG_RE: re.Pattern = re.compile('[#0+ -]*')
WIDTH_RE: re.Pattern = re.compile('(?:\\*|\\d*)')
PRECISION_RE: re.Pattern = re.compile('(?:\\.(?:\\*|\\d*))?')
LENGTH_RE: re.Pattern = re.compile('[hlL]?')
VALID_CONVERSIONS: Set[str] = frozenset('diouxXeEfFgGcrsa%')


def _must_match(regex: re.Pattern, string: str, pos: int) -> re.Match:
    match = regex.match(string, pos)
    assert match is not None
    return match


def parse_percent_format(s: str) -> Tuple[Tuple[str, Optional[Tuple[Optional[Union[int, str]], Optional[str], Optional[str], str]]], ...]:
    """Parses the string component of a `'...' % ...` format call

    Copied from https://github.com/asottile/pyupgrade at v1.20.1
    """

    def _parse_inner() -> Iterator[Tuple[str, Optional[Tuple[Optional[Union[int, str]], Optional[str], Optional[str], str]]]]:
        string_start = 0
        string_end = 0
        in_fmt = False
        i = 0
        while i < len(s):
            if not in_fmt:
                try:
                    i = s.index('%', i)
                except ValueError:
                    yield (s[string_start:], None)
                    return
                else:
                    string_end = i
                    i += 1
                    in_fmt = True
            else:
                key_match = MAPPING_KEY_RE.match(s, i)
                if key_match:
                    key = key_match.group(1)
                    i = key_match.end()
                else:
                    key = None
                conversion_flag_match = _must_match(CONVERSION_FLAG_RE, s, i)
                conversion_flag = conversion_flag_match.group() or None
                i = conversion_flag_match.end()
                width_match = _must_match(WIDTH_RE, s, i)
                width = width_match.group() or None
                i = width_match.end()
                precision_match = _must_match(PRECISION_RE, s, i)
                precision = precision_match.group() or None
                i = precision_match.end()
                i = _must_match(LENGTH_RE, s, i).end()
                try:
                    conversion = s[i]
                except IndexError:
                    raise ValueError('end-of-string while parsing format')
                i += 1
                fmt = (key, conversion_flag, width, precision, conversion)
                yield (s[string_start:string_end], fmt)
                in_fmt = False
                string_start = i
        if in_fmt:
            raise ValueError('end-of-string while parsing format')

    return tuple(_parse_inner())


class _FieldsOrder(dict):
    """Fix order of AST node fields."""

    def _get_fields(self, node_class: type) -> Tuple[str, ...]:
        fields = node_class._fields
        if 'iter' in fields:
            key_first = 'iter'.find
        elif 'generators' in fields:
            key_first = 'generators'.find
        else:
            key_first = 'value'.find
        return tuple(sorted(fields, key=key_first, reverse=True))

    def __missing__(self, node_class: type) -> Tuple[str, ...]:
        fields = self._get_fields(node_class)
        self[node_class] = fields
        return fields


def counter(items: Iterable[Any]) -> Dict[Any, int]:
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results: Dict[Any, int] = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results


def iter_child_nodes(node: ast.AST, omit: Optional[Union[str, Tuple[str, ...]]] = None, _fields_order: _FieldsOrder = _FieldsOrder()) -> Iterator[ast.AST]:
    """
    Yield all direct child nodes of *node*, that is, all fields that
    are nodes and all items of fields that are lists of nodes.

    :param node:          AST node to be iterated upon
    :param omit:          String or tuple of strings denoting the
                          attributes of the node to be omitted from
                          further parsing
    :param _fields_order: Order of AST node fields
    """
    for name in _fields_order[type(node)]:
        if omit and name in omit:
            continue
        field = getattr(node, name, None)
        if isinstance(field, ast.AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, ast.AST):
                    yield item


def convert_to_value(item: ast.AST) -> Any:
    if isinstance(item, ast.Str):
        return item.s
    elif hasattr(ast, 'Bytes') and isinstance(item, ast.Bytes):
        return item.s
    elif isinstance(item, ast.Tuple):
        return tuple((convert_to_value(i) for i in item.elts))
    elif isinstance(item, ast.Num):
        return item.n
    elif isinstance(item, ast.Name):
        result = VariableKey(item=item)
        constants_lookup = {'True': True, 'False': False, 'None': None}
        return constants_lookup.get(result.name, result)
    elif not PY2 and isinstance(item, ast.NameConstant):
        return item.value
    else:
        return UnhandledKeyType()


def is_notimplemented_name_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and getNodeName(node) == 'NotImplemented'


class Binding:
    """
    Represents the binding of a value to a name.

    The checker uses this to keep track of which names have been bound and
    which names have not. See L{Assignment} for a special type of binding that
    is checked with stricter rules.

    @ivar used: pair of (L{Scope}, node) indicating the scope and
                the node that this binding was last used.
    """

    def __init__(self, name: str, source: Optional[ast.AST]) -> None:
        self.name: str = name
        self.source: Optional[ast.AST] = source
        self.used: Union[bool, Tuple[Scope, ast.AST]] = False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} object {self.name!r} from line {getattr(self.source, "lineno", "unknown")} at {id(self):#x}>'

    def redefines(self, other: Any) -> bool:
        return isinstance(other, Definition) and self.name == other.name


class Definition(Binding):
    """
    A binding that defines a function or a class.
    """


class Builtin(Definition):
    """A definition created for all Python builtins."""

    def __init__(self, name: str) -> None:
        super().__init__(name, None)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} object {self.name!r} at {id(self):#x}>'


class UnhandledKeyType:
    """
    A dictionary key of a type that we cannot or do not check for duplicates.
    """


class VariableKey:
    """
    A dictionary key which is a variable.

    @ivar item: The variable AST object.
    """

    def __init__(self, item: ast.Name) -> None:
        self.name: str = item.id

    def __eq__(self, compare: Any) -> bool:
        return isinstance(compare, VariableKey) and compare.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class Importation(Definition):
    """
    A binding created by an import statement.

    @ivar fullName: The complete name given to the import statement,
        possibly including multiple dotted components.
    @type fullName: C{str}
    """

    def __init__(self, name: str, source: ast.AST, full_name: Optional[str] = None) -> None:
        self.fullName: str = full_name or name
        self.redefined: List[ast.AST] = []
        super().__init__(name, source)

    def redefines(self, other: Any) -> bool:
        if isinstance(other, SubmoduleImportation):
            return self.fullName == other.fullName
        return isinstance(other, Definition) and self.name == other.name

    def _has_alias(self) -> bool:
        """Return whether importation needs an as clause."""
        return not self.fullName.split('.')[-1] == self.name

    @property
    def source_statement(self) -> str:
        """Generate a source statement equivalent to the import."""
        if self._has_alias():
            return f'import {self.fullName} as {self.name}'
        else:
            return f'import {self.fullName}'

    def __str__(self) -> str:
        """Return import full name with alias."""
        if self._has_alias():
            return f'{self.fullName} as {self.name}'
        else:
            return self.fullName


class SubmoduleImportation(Importation):
    """
    A binding created by a submodule import statement.

    A submodule import is a special case where the root module is implicitly
    imported, without an 'as' clause, and the submodule is also imported.
    Python does not restrict which attributes of the root module may be used.

    This class is only used when the submodule import is without an 'as' clause.

    pyflakes handles this case by registering the root module name in the scope,
    allowing any attribute of the root module to be accessed.

    RedefinedWhileUnused is suppressed in `redefines` unless the submodule
    name is also the same, to avoid false positives.
    """

    def __init__(self, name: str, source: ast.AST) -> None:
        assert '.' in name and (not source or isinstance(source, ast.Import))
        package_name = name.split('.')[0]
        super().__init__(package_name, source)
        self.fullName: str = name

    def redefines(self, other: Any) -> bool:
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        return super().redefines(other)

    def __str__(self) -> str:
        return self.fullName

    @property
    def source_statement(self) -> str:
        return 'import ' + self.fullName


class ImportationFrom(Importation):

    def __init__(self, name: str, source: ast.AST, module: str, real_name: Optional[str] = None) -> None:
        self.module: str = module
        self.real_name: str = real_name or name
        if module.endswith('.'):
            full_name = module + self.real_name
        else:
            full_name = module + '.' + self.real_name
        super().__init__(name, source, full_name)

    def __str__(self) -> str:
        """Return import full name with alias."""
        if self.real_name != self.name:
            return f'{self.fullName} as {self.name}'
        else:
            return self.fullName

    @property
    def source_statement(self) -> str:
        if self.real_name != self.name:
            return f'from {self.module} import {self.real_name} as {self.name}'
        else:
            return f'from {self.module} import {self.name}'


class StarImportation(Importation):
    """A binding created by a 'from x import *' statement."""

    def __init__(self, name: str, source: ast.AST) -> None:
        super().__init__('*', source)
        self.name: str = name + '.*'
        self.fullName: str = name

    @property
    def source_statement(self) -> str:
        return 'from ' + self.fullName + ' import *'

    def __str__(self) -> str:
        if self.fullName.endswith('.'):
            return self.source_statement
        else:
            return self.name


class FutureImportation(ImportationFrom):
    """
    A binding created by a from `__future__` import statement.

    `__future__` imports are implicitly used.
    """

    def __init__(self, name: str, source: ast.AST, scope: Scope) -> None:
        super().__init__(name, source, '__future__')
        self.used: Union[bool, Tuple[Scope, ast.AST]] = (scope, source)


class Argument(Binding):
    """
    Represents binding a name as an argument.
    """

    def __init__(self, name: str, source: Optional[ast.AST]) -> None:
        super().__init__(name, source)


class Assignment(Binding):
    """
    Represents binding a name with an explicit assignment.

    The checker will raise warnings for any Assignment that isn't used. Also,
    the checker does not consider assignments in tuple/list unpacking to be
    Assignments, rather it treats them as simple Bindings.
    """

    def __init__(self, name: str, source: ast.AST) -> None:
        super().__init__(name, source)


class Annotation(Binding):
    """
    Represents binding a name to a type without an associated value.

    As long as this name is not assigned a value in another binding, it is considered
    undefined for most purposes. One notable exception is using the name as a type
    annotation.
    """

    def redefines(self, other: Any) -> bool:
        """An Annotation doesn't define any name, so it cannot redefine one."""
        return False


class FunctionDefinition(Definition):
    pass


class ClassDefinition(Definition):
    pass


class ExportBinding(Binding):
    """
    A binding created by an C{__all__} assignment.  If the names in the list
    can be determined statically, they will be treated as names for export and
    additional checking applied to them.

    The only recognized C{__all__} assignment via list/tuple concatenation is in the
    following format:

        __all__ = ['a'] + ['b'] + ['c']

    Names which are imported and not otherwise used but appear in the value of
    C{__all__} will not have an unused import warning reported for them.
    """

    def __init__(self, name: str, source: ast.AST, scope: Scope) -> None:
        if '__all__' in scope and isinstance(source, ast.AugAssign):
            self.names: List[str] = list(scope['__all__'].names)
        else:
            self.names = []

        def _add_to_names(container: ast.AST) -> None:
            for node in container.elts:
                if isinstance(node, ast.Str):
                    self.names.append(node.s)

        if isinstance(source.value, (ast.List, ast.Tuple)):
            _add_to_names(source.value)
        elif isinstance(source.value, ast.BinOp):
            currentValue = source.value
            while isinstance(currentValue.right, (ast.List, ast.Tuple)):
                left = currentValue.left
                right = currentValue.right
                _add_to_names(right)
                if isinstance(left, ast.BinOp):
                    currentValue = left
                elif isinstance(left, (ast.List, ast.Tuple)):
                    _add_to_names(left)
                    break
                else:
                    break
        super().__init__(name, source)

    def __init__(self, name: str, source: ast.AugAssign, scope: Scope) -> None:
        super().__init__(name, source)
        # Additional initialization logic if needed


class Scope(dict):
    importStarred: bool = False

    def __repr__(self) -> str:
        scope_cls = self.__class__.__name__
        return f'<{scope_cls} at {id(self):#x} {dict.__repr__(self)}>'


class ClassScope(Scope):
    pass


class FunctionScope(Scope):
    """
    I represent a name scope for a function.

    @ivar globals: Names declared 'global' in this function.
    """
    usesLocals: bool = False
    alwaysUsed: Set[str] = {'__tracebackhide__', '__traceback_info__', '__traceback_supplement__'}

    def __init__(self) -> None:
        super().__init__()
        self.globals: Set[str] = self.alwaysUsed.copy()
        self.returnValue: Optional[ast.AST] = None
        self.isGenerator: bool = False

    def unusedAssignments(self) -> Iterator[Tuple[str, Assignment]]:
        """
        Return a generator for the assignments which have not been used.
        """
        for name, binding in self.items():
            if (
                not binding.used
                and name != '_'
                and (name not in self.globals)
                and (not self.usesLocals)
                and isinstance(binding, Assignment)
            ):
                yield (name, binding)


class GeneratorScope(Scope):
    pass


class ModuleScope(Scope):
    """Scope for a module."""
    _futures_allowed: bool = True
    _annotations_future_enabled: bool = False


class DoctestScope(ModuleScope):
    """Scope for a doctest."""
    pass


class DummyNode:
    """Used in place of an `ast.AST` to set error message positions"""

    def __init__(self, lineno: int, col_offset: int) -> None:
        self.lineno: int = lineno
        self.col_offset: int = col_offset


class DetectClassScopedMagic:
    names: List[str] = dir()


_MAGIC_GLOBALS: List[str] = ['__file__', '__builtins__', 'WindowsError']
if PY36_PLUS:
    _MAGIC_GLOBALS.append('__annotations__')


def getNodeName(node: ast.AST) -> Optional[str]:
    if hasattr(node, 'id'):
        return node.id
    if hasattr(node, 'name'):
        return node.name
    if hasattr(node, 'rest'):
        return node.rest
    return None


TYPING_MODULES: Set[str] = frozenset(('typing', 'typing_extensions'))


def _is_typing_helper(
    node: ast.AST,
    is_name_match_fn: Callable[[str], bool],
    scope_stack: List[Scope],
) -> bool:
    """
    Internal helper to determine whether or not something is a member of a
    typing module. This is used as part of working out whether we are within a
    type annotation context.

    Note: you probably don't want to use this function directly. Instead see the
    utils below which wrap it (`_is_typing` and `_is_any_typing_member`).
    """

    def _bare_name_is_attr(name: str) -> bool:
        for scope in reversed(scope_stack):
            if name in scope:
                binding = scope[name]
                if isinstance(binding, ImportationFrom) and binding.module in TYPING_MODULES and is_name_match_fn(binding.real_name):
                    return True
        return False

    def _module_scope_is_typing(name: str) -> bool:
        for scope in reversed(scope_stack):
            if name in scope:
                binding = scope[name]
                if isinstance(binding, Importation) and binding.fullName in TYPING_MODULES:
                    return True
        return False

    return (
        (isinstance(node, ast.Name) and _bare_name_is_attr(node.id))
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and _module_scope_is_typing(node.value.id)
            and is_name_match_fn(node.attr)
        )
    )


def _is_typing(
    node: ast.AST, typing_attr: str, scope_stack: List[Scope]
) -> bool:
    """
    Determine whether `node` represents the member of a typing module specified
    by `typing_attr`.

    This is used as part of working out whether we are within a type annotation
    context.
    """
    return _is_typing_helper(node, lambda x: x == typing_attr, scope_stack)


def _is_any_typing_member(node: ast.AST, scope_stack: List[Scope]) -> bool:
    """
    Determine whether `node` represents any member of a typing module.

    This is used as part of working out whether we are within a type annotation
    context.
    """
    return _is_typing_helper(node, lambda x: True, scope_stack)


def is_typing_overload(value: Binding, scope_stack: List[Scope]) -> bool:
    if isinstance(value.source, FUNCTION_TYPES):
        return any(
            (_is_typing(dec, 'overload', scope_stack) for dec in value.source.decorator_list)
        )
    return False


class AnnotationState:
    NONE: int = 0
    STRING: int = 1
    BARE: int = 2


def in_annotation(func: Callable) -> Callable:
    @functools.wraps(func)
    def in_annotation_func(self: Checker, *args: Any, **kwargs: Any) -> Any:
        with self._enter_annotation():
            return func(self, *args, **kwargs)
    return in_annotation_func


def in_string_annotation(func: Callable) -> Callable:
    @functools.wraps(func)
    def in_annotation_func(self: Checker, *args: Any, **kwargs: Any) -> Any:
        with self._enter_annotation(AnnotationState.STRING):
            return func(self, *args, **kwargs)
    return in_annotation_func


def make_tokens(code: Union[str, bytes]) -> Tuple[tokenize.TokenInfo, ...]:
    if not isinstance(code, bytes):
        code = code.encode('UTF-8')
    lines = iter(code.splitlines(True))
    return tuple(tokenize_tokenize(lambda: next(lines, b'')))


class _TypeableVisitor(ast.NodeVisitor):
    """Collect the line number and nodes which are deemed typeable by
    PEP 484

    https://www.python.org/dev/peps/pep-0484/#type-comments
    """

    def __init__(self) -> None:
        self.typeable_lines: List[int] = []
        self.typeable_nodes: Dict[int, ast.AST] = {}

    def _typeable(self, node: ast.AST) -> None:
        self.typeable_lines.append(node.lineno)
        self.typeable_nodes[node.lineno] = node
        self.generic_visit(node)

    visit_Assign = visit_For = visit_FunctionDef = visit_With = _typeable
    if PY35_PLUS:
        visit_AsyncFor = visit_AsyncFunctionDef = visit_AsyncWith = _typeable


def _collect_type_comments(tree: ast.AST, tokens: Tuple[tokenize.TokenInfo, ...]) -> Dict[ast.AST, List[Tuple[Tuple[int, int], str]]]:
    visitor = _TypeableVisitor()
    visitor.visit(tree)
    type_comments: Dict[ast.AST, List[Tuple[Tuple[int, int], str]]] = collections.defaultdict(list)
    for tp, text, start, _, _ in tokens:
        if tp != tokenize.COMMENT or not TYPE_COMMENT_RE.match(text) or TYPE_IGNORE_RE.match(text):
            continue
        lineno, _ = start
        idx = bisect.bisect_right(visitor.typeable_lines, lineno)
        if idx == 0:
            continue
        node = visitor.typeable_nodes[visitor.typeable_lines[idx - 1]]
        type_comments[node].append(((lineno, start[1]), text))
    return type_comments


class Checker:
    """
    I check the cleanliness and sanity of Python code.

    @ivar _deferredFunctions: Tracking list used by L{deferFunction}.  Elements
        of the list are two-tuples.  The first element is the callable passed
        to L{deferFunction}.  The second element is a copy of the scope stack
        at the time L{deferFunction} was called.

    @ivar _deferredAssignments: Similar to C{_deferredFunctions}, but for
        callables which are deferred assignment checks.
    """
    _ast_node_scope: Dict[type, type] = {
        ast.Module: ModuleScope,
        ast.ClassDef: ClassScope,
        ast.FunctionDef: FunctionScope,
        ast.Lambda: FunctionScope,
        ast.ListComp: GeneratorScope,
        ast.SetComp: GeneratorScope,
        ast.GeneratorExp: GeneratorScope,
        ast.DictComp: GeneratorScope,
    }
    if PY35_PLUS:
        _ast_node_scope[ast.AsyncFunctionDef] = FunctionScope
    nodeDepth: int = 0
    offset: Optional[Tuple[int, int]] = None
    _in_annotation: int = AnnotationState.NONE
    _in_deferred: bool = False
    builtIns: Set[str] = set(builtin_vars).union(_MAGIC_GLOBALS)
    _customBuiltIns: Optional[str] = os.environ.get('PYFLAKES_BUILTINS')
    if _customBuiltIns:
        builtIns.update(_customBuiltIns.split(','))
    del _customBuiltIns

    def __init__(
        self,
        tree: ast.AST,
        filename: str = '(none)',
        builtins: Optional[Iterable[str]] = None,
        withDoctest: bool = 'PYFLAKES_DOCTEST' in os.environ,
        file_tokens: Tuple[tokenize.TokenInfo, ...] = (),
    ) -> None:
        self._nodeHandlers: Dict[type, Callable[[ast.AST], None]] = {}
        self._deferredFunctions: List[Tuple[Callable, List[Scope], Optional[Tuple[int, int]]]] = []
        self._deferredAssignments: List[Tuple[Callable, List[Scope], Optional[Tuple[int, int]]]] = []
        self.deadScopes: List[Scope] = []
        self.messages: List[messages.Message] = []
        self.filename: str = filename
        if builtins:
            self.builtIns = self.builtIns.union(builtins)
        self.withDoctest: bool = withDoctest
        try:
            self.scopeStack: List[Scope] = [Checker._ast_node_scope[type(tree)]()]
        except KeyError:
            raise RuntimeError(f'No scope implemented for the node {tree!r}')
        self.exceptHandlers: List[Tuple[str, ...]] = [()]
        self.root: ast.AST = tree
        self._type_comments: Dict[ast.AST, List[Tuple[Tuple[int, int], str]]] = _collect_type_comments(tree, file_tokens)
        for builtin in self.builtIns:
            self.addBinding(None, Builtin(builtin))
        self.handleChildren(tree)
        self._in_deferred = True
        self.runDeferred(self._deferredFunctions)
        self._deferredFunctions = []
        self.runDeferred(self._deferredAssignments)
        self._deferredAssignments = []
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

    def runDeferred(self, deferred: List[Tuple[Callable, List[Scope], Optional[Tuple[int, int]]]]) -> None:
        """
        Run the callables in C{deferred} using their associated scope stack.
        """
        for handler, scope, offset in deferred:
            self.scopeStack = scope
            self.offset = offset
            handler()

    def _in_doctest(self) -> bool:
        return len(self.scopeStack) >= 2 and isinstance(self.scopeStack[1], DoctestScope)

    @property
    def futuresAllowed(self) -> bool:
        if not all(isinstance(scope, ModuleScope) for scope in self.scopeStack):
            return False
        return self.scope._futures_allowed

    @futuresAllowed.setter
    def futuresAllowed(self, value: bool) -> None:
        assert value is False
        if isinstance(self.scope, ModuleScope):
            self.scope._futures_allowed = False

    @property
    def annotationsFutureEnabled(self) -> bool:
        scope = self.scopeStack[0]
        if not isinstance(scope, ModuleScope):
            return False
        return scope._annotations_future_enabled

    @annotationsFutureEnabled.setter
    def annotationsFutureEnabled(self, value: bool) -> None:
        assert value is True
        assert isinstance(self.scope, ModuleScope)
        self.scope._annotations_future_enabled = True

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
            if isinstance(scope, ClassScope):
                continue
            all_binding: Optional[ExportBinding] = scope.get('__all__')
            if all_binding and not isinstance(all_binding, ExportBinding):
                all_binding = None
            if all_binding:
                all_names = set(all_binding.names)
                undefined = [name for name in all_binding.names if name not in scope]
            else:
                all_names = undefined = []
            if undefined:
                if not scope.importStarred and os.path.basename(self.filename) != '__init__.py':
                    for name in undefined:
                        self.report(messages.UndefinedExport, scope['__all__'].source, name)
                if scope.importStarred:
                    from_list: List[str] = []
                    for binding in scope.values():
                        if isinstance(binding, StarImportation):
                            binding.used = all_binding
                            from_list.append(binding.fullName)
                    from_list = ', '.join(sorted(from_list))
                    for name in undefined:
                        self.report(messages.ImportStarUsage, scope['__all__'].source, name, from_list)
            for value in scope.values():
                if isinstance(value, Importation):
                    used: bool = value.used or value.name in all_names
                    if not used:
                        messg = messages.UnusedImport
                        self.report(messg, value.source, str(value))
                    for node in value.redefined:
                        if isinstance(self.getParent(node), FOR_TYPES):
                            messg = messages.ImportShadowedByLoopVar
                        elif used:
                            continue
                        else:
                            messg = messages.RedefinedWhileUnused
                        self.report(messg, node, value.name, value.source)

    def pushScope(self, scopeClass: type = FunctionScope) -> None:
        self.scopeStack.append(scopeClass())

    def report(self, messageClass: type, *args: Any, **kwargs: Any) -> None:
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def getParent(self, node: ast.AST) -> ast.AST:
        while True:
            node = node._pyflakes_parent
            if not hasattr(node, 'elts') and not hasattr(node, 'ctx'):
                return node

    def getCommonAncestor(self, lnode: ast.AST, rnode: ast.AST, stop: ast.AST) -> Optional[ast.AST]:
        if stop in (lnode, rnode) or not (hasattr(lnode, '_pyflakes_parent') and hasattr(rnode, '_pyflakes_parent')):
            return None
        if lnode is rnode:
            return lnode
        if lnode._pyflakes_depth > rnode._pyflakes_depth:
            return self.getCommonAncestor(lnode._pyflakes_parent, rnode, stop)
        if lnode._pyflakes_depth < rnode._pyflakes_depth:
            return self.getCommonAncestor(lnode, rnode._pyflakes_parent, stop)
        return self.getCommonAncestor(lnode._pyflakes_parent, rnode._pyflakes_parent, stop)

    def descendantOf(self, node: ast.AST, ancestors: Iterable[ast.AST], stop: ast.AST) -> bool:
        for a in ancestors:
            if self.getCommonAncestor(node, a, stop):
                return True
        return False

    def _getAncestor(self, node: ast.AST, ancestor_type: Union[type, Tuple[type, ...]]) -> Optional[ast.AST]:
        parent = node
        while True:
            if parent is self.root:
                return None
            parent = self.getParent(parent)
            if isinstance(parent, ancestor_type):
                return parent

    def getScopeNode(self, node: ast.AST) -> Optional[ast.AST]:
        return self._getAncestor(node, tuple(Checker._ast_node_scope.keys()))

    def differentForks(self, lnode: ast.AST, rnode: ast.AST) -> bool:
        """True, if lnode and rnode are located on different forks of IF/TRY"""
        ancestor = self.getCommonAncestor(lnode, rnode, self.root)
        if not ancestor:
            return False
        parts = getAlternatives(ancestor)
        if parts:
            for items in parts:
                if self.descendantOf(lnode, items, ancestor) ^ self.descendantOf(rnode, items, ancestor):
                    return True
        return False

    def addBinding(self, node: Optional[ast.AST], value: Binding) -> None:
        """
        Called when a binding is altered.

        - `node` is the statement responsible for the change
        - `value` is the new value, a Binding instance
        """
        for scope in self.scopeStack[::-1]:
            if value.name in scope:
                break
        existing = scope.get(value.name)
        if existing and (not isinstance(existing, Builtin)) and (not self.differentForks(node, existing.source)):
            parent_stmt = self.getParent(existing.source) if existing.source else None
            if isinstance(existing, Importation) and isinstance(parent_stmt, FOR_TYPES):
                self.report(messages.ImportShadowedByLoopVar, node, value.name, existing.source)
            elif scope is self.scope:
                if isinstance(existing.source, ast.comprehension) and not isinstance(self.getParent(existing.source), (FOR_TYPES, ast.comprehension)):
                    self.report(messages.RedefinedInListComp, node, value.name, existing.source)
                elif not existing.used and value.redefines(existing):
                    if value.name != '_' or isinstance(existing, Importation):
                        if not is_typing_overload(existing, self.scopeStack):
                            self.report(messages.RedefinedWhileUnused, node, value.name, existing.source)
            elif isinstance(existing, Importation) and value.redefines(existing):
                existing.redefined.append(node)
        if value.name in self.scope:
            value.used = self.scope[value.name].used
        if value.name not in self.scope or not isinstance(value, Annotation):
            self.scope[value.name] = value

    def _unknown_handler(self, node: ast.AST) -> None:
        if os.environ.get('PYFLAKES_ERROR_UNKNOWN'):
            raise NotImplementedError(f'Unexpected type: {type(node)}')
        else:
            self.handleChildren(node)

    def getNodeHandler(self, node_class: type) -> Callable[[ast.AST], None]:
        try:
            return self._nodeHandlers[node_class]
        except KeyError:
            nodeType = getNodeType(node_class)
        handler = getattr(self, nodeType, self._unknown_handler)
        self._nodeHandlers[node_class] = handler
        return handler

    def handleNodeLoad(self, node: ast.AST) -> None:
        name = getNodeName(node)
        if not name:
            return
        in_generators: Optional[bool] = None
        importStarred: Optional[bool] = None
        for scope in self.scopeStack[-1::-1]:
            if isinstance(scope, ClassScope):
                if not PY2 and name == '__class__':
                    return
                elif in_generators is False:
                    continue
            binding = scope.get(name, None)
            if isinstance(binding, Annotation) and (not self._in_postponed_annotation):
                continue
            if name == 'print' and isinstance(binding, Builtin):
                parent = self.getParent(node)
                if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.RShift):
                    self.report(messages.InvalidPrintSyntax, node)
            if binding:
                binding.used = (self.scope, node)
                if isinstance(binding, Importation) and binding._has_alias():
                    try:
                        scope[binding.fullName].used = (self.scope, node)
                    except KeyError:
                        pass
            else:
                pass
            importStarred = importStarred or scope.importStarred
            if in_generators is not False:
                in_generators = isinstance(scope, GeneratorScope)
        if importStarred:
            from_list: List[str] = []
            for scope in self.scopeStack[-1::-1]:
                for binding in scope.values():
                    if isinstance(binding, StarImportation):
                        binding.used = all_binding
                        from_list.append(binding.fullName)
            from_list = ', '.join(sorted(from_list))
            self.report(messages.ImportStarUsage, node, name, from_list)
            return
        if name == '__path__' and os.path.basename(self.filename) == '__init__.py':
            return
        if name in DetectClassScopedMagic.names and isinstance(self.scope, ClassScope):
            return
        if 'NameError' not in self.exceptHandlers[-1]:
            self.report(messages.UndefinedName, node, name)

    def handleNodeStore(self, node: ast.AST) -> None:
        name = getNodeName(node)
        if not name:
            return
        if isinstance(self.scope, FunctionScope) and name not in self.scope:
            for scope in self.scopeStack[:-1]:
                if not isinstance(scope, (FunctionScope, ModuleScope)):
                    continue
                used = name in scope and scope[name].used
                if used and used[0] is self.scope and (name not in self.scope.globals):
                    self.report(messages.UndefinedLocal, scope[name].used[1], name, scope[name].source)
                    break
        parent_stmt = self.getParent(node)
        if isinstance(parent_stmt, ANNASSIGN_TYPES) and parent_stmt.value is None:
            binding = Annotation(name, node)
        elif isinstance(parent_stmt, (FOR_TYPES, ast.comprehension)) or (parent_stmt != node._pyflakes_parent and (not self.isLiteralTupleUnpacking(parent_stmt))):
            binding = Binding(name, node)
        elif name == '__all__' and isinstance(self.scope, ModuleScope):
            binding = ExportBinding(name, parent_stmt, self.scope)
        elif PY2 and isinstance(getattr(node, 'ctx', None), ast.Param):
            binding = Argument(name, self.getScopeNode(node))
        else:
            binding = Assignment(name, node)
        self.addBinding(node, binding)

    def handleNodeDelete(self, node: ast.AST) -> None:
        def on_conditional_branch() -> bool:
            """
            Return `True` if node is part of a conditional body.
            """
            current = getattr(node, '_pyflakes_parent', None)
            while current:
                if isinstance(current, (ast.If, ast.While, ast.IfExp)):
                    return True
                current = getattr(current, '_pyflakes_parent', None)
            return False

        name = getNodeName(node)
        if not name:
            return
        if on_conditional_branch():
            return
        if isinstance(self.scope, FunctionScope) and name in self.scope.globals:
            self.scope.globals.remove(name)
        else:
            try:
                del self.scope[name]
            except KeyError:
                self.report(messages.UndefinedName, node, name)

    @contextlib.contextmanager
    def _enter_annotation(self, ann_type: int = AnnotationState.BARE) -> Iterator[None]:
        orig, self._in_annotation = self._in_annotation, ann_type
        try:
            yield
        finally:
            self._in_annotation = orig

    @property
    def _in_postponed_annotation(self) -> bool:
        return self._in_annotation == AnnotationState.STRING or self.annotationsFutureEnabled

    def _handle_type_comments(self, node: ast.AST) -> None:
        for (lineno, col_offset), comment in self._type_comments.get(node, []):
            comment = comment.split(':', 1)[1].strip()
            func_match = TYPE_FUNC_RE.match(comment)
            if func_match:
                parts = (func_match.group(1).replace('*', ''), func_match.group(2).strip())
            else:
                parts = (comment,)
            for part in parts:
                if PY2:
                    part = part.replace('...', 'Ellipsis')
                self.deferFunction(
                    functools.partial(
                        self.handleStringAnnotation,
                        part,
                        DummyNode(lineno, col_offset),
                        lineno,
                        col_offset,
                        messages.CommentAnnotationSyntaxError,
                    )
                )

    def handleChildren(self, tree: ast.AST, omit: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        self._handle_type_comments(tree)
        for node in iter_child_nodes(tree, omit=omit):
            self.handleNode(node, tree)

    def isLiteralTupleUnpacking(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Assign):
            for child in node.targets + [node.value]:
                if not hasattr(child, 'elts'):
                    return False
            return True
        return False

    def isDocstring(self, node: ast.AST) -> bool:
        """
        Determine if the given node is a docstring, as long as it is at the
        correct place in the node tree.
        """
        return isinstance(node, ast.Str) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Str))

    def getDocstring(self, node: ast.AST) -> Tuple[Optional[str], Optional[int]]:
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Str):
            return (None, None)
        if PYPY or PY38_PLUS:
            doctest_lineno = node.lineno - 1
        else:
            doctest_lineno = node.lineno - node.s.count('\n') - 1
        return (node.s, doctest_lineno)

    def handleNode(self, node: Optional[ast.AST], parent: ast.AST) -> None:
        if node is None:
            return
        if self.offset and hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            node.lineno += self.offset[0]
            node.col_offset += self.offset[1]
        if self.futuresAllowed and not (isinstance(node, ast.ImportFrom) or self.isDocstring(node)):
            self.futuresAllowed = False
        self.nodeDepth += 1
        node._pyflakes_depth: int = self.nodeDepth
        node._pyflakes_parent: ast.AST = parent
        try:
            handler = self.getNodeHandler(type(node))
            handler(node)
        finally:
            self.nodeDepth -= 1

    _getDoctestExamples = doctest.DocTestParser().get_examples

    def handleDoctests(self, node: ast.AST) -> None:
        try:
            if hasattr(node, 'docstring'):
                docstring = node.docstring
                node_lineno = node.lineno
                if hasattr(node, 'args'):
                    node_lineno = max([node_lineno] + [arg.lineno for arg in node.args.args])
            else:
                docstring, node_lineno = self.getDocstring(node.body[0])
            examples = docstring and self._getDoctestExamples(docstring)
        except (ValueError, IndexError):
            return
        if not examples:
            return
        saved_stack = self.scopeStack
        self.scopeStack = [self.scopeStack[0]]
        node_offset: Tuple[int, int] = self.offset or (0, 0)
        self.pushScope(DoctestScope)
        if '_' not in self.scopeStack[0]:
            self.addBinding(None, Builtin('_'))
        for example in examples:
            try:
                tree = ast.parse(example.source, '<doctest>')
            except SyntaxError as e:
                if PYPY:
                    e.offset += 1
                position = (node_lineno + example.lineno + (e.lineno or 0), example.indent + 4 + (e.offset or 0))
                self.report(messages.DoctestSyntaxError, node, position)
            else:
                self.offset = (
                    node_offset[0] + node_lineno + example.lineno,
                    node_offset[1] + example.indent + 4,
                )
                self.handleChildren(tree)
                self.offset = node_offset
        self.popScope()
        self.scopeStack = saved_stack

    @in_string_annotation
    def handleStringAnnotation(
        self,
        s: str,
        node: ast.AST,
        ref_lineno: int,
        ref_col_offset: int,
        err: type,
    ) -> None:
        try:
            tree = ast.parse(s)
        except SyntaxError:
            self.report(err, node, s)
            return
        body = tree.body
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            self.report(err, node, s)
            return
        parsed_annotation = tree.body[0].value
        for descendant in ast.walk(parsed_annotation):
            if 'lineno' in descendant._attributes and 'col_offset' in descendant._attributes:
                descendant.lineno = ref_lineno
                descendant.col_offset = ref_col_offset
        self.handleNode(parsed_annotation, node)

    @in_annotation
    def handleAnnotation(self, annotation: Optional[ast.AST], node: ast.AST) -> None:
        if isinstance(annotation, ast.Str):
            self.deferFunction(
                functools.partial(
                    self.handleStringAnnotation,
                    annotation.s,
                    node,
                    annotation.lineno,
                    annotation.col_offset,
                    messages.ForwardAnnotationSyntaxError,
                )
            )
        elif self.annotationsFutureEnabled:
            fn = in_annotation(self.handleNode)
            self.deferFunction(lambda: fn(self, annotation, node))
        else:
            self.handleNode(annotation, node)

    def ignore(self, node: ast.AST) -> None:
        pass

    DELETE = PRINT = FOR = ASYNCFOR = WHILE = WITH = WITHITEM = ASYNCWITH = ASYNCWITHITEM = TRYFINALLY = EXEC = EXPR = ASSIGN = handleChildren
    PASS = ignore
    BOOLOP = UNARYOP = SET = REPR = ATTRIBUTE = STARRED = NAMECONSTANT = NAMEDEXPR = handleChildren

    def SUBSCRIPT(self, node: ast.Subscript) -> None:
        if _is_name_or_attr(node.value, 'Literal'):
            with self._enter_annotation(AnnotationState.NONE):
                self.handleChildren(node)
        elif _is_name_or_attr(node.value, 'Annotated'):
            self.handleNode(node.value, node)
            if isinstance(node.slice, ast.Tuple):
                slice_tuple = node.slice
            elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Tuple):
                slice_tuple = node.slice.value
            else:
                slice_tuple = None
            if slice_tuple is None or len(slice_tuple.elts) < 2:
                self.handleNode(node.slice, node)
            else:
                self.handleNode(slice_tuple.elts[0], node)
                with self._enter_annotation(AnnotationState.NONE):
                    for arg in slice_tuple.elts[1:]:
                        self.handleNode(arg, node)
            self.handleNode(node.ctx, node)
        elif _is_any_typing_member(node.value, self.scopeStack):
            with self._enter_annotation():
                self.handleChildren(node)
        else:
            self.handleChildren(node)

    def _handle_string_dot_format(self, node: ast.Call) -> None:
        try:
            placeholders: Tuple[Tuple[str, Optional[Tuple[Optional[Union[int, str]], Optional[str], Optional[str], str]]], ...] = tuple(
                parse_format_string(node.func.value.s)
            )
        except ValueError as e:
            self.report(messages.StringDotFormatInvalidFormat, node, e)
            return

        class state:
            auto: Optional[int] = None
            next_auto: int = 0

        placeholder_positional: Set[Union[int, str]] = set()
        placeholder_named: Set[str] = set()

        def _add_key(fmtkey: Optional[str]) -> bool:
            """Returns True if there is an error which should early-exit"""
            if fmtkey is None:
                return False
            fmtkey, _, _ = fmtkey.partition('.')
            fmtkey, _, _ = fmtkey.partition('[')
            try:
                fmtkey_int = int(fmtkey)
                fmtkey = fmtkey_int
            except ValueError:
                pass
            else:
                if state.auto is True:
                    self.report(messages.StringDotFormatMixingAutomatic, node)
                    return True
                else:
                    state.auto = False
            if fmtkey == '':
                if state.auto is False:
                    self.report(messages.StringDotFormatMixingAutomatic, node)
                    return True
                else:
                    state.auto = True
                fmtkey = state.next_auto
                state.next_auto += 1
            if isinstance(fmtkey, int):
                placeholder_positional.add(fmtkey)
            else:
                placeholder_named.add(fmtkey)
            return False

        for _, fmtkey, spec, _ in placeholders:
            if _add_key(fmtkey):
                return
            if spec is not None:
                try:
                    spec_placeholders = tuple(parse_format_string(spec))
                except ValueError as e:
                    self.report(messages.StringDotFormatInvalidFormat, node, e)
                    return
                for _, spec_fmtkey, spec_spec, _ in spec_placeholders:
                    if spec_spec is not None and '{' in spec_spec:
                        self.report(messages.StringDotFormatInvalidFormat, node, 'Max string recursion exceeded')
                        return
                    if _add_key(spec_fmtkey):
                        return
        if getattr(node, 'starargs', None) or getattr(node, 'kwargs', None) or any(
            (isinstance(arg, getattr(ast, 'Starred', ())) for arg in node.args)
        ) or any((kwd.arg is None for kwd in node.keywords)):
            return
        substitution_positional: Set[int] = set(range(len(node.args)))
        substitution_named: Set[str] = {kwd.arg for kwd in node.keywords if kwd.arg}
        extra_positional = substitution_positional - placeholder_positional
        extra_named = substitution_named - placeholder_named
        missing_arguments_positional = placeholder_positional - substitution_positional
        missing_arguments_named = placeholder_named - substitution_named
        missing_arguments = missing_arguments_positional.union(missing_arguments_named)
        if extra_positional:
            self.report(messages.StringDotFormatExtraPositionalArguments, node, ', '.join(sorted((str(x) for x in extra_positional))))
        if extra_named:
            self.report(messages.StringDotFormatExtraNamedArguments, node, ', '.join(sorted(extra_named)))
        if missing_arguments:
            self.report(messages.StringDotFormatMissingArgument, node, ', '.join(sorted((str(x) for x in missing_arguments))))

    def CALL(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Str)
            and node.func.attr == 'format'
        ):
            self._handle_string_dot_format(node)
        omit: List[str] = []
        annotated: List[ast.AST] = []
        not_annotated: List[Tuple[ast.keyword, Optional[List[str]]]] = []
        if _is_typing(node.func, 'cast', self.scopeStack) and len(node.args) >= 1:
            with self._enter_annotation():
                self.handleNode(node.args[0], node)
        elif _is_typing(node.func, 'TypeVar', self.scopeStack):
            omit += ['args']
            annotated += list(node.args[1:])
            omit += ['keywords']
            annotated += [k.value for k in node.keywords if k.arg == 'bound']
            not_annotated += [(k, ['value']) for k in node.keywords]
        elif _is_typing(node.func, 'TypedDict', self.scopeStack):
            if len(node.args) > 1 and isinstance(node.args[1], ast.Dict):
                omit += ['args']
                annotated += node.args[1].values
                not_annotated += [(arg, ['values'] if i == 1 else None) for i, arg in enumerate(node.args)]
            omit += ['keywords']
            annotated += [k.value for k in node.keywords]
            not_annotated += [(k, ['value']) for k in node.keywords]
        elif _is_typing(node.func, 'NamedTuple', self.scopeStack):
            if (
                len(node.args) > 1
                and isinstance(node.args[1], (ast.Tuple, ast.List))
                and all(isinstance(x, (ast.Tuple, ast.List)) and len(x.elts) == 2 for x in node.args[1].elts)
            ):
                omit += ['args']
                annotated += [elt.elts[1] for elt in node.args[1].elts]
                not_annotated += [(elt.elts[0], None) for elt in node.args[1].elts]
                not_annotated += [(arg, ['elts'] if i == 1 else None) for i, arg in enumerate(node.args)]
                not_annotated += [(elt, 'elts') for elt in node.args[1].elts]
            omit += ['keywords']
            annotated += [k.value for k in node.keywords]
            not_annotated += [(k, ['value']) for k in node.keywords]
        if omit:
            with self._enter_annotation(AnnotationState.NONE):
                for na_node, na_omit in not_annotated:
                    self.handleChildren(na_node, omit=na_omit)
                self.handleChildren(node, omit=omit)
            with self._enter_annotation():
                for annotated_node in annotated:
                    self.handleNode(annotated_node, node)
        else:
            self.handleChildren(node)

    def _handle_percent_format(self, node: ast.BinOp) -> None:
        try:
            placeholders: Tuple[Tuple[str, Optional[Tuple[Optional[Union[int, str]], Optional[str], Optional[str], str]]], ...] = parse_percent_format(node.left.s)
        except ValueError:
            self.report(messages.PercentFormatInvalidFormat, node, 'incomplete format')
            return
        named: Set[str] = set()
        positional_count: int = 0
        positional: Optional[bool] = None
        for _, placeholder in placeholders:
            if placeholder is None:
                continue
            name, _, width, precision, conversion = placeholder
            if conversion == '%':
                continue
            if conversion not in VALID_CONVERSIONS:
                self.report(messages.PercentFormatUnsupportedFormatCharacter, node, conversion)
            if positional is None and conversion:
                positional = name is None
            for part in (width, precision):
                if part is not None and '*' in part:
                    if not positional:
                        self.report(messages.PercentFormatStarRequiresSequence, node)
                    else:
                        positional_count += 1
            if positional and name is not None:
                self.report(messages.PercentFormatMixingAutomatic, node)
                return
            elif not positional and name is None:
                self.report(messages.PercentFormatMixingAutomatic, node)
                return
            if positional:
                positional_count += 1
            else:
                named.add(name)
        if isinstance(node.right, (ast.List, ast.Tuple)) and not any(isinstance(elt, getattr(ast, 'Starred', ())) for elt in node.right.elts):
            substitution_count: int = len(node.right.elts)
            if positional and positional_count != substitution_count:
                self.report(messages.PercentFormatPositionalCountMismatch, node, positional_count, substitution_count)
            elif not positional:
                self.report(messages.PercentFormatExpectedMapping, node)
        if isinstance(node.right, ast.Dict) and all(isinstance(k, ast.Str) for k in node.right.keys):
            if positional and positional_count > 1:
                self.report(messages.PercentFormatExpectedSequence, node)
                return
            substitution_keys: Set[str] = {k.s for k in node.right.keys}
            extra_keys = substitution_keys - named
            missing_keys = named - substitution_keys
            if not positional and extra_keys:
                self.report(messages.PercentFormatExtraNamedArguments, node, ', '.join(sorted(extra_keys)))
            if not positional and missing_keys:
                self.report(messages.PercentFormatMissingArgument, node, ', '.join(sorted(missing_keys)))

    def BINOP(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Str):
            self._handle_percent_format(node)
        self.handleChildren(node)

    def STR(self, node: ast.Str) -> None:
        if self._in_annotation:
            fn = functools.partial(
                self.handleStringAnnotation,
                node.s,
                node,
                node.lineno,
                node.col_offset,
                messages.ForwardAnnotationSyntaxError,
            )
            if self._in_deferred:
                fn()
            else:
                self.deferFunction(fn)

    if PY38_PLUS:

        def CONSTANT(self, node: ast.Constant) -> None:
            if isinstance(node.value, str):
                self.STR(node)
    else:
        NUM = BYTES = ELLIPSIS = CONSTANT = ignore

    SUBSCRIPT: Callable[[ast.Subscript], None] = SUBSCRIPT
    SLICE = EXTSLICE = INDEX = handleChildren
    LOAD = STORE = DEL = AUGLOAD = AUGSTORE = PARAM = ignore
    AND = OR = ADD = SUB = MULT = DIV = MOD = POW = LSHIFT = RSHIFT = BITOR = BITXOR = BITAND = FLOORDIV = INVERT = NOT = UADD = USUB = EQ = NOTEQ = LT = LTE = GT = GTE = IS = ISNOT = IN = NOTIN = MATMULT = ignore

    def RAISE(self, node: ast.Raise) -> None:
        self.handleChildren(node)
        arg = get_raise_argument(node)
        if isinstance(arg, ast.Call):
            if is_notimplemented_name_node(arg.func):
                self.report(messages.RaiseNotImplemented, node)
        elif is_notimplemented_name_node(arg):
            self.report(messages.RaiseNotImplemented, node)

    COMPREHENSION = KEYWORD = FORMATTEDVALUE = handleChildren
    _in_fstring: bool = False

    def JOINEDSTR(self, node: ast.JoinedStr) -> None:
        if not self._in_fstring and not any(isinstance(x, ast.FormattedValue) for x in node.values):
            self.report(messages.FStringMissingPlaceholders, node)
        self._in_fstring, orig = True, self._in_fstring
        try:
            self.handleChildren(node)
        finally:
            self._in_fstring = orig

    def DICT(self, node: ast.Dict) -> None:
        keys = [convert_to_value(key) for key in node.keys]
        key_counts = counter(keys)
        duplicate_keys = [key for key, count in key_counts.items() if count > 1]
        for key in duplicate_keys:
            key_indices = [i for i, i_key in enumerate(keys) if i_key == key]
            values = counter((convert_to_value(node.values[index]) for index in key_indices))
            if any(count == 1 for count in values.values()):
                for key_index in key_indices:
                    key_node = node.keys[key_index]
                    if isinstance(key, VariableKey):
                        self.report(messages.MultiValueRepeatedKeyVariable, key_node, key.name)
                    else:
                        self.report(messages.MultiValueRepeatedKeyLiteral, key_node, key)
        self.handleChildren(node)

    def IF(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Tuple) and node.test.elts != []:
            self.report(messages.IfTuple, node)
        self.handleChildren(node)

    IFEXP = IF

    def ASSERT(self, node: ast.Assert) -> None:
        if isinstance(node.test, ast.Tuple) and node.test.elts != []:
            self.report(messages.AssertTuple, node)
        self.handleChildren(node)

    def GLOBAL(self, node: ast.Global) -> None:
        """
        Keep track of globals declarations.
        """
        global_scope_index = 1 if self._in_doctest() else 0
        global_scope = self.scopeStack[global_scope_index]
        if self.scope is not global_scope:
            for node_name in node.names:
                node_value = Assignment(node_name, node)
                self.messages = [
                    m for m in self.messages
                    if not (isinstance(m, messages.UndefinedName) and m.message_args[0] == node_name)
                ]
                global_scope.setdefault(node_name, node_value)
                node_value.used = (global_scope, node)
                for scope in self.scopeStack[global_scope_index + 1:]:
                    scope[node_name] = node_value

    NONLOCAL = GLOBAL

    def GENERATOREXP(self, node: ast.GeneratorExp) -> None:
        self.pushScope(GeneratorScope)
        self.handleChildren(node)
        self.popScope()

    LISTCOMP: Callable[[ast.ListComp], None] = GENERATOREXP if not PY2 else handleChildren
    DICTCOMP = SETCOMP = GENERATOREXP

    def NAME(self, node: ast.Name) -> None:
        """
        Handle occurrence of Name (which can be a load/store/delete access.)
        """
        if isinstance(node.ctx, ast.Load):
            self.handleNodeLoad(node)
            if node.id == 'locals' and isinstance(self.scope, FunctionScope) and isinstance(node._pyflakes_parent, ast.Call):
                self.scope.usesLocals = True
        elif isinstance(node.ctx, ast.Store):
            self.handleNodeStore(node)
        elif PY2 and isinstance(node.ctx, ast.Param):
            self.handleNodeStore(node)
        elif isinstance(node.ctx, ast.Del):
            self.handleNodeDelete(node)
        else:
            raise RuntimeError(f'Got impossible expression context: {node.ctx!r}')

    def CONTINUE(self, node: ast.Continue) -> None:
        n: ast.AST = node
        while hasattr(n, '_pyflakes_parent'):
            n, n_child = n._pyflakes_parent, n
            if isinstance(n, LOOP_TYPES):
                if n_child not in n.orelse:
                    return
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                break
            if hasattr(n, 'finalbody') and isinstance(node, ast.Continue):
                if n_child in n.finalbody and (not PY38_PLUS):
                    self.report(messages.ContinueInFinally, node)
                    return
        if isinstance(node, ast.Continue):
            self.report(messages.ContinueOutsideLoop, node)
        else:
            self.report(messages.BreakOutsideLoop, node)

    BREAK = CONTINUE

    def RETURN(self, node: ast.Return) -> None:
        if isinstance(self.scope, (ClassScope, ModuleScope)):
            self.report(messages.ReturnOutsideFunction, node)
            return
        if node.value and hasattr(self.scope, 'returnValue') and not self.scope.returnValue:
            self.scope.returnValue = node.value
        self.handleNode(node.value, node)

    def YIELD(self, node: ast.Yield) -> None:
        if isinstance(self.scope, (ClassScope, ModuleScope)):
            self.report(messages.YieldOutsideFunction, node)
            return
        self.scope.isGenerator = True
        self.handleNode(node.value, node)

    AWAIT = YIELDFROM = YIELD

    def FUNCTIONDEF(self, node: ast.FunctionDef) -> None:
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        self.LAMBDA(node)
        self.addBinding(node, FunctionDefinition(node.name, node))
        if self.withDoctest and not self._in_doctest() and not isinstance(self.scope, FunctionScope):
            self.deferFunction(lambda: self.handleDoctests(node))

    ASYNCFUNCTIONDEF = FUNCTIONDEF

    def LAMBDA(self, node: ast.Lambda) -> None:
        args: List[str] = []
        annotations: List[Optional[ast.AST]] = []
        if PY2:

            def addArgs(arglist: List[Union[ast.Name, ast.Tuple]]) -> None:
                for arg in arglist:
                    if isinstance(arg, ast.Tuple):
                        addArgs(arg.elts)
                    else:
                        args.append(arg.id)

            addArgs(node.args.args)
            defaults: List[ast.AST] = node.args.defaults
        else:
            if PY38_PLUS:
                for arg in node.args.posonlyargs:
                    args.append(arg.arg)
                    annotations.append(arg.annotation)
            for arg in node.args.args + node.args.kwonlyargs:
                args.append(arg.arg)
                annotations.append(arg.annotation)
            defaults = node.args.defaults + node.args.kw_defaults
        is_py3_func: bool = hasattr(node, 'returns')
        for arg_name in ('vararg', 'kwarg'):
            wildcard = getattr(node.args, arg_name, None)
            if not wildcard:
                continue
            args.append(wildcard if PY2 else wildcard.arg)
            if is_py3_func:
                if PY2:
                    argannotation = arg_name + 'annotation'
                    annotations.append(getattr(node.args, argannotation))
                else:
                    annotations.append(wildcard.annotation)
        if is_py3_func:
            annotations.append(node.returns)
        if len(set(args)) < len(args):
            for idx, arg in enumerate(args):
                if arg in args[:idx]:
                    self.report(messages.DuplicateArgument, node, arg)
        for annotation in annotations:
            self.handleAnnotation(annotation, node)
        for default in defaults:
            self.handleNode(default, node)

        def runFunction() -> None:
            self.pushScope()
            self.handleChildren(node, omit=['decorator_list', 'returns'])

            def checkUnusedAssignments() -> None:
                """
                Check to see if any assignments have not been used.
                """
                for name, binding in self.scope.unusedAssignments():
                    self.report(messages.UnusedVariable, binding.source, name)

            self.deferAssignment(checkUnusedAssignments)
            if PY2:

                def checkReturnWithArgumentInsideGenerator() -> None:
                    """
                    Check to see if there is any return statement with
                    arguments but the function is a generator.
                    """
                    if self.scope.isGenerator and self.scope.returnValue:
                        self.report(messages.ReturnWithArgsInsideGenerator, self.scope.returnValue)

                self.deferAssignment(checkReturnWithArgumentInsideGenerator)
            self.popScope()

        self.deferFunction(runFunction)

    def ARGUMENTS(self, node: ast.arguments) -> None:
        self.handleChildren(node, omit=('defaults', 'kw_defaults'))
        if PY2:
            scope_node = self.getScopeNode(node)
            if node.vararg:
                self.addBinding(node.vararg, Argument(node.vararg, scope_node))
            if node.kwarg:
                self.addBinding(node.kwarg, Argument(node.kwarg, scope_node))

    def ARG(self, node: ast.arg) -> None:
        self.addBinding(node, Argument(node.arg, self.getScopeNode(node)))

    def CLASSDEF(self, node: ast.ClassDef) -> None:
        """
        Check names used in a class definition, including its decorators, base
        classes, and the body of its definition.  Additionally, add its name to
        the current scope.
        """
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        for baseNode in node.bases:
            self.handleNode(baseNode, node)
        if not PY2:
            for keywordNode in node.keywords:
                self.handleNode(keywordNode, node)
        self.pushScope(ClassScope)
        if self.withDoctest and not self._in_doctest() and not isinstance(self.scope, FunctionScope):
            self.deferFunction(lambda: self.handleDoctests(node))
        for stmt in node.body:
            self.handleNode(stmt, node)
        self.popScope()
        self.addBinding(node, ClassDefinition(node.name, node))

    def AUGASSIGN(self, node: ast.AugAssign) -> None:
        self.handleNodeLoad(node.target)
        self.handleNode(node.value, node)
        self.handleNode(node.target, node)

    def TUPLE(self, node: ast.Tuple) -> None:
        if not PY2 and isinstance(node.ctx, ast.Store):
            has_starred = False
            star_loc = -1
            for i, n in enumerate(node.elts):
                if isinstance(n, ast.Starred):
                    if has_starred:
                        self.report(messages.TwoStarredExpressions, node)
                        break
                    has_starred = True
                    star_loc = i
            if star_loc >= 1 << 8 or len(node.elts) - star_loc - 1 >= 1 << 24:
                self.report(messages.TooManyExpressionsInStarredAssignment, node)
        self.handleChildren(node)

    LIST = TUPLE

    def IMPORT(self, node: ast.Import) -> None:
        for alias in node.names:
            if '.' in alias.name and (not alias.asname):
                importation = SubmoduleImportation(alias.name, node)
            else:
                name = alias.asname or alias.name
                importation = Importation(name, node, alias.name)
            self.addBinding(node, importation)

    def IMPORTFROM(self, node: ast.ImportFrom) -> None:
        if node.module == '__future__':
            if not self.futuresAllowed:
                self.report(messages.LateFutureImport, node, [n.name for n in node.names])
        else:
            self.futuresAllowed = False
        module = '.' * node.level + (node.module or '')
        for alias in node.names:
            if alias.name == '*':
                if not PY2 and not isinstance(self.scope, ModuleScope):
                    self.report(messages.ImportStarNotPermitted, node, module)
                    continue
                self.scope.importStarred = True
                self.report(messages.ImportStarUsed, node, module)
                importation = StarImportation(module, node)
            else:
                name = alias.asname or alias.name
                if node.module == '__future__':
                    importation = FutureImportation(name, node, self.scope)
                    if alias.name not in __future__.all_feature_names:
                        self.report(messages.FutureFeatureNotDefined, node, alias.name)
                    if alias.name == 'annotations':
                        self.annotationsFutureEnabled = True
                else:
                    importation = ImportationFrom(name, node, module, alias.name)
            self.addBinding(node, importation)

    def TRY(self, node: ast.Try) -> None:
        handler_names: Tuple[str, ...] = ()
        for i, handler in enumerate(node.handlers):
            if isinstance(handler.type, ast.Tuple):
                for exc_type in handler.type.elts:
                    handler_names += (getNodeName(exc_type),)
            elif handler.type:
                handler_names += (getNodeName(handler.type),)
            if handler.type is None and i < len(node.handlers) - 1:
                self.report(messages.DefaultExceptNotLast, handler)
        self.exceptHandlers.append(handler_names)
        for child in node.body:
            self.handleNode(child, node)
        self.exceptHandlers.pop()
        self.handleChildren(node, omit='body')

    TRYEXCEPT = TRY

    def EXCEPTHANDLER(self, node: ast.ExceptHandler) -> None:
        if PY2 or node.name is None:
            self.handleChildren(node)
            return
        if node.name in self.scope:
            self.handleNodeStore(node)
        try:
            prev_definition = self.scope.pop(node.name)
        except KeyError:
            prev_definition = None
        self.handleNodeStore(node)
        self.handleChildren(node)
        try:
            binding = self.scope.pop(node.name)
        except KeyError:
            pass
        else:
            if not binding.used:
                self.report(messages.UnusedVariable, node, node.name)
        if prev_definition:
            self.scope[node.name] = prev_definition

    def ANNASSIGN(self, node: ast.AnnAssign) -> None:
        self.handleNode(node.target, node)
        self.handleAnnotation(node.annotation, node)
        if node.value:
            self.handleNode(node.value, node)

    def COMPARE(self, node: ast.Compare) -> None:
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)) and (_is_const_non_singleton(left) or _is_const_non_singleton(right)):
                self.report(messages.IsLiteral, node)
            left = right
        self.handleChildren(node)

    MATCH = MATCH_CASE = MATCHCLASS = MATCHOR = MATCHSEQUENCE = handleChildren
    MATCHSINGLETON = MATCHVALUE = handleChildren

    def _match_target(self, node: ast.AST) -> None:
        self.handleNodeStore(node)
        self.handleChildren(node)

    MATCHAS = MATCHMAPPING = MATCHSTAR = _match_target

    def handleStringAnnotation(
        self,
        s: str,
        node: DummyNode,
        ref_lineno: int,
        ref_col_offset: int,
        err: type,
    ) -> None:
        try:
            tree = ast.parse(s)
        except SyntaxError:
            self.report(err, node, s)
            return
        body = tree.body
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            self.report(err, node, s)
            return
        parsed_annotation = tree.body[0].value
        for descendant in ast.walk(parsed_annotation):
            if 'lineno' in descendant._attributes and 'col_offset' in descendant._attributes:
                descendant.lineno = ref_lineno
                descendant.col_offset = ref_col_offset
        self.handleNode(parsed_annotation, node)

    def IGNORE(self, node: ast.AST) -> None:
        pass
