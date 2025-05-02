"""
Main module.

Implement the central Checker class.
Also, it models the Bindings and Scopes.
"""
from __future__ import annotations
import __future__
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
from typing import (
    Any, Callable, Dict, FrozenSet, Generator, Iterable, Iterator, List,
    Optional, Set, Tuple, Type, TypeVar, Union, cast, overload
)
from pyflakes import messages

T = TypeVar('T')
NodeT = TypeVar('NodeT', bound=ast.AST)
ScopeT = TypeVar('ScopeT', bound='Scope')

PY2: bool = sys.version_info < (3, 0)
PY35_PLUS: bool = sys.version_info >= (3, 5)
PY36_PLUS: bool = sys.version_info >= (3, 6)
PY38_PLUS: bool = sys.version_info >= (3, 8)
try:
    sys.pypy_version_info
    PYPY: bool = True
except AttributeError:
    PYPY: bool = False

builtin_vars: List[str] = dir(__import__('__builtin__' if PY2 else 'builtins'))
parse_format_string: Callable[[str], Iterator[Tuple[str, str, str, str]] = string.Formatter().parse

if PY2:
    tokenize_tokenize: Callable[..., Iterator[tokenize.TokenInfo]] = tokenize.generate_tokens
else:
    tokenize_tokenize: Callable[..., Iterator[tokenize.TokenInfo]] = tokenize.tokenize

if PY2:
    def getNodeType(node_class: Type[ast.AST]) -> str:
        return str(unicode(node_class.__name__).upper()

    def get_raise_argument(node: ast.Raise) -> ast.expr:
        return node.type
else:
    def getNodeType(node_class: Type[ast.AST]) -> str:
        return node_class.__name__.upper()

    def get_raise_argument(node: ast.Raise) -> ast.expr:
        return node.exc
    unicode = str

if PY2:
    def getAlternatives(n: ast.AST) -> List[List[ast.stmt]]:
        if isinstance(n, (ast.If, ast.TryFinally)):
            return [n.body]
        if isinstance(n, ast.TryExcept):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
else:
    def getAlternatives(n: ast.AST) -> List[List[ast.stmt]]:
        if isinstance(n, ast.If):
            return [n.body]
        if isinstance(n, ast.Try):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]

if PY35_PLUS:
    FOR_TYPES: Tuple[Type[ast.For], Type[ast.AsyncFor]] = (ast.For, ast.AsyncFor)
    LOOP_TYPES: Tuple[Type[ast.While], Type[ast.For], Type[ast.AsyncFor]] = (ast.While, ast.For, ast.AsyncFor)
    FUNCTION_TYPES: Tuple[Type[ast.FunctionDef], Type[ast.AsyncFunctionDef]] = (ast.FunctionDef, ast.AsyncFunctionDef)
else:
    FOR_TYPES: Tuple[Type[ast.For]] = (ast.For,)
    LOOP_TYPES: Tuple[Type[ast.While], Type[ast.For]] = (ast.While, ast.For)
    FUNCTION_TYPES: Tuple[Type[ast.FunctionDef]] = (ast.FunctionDef,)

if PY36_PLUS:
    ANNASSIGN_TYPES: Tuple[Type[ast.AnnAssign]] = (ast.AnnAssign,)
else:
    ANNASSIGN_TYPES: Tuple[()] = ()

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
    _const_tps: Tuple[Type[ast.Str], Type[ast.Num]] = (ast.Str, ast.Num)
    if not PY2:
        _const_tps += (ast.Bytes,)

    def _is_constant(node: ast.AST) -> bool:
        return isinstance(node, _const_tps) or _is_singleton(node) or _is_tuple_constant(node)

def _is_const_non_singleton(node: ast.AST) -> bool:
    return _is_constant(node) and (not _is_singleton(node))

def _is_name_or_attr(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name or (isinstance(node, ast.Attribute) and node.attr == name)

TYPE_COMMENT_RE: re.Pattern = re.compile('^#\\s*type:\\s*')
ASCII_NON_ALNUM: str = ''.join([chr(i) for i in range(128) if not chr(i).isalnum()])
TYPE_IGNORE_RE: re.Pattern = re.compile(TYPE_COMMENT_RE.pattern + 'ignore([{}]|$)'.format(ASCII_NON_ALNUM))
TYPE_FUNC_RE: re.Pattern = re.compile('^(\\(.*?\\))\\s*->\\s*(.*)$')
MAPPING_KEY_RE: re.Pattern = re.compile('\\(([^()]*)\\)')
CONVERSION_FLAG_RE: re.Pattern = re.compile('[#0+ -]*')
WIDTH_RE: re.Pattern = re.compile('(?:\\*|\\d*)')
PRECISION_RE: re.Pattern = re.compile('(?:\\.(?:\\*|\\d*))?')
LENGTH_RE: re.Pattern = re.compile('[hlL]?')
VALID_CONVERSIONS: FrozenSet[str] = frozenset('diouxXeEfFgGcrsa%')

def _must_match(regex: re.Pattern, string: str, pos: int) -> re.Match:
    match = regex.match(string, pos)
    assert match is not None
    return match

def parse_percent_format(s: str) -> Tuple[Tuple[Optional[str], Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]], ...]:
    """Parses the string component of a `'...' % ...` format call

    Copied from https://github.com/asottile/pyupgrade at v1.20.1
    """

    def _parse_inner() -> Generator[Tuple[str, Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]]], None, None]:
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

    def _get_fields(self, node_class: Type[ast.AST]) -> Tuple[str, ...]:
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

def counter(items: Iterable[T]) -> Dict[T, int]:
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results: Dict[T, int] = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results

def iter_child_nodes(node: ast.AST, omit: Optional[Union[str, Tuple[str, ...]]] = None, _fields_order: _FieldsOrder = _FieldsOrder()) -> Generator[ast.AST, None, None]:
    """
    Yield all direct child nodes of *node*, that is, all fields that
    are nodes and all items of fields that are lists of nodes.

    :param node:          AST node to be iterated upon
    :param omit:          String or tuple of strings denoting the
                          attributes of the node to be omitted from
                          further parsing
    :param _fields_order: Order of AST node fields
    """
    for name in _fields_order[node.__class__]:
        if omit and name in omit:
            continue
        field = getattr(node, name, None)
        if isinstance(field, ast.AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, ast.AST):
                    yield item

def convert_to_value(item: ast.AST) -> Union[str, bytes, Tuple[Any, ...], float, 'VariableKey', bool, type(None), 'UnhandledKeyType']:
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

class Binding(object):
    """
    Represents the binding of a value to a name.

    The checker uses this to keep track of which names have been bound and
    which names have not. See L{Assignment} for a special type of binding that
    is checked with stricter rules.

    @ivar used: pair of (L{Scope}, node) indicating the scope and
                the node that this binding was last used.
    """

    def __init__(self, name: str, source: ast.AST) -> None:
        self.name = name
        self.source = source
        self.used: Union[bool, Tuple['Scope', ast.AST]] = False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return '<%s object %r from line %r at 0x%x>' % (self.__class__.__name__, self.name, self.source.lineno, id(self))

    def redefines(self, other: 'Binding') -> bool:
        return isinstance(other, Definition) and self.name == other.name

class Definition(Binding):
    """
    A binding that defines a function or a class.
    """

class Builtin(Definition):
    """A definition created for all Python builtins."""

    def __init__(self, name: str) -> None:
        super(Builtin, self).__init__(name, None)

    def __repr__(self) -> str:
        return '<%s object %r at 0x%x>' % (self.__class__.__name__, self.name, id(self))

class UnhandledKeyType(object):
    """
    A dictionary key of a type that we cannot or do not check for duplicates.
    """

class VariableKey(object):
    """
    A dictionary key which is a variable.

    @ivar item: The variable AST object.
    """

    def __init__(self, item: ast.Name) -> None:
        self.name = item.id

    def __eq__(self, compare: object) -> bool:
        return compare.__class__ == self.__class__ and compare.name == self.name

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
        self.fullName = full_name or name
        self.redefined: List[ast.AST] = []
        super(Importation, self).__init__(name, source)

    def redefines(self, other: 'Binding') -> bool:
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
            return 'import %s as %s' % (self.fullName, self.name)
        else:
            return 'import %s' % self.fullName

    def __str__(self) -> str:
        """Return import full name with alias."""
        if self._has_alias():
            return self.fullName + ' as ' + self.name
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

    def __init__(self, name: str, source: ast.Import) -> None:
        assert '.' in name and (not source or isinstance(source, ast.Import))
        package_name = name.split('.')[0]
        super(SubmoduleImportation, self).__init__(package_name, source)
        self.fullName = name

    def redefines(self, other: 'Binding') -> bool:
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        return super(SubmoduleImportation, self).redefines(other)

    def __str__(self) -> str:
        return self.fullName

    @property
    def source_statement(self) -> str:
        return 'import ' + self.fullName

class ImportationFrom(Importation):

    def __init__(self, name: str, source: ast.AST, module: str, real_name: Optional[str] = None) -> None:
        self.module = module
        self.real_name = real_name or name
        if module.endswith('.'):
            full_name = module + self.real_name
        else:
            full_name = module + '.' + self.real_name
        super(ImportationFrom, self).__init__(name, source, full_name)

    def __str__(self) -> str:
        """Return import full name with alias."""
        if self.real_name != self.name:
            return self.fullName + ' as ' + self.name
        else:
            return self.fullName

    @property
    def source_statement(self) -> str:
        if self.real_name != self.name:
            return 'from %s import %s as %s' % (self.module, self.real_name, self.name)
        else:
            return 'from %s import %s' % (self.module, self.real_name)

class StarImportation(Importation):
    """A binding created by a 'from x import *' statement."""

    def __init__(self, name: str, source: ast.AST) -> None:
        super(StarImportation, self).__init__