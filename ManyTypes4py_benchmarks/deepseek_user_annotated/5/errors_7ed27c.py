# -*- coding: utf-8 -*-
import codecs
import warnings
import re
from contextlib import contextmanager
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Iterator, Generator,
    ContextManager, Pattern, Match, cast, Type, TypeVar, Callable, Iterable
)

from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
from parso.python.tree import (
    Leaf, Node, PythonNode, PythonLeaf, PythonErrorNode, PythonErrorLeaf
)
from parso.tree import BaseNode, Leaf as BaseLeaf

_BLOCK_STMTS = ('if_stmt', 'while_stmt', 'for_stmt', 'try_stmt', 'with_stmt')
_STAR_EXPR_PARENTS = ('testlist_star_expr', 'testlist_comp', 'exprlist')
# This is the maximal block size given by python.
_MAX_BLOCK_SIZE = 20
_MAX_INDENT_COUNT = 100
ALLOWED_FUTURES = (
    'nested_scopes', 'generators', 'division', 'absolute_import',
    'with_statement', 'print_function', 'unicode_literals', 'generator_stop',
)
_COMP_FOR_TYPES = ('comp_for', 'sync_comp_for')

T = TypeVar('T')
NodeOrLeaf = Union[BaseNode, BaseLeaf]
PythonNodeOrLeaf = Union[PythonNode, PythonLeaf]

def _get_rhs_name(node: PythonNodeOrLeaf, version: Tuple[int, int]) -> str:
    type_ = node.type
    if type_ == "lambdef":
        return "lambda"
    elif type_ == "atom":
        comprehension = _get_comprehension_type(node)
        first, second = node.children[:2]
        if comprehension is not None:
            return comprehension
        elif second.type == "dictorsetmaker":
            if version < (3, 8):
                return "literal"
            else:
                if second.children[1] == ":" or second.children[0] == "**":
                    return "dict display"
                else:
                    return "set display"
        elif (
            first == "("
            and (second == ")"
                 or (len(node.children) == 3 and node.children[1].type == "testlist_comp"))
        ):
            return "tuple"
        elif first == "(":
            return _get_rhs_name(_remove_parens(node), version=version)
        elif first == "[":
            return "list"
        elif first == "{" and second == "}":
            return "dict display"
        elif first == "{" and len(node.children) > 2:
            return "set display"
    elif type_ == "keyword":
        if "yield" in node.value:
            return "yield expression"
        if version < (3, 8):
            return "keyword"
        else:
            return str(node.value)
    elif type_ == "operator" and node.value == "...":
        return "Ellipsis"
    elif type_ == "comparison":
        return "comparison"
    elif type_ in ("string", "number", "strings"):
        return "literal"
    elif type_ == "yield_expr":
        return "yield expression"
    elif type_ == "test":
        return "conditional expression"
    elif type_ in ("atom_expr", "power"):
        if node.children[0] == "await":
            return "await expression"
        elif node.children[-1].type == "trailer":
            trailer = node.children[-1]
            if trailer.children[0] == "(":
                return "function call"
            elif trailer.children[0] == "[":
                return "subscript"
            elif trailer.children[0] == ".":
                return "attribute"
    elif (
        ("expr" in type_ and "star_expr" not in type_)  # is a substring
        or "_test" in type_
        or type_ in ("term", "factor")
    ):
        return "operator"
    elif type_ == "star_expr":
        return "starred"
    elif type_ == "testlist_star_expr":
        return "tuple"
    elif type_ == "fstring":
        return "f-string expression"
    return type_  # shouldn't reach here


def _iter_stmts(scope: PythonNode) -> Generator[PythonNodeOrLeaf, None, None]:
    """
    Iterates over all statements and splits up simple_stmt.
    """
    for child in scope.children:
        if child.type == 'simple_stmt':
            for child2 in child.children:
                if child2.type == 'newline' or child2 == ';':
                    continue
                yield child2
        else:
            yield child


def _get_comprehension_type(atom: PythonNode) -> Optional[str]:
    first, second = atom.children[:2]
    if second.type == 'testlist_comp' and second.children[1].type in _COMP_FOR_TYPES:
        if first == '[':
            return 'list comprehension'
        else:
            return 'generator expression'
    elif second.type == 'dictorsetmaker' and second.children[-1].type in _COMP_FOR_TYPES:
        if second.children[1] == ':':
            return 'dict comprehension'
        else:
            return 'set comprehension'
    return None


def _is_future_import(import_from: PythonNode) -> bool:
    from_names = import_from.get_from_names()
    return [n.value for n in from_names] == ['__future__']


def _remove_parens(atom: PythonNodeOrLeaf) -> PythonNodeOrLeaf:
    """
    Returns the inner part of an expression like `(foo)`. Also removes nested
    parens.
    """
    try:
        children = atom.children
    except AttributeError:
        pass
    else:
        if len(children) == 3 and children[0] == '(':
            return _remove_parens(atom.children[1])
    return atom


def _skip_parens_bottom_up(node: PythonNodeOrLeaf) -> Optional[PythonNodeOrLeaf]:
    """
    Returns an ancestor node of an expression, skipping all levels of parens
    bottom-up.
    """
    while node.parent is not None:
        node = node.parent
        if node.type != 'atom' or node.children[0] != '(':
            return node
    return None


def _iter_params(parent_node: PythonNode) -> Generator[PythonNodeOrLeaf, None, None]:
    return (n for n in parent_node.children if n.type == 'param' or n.type == 'operator')


def _is_future_import_first(import_from: PythonNode) -> bool:
    """
    Checks if the import is the first statement of a file.
    """
    found_docstring = False
    for stmt in _iter_stmts(import_from.get_root_node()):
        if stmt.type == 'string' and not found_docstring:
            continue
        found_docstring = True

        if stmt == import_from:
            return True
        if stmt.type == 'import_from' and _is_future_import(stmt):
            continue
        return False


def _iter_definition_exprs_from_lists(exprlist: PythonNode) -> Generator[PythonNodeOrLeaf, None, None]:
    def check_expr(child: PythonNodeOrLeaf) -> Generator[PythonNodeOrLeaf, None, None]:
        if child.type == 'atom':
            if child.children[0] == '(':
                testlist_comp = child.children[1]
                if testlist_comp.type == 'testlist_comp':
                    yield from _iter_definition_exprs_from_lists(testlist_comp)
                    return
                else:
                    # It's a paren that doesn't do anything, like 1 + (1)
                    yield from check_expr(testlist_comp)
                    return
            elif child.children[0] == '[':
                yield testlist_comp
                return
        yield child

    if exprlist.type in _STAR_EXPR_PARENTS:
        for child in exprlist.children[::2]:
            yield from check_expr(child)
    else:
        yield from check_expr(exprlist)


def _get_expr_stmt_definition_exprs(expr_stmt: PythonNode) -> List[PythonNodeOrLeaf]:
    exprs = []
    for list_ in expr_stmt.children[:-2:2]:
        if list_.type in ('testlist_star_expr', 'testlist'):
            exprs += list(_iter_definition_exprs_from_lists(list_))
        else:
            exprs.append(list_)
    return exprs


def _get_for_stmt_definition_exprs(for_stmt: PythonNode) -> List[PythonNodeOrLeaf]:
    exprlist = for_stmt.children[1]
    return list(_iter_definition_exprs_from_lists(exprlist))


def _is_argument_comprehension(argument: PythonNode) -> bool:
    return argument.children[1].type in _COMP_FOR_TYPES


def _any_fstring_error(version: Tuple[int, int], node: Optional[PythonNodeOrLeaf]) -> bool:
    if version < (3, 9) or node is None:
        return False
    if node.type == "error_node":
        return any(child.type == "fstring_start" for child in node.children)
    elif node.type == "fstring":
        return True
    else:
        return node.search_ancestor("fstring") is not None


class _Context:
    def __init__(self, node: PythonNode, add_syntax_error: Callable[[PythonNodeOrLeaf, str], None], parent_context: Optional['_Context'] = None):
        self.node = node
        self.blocks: List[PythonNode] = []
        self.parent_context = parent_context
        self._used_name_dict: Dict[str, List[PythonLeaf]] = {}
        self._global_names: List[PythonLeaf]] = []
        self._local_params_names: List[str] = []
        self._nonlocal_names: List[PythonLeaf]] = []
        self._nonlocal_names_in_subscopes: List[PythonLeaf]] = []
        self._add_syntax_error = add_syntax_error

    def is_async_funcdef(self) -> bool:
        return self.is_function() \
            and self.node.parent.type in ('async_funcdef', 'async_stmt')

    def is_function(self) -> bool:
        return self.node.type == 'funcdef'

    def add_name(self, name: PythonLeaf) -> None:
        parent_type = name.parent.type
        if parent_type == 'trailer':
            return

        if parent_type == 'global_stmt':
            self._global_names.append(name)
        elif parent_type == 'nonlocal_stmt':
            self._nonlocal_names.append(name)
        elif parent_type == 'funcdef':
            self._local_params_names.extend(
                [param.name.value for param in name.parent.get_params()]
            )
        else:
            self._used_name_dict.setdefault(name.value, []).append(name)

    def finalize(self) -> List[PythonLeaf]:
        self._analyze_names(self._global_names, 'global')
        self._analyze_names(self._nonlocal_names, 'nonlocal')

        global_name_strs = {n.value: n for n in self._global_names}
        for nonlocal_name in self._nonlocal_names:
            try:
                global_name = global_name_strs[nonlocal_name.value]
            except KeyError:
                continue

            message = "name '%s' is nonlocal and global" % global_name.value
            if global_name.start_pos < nonlocal_name.start_pos:
                error_name = global_name
            else:
                error_name = nonlocal_name
            self._add_syntax_error(error_name, message)

        nonlocals_not_handled = []
        for nonlocal_name in self._nonlocal_names_in_subscopes:
            search = nonlocal_name.value
            if search in self._local_params_names:
                continue
            if search in global_name_strs or self.parent_context is None:
                message = "no binding for nonlocal '%s' found" % nonlocal_name.value
                self._add_syntax_error(nonlocal_name, message)
            elif not self.is_function() or \
                    nonlocal_name.value not in self._used_name_dict:
                nonlocals_not_handled.append(nonlocal_name)
        return self._nonlocal_names + nonlocals_not_handled

    def _analyze_names(self, globals_or_nonlocals: List[PythonLeaf], type_: str) -> None:
        def raise_(message: str) -> None:
            self._add_syntax_error(base_name, message % (base_name.value, type_))

        params = []
        if self.node.type == 'funcdef':
            params = self.node.get_params()

        for base_name in globals_or_nonlocals:
            found_global_or_nonlocal = False
            for name in reversed(self._used_name_dict.get(base_name.value, [])):
                if name.start_pos > base_name.start_pos:
                    found_global_or_nonlocal = True

                parent = name.parent
                if parent.type == 'param' and parent.name == name:
                    continue

                if name.is_definition():
                    if parent.type == 'expr_stmt' \
                            and parent.children[1].type == 'annassign':
                        if found_global_or_nonlocal:
                            base_name = name
                        raise_("annotated name '%s' can't be %s")
                        break
                    else:
                        message = "name '%s' is assigned to before %s declaration"
                else:
                    message = "name '%s' is used prior to %s declaration"

                if not found_global_or_nonlocal:
                    raise_(message)
                    break

            for param in params:
                if param.name.value == base_name.value:
                    raise_("name '%s' is parameter and %s")

    @contextmanager
    def add_block(self, node: PythonNode) -> Generator[None, None, None]:
        self.blocks.append(node)
        yield
        self.blocks.pop()

    def add_context(self, node: PythonNode) -> '_Context':
        return _Context(node, self._add_syntax_error, parent_context=self)

    def close_child_context(self, child_context: '_Context') -> None:
        self._nonlocal_names_in_subscopes += child_context.finalize()


class ErrorFinder(Normalizer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._error_dict: Dict[int, Tuple[int, str, PythonNodeOrLeaf]] = {}
        self.version = self.grammar.version_info

    def initialize(self, node: PythonNode) -> None:
        def create_context(node: Optional[PythonNode]) -> Optional[_Context]:
            if node is None:
                return None

            parent_context = create_context(node.parent)
            if node.type in ('classdef', 'funcdef', 'file_input'):
                return _Context(node, self._add_syntax_error, parent_context)
            return parent_context

        self.context = create_context(node) or _Context(node, self._add_syntax_error)
        self._indentation_count = 0

    def visit(self, node: PythonNodeOrLeaf) -> str:
        if node.type == 'error_node':
            with self.visit_node(node):
                return ''
        return super().visit(node)

    @contextmanager
    def visit_node(self, node: PythonNodeOrLeaf) -> Generator[None, None, None]:
        self._check_type_rules(node)

        if node.type in _BLOCK_STMTS:
            with self.context.add_block(node):
                if len(self.context.blocks) == _MAX_BLOCK_SIZE:
                    self._add_syntax_error(node, "too many statically nested blocks")
                yield
            return
        elif node.type == 'suite':
            self._indentation_count += 1
            if self._indentation_count == _MAX_INDENT_COUNT:
                self._add_indentation_error(node.children[1], "too many levels of indentation")

        yield

        if node.type == 'suite':
            self._indentation_count -= 1
        elif node.type in ('classdef', 'funcdef'):
            context = self.context
            self.context = context.parent_context
            self.context.close_child_context(context)

    def visit_leaf(self, leaf: PythonLeaf) -> str:
        if leaf.type == 'error_leaf':
            if leaf.token_type in ('INDENT', 'ERROR_DEDENT'):
                spacing = list(leaf.get_next_leaf()._split_prefix())[-1]
                if leaf.token_type == 'INDENT':
                    message = 'unexpected indent'
                else:
                    message = 'unindent does not match any outer indentation level'
                self._add_indentation_error(spacing, message)
            else:
                if leaf.value.startswith('\\'):
                    message = 'unexpected character after line continuation character'
                else:
                    match = re.match('\\w{,2}("{1,3}|\'{1,3})', leaf.value)
                    if match is None:
                        message = 'invalid syntax'
                        if (
                            self.version >= (3, 9)
                            and leaf.value in _get_token_collection(
                                self.version
                            ).always_break_tokens
                        ):
                            message = "f-string: " + message
                    else:
                        if len(match.group(1)) == 1:
                            message = 'EOL while scanning string literal'
                        else:
                            message = 'EOF while scanning triple-quoted string literal'
                self._add_syntax_error(leaf, message)
            return ''
        elif leaf.value == ':':
            parent = leaf.parent
            if parent.type in ('classdef', 'funcdef'):
                self.context = self.context.add_context(parent)

        return super().visit_leaf(leaf)

    def _add_indentation_error(self, spacing: PythonLeaf, message: str) -> None:
        self.add_issue(spacing, 903, "IndentationError: " + message)

    def _add_syntax_error(self, node: PythonNodeOrLeaf, message: str) -> None:
        self.add_issue(node, 901, "SyntaxError: " + message)

    def add_issue(self, node: PythonNodeOrLeaf, code: int, message: str) -> None:
        line = node.start_pos[0]
        args = (code, message, node)
        self._error_dict.setdefault(line, args)

    def finalize(self) -> None:
        self.context.finalize()

        for code, message, node in self._error_dict.values():
            self.issues.append(Issue(node, code, message))


class IndentationRule(Rule):
   