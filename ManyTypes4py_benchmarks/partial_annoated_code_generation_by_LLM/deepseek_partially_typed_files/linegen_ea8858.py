"""
Generating lines of code.
"""
import re
import sys
from collections.abc import Collection, Iterator
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Optional, Union, cast, Any, Callable, Type
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, STRING_PRIORITY, get_leaves_inside_matching_brackets, max_delimiter_priority_in_atom
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import Line, RHSResult, append_leaves, can_be_split, can_omit_invisible_parens, is_line_short_enough, line_to_string
from black.mode import Feature, Mode, Preview
from black.nodes import ASSIGNMENTS, BRACKETS, CLOSING_BRACKETS, OPENING_BRACKETS, STANDALONE_COMMENT, STATEMENT, WHITESPACE, Visitor, ensure_visible, fstring_to_string, get_annotation_type, is_arith_like, is_async_stmt_or_funcdef, is_atom_with_invisible_parens, is_docstring, is_empty_tuple, is_generator, is_lpar_token, is_multiline_string, is_name_token, is_one_sequence_between, is_one_tuple, is_parent_function_or_class, is_part_of_annotation, is_rpar_token, is_stub_body, is_stub_suite, is_tuple_containing_star, is_tuple_containing_walrus, is_type_ignore_comment_string, is_vararg, is_walrus_assignment, is_yield, syms, wrap_in_parentheses
from black.numerics import normalize_numeric_literal
from black.strings import fix_multiline_docstring, get_string_prefix, normalize_string_prefix, normalize_string_quotes, normalize_unicode_escape_sequences
from black.trans import CannotTransform, StringMerger, StringParenStripper, StringParenWrapper, StringSplitter, Transformer, hug_power_op
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
LeafID = int
LN = Union[Leaf, Node]

class CannotSplit(CannotTransform):
    """A readable split that fits the allotted line length is impossible."""

class LineGenerator(Visitor[Line]):
    """Generates reformatted Line objects.  Empty lines are not emitted.

    Note: destroys the tree it's visiting by mutating prefixes of its leaves
    in ways that will no longer stringify to valid Python code on the tree.
    """

    def __init__(self, mode: Mode, features: Collection[Feature]) -> None:
        self.mode: Mode = mode
        self.features: Collection[Feature] = features
        self.current_line: Line
        self.__post_init__()

    def line(self, indent: int = 0) -> Iterator[Line]:
        """Generate a line.

        If the line is empty, only emit if it makes sense.
        If the line is too long, split it first and then generate.

        If any lines were generated, set up a new current_line.
        """
        if not self.current_line:
            self.current_line.depth += indent
            return
        if len(self.current_line.leaves) == 1 and is_async_stmt_or_funcdef(self.current_line.leaves[0]):
            return
        complete_line: Line = self.current_line
        self.current_line = Line(mode=self.mode, depth=complete_line.depth + indent)
        yield complete_line

    def visit_default(self, node: LN) -> Iterator[Line]:
        """Default `visit_*()` implementation. Recurses to children of `node`."""
        if isinstance(node, Leaf):
            any_open_brackets: bool = self.current_line.bracket_tracker.any_open_brackets()
            for comment in generate_comments(node):
                if any_open_brackets:
                    self.current_line.append(comment)
                elif comment.type == token.COMMENT:
                    self.current_line.append(comment)
                    yield from self.line()
                else:
                    yield from self.line()
                    self.current_line.append(comment)
                    yield from self.line()
            if any_open_brackets:
                node.prefix = ''
            if node.type not in WHITESPACE:
                self.current_line.append(node)
        yield from super().visit_default(node)

    def visit_test(self, node: Node) -> Iterator[Line]:
        """Visit an `x if y else z` test"""
        already_parenthesized: bool = node.prev_sibling and node.prev_sibling.type == token.LPAR
        if not already_parenthesized:
            lpar: Leaf = Leaf(token.LPAR, '')
            rpar: Leaf = Leaf(token.RPAR, '')
            prefix: str = node.prefix
            node.prefix = ''
            lpar.prefix = prefix
            node.insert_child(0, lpar)
            node.append_child(rpar)
        yield from self.visit_default(node)

    def visit_INDENT(self, node: Leaf) -> Iterator[Line]:
        """Increase indentation level, maybe yield a line."""
        yield from self.line(+1)
        yield from self.visit_default(node)

    def visit_DEDENT(self, node: Leaf) -> Iterator[Line]:
        """Decrease indentation level, maybe yield a line."""
        yield from self.line()
        yield from self.visit_default(node)
        yield from self.line(-1)

    def visit_stmt(self, node: Node, keywords: set[str], parens: set[str]) -> Iterator[Line]:
        """Visit a statement.

        This implementation is shared for `if`, `while`, `for`, `try`, `except`,
        `def`, `with`, `class`, `assert`, and assignments.

        The relevant Python language `keywords` for a given statement will be
        NAME leaves within it. This methods puts those on a separate line.

        `parens` holds a set of string leaf values immediately after which
        invisible parens should be put.
        """
        normalize_invisible_parens(node, parens_after=parens, mode=self.mode, features=self.features)
        for child in node.children:
            if is_name_token(child) and child.value in keywords:
                yield from self.line()
            yield from self.visit(child)

    def visit_typeparams(self, node: Node) -> Iterator[Line]:
        yield from self.visit_default(node)
        node.children[0].prefix = ''

    def visit_typevartuple(self, node: Node) -> Iterator[Line]:
        yield from self.visit_default(node)
        node.children[1].prefix = ''

    def visit_paramspec(self, node: Node) -> Iterator[Line]:
        yield from self.visit_default(node)
        node.children[1].prefix = ''

    def visit_dictsetmaker(self, node: Node) -> Iterator[Line]:
        if Preview.wrap_long_dict_values_in_parens in self.mode:
            for (i, child) in enumerate(node.children):
                if i == 0:
                    continue
                if node.children[i - 1].type == token.COLON:
                    if child.type == syms.atom and child.children[0].type in OPENING_BRACKETS and (not is_walrus_assignment(child)):
                        maybe_make_parens_invisible_in_atom(child, parent=node, remove_brackets_around_comma=False)
                    else:
                        wrap_in_parentheses(node, child, visible=False)
        yield from self.visit_default(node)

    def visit_funcdef(self, node: Node) -> Iterator[Line]:
        """Visit function definition."""
        yield from self.line()
        is_return_annotation: bool = False
        for child in node.children:
            if child.type == token.RARROW:
                is_return_annotation = True
            elif is_return_annotation:
                if child.type == syms.atom and child.children[0].type == token.LPAR:
                    if maybe_make_parens_invisible_in_atom(child, parent=node, remove_brackets_around_comma=False):
                        wrap_in_parentheses(node, child, visible=False)
                else:
                    wrap_in_parentheses(node, child, visible=False)
                is_return_annotation = False
        for child in node.children:
            yield from self.visit(child)

    def visit_match_case(self, node: Node) -> Iterator[Line]:
        """Visit either a match or case statement."""
        normalize_invisible_parens(node, parens_after=set(), mode=self.mode, features=self.features)
        yield from self.line()
        for child in node.children:
            yield from self.visit(child)

    def visit_suite(self, node: Node) -> Iterator[Line]:
        """Visit a suite."""
        if is_stub_suite(node):
            yield from self.visit(node.children[2])
        else:
            yield from self.visit_default(node)

    def visit_simple_stmt(self, node: Node) -> Iterator[Line]:
        """Visit a statement without nested statements."""
        prev_type: Optional[int] = None
        for child in node.children:
            if (prev_type is None or prev_type == token.SEMI) and is_arith_like(child):
                wrap_in_parentheses(node, child, visible=False)
            prev_type = child.type
        if node.parent and node.parent.type in STATEMENT:
            if is_parent_function_or_class(node) and is_stub_body(node):
                yield from self.visit_default(node)
            else:
                yield from self.line(+1)
                yield from self.visit_default(node)
                yield from self.line(-1)
        else:
            if node.parent and is_stub_suite(node.parent):
                node.prefix = ''
                yield from self.visit_default(node)
                return
            yield from self.line()
            yield from self.visit_default(node)

    def visit_async_stmt(self, node: Node) -> Iterator[Line]:
        """Visit `async def`, `async for`, `async with`."""
        yield from self.line()
        children = iter(node.children)
        for child in children:
            yield from self.visit(child)
            if child.type == token.ASYNC or child.type == STANDALONE_COMMENT:
                break
        internal_stmt: LN = next(children)
        yield from self.visit(internal_stmt)

    def visit_decorators(self, node: Node) -> Iterator[Line]:
        """Visit decorators."""
        for child in node.children:
            yield from self.line()
            yield from self.visit(child)

    def visit_power(self, node: Node) -> Iterator[Line]:
        for (idx, leaf) in enumerate(node.children[:-1]):
            next_leaf: LN = node.children[idx + 1]
            if not isinstance(leaf, Leaf):
                continue
            value: str = leaf.value.lower()
            if leaf.type == token.NUMBER and next_leaf.type == syms.trailer and (next_leaf.children[0].type == token.DOT) and (not value.startswith(('0x', '0b', '0o'))) and ('j' not in value):
                wrap_in_parentheses(node, leaf)
        remove_await_parens(node)
        yield from self.visit_default(node)

    def visit_SEMI(self, leaf: Leaf) -> Iterator[Line]:
        """Remove a semicolon and put the other statement on a separate line."""
        yield from self.line()

    def visit_ENDMARKER(self, leaf: Leaf) -> Iterator[Line]:
        """End of file. Process outstanding comments and end with a newline."""
        yield from self.visit_default(leaf)
        yield from self.line()

    def visit_STANDALONE_COMMENT(self, leaf: Leaf) -> Iterator[Line]:
        if not self.current_line.bracket_tracker.any_open_brackets():
            yield from self.line()
        yield from self.visit_default(leaf)

    def visit_factor(self, node: Node) -> Iterator[Line]:
        """Force parentheses between a unary op and a binary power:

        -2 ** 8 -> -(2 ** 8)
        """
        (_operator, operand) = node.children
        if operand.type == syms.power and len(operand.children) == 3 and (operand.children[1].type == token.DOUBLESTAR):
            lpar: Leaf = Leaf(token.LPAR, '(')
            rpar: Leaf = Leaf(token.RPAR, ')')
            index: int = operand.remove() or 0
            node.insert_child(index, Node(syms.atom, [lpar, operand, rpar]))
        yield from self.visit_default(node)

    def visit_tname(self, node: Node) -> Iterator[Line]:
        """
        Add potential parentheses around types in function parameter lists to be made
        into real parentheses in case the type hint is too long to fit on a line
        Examples:
        def foo(a: int, b: float = 7): ...

        ->

        def foo(a: (int), b: (float) = 7): ...
        """
        assert len(node.children) == 3
        if maybe_make_parens_invisible_in_atom(node.children[2], parent=node):
            wrap_in_parentheses(node, node.children[2], visible=False)
        yield from self.visit_default(node)

    def visit_STRING(self, leaf: Leaf) -> Iterator[Line]:
        normalize_unicode_escape_sequences(leaf)
        if is_docstring(leaf) and (not re.search('\\\\\\s*\\n', leaf.value)):
            if self.mode.string_normalization:
                docstring: str = normalize_string_prefix(leaf.value)
                docstring = normalize_string_quotes(docstring)
            else:
                docstring = leaf.value
            prefix: str = get_string_prefix(docstring)
            docstring = docstring[len(prefix):]
            quote_char: str = docstring[0]
            quote_len: int = 1 if docstring[1] != quote_char else 3
            docstring = docstring[quote_len:-quote_len]
            docstring_started_empty: bool = not docstring
            indent: str = ' ' * 4 * self.current_line.depth
            if is_multiline_string(leaf):
                docstring = fix_multiline_docstring(docstring, indent)
            else:
                docstring = docstring.strip()
            has_trailing_backslash: bool = False
            if docstring:
                if docstring[0] == quote_char:
                    docstring = ' ' + docstring
                if docstring[-1] == quote_char:
                    docstring += ' '
                if docstring[-1] == '\\':
                    backslash_count: int = len(docstring) - len(docstring.rstrip('\\'))
                    if backslash_count % 2:
                        docstring += ' '
                        has_trailing_backslash = True
            elif not docstring_started_empty:
                docstring = ' '
            quote: str = quote_char * quote_len
            if quote_len == 3:
                lines: list[str] = docstring.splitlines()
                last_line_length: int = len(lines[-1]) if docstring else 0
                if len(lines) > 1 and last_line_length + quote_len > self.mode.line_length and (len(indent) + quote_len <= self.mode.line_length) and (not has_trailing_backslash):
                    if leaf.value[-1 - quote_len] == '\n':
                        leaf.value = prefix + quote + docstring + quote
                    else:
                        leaf.value = prefix + quote + docstring + '\n' + indent + quote
                else:
                    leaf.value = prefix + quote + docstring + quote
            else:
                leaf.value = prefix + quote + docstring + quote
        if self.mode.string_normalization and leaf.type == token.STRING:
            leaf.value = normalize_string_prefix(leaf.value)
            leaf.value = normalize_string_quotes(leaf.value)
        yield from self.visit_default(leaf)

    def visit_NUMBER(self, leaf: Leaf) -> Iterator[Line]:
        normalize_numeric_literal(leaf)
        yield from self.visit_default(leaf)

    def visit_atom(self, node: Node) -> Iterator[Line]:
        """Visit any atom"""
        if len(node.children) == 3:
            first: LN = node.children[0]
            last: LN = node.children[-1]
            if first.type == token.LSQB and last.type == token.RSQB or (first.type == token.LBRACE and last.type == token.RBRACE):
                maybe_make_parens_invisible_in_atom(node.children[1], parent=node)
        yield from self.visit_default(node)

    def visit_fstring(self, node: Node) -> Iterator[Line]:
        string_leaf: Leaf = fstring_to_string(node)
        node.replace(string_leaf)
        if '\\' in string_leaf.value and any(('\\' in str(child) for child in node.children if child.type == syms.fstring_replacement_field)):
            yield from self.visit_default(string_leaf)
            return
        yield from self.visit_STRING(string_leaf)

    def __post_init__(self) -> None:
        """You are in a twisty little maze of passages."""
        self.current_line = Line(mode=self.mode)
        v: Callable[[Node, set[str], set[str]], Iterator[Line]] = self.visit_stmt
        Ø: set[str] = set()
        self.visit_assert_stmt: Callable[[Node], Iterator[Line]] = partial(v, keywords={'assert'}, parens={'assert', ','})
        self.visit_if_stmt: Callable[[Node], Iterator[Line]] = partial(v, keywords={'if', 'else', 'elif'}, parens={'if', 'elif'})
        self.visit_while_stmt: Callable[[Node], Iterator[Line]] = partial(v, keywords={'while', 'else'}, parens={'while'})
        self.visit_for_stmt: Callable[[Node], Iterator[Line]] = partial(v, keywords={'for', 'else'}, parens={'for', 'in'})
        self.visit_try_stmt: Callable[[Node], Iterator[Line]] = partial(v, keywords={'try', 'except', 'else', 'finally'}, parens=Ø)
        self.visit_except_clause: Callable[[Node], Iterator[Line]] = partial(v, keywords={'except'}, parens={'except'})
        self.visit_with_stmt: Callable[[Node], Iterator[Line]] = partial(v, keywords={'with'}, parens={'with'})
        self.visit_classdef: Callable[[Node], Iterator[Line]] = partial(v, keywords={'class'}, parens=Ø)
        self.visit_expr_stmt: Callable[[