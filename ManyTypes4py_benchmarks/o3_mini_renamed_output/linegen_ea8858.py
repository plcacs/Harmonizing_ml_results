#!/usr/bin/env python3
"""
Generating lines of code.
"""
import re
import sys
from collections.abc import Collection, Iterator, Callable
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Optional, Union, cast, List, Set

from black.brackets import (
    COMMA_PRIORITY,
    DOT_PRIORITY,
    STRING_PRIORITY,
    get_leaves_inside_matching_brackets,
    max_delimiter_priority_in_atom,
)
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
    Line,
    RHSResult,
    append_leaves,
    can_be_split,
    can_omit_invisible_parens,
    is_line_short_enough,
    line_to_string,
)
from black.mode import Feature, Mode, Preview
from black.nodes import (
    ASSIGNMENTS,
    BRACKETS,
    CLOSING_BRACKETS,
    OPENING_BRACKETS,
    STANDALONE_COMMENT,
    STATEMENT,
    WHITESPACE,
    Visitor,
    ensure_visible,
    fstring_to_string,
    get_annotation_type,
    is_arith_like,
    is_async_stmt_or_funcdef,
    is_atom_with_invisible_parens,
    is_docstring,
    is_empty_tuple,
    is_generator,
    is_lpar_token,
    is_multiline_string,
    is_name_token,
    is_one_sequence_between,
    is_one_tuple,
    is_parent_function_or_class,
    is_part_of_annotation,
    is_rpar_token,
    is_stub_body,
    is_stub_suite,
    is_tuple_containing_star,
    is_tuple_containing_walrus,
    is_type_ignore_comment_string,
    is_vararg,
    is_walrus_assignment,
    is_yield,
    syms,
    wrap_in_parentheses,
)
from black.numerics import normalize_numeric_literal
from black.strings import (
    fix_multiline_docstring,
    get_string_prefix,
    normalize_string_prefix,
    normalize_string_quotes,
    normalize_unicode_escape_sequences,
)
from black.trans import (
    CannotTransform,
    StringMerger,
    StringParenStripper,
    StringParenWrapper,
    StringSplitter,
    Transformer,
    hug_power_op,
)
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
        self.__post_init__()

    def func_d1qjl4yo(self, indent: int = 0) -> Iterator[Line]:
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

    def func_n91q18bp(self, node: LN) -> Iterator[Line]:
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

    def func_2mcou69j(self, node: LN) -> Iterator[Line]:
        """Visit an `x if y else z` test"""
        already_parenthesized: bool = (node.prev_sibling and node.prev_sibling.type == token.LPAR)
        if not already_parenthesized:
            lpar: Leaf = Leaf(token.LPAR, '')
            rpar: Leaf = Leaf(token.RPAR, '')
            prefix: str = node.prefix
            node.prefix = ''
            node.insert_child(0, lpar)
            node.append_child(rpar)
        yield from self.visit_default(node)

    def func_12dciboj(self, node: LN) -> Iterator[Line]:
        """Increase indentation level, maybe yield a line."""
        yield from self.line(+1)
        yield from self.visit_default(node)

    def func_kthy8dh1(self, node: LN) -> Iterator[Line]:
        """Decrease indentation level, maybe yield a line."""
        yield from self.line()
        yield from self.visit_default(node)
        yield from self.line(-1)

    def func_84v9cibt(self, node: LN, keywords: Collection[str], parens: Collection[str]) -> Iterator[Line]:
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

    def func_5shl0xua(self, node: LN) -> Iterator[Line]:
        yield from self.visit_default(node)
        node.children[0].prefix = ''
        yield from []  # To ensure Iterator return

    def func_93urnivp(self, node: LN) -> Iterator[Line]:
        yield from self.visit_default(node)
        node.children[1].prefix = ''
        yield from []

    def func_03xnperv(self, node: LN) -> Iterator[Line]:
        yield from self.visit_default(node)
        node.children[1].prefix = ''
        yield from []

    def func_4pkfe35e(self, node: LN) -> Iterator[Line]:
        if Preview.wrap_long_dict_values_in_parens in self.mode:
            for i, child in enumerate(node.children):
                if i == 0:
                    continue
                if node.children[i - 1].type == token.COLON:
                    if child.type == syms.atom and child.children[0].type in OPENING_BRACKETS and not is_walrus_assignment(child):
                        maybe_make_parens_invisible_in_atom(child, parent=node, remove_brackets_around_comma=False)
                    else:
                        wrap_in_parentheses(node, child, visible=False)
        yield from self.visit_default(node)

    def func_2n0anyrs(self, node: LN) -> Iterator[Line]:
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

    def func_14xo5l7j(self, node: LN) -> Iterator[Line]:
        """Visit either a match or case statement."""
        normalize_invisible_parens(node, parens_after=set(), mode=self.mode, features=self.features)
        yield from self.line()
        for child in node.children:
            yield from self.visit(child)

    def func_327d027a(self, node: LN) -> Iterator[Line]:
        """Visit a suite."""
        if is_stub_suite(node):
            yield from self.visit(node.children[2])
        else:
            yield from self.visit_default(node)

    def func_3qhreym7(self, node: LN) -> Iterator[Line]:
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

    def func_gnou2ofu(self, node: LN) -> Iterator[Line]:
        """Visit `async def`, `async for`, `async with`."""
        yield from self.line()
        children = iter(node.children)
        for child in children:
            yield from self.visit(child)
            if child.type == token.ASYNC or child.type == STANDALONE_COMMENT:
                break
        internal_stmt = next(children)
        yield from self.visit(internal_stmt)

    def func_cm6lxtig(self, node: LN) -> Iterator[Line]:
        """Visit decorators."""
        for child in node.children:
            yield from self.line()
            yield from self.visit(child)

    def func_i59it9wh(self, node: LN) -> Iterator[Line]:
        for idx, leaf in enumerate(node.children[:-1]):
            next_leaf = node.children[idx + 1]
            if not isinstance(leaf, Leaf):
                continue
            value: str = leaf.value.lower()
            if (leaf.type == token.NUMBER and next_leaf.type == syms.trailer and next_leaf.children[0].type == token.DOT and not value.startswith(('0x', '0b', '0o')) and 'j' not in value):
                wrap_in_parentheses(node, leaf)
        remove_await_parens(node)
        yield from self.visit_default(node)

    def func_2ct4c8uq(self, leaf: Leaf) -> Iterator[Line]:
        """Remove a semicolon and put the other statement on a separate line."""
        yield from self.line()

    def func_yrfoscmk(self, leaf: Leaf) -> Iterator[Line]:
        """End of file. Process outstanding comments and end with a newline."""
        yield from self.visit_default(leaf)
        yield from self.line()

    def func_dvq1znb2(self, leaf: Leaf) -> Iterator[Line]:
        if not self.current_line.bracket_tracker.any_open_brackets():
            yield from self.line()
        yield from self.visit_default(leaf)

    def func_gxhnkja6(self, node: LN) -> Iterator[Line]:
        """Force parentheses between a unary op and a binary power:

        -2 ** 8 -> -(2 ** 8)
        """
        _operator, operand = node.children  # type: ignore
        if operand.type == syms.power and len(operand.children) == 3 and operand.children[1].type == token.DOUBLESTAR:
            lpar: Leaf = Leaf(token.LPAR, '(')
            rpar: Leaf = Leaf(token.RPAR, ')')
            index = operand.remove() or 0  # type: ignore
            node.insert_child(index, Node(syms.atom, [lpar, operand, rpar]))
        yield from self.visit_default(node)

    def func_bfonbcn9(self, node: LN) -> Iterator[Line]:
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

    def func_3c2odbyy(self, leaf: Leaf) -> Iterator[Line]:
        normalize_unicode_escape_sequences(leaf)
        if is_docstring(leaf) and not re.search(r'\\\s*\n', leaf.value):
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
                lines: List[str] = docstring.splitlines()
                last_line_length: int = len(lines[-1]) if docstring else 0
                if (len(lines) > 1 and last_line_length + quote_len > self.mode.line_length and len(indent) + quote_len <= self.mode.line_length and not has_trailing_backslash):
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

    def func_zek5znb(self, leaf: Leaf) -> Iterator[Line]:
        normalize_numeric_literal(leaf)
        yield from self.visit_default(leaf)

    def func_fl5aoqpp(self, node: LN) -> Iterator[Line]:
        """Visit any atom"""
        if len(node.children) == 3:
            first = node.children[0]
            last = node.children[-1]
            if (first.type == token.LSQB and last.type == token.RSQB or 
                first.type == token.LBRACE and last.type == token.RBRACE):
                maybe_make_parens_invisible_in_atom(node.children[1], parent=node)
        yield from self.visit_default(node)

    def func_w1zhw9bg(self, node: LN) -> Iterator[Line]:
        string_leaf: Leaf = fstring_to_string(node)
        node.replace(string_leaf)
        if '\\' in string_leaf.value and any('\\' in str(child) for child in node.children if child.type == syms.fstring_replacement_field):
            yield from self.visit_default(string_leaf)
            return
        yield from self.visit_STRING(string_leaf)

    def __post_init__(self) -> None:
        """You are in a twisty little maze of passages."""
        self.current_line: Line = Line(mode=self.mode)
        v: Callable[..., Iterator[Line]] = self.visit_stmt  # type: ignore
        Ø: Set[str] = set()
        self.visit_assert_stmt = partial(v, keywords={'assert'}, parens={'assert', ','})
        self.visit_if_stmt = partial(v, keywords={'if', 'else', 'elif'}, parens={'if', 'elif'})
        self.visit_while_stmt = partial(v, keywords={'while', 'else'}, parens={'while'})
        self.visit_for_stmt = partial(v, keywords={'for', 'else'}, parens={'for', 'in'})
        self.visit_try_stmt = partial(v, keywords={'try', 'except', 'else', 'finally'}, parens=Ø)
        self.visit_except_clause = partial(v, keywords={'except'}, parens={'except'})
        self.visit_with_stmt = partial(v, keywords={'with'}, parens={'with'})
        self.visit_classdef = partial(v, keywords={'class'}, parens=Ø)
        self.visit_expr_stmt = partial(v, keywords=Ø, parens=ASSIGNMENTS)
        self.visit_return_stmt = partial(v, keywords={'return'}, parens={'return'})
        self.visit_import_from = partial(v, keywords=Ø, parens={'import'})
        self.visit_del_stmt = partial(v, keywords=Ø, parens={'del'})
        self.visit_async_funcdef = self.visit_async_stmt
        self.visit_decorated = self.visit_decorators
        self.visit_match_stmt = self.visit_match_case
        self.visit_case_block = self.visit_match_case
        self.visit_guard = partial(v, keywords=Ø, parens={'if'})


def func_8dca9rxi(line: Line, features: Collection[Feature], mode: Mode) -> Optional[str]:
    try:
        return line_to_string(next(hug_power_op(line, features, mode)))
    except CannotTransform:
        return None


def func_36yftdol(line: Line, mode: Mode, features: Collection[Feature] = ()) -> Iterator[Line]:
    """Transform a `line`, potentially splitting it into many lines.

    They should fit in the allotted `line_length` but might not be able to.

    `features` are syntactical features that may be used in the output.
    """
    if line.is_comment:
        yield line
        return
    line_str: str = line_to_string(line)
    line_str_hugging_power_ops: str = func_8dca9rxi(line, features, mode) or line_str
    ll: int = mode.line_length
    sn: bool = mode.string_normalization
    string_merge: StringMerger = StringMerger(ll, sn)
    string_paren_strip: StringParenStripper = StringParenStripper(ll, sn)
    string_split: StringSplitter = StringSplitter(ll, sn)
    string_paren_wrap: StringParenWrapper = StringParenWrapper(ll, sn)
    if (not func_d1qjl4yo.contains_uncollapsable_type_comments() and not line.should_split_rhs and not line.magic_trailing_comma and
        (is_line_short_enough(line, mode=mode, line_str=line_str_hugging_power_ops) or func_d1qjl4yo.contains_unsplittable_type_ignore()) and
        not (line.inside_brackets and func_d1qjl4yo.contains_standalone_comments()) and
        not func_d1qjl4yo.contains_implicit_multiline_string_with_comments()):
        if Preview.string_processing in mode:
            transformers: List[Callable[[Line, Collection[Feature], Mode], Iterator[Line]]] = [string_merge, string_paren_strip]
        else:
            transformers = []
    elif line.is_def and not should_split_funcdef_with_rhs(line, mode):
        transformers = [left_hand_split]
    else:
        def func_knrdhg0u(self, line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
            """Wraps calls to `right_hand_split`.

            The calls increasingly `omit` right-hand trailers (bracket pairs with
            content), meaning the trailers get glued together to split on another
            bracket pair instead.
            """
            for omit in generate_trailers_to_omit(line, mode.line_length):
                lines = list(right_hand_split(line, mode, features, omit=omit))
                if is_line_short_enough(lines[0], mode=mode):
                    yield from lines
                    return
            yield from right_hand_split(line, mode, features=features)
        rhs = type('rhs', (), {'__call__': _rhs})()  # type: ignore
        if Preview.string_processing in mode:
            if line.inside_brackets:
                transformers = [string_merge, string_paren_strip, string_split, delimiter_split, standalone_comment_split, string_paren_wrap, rhs]
            else:
                transformers = [string_merge, string_paren_strip, string_split, string_paren_wrap, rhs]
        elif line.inside_brackets:
            transformers = [delimiter_split, standalone_comment_split, rhs]
        else:
            transformers = [rhs]
    transformers.append(hug_power_op)
    for transform in transformers:
        try:
            result = run_transformer(line, transform, mode, features, line_str=line_str)
        except CannotTransform:
            continue
        else:
            yield from result
            break
    else:
        yield line


def func_tbiqdp0g(line: Line, mode: Mode) -> bool:
    """If a funcdef has a magic trailing comma in the return type, then we should first
    split the line with rhs to respect the comma.
    """
    return_type_leaves: List[Leaf] = []
    in_return_type: bool = False
    for leaf in line.leaves:
        if leaf.type == token.COLON:
            in_return_type = False
        if in_return_type:
            return_type_leaves.append(leaf)
        if leaf.type == token.RARROW:
            in_return_type = True
    result: Line = Line(mode=line.mode, depth=line.depth)
    leaves_to_track: Set[int] = get_leaves_inside_matching_brackets(return_type_leaves)
    for leaf in return_type_leaves:
        result.append(leaf, preformatted=True, track_bracket=id(leaf) in leaves_to_track)
    return result.magic_trailing_comma is not None


class _BracketSplitComponent(Enum):
    head = auto()
    body = auto()
    tail = auto()


def func_s0es0qrs(line: Line, _features: Collection[Feature], mode: Mode) -> Iterator[Line]:
    """Split line into many lines, starting with the first matching bracket pair.

    Note: this usually looks weird, only use this for function definitions.
    Prefer RHS otherwise.  This is why this function is not symmetrical with
    :func:`right_hand_split` which also handles optional parentheses.
    """
    for leaf_type in [token.LPAR, token.LSQB]:
        tail_leaves: List[Leaf] = []
        body_leaves: List[Leaf] = []
        head_leaves: List[Leaf] = []
        current_leaves: List[Leaf] = head_leaves
        matching_bracket: Optional[Leaf] = None
        for leaf in line.leaves:
            if (current_leaves is body_leaves and leaf.type in CLOSING_BRACKETS and leaf.opening_bracket is matching_bracket and isinstance(matching_bracket, Leaf)):
                ensure_visible(leaf)
                ensure_visible(matching_bracket)
                current_leaves = tail_leaves if body_leaves else head_leaves
            current_leaves.append(leaf)
            if current_leaves is head_leaves:
                if leaf.type == leaf_type:
                    matching_bracket = leaf
                    current_leaves = body_leaves
        if matching_bracket and tail_leaves:
            break
    if not matching_bracket or not tail_leaves:
        raise CannotSplit('No brackets found')
    head = bracket_split_build_line(head_leaves, line, matching_bracket, component=_BracketSplitComponent.head)
    body = bracket_split_build_line(body_leaves, line, matching_bracket, component=_BracketSplitComponent.body)
    tail = bracket_split_build_line(tail_leaves, line, matching_bracket, component=_BracketSplitComponent.tail)
    bracket_split_succeeded_or_raise(head, body, tail)
    for result in (head, body, tail):
        if result:
            yield result


def func_xxwfmlqr(line: Line, mode: Mode, features: Collection[Feature] = (), omit: Collection[int] = ()) -> Iterator[Line]:
    """Split line into many lines, starting with the last matching bracket pair.

    If the split was by optional parentheses, attempt splitting without them, too.
    `omit` is a collection of closing bracket IDs that shouldn't be considered for
    this split.

    Note: running this function modifies `bracket_depth` on the leaves of `line`.
    """
    rhs_result: RHSResult = _first_right_hand_split(line, omit=omit)
    yield from _maybe_split_omitting_optional_parens(rhs_result, line, mode, features=features, omit=omit)


def func_l05yw2vb(line: Line, omit: Collection[int] = ()) -> RHSResult:
    """Split the line into head, body, tail starting with the last bracket pair.

    Note: this function should not have side effects. It's relied upon by
    _maybe_split_omitting_optional_parens to get an opinion whether to prefer
    splitting on the right side of an assignment statement.
    """
    tail_leaves: List[Leaf] = []
    body_leaves: List[Leaf] = []
    head_leaves: List[Leaf] = []
    current_leaves: List[Leaf] = tail_leaves
    opening_bracket: Optional[Leaf] = None
    closing_bracket: Optional[Leaf] = None
    for leaf in reversed(line.leaves):
        if current_leaves is body_leaves:
            if leaf is opening_bracket:
                current_leaves = head_leaves if body_leaves else tail_leaves
        current_leaves.append(leaf)
        if current_leaves is tail_leaves:
            if leaf.type in CLOSING_BRACKETS and id(leaf) not in omit:
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf
                current_leaves = body_leaves
    if not (opening_bracket and closing_bracket and head_leaves):
        raise CannotSplit('No brackets found')
    tail_leaves.reverse()
    body_leaves.reverse()
    head_leaves.reverse()
    body = None
    if (Preview.hug_parens_with_braces_and_square_brackets in line.mode and tail_leaves[0].value and tail_leaves[0].opening_bracket is head_leaves[-1]):
        inner_body_leaves: List[Leaf] = list(body_leaves)
        hugged_opening_leaves: List[Leaf] = []
        hugged_closing_leaves: List[Leaf] = []
        is_unpacking: bool = body_leaves[0].type in [token.STAR, token.DOUBLESTAR]
        unpacking_offset: int = 1 if is_unpacking else 0
        while len(inner_body_leaves) >= 2 + unpacking_offset and inner_body_leaves[-1].type in CLOSING_BRACKETS and inner_body_leaves[-1].opening_bracket is inner_body_leaves[unpacking_offset]:
            if unpacking_offset:
                hugged_opening_leaves.append(inner_body_leaves.pop(0))
                unpacking_offset = 0
            hugged_opening_leaves.append(inner_body_leaves.pop(0))
            hugged_closing_leaves.insert(0, inner_body_leaves.pop())
        if hugged_opening_leaves and inner_body_leaves:
            inner_body = bracket_split_build_line(inner_body_leaves, line, hugged_opening_leaves[-1], component=_BracketSplitComponent.body)
            if line.mode.magic_trailing_comma and inner_body_leaves[-1].type == token.COMMA:
                should_hug: bool = True
            else:
                line_length: int = line.mode.line_length - sum(len(str(leaf)) for leaf in hugged_opening_leaves + hugged_closing_leaves)
                if is_line_short_enough(inner_body, mode=replace(line.mode, line_length=line_length)):
                    should_hug = False
                else:
                    should_hug = True
            if should_hug:
                body_leaves = inner_body_leaves
                head_leaves.extend(hugged_opening_leaves)
                tail_leaves = hugged_closing_leaves + tail_leaves
                body = inner_body
    head = bracket_split_build_line(head_leaves, line, opening_bracket, component=_BracketSplitComponent.head)
    if body is None:
        body = bracket_split_build_line(body_leaves, line, opening_bracket, component=_BracketSplitComponent.body)
    tail = bracket_split_build_line(tail_leaves, line, opening_bracket, component=_BracketSplitComponent.tail)
    bracket_split_succeeded_or_raise(head, body, tail)
    return RHSResult(head, body, tail, opening_bracket, closing_bracket)  # type: ignore


def func_4dnkxw2d(rhs: RHSResult, line: Line, mode: Mode, features: Collection[Feature] = (), omit: Collection[int] = ()) -> Iterator[Line]:
    if (Feature.FORCE_OPTIONAL_PARENTHESES not in features and rhs.opening_bracket.type == token.LPAR and not rhs.opening_bracket.value and rhs.closing_bracket.type == token.RPAR and not rhs.closing_bracket.value and not line.is_import and
        can_omit_invisible_parens(rhs, mode.line_length)):
        omit_updated: Set[int] = {id(rhs.closing_bracket), *omit}
        try:
            rhs_oop: RHSResult = func_l05yw2vb(line, omit=omit_updated)
            if _prefer_split_rhs_oop_over_rhs(rhs_oop, rhs, mode):
                yield from func_4dnkxw2d(rhs_oop, line, mode, features=features, omit=omit_updated)
                return
        except CannotSplit as e:
            if line.is_chained_assignment:
                pass
            elif not can_be_split(rhs.body) and not is_line_short_enough(rhs.body, mode=mode) and not (Preview.wrap_long_dict_values_in_parens and rhs.opening_bracket.parent and rhs.opening_bracket.parent.parent and rhs.opening_bracket.parent.parent.type == syms.dictsetmaker):
                raise CannotSplit("Splitting failed, body is still too long and can't be split.") from e
            elif rhs.head.contains_multiline_strings() or rhs.tail.contains_multiline_strings():
                raise CannotSplit('The current optional pair of parentheses is bound to fail to satisfy the splitting algorithm because the head or the tail contains multiline strings which by definition never fit one line.') from e
    ensure_visible(rhs.opening_bracket)
    ensure_visible(rhs.closing_bracket)
    for result in (rhs.head, rhs.body, rhs.tail):
        if result:
            yield result


def func_9xb09gat(rhs_oop: RHSResult, rhs: RHSResult, mode: Mode) -> bool:
    """
    Returns whether we should prefer the result from a split omitting optional parens
    (rhs_oop) over the original (rhs).
    """
    if rhs_oop.head.contains_unsplittable_type_ignore() or rhs_oop.body.contains_unsplittable_type_ignore() or rhs_oop.tail.contains_unsplittable_type_ignore():
        return True
    if (Preview.wrap_long_dict_values_in_parens and rhs.opening_bracket.parent and rhs.opening_bracket.parent.parent and rhs.opening_bracket.parent.parent.type == syms.dictsetmaker and rhs.body.bracket_tracker.delimiters):
        return any(leaf.type == token.COLON for leaf in rhs_oop.tail.leaves)
    if not (len(rhs.head.leaves) >= 2 and rhs.head.leaves[-2].type == token.EQUAL):
        return True
    if not any(leaf.type in BRACKETS for leaf in rhs.head.leaves[:-1]):
        return True
    if not is_line_short_enough(rhs.head, mode=replace(mode, line_length=mode.line_length - 1)):
        return True
    if rhs.head.magic_trailing_comma is not None:
        return True
    rhs_head_equal_count: int = [leaf.type for leaf in rhs.head.leaves].count(token.EQUAL)
    rhs_oop_head_equal_count: int = [leaf.type for leaf in rhs_oop.head.leaves].count(token.EQUAL)
    if (rhs_head_equal_count > 1 and rhs_head_equal_count > rhs_oop_head_equal_count):
        return False
    has_closing_bracket_after_assign: bool = False
    for leaf in reversed(rhs_oop.head.leaves):
        if leaf.type == token.EQUAL:
            break
        if leaf.type in CLOSING_BRACKETS:
            has_closing_bracket_after_assign = True
            break
    return has_closing_bracket_after_assign or (any(leaf.type == token.EQUAL for leaf in rhs_oop.head.leaves) and is_line_short_enough(rhs_oop.head, mode=mode))


def func_umrx812m(head: Line, body: Line, tail: Line) -> None:
    """Raise :exc:`CannotSplit` if the last left- or right-hand split failed.

    Do nothing otherwise.

    A left- or right-hand split is based on a pair of brackets. Content before
    (and including) the opening bracket is left on one line, content inside the
    brackets is put on a separate line, and finally content starting with and
    following the closing bracket is put on a separate line.

    Those are called `head`, `body`, and `tail`, respectively. If the split
    produced the same line (all content in `head`) or ended up with an empty `body`
    and the `tail` is just the closing bracket, then it's considered failed.
    """
    tail_len: int = len(str(tail).strip())
    if not body:
        if tail_len == 0:
            raise CannotSplit('Splitting brackets produced the same line')
        elif tail_len < 3:
            raise CannotSplit(f'Splitting brackets on an empty body to save {tail_len} characters is not worth it')


def func_wj1f3qbq(leaves: List[Leaf], original: Line, opening_bracket: Leaf) -> bool:
    if not leaves:
        return False
    if original.is_import:
        return True
    if not original.is_def:
        return False
    if opening_bracket.value != '(':
        return False
    if any(leaf.type == token.COMMA and not is_part_of_annotation(leaf) for leaf in leaves):
        return False
    leaf_with_parent: Optional[Leaf] = next((leaf for leaf in leaves if leaf.parent), None)
    if leaf_with_parent is None:
        return True
    if get_annotation_type(leaf_with_parent) == 'return':
        return False
    if (leaf_with_parent.parent and leaf_with_parent.parent.next_sibling and leaf_with_parent.parent.next_sibling.type == token.VBAR):
        return False
    return True


def func_3150bj6e(leaves: List[Leaf], original: Line, opening_bracket: Leaf, *, component: _BracketSplitComponent) -> Line:
    """Return a new line with given `leaves` and respective comments from `original`.

    If it's the head component, brackets will be tracked so trailing commas are
    respected.

    If it's the body component, the result line is one-indented inside brackets and as
    such has its first leaf's prefix normalized and a trailing comma added when
    expected.
    """
    result: Line = Line(mode=original.mode, depth=original.depth)
    if component is _BracketSplitComponent.body:
        result.inside_brackets = True
        result.depth += 1
        if func_wj1f3qbq(leaves, original, opening_bracket):
            for i in range(len(leaves) - 1, -1, -1):
                if leaves[i].type == STANDALONE_COMMENT:
                    continue
                if leaves[i].type != token.COMMA:
                    new_comma: Leaf = Leaf(token.COMMA, ',')
                    leaves.insert(i + 1, new_comma)
                break
    leaves_to_track: Set[int] = set()
    if component is _BracketSplitComponent.head:
        leaves_to_track = get_leaves_inside_matching_brackets(leaves)
    for leaf in leaves:
        result.append(leaf, preformatted=True, track_bracket=id(leaf) in leaves_to_track)
        for comment_after in original.comments_after(leaf):
            result.append(comment_after, preformatted=True)
    if component is _BracketSplitComponent.body and should_split_line(result, opening_bracket):
        result.should_split_rhs = True
    return result


def func_sypy2dl6(split_func: Callable[[Line, Collection[Feature], Mode], Iterator[Line]]) -> Callable[[Line, Collection[Feature], Mode], Iterator[Line]]:
    """Normalize prefix of the first leaf in every line returned by `split_func`.

    This is a decorator over relevant split functions.
    """
    @wraps(split_func)
    def func_0dk3pd9m(line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
        for split_line in split_func(line, features, mode):
            split_line.leaves[0].prefix = ''
            yield split_line
    return func_0dk3pd9m


def func_2jhvmku0(line: Line) -> Optional[int]:
    for leaf_idx in range(len(line.leaves) - 1, 0, -1):
        if line.leaves[leaf_idx].type != STANDALONE_COMMENT:
            return leaf_idx
    return None


def func_hc3kdz2q(leaf: Leaf, features: Collection[Feature]) -> bool:
    if is_vararg(leaf, within={syms.typedargslist}):
        return Feature.TRAILING_COMMA_IN_DEF in features
    if is_vararg(leaf, within={syms.arglist, syms.argument}):
        return Feature.TRAILING_COMMA_IN_CALL in features
    return True


def func_b57a9znb(safe: bool, delimiter_priority: int, line: Line) -> Line:
    if safe and delimiter_priority == COMMA_PRIORITY and line.leaves[-1].type != token.COMMA and line.leaves[-1].type != STANDALONE_COMMENT:
        new_comma: Leaf = Leaf(token.COMMA, ',')
        func_d1qjl4yo.append(new_comma)
    return line


MIGRATE_COMMENT_DELIMITERS: Set[int] = {STRING_PRIORITY, COMMA_PRIORITY}


@dont_increase_indentation
def func_pey2dcug(line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
    """Split according to delimiters of the highest priority.

    If the appropriate Features are given, the split will add trailing commas
    also in function signatures and calls that contain `*` and `**`.
    """
    if len(line.leaves) == 0:
        raise CannotSplit('Line empty') from None
    last_leaf: Leaf = line.leaves[-1]
    bt = line.bracket_tracker
    try:
        delimiter_priority: int = bt.max_delimiter_priority(exclude={id(last_leaf)})
    except ValueError:
        raise CannotSplit('No delimiters found') from None
    if delimiter_priority == DOT_PRIORITY and bt.delimiter_count_with_priority(delimiter_priority) == 1:
        raise CannotSplit('Splitting a single attribute from its owner looks wrong')
    current_line: Line = Line(mode=line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
    lowest_depth: int = sys.maxsize
    trailing_comma_safe: bool = True

    def func_ikmv5a2j(leaf: Leaf) -> Iterator[Line]:
        """Append `leaf` to current line or to new line if appending impossible."""
        nonlocal current_line
        try:
            current_line.append_safe(leaf, preformatted=True)
        except ValueError:
            yield current_line
            current_line = Line(mode=line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
            current_line.append(leaf)
        return
        yield  # for type-checker

    def func_1sxdjexs(leaf: Leaf) -> Iterator[Line]:
        for comment_after in func_d1qjl4yo.comments_after(leaf):
            yield from func_ikmv5a2j(comment_after)
    last_non_comment_leaf: Optional[int] = func_2jhvmku0(line)
    for leaf_idx, leaf in enumerate(line.leaves):
        yield from func_ikmv5a2j(leaf)
        previous_priority: Optional[int] = leaf_idx > 0 and bt.delimiters.get(id(line.leaves[leaf_idx - 1]))
        if (previous_priority != delimiter_priority or delimiter_priority in MIGRATE_COMMENT_DELIMITERS):
            yield from func_1sxdjexs(leaf)
        lowest_depth = min(lowest_depth, leaf.bracket_depth)
        if trailing_comma_safe and leaf.bracket_depth == lowest_depth:
            trailing_comma_safe = func_hc3kdz2q(leaf, features)
        if (last_leaf.type == STANDALONE_COMMENT and leaf_idx == last_non_comment_leaf):
            current_line = func_b57a9znb(trailing_comma_safe, delimiter_priority, current_line)
        leaf_priority: Optional[int] = bt.delimiters.get(id(leaf))
        if leaf_priority == delimiter_priority:
            if leaf_idx + 1 < len(line.leaves) and delimiter_priority not in MIGRATE_COMMENT_DELIMITERS:
                yield from func_1sxdjexs(line.leaves[leaf_idx + 1])
            yield current_line
            current_line = Line(mode=line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
    if current_line:
        current_line = func_b57a9znb(trailing_comma_safe, delimiter_priority, current_line)
        yield current_line


@dont_increase_indentation
def func_pw4mywdp(line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
    """Split standalone comments from the rest of the line."""
    if not func_d1qjl4yo.contains_standalone_comments():
        raise CannotSplit('Line does not have any standalone comments')
    current_line: Line = Line(mode=line.mode, depth=line.depth, inside_brackets=line.inside_brackets)

    def func_ikmv5a2j(leaf: Leaf) -> Iterator[Line]:
        """Append `leaf` to current line or to new line if appending impossible."""
        nonlocal current_line
        try:
            current_line.append_safe(leaf, preformatted=True)
        except ValueError:
            yield current_line
            current_line = Line(line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
            current_line.append(leaf)
        return
        yield  # for type-checker

    for leaf in line.leaves:
        yield from func_ikmv5a2j(leaf)
        for comment_after in func_d1qjl4yo.comments_after(leaf):
            yield from func_ikmv5a2j(comment_after)
    if current_line:
        yield current_line


def func_d1a7sc21(node: LN, parens_after: Collection[str], *, mode: Mode, features: Collection[Feature]) -> None:
    """Make existing optional parentheses invisible or create new ones.

    `parens_after` is a set of string leaf values immediately after which parens
    should be put.

    Standardizes on visible parentheses for single-element tuples, and keeps
    existing visible parentheses for other tuples and generator expressions.
    """
    for pc in list_comments(node.prefix, is_endmarker=False):
        if pc.value in FMT_OFF:
            return
    if node.type == syms.with_stmt:
        _maybe_wrap_cms_in_parens(node, mode, features)
    check_lpar: bool = False
    for index, child in enumerate(list(node.children)):
        if isinstance(child, Node) and child.type == syms.annassign:
            func_d1a7sc21(child, parens_after=parens_after, mode=mode, features=features)
        if isinstance(child, Node) and child.type == syms.case_block:
            func_d1a7sc21(child, parens_after={'case'}, mode=mode, features=features)
        if isinstance(child, Node) and child.type == syms.guard:
            func_d1a7sc21(child, parens_after={'if'}, mode=mode, features=features)
        if index == 0 and isinstance(child, Node) and child.type == syms.testlist_star_expr:
            check_lpar = True
        if check_lpar:
            if (child.type == syms.atom and node.type == syms.for_stmt and isinstance(child.prev_sibling, Leaf) and child.prev_sibling.type == token.NAME and child.prev_sibling.value == 'for'):
                if maybe_make_parens_invisible_in_atom(child, parent=node, remove_brackets_around_comma=True):
                    wrap_in_parentheses(node, child, visible=False)
            elif isinstance(child, Node) and node.type == syms.with_stmt:
                remove_with_parens(child, node)
            elif child.type == syms.atom:
                if maybe_make_parens_invisible_in_atom(child, parent=node):
                    wrap_in_parentheses(node, child, visible=False)
            elif is_one_tuple(child):
                wrap_in_parentheses(node, child, visible=True)
            elif node.type == syms.import_from:
                _normalize_import_from(node, child, index)
                break
            elif index == 1 and child.type == token.STAR and node.type == syms.except_clause:
                continue
            elif isinstance(child, Leaf) and child.next_sibling is not None and child.next_sibling.type == token.COLON and child.value == 'case':
                break
            elif not is_multiline_string(child):
                wrap_in_parentheses(node, child, visible=False)
        comma_check: bool = child.type == token.COMMA
        check_lpar = isinstance(child, Leaf) and (child.value in parens_after or comma_check)


def func_9jfa6sp9(parent: Node, child: LN, index: int) -> None:
    if is_lpar_token(child):
        assert is_rpar_token(parent.children[-1])
        child.value = ''
        parent.children[-1].value = ''
    elif child.type != token.STAR:
        parent.insert_child(index, Leaf(token.LPAR, ''))
        parent.append_child(Leaf(token.RPAR, ''))


def func_gjsf7w28(node: LN) -> None:
    if node.children[0].type == token.AWAIT and len(node.children) > 1:
        if node.children[1].type == syms.atom and node.children[1].children[0].type == token.LPAR:
            if maybe_make_parens_invisible_in_atom(node.children[1], parent=node, remove_brackets_around_comma=True):
                wrap_in_parentheses(node, node.children[1], visible=False)
            opening_bracket: Leaf = cast(Leaf, node.children[1].children[0])
            closing_bracket: Leaf = cast(Leaf, node.children[1].children[-1])
            bracket_contents = node.children[1].children[1]
            if isinstance(bracket_contents, Node) and (bracket_contents.type != syms.power or bracket_contents.children[0].type == token.AWAIT or any(isinstance(child, Leaf) and child.type == token.DOUBLESTAR for child in bracket_contents.children)):
                ensure_visible(opening_bracket)
                ensure_visible(closing_bracket)


def func_p2kk8xgb(node: LN, mode: Mode, features: Collection[Feature]) -> None:
    """When enabled and safe, wrap the multiple context managers in invisible parens.

    It is only safe when `features` contain Feature.PARENTHESIZED_CONTEXT_MANAGERS.
    """
    if Feature.PARENTHESIZED_CONTEXT_MANAGERS not in features or len(node.children) <= 2 or node.children[1].type == syms.atom:
        return
    colon_index: Optional[int] = None
    for i in range(2, len(node.children)):
        if node.children[i].type == token.COLON:
            colon_index = i
            break
    if colon_index is not None:
        lpar: Leaf = Leaf(token.LPAR, '')
        rpar: Leaf = Leaf(token.RPAR, '')
        context_managers: List[Node] = node.children[1:colon_index]  # type: ignore
        for child in context_managers:
            child.remove()
        new_child: Node = Node(syms.atom, [lpar, Node(syms.testlist_gexp, context_managers), rpar])
        node.insert_child(1, new_child)


def func_9ojxar8g(node: LN, parent: Node) -> None:
    """Recursively hide optional parens in `with` statements."""
    if node.type == syms.atom:
        if maybe_make_parens_invisible_in_atom(node, parent=parent, remove_brackets_around_comma=True):
            wrap_in_parentheses(parent, node, visible=False)
        if isinstance(node.children[1], Node):
            func_9ojxar8g(node.children[1], node)
    elif node.type == syms.testlist_gexp:
        for child in node.children:
            if isinstance(child, Node):
                func_9ojxar8g(child, node)
    elif node.type == syms.asexpr_test and not any(leaf.type == token.COLONEQUAL for leaf in node.leaves()):
        if maybe_make_parens_invisible_in_atom(node.children[0], parent=node, remove_brackets_around_comma=True):
            wrap_in_parentheses(node, node.children[0], visible=False)


def func_hoghf4l8(node: LN, parent: LN, remove_brackets_around_comma: bool = False) -> bool:
    """If it's safe, make the parens in the atom `node` invisible, recursively.
    Additionally, remove repeated, adjacent invisible parens from the atom `node`
    as they are redundant.

    Returns whether the node should itself be wrapped in invisible parentheses.
    """
    if (node.type not in (syms.atom, syms.expr) or is_empty_tuple(node) or is_one_tuple(node) or (is_yield(node) and parent.type != syms.expr_stmt) or 
        (not remove_brackets_around_comma and max_delimiter_priority_in_atom(node) >= COMMA_PRIORITY) or
        is_tuple_containing_walrus(node) or is_tuple_containing_star(node) or is_generator(node)):
        return False
    if is_walrus_assignment(node):
        if parent.type in [syms.annassign, syms.expr_stmt, syms.assert_stmt, syms.return_stmt, syms.except_clause, syms.funcdef, syms.with_stmt, syms.tname, syms.for_stmt, syms.del_stmt, syms.for_stmt]:
            return False
    first: Leaf = node.children[0]
    last: Leaf = node.children[-1]
    if is_lpar_token(first) and is_rpar_token(last):
        middle = node.children[1]
        if not is_type_ignore_comment_string(middle.prefix.strip()):
            first.value = ''
            last.value = ''
        func_hoghf4l8(middle, parent=parent, remove_brackets_around_comma=remove_brackets_around_comma)
        if is_atom_with_invisible_parens(middle):
            middle.replace(middle.children[1])
            if middle.children[0].prefix.strip():
                middle.children[1].prefix = middle.children[0].prefix + middle.children[1].prefix
            if middle.children[-1].prefix.strip():
                last.prefix = middle.children[-1].prefix + last.prefix
        return False
    return True


def func_iypuzfum(line: Line, opening_bracket: Leaf) -> bool:
    """Should `line` be immediately split with `delimiter_split()` after RHS?"""
    if not (opening_bracket.parent and opening_bracket.value in '[{('):
        return False
    exclude: Set[int] = set()
    trailing_comma: bool = False
    try:
        last_leaf: Leaf = line.leaves[-1]
        if last_leaf.type == token.COMMA:
            trailing_comma = True
            exclude.add(id(last_leaf))
        max_priority: int = line.bracket_tracker.max_delimiter_priority(exclude=exclude)
    except (IndexError, ValueError):
        return False
    return max_priority == COMMA_PRIORITY and ((line.mode.magic_trailing_comma and trailing_comma) or (opening_bracket.parent.type in {syms.atom, syms.import_from}))


def func_yesbcuw5(line: Line, line_length: int) -> Iterator[Set[int]]:
    """Generate sets of closing bracket IDs that should be omitted in a RHS.

    Brackets can be omitted if the entire trailer up to and including
    a preceding closing bracket fits in one line.

    Yielded sets are cumulative (contain results of previous yields, too).  First
    set is empty, unless the line should explode, in which case bracket pairs until
    the one that needs to explode are omitted.
    """
    omit: Set[int] = set()
    if not line.magic_trailing_comma:
        yield omit
    length: int = 4 * line.depth
    opening_bracket: Optional[Leaf] = None
    closing_bracket: Optional[Leaf] = None
    inner_brackets: Set[int] = set()
    for index, leaf, leaf_length in func_d1qjl4yo.enumerate_with_length(is_reversed=True):
        length += leaf_length
        if length > line_length:
            break
        has_inline_comment: bool = leaf_length > len(leaf.value) + len(leaf.prefix)
        if leaf.type == STANDALONE_COMMENT or has_inline_comment:
            break
        if opening_bracket:
            if leaf is opening_bracket:
                opening_bracket = None
            elif leaf.type in CLOSING_BRACKETS:
                prev = line.leaves[index - 1] if index > 0 else None
                if (prev and prev.type == token.COMMA and leaf.opening_bracket is not None and not is_one_sequence_between(leaf.opening_bracket, leaf, line.leaves)):
                    break
                inner_brackets.add(id(leaf))
        elif leaf.type in CLOSING_BRACKETS:
            prev = line.leaves[index - 1] if index > 0 else None
            if prev and prev.type in OPENING_BRACKETS:
                inner_brackets.add(id(leaf))
                continue
            if closing_bracket:
                omit.add(id(closing_bracket))
                omit.update(inner_brackets)
                inner_brackets.clear()
                yield omit
            if (prev and prev.type == token.COMMA and leaf.opening_bracket is not None and not is_one_sequence_between(leaf.opening_bracket, leaf, line.leaves)):
                break
            if leaf.value:
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf


def func_3fs4u0j4(line: Line, transform: Callable[[Line, Collection[Feature], Mode], Iterator[Line]], mode: Mode, features: Collection[Feature], *, line_str: str = '') -> List[Line]:
    if not line_str:
        line_str = line_to_string(line)
    result: List[Line] = []
    for transformed_line in transform(line, features, mode):
        if str(transformed_line).strip('\n') == line_str:
            raise CannotTransform('Line transformer returned an unchanged result')
        result.extend(func_36yftdol(transformed_line, mode=mode, features=features))
    features_set: Set[Feature] = set(features)
    if (Feature.FORCE_OPTIONAL_PARENTHESES in features_set or transform.__class__.__name__ != 'rhs' or not line.bracket_tracker.invisible or any(bracket.value for bracket in line.bracket_tracker.invisible) or
        func_d1qjl4yo.contains_multiline_strings() or result[0].contains_uncollapsable_type_comments() or result[0].contains_unsplittable_type_ignore() or is_line_short_enough(result[0], mode=mode) or any(leaf.parent is None for leaf in line.leaves)):
        return result
    line_copy: Line = func_d1qjl4yo.clone()
    append_leaves(line_copy, line, line.leaves)
    features_fop: Set[Feature] = features_set | {Feature.FORCE_OPTIONAL_PARENTHESES}
    second_opinion: List[Line] = func_3fs4u0j4(line_copy, transform, mode, features_fop, line_str=line_str)
    if all(is_line_short_enough(ln, mode=mode) for ln in second_opinion):
        result = second_opinion
    return result
