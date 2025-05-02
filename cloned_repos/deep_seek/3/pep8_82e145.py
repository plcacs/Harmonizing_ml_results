import re
from contextlib import contextmanager
from typing import Tuple, List, Dict, Set, Optional, Union, Any, Iterator, Type, TypeVar, Generic, Sequence, cast
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope, Leaf, Node, BaseNode

_IMPORT_TYPES = ('import_name', 'import_from')
_SUITE_INTRODUCERS = ('classdef', 'funcdef', 'if_stmt', 'while_stmt', 'for_stmt', 'try_stmt', 'with_stmt')
_NON_STAR_TYPES = ('term', 'import_from', 'power')
_OPENING_BRACKETS = ('(', '[', '{')
_CLOSING_BRACKETS = (')', ']', '}')
_FACTOR = ('+', '-', '~')
_ALLOW_SPACE = ('*', '+', '-', '**', '/', '//', '@')
_BITWISE_OPERATOR = ('<<', '>>', '|', '&', '^')
_NEEDS_SPACE = ('=', '%', '->', '<', '>', '==', '>=', '<=', '<>', '!=', '+=', '-=', '*=', '@=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**=', '//=')
_NEEDS_SPACE += _BITWISE_OPERATOR
_IMPLICIT_INDENTATION_TYPES = ('dictorsetmaker', 'argument')
_POSSIBLE_SLICE_PARENTS = ('subscript', 'subscriptlist', 'sliceop')

class IndentationTypes:
    VERTICAL_BRACKET = object()
    HANGING_BRACKET = object()
    BACKSLASH = object()
    SUITE = object()
    IMPLICIT = object()

class IndentationNode:
    type: Any = IndentationTypes.SUITE

    def __init__(self, config: 'PEP8NormalizerConfig', indentation: str, parent: Optional['IndentationNode'] = None) -> None:
        self.bracket_indentation: str = self.indentation = indentation
        self.parent: Optional['IndentationNode'] = parent

    def __repr__(self) -> str:
        return '<%s>' % self.__class__.__name__

    def get_latest_suite_node(self) -> Optional['IndentationNode']:
        n = self
        while n is not None:
            if n.type == IndentationTypes.SUITE:
                return n
            n = n.parent
        return None

class BracketNode(IndentationNode):
    def __init__(self, config: 'PEP8NormalizerConfig', leaf: Leaf, parent: IndentationNode, in_suite_introducer: bool = False) -> None:
        self.leaf: Leaf = leaf
        previous_leaf: Leaf = leaf
        n: IndentationNode = parent
        if n.type == IndentationTypes.IMPLICIT:
            n = n.parent
        while True:
            if hasattr(n, 'leaf') and previous_leaf.line != n.leaf.line:
                break
            previous_leaf = previous_leaf.get_previous_leaf()
            if not isinstance(n, BracketNode) or previous_leaf != n.leaf:
                break
            n = n.parent
        parent_indentation: str = n.indentation
        next_leaf: Leaf = leaf.get_next_leaf()
        if '\n' in next_leaf.prefix or '\r' in next_leaf.prefix:
            self.bracket_indentation = parent_indentation + config.closing_bracket_hanging_indentation
            self.indentation = parent_indentation + config.indentation
            self.type = IndentationTypes.HANGING_BRACKET
        else:
            expected_end_indent: int = leaf.end_pos[1]
            if '\t' in config.indentation:
                self.indentation = None
            else:
                self.indentation = ' ' * expected_end_indent
            self.bracket_indentation = self.indentation
            self.type = IndentationTypes.VERTICAL_BRACKET
        if in_suite_introducer and parent.type == IndentationTypes.SUITE and (self.indentation == parent_indentation + config.indentation):
            self.indentation += config.indentation
            self.bracket_indentation = self.indentation
        self.parent = parent

class ImplicitNode(BracketNode):
    def __init__(self, config: 'PEP8NormalizerConfig', leaf: Leaf, parent: IndentationNode) -> None:
        super().__init__(config, leaf, parent)
        self.type = IndentationTypes.IMPLICIT
        next_leaf: Leaf = leaf.get_next_leaf()
        if leaf == ':' and '\n' not in next_leaf.prefix and ('\r' not in next_leaf.prefix):
            self.indentation += ' '

class BackslashNode(IndentationNode):
    type = IndentationTypes.BACKSLASH

    def __init__(self, config: 'PEP8NormalizerConfig', parent_indentation: str, containing_leaf: Leaf, spacing: Any, parent: Optional[IndentationNode] = None) -> None:
        expr_stmt: Optional[Node] = containing_leaf.search_ancestor('expr_stmt')
        if expr_stmt is not None:
            equals: Leaf = expr_stmt.children[-2]
            if '\t' in config.indentation:
                self.indentation = None
            elif equals.end_pos == spacing.start_pos:
                self.indentation = parent_indentation + config.indentation
            else:
                self.indentation = ' ' * (equals.end_pos[1] + 1)
        else:
            self.indentation = parent_indentation + config.indentation
        self.bracket_indentation = self.indentation
        self.parent = parent

def _is_magic_name(name: Leaf) -> bool:
    return name.value.startswith('__') and name.value.endswith('__')

class PEP8Normalizer(ErrorFinder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._previous_part: Optional[Leaf] = None
        self._previous_leaf: Optional[Leaf] = None
        self._on_newline: bool = True
        self._newline_count: int = 0
        self._wanted_newline_count: Optional[int] = None
        self._max_new_lines_in_prefix: int = 0
        self._new_statement: bool = True
        self._implicit_indentation_possible: bool = False
        self._indentation_tos: IndentationNode = self._last_indentation_tos = IndentationNode(self._config, indentation='')
        self._in_suite_introducer: bool = False
        if ' ' in self._config.indentation:
            self._indentation_type = 'spaces'
            self._wrong_indentation_char = '\t'
        else:
            self._indentation_type = 'tabs'
            self._wrong_indentation_char = ' '

    @contextmanager
    def visit_node(self, node: Node) -> Iterator[None]:
        with super().visit_node(node):
            with self._visit_node(node):
                yield

    @contextmanager
    def _visit_node(self, node: Node) -> Iterator[None]:
        typ = node.type
        if typ in 'import_name':
            names: List[Leaf] = node.get_defined_names()
            if len(names) > 1:
                for name in names[:1]:
                    self.add_issue(name, 401, 'Multiple imports on one line')
        elif typ == 'lambdef':
            expr_stmt: Node = node.parent
            if expr_stmt.type == 'expr_stmt' and any((n.type == 'name' for n in expr_stmt.children[:-2:2])):
                self.add_issue(node, 731, 'Do not assign a lambda expression, use a def')
        elif typ == 'try_stmt':
            for child in node.children:
                if child.type == 'keyword' and child.value == 'except':
                    self.add_issue(child, 722, 'Do not use bare except, specify exception instead')
        elif typ == 'comparison':
            for child in node.children:
                if child.type not in ('atom_expr', 'power'):
                    continue
                if len(child.children) > 2:
                    continue
                trailer: Node = child.children[1]
                atom: Node = child.children[0]
                if trailer.type == 'trailer' and atom.type == 'name' and (atom.value == 'type'):
                    self.add_issue(node, 721, "Do not compare types, use 'isinstance()")
                    break
        elif typ == 'file_input':
            endmarker: Leaf = node.children[-1]
            prev: Optional[Leaf] = endmarker.get_previous_leaf()
            prefix: str = endmarker.prefix
            if not prefix.endswith('\n') and (not prefix.endswith('\r')) and (prefix or prev is None or prev.value not in {'\n', '\r\n', '\r'}):
                self.add_issue(endmarker, 292, 'No newline at end of file')
        if typ in _IMPORT_TYPES:
            simple_stmt: Node = node.parent
            module: Node = simple_stmt.parent
            if module.type == 'file_input':
                index: int = module.children.index(simple_stmt)
                for child in module.children[:index]:
                    children: List[Node] = [child]
                    if child.type == 'simple_stmt':
                        children = child.children[:-1]
                    found_docstring: bool = False
                    for c in children:
                        if c.type == 'string' and (not found_docstring):
                            continue
                        found_docstring = True
                        if c.type == 'expr_stmt' and all((_is_magic_name(n) for n in c.get_defined_names()):
                            continue
                        if c.type in _IMPORT_TYPES or isinstance(c, Flow):
                            continue
                        self.add_issue(node, 402, 'Module level import not at top of file')
                        break
                    else:
                        continue
                    break
        implicit_indentation_possible: bool = typ in _IMPLICIT_INDENTATION_TYPES
        in_introducer: bool = typ in _SUITE_INTRODUCERS
        if in_introducer:
            self._in_suite_introducer = True
        elif typ == 'suite':
            if self._indentation_tos.type == IndentationTypes.BACKSLASH:
                self._indentation_tos = self._indentation_tos.parent
            self._indentation_tos = IndentationNode(self._config, self._indentation_tos.indentation + self._config.indentation, parent=self._indentation_tos)
        elif implicit_indentation_possible:
            self._implicit_indentation_possible = True
        yield
        if typ == 'suite':
            assert self._indentation_tos.type == IndentationTypes.SUITE
            self._indentation_tos = self._indentation_tos.parent
            self._wanted_newline_count = None
        elif implicit_indentation_possible:
            self._implicit_indentation_possible = False
            if self._indentation_tos.type == IndentationTypes.IMPLICIT:
                self._indentation_tos = self._indentation_tos.parent
        elif in_introducer:
            self._in_suite_introducer = False
            if typ in ('classdef', 'funcdef'):
                self._wanted_newline_count = self._get_wanted_blank_lines_count()

    def _check_tabs_spaces(self, spacing: Leaf) -> bool:
        if self._wrong_indentation_char in spacing.value:
            self.add_issue(spacing, 101, 'Indentation contains ' + self._indentation_type)
            return True
        return False

    def _get_wanted_blank_lines_count(self) -> int:
        suite_node: Optional[IndentationNode] = self._indentation_tos.get_latest_suite_node()
        return int(suite_node.parent is None) + 1

    def _reset_newlines(self, spacing: Leaf, leaf: Leaf, is_comment: bool = False) -> None:
        self._max_new_lines_in_prefix = max(self._max_new_lines_in_prefix, self._newline_count)
        wanted: Optional[int] = self._wanted_newline_count
        if wanted is not None:
            blank_lines: int = self._newline_count - 1
            if wanted > blank_lines and leaf.type != 'endmarker':
                if not is_comment:
                    code: int = 302 if wanted == 2 else 301
                    message: str = 'expected %s blank line, found %s' % (wanted, blank_lines)
                    self.add_issue(spacing, code, message)
                    self._wanted_newline_count = None
            else:
                self._wanted_newline_count = None
        if not is_comment:
            wanted = self._get_wanted_blank_lines_count()
            actual: int = self._max_new_lines_in_prefix - 1
            val: str = leaf.value
            needs_lines: bool = val == '@' and leaf.parent.type == 'decorator' or ((val == 'class' or (val == 'async' and leaf.get_next_leaf() == 'def') or (val == 'def' and self._previous_leaf != 'async')) and leaf.parent.parent.type != 'decorated')
            if needs_lines and actual < wanted:
                func_or_cls: Node = leaf.parent
                suite: Node = func_or_cls.parent
                if suite.type == 'decorated':
                    suite = suite.parent
                if suite.children[int(suite.type == 'suite')] != func_or_cls:
                    code = 302 if wanted == 2 else 301
                    message = 'expected %s blank line, found %s' % (wanted, actual)
                    self.add_issue(spacing, code, message)
            self._max_new_lines_in_prefix = 0
        self._newline_count = 0

    def visit_leaf(self, leaf: Leaf) -> str:
        super().visit_leaf(leaf)
        for part in leaf._split_prefix():
            if part.type == 'spacing':
                break
            self._visit_part(part, part.create_spacing_part(), leaf)
        self._analyse_non_prefix(leaf)
        self._visit_part(leaf, part, leaf)
        self._last_indentation_tos = self._indentation_tos
        self._new_statement = leaf.type == 'newline'
        if leaf.type == 'newline' and self._indentation_tos.type == IndentationTypes.BACKSLASH:
            self._indentation_tos = self._indentation_tos.parent
        if leaf.value == ':' and leaf.parent.type in _SUITE_INTRODUCERS:
            self._in_suite_introducer = False
        elif leaf.value == 'elif':
            self._in_suite_introducer = True
        if not self._new_statement:
            self._reset_newlines(part, leaf)
            self._max_blank_lines = 0
        self._previous_leaf = leaf
        return leaf.value

    def _visit_part(self, part: Leaf, spacing: Any, leaf: Leaf) -> None:
        value: str = part.value
        type_: str = part.type
        if type_ == 'error_leaf':
            return
        if value == ',' and part.parent.type == 'dictorsetmaker':
            self._indentation_tos = self._indentation_tos.parent
        node: IndentationNode = self._indentation_tos
        if type_ == 'comment':
            if value.startswith('##'):
                if value.lstrip('#'):
                    self.add_issue(part, 266, "Too many leading '#' for block comment.")
            elif self._on_newline:
                if not re.match('#:? ', value) and (not value == '#') and (not (value.startswith('#!') and part.start_pos == (1, 0))):
                    self.add_issue(part, 265, "Block comment should start with '# '")
            elif not re.match('#:? [^ ]', value):
                self.add_issue(part, 262, "Inline comment should start with '# '")
            self._reset_newlines(spacing, leaf, is_comment=True)
        elif type_ == 'newline':
            if self._newline_count > self._get_wanted_blank_lines_count():
                self.add_issue(part, 303, 'Too many blank lines (%s)' % self._newline_count)
            elif leaf in ('def', 'class') and leaf.parent.parent.type == 'decorated':
                self.add_issue(part, 304, 'Blank lines found after function decorator')
            self._newline_count += 1
        if type_ == 'backslash':
            if node.type != IndentationTypes.BACKSLASH:
                if node.type != IndentationTypes.SUITE:
                    self.add_issue(part, 502, 'The backslash is redundant between brackets')
                else:
                    indentation: str = node.indentation
                    if self._in_suite_introducer and node.type == IndentationTypes.SUITE:
                        indentation += self._config.indentation
                    self._indentation_tos = BackslashNode(self._config, indentation, part, spacing, parent=self._indentation_tos)
        elif self._on_newline:
            indentation: str = spacing.value
            if node.type == IndentationTypes.BACKSLASH and self._previous_part.type == 'newline':
                self._indentation_tos = self._indentation_tos.parent
            if not self._check_tabs_spaces(spacing):
                should_be_indentation: str = node.indentation
                if type_ == 'comment':
                    n: IndentationNode = self._last_indentation_tos
                    while True:
                        if len(indentation) > len(n.indentation):
                            break
                        should_be_indentation = n.indentation
                        self._last_indentation_tos = n
                        if n == node:
                            break
                        n = n.parent
                if self._new_statement:
                    if type_ == 'newline':
                        if indentation:
                            self.add_issue(spacing, 291, 'Trailing whitespace')
                    elif indentation != should_be_indentation:
                        s: str = '%s %s' % (len(self._config.indentation), self