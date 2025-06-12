"""
The diff parser is trying to be a faster version of the normal parser by trying
to reuse the nodes of a previous pass over the same file. This is also called
incremental parsing in parser literature. The difference is mostly that with
incremental parsing you get a range that needs to be reparsed. Here we
calculate that range ourselves by using difflib. After that it's essentially
incremental parsing.

The biggest issue of this approach is that we reuse nodes in a mutable way. The
intial design and idea is quite problematic for this parser, but it is also
pretty fast. Measurements showed that just copying nodes in Python is simply
quite a bit slower (especially for big files >3 kLOC). Therefore we did not
want to get rid of the mutable nodes, since this is usually not an issue.

This is by far the hardest software I ever wrote, exactly because the initial
design is crappy. When you have to account for a lot of mutable state, it
creates a ton of issues that you would otherwise not have. This file took
probably 3-6 months to write, which is insane for a parser.

There is a fuzzer in that helps test this whole thing. Please use it if you
make changes here. If you run the fuzzer like::

    test/fuzz_diff_parser.py random -n 100000

you can be pretty sure that everything is still fine. I sometimes run the
fuzzer up to 24h to make sure everything is still ok.
"""
import re
import difflib
from collections import namedtuple
import logging
from typing import List, Tuple, Dict, Optional, Iterator, Any, Set, Union, NamedTuple, Sequence, cast
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker, Leaf, Node, BaseNode
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
from parso.python.tree import Module
from parso.grammar import Grammar
from parso.python.tokenize import PythonTokenizer
from parso.python.parser import StackNode

LOG = logging.getLogger(__name__)
DEBUG_DIFF_PARSER = False
_INDENTATION_TOKENS = ('INDENT', 'ERROR_DEDENT', 'DEDENT')
NEWLINE = PythonTokenTypes.NEWLINE
DEDENT = PythonTokenTypes.DEDENT
NAME = PythonTokenTypes.NAME
ERROR_DEDENT = PythonTokenTypes.ERROR_DEDENT
ENDMARKER = PythonTokenTypes.ENDMARKER

def _is_indentation_error_leaf(node: BaseNode) -> bool:
    return node.type == 'error_leaf' and node.token_type in _INDENTATION_TOKENS

def _get_previous_leaf_if_indentation(leaf: Optional[Leaf]) -> Optional[Leaf]:
    while leaf and _is_indentation_error_leaf(leaf):
        leaf = leaf.get_previous_leaf()
    return leaf

def _get_next_leaf_if_indentation(leaf: Optional[Leaf]) -> Optional[Leaf]:
    while leaf and _is_indentation_error_leaf(leaf):
        leaf = leaf.get_next_leaf()
    return leaf

def _get_suite_indentation(tree_node: Node) -> int:
    return _get_indentation(tree_node.children[1])

def _get_indentation(tree_node: BaseNode) -> int:
    return tree_node.start_pos[1]

def _assert_valid_graph(node: BaseNode) -> None:
    """
    Checks if the parent/children relationship is correct.

    This is a check that only runs during debugging/testing.
    """
    try:
        children = node.children
    except AttributeError:
        if node.type == 'error_leaf' and node.token_type in _INDENTATION_TOKENS:
            assert not node.value
            assert not node.prefix
            return
        previous_leaf = _get_previous_leaf_if_indentation(node.get_previous_leaf())
        if previous_leaf is None:
            content = node.prefix
            previous_start_pos = (1, 0)
        else:
            assert previous_leaf.end_pos <= node.start_pos, (previous_leaf, node)
            content = previous_leaf.value + node.prefix
            previous_start_pos = previous_leaf.start_pos
        if '\n' in content or '\r' in content:
            splitted = split_lines(content)
            line = previous_start_pos[0] + len(splitted) - 1
            actual = (line, len(splitted[-1]))
        else:
            actual = (previous_start_pos[0], previous_start_pos[1] + len(content))
            if content.startswith(BOM_UTF8_STRING) and node.get_start_pos_of_prefix() == (1, 0):
                actual = (actual[0], actual[1] - 1)
        assert node.start_pos == actual, (node.start_pos, actual)
    else:
        for child in children:
            assert child.parent == node, (node, child)
            _assert_valid_graph(child)

def _assert_nodes_are_equal(node1: BaseNode, node2: BaseNode) -> None:
    try:
        children1 = node1.children
    except AttributeError:
        assert not hasattr(node2, 'children'), (node1, node2)
        assert node1.value == node2.value, (node1, node2)
        assert node1.type == node2.type, (node1, node2)
        assert node1.prefix == node2.prefix, (node1, node2)
        assert node1.start_pos == node2.start_pos, (node1, node2)
        return
    else:
        try:
            children2 = node2.children
        except AttributeError:
            assert False, (node1, node2)
    for n1, n2 in zip(children1, children2):
        _assert_nodes_are_equal(n1, n2)
    assert len(children1) == len(children2), '\n' + repr(children1) + '\n' + repr(children2)

def _get_debug_error_message(module: Module, old_lines: List[str], new_lines: List[str]) -> str:
    current_lines = split_lines(module.get_code(), keepends=True)
    current_diff = difflib.unified_diff(new_lines, current_lines)
    old_new_diff = difflib.unified_diff(old_lines, new_lines)
    import parso
    return "There's an issue with the diff parser. Please report (parso v%s) - Old/New:\n%s\nActual Diff (May be empty):\n%s" % (parso.__version__, ''.join(old_new_diff), ''.join(current_diff))

def _get_last_line(node_or_leaf: BaseNode) -> int:
    last_leaf = node_or_leaf.get_last_leaf()
    if _ends_with_newline(last_leaf):
        return last_leaf.start_pos[0]
    else:
        n = last_leaf.get_next_leaf()
        if n.type == 'endmarker' and '\n' in n.prefix:
            return last_leaf.end_pos[0] + 1
        return last_leaf.end_pos[0]

def _skip_dedent_error_leaves(leaf: Optional[Leaf]) -> Optional[Leaf]:
    while leaf is not None and leaf.type == 'error_leaf' and (leaf.token_type == 'DEDENT'):
        leaf = leaf.get_previous_leaf()
    return leaf

def _ends_with_newline(leaf: Optional[Leaf], suffix: str = '') -> bool:
    if leaf is None:
        return False
    leaf = _skip_dedent_error_leaves(leaf)
    if leaf.type == 'error_leaf':
        typ = leaf.token_type.lower()
    else:
        typ = leaf.type
    return typ == 'newline' or suffix.endswith('\n') or suffix.endswith('\r')

def _flows_finished(pgen_grammar: Grammar, stack: List[StackNode]) -> bool:
    """
    if, while, for and try might not be finished, because another part might
    still be parsed.
    """
    for stack_node in stack:
        if stack_node.nonterminal in ('if_stmt', 'while_stmt', 'for_stmt', 'try_stmt'):
            return False
    return True

def _func_or_class_has_suite(node: Node) -> bool:
    if node.type == 'decorated':
        node = node.children[-1]
    if node.type in ('async_funcdef', 'async_stmt'):
        node = node.children[-1]
    return node.type in ('classdef', 'funcdef') and node.children[-1].type == 'suite'

def _suite_or_file_input_is_valid(pgen_grammar: Grammar, stack: List[StackNode]) -> bool:
    if not _flows_finished(pgen_grammar, stack):
        return False
    for stack_node in reversed(stack):
        if stack_node.nonterminal == 'decorator':
            return False
        if stack_node.nonterminal == 'suite':
            return len(stack_node.nodes) > 1
    return True

def _is_flow_node(node: BaseNode) -> bool:
    if node.type == 'async_stmt':
        node = node.children[1]
    try:
        value = node.children[0].value
    except AttributeError:
        return False
    return value in ('if', 'for', 'while', 'try', 'with')

class _PositionUpdatingFinished(Exception):
    pass

def _update_positions(nodes: List[BaseNode], line_offset: int, last_leaf: Leaf) -> None:
    for node in nodes:
        try:
            children = node.children
        except AttributeError:
            node.line += line_offset
            if node is last_leaf:
                raise _PositionUpdatingFinished
        else:
            _update_positions(children, line_offset, last_leaf)

class DiffParser:
    """
    An advanced form of parsing a file faster. Unfortunately comes with huge
    side effects. It changes the given module.
    """

    def __init__(self, pgen_grammar: Grammar, tokenizer: PythonTokenizer, module: Module) -> None:
        self._pgen_grammar = pgen_grammar
        self._tokenizer = tokenizer
        self._module = module
        self._parser_lines_new: List[str] = []
        self._active_parser: Optional[Parser] = None
        self._copy_count: int = 0
        self._parser_count: int = 0
        self._nodes_tree: Optional[_NodesTree] = None
        self._replace_tos_indent: Optional[int] = None
        self._keyword_token_indents: Dict[Tuple[int, int], List[int]] = {}

    def _reset(self) -> None:
        self._copy_count = 0
        self._parser_count = 0
        self._nodes_tree = _NodesTree(self._module)

    def update(self, old_lines: List[str], new_lines: List[str]) -> Module:
        """
        The algorithm works as follows:

        Equal:
            - Assure that the start is a newline, otherwise parse until we get
              one.
            - Copy from parsed_until_line + 1 to max(i2 + 1)
            - Make sure that the indentation is correct (e.g. add DEDENT)
            - Add old and change positions
        Insert:
            - Parse from parsed_until_line + 1 to min(j2 + 1), hopefully not
              much more.

        Returns the new module node.
        """
        LOG.debug('diff parser start')
        self._module._used_names = None
        self._parser_lines_new = new_lines
        self._reset()
        line_length = len(new_lines)
        sm = difflib.SequenceMatcher(None, old_lines, self._parser_lines_new)
        opcodes = sm.get_opcodes()
        LOG.debug('line_lengths old: %s; new: %s' % (len(old_lines), line_length))
        for operation, i1, i2, j1, j2 in opcodes:
            LOG.debug('-> code[%s] old[%s:%s] new[%s:%s]', operation, i1 + 1, i2, j1 + 1, j2)
            if j2 == line_length and new_lines[-1] == '':
                j2 -= 1
            if operation == 'equal':
                line_offset = j1 - i1
                self._copy_from_old_parser(line_offset, i1 + 1, i2, j2)
            elif operation == 'replace':
                self._parse(until_line=j2)
            elif operation == 'insert':
                self._parse(until_line=j2)
            else:
                assert operation == 'delete'
        assert self._nodes_tree is not None
        self._nodes_tree.close()
        if DEBUG_DIFF_PARSER:
            try:
                code = ''.join(new_lines)
                assert self._module.get_code() == code
                _assert_valid_graph(self._module)
                without_diff_parser_module = Parser(self._pgen_grammar, error_recovery=True).parse(self._tokenizer(new_lines))
                _assert_nodes_are_equal(self._module, without_diff_parser_module)
            except AssertionError:
                print(_get_debug_error_message(self._module, old_lines, new_lines))
                raise
        last_pos = self._module.end_pos[0]
        if last_pos != line_length:
            raise Exception('(%s != %s) ' % (last_pos, line_length) + _get_debug_error_message(self._module, old_lines, new_lines))
        LOG.debug('diff parser end')
        return self._module

    def _enabled_debugging(self, old_lines: List[str], lines_new: List[str]) -> None:
        if self._module.get_code() != ''.join(lines_new):
            LOG.warning('parser issue:\n%s\n%s', ''.join(old_lines), ''.join(lines_new))

    def _copy_from_old_parser(self, line_offset: int, start_line_old: int, until_line_old: int, until_line_new: int) -> None:
        last_until_line = -1
        while until_line_new > self._nodes_tree.parsed_until_line:
            parsed_until_line_old = self._nodes_tree.parsed_until_line - line_offset
            line_stmt = self._get_old_line_stmt(parsed_until_line_old + 1)
            if line_stmt is None:
                self._parse(self._nodes_tree.parsed_until_line + 1)
            else:
                p_children = line_stmt.parent.children
                index = p_children.index(line_stmt)
                if start_line_old == 1 and p_children[0].get_first_leaf().prefix.startswith(BOM_UTF8_STRING):
                    copied_nodes = []
                else:
                    from_ = self._nodes_tree.parsed_until_line + 1
                    copied_nodes = self._nodes_tree.copy_nodes(p_children[index:], until_line_old, line_offset)
                if copied_nodes:
                    self._copy_count += 1
                    to = self._nodes_tree.parsed_until_line
                    LOG.debug('copy old[%s:%s] new[%s:%s]', copied_nodes[0].start_pos[0], copied_nodes[-1].end_pos[0] - 1, from_, to)
                else:
                    self._parse(self._nodes_tree.parsed_until_line + 1)
            assert last_until_line != self._nodes_tree.parsed_until_line, last_until_line
            last_until_line = self._nodes_tree.parsed_until_line

    def _get_old_line_stmt(self, old_line: int) -> Optional[Node]:
        leaf = self._module.get_leaf_for_position((old_line, 0), include_prefixes=True)
        if _ends_with_newline(leaf):
            leaf = leaf.get_next_leaf()
        if leaf is None or leaf.get_start_pos_of_prefix()[0] != old_line:
            return None
        node = leaf
        while node.parent.type not in ('file_input', 'suite'):
            node = node.parent
        if node.start_pos[0] >= old_line:
            return node
        return None

    def _parse(self, until_line: int) -> None:
        """
        Parses at least until the given line, but might just parse more until a
        valid state is reached.
        """
        last_until_line = 0
        while until_line > self._nodes_tree.parsed_until_line:
            node = self._try_parse_part(until_line)
            nodes = node.children
            self._nodes_tree.add_parsed_nodes(nodes, self._keyword_token_indents)
            if self._replace_tos_indent is not None:
                self._nodes_tree.indents[-1] = self._replace_tos_indent
            LOG.debug('parse_part from %s to %s (to %s in part parser)', nodes[0].get_start_pos_of_prefix()[0], self._nodes_tree.parsed_until_line, node.end_pos[0] - 1)
            assert last_until_line != self._nodes_tree.parsed_until_line, last_until_line
            last_until_line = self._nodes_tree.parsed_until_line

    def _try_parse_part(self, until_line: int) -> Node:
        """
        Sets up a normal parser that uses a spezialized tokenizer to only parse
        until a certain position (or a bit longer if the statement hasn't
        ended.
        """
        self._parser_count += 1
        parsed_until_line = self._nodes_tree.parsed_until_line
        lines_after = self._parser_lines_new[parsed_until_line:]
        tokens = self._diff_tokenize(lines_after, until_line, line_offset=parsed_until_line)
        self._active_parser = Parser(self._pgen_grammar, error_recovery=True)
        return self._active_parser.parse(tokens=tokens)

    def _diff_tokenize(self, lines: List[str], until_line: int, line_offset: int = 0) -> Iterator[PythonToken]:
        was_newline = False
        assert self._nodes_tree is not None
        indents = self._nodes_tree.indents
        initial_indentation_count = len(indents)
        tokens = self._tokenizer(lines, start_pos=(line_offset + 1, 0), indents=indents, is_first_token=line_offset == 0)
        assert