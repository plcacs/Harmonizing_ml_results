import re
import difflib
from typing import Any, Callable, Dict, Iterator, Iterable, List, Optional, Sequence, Tuple, NamedTuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes

LOG: logging.Logger = logging.getLogger(__name__)
DEBUG_DIFF_PARSER: bool = False
_INDENTATION_TOKENS: Tuple[str, ...] = ('INDENT', 'ERROR_DEDENT', 'DEDENT')
NEWLINE: int = PythonTokenTypes.NEWLINE
DEDENT: int = PythonTokenTypes.DEDENT
NAME: int = PythonTokenTypes.NAME
ERROR_DEDENT: int = PythonTokenTypes.ERROR_DEDENT
ENDMARKER: int = PythonTokenTypes.ENDMARKER


def _is_indentation_error_leaf(node: Any) -> bool:
    return node.type == 'error_leaf' and node.token_type in _INDENTATION_TOKENS


def _get_previous_leaf_if_indentation(leaf: Optional[Any]) -> Optional[Any]:
    while leaf and _is_indentation_error_leaf(leaf):
        leaf = leaf.get_previous_leaf()
    return leaf


def _get_next_leaf_if_indentation(leaf: Optional[Any]) -> Optional[Any]:
    while leaf and _is_indentation_error_leaf(leaf):
        leaf = leaf.get_next_leaf()
    return leaf


def _get_suite_indentation(tree_node: Any) -> int:
    return _get_indentation(tree_node.children[1])


def _get_indentation(tree_node: Any) -> int:
    return tree_node.start_pos[1]


def _assert_valid_graph(node: Any) -> None:
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


def _assert_nodes_are_equal(node1: Any, node2: Any) -> None:
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


def _get_debug_error_message(module: Any, old_lines: Sequence[str], new_lines: Sequence[str]) -> str:
    current_lines = split_lines(module.get_code(), keepends=True)
    current_diff = difflib.unified_diff(new_lines, current_lines)
    old_new_diff = difflib.unified_diff(old_lines, new_lines)
    import parso
    return "There's an issue with the diff parser. Please report (parso v%s) - Old/New:\n%s\nActual Diff (May be empty):\n%s" % (parso.__version__, ''.join(old_new_diff), ''.join(current_diff))


def _get_last_line(node_or_leaf: Any) -> int:
    last_leaf = node_or_leaf.get_last_leaf()
    if _ends_with_newline(last_leaf):
        return last_leaf.start_pos[0]
    else:
        n = last_leaf.get_next_leaf()
        if n.type == 'endmarker' and '\n' in n.prefix:
            return last_leaf.end_pos[0] + 1
        return last_leaf.end_pos[0]


def _skip_dedent_error_leaves(leaf: Optional[Any]) -> Optional[Any]:
    while leaf is not None and leaf.type == 'error_leaf' and (leaf.token_type == 'DEDENT'):
        leaf = leaf.get_previous_leaf()
    return leaf


def _ends_with_newline(leaf: Any, suffix: str = '') -> bool:
    leaf = _skip_dedent_error_leaves(leaf)
    if leaf.type == 'error_leaf':
        typ = leaf.token_type.lower()
    else:
        typ = leaf.type
    return typ == 'newline' or suffix.endswith('\n') or suffix.endswith('\r')


def _flows_finished(pgen_grammar: Any, stack: Any) -> bool:
    """
    if, while, for and try might not be finished, because another part might
    still be parsed.
    """
    for stack_node in stack:
        if stack_node.nonterminal in ('if_stmt', 'while_stmt', 'for_stmt', 'try_stmt'):
            return False
    return True


def _func_or_class_has_suite(node: Any) -> bool:
    if node.type == 'decorated':
        node = node.children[-1]
    if node.type in ('async_funcdef', 'async_stmt'):
        node = node.children[-1]
    return node.type in ('classdef', 'funcdef') and node.children[-1].type == 'suite'


def _suite_or_file_input_is_valid(pgen_grammar: Any, stack: Any) -> bool:
    if not _flows_finished(pgen_grammar, stack):
        return False
    for stack_node in reversed(stack):
        if stack_node.nonterminal == 'decorator':
            return False
        if stack_node.nonterminal == 'suite':
            return len(stack_node.nodes) > 1
    return True


def _is_flow_node(node: Any) -> bool:
    if node.type == 'async_stmt':
        node = node.children[1]
    try:
        value = node.children[0].value
    except AttributeError:
        return False
    return value in ('if', 'for', 'while', 'try', 'with')


class _PositionUpdatingFinished(Exception):
    pass


def _update_positions(nodes: Sequence[Any], line_offset: int, last_leaf: Any) -> None:
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

    def __init__(
        self,
        pgen_grammar: Any,
        tokenizer: Callable[..., Iterator[PythonToken]],
        module: Any
    ) -> None:
        self._pgen_grammar: Any = pgen_grammar
        self._tokenizer: Callable[..., Iterator[PythonToken]] = tokenizer
        self._module: Any = module

        self._copy_count: int = 0
        self._parser_count: int = 0
        self._nodes_tree: _NodesTree = _NodesTree(self._module)

        self._parser_lines_new: List[str] = []
        self._active_parser: Optional[Parser] = None
        self._replace_tos_indent: Optional[int] = None
        self._keyword_token_indents: Dict[Tuple[int, int], List[int]] = {}

    def _reset(self) -> None:
        self._copy_count = 0
        self._parser_count = 0
        self._nodes_tree = _NodesTree(self._module)

    def update(self, old_lines: Sequence[str], new_lines: Sequence[str]) -> Any:
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
        self._parser_lines_new = list(new_lines)
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

    def _enabled_debugging(self, old_lines: Sequence[str], lines_new: Sequence[str]) -> None:
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
                    copied_nodes: List[Any] = []
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

    def _get_old_line_stmt(self, old_line: int) -> Optional[Any]:
        leaf = self._module.get_leaf_for_position((old_line, 0), include_prefixes=True)
        if _ends_with_newline(leaf):
            leaf = leaf.get_next_leaf()
        if leaf.get_start_pos_of_prefix()[0] == old_line:
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

    def _try_parse_part(self, until_line: int) -> Any:
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

    def _diff_tokenize(self, lines: Sequence[str], until_line: int, line_offset: int = 0) -> Iterator[PythonToken]:
        was_newline = False
        indents = self._nodes_tree.indents
        initial_indentation_count = len(indents)
        tokens = self._tokenizer(lines, start_pos=(line_offset + 1, 0), indents=indents, is_first_token=line_offset == 0)
        stack = self._active_parser.stack  # type: ignore[union-attr]
        self._replace_tos_indent = None
        self._keyword_token_indents = {}
        for token in tokens:
            typ = token.type
            if typ == DEDENT:
                if len(indents) < initial_indentation_count:
                    while True:
                        typ, string, start_pos, prefix = token = next(tokens)
                        if typ in (DEDENT, ERROR_DEDENT):
                            if typ == ERROR_DEDENT:
                                self._replace_tos_indent = start_pos[1] + 1
                                pass
                        else:
                            break
                    if '\n' in prefix or '\r' in prefix:
                        prefix = re.sub('[^\\n\\r]+\\Z', '', prefix)
                    else:
                        assert start_pos[1] >= len(prefix), repr(prefix)
                        if start_pos[1] - len(prefix) == 0:
                            prefix = ''
                    yield PythonToken(ENDMARKER, '', start_pos, prefix)
                    break
            elif typ == NEWLINE and token.start_pos[0] >= until_line:
                was_newline = True
            elif was_newline:
                was_newline = False
                if len(indents) == initial_indentation_count:
                    if _suite_or_file_input_is_valid(self._pgen_grammar, stack):
                        yield PythonToken(ENDMARKER, '', token.start_pos, '')
                        break
            if typ == NAME and token.string in ('class', 'def'):
                self._keyword_token_indents[token.start_pos] = list(indents)
            yield token


class _NodesTreeNode:
    class _ChildrenGroup(NamedTuple):
        prefix: str
        children: List[Any]
        line_offset: int
        last_line_offset_leaf: Any

    def __init__(self, tree_node: Any, parent: Optional['._NodesTreeNode'] = None, indentation: int = 0) -> None:
        self.tree_node: Any = tree_node
        self._children_groups: List[_NodesTreeNode._ChildrenGroup] = []
        self.parent: Optional[_NodesTreeNode] = parent
        self._node_children: List[_NodesTreeNode] = []
        self.indentation: int = indentation

    def finish(self) -> None:
        children: List[Any] = []
        for prefix, children_part, line_offset, last_line_offset_leaf in self._children_groups:
            first_leaf = _get_next_leaf_if_indentation(children_part[0].get_first_leaf())
            first_leaf.prefix = prefix + first_leaf.prefix
            if line_offset != 0:
                try:
                    _update_positions(children_part, line_offset, last_line_offset_leaf)
                except _PositionUpdatingFinished:
                    pass
            children += children_part
        self.tree_node.children = children
        for node in children:
            node.parent = self.tree_node
        for node_child in self._node_children:
            node_child.finish()

    def add_child_node(self, child_node: '_NodesTreeNode') -> None:
        self._node_children.append(child_node)

    def add_tree_nodes(self, prefix: str, children: List[Any], line_offset: int = 0, last_line_offset_leaf: Optional[Any] = None) -> None:
        if last_line_offset_leaf is None:
            last_line_offset_leaf = children[-1].get_last_leaf()
        group = self._ChildrenGroup(prefix, children, line_offset, last_line_offset_leaf)
        self._children_groups.append(group)

    def get_last_line(self, suffix: str) -> int:
        line = 0
        if self._children_groups:
            children_group = self._children_groups[-1]
            last_leaf = _get_previous_leaf_if_indentation(children_group.last_line_offset_leaf)
            line = last_leaf.end_pos[0] + children_group.line_offset
            if _ends_with_newline(last_leaf, suffix):
                line -= 1
        line += len(split_lines(suffix)) - 1
        if suffix and (not suffix.endswith('\n')) and (not suffix.endswith('\r')):
            line += 1
        if self._node_children:
            return max(line, self._node_children[-1].get_last_line(suffix))
        return line

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.tree_node)


class _NodesTree:

    def __init__(self, module: Any) -> None:
        self._base_node: _NodesTreeNode = _NodesTreeNode(module)
        self._working_stack: List[_NodesTreeNode] = [self._base_node]
        self._module: Any = module
        self._prefix_remainder: str = ''
        self.prefix: str = ''
        self.indents: List[int] = [0]

    @property
    def parsed_until_line(self) -> int:
        return self._working_stack[-1].get_last_line(self.prefix)

    def _update_insertion_node(self, indentation: int) -> _NodesTreeNode:
        for node in reversed(list(self._working_stack)):
            if node.indentation < indentation or node is self._working_stack[0]:
                return node
            self._working_stack.pop()
        return self._working_stack[-1]

    def add_parsed_nodes(self, tree_nodes: List[Any], keyword_token_indents: Dict[Tuple[int, int], List[int]]) -> None:
        old_prefix = self.prefix
        tree_nodes = self._remove_endmarker(tree_nodes)
        if not tree_nodes:
            self.prefix = old_prefix + self.prefix
            return
        assert tree_nodes[0].type != 'newline'
        node = self._update_insertion_node(tree_nodes[0].start_pos[1])
        assert node.tree_node.type in ('suite', 'file_input')
        node.add_tree_nodes(old_prefix, tree_nodes)
        self._update_parsed_node_tos(tree_nodes[-1], keyword_token_indents)

    def _update_parsed_node_tos(self, tree_node: Any, keyword_token_indents: Dict[Tuple[int, int], List[int]]) -> None:
        if tree_node.type == 'suite':
            def_leaf = tree_node.parent.children[0]
            new_tos = _NodesTreeNode(tree_node, indentation=keyword_token_indents[def_leaf.start_pos][-1])
            new_tos.add_tree_nodes('', list(tree_node.children))
            self._working_stack[-1].add_child_node(new_tos)
            self._working_stack.append(new_tos)
            self._update_parsed_node_tos(tree_node.children[-1], keyword_token_indents)
        elif _func_or_class_has_suite(tree_node):
            self._update_parsed_node_tos(tree_node.children[-1], keyword_token_indents)

    def _remove_endmarker(self, tree_nodes: List[Any]) -> List[Any]:
        """
        Helps cleaning up the tree nodes that get inserted.
        """
        last_leaf = tree_nodes[-1].get_last_leaf()
        is_endmarker = last_leaf.type == 'endmarker'
        self._prefix_remainder = ''
        if is_endmarker:
            prefix = last_leaf.prefix
            separation = max(prefix.rfind('\n'), prefix.rfind('\r'))
            if separation > -1:
                last_leaf.prefix, self._prefix_remainder = (last_leaf.prefix[:separation + 1], last_leaf.prefix[separation + 1:])
        self.prefix = ''
        if is_endmarker:
            self.prefix = last_leaf.prefix
            tree_nodes = tree_nodes[:-1]
        return tree_nodes

    def _get_matching_indent_nodes(self, tree_nodes: List[Any], is_new_suite: bool) -> Iterator[Any]:
        node_iterator = iter(tree_nodes)
        if is_new_suite:
            yield next(node_iterator)
        first_node = next(node_iterator)
        indent = _get_indentation(first_node)
        if not is_new_suite and indent not in self.indents:
            return
        yield first_node
        for n in node_iterator:
            if _get_indentation(n) != indent:
                return
            yield n

    def copy_nodes(self, tree_nodes: List[Any], until_line: int, line_offset: int) -> List[Any]:
        """
        Copies tree nodes from the old parser tree.

        Returns the number of tree nodes that were copied.
        """
        if tree_nodes[0].type in ('error_leaf', 'error_node'):
            return []
        indentation = _get_indentation(tree_nodes[0])
        old_working_stack = list(self._working_stack)
        old_prefix = self.prefix
        old_indents = self.indents
        self.indents = [i for i in self.indents if i <= indentation]
        self._update_insertion_node(indentation)
        new_nodes, self._working_stack, self.prefix, added_indents = self._copy_nodes(list(self._working_stack), tree_nodes, until_line, line_offset, self.prefix)
        if new_nodes:
            self.indents += added_indents
        else:
            self._working_stack = old_working_stack
            self.prefix = old_prefix
            self.indents = old_indents
        return new_nodes

    def _copy_nodes(
        self,
        working_stack: List[_NodesTreeNode],
        nodes: List[Any],
        until_line: int,
        line_offset: int,
        prefix: str = '',
        is_nested: bool = False
    ) -> Tuple[List[Any], List[_NodesTreeNode], str, List[int]]:
        new_nodes: List[Any] = []
        added_indents: List[int] = []
        nodes = list(self._get_matching_indent_nodes(nodes, is_new_suite=is_nested))
        new_prefix = ''
        for node in nodes:
            if node.start_pos[0] > until_line:
                break
            if node.type == 'endmarker':
                break
            if node.type == 'error_leaf' and node.token_type in ('DEDENT', 'ERROR_DEDENT'):
                break
            if _get_last_line(node) > until_line:
                if _func_or_class_has_suite(node):
                    new_nodes.append(node)
                break
            try:
                c = node.children
            except AttributeError:
                pass
            else:
                n = node
                if n.type == 'decorated':
                    n = n.children[-1]
                if n.type in ('async_funcdef', 'async_stmt'):
                    n = n.children[-1]
                if n.type in ('classdef', 'funcdef'):
                    suite_node = n.children[-1]
                else:
                    suite_node = c[-1]
                if suite_node.type in ('error_leaf', 'error_node'):
                    break
            new_nodes.append(node)
        if new_nodes:
            while new_nodes:
                last_node = new_nodes[-1]
                if last_node.type in ('error_leaf', 'error_node') or _is_flow_node(new_nodes[-1]):
                    new_prefix = ''
                    new_nodes.pop()
                    while new_nodes:
                        last_node = new_nodes[-1]
                        if last_node.get_last_leaf().type == 'newline':
                            break
                        new_nodes.pop()
                    continue
                if len(new_nodes) > 1 and new_nodes[-2].type == 'error_node':
                    new_nodes.pop()
                    continue
                break
        if not new_nodes:
            return ([], working_stack, prefix, added_indents)
        tos = working_stack[-1]
        last_node = new_nodes[-1]
        had_valid_suite_last = False
        if _func_or_class_has_suite(last_node):
            suite = last_node
            while suite.type != 'suite':
                suite = suite.children[-1]
            indent = _get_suite_indentation(suite)
            added_indents.append(indent)
            suite_tos = _NodesTreeNode(suite, indentation=_get_indentation(last_node))
            suite_nodes, new_working_stack, new_prefix, ai = self._copy_nodes(working_stack + [suite_tos], suite.children, until_line, line_offset, is_nested=True)
            added_indents += ai
            if len(suite_nodes) < 2:
                new_nodes.pop()
                new_prefix = ''
            else:
                assert new_nodes
                tos.add_child_node(suite_tos)
                working_stack = new_working_stack
                had_valid_suite_last = True
        if new_nodes:
            if not _ends_with_newline(new_nodes[-1].get_last_leaf()) and (not had_valid_suite_last):
                p = new_nodes[-1].get_next_leaf().prefix
                new_prefix = split_lines(p, keepends=True)[0]
            if had_valid_suite_last:
                last = new_nodes[-1]
                if last.type == 'decorated':
                    last = last.children[-1]
                if last.type in ('async_funcdef', 'async_stmt'):
                    last = last.children[-1]
                last_line_offset_leaf = last.children[-2].get_last_leaf()
                assert last_line_offset_leaf == ':'
            else:
                last_line_offset_leaf = new_nodes[-1].get_last_leaf()
            tos.add_tree_nodes(prefix, new_nodes, line_offset, last_line_offset_leaf)
            prefix = new_prefix
            self._prefix_remainder = ''
        return (new_nodes, working_stack, prefix, added_indents)

    def close(self) -> None:
        self._base_node.finish()
        try:
            last_leaf = self._module.get_last_leaf()
        except IndexError:
            end_pos: List[int] = [1, 0]
        else:
            last_leaf = _skip_dedent_error_leaves(last_leaf)
            end_pos = list(last_leaf.end_pos)
        lines = split_lines(self.prefix)
        assert len(lines) > 0
        if len(lines) == 1:
            if lines[0].startswith(BOM_UTF8_STRING) and end_pos == [1, 0]:
                end_pos[1] -= 1
            end_pos[1] += len(lines[0])
        else:
            end_pos[0] += len(lines) - 1
            end_pos[1] = len(lines[-1])
        endmarker = EndMarker('', tuple(end_pos), self.prefix + self._prefix_remainder)
        endmarker.parent = self._module
        self._module.children.append(endmarker)