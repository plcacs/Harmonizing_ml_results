import codecs
import warnings
import re
from contextlib import contextmanager
from typing import List, Dict, Optional, Generator, Tuple, Any, Iterator, Union
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
from parso.tree import Node, Leaf

_BLOCK_STMTS: Tuple[str, ...] = ('if_stmt', 'while_stmt', 'for_stmt', 'try_stmt', 'with_stmt')
_STAR_EXPR_PARENTS: Tuple[str, ...] = ('testlist_star_expr', 'testlist_comp', 'exprlist')
_MAX_BLOCK_SIZE: int = 20
_MAX_INDENT_COUNT: int = 100
ALLOWED_FUTURES: Tuple[str, ...] = (
    'nested_scopes', 'generators', 'division', 'absolute_import',
    'with_statement', 'print_function', 'unicode_literals', 'generator_stop'
)
_COMP_FOR_TYPES: Tuple[str, ...] = ('comp_for', 'sync_comp_for')


def _get_rhs_name(node: Node, version: Tuple[int, int]) -> str:
    type_ = node.type
    if type_ == 'lambdef':
        return 'lambda'
    elif type_ == 'atom':
        comprehension = _get_comprehension_type(node)
        first, second = node.children[:2]
        if comprehension is not None:
            return comprehension
        elif second.type == 'dictorsetmaker':
            if version < (3, 8):
                return 'literal'
            elif second.children[1] == ':' or second.children[0] == '**':
                return 'dict display'
            else:
                return 'set display'
        elif first == '(' and (second == ')' or (len(node.children) == 3 and node.children[1].type == 'testlist_comp')):
            return 'tuple'
        elif first == '(':
            return _get_rhs_name(_remove_parens(node), version=version)
        elif first == '[':
            return 'list'
        elif first == '{' and second == '}':
            return 'dict display'
        elif first == '{' and len(node.children) > 2:
            return 'set display'
    elif type_ == 'keyword':
        if 'yield' in node.value:
            return 'yield expression'
        if version < (3, 8):
            return 'keyword'
        else:
            return str(node.value)
    elif type_ == 'operator' and node.value == '...':
        return 'Ellipsis'
    elif type_ == 'comparison':
        return 'comparison'
    elif type_ in ('string', 'number', 'strings'):
        return 'literal'
    elif type_ == 'yield_expr':
        return 'yield expression'
    elif type_ == 'test':
        return 'conditional expression'
    elif type_ in ('atom_expr', 'power'):
        if node.children[0] == 'await':
            return 'await expression'
        elif node.children[-1].type == 'trailer':
            trailer = node.children[-1]
            if trailer.children[0] == '(':
                return 'function call'
            elif trailer.children[0] == '[':
                return 'subscript'
            elif trailer.children[0] == '.':
                return 'attribute'
    elif 'expr' in type_ and 'star_expr' not in type_ or '_test' in type_ or type_ in ('term', 'factor'):
        return 'operator'
    elif type_ == 'star_expr':
        return 'starred'
    elif type_ == 'testlist_star_expr':
        return 'tuple'
    elif type_ == 'fstring':
        return 'f-string expression'
    return type_


def _iter_stmts(scope: Node) -> Iterator[Node]:
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


def _get_comprehension_type(atom: Node) -> Optional[str]:
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


def _is_future_import(import_from: Node) -> bool:
    from_names = import_from.get_from_names()
    return [n.value for n in from_names] == ['__future__']


def _remove_parens(atom: Node) -> Node:
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


def _skip_parens_bottom_up(node: Node) -> Optional[Node]:
    """
    Returns an ancestor node of an expression, skipping all levels of parens
    bottom-up.
    """
    while node.parent is not None:
        node = node.parent
        if node.type != 'atom' or node.children[0] != '(':
            return node
    return None


def _iter_params(parent_node: Node) -> Iterator[Node]:
    return (n for n in parent_node.children if n.type == 'param' or n.type == 'operator')


def _is_future_import_first(import_from: Node) -> bool:
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
    return False


def _iter_definition_exprs_from_lists(exprlist: Node) -> Iterator[Node]:
    
    def check_expr(child: Node) -> Iterator[Node]:
        if child.type == 'atom':
            if child.children[0] == '(':
                testlist_comp = child.children[1]
                if testlist_comp.type == 'testlist_comp':
                    yield from _iter_definition_exprs_from_lists(testlist_comp)
                    return
                else:
                    yield from check_expr(testlist_comp)
                    return
            elif child.children[0] == '[':
                yield child.children[1]
                return
        yield child
    
    if exprlist.type in _STAR_EXPR_PARENTS:
        for child in exprlist.children[::2]:
            yield from check_expr(child)
    else:
        yield from check_expr(exprlist)


def _get_expr_stmt_definition_exprs(expr_stmt: Node) -> List[Node]:
    exprs: List[Node] = []
    for list_ in expr_stmt.children[:-2:2]:
        if list_.type in ('testlist_star_expr', 'testlist'):
            exprs += list(_iter_definition_exprs_from_lists(list_))
        else:
            exprs.append(list_)
    return exprs


def _get_for_stmt_definition_exprs(for_stmt: Node) -> List[Node]:
    exprlist = for_stmt.children[1]
    return list(_iter_definition_exprs_from_lists(exprlist))


def _is_argument_comprehension(argument: Node) -> bool:
    return argument.children[1].type in _COMP_FOR_TYPES


def _any_fstring_error(version: Tuple[int, int], node: Optional[Node]) -> bool:
    if version < (3, 9) or node is None:
        return False
    if node.type == 'error_node':
        return any(child.type == 'fstring_start' for child in node.children)
    elif node.type == 'fstring':
        return True
    else:
        return bool(node.search_ancestor('fstring'))


class _Context:
    
    def __init__(self, node: Node, add_syntax_error: Any, parent_context: Optional['_Context'] = None) -> None:
        self.node: Node = node
        self.blocks: List[Node] = []
        self.parent_context: Optional['_Context'] = parent_context
        self._used_name_dict: Dict[str, List[Node]] = {}
        self._global_names: List[Node] = []
        self._local_params_names: List[str] = []
        self._nonlocal_names: List[Node] = []
        self._nonlocal_names_in_subscopes: List[Node] = []
        self._add_syntax_error = add_syntax_error
    
    def is_async_funcdef(self) -> bool:
        return self.is_function() and self.node.parent.type in ('async_funcdef', 'async_stmt')
    
    def is_function(self) -> bool:
        return self.node.type == 'funcdef'
    
    def add_name(self, name: Node) -> None:
        parent_type = name.parent.type
        if parent_type == 'trailer':
            return
        if parent_type == 'global_stmt':
            self._global_names.append(name)
        elif parent_type == 'nonlocal_stmt':
            self._nonlocal_names.append(name)
        elif parent_type == 'funcdef':
            self._local_params_names.extend(param.name.value for param in name.parent.get_params())
        else:
            self._used_name_dict.setdefault(name.value, []).append(name)
    
    def finalize(self) -> List[Node]:
        """
        Returns a list of nonlocal names that need to be part of that scope.
        """
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
        nonlocals_not_handled: List[Node] = []
        for nonlocal_name in self._nonlocal_names_in_subscopes:
            search = nonlocal_name.value
            if search in self._local_params_names:
                continue
            if search in global_name_strs or self.parent_context is None:
                message = "no binding for nonlocal '%s' found" % nonlocal_name.value
                self._add_syntax_error(nonlocal_name, message)
            elif not self.is_function() or nonlocal_name.value not in self._used_name_dict:
                nonlocals_not_handled.append(nonlocal_name)
        return self._nonlocal_names + nonlocals_not_handled
    
    def _analyze_names(self, globals_or_nonlocals: List[Node], type_: str) -> None:
    
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
                    if parent.type == 'expr_stmt' and parent.children[1].type == 'annassign':
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
    def add_block(self, node: Node) -> Iterator[None]:
        self.blocks.append(node)
        yield
        self.blocks.pop()
    
    def add_context(self, node: Node) -> ' _Context':
        return _Context(node, self._add_syntax_error, parent_context=self)
    
    def close_child_context(self, child_context: ' _Context') -> None:
        self._nonlocal_names_in_subscopes += child_context.finalize()


class ErrorFinder(Normalizer):
    """
    Searches for errors in the syntax tree.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._error_dict: Dict[int, Tuple[int, str, Node]] = {}
        self.version: Tuple[int, int] = self.grammar.version_info

    def initialize(self, node: Node) -> None:

        def create_context(node: Optional[Node]) -> Optional[_Context]:
            if node is None:
                return None
            parent_context = create_context(node.parent)
            if node.type in ('classdef', 'funcdef', 'file_input'):
                return _Context(node, self._add_syntax_error, parent_context)
            return parent_context
        self.context: _Context = create_context(node) or _Context(node, self._add_syntax_error)
        self._indentation_count: int = 0

    def visit(self, node: Node) -> str:
        if node.type == 'error_node':
            with self.visit_node(node):
                return ''
        return super().visit(node)

    @contextmanager
    def visit_node(self, node: Node) -> Iterator[None]:
        self._check_type_rules(node)
        if node.type in _BLOCK_STMTS:
            with self.context.add_block(node):
                if len(self.context.blocks) == _MAX_BLOCK_SIZE:
                    self._add_syntax_error(node, 'too many statically nested blocks')
                yield
            return
        elif node.type == 'suite':
            self._indentation_count += 1
            if self._indentation_count == _MAX_INDENT_COUNT:
                self._add_indentation_error(node.children[1], 'too many levels of indentation')
        yield
        if node.type == 'suite':
            self._indentation_count -= 1
        elif node.type in ('classdef', 'funcdef'):
            context = self.context
            self.context = context.parent_context
            if context.parent_context:
                self.context.close_child_context(context)

    def visit_leaf(self, leaf: Leaf) -> str:
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
                    match = re.match(r'\w{,2}("{1,3}|\'{1,3})', leaf.value)
                    if match is None:
                        message = 'invalid syntax'
                        if self.version >= (3, 9) and leaf.value in _get_token_collection(self.version).always_break_tokens:
                            message = 'f-string: ' + message
                    elif len(match.group(1)) == 1:
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

    def _add_indentation_error(self, spacing: Any, message: str) -> None:
        self.add_issue(spacing, 903, 'IndentationError: ' + message)

    def _add_syntax_error(self, node: Node, message: str) -> None:
        self.add_issue(node, 901, 'SyntaxError: ' + message)

    def add_issue(self, node: Node, code: int, message: str) -> None:
        line = node.start_pos[0]
        args: Tuple[int, str, Node] = (code, message, node)
        self._error_dict.setdefault(line, args)

    def finalize(self) -> None:
        self.context.finalize()
        for code, message, node in self._error_dict.values():
            self.issues.append(Issue(node, code, message))


class IndentationRule(Rule):
    code: int = 903

    def _get_message(self, message: str, node: Node) -> str:
        message = super()._get_message(message, node)
        return 'IndentationError: ' + message


@ErrorFinder.register_rule(type='error_node')
class _ExpectIndentedBlock(IndentationRule):
    message: str = 'expected an indented block'

    def get_node(self, node: Node) -> Node:
        leaf: Leaf = node.get_next_leaf()
        return list(leaf._split_prefix())[-1]

    def is_issue(self, node: Node) -> bool:
        return node.children[-1].type == 'newline'


class ErrorFinderConfig(NormalizerConfig):
    normalizer_class = ErrorFinder


class SyntaxRule(Rule):
    code: int = 901

    def _get_message(self, message: str, node: Node) -> str:
        message = super()._get_message(message, node)
        if 'f-string' not in message and _any_fstring_error(self._normalizer.version, node):
            message = 'f-string: ' + message
        return 'SyntaxError: ' + message


@ErrorFinder.register_rule(type='error_node')
class _InvalidSyntaxRule(SyntaxRule):
    message: str = 'invalid syntax'
    fstring_message: str = 'f-string: invalid syntax'

    def get_node(self, node: Node) -> Node:
        return node.get_next_leaf()

    def is_issue(self, node: Node) -> bool:
        error = node.get_next_leaf().type != 'error_leaf'
        if error and _any_fstring_error(self._normalizer.version, node):
            self.add_issue(node, message=self.fstring_message)
        else:
            return error


@ErrorFinder.register_rule(value='await')
class _AwaitOutsideAsync(SyntaxRule):
    message: str = "'await' outside async function"

    def is_issue(self, leaf: Leaf) -> bool:
        return not self._normalizer.context.is_async_funcdef()

    def get_error_node(self, node: Node) -> Node:
        return node.parent


@ErrorFinder.register_rule(value='break')
class _BreakOutsideLoop(SyntaxRule):
    message: str = "'break' outside loop"

    def is_issue(self, leaf: Leaf) -> bool:
        in_loop = False
        for block in self._normalizer.context.blocks:
            if block.type in ('for_stmt', 'while_stmt'):
                in_loop = True
        return not in_loop


@ErrorFinder.register_rule(value='continue')
class _ContinueChecks(SyntaxRule):
    message: str = "'continue' not properly in loop"
    message_in_finally: str = "'continue' not supported inside 'finally' clause"

    def is_issue(self, leaf: Leaf) -> bool:
        in_loop = False
        for block in self._normalizer.context.blocks:
            if block.type in ('for_stmt', 'while_stmt'):
                in_loop = True
            if block.type == 'try_stmt':
                last_block = block.children[-3]
                if last_block == 'finally' and leaf.start_pos > last_block.start_pos and (self._normalizer.version < (3, 8)):
                    self.add_issue(leaf, message=self.message_in_finally)
                    return False
        if not in_loop:
            return True
        return False


@ErrorFinder.register_rule(value='from')
class _YieldFromCheck(SyntaxRule):
    message: str = "'yield from' inside async function"

    def get_node(self, leaf: Leaf) -> Node:
        return leaf.parent.parent

    def is_issue(self, leaf: Leaf) -> bool:
        return leaf.parent.type == 'yield_arg' and self._normalizer.context.is_async_funcdef()


@ErrorFinder.register_rule(type='name')
class _NameChecks(SyntaxRule):
    message: str = 'cannot assign to __debug__'
    message_none: str = 'cannot assign to None'

    def is_issue(self, leaf: Leaf) -> bool:
        self._normalizer.context.add_name(leaf)
        if leaf.value == '__debug__' and leaf.is_definition():
            return True
        return False


@ErrorFinder.register_rule(type='string')
class _StringChecks(SyntaxRule):
    message: str = 'bytes can only contain ASCII literal characters.'

    def is_issue(self, leaf: Leaf) -> bool:
        string_prefix = leaf.string_prefix.lower()
        if 'b' in string_prefix and any(ord(c) > 127 for c in leaf.value):
            return True
        if 'r' not in string_prefix:
            payload = leaf._get_payload()
            if 'b' in string_prefix:
                payload = payload.encode('utf-8')
                func = codecs.escape_decode
            else:
                func = codecs.unicode_escape_decode
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    func(payload)
            except UnicodeDecodeError as e:
                self.add_issue(leaf, message='(unicode error) ' + str(e))
            except ValueError as e:
                self.add_issue(leaf, message='(value error) ' + str(e))
        return False


@ErrorFinder.register_rule(value='*')
class _StarCheck(SyntaxRule):
    message: str = 'named arguments must follow bare *'

    def is_issue(self, leaf: Leaf) -> bool:
        params = leaf.parent
        if params.type == 'parameters' and params:
            after = params.children[params.children.index(leaf) + 1:]
            after = [child for child in after if child not in (',', ')') and (not hasattr(child, 'star_count') or not child.star_count)]
            return len(after) == 0
        return False


@ErrorFinder.register_rule(value='**')
class _StarStarCheck(SyntaxRule):
    message: str = 'dict unpacking cannot be used in dict comprehension'

    def is_issue(self, leaf: Leaf) -> bool:
        if leaf.parent.type == 'dictorsetmaker':
            comp_for = leaf.get_next_sibling().get_next_sibling()
            return comp_for is not None and comp_for.type in _COMP_FOR_TYPES
        return False


@ErrorFinder.register_rule(value='yield')
@ErrorFinder.register_rule(value='return')
class _ReturnAndYieldChecks(SyntaxRule):
    message: str = "'return' with value in async generator"
    message_async_yield: str = "'yield' inside async function"

    def get_node(self, leaf: Leaf) -> Node:
        return leaf.parent

    def is_issue(self, leaf: Leaf) -> bool:
        if self._normalizer.context.node.type != 'funcdef':
            self.add_issue(self.get_node(leaf), message=f"'{leaf.value}' outside function")
            return False
        elif self._normalizer.context.is_async_funcdef() and any(self._normalizer.context.node.iter_yield_exprs()):
            if leaf.value == 'return' and leaf.parent.type == 'return_stmt':
                return True
        return False


@ErrorFinder.register_rule(type='strings')
class _BytesAndStringMix(SyntaxRule):
    message: str = 'cannot mix bytes and nonbytes literals'

    def _is_bytes_literal(self, string: Node) -> bool:
        if string.type == 'fstring':
            return False
        return 'b' in string.string_prefix.lower()

    def is_issue(self, node: Node) -> bool:
        first = node.children[0]
        first_is_bytes = self._is_bytes_literal(first)
        for string in node.children[1:]:
            if first_is_bytes != self._is_bytes_literal(string):
                return True
        return False


@ErrorFinder.register_rule(type='import_as_names')
class _TrailingImportComma(SyntaxRule):
    message: str = 'trailing comma not allowed without surrounding parentheses'

    def is_issue(self, node: Node) -> bool:
        if node.children[-1] == ',' and node.parent.children[-1] != ')':
            return True
        return False


@ErrorFinder.register_rule(type='import_from')
class _ImportStarInFunction(SyntaxRule):
    message: str = 'import * only allowed at module level'

    def is_issue(self, node: Node) -> bool:
        return node.is_star_import() and self._normalizer.context.parent_context is not None


@ErrorFinder.register_rule(type='import_from')
class _FutureImportRule(SyntaxRule):
    message: str = 'from __future__ imports must occur at the beginning of the file'

    def is_issue(self, node: Node) -> bool:
        if _is_future_import(node):
            if not _is_future_import_first(node):
                return True
            for from_name, future_name in node.get_paths():
                name = future_name.value
                allowed_futures = list(ALLOWED_FUTURES)
                if self._normalizer.version >= (3, 7):
                    allowed_futures.append('annotations')
                if name == 'braces':
                    self.add_issue(node, message='not a chance')
                elif name == 'barry_as_FLUFL':
                    m = "Seriously I'm not implementing this :) ~ Dave"
                    self.add_issue(node, message=m)
                elif name not in allowed_futures:
                    message = f'future feature {name} is not defined'
                    self.add_issue(node, message=message)
        return False


@ErrorFinder.register_rule(type='star_expr')
class _StarExprRule(SyntaxRule):
    message_iterable_unpacking: str = 'iterable unpacking cannot be used in comprehension'

    def is_issue(self, node: Node) -> bool:

        def check_delete_starred(node: Node) -> bool:
            while node.parent is not None:
                node = node.parent
                if node.type == 'del_stmt':
                    return True
                if node.type not in (*_STAR_EXPR_PARENTS, 'atom'):
                    return False
            return False

        if self._normalizer.version >= (3, 9):
            ancestor = node.parent
        else:
            ancestor = _skip_parens_bottom_up(node)
        if ancestor.type not in (*_STAR_EXPR_PARENTS, 'dictorsetmaker') and not (ancestor.type == 'atom' and ancestor.children[0] != '('):
            self.add_issue(node, message="can't use starred expression here")
            return True
        if check_delete_starred(node):
            if self._normalizer.version >= (3, 9):
                self.add_issue(node, message='cannot delete starred')
            else:
                self.add_issue(node, message="can't use starred expression here")
            return True
        if node.parent.type == 'testlist_comp':
            if node.parent.children[1].type in _COMP_FOR_TYPES:
                self.add_issue(node, message=self.message_iterable_unpacking)
                return True
        return False


@ErrorFinder.register_rule(types=_STAR_EXPR_PARENTS)
class _StarExprParentRule(SyntaxRule):

    def is_issue(self, node: Node) -> bool:

        def is_definition(node: Node, ancestor: Optional[Node]) -> bool:
            if ancestor is None:
                return False
            type_ = ancestor.type
            if type_ == 'trailer':
                return False
            if type_ == 'expr_stmt':
                return node.start_pos < ancestor.children[-1].start_pos
            return is_definition(node, ancestor.parent)
        
        if is_definition(node, node.parent):
            args = [c for c in node.children if c != ',']
            starred = [c for c in args if c.type == 'star_expr']
            if len(starred) > 1:
                if self._normalizer.version < (3, 9):
                    message = 'two starred expressions in assignment'
                else:
                    message = 'multiple starred expressions in assignment'
                self.add_issue(starred[1], message=message)
            elif starred:
                count = args.index(starred[0])
                if count >= 256:
                    message = 'too many expressions in star-unpacking assignment'
                    self.add_issue(starred[0], message=message)
        return False


@ErrorFinder.register_rule(type='annassign')
class _AnnotatorRule(SyntaxRule):
    message: str = 'illegal target for annotation'

    def get_node(self, node: Node) -> Node:
        return node.parent

    def is_issue(self, node: Node) -> bool:
        type_: Optional[str] = None
        lhs = node.parent.children[0]
        lhs = _remove_parens(lhs)
        try:
            children = lhs.children
        except AttributeError:
            pass
        else:
            if ',' in children or (lhs.type == 'atom' and children[0] == '('):
                type_ = 'tuple'
            elif lhs.type == 'atom' and children[0] == '[':
                type_ = 'list'
            trailer = children[-1]
        if type_ is None:
            if not (lhs.type == 'name' or (lhs.type in ('atom_expr', 'power') and trailer.type == 'trailer' and trailer.children[0] != '(')):
                self.add_issue(lhs.parent, message=self.message)
                return True
        else:
            message = f'only single target (not {type_}) can be annotated'
            self.add_issue(lhs.parent, message=message)
            return True
        return False


@ErrorFinder.register_rule(type='argument')
class _ArgumentRule(SyntaxRule):

    def is_issue(self, node: Node) -> bool:
        first = node.children[0]
        if self._normalizer.version < (3, 8):
            first = _remove_parens(first)
        if node.children[1] == '=' and first.type != 'name':
            if first.type == 'lambdef':
                if self._normalizer.version < (3, 8):
                    message = 'lambda cannot contain assignment'
                else:
                    message = 'expression cannot contain assignment, perhaps you meant "=="?'
            elif self._normalizer.version < (3, 8):
                message = "keyword can't be an expression"
            else:
                message = 'expression cannot contain assignment, perhaps you meant "=="?'
            self.add_issue(first, message=message)
        if _is_argument_comprehension(node) and node.parent.type == 'classdef':
            self.add_issue(node, message='invalid syntax')
        return False


@ErrorFinder.register_rule(type='nonlocal_stmt')
class _NonlocalModuleLevelRule(SyntaxRule):
    message: str = 'nonlocal declaration not allowed at module level'

    def is_issue(self, node: Node) -> bool:
        return self._normalizer.context.parent_context is None


@ErrorFinder.register_rule(type='arglist')
class _ArglistRule(SyntaxRule):

    @property
    def message(self) -> str:
        if self._normalizer.version < (3, 7):
            return 'Generator expression must be parenthesized if not sole argument'
        else:
            return 'Generator expression must be parenthesized'

    def is_issue(self, node: Node) -> bool:
        arg_set: set = set()
        kw_only: bool = False
        kw_unpacking_only: bool = False
        for argument in node.children:
            if argument == ',':
                continue
            if argument.type == 'argument':
                first = argument.children[0]
                if _is_argument_comprehension(argument) and len(node.children) >= 2:
                    self.add_issue(argument, message=self.message)
                    return True
                if first in ('*', '**'):
                    if first == '*':
                        if kw_unpacking_only:
                            message = 'iterable argument unpacking follows keyword argument unpacking'
                            self.add_issue(argument, message=message)
                    else:
                        kw_unpacking_only = True
                else:
                    kw_only = True
                    if first.type == 'name':
                        if first.value in arg_set:
                            message = 'keyword argument repeated'
                            if self._normalizer.version >= (3, 9):
                                message += f': {first.value}'
                            self.add_issue(first, message=message)
                        else:
                            arg_set.add(first.value)
            elif kw_unpacking_only:
                message = 'positional argument follows keyword argument unpacking'
                self.add_issue(argument, message=message)
            elif kw_only:
                message = 'positional argument follows keyword argument'
                self.add_issue(argument, message=message)
        return False


@ErrorFinder.register_rule(type='parameters')
@ErrorFinder.register_rule(type='lambdef')
class _ParameterRule(SyntaxRule):
    message: str = 'non-default argument follows default argument'

    def is_issue(self, node: Node) -> bool:
        param_names: set = set()
        default_only: bool = False
        star_seen: bool = False
        for p in _iter_params(node):
            if p.type == 'operator':
                if p.value == '*':
                    star_seen = True
                    default_only = False
                continue
            if p.name.value in param_names:
                message = f"duplicate argument '{p.name.value}' in function definition"
                self.add_issue(p.name, message=message)
            param_names.add(p.name.value)
            if not star_seen:
                if p.default is None and not p.star_count:
                    if default_only:
                        self.add_issue(p, message=self.message)
                        return True
                elif p.star_count:
                    star_seen = True
                    default_only = False
                else:
                    default_only = True
        return False


@ErrorFinder.register_rule(type='try_stmt')
class _TryStmtRule(SyntaxRule):
    message: str = "default 'except:' must be last"

    def is_issue(self, try_stmt: Node) -> bool:
        default_except: Optional[Node] = None
        for except_clause in try_stmt.children[3::3]:
            if except_clause in ('else', 'finally'):
                break
            if except_clause == 'except':
                default_except = except_clause
            elif default_except is not None:
                self.add_issue(default_except, message=self.message)
        return False


@ErrorFinder.register_rule(type='fstring')
class _FStringRule(SyntaxRule):
    _fstring_grammar: Optional[Any] = None
    message_expr: str = 'f-string expression part cannot include a backslash'
    message_nested: str = 'f-string: expressions nested too deeply'
    message_conversion: str = "f-string: invalid conversion character: expected 's', 'r', or 'a'"

    def _check_format_spec(self, format_spec: Node, depth: int) -> None:
        self._check_fstring_contents(format_spec.children[1:], depth)

    def _check_fstring_expr(self, fstring_expr: Node, depth: int) -> None:
        if depth >= 2:
            self.add_issue(fstring_expr, message=self.message_nested)
        expr = fstring_expr.children[1]
        if '\\' in expr.get_code():
            self.add_issue(expr, message=self.message_expr)
        children_2 = fstring_expr.children[2]
        if children_2.type == 'operator' and children_2.value == '=':
            conversion = fstring_expr.children[3]
        else:
            conversion = children_2
        if conversion.type == 'fstring_conversion':
            name = conversion.children[1]
            if name.value not in ('s', 'r', 'a'):
                self.add_issue(name, message=self.message_conversion)
        format_spec = fstring_expr.children[-2]
        if format_spec.type == 'fstring_format_spec':
            self._check_format_spec(format_spec, depth + 1)

    def is_issue(self, fstring: Node) -> bool:
        self._check_fstring_contents(fstring.children[1:-1])
        return False

    def _check_fstring_contents(self, children: List[Node], depth: int = 0) -> None:
        for fstring_content in children:
            if fstring_content.type == 'fstring_expr':
                self._check_fstring_expr(fstring_content, depth)


class _CheckAssignmentRule(SyntaxRule):

    def _check_assignment(self, node: Node, is_deletion: bool = False, is_namedexpr: bool = False, is_aug_assign: bool = False) -> None:
        error: Optional[str] = None
        type_ = node.type
        if type_ == 'lambdef':
            error = 'lambda'
        elif type_ == 'atom':
            first, second = node.children[:2]
            error = _get_comprehension_type(node)
            if error is None:
                if second.type == 'dictorsetmaker':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    elif second.children[1] == ':':
                        error = 'dict display'
                    else:
                        error = 'set display'
                elif first == '{' and second == '}':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'dict display'
                elif first == '{' and len(node.children) > 2:
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'set display'
                elif first in ('(', '['):
                    if second.type == 'yield_expr':
                        error = 'yield expression'
                    elif second.type == 'testlist_comp':
                        if is_namedexpr:
                            if first == '(':
                                error = 'tuple'
                            elif first == '[':
                                error = 'list'
                        for child in second.children[::2]:
                            self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
                    else:
                        self._check_assignment(second, is_deletion, is_namedexpr, is_aug_assign)
        elif type_ == 'keyword':
            if node.value == 'yield':
                error = 'yield expression'
            elif self._normalizer.version < (3, 8):
                error = 'keyword'
            else:
                error = str(node.value)
        elif type_ == 'operator':
            if node.value == '...':
                error = 'Ellipsis'
        elif type_ == 'comparison':
            error = 'comparison'
        elif type_ in ('string', 'number', 'strings'):
            error = 'literal'
        elif type_ == 'yield_expr':
            message = 'assignment to yield expression not possible'
            self.add_issue(node, message=message)
        elif type_ == 'test':
            error = 'conditional expression'
        elif type_ in ('atom_expr', 'power'):
            if node.children[0] == 'await':
                error = 'await expression'
            elif node.children[-2] == '**':
                error = 'operator'
            else:
                trailer = node.children[-1]
                assert trailer.type == 'trailer'
                if trailer.children[0] == '(':
                    error = 'function call'
                elif is_namedexpr and trailer.children[0] == '[':
                    error = 'subscript'
                elif is_namedexpr and trailer.children[0] == '.':
                    error = 'attribute'
        elif type_ == 'fstring':
            if self._normalizer.version < (3, 8):
                error = 'literal'
            else:
                error = 'f-string expression'
        elif type_ in ('testlist_star_expr', 'exprlist', 'testlist'):
            for child in node.children[::2]:
                self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
        elif 'expr' in type_ and type_ != 'star_expr' or '_test' in type_ or type_ in ('term', 'factor'):
            error = 'operator'
        elif type_ == 'star_expr':
            if is_deletion:
                if self._normalizer.version >= (3, 9):
                    error = 'starred'
                else:
                    self.add_issue(node, message="can't use starred expression here")
            else:
                if self._normalizer.version >= (3, 9):
                    ancestor = node.parent
                else:
                    ancestor = _skip_parens_bottom_up(node)
                if ancestor.type not in _STAR_EXPR_PARENTS and not is_aug_assign and not (ancestor.type == 'atom' and ancestor.children[0] == '['):
                    message = 'starred assignment target must be in a list or tuple'
                    self.add_issue(node, message=message)
            self._check_assignment(node.children[1])
        if error is not None:
            if is_namedexpr:
                message = f'cannot use assignment expressions with {error}'
            else:
                cannot = "can't" if self._normalizer.version < (3, 8) else 'cannot'
                msg_action = 'delete' if is_deletion else 'assign to'
                message = f"{cannot} {msg_action} {error}"
            self.add_issue(node, message=message)

    
@ErrorFinder.register_rule(type='sync_comp_for')
class _CompForRule(_CheckAssignmentRule):
    message: str = 'asynchronous comprehension outside of an asynchronous function'

    def is_issue(self, node: Node) -> bool:
        expr_list = node.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)
        return node.parent.children[0] == 'async' and not self._normalizer.context.is_async_funcdef()


@ErrorFinder.register_rule(type='expr_stmt')
class _ExprStmtRule(_CheckAssignmentRule):
    message: str = 'illegal expression for augmented assignment'
    extended_message: str = "'{target}' is an " + message

    def is_issue(self, node: Node) -> bool:
        augassign = node.children[1]
        is_aug_assign = augassign != '=' and augassign.type != 'annassign'
        if self._normalizer.version <= (3, 8) or not is_aug_assign:
            for before_equal in node.children[:-2:2]:
                self._check_assignment(before_equal, is_aug_assign=is_aug_assign)
        if is_aug_assign:
            target = _remove_parens(node.children[0])
            if target.type == 'name' or (target.type in ('atom_expr', 'power') and target.children[1].type == 'trailer' and target.children[-1].children[0] != '('):
                return False
            if self._normalizer.version <= (3, 8):
                return True
            else:
                rhs_name = _get_rhs_name(node.children[0], self._normalizer.version)
                message = self.extended_message.format(target=rhs_name)
                self.add_issue(node, message=message)
                return True
        return False


@ErrorFinder.register_rule(type='with_item')
class _WithItemRule(_CheckAssignmentRule):

    def is_issue(self, with_item: Node) -> bool:
        self._check_assignment(with_item.children[2])
        return False


@ErrorFinder.register_rule(type='del_stmt')
class _DelStmtRule(_CheckAssignmentRule):

    def is_issue(self, del_stmt: Node) -> bool:
        child = del_stmt.children[1]
        if child.type != 'expr_list':
            self._check_assignment(child, is_deletion=True)
        return False


@ErrorFinder.register_rule(type='expr_list')
class _ExprListRule(_CheckAssignmentRule):

    def is_issue(self, expr_list: Node) -> bool:
        for expr in expr_list.children[::2]:
            self._check_assignment(expr)
        return False


@ErrorFinder.register_rule(type='for_stmt')
class _ForStmtRule(_CheckAssignmentRule):

    def is_issue(self, for_stmt: Node) -> bool:
        expr_list = for_stmt.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)
        return False


@ErrorFinder.register_rule(type='namedexpr_test')
class _NamedExprRule(_CheckAssignmentRule):

    def is_issue(self, namedexpr_test: Node) -> bool:
        first = namedexpr_test.children[0]

        def search_namedexpr_in_comp_for(node: Node) -> Optional[Node]:
            while True:
                parent = node.parent
                if parent is None:
                    return parent
                if parent.type == 'sync_comp_for' and parent.children[3] == node:
                    return parent
                node = parent

        if search_namedexpr_in_comp_for(namedexpr_test):
            message = 'assignment expression cannot be used in a comprehension iterable expression'
            self.add_issue(namedexpr_test, message=message)
        exprlist: List[Node] = []

        def process_comp_for(comp_for: Node) -> None:
            if comp_for.type == 'sync_comp_for':
                comp = comp_for
            elif comp_for.type == 'comp_for':
                comp = comp_for.children[1]
            exprlist.extend(_get_for_stmt_definition_exprs(comp))

        def search_all_comp_ancestors(node: Node) -> bool:
            has_ancestors = False
            while True:
                node = node.search_ancestor('testlist_comp', 'dictorsetmaker')
                if node is None:
                    break
                for child in node.children:
                    if child.type in _COMP_FOR_TYPES:
                        process_comp_for(child)
                        has_ancestors = True
                        break
            return has_ancestors

        search_all = search_all_comp_ancestors(namedexpr_test)
        if search_all:
            if self._normalizer.context.node.type == 'classdef':
                message = 'assignment expression within a comprehension cannot be used in a class body'
                self.add_issue(namedexpr_test, message=message)
            namelist = [expr.value for expr in exprlist if expr.type == 'name']
            if first.type == 'name' and first.value in namelist:
                message = f"assignment expression cannot rebind comprehension iteration variable {first.value!r}"
                self.add_issue(namedexpr_test, message=message)
        self._check_assignment(first, is_namedexpr=True)
        return False


@ErrorFinder.register_rule(type='strings')
class _BytesAndStringMix(SyntaxRule):
    message: str = 'cannot mix bytes and nonbytes literals'

    def _is_bytes_literal(self, string: Node) -> bool:
        if string.type == 'fstring':
            return False
        return 'b' in string.string_prefix.lower()

    def is_issue(self, node: Node) -> bool:
        first = node.children[0]
        first_is_bytes = self._is_bytes_literal(first)
        for string in node.children[1:]:
            if first_is_bytes != self._is_bytes_literal(string):
                return True
        return False


@ErrorFinder.register_rule(type='import_as_names')
class _TrailingImportComma(SyntaxRule):
    message: str = 'trailing comma not allowed without surrounding parentheses'

    def is_issue(self, node: Node) -> bool:
        if node.children[-1] == ',' and node.parent.children[-1] != ')':
            return True
        return False


@ErrorFinder.register_rule(type='import_from')
class _ImportStarInFunction(SyntaxRule):
    message: str = 'import * only allowed at module level'

    def is_issue(self, node: Node) -> bool:
        return node.is_star_import() and self._normalizer.context.parent_context is not None


@ErrorFinder.register_rule(type='import_from')
class _FutureImportRule(SyntaxRule):
    message: str = 'from __future__ imports must occur at the beginning of the file'

    def is_issue(self, node: Node) -> bool:
        if _is_future_import(node):
            if not _is_future_import_first(node):
                return True
            for from_name, future_name in node.get_paths():
                name = future_name.value
                allowed_futures = list(ALLOWED_FUTURES)
                if self._normalizer.version >= (3, 7):
                    allowed_futures.append('annotations')
                if name == 'braces':
                    self.add_issue(node, message='not a chance')
                elif name == 'barry_as_FLUFL':
                    m = "Seriously I'm not implementing this :) ~ Dave"
                    self.add_issue(node, message=m)
                elif name not in allowed_futures:
                    message = f'future feature {name} is not defined'
                    self.add_issue(node, message=message)
        return False


@ErrorFinder.register_rule(type='star_expr')
class _StarExprRule(SyntaxRule):
    message_iterable_unpacking: str = 'iterable unpacking cannot be used in comprehension'

    def is_issue(self, node: Node) -> bool:

        def check_delete_starred(node: Node) -> bool:
            while node.parent is not None:
                node = node.parent
                if node.type == 'del_stmt':
                    return True
                if node.type not in (*_STAR_EXPR_PARENTS, 'atom'):
                    return False
            return False

        if self._normalizer.version >= (3, 9):
            ancestor = node.parent
        else:
            ancestor = _skip_parens_bottom_up(node)
        if ancestor.type not in (*_STAR_EXPR_PARENTS, 'dictorsetmaker') and not (ancestor.type == 'atom' and ancestor.children[0] != '('):
            self.add_issue(node, message="can't use starred expression here")
            return True
        if check_delete_starred(node):
            if self._normalizer.version >= (3, 9):
                self.add_issue(node, message='cannot delete starred')
            else:
                self.add_issue(node, message="can't use starred expression here")
            return True
        if node.parent.type == 'testlist_comp':
            if node.parent.children[1].type in _COMP_FOR_TYPES:
                self.add_issue(node, message=self.message_iterable_unpacking)
                return True
        return False


@ErrorFinder.register_rule(types=_STAR_EXPR_PARENTS)
class _StarExprParentRule(SyntaxRule):

    def is_issue(self, node: Node) -> bool:

        def is_definition(node: Node, ancestor: Optional[Node]) -> bool:
            if ancestor is None:
                return False
            type_ = ancestor.type
            if type_ == 'trailer':
                return False
            if type_ == 'expr_stmt':
                return node.start_pos < ancestor.children[-1].start_pos
            return is_definition(node, ancestor.parent)
        
        if is_definition(node, node.parent):
            args = [c for c in node.children if c != ',']
            starred = [c for c in args if c.type == 'star_expr']
            if len(starred) > 1:
                if self._normalizer.version < (3, 9):
                    message = 'two starred expressions in assignment'
                else:
                    message = 'multiple starred expressions in assignment'
                self.add_issue(starred[1], message=message)
            elif starred:
                count = args.index(starred[0])
                if count >= 256:
                    message = 'too many expressions in star-unpacking assignment'
                    self.add_issue(starred[0], message=message)
        return False


@ErrorFinder.register_rule(type='annassign')
class _AnnotatorRule(SyntaxRule):
    message: str = 'illegal target for annotation'

    def get_node(self, node: Node) -> Node:
        return node.parent

    def is_issue(self, node: Node) -> bool:
        type_: Optional[str] = None
        lhs = node.parent.children[0]
        lhs = _remove_parens(lhs)
        try:
            children = lhs.children
        except AttributeError:
            pass
        else:
            if ',' in children or (lhs.type == 'atom' and children[0] == '('):
                type_ = 'tuple'
            elif lhs.type == 'atom' and children[0] == '[':
                type_ = 'list'
            trailer = children[-1]
        if type_ is None:
            if not (lhs.type == 'name' or (lhs.type in ('atom_expr', 'power') and trailer.type == 'trailer' and trailer.children[0] != '(')):
                self.add_issue(lhs.parent, message=self.message)
                return True
        else:
            message = f'only single target (not {type_}) can be annotated'
            self.add_issue(lhs.parent, message=message)
            return True
        return False


@ErrorFinder.register_rule(type='argument')
class _ArgumentRule(SyntaxRule):

    def is_issue(self, node: Node) -> bool:
        first = node.children[0]
        if self._normalizer.version < (3, 8):
            first = _remove_parens(first)
        if node.children[1] == '=' and first.type != 'name':
            if first.type == 'lambdef':
                if self._normalizer.version < (3, 8):
                    message = 'lambda cannot contain assignment'
                else:
                    message = 'expression cannot contain assignment, perhaps you meant "=="?'
            elif self._normalizer.version < (3, 8):
                message = "keyword can't be an expression"
            else:
                message = 'expression cannot contain assignment, perhaps you meant "=="?'
            self.add_issue(first, message=message)
        if _is_argument_comprehension(node) and node.parent.type == 'classdef':
            self.add_issue(node, message='invalid syntax')
        return False


@ErrorFinder.register_rule(type='nonlocal_stmt')
class _NonlocalModuleLevelRule(SyntaxRule):
    message: str = 'nonlocal declaration not allowed at module level'

    def is_issue(self, node: Node) -> bool:
        return self._normalizer.context.parent_context is None


@ErrorFinder.register_rule(type='arglist')
class _ArglistRule(SyntaxRule):

    @property
    def message(self) -> str:
        if self._normalizer.version < (3, 7):
            return 'Generator expression must be parenthesized if not sole argument'
        else:
            return 'Generator expression must be parenthesized'

    def is_issue(self, node: Node) -> bool:
        arg_set: set = set()
        kw_only: bool = False
        kw_unpacking_only: bool = False
        for argument in node.children:
            if argument == ',':
                continue
            if argument.type == 'argument':
                first = argument.children[0]
                if _is_argument_comprehension(argument) and len(node.children) >= 2:
                    return True
                if first in ('*', '**'):
                    if first == '*':
                        if kw_unpacking_only:
                            message = 'iterable argument unpacking follows keyword argument unpacking'
                            self.add_issue(argument, message=message)
                    else:
                        kw_unpacking_only = True
                else:
                    kw_only = True
                    if first.type == 'name':
                        if first.value in arg_set:
                            message = 'keyword argument repeated'
                            if self._normalizer.version >= (3, 9):
                                message += f': {first.value}'
                            self.add_issue(first, message=message)
                        else:
                            arg_set.add(first.value)
            elif kw_unpacking_only:
                message = 'positional argument follows keyword argument unpacking'
                self.add_issue(argument, message=message)
            elif kw_only:
                message = 'positional argument follows keyword argument'
                self.add_issue(argument, message=message)
        return False


@ErrorFinder.register_rule(type='parameters')
@ErrorFinder.register_rule(type='lambdef')
class _ParameterRule(SyntaxRule):
    message: str = 'non-default argument follows default argument'

    def is_issue(self, node: Node) -> bool:
        param_names: set = set()
        default_only: bool = False
        star_seen: bool = False
        for p in _iter_params(node):
            if p.type == 'operator':
                if p.value == '*':
                    star_seen = True
                    default_only = False
                continue
            if p.name.value in param_names:
                message = f"duplicate argument '{p.name.value}' in function definition"
                self.add_issue(p.name, message=message)
            param_names.add(p.name.value)
            if not star_seen:
                if p.default is None and not p.star_count:
                    if default_only:
                        self.add_issue(p, message=self.message)
                        return True
                elif p.star_count:
                    star_seen = True
                    default_only = False
                else:
                    default_only = True
        return False


@ErrorFinder.register_rule(type='try_stmt')
class _TryStmtRule(SyntaxRule):
    message: str = "default 'except:' must be last"

    def is_issue(self, try_stmt: Node) -> bool:
        default_except: Optional[Node] = None
        for except_clause in try_stmt.children[3::3]:
            if except_clause in ('else', 'finally'):
                break
            if except_clause == 'except':
                default_except = except_clause
            elif default_except is not None:
                self.add_issue(default_except, message=self.message)
        return False


@ErrorFinder.register_rule(type='fstring')
class _FStringRule(SyntaxRule):
    _fstring_grammar: Optional[Any] = None
    message_expr: str = 'f-string expression part cannot include a backslash'
    message_nested: str = 'f-string: expressions nested too deeply'
    message_conversion: str = "f-string: invalid conversion character: expected 's', 'r', or 'a'"

    def _check_format_spec(self, format_spec: Node, depth: int) -> None:
        self._check_fstring_contents(format_spec.children[1:], depth)

    def _check_fstring_expr(self, fstring_expr: Node, depth: int) -> None:
        if depth >= 2:
            self.add_issue(fstring_expr, message=self.message_nested)
        expr = fstring_expr.children[1]
        if '\\' in expr.get_code():
            self.add_issue(expr, message=self.message_expr)
        children_2 = fstring_expr.children[2]
        if children_2.type == 'operator' and children_2.value == '=':
            conversion = fstring_expr.children[3]
        else:
            conversion = children_2
        if conversion.type == 'fstring_conversion':
            name = conversion.children[1]
            if name.value not in ('s', 'r', 'a'):
                self.add_issue(name, message=self.message_conversion)
        format_spec = fstring_expr.children[-2]
        if format_spec.type == 'fstring_format_spec':
            self._check_format_spec(format_spec, depth + 1)

    def is_issue(self, fstring: Node) -> bool:
        self._check_fstring_contents(fstring.children[1:-1])
        return False


class _CheckAssignmentRule(SyntaxRule):

    def _check_assignment(self, node: Node, is_deletion: bool = False, is_namedexpr: bool = False, is_aug_assign: bool = False) -> None:
        error: Optional[str] = None
        type_ = node.type
        if type_ == 'lambdef':
            error = 'lambda'
        elif type_ == 'atom':
            first, second = node.children[:2]
            error = _get_comprehension_type(node)
            if error is None:
                if second.type == 'dictorsetmaker':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    elif second.children[1] == ':':
                        error = 'dict display'
                    else:
                        error = 'set display'
                elif first == '{' and second == '}':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'dict display'
                elif first == '{' and len(node.children) > 2:
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'set display'
                elif first in ('(', '['):
                    if second.type == 'yield_expr':
                        error = 'yield expression'
                    elif second.type == 'testlist_comp':
                        if is_namedexpr:
                            if first == '(':
                                error = 'tuple'
                            elif first == '[':
                                error = 'list'
                        for child in second.children[::2]:
                            self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
                    else:
                        self._check_assignment(second, is_deletion, is_namedexpr, is_aug_assign)
        elif type_ == 'keyword':
            if node.value == 'yield':
                error = 'yield expression'
            elif self._normalizer.version < (3, 8):
                error = 'keyword'
            else:
                error = str(node.value)
        elif type_ == 'operator':
            if node.value == '...':
                error = 'Ellipsis'
        elif type_ == 'comparison':
            error = 'comparison'
        elif type_ in ('string', 'number', 'strings'):
            error = 'literal'
        elif type_ == 'yield_expr':
            message = 'assignment to yield expression not possible'
            self.add_issue(node, message=message)
        elif type_ == 'test':
            error = 'conditional expression'
        elif type_ in ('atom_expr', 'power'):
            if node.children[0] == 'await':
                error = 'await expression'
            elif node.children[-2] == '**':
                error = 'operator'
            else:
                trailer = node.children[-1]
                assert trailer.type == 'trailer'
                if trailer.children[0] == '(':
                    error = 'function call'
                elif is_namedexpr and trailer.children[0] == '[':
                    error = 'subscript'
                elif is_namedexpr and trailer.children[0] == '.':
                    error = 'attribute'
        elif type_ == 'fstring':
            if self._normalizer.version < (3, 8):
                error = 'literal'
            else:
                error = 'f-string expression'
        elif type_ in ('testlist_star_expr', 'exprlist', 'testlist'):
            for child in node.children[::2]:
                self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
        elif 'expr' in type_ and type_ != 'star_expr' or '_test' in type_ or type_ in ('term', 'factor'):
            error = 'operator'
        elif type_ == 'star_expr':
            if is_deletion:
                if self._normalizer.version >= (3, 9):
                    error = 'starred'
                else:
                    self.add_issue(node, message="can't use starred expression here")
            else:
                if self._normalizer.version >= (3, 9):
                    ancestor = node.parent
                else:
                    ancestor = _skip_parens_bottom_up(node)
                if ancestor.type not in _STAR_EXPR_PARENTS and not is_aug_assign and not (ancestor.type == 'atom' and ancestor.children[0] == '['):
                    message = 'starred assignment target must be in a list or tuple'
                    self.add_issue(node, message=message)
            self._check_assignment(node.children[1])
        if error is not None:
            if is_namedexpr:
                message = f'cannot use assignment expressions with {error}'
            else:
                cannot = "can't" if self._normalizer.version < (3, 8) else 'cannot'
                msg_action = 'delete' if is_deletion else 'assign to'
                message = f"{cannot} {msg_action} {error}"
            self.add_issue(node, message=message)


@ErrorFinder.register_rule(type='sync_comp_for')
class _CompForRule(_CheckAssignmentRule):
    message: str = 'asynchronous comprehension outside of an asynchronous function'

    def is_issue(self, node: Node) -> bool:
        expr_list = node.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)
        return node.parent.children[0] == 'async' and not self._normalizer.context.is_async_funcdef()


@ErrorFinder.register_rule(type='expr_stmt')
class _ExprStmtRule(_CheckAssignmentRule):
    message: str = 'illegal expression for augmented assignment'
    extended_message: str = "'{target}' is an " + message

    def is_issue(self, node: Node) -> bool:
        augassign = node.children[1]
        is_aug_assign = augassign != '=' and augassign.type != 'annassign'
        if self._normalizer.version <= (3, 8) or not is_aug_assign:
            for before_equal in node.children[:-2:2]:
                self._check_assignment(before_equal, is_aug_assign=is_aug_assign)
        if is_aug_assign:
            target = _remove_parens(node.children[0])
            if target.type == 'name' or (target.type in ('atom_expr', 'power') and target.children[1].type == 'trailer' and target.children[-1].children[0] != '('):
                return False
            if self._normalizer.version <= (3, 8):
                return True
            else:
                rhs_name = _get_rhs_name(node.children[0], self._normalizer.version)
                message = self.extended_message.format(target=rhs_name)
                self.add_issue(node, message=message)
                return True
        return False


@ErrorFinder.register_rule(type='with_item')
class _WithItemRule(_CheckAssignmentRule):

    def is_issue(self, with_item: Node) -> bool:
        self._check_assignment(with_item.children[2])
        return False


@ErrorFinder.register_rule(type='del_stmt')
class _DelStmtRule(_CheckAssignmentRule):

    def is_issue(self, del_stmt: Node) -> bool:
        child = del_stmt.children[1]
        if child.type != 'expr_list':
            self._check_assignment(child, is_deletion=True)
        return False


@ErrorFinder.register_rule(type='expr_list')
class _ExprListRule(_CheckAssignmentRule):

    def is_issue(self, expr_list: Node) -> bool:
        for expr in expr_list.children[::2]:
            self._check_assignment(expr)
        return False


@ErrorFinder.register_rule(type='for_stmt')
class _ForStmtRule(_CheckAssignmentRule):

    def is_issue(self, for_stmt: Node) -> bool:
        expr_list = for_stmt.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)
        return False


@ErrorFinder.register_rule(type='namedexpr_test')
class _NamedExprRule(_CheckAssignmentRule):

    def is_issue(self, namedexpr_test: Node) -> bool:
        first = namedexpr_test.children[0]

        def search_namedexpr_in_comp_for(node: Node) -> Optional[Node]:
            while True:
                parent = node.parent
                if parent is None:
                    return parent
                if parent.type == 'sync_comp_for' and parent.children[3] == node:
                    return parent
                node = parent

        if search_namedexpr_in_comp_for(namedexpr_test):
            message = 'assignment expression cannot be used in a comprehension iterable expression'
            self.add_issue(namedexpr_test, message=message)
        exprlist: List[Node] = []

        def process_comp_for(comp_for: Node) -> None:
            if comp_for.type == 'sync_comp_for':
                comp = comp_for
            elif comp_for.type == 'comp_for':
                comp = comp_for.children[1]
            exprlist.extend(_get_for_stmt_definition_exprs(comp))

        def search_all_comp_ancestors(node: Node) -> bool:
            has_ancestors = False
            while True:
                node = node.search_ancestor('testlist_comp', 'dictorsetmaker')
                if node is None:
                    break
                for child in node.children:
                    if child.type in _COMP_FOR_TYPES:
                        process_comp_for(child)
                        has_ancestors = True
                        break
            return has_ancestors

        search_all = search_all_comp_ancestors(namedexpr_test)
        if search_all:
            if self._normalizer.context.node.type == 'classdef':
                message = 'assignment expression within a comprehension cannot be used in a class body'
                self.add_issue(namedexpr_test, message=message)
            namelist = [expr.value for expr in exprlist if expr.type == 'name']
            if first.type == 'name' and first.value in namelist:
                message = f"assignment expression cannot rebind comprehension iteration variable {first.value!r}"
                self.add_issue(namedexpr_test, message=message)
        self._check_assignment(first, is_namedexpr=True)
        return False


@ErrorFinder.register_rule(type='strings')
class _BytesAndStringMix(SyntaxRule):
    message: str = 'cannot mix bytes and nonbytes literals'

    def _is_bytes_literal(self, string: Node) -> bool:
        if string.type == 'fstring':
            return False
        return 'b' in string.string_prefix.lower()

    def is_issue(self, node: Node) -> bool:
        first = node.children[0]
        first_is_bytes = self._is_bytes_literal(first)
        for string in node.children[1:]:
            if first_is_bytes != self._is_bytes_literal(string):
                return True
        return False


@ErrorFinder.register_rule(type='import_as_names')
class _TrailingImportComma(SyntaxRule):
    message: str = 'trailing comma not allowed without surrounding parentheses'

    def is_issue(self, node: Node) -> bool:
        if node.children[-1] == ',' and node.parent.children[-1] != ')':
            return True
        return False


@ErrorFinder.register_rule(type='import_from')
class _ImportStarInFunction(SyntaxRule):
    message: str = 'import * only allowed at module level'

    def is_issue(self, node: Node) -> bool:
        return node.is_star_import() and self._normalizer.context.parent_context is not None


@ErrorFinder.register_rule(type='import_from')
class _FutureImportRule(SyntaxRule):
    message: str = 'from __future__ imports must occur at the beginning of the file'

    def is_issue(self, node: Node) -> bool:
        if _is_future_import(node):
            if not _is_future_import_first(node):
                return True
            for from_name, future_name in node.get_paths():
                name = future_name.value
                allowed_futures = list(ALLOWED_FUTURES)
                if self._normalizer.version >= (3, 7):
                    allowed_futures.append('annotations')
                if name == 'braces':
                    self.add_issue(node, message='not a chance')
                elif name == 'barry_as_FLUFL':
                    m = "Seriously I'm not implementing this :) ~ Dave"
                    self.add_issue(node, message=m)
                elif name not in allowed_futures:
                    message = f'future feature {name} is not defined'
                    self.add_issue(node, message=message)
        return False


@ErrorFinder.register_rule(type='star_expr')
class _StarExprRule(SyntaxRule):
    message_iterable_unpacking: str = 'iterable unpacking cannot be used in comprehension'

    def is_issue(self, node: Node) -> bool:
        def check_delete_starred(node: Node) -> bool:
            while node.parent is not None:
                node = node.parent
                if node.type == 'del_stmt':
                    return True
                if node.type not in _STAR_EXPR_PARENTS and node.type != 'atom':
                    return False
            return False

        if self._normalizer.version >= (3, 9):
            ancestor = node.parent
        else:
            ancestor = _skip_parens_bottom_up(node)
        if ancestor.type not in (*_STAR_EXPR_PARENTS, 'dictorsetmaker') and not (ancestor.type == 'atom' and ancestor.children[0] != '('):
            self.add_issue(node, message="can't use starred expression here")
            return True
        if check_delete_starred(node):
            if self._normalizer.version >= (3, 9):
                self.add_issue(node, message='cannot delete starred')
            else:
                self.add_issue(node, message="can't use starred expression here")
            return True
        if node.parent.type == 'testlist_comp':
            if node.parent.children[1].type in _COMP_FOR_TYPES:
                self.add_issue(node, message=self.message_iterable_unpacking)
                return True
        return False


@ErrorFinder.register_rule(types=_STAR_EXPR_PARENTS)
class _StarExprParentRule(SyntaxRule):

    def is_issue(self, node: Node) -> bool:
        def is_definition(node: Node, ancestor: Optional[Node]) -> bool:
            if ancestor is None:
                return False
            type_ = ancestor.type
            if type_ == 'trailer':
                return False
            if type_ == 'expr_stmt':
                return node.start_pos < ancestor.children[-1].start_pos
            return is_definition(node, ancestor.parent)
        
        if is_definition(node, node.parent):
            args = [c for c in node.children if c != ',']
            starred = [c for c in args if c.type == 'star_expr']
            if len(starred) > 1:
                if self._normalizer.version < (3, 9):
                    message = 'two starred expressions in assignment'
                else:
                    message = 'multiple starred expressions in assignment'
                self.add_issue(starred[1], message=message)
            elif starred:
                count = args.index(starred[0])
                if count >= 256:
                    message = 'too many expressions in star-unpacking assignment'
                    self.add_issue(starred[0], message=message)
        return False


@ErrorFinder.register_rule(type='annassign')
class _AnnotatorRule(SyntaxRule):
    message: str = 'illegal target for annotation'

    def get_node(self, node: Node) -> Node:
        return node.parent

    def is_issue(self, node: Node) -> bool:
        type_: Optional[str] = None
        lhs = node.parent.children[0]
        lhs = _remove_parens(lhs)
        try:
            children = lhs.children
        except AttributeError:
            pass
        else:
            if ',' in children or (lhs.type == 'atom' and children[0] == '('):
                type_ = 'tuple'
            elif lhs.type == 'atom' and children[0] == '[':
                type_ = 'list'
            trailer = children[-1]
        if type_ is None:
            if not (lhs.type == 'name' or (lhs.type in ('atom_expr', 'power') and trailer.type == 'trailer' and trailer.children[0] != '(')):
                self.add_issue(lhs.parent, message=self.message)
                return True
        else:
            message = f'only single target (not {type_}) can be annotated'
            self.add_issue(lhs.parent, message=message)
            return True
        return False


@ErrorFinder.register_rule(type='argument')
class _ArgumentRule(SyntaxRule):

    def is_issue(self, node: Node) -> bool:
        first = node.children[0]
        if self._normalizer.version < (3, 8):
            first = _remove_parens(first)
        if node.children[1] == '=' and first.type != 'name':
            if first.type == 'lambdef':
                if self._normalizer.version < (3, 8):
                    message = 'lambda cannot contain assignment'
                else:
                    message = 'expression cannot contain assignment, perhaps you meant "=="?'
            elif self._normalizer.version < (3, 8):
                message = "keyword can't be an expression"
            else:
                message = 'expression cannot contain assignment, perhaps you meant "=="?'
            self.add_issue(first, message=message)
        if _is_argument_comprehension(node) and node.parent.type == 'classdef':
            self.add_issue(node, message='invalid syntax')
        return False


@ErrorFinder.register_rule(type='nonlocal_stmt')
class _NonlocalModuleLevelRule(SyntaxRule):
    message: str = 'nonlocal declaration not allowed at module level'

    def is_issue(self, node: Node) -> bool:
        return self._normalizer.context.parent_context is None


@ErrorFinder.register_rule(type='arglist')
class _ArglistRule(SyntaxRule):

    @property
    def message(self) -> str:
        if self._normalizer.version < (3, 7):
            return 'Generator expression must be parenthesized if not sole argument'
        else:
            return 'Generator expression must be parenthesized'

    def is_issue(self, node: Node) -> bool:
        arg_set: set = set()
        kw_only: bool = False
        kw_unpacking_only: bool = False
        for argument in node.children:
            if argument == ',':
                continue
            if argument.type == 'argument':
                first = argument.children[0]
                if _is_argument_comprehension(argument) and len(node.children) >= 2:
                    return True
                if first in ('*', '**'):
                    if first == '*':
                        if kw_unpacking_only:
                            message = 'iterable argument unpacking follows keyword argument unpacking'
                            self.add_issue(argument, message=message)
                    else:
                        kw_unpacking_only = True
                else:
                    kw_only = True
                    if first.type == 'name':
                        if first.value in arg_set:
                            message = 'keyword argument repeated'
                            if self._normalizer.version >= (3, 9):
                                message += f': {first.value}'
                            self.add_issue(first, message=message)
                        else:
                            arg_set.add(first.value)
            elif kw_unpacking_only:
                message = 'positional argument follows keyword argument unpacking'
                self.add_issue(argument, message=message)
            elif kw_only:
                message = 'positional argument follows keyword argument'
                self.add_issue(argument, message=message)
        return False


@ErrorFinder.register_rule(type='parameters')
@ErrorFinder.register_rule(type='lambdef')
class _ParameterRule(SyntaxRule):
    message: str = 'non-default argument follows default argument'

    def is_issue(self, node: Node) -> bool:
        param_names: set = set()
        default_only: bool = False
        star_seen: bool = False
        for p in _iter_params(node):
            if p.type == 'operator':
                if p.value == '*':
                    star_seen = True
                    default_only = False
                continue
            if p.name.value in param_names:
                message = f"duplicate argument '{p.name.value}' in function definition"
                self.add_issue(p.name, message=message)
            param_names.add(p.name.value)
            if not star_seen:
                if p.default is None and not p.star_count:
                    if default_only:
                        self.add_issue(p, message=self.message)
                        return True
                elif p.star_count:
                    star_seen = True
                    default_only = False
                else:
                    default_only = True
        return False


@ErrorFinder.register_rule(type='try_stmt')
class _TryStmtRule(SyntaxRule):
    message: str = "default 'except:' must be last"

    def is_issue(self, try_stmt: Node) -> bool:
        default_except: Optional[Node] = None
        for except_clause in try_stmt.children[3::3]:
            if except_clause in ('else', 'finally'):
                break
            if except_clause == 'except':
                default_except = except_clause
            elif default_except is not None:
                self.add_issue(default_except, message=self.message)
        return False


@ErrorFinder.register_rule(type='fstring')
class _FStringRule(SyntaxRule):
    _fstring_grammar: Optional[Any] = None
    message_expr: str = 'f-string expression part cannot include a backslash'
    message_nested: str = 'f-string: expressions nested too deeply'
    message_conversion: str = "f-string: invalid conversion character: expected 's', 'r', or 'a'"

    def _check_format_spec(self, format_spec: Node, depth: int) -> None:
        self._check_fstring_contents(format_spec.children[1:], depth)

    def _check_fstring_expr(self, fstring_expr: Node, depth: int) -> None:
        if depth >= 2:
            self.add_issue(fstring_expr, message=self.message_nested)
        expr = fstring_expr.children[1]
        if '\\' in expr.get_code():
            self.add_issue(expr, message=self.message_expr)
        children_2 = fstring_expr.children[2]
        if children_2.type == 'operator' and children_2.value == '=':
            conversion = fstring_expr.children[3]
        else:
            conversion = children_2
        if conversion.type == 'fstring_conversion':
            name = conversion.children[1]
            if name.value not in ('s', 'r', 'a'):
                self.add_issue(name, message=self.message_conversion)
        format_spec = fstring_expr.children[-2]
        if format_spec.type == 'fstring_format_spec':
            self._check_format_spec(format_spec, depth + 1)

    def is_issue(self, fstring: Node) -> bool:
        self._check_fstring_contents(fstring.children[1:-1])
        return False


class _CheckAssignmentRule(SyntaxRule):

    def _check_assignment(self, node: Node, is_deletion: bool = False, is_namedexpr: bool = False, is_aug_assign: bool = False) -> None:
        error: Optional[str] = None
        type_ = node.type
        if type_ == 'lambdef':
            error = 'lambda'
        elif type_ == 'atom':
            first, second = node.children[:2]
            error = _get_comprehension_type(node)
            if error is None:
                if second.type == 'dictorsetmaker':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    elif second.children[1] == ':':
                        error = 'dict display'
                    else:
                        error = 'set display'
                elif first == '{' and second == '}':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'dict display'
                elif first == '{' and len(node.children) > 2:
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'set display'
                elif first in ('(', '['):
                    if second.type == 'yield_expr':
                        error = 'yield expression'
                    elif second.type == 'testlist_comp':
                        if is_namedexpr:
                            if first == '(':
                                error = 'tuple'
                            elif first == '[':
                                error = 'list'
                        for child in second.children[::2]:
                            self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
                    else:
                        self._check_assignment(second, is_deletion, is_namedexpr, is_aug_assign)
        elif type_ == 'keyword':
            if node.value == 'yield':
                error = 'yield expression'
            elif self._normalizer.version < (3, 8):
                error = 'keyword'
            else:
                error = str(node.value)
        elif type_ == 'operator':
            if node.value == '...':
                error = 'Ellipsis'
        elif type_ == 'comparison':
            error = 'comparison'
        elif type_ in ('string', 'number', 'strings'):
            error = 'literal'
        elif type_ == 'yield_expr':
            message: str = 'assignment to yield expression not possible'
            self.add_issue(node, message=message)
        elif type_ == 'test':
            error = 'conditional expression'
        elif type_ in ('atom_expr', 'power'):
            if node.children[0] == 'await':
                error = 'await expression'
            elif node.children[-2] == '**':
                error = 'operator'
            else:
                trailer = node.children[-1]
                assert trailer.type == 'trailer'
                if trailer.children[0] == '(':
                    error = 'function call'
                elif is_namedexpr and trailer.children[0] == '[':
                    error = 'subscript'
                elif is_namedexpr and trailer.children[0] == '.':
                    error = 'attribute'
        elif type_ == 'fstring':
            if self._normalizer.version < (3, 8):
                error = 'literal'
            else:
                error = 'f-string expression'
        elif type_ in ('testlist_star_expr', 'exprlist', 'testlist'):
            for child in node.children[::2]:
                self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
        elif 'expr' in type_ and type_ != 'star_expr' or '_test' in type_ or type_ in ('term', 'factor'):
            error = 'operator'
        elif type_ == 'star_expr':
            if is_deletion:
                if self._normalizer.version >= (3, 9):
                    error = 'starred'
                else:
                    self.add_issue(node, message="can't use starred expression here")
            else:
                if self._normalizer.version >= (3, 9):
                    ancestor = node.parent
                else:
                    ancestor = _skip_parens_bottom_up(node)
                if ancestor.type not in _STAR_EXPR_PARENTS and not is_aug_assign and not (ancestor.type == 'atom' and ancestor.children[0] == '['):
                    message: str = 'starred assignment target must be in a list or tuple'
                    self.add_issue(node, message=message)
            self._check_assignment(node.children[1])
        if error is not None:
            if is_namedexpr:
                message: str = f'cannot use assignment expressions with {error}'
            else:
                cannot: str = "can't" if self._normalizer.version < (3, 8) else 'cannot'
                msg_action: str = 'delete' if is_deletion else 'assign to'
                message = f"{cannot} {msg_action} {error}"
            self.add_issue(node, message=message)


@ErrorFinder.register_rule(type='sync_comp_for')
class _CompForRule(_CheckAssignmentRule):
    message: str = 'asynchronous comprehension outside of an asynchronous function'

    def is_issue(self, node: Node) -> bool:
        expr_list = node.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)
        return node.parent.children[0] == 'async' and not self._normalizer.context.is_async_funcdef()


@ErrorFinder.register_rule(type='expr_stmt')
class _ExprStmtRule(_CheckAssignmentRule):
    message: str = 'illegal expression for augmented assignment'
    extended_message: str = "'{target}' is an " + message

    def is_issue(self, node: Node) -> bool:
        augassign = node.children[1]
        is_aug_assign = augassign != '=' and augassign.type != 'annassign'
        if self._normalizer.version <= (3, 8) or not is_aug_assign:
            for before_equal in node.children[:-2:2]:
                self._check_assignment(before_equal, is_aug_assign=is_aug_assign)
        if is_aug_assign:
            target = _remove_parens(node.children[0])
            if target.type == 'name' or (target.type in ('atom_expr', 'power') and target.children[1].type == 'trailer' and target.children[-1].children[0] != '('):
                return False
            if self._normalizer.version <= (3, 8):
                return True
            else:
                rhs_name = _get_rhs_name(node.children[0], self._normalizer.version)
                message = self.extended_message.format(target=rhs_name)
                self.add_issue(node, message=message)
                return True
        return False


@ErrorFinder.register_rule(type='with_item')
class _WithItemRule(_CheckAssignmentRule):

    def is_issue(self, with_item: Node) -> bool:
        self._check_assignment(with_item.children[2])
        return False


@ErrorFinder.register_rule(type='del_stmt')
class _DelStmtRule(_CheckAssignmentRule):

    def is_issue(self, del_stmt: Node) -> bool:
        child = del_stmt.children[1]
        if child.type != 'expr_list':
            self._check_assignment(child, is_deletion=True)
        return False


@ErrorFinder.register_rule(type='expr_list')
class _ExprListRule(_CheckAssignmentRule):

    def is_issue(self, expr_list: Node) -> bool:
        for expr in expr_list.children[::2]:
            self._check_assignment(expr)
        return False


@ErrorFinder.register_rule(type='for_stmt')
class _ForStmtRule(_CheckAssignmentRule):

    def is_issue(self, for_stmt: Node) -> bool:
        expr_list = for_stmt.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)
        return False


@ErrorFinder.register_rule(type='namedexpr_test')
class _NamedExprRule(_CheckAssignmentRule):

    def is_issue(self, namedexpr_test: Node) -> bool:
        first = namedexpr_test.children[0]

        def search_namedexpr_in_comp_for(node: Node) -> Optional[Node]:
            while True:
                parent = node.parent
                if parent is None:
                    return parent
                if parent.type == 'sync_comp_for' and parent.children[3] == node:
                    return parent
                node = parent

        if search_namedexpr_in_comp_for(namedexpr_test):
            message = 'assignment expression cannot be used in a comprehension iterable expression'
            self.add_issue(namedexpr_test, message=message)
        exprlist: List[Node] = []

        def process_comp_for(comp_for: Node) -> None:
            if comp_for.type == 'sync_comp_for':
                comp = comp_for
            elif comp_for.type == 'comp_for':
                comp = comp_for.children[1]
            exprlist.extend(_get_for_stmt_definition_exprs(comp))

        def search_all_comp_ancestors(node: Node) -> bool:
            has_ancestors = False
            while True:
                node = node.search_ancestor('testlist_comp', 'dictorsetmaker')
                if node is None:
                    break
                for child in node.children:
                    if child.type in _COMP_FOR_TYPES:
                        process_comp_for(child)
                        has_ancestors = True
                        break
            return has_ancestors

        search_all = search_all_comp_ancestors(namedexpr_test)
        if search_all:
            if self._normalizer.context.node.type == 'classdef':
                message = 'assignment expression within a comprehension cannot be used in a class body'
                self.add_issue(namedexpr_test, message=message)
            namelist = [expr.value for expr in exprlist if expr.type == 'name']
            if first.type == 'name' and first.value in namelist:
                message = f"assignment expression cannot rebind comprehension iteration variable {first.value!r}"
                self.add_issue(namedexpr_test, message=message)
        self._check_assignment(first, is_namedexpr=True)
        return False
