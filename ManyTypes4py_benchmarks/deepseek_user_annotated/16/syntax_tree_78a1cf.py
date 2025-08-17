"""
Functions inferring the syntax tree.
"""
import copy
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable, Iterator, Iterable

from parso.python import tree
from parso.python.tree import PythonNode, PythonLeaf

from jedi import debug
from jedi import parser_utils
from jedi.inference.base_value import ValueSet, NO_VALUES, ContextualizedNode, \
    iterator_to_value_set, iterate_values
from jedi.inference.lazy_value import LazyTreeValue
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import analysis
from jedi.inference import imports
from jedi.inference import arguments
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.value import iterable
from jedi.inference.value.dynamic_arrays import ListModification, DictModification
from jedi.inference.value import TreeInstance
from jedi.inference.helpers import is_string, is_literal, is_number, \
    get_names_of_node, is_big_annoying_library
from jedi.inference.compiled.access import COMPARISON_OPERATORS
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.gradual.stub_value import VersionInfo
from jedi.inference.gradual import annotation
from jedi.inference.names import TreeNameDefinition
from jedi.inference.context import CompForContext
from jedi.inference.value.decorator import Decoratee
from jedi.plugins import plugin_manager
from jedi.inference.value.instance import BoundMethod
from jedi.inference.value.module import ModuleValue
from jedi.inference.context import FunctionContext, ClassContext
from jedi.inference import InferenceState
from jedi.inference.value import TreeValue

operator_to_magic_method = {
    '+': '__add__',
    '-': '__sub__',
    '*': '__mul__',
    '@': '__matmul__',
    '/': '__truediv__',
    '//': '__floordiv__',
    '%': '__mod__',
    '**': '__pow__',
    '<<': '__lshift__',
    '>>': '__rshift__',
    '&': '__and__',
    '|': '__or__',
    '^': '__xor__',
}

reverse_operator_to_magic_method = {
    k: '__r' + v[2:] for k, v in operator_to_magic_method.items()
}

def _limit_value_infers(func: Callable) -> Callable:
    def wrapper(context: Any, *args: Any, **kwargs: Any) -> ValueSet:
        n = context.tree_node
        inference_state = context.inference_state
        try:
            inference_state.inferred_element_counts[n] += 1
            maximum = 300
            if context.parent_context is None \
                    and context.get_value() is inference_state.builtins_module:
                maximum *= 100

            if inference_state.inferred_element_counts[n] > maximum:
                debug.warning('In value %s there were too many inferences.', n)
                return NO_VALUES
        except KeyError:
            inference_state.inferred_element_counts[n] = 1
        return func(context, *args, **kwargs)
    return wrapper

def infer_node(context: Any, element: PythonNode) -> ValueSet:
    if isinstance(context, CompForContext):
        return _infer_node(context, element)

    if_stmt = element
    while if_stmt is not None:
        if_stmt = if_stmt.parent
        if if_stmt.type in ('if_stmt', 'for_stmt'):
            break
        if parser_utils.is_scope(if_stmt):
            if_stmt = None
            break
    predefined_if_name_dict = context.predefined_names.get(if_stmt)
    if predefined_if_name_dict is None and if_stmt \
            and if_stmt.type == 'if_stmt' and context.inference_state.is_analysis:
        if_stmt_test = if_stmt.children[1]
        name_dicts = [{}]
        if element.start_pos > if_stmt_test.end_pos:
            if_names = get_names_of_node(if_stmt_test)
            element_names = get_names_of_node(element)
            str_element_names = [e.value for e in element_names]
            if any(i.value in str_element_names for i in if_names):
                for if_name in if_names:
                    definitions = context.inference_state.infer(context, if_name)
                    if len(definitions) > 1:
                        if len(name_dicts) * len(definitions) > 16:
                            debug.dbg('Too many options for if branch inference %s.', if_stmt)
                            name_dicts = [{}]
                            break

                        original_name_dicts = list(name_dicts)
                        name_dicts = []
                        for definition in definitions:
                            new_name_dicts = list(original_name_dicts)
                            for i, name_dict in enumerate(new_name_dicts):
                                new_name_dicts[i] = name_dict.copy()
                                new_name_dicts[i][if_name.value] = ValueSet([definition])
                            name_dicts += new_name_dicts
                    else:
                        for name_dict in name_dicts:
                            name_dict[if_name.value] = definitions
        if len(name_dicts) > 1:
            result = NO_VALUES
            for name_dict in name_dicts:
                with context.predefine_names(if_stmt, name_dict):
                    result |= _infer_node(context, element)
            return result
        else:
            return _infer_node_if_inferred(context, element)
    else:
        if predefined_if_name_dict:
            return _infer_node(context, element)
        else:
            return _infer_node_if_inferred(context, element)

def _infer_node_if_inferred(context: Any, element: PythonNode) -> ValueSet:
    parent = element
    while parent is not None:
        parent = parent.parent
        predefined_if_name_dict = context.predefined_names.get(parent)
        if predefined_if_name_dict is not None:
            return _infer_node(context, element)
    return _infer_node_cached(context, element)

@inference_state_method_cache(default=NO_VALUES)
def _infer_node_cached(context: Any, element: PythonNode) -> ValueSet:
    return _infer_node(context, element)

@debug.increase_indent
@_limit_value_infers
def _infer_node(context: Any, element: PythonNode) -> ValueSet:
    debug.dbg('infer_node %s@%s in %s', element, element.start_pos, context)
    inference_state = context.inference_state
    typ = element.type
    if typ in ('name', 'number', 'string', 'atom', 'strings', 'keyword', 'fstring'):
        return infer_atom(context, element)
    elif typ == 'lambdef':
        return ValueSet([FunctionValue.from_context(context, element)])
    elif typ == 'expr_stmt':
        return infer_expr_stmt(context, element)
    elif typ in ('power', 'atom_expr'):
        first_child = element.children[0]
        children = element.children[1:]
        had_await = False
        if first_child.type == 'keyword' and first_child.value == 'await':
            had_await = True
            first_child = children.pop(0)

        value_set = context.infer_node(first_child)
        for (i, trailer) in enumerate(children):
            if trailer == '**':  # has a power operation.
                right = context.infer_node(children[i + 1])
                value_set = _infer_comparison(
                    context,
                    value_set,
                    trailer,
                    right
                )
                break
            value_set = infer_trailer(context, value_set, trailer)

        if had_await:
            return value_set.py__await__().py__stop_iteration_returns()
        return value_set
    elif typ in ('testlist_star_expr', 'testlist',):
        return ValueSet([iterable.SequenceLiteralValue(inference_state, context, element)])
    elif typ in ('not_test', 'factor'):
        value_set = context.infer_node(element.children[-1])
        for operator in element.children[:-1]:
            value_set = infer_factor(value_set, operator)
        return value_set
    elif typ == 'test':
        return (context.infer_node(element.children[0])
                | context.infer_node(element.children[-1]))
    elif typ == 'operator':
        if element.value != '...':
            origin = element.parent
            raise AssertionError("unhandled operator %s in %s " % (repr(element.value), origin))
        return ValueSet([compiled.builtin_from_name(inference_state, 'Ellipsis')])
    elif typ == 'dotted_name':
        value_set = infer_atom(context, element.children[0])
        for next_name in element.children[2::2]:
            value_set = value_set.py__getattribute__(next_name, name_context=context)
        return value_set
    elif typ == 'eval_input':
        return context.infer_node(element.children[0])
    elif typ == 'annassign':
        return annotation.infer_annotation(context, element.children[1]) \
            .execute_annotation()
    elif typ == 'yield_expr':
        if len(element.children) and element.children[1].type == 'yield_arg':
            element = element.children[1].children[1]
            generators = context.infer_node(element) \
                .py__getattribute__('__iter__').execute_with_values()
            return generators.py__stop_iteration_returns()
        return NO_VALUES
    elif typ == 'namedexpr_test':
        return context.infer_node(element.children[2])
    else:
        return infer_or_test(context, element)

def infer_trailer(context: Any, atom_values: ValueSet, trailer: PythonNode) -> ValueSet:
    trailer_op, node = trailer.children[:2]
    if node == ')':  # `arglist` is optional.
        node = None

    if trailer_op == '[':
        trailer_op, node, _ = trailer.children
        return atom_values.get_item(
            _infer_subscript_list(context, node),
            ContextualizedNode(context, trailer)
        )
    else:
        debug.dbg('infer_trailer: %s in %s', trailer, atom_values)
        if trailer_op == '.':
            return atom_values.py__getattribute__(
                name_context=context,
                name_or_str=node
            )
        else:
            assert trailer_op == '(', 'trailer_op is actually %s' % trailer_op
            args = arguments.TreeArguments(context.inference_state, context, node, trailer)
            return atom_values.execute(args)

def infer_atom(context: Any, atom: Union[PythonNode, PythonLeaf]) -> ValueSet:
    state = context.inference_state
    if atom.type == 'name':
        stmt = tree.search_ancestor(atom, 'expr_stmt', 'lambdef', 'if_stmt') or atom
        if stmt.type == 'if_stmt':
            if not any(n.start_pos <= atom.start_pos < n.end_pos for n in stmt.get_test_nodes()):
                stmt = atom
        elif stmt.type == 'lambdef':
            stmt = atom
        position = stmt.start_pos
        if _is_annotation_name(atom):
            position = None
        return context.py__getattribute__(atom, position=position)
    elif atom.type == 'keyword':
        if atom.value in ('False', 'True', 'None'):
            return ValueSet([compiled.builtin_from_name(state, atom.value)])
        elif atom.value == 'yield':
            return NO_VALUES
        assert False, 'Cannot infer the keyword %s' % atom

    elif isinstance(atom, tree.Literal):
        string = state.compiled_subprocess.safe_literal_eval(atom.value)
        return ValueSet([compiled.create_simple_object(state, string)])
    elif atom.type == 'strings':
        value_set = infer_atom(context, atom.children[0])
        for string in atom.children[1:]:
            right = infer_atom(context, string)
            value_set = _infer_comparison(context, value_set, '+', right)
        return value_set
    elif atom.type == 'fstring':
        return compiled.get_string_value_set(state)
    else:
        c = atom.children
        if c[0] == '(' and not len(c) == 2 \
                and not(c[1].type == 'testlist_comp'
                        and len(c[1].children) > 1):
            return context.infer_node(c[1])

        try:
            comp_for = c[1].children[1]
        except (IndexError, AttributeError):
            pass
        else:
            if comp_for == ':':
                try:
                    comp_for = c[1].children[3]
                except IndexError:
                    pass

            if comp_for.type in ('comp_for', 'sync_comp_for'):
                return ValueSet([iterable.comprehension_from_atom(
                    state, context, atom
                )])

        array_node = c[1]
        try:
            array_node_c = array_node.children
        except AttributeError:
            array_node_c = []
        if c[0] == '{' and (array_node == '}' or ':' in array_node_c
                            or '**' in array_node_c):
            new_value = iterable.DictLiteralValue(state, context, atom)
        else:
            new_value = iterable.SequenceLiteralValue(state, context, atom)
        return ValueSet([new_value])

@_limit_value_infers
def infer_expr_stmt(context: Any, stmt: PythonNode, seek_name: Optional[PythonLeaf] = None) -> ValueSet:
    with recursion.execution_allowed(context.inference_state, stmt) as allowed:
        if allowed:
            if seek_name is not None:
                pep0484_values = \
                    annotation.find_type_from_comment_hint_assign(context, stmt, seek_name)
                if pep0484_values:
                    return pep0484_values

            return _infer_expr_stmt(context, stmt, seek_name)
    return NO_VALUES

@debug.increase_indent
def _infer_expr_stmt(context: Any, stmt: PythonNode, seek_name: Optional[PythonLeaf] = None) -> ValueSet:
    def check_setitem(stmt: PythonNode) -> Tuple[bool, Optional[PythonNode]]:
        atom_expr = stmt.children[0]
        if atom_expr.type not in ('atom_expr', 'power'):
            return False, None
        name = atom_expr.children[0]
        if name.type != 'name' or len(atom_expr.children) != 2:
            return False, None
        trailer = atom_expr.children[-1]
        return trailer.children[0] == '[', trailer.children[1]

    debug.dbg('infer_expr_stmt %s (%s)', stmt, seek_name)
    rhs = stmt.get_rhs()
    value_set = context.infer_node(rhs)

    if seek_name:
        n = TreeNameDefinition(context, seek_name)
        value_set = check_tuple_assignments(n, value_set)

    first_operator = next(stmt.yield_operators(), None)
    is_setitem, subscriptlist = check_setitem(stmt)
    is_annassign = first_operator not in ('=', None) and first_operator.type == 'operator'
    if is_annassign or is_setitem:
        name = stmt.get_defined_names(include_setitem=True)[0].value
        left_values = context.py__getattribute__(name, position=stmt.start_pos)

        if is_setitem:
            def to_mod(v: TreeValue) -> TreeValue:
                c = ContextualizedSubscriptListNode(context, subscriptlist)
                if v.array_type == 'dict':
                    return DictModification(v, value_set, c)
                elif v.array_type == 'list':
                    return ListModification(v, value_set, c)
                return v

            value_set = ValueSet(to_mod(v) for v in left_values)
        else:
            operator = copy.copy(first_operator)
            operator.value = operator.value[:-1]
            for_stmt = tree.search_ancestor(stmt, 'for_stmt')
            if for_stmt is not None and for_stmt.type == 'for_stmt' and value_set \
                    and parser_utils.for_stmt_defines_one_name(for_stmt):
                node = for_stmt.get_testlist()
                cn = ContextualizedNode(context, node)
                ordered = list(cn.infer().iterate(cn))

                for lazy_value in ordered:
                    dct = {for_stmt.children[1].value: lazy_value.infer()}
                    with context.predefine_names(for_stmt, dct):
                        t = context.infer_node(rhs)
                        left_values = _infer_comparison(context, left_values, operator, t)
                value_set = left_values
            else:
                value_set = _infer_comparison(context, left_values, operator, value_set)
    debug.dbg('infer_expr_stmt result %s', value_set)
    return value_set

def infer_or_test(context: Any, or_test: PythonNode) -> ValueSet:
    iterator = iter(or_test.children)
    types = context.infer_node(next(iterator))
    for operator in iterator:
        right = next(iterator)
        if operator.type == 'comp_op':
            operator = ' '.join(c.value for c in operator.children)

        if operator in ('and', 'or'):
            left_bools = set(left.py__bool__() for left in types)
            if left_bools == {True}:
                if operator == 'and':
                    types = context.infer_node(right)
            elif left_bools == {False}:
                if operator != 'and':
                    types = context.infer_node(right)
        else:
            types = _infer_comparison(context, types, operator,
                                      context.infer_node(right))
    debug.dbg('infer_or_test types %s', types)
    return types

@iterator_to_value_set
def infer_factor(value_set: ValueSet, operator: PythonLeaf) -> Iterator[TreeValue]:
    for value in value_set:
        if operator == '-':
            if is_number(value):
                yield value.negate()
        elif operator == 'not':
            b = value.py__bool__()
            if b is None:
                return
            yield compiled.create_simple_object(value.inference_state, not b)
        else:
            yield value

def _literals_to_types(inference_state: InferenceState, result: ValueSet) -> ValueSet:
    new_result = NO_VALUES
    for typ in result:
        if is_literal(typ):
            cls = compiled.builtin_from_name(inference_state, typ.name.string_name