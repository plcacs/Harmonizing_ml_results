"""
Functions inferring the syntax tree.
"""
import copy
from typing import Callable, Optional, Any, Union, Iterator, List, Dict, Tuple
from parso.python import tree
from jedi import debug
from jedi import parser_utils
from jedi.inference.base_value import (
    ValueSet,
    NO_VALUES,
    ContextualizedNode,
    iterator_to_value_set,
    iterate_values,
)
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
from jedi.inference.helpers import (
    is_string,
    is_literal,
    is_number,
    get_names_of_node,
    is_big_annoying_library,
)
from jedi.inference.compiled.access import COMPARISON_OPERATORS
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.gradual.stub_value import VersionInfo
from jedi.inference.gradual import annotation
from jedi.inference.names import TreeNameDefinition
from jedi.inference.context import CompForContext
from jedi.inference.value.decorator import Decoratee
from jedi.plugins import plugin_manager

from parso.python.tree import NodeOrLeaf

operator_to_magic_method: Dict[str, str] = {
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "@": "__matmul__",
    "/": "__truediv__",
    "//": "__floordiv__",
    "%": "__mod__",
    "**": "__pow__",
    "<<": "__lshift__",
    ">>": "__rshift__",
    "&": "__and__",
    "|": "__or__",
    "^": "__xor__",
}
reverse_operator_to_magic_method: Dict[str, str] = {
    k: "__r" + v[2:] for k, v in operator_to_magic_method.items()
}


def _limit_value_infers(func: Callable[..., ValueSet]) -> Callable[..., ValueSet]:
    """
    This is for now the way how we limit type inference going wild. There are
    other ways to ensure recursion limits as well. This is mostly necessary
    because of instance (self) access that can be quite tricky to limit.

    I'm still not sure this is the way to go, but it looks okay for now and we
    can still go anther way in the future. Tests are there. ~ dave
    """

    def wrapper(context: "Context", *args: Any, **kwargs: Any) -> ValueSet:
        n: NodeOrLeaf = context.tree_node
        inference_state = context.inference_state
        try:
            inference_state.inferred_element_counts[n] += 1
            maximum: int = 300
            if (
                context.parent_context is None
                and context.get_value() is inference_state.builtins_module
            ):
                maximum *= 100
            if inference_state.inferred_element_counts[n] > maximum:
                debug.warning("In value %s there were too many inferences.", n)
                return NO_VALUES
        except KeyError:
            inference_state.inferred_element_counts[n] = 1
        return func(context, *args, **kwargs)

    return wrapper


def infer_node(context: "Context", element: NodeOrLeaf) -> ValueSet:
    if isinstance(context, CompForContext):
        return _infer_node(context, element)
    if_stmt: Optional[NodeOrLeaf] = element
    while if_stmt is not None:
        if_stmt = if_stmt.parent
        if if_stmt and if_stmt.type in ("if_stmt", "for_stmt"):
            break
        if if_stmt and parser_utils.is_scope(if_stmt):
            if_stmt = None
            break
    predefined_if_name_dict: Optional[Dict[str, ValueSet]] = context.predefined_names.get(if_stmt)
    if (
        predefined_if_name_dict is None
        and if_stmt
        and if_stmt.type == "if_stmt"
        and context.inference_state.is_analysis
    ):
        if_stmt_test: NodeOrLeaf = if_stmt.children[1]
        name_dicts: List[Dict[str, ValueSet]] = [{}]
        if element.start_pos > if_stmt_test.end_pos:
            if_names = get_names_of_node(if_stmt_test)
            element_names = get_names_of_node(element)
            str_element_names = [e.value for e in element_names]
            if any((i.value in str_element_names for i in if_names)):
                for if_name in if_names:
                    definitions: List["Definition"] = context.inference_state.infer(context, if_name)
                    if len(definitions) > 1:
                        if len(name_dicts) * len(definitions) > 16:
                            debug.dbg("Too many options for if branch inference %s.", if_stmt)
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
            result: ValueSet = NO_VALUES
            for name_dict in name_dicts:
                with context.predefine_names(if_stmt, name_dict):
                    result |= _infer_node(context, element)
            return result
        else:
            return _infer_node_if_inferred(context, element)
    elif predefined_if_name_dict:
        return _infer_node(context, element)
    else:
        return _infer_node_if_inferred(context, element)


def _infer_node_if_inferred(context: "Context", element: NodeOrLeaf) -> ValueSet:
    """
    TODO This function is temporary: Merge with infer_node.
    """
    parent: Optional[NodeOrLeaf] = element
    while parent is not None:
        parent = parent.parent
        predefined_if_name_dict: Optional[Dict[str, ValueSet]] = context.predefined_names.get(parent)
        if predefined_if_name_dict is not None:
            return _infer_node(context, element)
    return _infer_node_cached(context, element)


@inference_state_method_cache(default=NO_VALUES)
def _infer_node_cached(context: "Context", element: NodeOrLeaf) -> ValueSet:
    return _infer_node(context, element)


@debug.increase_indent
@_limit_value_infers
def _infer_node(context: "Context", element: NodeOrLeaf) -> ValueSet:
    debug.dbg("infer_node %s@%s in %s", element, element.start_pos, context)
    inference_state = context.inference_state
    typ: str = element.type
    if typ in ("name", "number", "string", "atom", "strings", "keyword", "fstring"):
        return infer_atom(context, element)
    elif typ == "lambdef":
        return ValueSet([FunctionValue.from_context(context, element)])
    elif typ == "expr_stmt":
        return infer_expr_stmt(context, element)
    elif typ in ("power", "atom_expr"):
        first_child: NodeOrLeaf = element.children[0]
        children: List[NodeOrLeaf] = element.children[1:]
        had_await: bool = False
        if first_child.type == "keyword" and first_child.value == "await":
            had_await = True
            first_child = children.pop(0)
        value_set: ValueSet = context.infer_node(first_child)
        for i, trailer in enumerate(children):
            if trailer == "**":
                right: ValueSet = context.infer_node(children[i + 1])
                value_set = _infer_comparison(context, value_set, trailer, right)
                break
            value_set = infer_trailer(context, value_set, trailer)
        if had_await:
            return value_set.py__await__().py__stop_iteration_returns()
        return value_set
    elif typ in ("testlist_star_expr", "testlist"):
        return ValueSet([iterable.SequenceLiteralValue(inference_state, context, element)])
    elif typ in ("not_test", "factor"):
        value_set: ValueSet = context.infer_node(element.children[-1])
        for operator in element.children[:-1]:
            value_set = infer_factor(value_set, operator)
        return value_set
    elif typ == "test":
        return context.infer_node(element.children[0]) | context.infer_node(element.children[-1])
    elif typ == "operator":
        if element.value != "...":
            origin: NodeOrLeaf = element.parent
            raise AssertionError(
                "unhandled operator %s in %s " % (repr(element.value), origin)
            )
        return ValueSet([compiled.builtin_from_name(inference_state, "Ellipsis")])
    elif typ == "dotted_name":
        value_set: ValueSet = infer_atom(context, element.children[0])
        for next_name in element.children[2::2]:
            value_set = value_set.py__getattribute__(next_name, name_context=context)
        return value_set
    elif typ == "eval_input":
        return context.infer_node(element.children[0])
    elif typ == "annassign":
        return annotation.infer_annotation(context, element.children[1]).execute_annotation()
    elif typ == "yield_expr":
        if len(element.children) and element.children[1].type == "yield_arg":
            element = cast(NodeOrLeaf, element.children[1].children[1])
            generators: ValueSet = context.infer_node(element).py__getattribute__("__iter__").execute_with_values()
            return generators.py__stop_iteration_returns()
        return NO_VALUES
    elif typ == "namedexpr_test":
        return context.infer_node(element.children[2])
    else:
        return infer_or_test(context, element)


def infer_trailer(
    context: "Context", atom_values: ValueSet, trailer: NodeOrLeaf
) -> ValueSet:
    trailer_op: str = trailer.children[0]
    node: Optional[NodeOrLeaf] = trailer.children[1] if len(trailer.children) >= 2 else None
    if node == ")":
        node = None
    if trailer_op == "[":
        trailer_op, node, _ = trailer.children
        return atom_values.get_item(
            _infer_subscript_list(context, node), ContextualizedNode(context, trailer)
        )
    else:
        debug.dbg("infer_trailer: %s in %s", trailer, atom_values)
        if trailer_op == ".":
            return atom_values.py__getattribute__(name_context=context, name_or_str=node)
        else:
            assert trailer_op == "(", "trailer_op is actually %s" % trailer_op
            args: arguments.TreeArguments = arguments.TreeArguments(
                context.inference_state, context, node, trailer
            )
            return atom_values.execute(args)


def infer_atom(context: "Context", atom: NodeOrLeaf) -> ValueSet:
    """
    Basically to process ``atom`` nodes. The parser sometimes doesn't
    generate the node (because it has just one child). In that case an atom
    might be a name or a literal as well.
    """
    state = context.inference_state
    if atom.type == "name":
        stmt: NodeOrLeaf = tree.search_ancestor(atom, "expr_stmt", "lambdef", "if_stmt") or atom
        if stmt.type == "if_stmt":
            if not any(
                n.start_pos <= atom.start_pos < n.end_pos for n in stmt.get_test_nodes()
            ):
                stmt = atom
        elif stmt.type == "lambdef":
            stmt = atom
        position: Optional[Tuple[int, int]] = stmt.start_pos
        if _is_annotation_name(atom):
            position = None
        return context.py__getattribute__(atom, position=position)
    elif atom.type == "keyword":
        if atom.value in ("False", "True", "None"):
            return ValueSet([compiled.builtin_from_name(state, atom.value)])
        elif atom.value == "yield":
            return NO_VALUES
        assert False, "Cannot infer the keyword %s" % atom
    elif isinstance(atom, tree.Literal):
        string: Any = state.compiled_subprocess.safe_literal_eval(atom.value)
        return ValueSet([compiled.create_simple_object(state, string)])
    elif atom.type == "strings":
        value_set: ValueSet = infer_atom(context, atom.children[0])
        for string in atom.children[1:]:
            right: ValueSet = infer_atom(context, string)
            value_set = _infer_comparison(context, value_set, "+", right)
        return value_set
    elif atom.type == "fstring":
        return compiled.get_string_value_set(state)
    else:
        c: List[Any] = atom.children
        if c[0] == "(" and not len(c) == 2 and not (
            c[1].type == "testlist_comp" and len(c[1].children) > 1
        ):
            return context.infer_node(c[1])
        try:
            comp_for: Any = c[1].children[1]
        except (IndexError, AttributeError):
            pass
        else:
            if comp_for == ":":
                try:
                    comp_for = c[1].children[3]
                except IndexError:
                    pass
            if comp_for.type in ("comp_for", "sync_comp_for"):
                return ValueSet(
                    [iterable.comprehension_from_atom(state, context, atom)]
                )
        array_node: NodeOrLeaf = c[1]
        try:
            array_node_c: List[Any] = array_node.children
        except AttributeError:
            array_node_c = []
        if c[0] == "{" and (array_node == "}" or ":" in array_node_c or "**" in array_node_c):
            new_value: iterable.DictLiteralValue = iterable.DictLiteralValue(
                state, context, atom
            )
        else:
            new_value: iterable.SequenceLiteralValue = iterable.SequenceLiteralValue(
                state, context, atom
            )
        return ValueSet([new_value])


@_limit_value_infers
def infer_expr_stmt(
    context: "Context", stmt: NodeOrLeaf, seek_name: Optional[str] = None
) -> ValueSet:
    with recursion.execution_allowed(context.inference_state, stmt) as allowed:
        if allowed:
            if seek_name is not None:
                pep0484_values: ValueSet = annotation.find_type_from_comment_hint_assign(
                    context, stmt, seek_name
                )
                if pep0484_values:
                    return pep0484_values
            return _infer_expr_stmt(context, stmt, seek_name)
    return NO_VALUES


@debug.increase_indent
def _infer_expr_stmt(
    context: "Context", stmt: NodeOrLeaf, seek_name: Optional[str] = None
) -> ValueSet:
    """
    The starting point of the completion. A statement always owns a call
    list, which are the calls, that a statement does. In case multiple
    names are defined in the statement, `seek_name` returns the result for
    this name.

    expr_stmt: testlist_star_expr (annassign | augassign (yield_expr|testlist) |
                     ('=' (yield_expr|testlist_star_expr))*)
    annassign: ':' test ['=' test]
    augassign: ('+=' | '-=' | '*=' | '@=' | '/=' | '%=' | '&=' | '|=' | '^=' |
                '<<=' | '>>=' | '**=' | '//=')

    :param stmt: A `tree.ExprStmt`.
    """

    def check_setitem(stmt_inner: NodeOrLeaf) -> Tuple[bool, Optional[NodeOrLeaf]]:
        atom_expr: NodeOrLeaf = stmt_inner.children[0]
        if atom_expr.type not in ("atom_expr", "power"):
            return (False, None)
        name: NodeOrLeaf = atom_expr.children[0]
        if name.type != "name" or len(atom_expr.children) != 2:
            return (False, None)
        trailer: NodeOrLeaf = atom_expr.children[-1]
        return (trailer.children[0] == "[", trailer.children[1])

    debug.dbg("infer_expr_stmt %s (%s)", stmt, seek_name)
    rhs: NodeOrLeaf = stmt.get_rhs()
    value_set: ValueSet = context.infer_node(rhs)
    if seek_name:
        n: TreeNameDefinition = TreeNameDefinition(context, seek_name)
        value_set = check_tuple_assignments(n, value_set)
    first_operator: Optional[NodeOrLeaf] = next(stmt.yield_operators(), None)
    is_setitem: bool
    subscriptlist: Optional[NodeOrLeaf]
    is_setitem, subscriptlist = check_setitem(stmt)
    is_annassign: bool = (
        first_operator not in ("=", None) and first_operator.type == "operator"
    )
    if is_annassign or is_setitem:
        name: str = stmt.get_defined_names(include_setitem=True)[0].value
        left_values: ValueSet = context.py__getattribute__(
            name, position=stmt.start_pos
        )
        if is_setitem:
            def to_mod(v: "Value") -> Union[ListModification, DictModification, "Value"]:
                c: ContextualizedNode = ContextualizedSubscriptListNode(
                    context, subscriptlist
                )
                if v.array_type == "dict":
                    return DictModification(v, value_set, c)
                elif v.array_type == "list":
                    return ListModification(v, value_set, c)
                return v

            value_set = ValueSet((to_mod(v) for v in left_values))
        else:
            operator: NodeOrLeaf = copy.copy(first_operator)
            operator.value = operator.value[:-1]
            for_stmt: Optional[NodeOrLeaf] = tree.search_ancestor(stmt, "for_stmt")
            if (
                for_stmt is not None
                and for_stmt.type == "for_stmt"
                and value_set
                and parser_utils.for_stmt_defines_one_name(for_stmt)
            ):
                node: NodeOrLeaf = for_stmt.get_testlist()
                cn: ContextualizedNode = ContextualizedNode(context, node)
                ordered: List[ValueSet] = list(cn.infer().iterate(cn))
                for lazy_value in ordered:
                    dct: Dict[str, ValueSet] = {for_stmt.children[1].value: lazy_value.infer()}
                    with context.predefine_names(for_stmt, dct):
                        t: ValueSet = context.infer_node(rhs)
                        left_values = _infer_comparison(context, left_values, operator, t)
                value_set = left_values
            else:
                value_set = _infer_comparison(context, left_values, operator, value_set)
    debug.dbg("infer_expr_stmt result %s", value_set)
    return value_set


def infer_or_test(context: "Context", or_test: NodeOrLeaf) -> ValueSet:
    iterator: Iterator[NodeOrLeaf] = iter(or_test.children)
    types: ValueSet = context.infer_node(next(iterator))
    for operator in iterator:
        right: NodeOrLeaf = next(iterator)
        if operator.type == "comp_op":
            operator = " ".join((c.value for c in operator.children))
        if operator in ("and", "or"):
            left_bools: set = set((left.py__bool__() for left in types))
            if left_bools == {True}:
                if operator == "and":
                    types = context.infer_node(right)
            elif left_bools == {False}:
                if operator != "and":
                    types = context.infer_node(right)
        else:
            types = _infer_comparison(context, types, operator, context.infer_node(right))
    debug.dbg("infer_or_test types %s", types)
    return types


@iterator_to_value_set
def infer_factor(value_set: ValueSet, operator: NodeOrLeaf) -> Iterator["Value"]:
    """
    Calculates `+`, `-`, `~` and `not` prefixes.
    """
    for value in value_set:
        if operator == "-":
            if is_number(value):
                yield value.negate()
        elif operator == "not":
            b: Optional[bool] = value.py__bool__()
            if b is None:
                return
            yield compiled.create_simple_object(
                value.inference_state, not b
            )
        else:
            yield value


def _literals_to_types(inference_state: "InferenceState", result: ValueSet) -> ValueSet:
    new_result: ValueSet = NO_VALUES
    for typ in result:
        if is_literal(typ):
            cls = compiled.builtin_from_name(
                inference_state, typ.name.string_name
            )
            new_result |= cls.execute_with_values()
        else:
            new_result |= ValueSet([typ])
    return new_result


def _infer_comparison(
    context: "Context",
    left_values: ValueSet,
    operator: NodeOrLeaf,
    right_values: ValueSet,
) -> ValueSet:
    state = context.inference_state
    if not left_values or not right_values:
        result: ValueSet = (left_values or NO_VALUES) | (right_values or NO_VALUES)
        return _literals_to_types(state, result)
    elif len(left_values) * len(right_values) > 6:
        return _literals_to_types(state, left_values | right_values)
    else:
        return ValueSet.from_sets(
            (
                _infer_comparison_part(state, context, left, operator, right)
                for left in left_values
                for right in right_values
            )
        )


def _is_annotation_name(name: NodeOrLeaf) -> bool:
    ancestor: Optional[NodeOrLeaf] = tree.search_ancestor(
        name, "param", "funcdef", "expr_stmt"
    )
    if ancestor is None:
        return False
    if ancestor.type in ("param", "funcdef"):
        ann: Optional[NodeOrLeaf] = ancestor.annotation
        if ann is not None:
            return ann.start_pos <= name.start_pos < ann.end_pos
    elif ancestor.type == "expr_stmt":
        c: List[Any] = ancestor.children
        if len(c) > 1 and c[1].type == "annassign":
            return c[1].start_pos <= name.start_pos < c[1].end_pos
    return False


def _is_list(value: "Value") -> bool:
    return value.array_type == "list"


def _is_tuple(value: "Value") -> bool:
    return value.array_type == "tuple"


def _bool_to_value(inference_state: "InferenceState", bool_: bool) -> "Value":
    return compiled.builtin_from_name(inference_state, str(bool_))


def _get_tuple_ints(value: "Value") -> Optional[List[int]]:
    if not isinstance(value, iterable.SequenceLiteralValue):
        return None
    numbers: List[int] = []
    for lazy_value in value.py__iter__():
        if not isinstance(lazy_value, LazyTreeValue):
            return None
        node: NodeOrLeaf = lazy_value.data
        if node.type != "number":
            return None
        try:
            numbers.append(int(node.value))
        except ValueError:
            return None
    return numbers


def _infer_comparison_part(
    inference_state: "InferenceState",
    context: "Context",
    left: "Value",
    operator: NodeOrLeaf,
    right: "Value",
) -> ValueSet:
    l_is_num: bool = is_number(left)
    r_is_num: bool = is_number(right)
    if isinstance(operator, str):
        str_operator: str = operator
    else:
        str_operator: str = str(operator.value)
    if str_operator == "*":
        if isinstance(left, iterable.Sequence) or is_string(left):
            return ValueSet([left])
        elif isinstance(right, iterable.Sequence) or is_string(right):
            return ValueSet([right])
    elif str_operator == "+":
        if (l_is_num and r_is_num) or (is_string(left) and is_string(right)):
            return left.execute_operation(right, str_operator)
        elif (_is_list(left) and _is_list(right)) or (_is_tuple(left) and _is_tuple(right)):
            return ValueSet([iterable.MergedArray(inference_state, (left, right))])
    elif str_operator == "-":
        if l_is_num and r_is_num:
            return left.execute_operation(right, str_operator)
    elif str_operator == "%":
        return ValueSet([left])
    elif str_operator in COMPARISON_OPERATORS:
        if left.is_compiled() and right.is_compiled():
            result: Optional[ValueSet] = left.execute_operation(right, str_operator)
            if result:
                return result
        else:
            if str_operator in ("is", "!=", "==", "is not"):
                operation = COMPARISON_OPERATORS[str_operator]
                bool_: bool = operation(left, right)
                if (str_operator in ("is", "==")) == bool_:
                    return ValueSet([_bool_to_value(inference_state, bool_)])
            if isinstance(left, VersionInfo):
                version_info: Optional[List[int]] = _get_tuple_ints(right)
                if version_info is not None:
                    bool_result: bool = compiled.access.COMPARISON_OPERATORS[
                        operator
                    ](inference_state.environment.version_info, tuple(version_info))
                    return ValueSet([_bool_to_value(inference_state, bool_result)])
        return ValueSet(
            [_bool_to_value(inference_state, True), _bool_to_value(inference_state, False)]
        )
    elif str_operator in ("in", "not in"):
        return NO_VALUES

    def check(obj: Any) -> bool:
        """Checks if a Jedi object is either a float or an int."""
        return isinstance(obj, TreeInstance) and obj.name.string_name in ("int", "float")

    if (
        str_operator in ("+", "-")
        and l_is_num != r_is_num
        and (not (check(left) or check(right)))
    ):
        message: str = "TypeError: unsupported operand type(s) for +: %s and %s"
        analysis.add(
            context, "type-error-operation", operator, message % (left, right)
        )
    if left.is_class() or right.is_class():
        return NO_VALUES
    method_name: str = operator_to_magic_method.get(str_operator, "")
    magic_methods: ValueSet = left.py__getattribute__(method_name)
    if magic_methods:
        result: ValueSet = magic_methods.execute_with_values(right)
        if result:
            return result
    if not magic_methods:
        reverse_method_name: str = reverse_operator_to_magic_method.get(str_operator, "")
        magic_methods = right.py__getattribute__(reverse_method_name)
        result: ValueSet = magic_methods.execute_with_values(left)
        if result:
            return result
    result = ValueSet([left, right])
    debug.dbg("Used operator %s resulting in %s", operator, result)
    return result


@plugin_manager.decorate()
def tree_name_to_values(
    inference_state: "InferenceState",
    context: "Context",
    tree_name: NodeOrLeaf,
) -> ValueSet:
    value_set: ValueSet = NO_VALUES
    module_node: Optional[NodeOrLeaf] = context.get_root_context().tree_node
    if module_node is not None:
        names: List[NodeOrLeaf] = module_node.get_used_names().get(tree_name.value, [])
        found_annotation: bool = False
        for name in names:
            expr_stmt: NodeOrLeaf = name.parent
            if expr_stmt.type == "expr_stmt" and expr_stmt.children[1].type == "annassign":
                correct_scope: bool = parser_utils.get_parent_scope(name) == context.tree_node
                if correct_scope:
                    found_annotation = True
                    value_set |= annotation.infer_annotation(
                        context, expr_stmt.children[1].children[1]
                    ).execute_annotation()
        if found_annotation:
            return value_set
    types: ValueSet = ValueSet()
    node: Optional[NodeOrLeaf] = tree_name.get_definition(
        import_name_always=True, include_setitem=True
    )
    if node is None:
        node = tree_name.parent
        if node.type == "global_stmt":
            c: "Context" = context.create_context(tree_name)
            if c.is_module():
                return NO_VALUES
            filter_obj = next(c.get_filters())
            names: List["Name"] = filter_obj.get(tree_name.value)
            return ValueSet.from_sets((name.infer() for name in names))
        elif node.type not in ("import_from", "import_name"):
            c: "Context" = context.create_context(tree_name)
            return infer_atom(c, tree_name)
    typ: str = node.type
    if typ == "for_stmt":
        types = annotation.find_type_from_comment_hint_for(context, node, tree_name)
        if types:
            return types
    if typ == "with_stmt":
        types = annotation.find_type_from_comment_hint_with(context, node, tree_name)
        if types:
            return types
    if typ in ("for_stmt", "comp_for", "sync_comp_for"):
        try:
            types = context.predefined_names[node][tree_name.value]
        except KeyError:
            cn: ContextualizedNode = ContextualizedNode(context, node.children[3])
            for_types: Iterator["Value"] = iterate_values(
                cn.infer(), contextualized_node=cn, is_async=node.parent.type == "async_stmt"
            )
            n: TreeNameDefinition = TreeNameDefinition(context, tree_name)
            types = check_tuple_assignments(n, for_types)
    elif typ == "expr_stmt":
        types = infer_expr_stmt(context, node, tree_name)
    elif typ == "with_stmt":
        value_managers: ValueSet = context.infer_node(
            node.get_test_node_from_name(tree_name)
        )
        if node.parent.type == "async_stmt":
            enter_methods: ValueSet = value_managers.py__getattribute__("__aenter__")
            coro: ValueSet = enter_methods.execute_with_values()
            return coro.py__await__().py__stop_iteration_returns()
        enter_methods: ValueSet = value_managers.py__getattribute__("__enter__")
        return enter_methods.execute_with_values()
    elif typ in ("import_from", "import_name"):
        types = imports.infer_import(context, tree_name)
    elif typ in ("funcdef", "classdef"):
        types = _apply_decorators(context, node)
    elif typ == "try_stmt":
        exceptions: ValueSet = context.infer_node(
            tree_name.get_previous_sibling().get_previous_sibling()
        )
        types = exceptions.execute_with_values()
    elif typ == "param":
        types = NO_VALUES
    elif typ == "del_stmt":
        types = NO_VALUES
    elif typ == "namedexpr_test":
        types = infer_node(context, node)
    else:
        raise ValueError("Should not happen. type: %s" % typ)
    return types


@inference_state_method_cache()
def _apply_decorators(context: "Context", node: NodeOrLeaf) -> ValueSet:
    """
    Returns the function, that should to be executed in the end.
    This is also the places where the decorators are processed.
    """
    if node.type == "classdef":
        decoratee_value: ClassValue = ClassValue(
            context.inference_state, parent_context=context, tree_node=node
        )
    else:
        decoratee_value: FunctionValue = FunctionValue.from_context(context, node)
    initial: ValueSet = values: ValueSet = ValueSet([decoratee_value])
    if is_big_annoying_library(context):
        return values
    for dec in reversed(node.get_decorators()):
        debug.dbg("decorator: %s %s", dec, values, color="MAGENTA")
        with debug.increase_indent_cm():
            dec_values: ValueSet = context.infer_node(dec.children[1])
            trailer_nodes: List[NodeOrLeaf] = dec.children[2:-1]
            if trailer_nodes:
                trailer: tree.PythonNode = tree.PythonNode("trailer", trailer_nodes)
                trailer.parent = dec
                dec_values = infer_trailer(context, dec_values, trailer)
            if not len(dec_values):
                code: str = dec.get_code(include_prefix=False)
                if code != "@runtime\n":
                    debug.warning("decorator not found: %s on %s", dec, node)
                return initial
            values = dec_values.execute(arguments.ValuesArguments([values]))
            if not len(values):
                debug.warning("not possible to resolve wrappers found %s", node)
                return initial
        debug.dbg("decorator end %s", values, color="MAGENTA")
    if values != initial:
        return ValueSet([Decoratee(c, decoratee_value) for c in values])
    return values


def check_tuple_assignments(name: "TreeNameDefinition", value_set: ValueSet) -> ValueSet:
    """
    Checks if tuples are assigned.
    """
    lazy_value: Optional["LazyValue"] = None
    for index, node in name.assignment_indexes():
        cn: ContextualizedNode = ContextualizedNode(name.parent_context, node)
        iterated: Iterator["Value"] = value_set.iterate(cn)
        if isinstance(index, slice):
            return NO_VALUES
        i: int = 0
        while i <= index:
            try:
                lazy_value = next(iterated)
            except StopIteration:
                return NO_VALUES
            else:
                i += lazy_value.max
        value_set = lazy_value.infer()
    return value_set


class ContextualizedSubscriptListNode(ContextualizedNode):
    def infer(self) -> ValueSet:
        return _infer_subscript_list(self.context, self.node)


def _infer_subscript_list(context: "Context", index: NodeOrLeaf) -> ValueSet:
    """
    Handles slices in subscript nodes.
    """
    if index == ":":
        return ValueSet([iterable.Slice(context, None, None, None)])
    elif index.type == "subscript" and not index.children[0] == ".":
        result: List[Optional[NodeOrLeaf]] = []
        for el in index.children:
            if el == ":":
                if not result:
                    result.append(None)
            elif el.type == "sliceop":
                if len(el.children) == 2:
                    result.append(el.children[1])
            else:
                result.append(el)
        result += [None] * (3 - len(result))
        return ValueSet([iterable.Slice(context, *result)])
    elif index.type == "subscriptlist":
        return ValueSet([iterable.SequenceLiteralValue(context.inference_state, context, index)])
    return context.infer_node(index)


def _apply_decorators(context: "Context", node: NodeOrLeaf) -> ValueSet:
    """
    Returns the function, that should to be executed in the end.
    This is also the places where the decorators are processed.
    """
    if node.type == "classdef":
        decoratee_value: ClassValue = ClassValue(
            context.inference_state, parent_context=context, tree_node=node
        )
    else:
        decoratee_value: FunctionValue = FunctionValue.from_context(context, node)
    initial: ValueSet = values: ValueSet = ValueSet([decoratee_value])
    if is_big_annoying_library(context):
        return values
    for dec in reversed(node.get_decorators()):
        debug.dbg("decorator: %s %s", dec, values, color="MAGENTA")
        with debug.increase_indent_cm():
            dec_values: ValueSet = context.infer_node(dec.children[1])
            trailer_nodes: List[NodeOrLeaf] = dec.children[2:-1]
            if trailer_nodes:
                trailer: tree.PythonNode = tree.PythonNode("trailer", trailer_nodes)
                trailer.parent = dec
                dec_values = infer_trailer(context, dec_values, trailer)
            if not len(dec_values):
                code: str = dec.get_code(include_prefix=False)
                if code != "@runtime\n":
                    debug.warning("decorator not found: %s on %s", dec, node)
                return initial
            values = dec_values.execute(arguments.ValuesArguments([values]))
            if not len(values):
                debug.warning("not possible to resolve wrappers found %s", node)
                return initial
        debug.dbg("decorator end %s", values, color="MAGENTA")
    if values != initial:
        return ValueSet([Decoratee(c, decoratee_value) for c in values])
    return values


def check_tuple_assignments(name: "TreeNameDefinition", value_set: ValueSet) -> ValueSet:
    """
    Checks if tuples are assigned.
    """
    lazy_value: Optional["LazyValue"] = None
    for index, node in name.assignment_indexes():
        cn: ContextualizedNode = ContextualizedNode(name.parent_context, node)
        iterated: Iterator["Value"] = value_set.iterate(cn)
        if isinstance(index, slice):
            return NO_VALUES
        i: int = 0
        while i <= index:
            try:
                lazy_value = next(iterated)
            except StopIteration:
                return NO_VALUES
            else:
                i += lazy_value.max
        value_set = lazy_value.infer()
    return value_set


class ContextualizedSubscriptListNode(ContextualizedNode):
    def infer(self) -> ValueSet:
        return _infer_subscript_list(self.context, self.node)
