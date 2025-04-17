import re
from itertools import zip_longest
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, LazyTreeValue, get_merged_lazy_value
from jedi.inference.names import ParamName, TreeNameDefinition, AnonymousParamName
from jedi.inference.base_value import NO_VALUES, ValueSet, ContextualizedNode
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache


def try_iter_content(types, depth=0):
    """Helper method for static analysis."""
    if depth > 10:
        return
    for typ in types:
        try:
            f = typ.py__iter__
        except AttributeError:
            pass
        else:
            for lazy_value in f():
                try_iter_content(lazy_value.infer(), depth + 1)


class ParamIssue(Exception):
    pass


def repack_with_argument_clinic(clinic_string):
    """
    Transforms a function or method with arguments to the signature that is
    given as an argument clinic notation.

    Argument clinic is part of CPython and used for all the functions that are
    implemented in C (Python 3.7):

        str.split.__text_signature__
        # Results in: '($self, /, sep=None, maxsplit=-1)'
    """

    def decorator(func):

        def wrapper(value, arguments):
            try:
                args = tuple(iterate_argument_clinic(value.inference_state,
                    arguments, clinic_string))
            except ParamIssue:
                return NO_VALUES
            else:
                return func(value, *args)
        return wrapper
    return decorator


def iterate_argument_clinic(inference_state, arguments, clinic_string):
    """Uses a list with argument clinic information (see PEP 436)."""
    clinic_args = list(_parse_argument_clinic(clinic_string))
    iterator = PushBackIterator(arguments.unpack())
    for i, (name, optional, allow_kwargs, stars) in enumerate(clinic_args):
        if stars == 1:
            lazy_values: List[Any] = []
            for key, argument in iterator:
                if key is not None:
                    iterator.push_back((key, argument))
                    break
                lazy_values.append(argument)
            yield iterable.FakeTuple(inference_state, lazy_values)
            continue
        elif stars == 2:
            raise NotImplementedError()
        key, argument = next(iterator, (None, None))
        if key is not None:
            debug.warning(
                'Keyword arguments in argument clinic are currently not supported.'
                )
            raise ParamIssue
        if argument is None and not optional:
            debug.warning(
                'TypeError: %s expected at least %s arguments, got %s',
                name, len(clinic_args), i)
            raise ParamIssue
        value_set: ValueSet = (NO_VALUES if argument is None else argument.
            infer())
        if not value_set and not optional:
            debug.warning('argument_clinic "%s" not resolvable.', name)
            raise ParamIssue
        yield value_set


def _parse_argument_clinic(string):
    allow_kwargs = False
    optional = False
    while string:
        match = re.match('(?:(?:(\\[),? ?|, ?|)(\\**\\w+)|, ?/)\\]*', string)
        if not match:
            break
        string = string[len(match.group(0)):]
        if not match.group(2):
            allow_kwargs = True
            continue
        optional = optional or bool(match.group(1))
        word = match.group(2)
        stars = word.count('*')
        word = word[stars:]
        yield word, optional, allow_kwargs, stars
        if stars:
            allow_kwargs = True


class _AbstractArgumentsMixin:

    def unpack(self, funcdef=None):
        raise NotImplementedError

    def get_calling_nodes(self):
        return []


class AbstractArguments(_AbstractArgumentsMixin):
    context: Any
    argument_node: Any
    trailer: Optional[Any]


def unpack_arglist(arglist):
    if arglist is None:
        return
    if arglist.type != 'arglist' and not (arglist.type == 'argument' and 
        arglist.children[0] in ('*', '**')):
        yield 0, arglist
        return
    iterator = iter(arglist.children)
    for child in iterator:
        if child == ',':
            continue
        elif child in ('*', '**'):
            c = next(iterator, None)
            assert c is not None
            yield len(child.value), c
        elif child.type == 'argument' and child.children[0] in ('*', '**'):
            assert len(child.children) == 2
            yield len(child.children[0].value), child.children[1]
        else:
            yield 0, child


class TreeArguments(AbstractArguments):
    context: Any
    argument_node: Any
    trailer: Optional[Any]
    _inference_state: Any

    def __init__(self, inference_state, context, argument_node, trailer=None):
        """
        :param argument_node: May be an argument_node or a list of nodes.
        """
        self.argument_node = argument_node
        self.context = context
        self._inference_state = inference_state
        self.trailer = trailer

    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls, inference_state, context, argument_node, trailer
        =None):
        return cls(inference_state, context, argument_node, trailer)

    def unpack(self, funcdef=None):
        named_args: List[Tuple[str, LazyTreeValue]] = []
        for star_count, el in unpack_arglist(self.argument_node):
            if star_count == 1:
                arrays = self.context.infer_node(el)
                iterators = [_iterate_star_args(self.context, a, el,
                    funcdef) for a in arrays]
                for values in list(zip_longest(*iterators)):
                    merged = get_merged_lazy_value([v for v in values if v
                         is not None])
                    yield None, merged
            elif star_count == 2:
                arrays = self.context.infer_node(el)
                for dct in arrays:
                    yield from _star_star_dict(self.context, dct, el, funcdef)
            elif el.type == 'argument':
                c = el.children
                if len(c) == 3:
                    named_args.append((c[0].value, LazyTreeValue(self.
                        context, c[2])))
                else:
                    sync_comp_for = el.children[1]
                    if sync_comp_for.type == 'comp_for':
                        sync_comp_for = sync_comp_for.children[1]
                    comp = iterable.GeneratorComprehension(self.
                        _inference_state, defining_context=self.context,
                        sync_comp_for_node=sync_comp_for, entry_node=el.
                        children[0])
                    yield None, LazyKnownValue(comp)
            else:
                yield None, LazyTreeValue(self.context, el)
        for name, value in named_args:
            yield None, value

    def _as_tree_tuple_objects(self):
        for star_count, argument in unpack_arglist(self.argument_node):
            default: Optional[Any] = None
            if argument.type == 'argument':
                if len(argument.children) == 3:
                    argument, default = argument.children[::2]
            yield argument, default, star_count

    def iter_calling_names_with_star(self):
        for name, default, star_count in self._as_tree_tuple_objects():
            if not star_count or not isinstance(name, tree.Name):
                continue
            yield TreeNameDefinition(self.context, name)

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.argument_node}>'

    def get_calling_nodes(self):
        old_arguments_list: List['TreeArguments'] = []
        arguments: AbstractArguments = self
        while arguments not in old_arguments_list:
            if not isinstance(arguments, TreeArguments):
                break
            old_arguments_list.append(arguments)
            for calling_name in reversed(list(arguments.
                iter_calling_names_with_star())):
                names = calling_name.goto()
                if len(names) != 1:
                    break
                if isinstance(names[0], AnonymousParamName):
                    return []
                if not isinstance(names[0], ParamName):
                    break
                executed_param_name = names[0].get_executed_param_name()
                arguments = executed_param_name.arguments
                break
        else:
            pass
        if arguments.argument_node is not None:
            return [ContextualizedNode(arguments.context, arguments.
                argument_node)]
        if arguments.trailer is not None:
            return [ContextualizedNode(arguments.context, arguments.trailer)]
        return []


class ValuesArguments(AbstractArguments):
    _values_list: List[Any]

    def __init__(self, values_list):
        self._values_list = values_list

    def unpack(self, funcdef=None):
        for values in self._values_list:
            yield None, LazyKnownValues(values)

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self._values_list}>'


class TreeArgumentsWrapper(_AbstractArgumentsMixin):
    _wrapped_arguments: AbstractArguments

    def __init__(self, arguments):
        self._wrapped_arguments = arguments

    @property
    def context(self):
        return self._wrapped_arguments.context

    @property
    def argument_node(self):
        return self._wrapped_arguments.argument_node

    @property
    def trailer(self):
        return self._wrapped_arguments.trailer

    def unpack(self, funcdef=None):
        raise NotImplementedError

    def get_calling_nodes(self):
        return self._wrapped_arguments.get_calling_nodes()

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self._wrapped_arguments}>'


def _iterate_star_args(context, array, input_node, funcdef=None):
    if not array.py__getattribute__('__iter__'):
        if funcdef is not None:
            m = (
                f'TypeError: {funcdef.name.value}() argument after * must be a sequence, not {array}'
                )
            analysis.add(context, 'type-error-star', input_node, message=m)
    try:
        iter_: Callable[[], Iterator[Any]] = array.py__iter__
    except AttributeError:
        pass
    else:
        yield from iter_()


def _star_star_dict(context, array, input_node, funcdef):
    from jedi.inference.value.instance import CompiledInstance
    if isinstance(array, CompiledInstance
        ) and array.name.string_name == 'dict':
        return {}
    elif isinstance(array, iterable.Sequence) and array.array_type == 'dict':
        return array.exact_key_items()
    else:
        if funcdef is not None:
            m = (
                f'TypeError: {funcdef.name.value} argument after ** must be a mapping, not {array}'
                )
            analysis.add(context, 'type-error-star-star', input_node, message=m
                )
        return {}
