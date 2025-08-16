"""
Implementations of standard library functions, because it's not possible to
understand them with Jedi.

To add a new implementation, create a function and add it to the
``_implemented`` dict at the bottom of this module.

Note that this module exists only to implement very specific functionality in
the standard library. The usual way to understand the standard library is the
compiled module that returns the types for C-builtins.
"""
import parso
import os
from inspect import Parameter
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar, Union
)

from jedi import debug
from jedi.inference.utils import safe_property
from jedi.inference.helpers import get_str_or_none
from jedi.inference.arguments import iterate_argument_clinic, ParamIssue, \
    repack_with_argument_clinic, AbstractArguments, TreeArgumentsWrapper
from jedi.inference import analysis
from jedi.inference import compiled
from jedi.inference.value.instance import \
    AnonymousMethodExecutionContext, MethodExecutionContext
from jedi.inference.base_value import ContextualizedNode, \
    NO_VALUES, ValueSet, ValueWrapper, LazyValueWrapper
from jedi.inference.value import ClassValue, ModuleValue
from jedi.inference.value.klass import ClassMixin
from jedi.inference.value.function import FunctionMixin
from jedi.inference.value import iterable
from jedi.inference.lazy_value import LazyTreeValue, LazyKnownValue, \
    LazyKnownValues
from jedi.inference.names import ValueName, BaseTreeParamName
from jedi.inference.filters import AttributeOverwrite, publish_method, \
    ParserTreeFilter, DictFilter
from jedi.inference.signature import AbstractSignature, SignatureWrapper


# Copied from Python 3.6's stdlib.
_NAMEDTUPLE_CLASS_TEMPLATE = """\
_property = property
_tuple = tuple
from operator import itemgetter as _itemgetter
from collections import OrderedDict

class {typename}(tuple):
    __slots__ = ()

    _fields = {field_names!r}

    def __new__(_cls, {arg_list}):
        'Create new instance of {typename}({arg_list})'
        return _tuple.__new__(_cls, ({arg_list}))

    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        'Make a new {typename} object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != {num_fields:d}:
            raise TypeError('Expected {num_fields:d} arguments, got %d' % len(result))
        return result

    def _replace(_self, **kwds):
        'Return a new {typename} object replacing specified fields with new values'
        result = _self._make(map(kwds.pop, {field_names!r}, _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds))
        return result

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + '({repr_fmt})' % self

    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values.'
        return OrderedDict(zip(self._fields, self))

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)

    # These methods were added by Jedi.
    # __new__ doesn't really work with Jedi. So adding this to nametuples seems
    # like the easiest way.
    def __init__(self, {arg_list}):
        'A helper function for namedtuple.'
        self.__iterable = ({arg_list})

    def __iter__(self):
        for i in self.__iterable:
            yield i

    def __getitem__(self, y):
        return self.__iterable[y]

{field_defs}
"""

_NAMEDTUPLE_FIELD_TEMPLATE = '''\
    {name} = _property(_itemgetter({index:d}), doc='Alias for field number {index:d}')
'''

T = TypeVar('T')
Callback = Callable[[], ValueSet]

def execute(callback: Callable[[], ValueSet]) -> Callable[[Any, AbstractArguments], ValueSet]:
    def wrapper(value: Any, arguments: AbstractArguments) -> ValueSet:
        def call() -> ValueSet:
            return callback()

        try:
            obj_name = value.name.string_name
        except AttributeError:
            pass
        else:
            p = value.parent_context
            if p is not None and p.is_builtins_module():
                module_name = 'builtins'
            elif p is not None and p.is_module():
                module_name = p.py__name__()
            else:
                return call()

            if value.is_bound_method() or value.is_instance():
                return call()

            try:
                func = _implemented[module_name][obj_name]
            except KeyError:
                pass
            else:
                return func(value, arguments=arguments, callback=call)
        return call()

    return wrapper


def _follow_param(inference_state: Any, arguments: AbstractArguments, index: int) -> ValueSet:
    try:
        key, lazy_value = list(arguments.unpack())[index]
    except IndexError:
        return NO_VALUES
    else:
        return lazy_value.infer()


def argument_clinic(
    clinic_string: str,
    want_value: bool = False,
    want_context: bool = False,
    want_arguments: bool = False,
    want_inference_state: bool = False,
    want_callback: bool = False
) -> Callable[[Callable[..., ValueSet]], Callable[[Any, AbstractArguments, Callback], ValueSet]]:
    def f(func: Callable[..., ValueSet]) -> Callable[[Any, AbstractArguments, Callback], ValueSet]:
        def wrapper(value: Any, arguments: AbstractArguments, callback: Callback) -> ValueSet:
            try:
                args = tuple(iterate_argument_clinic(
                    value.inference_state, arguments, clinic_string))
            except ParamIssue:
                return NO_VALUES

            debug.dbg('builtin start %s' % value, color='MAGENTA')
            kwargs: Dict[str, Any] = {}
            if want_context:
                kwargs['context'] = arguments.context
            if want_value:
                kwargs['value'] = value
            if want_inference_state:
                kwargs['inference_state'] = value.inference_state
            if want_arguments:
                kwargs['arguments'] = arguments
            if want_callback:
                kwargs['callback'] = callback
            result = func(*args, **kwargs)
            debug.dbg('builtin end: %s', result, color='MAGENTA')
            return result

        return wrapper
    return f


@argument_clinic('iterator[, default], /', want_inference_state=True)
def builtins_next(iterators: ValueSet, defaults: ValueSet, inference_state: Any) -> ValueSet:
    return defaults | iterators.py__getattribute__('__next__').execute_with_values()


@argument_clinic('iterator[, default], /')
def builtins_iter(iterators_or_callables: ValueSet, defaults: ValueSet) -> ValueSet:
    return iterators_or_callables.py__getattribute__('__iter__').execute_with_values()


@argument_clinic('object, name[, default], /')
def builtins_getattr(objects: ValueSet, names: ValueSet, defaults: Optional[ValueSet] = None) -> ValueSet:
    for value in objects:
        for name in names:
            string = get_str_or_none(name)
            if string is None:
                debug.warning('getattr called without str')
                continue
            else:
                return value.py__getattribute__(string)
    return NO_VALUES


@argument_clinic('object[, bases, dict], /')
def builtins_type(objects: ValueSet, bases: ValueSet, dicts: ValueSet) -> ValueSet:
    if bases or dicts:
        return NO_VALUES
    else:
        return objects.py__class__()


class SuperInstance(LazyValueWrapper):
    def __init__(self, inference_state: Any, instance: Any) -> None:
        self.inference_state = inference_state
        self._instance = instance

    def _get_bases(self) -> ValueSet:
        return self._instance.py__class__().py__bases__()

    def _get_wrapped_value(self) -> Any:
        objs = self._get_bases()[0].infer().execute_with_values()
        if not objs:
            return self._instance
        return next(iter(objs))

    def get_filters(self, origin_scope: Optional[Any] = None) -> Iterator[Any]:
        for b in self._get_bases():
            for value in b.infer().execute_with_values():
                for f in value.get_filters():
                    yield f


@argument_clinic('[type[, value]], /', want_context=True)
def builtins_super(types: ValueSet, objects: ValueSet, context: Any) -> ValueSet:
    instance = None
    if isinstance(context, AnonymousMethodExecutionContext):
        instance = context.instance
    elif isinstance(context, MethodExecutionContext):
        instance = context.instance
    if instance is None:
        return NO_VALUES
    return ValueSet({SuperInstance(instance.inference_state, instance)})


class ReversedObject(AttributeOverwrite):
    def __init__(self, reversed_obj: Any, iter_list: List[Any]) -> None:
        super().__init__(reversed_obj)
        self._iter_list = iter_list

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> List[Any]:
        return self._iter_list

    @publish_method('__next__')
    def _next(self, arguments: AbstractArguments) -> ValueSet:
        return ValueSet.from_sets(
            lazy_value.infer() for lazy_value in self._iter_list
        )


@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_reversed(sequences: ValueSet, value: Any, arguments: AbstractArguments) -> ValueSet:
    key, lazy_value = next(arguments.unpack())
    cn = None
    if isinstance(lazy_value, LazyTreeValue):
        cn = ContextualizedNode(lazy_value.context, lazy_value.data)
    ordered = list(sequences.iterate(cn))

    seq, = value.inference_state.typing_module.py__getattribute__('Iterator').execute_with_values()
    return ValueSet([ReversedObject(seq, list(reversed(ordered)))])


@argument_clinic('value, type, /', want_arguments=True, want_inference_state=True)
def builtins_isinstance(objects: ValueSet, types: ValueSet, arguments: AbstractArguments, inference_state: Any) -> ValueSet:
    bool_results = set()
    for o in objects:
        cls = o.py__class__()
        try:
            cls.py__bases__
        except AttributeError:
            bool_results = set([True, False])
            break

        mro = list(cls.py__mro__())

        for cls_or_tup in types:
            if cls_or_tup.is_class():
                bool_results.add(cls_or_tup in mro)
            elif cls_or_tup.name.string_name == 'tuple' \
                    and cls_or_tup.get_root_context().is_builtins_module():
                classes = ValueSet.from_sets(
                    lazy_value.infer()
                    for lazy_value in cls_or_tup.iterate()
                )
                bool_results.add(any(cls in mro for cls in classes))
            else:
                _, lazy_value = list(arguments.unpack())[1]
                if isinstance(lazy_value, LazyTreeValue):
                    node = lazy_value.data
                    message = 'TypeError: isinstance() arg 2 must be a ' \
                              'class, type, or tuple of classes and types, ' \
                              'not %s.' % cls_or_tup
                    analysis.add(lazy_value.context, 'type-error-isinstance', node, message)

    return ValueSet(
        compiled.builtin_from_name(inference_state, str(b))
        for b in bool_results
    )


class StaticMethodObject(ValueWrapper):
    def py__get__(self, instance: Optional[Any], class_value: Optional[Any]) -> ValueSet:
        return ValueSet([self._wrapped_value])


@argument_clinic('sequence, /')
def builtins_staticmethod(functions: ValueSet) -> ValueSet:
    return ValueSet(StaticMethodObject(f) for f in functions)


class ClassMethodObject(ValueWrapper):
    def __init__(self, class_method_obj: Any, function: Any) -> None:
        super().__init__(class_method_obj)
        self._function = function

    def py__get__(self, instance: Optional[Any], class_value: Optional[Any]) -> ValueSet:
        return ValueSet([
            ClassMethodGet(__get__, class_value, self._function)
            for __get__ in self._wrapped_value.py__getattribute__('__get__')
        ])


class ClassMethodGet(ValueWrapper):
    def __init__(self, get_method: Any, klass: Any, function: Any) -> None:
        super().__init__(get_method)
        self._class = klass
        self._function = function

    def get_signatures(self) -> List[Any]:
        return [sig.bind(self._function) for sig in self._function.get_signatures()]

    def py__call__(self, arguments: AbstractArguments) -> ValueSet:
        return self._function.execute(ClassMethodArguments(self._class, arguments))


class ClassMethodArguments(TreeArgumentsWrapper):
    def __init__(self, klass: Any, arguments: AbstractArguments) -> None:
        super().__init__(arguments)
        self._class = klass

    def unpack(self, func: Optional[Any] = None) -> Iterator[Tuple[Optional[str], Any]]:
        yield None, LazyKnownValue(self._class)
        for values in self._wrapped_arguments.unpack(func):
            yield values


@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_classmethod(functions: ValueSet, value: Any, arguments: AbstractArguments) -> ValueSet:
    return ValueSet(
        ClassMethodObject(class_method_object, function)
        for class_method_object in value.py__call__(arguments=arguments)
        for function in functions
    )


class PropertyObject(AttributeOverwrite, ValueWrapper):
    api_type = 'property'

    def __init__(self, property_obj: Any, function: Any) -> None:
        super().__init__(property_obj)
        self._function = function

    def py__get__(self, instance: Optional[Any], class_value: Optional[Any]) -> ValueSet:
        if instance is None:
            return ValueSet([self])
        return self._function.execute_with_values(instance)

    @publish_method('deleter')
    @publish_method('getter')
    @publish_method('setter')
    def _return_self(self, arguments: AbstractArguments) -> ValueSet:
        return ValueSet({self})


@argument_clinic('func, /', want_callback=True)
def builtins_property(functions: ValueSet, callback: Callback) -> ValueSet:
    return ValueSet(
        PropertyObject(property_value, function)
        for property_value in callback()
        for function in functions
    )


def collections_namedtuple(value: Any, arguments: AbstractArguments, callback: Callback) -> ValueSet:
    inference_state = value.inference_state

    name = 'jedi_unknown_namedtuple'
    for c in _follow_param(inference_state, arguments, 0):
        x = get_str_or_none(c)
        if x is not None:
            name = x
            break

    param_values = _follow_param(inference_state, arguments, 1)
    if not param_values:
        return NO_VALUES
    _fields = list(param_values)[0]
    string = get_str_or_none(_fields)
    if string is not None:
        fields = string.replace(',', ' ').split()
    elif isinstance(_fields, iterable.Sequence):
        fields = [
            get_str_or_none(v)
            for lazy_value in _fields.py__iter__()
            for v in lazy_value.infer()
        ]
        fields = [f for f in fields if f is not None]
    else:
        return NO_VALUES

    code = _NAMEDTUPLE_CLASS_TEMPLATE.format(
        typename=name,
        field_names=tuple(fields),
        num_fields=len(fields),
        arg_list=repr(tuple(fields)).replace("'", "")[1:-1],
        repr_fmt='',
        field_defs='\n'.join(_NAMEDTUPLE_FIELD_TEMPLATE.format(index=index, name=name)
                             for index, name in enumerate(fields))
    )

    module = inference_state.grammar.parse(code)
    generated_class = next(module.iter_classdefs())
    parent_context = ModuleValue(
        inference_state, module,
        code_lines=parso.split_lines(code, keepends=True),
    ).as_context()

    return ValueSet([ClassValue(inference_state, parent_context, generated_class)])


class PartialObject(ValueWrapper):
    def __init__(self, actual_value: Any, arguments: AbstractArguments, instance: Optional[Any] = None) -> None:
        super().__init__(actual_value)
        self._arguments = arguments
        self._instance = instance

    def _get_functions(self, unpacked_arguments: Iterator[Tuple[Optional[str], Any]]) -> Optional[ValueSet]:
        key, lazy_value = next(unpacked_arguments, (None, None))
        if key is not None or lazy_value is None:
            debug.warning("Partial should have a proper function %s", self._arguments)
            return None
        return lazy_value.infer()

    def get_signatures(self) -> List[Any]:
        unpacked_arguments = self._arguments.unpack()
        funcs = self._get_functions(unpacked_arguments)
        if funcs is None:
            return []

        arg_count = 0
        if self._instance is not None:
            arg_count = 1
        keys = set()
        for key, _ in unpacked_arguments:
            if key is None:
                arg_count += 1
            else:
                keys.add(key)
        return [PartialSignature(s, arg_count, keys) for s in funcs.get_signatures()]

    def py__call