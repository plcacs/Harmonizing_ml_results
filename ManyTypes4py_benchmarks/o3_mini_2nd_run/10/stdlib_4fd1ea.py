from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union
import parso
import os
from inspect import Parameter
from jedi import debug
from jedi.inference.utils import safe_property
from jedi.inference.helpers import get_str_or_none
from jedi.inference.arguments import iterate_argument_clinic, ParamIssue, repack_with_argument_clinic, AbstractArguments, TreeArgumentsWrapper
from jedi.inference import analysis
from jedi.inference import compiled
from jedi.inference.value.instance import AnonymousMethodExecutionContext, MethodExecutionContext
from jedi.inference.base_value import ContextualizedNode, NO_VALUES, ValueSet, ValueWrapper, LazyValueWrapper
from jedi.inference.value import ClassValue, ModuleValue
from jedi.inference.value.klass import ClassMixin
from jedi.inference.value.function import FunctionMixin
from jedi.inference.value import iterable
from jedi.inference.lazy_value import LazyTreeValue, LazyKnownValue, LazyKnownValues
from jedi.inference.names import ValueName, BaseTreeParamName
from jedi.inference.filters import AttributeOverwrite, publish_method, ParserTreeFilter, DictFilter
from jedi.inference.signature import AbstractSignature, SignatureWrapper

_NAMEDTUPLE_CLASS_TEMPLATE: str = (
    "_property = property\n"
    "_tuple = tuple\n"
    "from operator import itemgetter as _itemgetter\n"
    "from collections import OrderedDict\n\n"
    "class {typename}(tuple):\n"
    "    __slots__ = ()\n\n"
    "    _fields = {field_names!r}\n\n"
    "    def __new__(_cls, {arg_list}):\n"
    "        'Create new instance of {typename}({arg_list})'\n"
    "        return _tuple.__new__(_cls, ({arg_list}))\n\n"
    "    @classmethod\n"
    "    def _make(cls, iterable, new=tuple.__new__, len=len):\n"
    "        'Make a new {typename} object from a sequence or iterable'\n"
    "        result = new(cls, iterable)\n"
    "        if len(result) != {num_fields:d}:\n"
    "            raise TypeError('Expected {num_fields:d} arguments, got %d' % len(result))\n"
    "        return result\n\n"
    "    def _replace(_self, **kwds):\n"
    "        'Return a new {typename} object replacing specified fields with new values'\n"
    "        result = _self._make(map(kwds.pop, {field_names!r}, _self))\n"
    "        if kwds:\n"
    "            raise ValueError('Got unexpected field names: %r' % list(kwds))\n"
    "        return result\n\n"
    "    def __repr__(self):\n"
    "        'Return a nicely formatted representation string'\n"
    "        return self.__class__.__name__ + '({repr_fmt})' % self\n\n"
    "    def _asdict(self):\n"
    "        'Return a new OrderedDict which maps field names to their values.'\n"
    "        return OrderedDict(zip(self._fields, self))\n\n"
    "    def __getnewargs__(self):\n"
    "        'Return self as a plain tuple.  Used by copy and pickle.'\n"
    "        return tuple(self)\n\n"
    "    # These methods were added by Jedi.\n"
    "    # __new__ doesn't really work with Jedi. So adding this to nametuples seems\n"
    "    # like the easiest way.\n"
    "    def __init__(self, {arg_list}):\n"
    "        'A helper function for namedtuple.'\n"
    "        self.__iterable = ({arg_list})\n\n"
    "    def __iter__(self):\n"
    "        for i in self.__iterable:\n"
    "            yield i\n\n"
    "    def __getitem__(self, y):\n"
    "        return self.__iterable[y]\n\n"
    "{field_defs}\n"
)
_NAMEDTUPLE_FIELD_TEMPLATE: str = "    {name} = _property(_itemgetter({index:d}), doc='Alias for field number {index:d}')\n"


def execute(callback: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def wrapper(value: Any, arguments: Any) -> Any:
        def call() -> Any:
            return callback(value, arguments=arguments)
        try:
            obj_name: str = value.name.string_name
        except AttributeError:
            pass
        else:
            p = value.parent_context
            if p is not None and p.is_builtins_module():
                module_name: str = 'builtins'
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


def _follow_param(inference_state: Any, arguments: Any, index: int) -> Any:
    try:
        key, lazy_value = list(arguments.unpack())[index]
    except IndexError:
        return NO_VALUES
    else:
        return lazy_value.infer()


def argument_clinic(clinic_string: str, want_value: bool = False, want_context: bool = False,
                      want_arguments: bool = False, want_inference_state: bool = False,
                      want_callback: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Works like Argument Clinic (PEP 436), to validate function params.
    """
    def f(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(value: Any, arguments: Any, callback: Callable[..., Any]) -> Any:
            try:
                args = tuple(iterate_argument_clinic(value.inference_state, arguments, clinic_string))
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
def builtins_next(iterators: Any, defaults: Any, inference_state: Any) -> Any:
    return defaults | iterators.py__getattribute__('__next__').execute_with_values()


@argument_clinic('iterator[, default], /')
def builtins_iter(iterators_or_callables: Any, defaults: Any) -> Any:
    return iterators_or_callables.py__getattribute__('__iter__').execute_with_values()


@argument_clinic('object, name[, default], /')
def builtins_getattr(objects: Any, names: Any, defaults: Optional[Any] = None) -> Any:
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
def builtins_type(objects: Any, bases: Any, dicts: Any) -> Any:
    if bases or dicts:
        return NO_VALUES
    else:
        return objects.py__class__()


class SuperInstance(LazyValueWrapper):
    """To be used like the object ``super`` returns."""

    def __init__(self, inference_state: Any, instance: Any) -> None:
        self.inference_state = inference_state
        self._instance = instance

    def _get_bases(self) -> Any:
        return self._instance.py__class__().py__bases__()

    def _get_wrapped_value(self) -> Any:
        objs = self._get_bases()[0].infer().execute_with_values()
        if not objs:
            return self._instance
        return next(iter(objs))

    def get_filters(self, origin_scope: Optional[Any] = None) -> Any:
        for b in self._get_bases():
            for value in b.infer().execute_with_values():
                for f in value.get_filters():
                    yield f


@argument_clinic('[type[, value]], /', want_context=True)
def builtins_super(types: Any, objects: Any, context: Any) -> Any:
    instance = None
    if isinstance(context, AnonymousMethodExecutionContext):
        instance = context.instance
    elif isinstance(context, MethodExecutionContext):
        instance = context.instance
    if instance is None:
        return NO_VALUES
    return ValueSet({SuperInstance(instance.inference_state, instance)})


class ReversedObject(AttributeOverwrite):

    def __init__(self, reversed_obj: Any, iter_list: Any) -> None:
        super().__init__(reversed_obj)
        self._iter_list = iter_list

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Any:
        return self._iter_list

    @publish_method('__next__')
    def _next(self, arguments: Any) -> Any:
        return ValueSet.from_sets((lazy_value.infer() for lazy_value in self._iter_list))


@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_reversed(sequences: Any, value: Any, arguments: Any) -> Any:
    key, lazy_value = next(arguments.unpack())
    cn = None
    if isinstance(lazy_value, LazyTreeValue):
        cn = ContextualizedNode(lazy_value.context, lazy_value.data)
    ordered = list(sequences.iterate(cn))
    seq, = value.inference_state.typing_module.py__getattribute__('Iterator').execute_with_values()
    return ValueSet([ReversedObject(seq, list(reversed(ordered)))])


@argument_clinic('value, type, /', want_arguments=True, want_inference_state=True)
def builtins_isinstance(objects: Any, types: Any, arguments: Any, inference_state: Any) -> Any:
    bool_results: Set[bool] = set()
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
            elif cls_or_tup.name.string_name == 'tuple' and cls_or_tup.get_root_context().is_builtins_module():
                classes = ValueSet.from_sets((lazy_value.infer() for lazy_value in cls_or_tup.iterate()))
                bool_results.add(any((cls in mro for cls in classes)))
            else:
                _, lazy_value = list(arguments.unpack())[1]
                if isinstance(lazy_value, LazyTreeValue):
                    node = lazy_value.data
                    message = 'TypeError: isinstance() arg 2 must be a class, type, or tuple of classes and types, not %s.' % cls_or_tup
                    analysis.add(lazy_value.context, 'type-error-isinstance', node, message)
    return ValueSet((compiled.builtin_from_name(inference_state, str(b)) for b in bool_results))


class StaticMethodObject(ValueWrapper):

    def py__get__(self, instance: Any, class_value: Any) -> Any:
        return ValueSet([self._wrapped_value])


@argument_clinic('sequence, /')
def builtins_staticmethod(functions: Any) -> Any:
    return ValueSet((StaticMethodObject(f) for f in functions))


class ClassMethodObject(ValueWrapper):

    def __init__(self, class_method_obj: Any, function: Any) -> None:
        super().__init__(class_method_obj)
        self._function = function

    def py__get__(self, instance: Any, class_value: Any) -> Any:
        return ValueSet([ClassMethodGet(__get__, class_value, self._function) for __get__ in self._wrapped_value.py__getattribute__('__get__')])


class ClassMethodGet(ValueWrapper):

    def __init__(self, get_method: Any, klass: Any, function: Any) -> None:
        super().__init__(get_method)
        self._class = klass
        self._function = function

    def get_signatures(self) -> List[Any]:
        return [sig.bind(self._function) for sig in self._function.get_signatures()]

    def py__call__(self, arguments: Any) -> Any:
        return self._function.execute(ClassMethodArguments(self._class, arguments))


class ClassMethodArguments(TreeArgumentsWrapper):

    def __init__(self, klass: Any, arguments: Any) -> None:
        super().__init__(arguments)
        self._class = klass

    def unpack(self, func: Optional[Any] = None) -> Any:
        yield (None, LazyKnownValue(self._class))
        for values in self._wrapped_arguments.unpack(func):
            yield values


@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_classmethod(functions: Any, value: Any, arguments: Any) -> Any:
    return ValueSet((ClassMethodObject(class_method_object, function)
                     for class_method_object in value.py__call__(arguments=arguments)
                     for function in functions))


class PropertyObject(AttributeOverwrite, ValueWrapper):
    api_type: str = 'property'

    def __init__(self, property_obj: Any, function: Any) -> None:
        super().__init__(property_obj)
        self._function = function

    def py__get__(self, instance: Any, class_value: Any) -> Any:
        if instance is None:
            return ValueSet([self])
        return self._function.execute_with_values(instance)

    @publish_method('deleter')
    @publish_method('getter')
    @publish_method('setter')
    def _return_self(self, arguments: Any) -> Any:
        return ValueSet({self})


@argument_clinic('func, /', want_callback=True)
def builtins_property(functions: Any, callback: Callable[..., Any]) -> Any:
    return ValueSet((PropertyObject(property_value, function)
                     for property_value in callback()
                     for function in functions))


def collections_namedtuple(value: Any, arguments: Any, callback: Callable[..., Any]) -> Any:
    """
    Implementation of the namedtuple function.

    This has to be done by processing the namedtuple class template and
    inferring the result.

    """
    inference_state = value.inference_state
    name: str = 'jedi_unknown_namedtuple'
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
        fields = [get_str_or_none(v) for lazy_value in _fields.py__iter__() for v in lazy_value.infer()]
        fields = [f for f in fields if f is not None]
    else:
        return NO_VALUES
    code: str = _NAMEDTUPLE_CLASS_TEMPLATE.format(
        typename=name,
        field_names=tuple(fields),
        num_fields=len(fields),
        arg_list=repr(tuple(fields)).replace("'", '')[1:-1],
        repr_fmt='',
        field_defs='\n'.join((_NAMEDTUPLE_FIELD_TEMPLATE.format(index=index, name=name) for index, name in enumerate(fields)))
    )
    module = inference_state.grammar.parse(code)
    generated_class = next(module.iter_classdefs())
    parent_context = ModuleValue(inference_state, module, code_lines=parso.split_lines(code, keepends=True)).as_context()
    return ValueSet([ClassValue(inference_state, parent_context, generated_class)])


class PartialObject(ValueWrapper):

    def __init__(self, actual_value: Any, arguments: Any, instance: Optional[Any] = None) -> None:
        super().__init__(actual_value)
        self._arguments = arguments
        self._instance = instance

    def _get_functions(self, unpacked_arguments: Any) -> Any:
        key, lazy_value = next(unpacked_arguments, (None, None))
        if key is not None or lazy_value is None:
            debug.warning('Partial should have a proper function %s', self._arguments)
            return None
        return lazy_value.infer()

    def get_signatures(self) -> List[Any]:
        unpacked_arguments = self._arguments.unpack()
        funcs = self._get_functions(unpacked_arguments)
        if funcs is None:
            return []
        arg_count: int = 0
        if self._instance is not None:
            arg_count = 1
        keys: Set[Any] = set()
        for key, _ in unpacked_arguments:
            if key is None:
                arg_count += 1
            else:
                keys.add(key)
        return [PartialSignature(s, arg_count, keys) for s in funcs.get_signatures()]

    def py__call__(self, arguments: Any) -> Any:
        funcs = self._get_functions(self._arguments.unpack())
        if funcs is None:
            return NO_VALUES
        return funcs.execute(MergedPartialArguments(self._arguments, arguments, self._instance))

    def py__doc__(self) -> Any:
        """
        In CPython partial does not replace the docstring. However we are still
        imitating it here, because we want this docstring to be worth something
        for the user.
        """
        callables = self._get_functions(self._arguments.unpack())
        if callables is None:
            return ''
        for callable_ in callables:
            return callable_.py__doc__()
        return ''

    def py__get__(self, instance: Any, class_value: Any) -> Any:
        return ValueSet([self])


class PartialMethodObject(PartialObject):

    def py__get__(self, instance: Any, class_value: Any) -> Any:
        if instance is None:
            return ValueSet([self])
        return ValueSet([PartialObject(self._wrapped_value, self._arguments, instance)])


class PartialSignature(SignatureWrapper):

    def __init__(self, wrapped_signature: Any, skipped_arg_count: int, skipped_arg_set: Set[Any]) -> None:
        super().__init__(wrapped_signature)
        self._skipped_arg_count = skipped_arg_count
        self._skipped_arg_set = skipped_arg_set

    def get_param_names(self, resolve_stars: bool = False) -> Any:
        names = self._wrapped_signature.get_param_names()[self._skipped_arg_count:]
        return [n for n in names if n.string_name not in self._skipped_arg_set]


class MergedPartialArguments(AbstractArguments):

    def __init__(self, partial_arguments: Any, call_arguments: Any, instance: Optional[Any] = None) -> None:
        self._partial_arguments = partial_arguments
        self._call_arguments = call_arguments
        self._instance = instance

    def unpack(self, func: Optional[Any] = None) -> Any:
        unpacked = self._partial_arguments.unpack(func)
        next(unpacked, None)
        if self._instance is not None:
            yield (None, LazyKnownValue(self._instance))
        for key_lazy_value in unpacked:
            yield key_lazy_value
        for key_lazy_value in self._call_arguments.unpack(func):
            yield key_lazy_value


def functools_partial(value: Any, arguments: Any, callback: Callable[..., Any]) -> Any:
    return ValueSet((PartialObject(instance, arguments) for instance in value.py__call__(arguments)))


def functools_partialmethod(value: Any, arguments: Any, callback: Callable[..., Any]) -> Any:
    return ValueSet((PartialMethodObject(instance, arguments) for instance in value.py__call__(arguments)))


@argument_clinic('first, /')
def _return_first_param(firsts: Any) -> Any:
    return firsts


@argument_clinic('seq')
def _random_choice(sequences: Any) -> Any:
    return ValueSet.from_sets((lazy_value.infer() for sequence in sequences for lazy_value in sequence.py__iter__()))


def _dataclass(value: Any, arguments: Any, callback: Callable[..., Any]) -> Any:
    for c in _follow_param(value.inference_state, arguments, 0):
        if c.is_class():
            return ValueSet([DataclassWrapper(c)])
        else:
            return ValueSet([value])
    return NO_VALUES


class DataclassWrapper(ValueWrapper, ClassMixin):

    def get_signatures(self) -> List[Any]:
        param_names: List[Any] = []
        for cls in reversed(list(self.py__mro__())):
            if isinstance(cls, DataclassWrapper):
                filter_ = cls.as_context().get_global_filter()
                for name in sorted(filter_.values(), key=lambda name: name.start_pos):
                    d = name.tree_name.get_definition()
                    annassign = d.children[1]
                    if d.type == 'expr_stmt' and annassign.type == 'annassign':
                        if len(annassign.children) < 4:
                            default = None
                        else:
                            default = annassign.children[3]
                        param_names.append(DataclassParamName(parent_context=cls.parent_context,
                                                                tree_name=name.tree_name,
                                                                annotation_node=annassign.children[1],
                                                                default_node=default))
        return [DataclassSignature(cls, param_names)]


class DataclassSignature(AbstractSignature):

    def __init__(self, value: Any, param_names: Any) -> None:
        super().__init__(value)
        self._param_names = param_names

    def get_param_names(self, resolve_stars: bool = False) -> Any:
        return self._param_names


class DataclassParamName(BaseTreeParamName):

    def __init__(self, parent_context: Any, tree_name: Any, annotation_node: Any, default_node: Any) -> None:
        super().__init__(parent_context, tree_name)
        self.annotation_node = annotation_node
        self.default_node = default_node

    def get_kind(self) -> Any:
        return Parameter.POSITIONAL_OR_KEYWORD

    def infer(self) -> Any:
        if self.annotation_node is None:
            return NO_VALUES
        else:
            return self.parent_context.infer_node(self.annotation_node)


class ItemGetterCallable(ValueWrapper):

    def __init__(self, instance: Any, args_value_set: Any) -> None:
        super().__init__(instance)
        self._args_value_set = args_value_set

    @repack_with_argument_clinic('item, /')
    def py__call__(self, item_value_set: Any) -> Any:
        value_set = NO_VALUES
        for args_value in self._args_value_set:
            lazy_values = list(args_value.py__iter__())
            if len(lazy_values) == 1:
                value_set |= item_value_set.get_item(lazy_values[0].infer(), None)
            else:
                value_set |= ValueSet([iterable.FakeList(self._wrapped_value.inference_state, [LazyKnownValues(item_value_set.get_item(lazy_value.infer(), None)) for lazy_value in lazy_values])])
        return value_set


@argument_clinic('func, /')
def _functools_wraps(funcs: Any) -> Any:
    return ValueSet((WrapsCallable(func) for func in funcs))


class WrapsCallable(ValueWrapper):

    @repack_with_argument_clinic('func, /')
    def py__call__(self, funcs: Any) -> Any:
        return ValueSet({Wrapped(func, self._wrapped_value) for func in funcs})


class Wrapped(ValueWrapper, FunctionMixin):

    def __init__(self, func: Any, original_function: Any) -> None:
        super().__init__(func)
        self._original_function = original_function

    @property
    def name(self) -> Any:
        return self._original_function.name

    def get_signature_functions(self) -> List[Any]:
        return [self]


@argument_clinic('*args, /', want_value=True, want_arguments=True)
def _operator_itemgetter(args_value_set: Any, value: Any, arguments: Any) -> Any:
    return ValueSet([ItemGetterCallable(instance, args_value_set) for instance in value.py__call__(arguments)])


def _create_string_input_function(func: Callable[[str], str]) -> Callable[[Any, Any, Any], Any]:
    @argument_clinic('string, /', want_value=True, want_arguments=True)
    def wrapper(strings: Any, value: Any, arguments: Any) -> Any:
        def iterate() -> Any:
            for value in strings:
                s = get_str_or_none(value)
                if s is not None:
                    s = func(s)
                    yield compiled.create_simple_object(value.inference_state, s)
        values = ValueSet(iterate())
        if values:
            return values
        return value.py__call__(arguments)
    return wrapper


@argument_clinic('*args, /', want_callback=True)
def _os_path_join(args_set: Any, callback: Callable[..., Any]) -> Any:
    if len(args_set) == 1:
        string: str = ''
        sequence, = args_set
        is_first: bool = True
        for lazy_value in sequence.py__iter__():
            string_values = lazy_value.infer()
            if len(string_values) != 1:
                break
            s = get_str_or_none(next(iter(string_values)))
            if s is None:
                break
            if not is_first:
                string += os.path.sep
            string += s
            is_first = False
        else:
            return ValueSet([compiled.create_simple_object(sequence.inference_state, string)])
    return callback()


_implemented: Dict[str, Dict[str, Callable[..., Any]]] = {
    'builtins': {
        'getattr': builtins_getattr,
        'type': builtins_type,
        'super': builtins_super,
        'reversed': builtins_reversed,
        'isinstance': builtins_isinstance,
        'next': builtins_next,
        'iter': builtins_iter,
        'staticmethod': builtins_staticmethod,
        'classmethod': builtins_classmethod,
        'property': builtins_property
    },
    'copy': {
        'copy': _return_first_param,
        'deepcopy': _return_first_param
    },
    'json': {
        'load': lambda value, arguments, callback: NO_VALUES,
        'loads': lambda value, arguments, callback: NO_VALUES
    },
    'collections': {
        'namedtuple': collections_namedtuple
    },
    'functools': {
        'partial': functools_partial,
        'partialmethod': functools_partialmethod,
        'wraps': _functools_wraps
    },
    '_weakref': {
        'proxy': _return_first_param
    },
    'random': {
        'choice': _random_choice
    },
    'operator': {
        'itemgetter': _operator_itemgetter
    },
    'abc': {
        'abstractmethod': _return_first_param
    },
    'typing': {
        '_alias': lambda value, arguments, callback: NO_VALUES,
        'runtime_checkable': lambda value, arguments, callback: NO_VALUES
    },
    'dataclasses': {
        'dataclass': _dataclass
    },
    'os.path': {
        'dirname': _create_string_input_function(os.path.dirname),
        'abspath': _create_string_input_function(os.path.abspath),
        'relpath': _create_string_input_function(os.path.relpath),
        'join': _os_path_join
    }
}


def get_metaclass_filters(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(cls: Any, metaclasses: Any, is_instance: Any) -> Any:
        for metaclass in metaclasses:
            if metaclass.py__name__() == 'EnumMeta' and metaclass.get_root_context().py__name__() == 'enum':
                filter_ = ParserTreeFilter(parent_context=cls.as_context())
                return [DictFilter({name.string_name: EnumInstance(cls, name).name for name in filter_.values()})]
        return func(cls, metaclasses, is_instance)
    return wrapper


class EnumInstance(LazyValueWrapper):

    def __init__(self, cls: Any, name: Any) -> None:
        self.inference_state = cls.inference_state
        self._cls = cls
        self._name = name
        self.tree_node = self._name.tree_name

    @safe_property
    def name(self) -> Any:
        return ValueName(self, self._name.tree_name)

    def _get_wrapped_value(self) -> Any:
        n = self._name.string_name
        if n.startswith('__') and n.endswith('__') or self._name.api_type == 'function':
            inferred = self._name.infer()
            if inferred:
                return next(iter(inferred))
            o, = self.inference_state.builtins_module.py__getattribute__('object')
            return o
        value, = self._cls.execute_with_values()
        return value

    def get_filters(self, origin_scope: Optional[Any] = None) -> Any:
        yield DictFilter(dict(name=compiled.create_simple_object(self.inference_state, self._name.string_name).name,
                              value=self._name))
        for f in self._get_wrapped_value().get_filters():
            yield f


def tree_name_to_values(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(inference_state: Any, context: Any, tree_name: Any) -> Any:
        if tree_name.value == 'sep' and context.is_module() and (context.py__name__() == 'os.path'):
            return ValueSet({compiled.create_simple_object(inference_state, os.path.sep)})
        return func(inference_state, context, tree_name)
    return wrapper