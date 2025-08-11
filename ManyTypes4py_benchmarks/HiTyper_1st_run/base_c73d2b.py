from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, iterator_to_value_set, LazyValueWrapper, ValueWrapper
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager

class _BoundTypeVarName(AbstractNameDefinition):
    """
    This type var was bound to a certain type, e.g. int.
    """

    def __init__(self, type_var, value_set) -> None:
        self._type_var = type_var
        self.parent_context = type_var.parent_context
        self._value_set = value_set

    def infer(self) -> typing.Generator:

        def iter_() -> typing.Generator:
            for value in self._value_set:
                from jedi.inference.gradual.typing import AnyClass
                if isinstance(value, AnyClass):
                    yield from self._type_var.constraints
                else:
                    yield value
        return ValueSet(iter_())

    def py__name__(self) -> str:
        return self._type_var.py__name__()

    def __repr__(self) -> typing.Text:
        return '<%s %s -> %s>' % (self.__class__.__name__, self.py__name__(), self._value_set)

class _TypeVarFilter:
    """
    A filter for all given variables in a class.

        A = TypeVar('A')
        B = TypeVar('B')
        class Foo(Mapping[A, B]):
            ...

    In this example we would have two type vars given: A and B
    """

    def __init__(self, generics: typing.Callable, type_vars: Any) -> None:
        self._generics = generics
        self._type_vars = type_vars

    def get(self, name: str) -> Union[list[_BoundTypeVarName], list]:
        for i, type_var in enumerate(self._type_vars):
            if type_var.py__name__() == name:
                try:
                    return [_BoundTypeVarName(type_var, self._generics[i])]
                except IndexError:
                    return [type_var.name]
        return []

    def values(self) -> list:
        return []

class _AnnotatedClassContext(ClassContext):

    def get_filters(self, *args, **kwargs) -> typing.Generator[EmptyFilter]:
        filters = super().get_filters(*args, **kwargs)
        yield from filters
        yield self._value.get_type_var_filter()

class DefineGenericBaseClass(LazyValueWrapper):

    def __init__(self, generics_manager) -> None:
        self._generics_manager = generics_manager

    def _create_instance_with_generics(self, generics_manager: Union[str, typing.Type]) -> GenericClass:
        raise NotImplementedError

    @inference_state_method_cache()
    def get_generics(self):
        return self._generics_manager.to_tuple()

    def define_generics(self, type_var_dict: typing.Type) -> ValueSet:
        from jedi.inference.gradual.type_var import TypeVar
        changed = False
        new_generics = []
        for generic_set in self.get_generics():
            values = NO_VALUES
            for generic in generic_set:
                if isinstance(generic, (DefineGenericBaseClass, TypeVar)):
                    result = generic.define_generics(type_var_dict)
                    values |= result
                    if result != ValueSet({generic}):
                        changed = True
                else:
                    values |= ValueSet([generic])
            new_generics.append(values)
        if not changed:
            return ValueSet([self])
        return ValueSet([self._create_instance_with_generics(TupleGenericManager(tuple(new_generics)))])

    def is_same_class(self, other: typing.Type) -> bool:
        if not isinstance(other, DefineGenericBaseClass):
            return False
        if self.tree_node != other.tree_node:
            return False
        given_params1 = self.get_generics()
        given_params2 = other.get_generics()
        if len(given_params1) != len(given_params2):
            return False
        return all((any((cls2.is_same_class(cls1) for cls1 in class_set1.gather_annotation_classes() for cls2 in class_set2.gather_annotation_classes())) for class_set1, class_set2 in zip(given_params1, given_params2)))

    def get_signatures(self) -> list:
        return []

    def __repr__(self) -> typing.Text:
        return '<%s: %s%s>' % (self.__class__.__name__, self._wrapped_value, list(self.get_generics()))

class GenericClass(DefineGenericBaseClass, ClassMixin):
    """
    A class that is defined with generics, might be something simple like:

        class Foo(Generic[T]): ...
        my_foo_int_cls = Foo[int]
    """

    def __init__(self, class_value, generics_manager) -> None:
        super().__init__(generics_manager)
        self._class_value = class_value

    def _get_wrapped_value(self):
        return self._class_value

    def get_type_hint(self, add_class_info: bool=True):
        n = self.py__name__()
        n = dict(list='List', dict='Dict', set='Set', tuple='Tuple').get(n, n)
        s = n + self._generics_manager.get_type_hint()
        if add_class_info:
            return 'Type[%s]' % s
        return s

    def get_type_var_filter(self) -> _TypeVarFilter:
        return _TypeVarFilter(self.get_generics(), self.list_type_vars())

    def py__call__(self, arguments: Union[str, typing.Callable]) -> ValueSet:
        instance, = super().py__call__(arguments)
        return ValueSet([_GenericInstanceWrapper(instance)])

    def _as_context(self) -> _AnnotatedClassContext:
        return _AnnotatedClassContext(self)

    @to_list
    def py__bases__(self) -> typing.Generator[_LazyGenericBaseClass]:
        for base in self._wrapped_value.py__bases__():
            yield _LazyGenericBaseClass(self, base, self._generics_manager)

    def _create_instance_with_generics(self, generics_manager: Union[str, typing.Type]) -> GenericClass:
        return GenericClass(self._class_value, generics_manager)

    def is_sub_class_of(self, class_value: Union[str, typing.Any, None]) -> bool:
        if super().is_sub_class_of(class_value):
            return True
        return self._class_value.is_sub_class_of(class_value)

    def with_generics(self, generics_tuple: Union[typing.Type, dict]) -> typing.Type:
        return self._class_value.with_generics(generics_tuple)

    def infer_type_vars(self, value_set: Any) -> dict:
        from jedi.inference.gradual.annotation import merge_pairwise_generics, merge_type_var_dicts
        annotation_name = self.py__name__()
        type_var_dict = {}
        if annotation_name == 'Iterable':
            annotation_generics = self.get_generics()
            if annotation_generics:
                return annotation_generics[0].infer_type_vars(value_set.merge_types_of_iterate())
        else:
            for py_class in value_set:
                if py_class.is_instance() and (not py_class.is_compiled()):
                    py_class = py_class.get_annotated_class_object()
                else:
                    continue
                if py_class.api_type != 'class':
                    continue
                for parent_class in py_class.py__mro__():
                    class_name = parent_class.py__name__()
                    if annotation_name == class_name:
                        merge_type_var_dicts(type_var_dict, merge_pairwise_generics(self, parent_class))
                        break
        return type_var_dict

class _LazyGenericBaseClass:

    def __init__(self, class_value, lazy_base_class, generics_manager) -> None:
        self._class_value = class_value
        self._lazy_base_class = lazy_base_class
        self._generics_manager = generics_manager

    @iterator_to_value_set
    def infer(self) -> typing.Generator:
        for base in self._lazy_base_class.infer():
            if isinstance(base, GenericClass):
                yield GenericClass.create_cached(base.inference_state, base._wrapped_value, TupleGenericManager(tuple(self._remap_type_vars(base))))
            elif base.is_class_mixin():
                yield GenericClass.create_cached(base.inference_state, base, self._generics_manager)
            else:
                yield base

    def _remap_type_vars(self, base: Union[str, int, llvmlite.ir.Value]) -> typing.Generator:
        from jedi.inference.gradual.type_var import TypeVar
        filter = self._class_value.get_type_var_filter()
        for type_var_set in base.get_generics():
            new = NO_VALUES
            for type_var in type_var_set:
                if isinstance(type_var, TypeVar):
                    names = filter.get(type_var.py__name__())
                    new |= ValueSet.from_sets((name.infer() for name in names))
                else:
                    new |= ValueSet([type_var])
            yield new

    def __repr__(self) -> typing.Text:
        return '<%s: %s>' % (self.__class__.__name__, self._lazy_base_class)

class _GenericInstanceWrapper(ValueWrapper):

    def py__stop_iteration_returns(self) -> Union[ValueSet, str, typing.ChainMap, None]:
        for cls in self._wrapped_value.class_value.py__mro__():
            if cls.py__name__() == 'Generator':
                generics = cls.get_generics()
                try:
                    return generics[2].execute_annotation()
                except IndexError:
                    pass
            elif cls.py__name__() == 'Iterator':
                return ValueSet([builtin_from_name(self.inference_state, 'None')])
        return self._wrapped_value.py__stop_iteration_returns()

    def get_type_hint(self, add_class_info: bool=True):
        return self._wrapped_value.class_value.get_type_hint(add_class_info=False)

class _PseudoTreeNameClass(Value):
    """
    In typeshed, some classes are defined like this:

        Tuple: _SpecialForm = ...

    Now this is not a real class, therefore we have to do some workarounds like
    this class. Essentially this class makes it possible to goto that `Tuple`
    name, without affecting anything else negatively.
    """
    api_type = 'class'

    def __init__(self, parent_context, tree_name) -> None:
        super().__init__(parent_context.inference_state, parent_context)
        self._tree_name = tree_name

    @property
    def tree_node(self):
        return self._tree_name

    def get_filters(self, *args, **kwargs) -> typing.Generator[EmptyFilter]:

        class EmptyFilter(ClassFilter):

            def __init__(self) -> None:
                pass

            def get(self, name: str, **kwargs) -> Union[list[_BoundTypeVarName], list]:
                return []

            def values(self, **kwargs) -> list:
                return []
        yield EmptyFilter()

    def py__class__(self):
        return builtin_from_name(self.inference_state, 'type')

    @property
    def name(self) -> ValueName:
        return ValueName(self, self._tree_name)

    def get_qualified_names(self) -> tuple:
        return (self._tree_name.value,)

    def __repr__(self) -> typing.Text:
        return '%s(%s)' % (self.__class__.__name__, self._tree_name.value)

class BaseTypingValue(LazyValueWrapper):

    def __init__(self, parent_context, tree_name) -> None:
        self.inference_state = parent_context.inference_state
        self.parent_context = parent_context
        self._tree_name = tree_name

    @property
    def name(self) -> ValueName:
        return ValueName(self, self._tree_name)

    def _get_wrapped_value(self):
        return _PseudoTreeNameClass(self.parent_context, self._tree_name)

    def get_signatures(self) -> list:
        return self._wrapped_value.get_signatures()

    def __repr__(self) -> typing.Text:
        return '%s(%s)' % (self.__class__.__name__, self._tree_name.value)

class BaseTypingClassWithGenerics(DefineGenericBaseClass):

    def __init__(self, parent_context, tree_name, generics_manager) -> None:
        super().__init__(generics_manager)
        self.inference_state = parent_context.inference_state
        self.parent_context = parent_context
        self._tree_name = tree_name

    def _get_wrapped_value(self):
        return _PseudoTreeNameClass(self.parent_context, self._tree_name)

    def __repr__(self) -> typing.Text:
        return '%s(%s%s)' % (self.__class__.__name__, self._tree_name.value, self._generics_manager)

class BaseTypingInstance(LazyValueWrapper):

    def __init__(self, parent_context, class_value, tree_name, generics_manager) -> None:
        self.inference_state = class_value.inference_state
        self.parent_context = parent_context
        self._class_value = class_value
        self._tree_name = tree_name
        self._generics_manager = generics_manager

    def py__class__(self):
        return self._class_value

    def get_annotated_class_object(self):
        return self._class_value

    def get_qualified_names(self) -> tuple:
        return (self.py__name__(),)

    @property
    def name(self) -> ValueName:
        return ValueName(self, self._tree_name)

    def _get_wrapped_value(self):
        object_, = builtin_from_name(self.inference_state, 'object').execute_annotation()
        return object_

    def __repr__(self) -> typing.Text:
        return '<%s: %s>' % (self.__class__.__name__, self._generics_manager)