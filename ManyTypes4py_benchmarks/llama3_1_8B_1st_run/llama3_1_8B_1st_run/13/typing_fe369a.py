"""
We need to somehow work with the typing objects. Since the typing objects are
pretty bare we need to add all the Jedi customizations to make them work as
values.

This file deals with all the typing.py cases.
"""
import itertools
from jedi import debug
from jedi.inference.compiled import builtin_from_name, create_simple_object
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, LazyValueWrapper, ValueWrapper
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.arguments import repack_with_argument_clinic
from jedi.inference.filters import FilterWrapper
from jedi.inference.names import NameWrapper, ValueName
from jedi.inference.value.klass import ClassMixin
from jedi.inference.gradual.base import BaseTypingValue, BaseTypingClassWithGenerics, BaseTypingInstance
from jedi.inference.gradual.type_var import TypeVarClass
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from typing import List, Dict, Set, FrozenSet, ChainMap, Counter, DefaultDict, Deque

_PROXY_CLASS_TYPES: List[str] = 'Tuple Generic Protocol Callable Type'.split()
_TYPE_ALIAS_TYPES: Dict[str, str] = {'List': 'builtins.list', 'Dict': 'builtins.dict', 'Set': 'builtins.set', 'FrozenSet': 'builtins.frozenset', 'ChainMap': 'collections.ChainMap', 'Counter': 'collections.Counter', 'DefaultDict': 'collections.defaultdict', 'Deque': 'collections.deque'}
_PROXY_TYPES: List[str] = 'Optional Union ClassVar'.split()

class TypingModuleName(NameWrapper):
    """A NameWrapper for typing module names."""

    def infer(self) -> ValueSet:
        """Infer the typing module name."""
        return self._remap()

    def _remap(self) -> ValueSet:
        """Remap the typing module name."""
        name = self.string_name
        inference_state = self.parent_context.inference_state
        try:
            actual = _TYPE_ALIAS_TYPES[name]
        except KeyError:
            pass
        else:
            yield TypeAlias.create_cached(inference_state, self.parent_context, self.tree_name, actual)
            return
        if name in _PROXY_CLASS_TYPES:
            yield ProxyTypingClassValue.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name in _PROXY_TYPES:
            yield ProxyTypingValue.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name == 'runtime':
            return
        elif name == 'TypeVar':
            cls, = self._wrapped_name.infer()
            yield TypeVarClass.create_cached(inference_state, cls)
        elif name == 'Any':
            yield AnyClass.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name == 'TYPE_CHECKING':
            yield builtin_from_name(inference_state, 'True')
        elif name == 'overload':
            yield OverloadFunction.create_cached(inference_state, self.parent_context, self.tree_name)
        elif name == 'NewType':
            v, = self._wrapped_name.infer()
            yield NewTypeFunction.create_cached(inference_state, v)
        elif name == 'cast':
            cast_fn, = self._wrapped_name.infer()
            yield CastFunction.create_cached(inference_state, cast_fn)
        elif name == 'TypedDict':
            yield TypedDictClass.create_cached(inference_state, self.parent_context, self.tree_name)
        else:
            yield from self._wrapped_name.infer()

class TypingModuleFilterWrapper(FilterWrapper):
    """A FilterWrapper for typing module names."""
    name_wrapper_class = TypingModuleName

class ProxyWithGenerics(BaseTypingClassWithGenerics):
    """A ProxyWithGenerics class."""

    def execute_annotation(self) -> ValueSet:
        """Execute the annotation."""
        string_name = self._tree_name.value
        if string_name == 'Union':
            return self.gather_annotation_classes().execute_annotation()
        elif string_name == 'Optional':
            return self.gather_annotation_classes().execute_annotation() | ValueSet([builtin_from_name(self.inference_state, 'None')])
        elif string_name == 'Type':
            return self._generics_manager[0]
        elif string_name == 'ClassVar':
            return self._generics_manager[0].execute_annotation()
        mapped = {'Tuple': Tuple, 'Generic': Generic, 'Protocol': Protocol, 'Callable': Callable}
        cls = mapped[string_name]
        return ValueSet([cls(self.parent_context, self, self._tree_name, generics_manager=self._generics_manager)])

    def gather_annotation_classes(self) -> ValueSet:
        """Gather annotation classes."""
        return ValueSet.from_sets(self._generics_manager.to_tuple())

    def _create_instance_with_generics(self, generics_manager: LazyGenericManager) -> ProxyWithGenerics:
        """Create an instance with generics."""
        return ProxyWithGenerics(self.parent_context, self._tree_name, generics_manager)

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, TypeVarClass]:
        """Infer type variables."""
        annotation_generics = self.get_generics()
        if not annotation_generics:
            return {}
        annotation_name = self.py__name__()
        if annotation_name == 'Optional':
            none = builtin_from_name(self.inference_state, 'None')
            return annotation_generics[0].infer_type_vars(value_set.filter(lambda x: x != none))
        return {}

class ProxyTypingValue(BaseTypingValue):
    """A ProxyTypingValue class."""
    index_class = ProxyWithGenerics

    def with_generics(self, generics_tuple: TupleGenericManager) -> ProxyWithGenerics:
        """Add generics."""
        return self.index_class.create_cached(self.inference_state, self.parent_context, self._tree_name, generics_manager=generics_tuple)

    def py__getitem__(self, index_value_set: ValueSet, contextualized_node) -> ValueSet:
        """Get an item."""
        return ValueSet((self.index_class.create_cached(self.inference_state, self.parent_context, self._tree_name, generics_manager=LazyGenericManager(context_of_index=contextualized_node.context, index_value=index_value)) for index_value in index_value_set))

class _TypingClassMixin(ClassMixin):
    """A _TypingClassMixin class."""

    def py__bases__(self) -> ValueSet:
        """Get bases."""
        return [LazyKnownValues(self.inference_state.builtins_module.py__getattribute__('object'))]

    def get_metaclasses(self) -> List[Value]:
        """Get metaclasses."""
        return []

    @property
    def name(self) -> ValueName:
        """Get name."""
        return ValueName(self, self._tree_name)

class TypingClassWithGenerics(ProxyWithGenerics, _TypingClassMixin):
    """A TypingClassWithGenerics class."""

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, TypeVarClass]:
        """Infer type variables."""
        type_var_dict = {}
        annotation_generics = self.get_generics()
        if not annotation_generics:
            return type_var_dict
        annotation_name = self.py__name__()
        if annotation_name == 'Type':
            return annotation_generics[0].infer_type_vars(value_set.execute_annotation())
        elif annotation_name == 'Callable':
            if len(annotation_generics) == 2:
                return annotation_generics[1].infer_type_vars(value_set.execute_annotation())
        elif annotation_name == 'Tuple':
            tuple_annotation, = self.execute_annotation()
            return tuple_annotation.infer_type_vars(value_set)
        return type_var_dict

    def _create_instance_with_generics(self, generics_manager: LazyGenericManager) -> TypingClassWithGenerics:
        """Create an instance with generics."""
        return TypingClassWithGenerics(self.parent_context, self._tree_name, generics_manager)

class ProxyTypingClassValue(ProxyTypingValue, _TypingClassMixin):
    """A ProxyTypingClassValue class."""
    index_class = TypingClassWithGenerics

class TypeAlias(LazyValueWrapper):
    """A TypeAlias class."""

    def __init__(self, parent_context: 'Context', origin_tree_name: 'TreeNode', actual: str) -> None:
        """Initialize TypeAlias."""
        self.inference_state = parent_context.inference_state
        self.parent_context = parent_context
        self._origin_tree_name = origin_tree_name
        self._actual = actual

    @property
    def name(self) -> ValueName:
        """Get name."""
        return ValueName(self, self._origin_tree_name)

    def py__name__(self) -> str:
        """Get name."""
        return self.name.string_name

    def __repr__(self) -> str:
        """Get representation."""
        return '<%s: %s>' % (self.__class__.__name__, self._actual)

    def _get_wrapped_value(self) -> Value:
        """Get wrapped value."""
        module_name, class_name = self._actual.split('.')
        from jedi.inference.imports import Importer
        module, = Importer(self.inference_state, [module_name], self.inference_state.builtins_module).follow()
        classes = module.py__getattribute__(class_name)
        assert len(classes) == 1, classes
        cls = next(iter(classes))
        return cls

    def gather_annotation_classes(self) -> ValueSet:
        """Gather annotation classes."""
        return ValueSet([self._get_wrapped_value()])

    def get_signatures(self) -> List['Signature']:
        """Get signatures."""
        return []

class Callable(BaseTypingInstance):
    """A Callable class."""

    def py__call__(self, arguments: 'Arguments') -> ValueSet:
        """
            def x() -> Callable[[Callable[..., _T]], _T]: ...
        """
        try:
            param_values = self._generics_manager[0]
            result_values = self._generics_manager[1]
        except IndexError:
            debug.warning('Callable[...] defined without two arguments')
            return NO_VALUES
        else:
            from jedi.inference.gradual.annotation import infer_return_for_callable
            return infer_return_for_callable(arguments, param_values, result_values)

    def py__get__(self, instance: 'Instance', class_value: 'Value') -> ValueSet:
        """Get attribute."""
        return ValueSet([self])

class Tuple(BaseTypingInstance):
    """A Tuple class."""

    def _is_homogenous(self) -> bool:
        """Check if tuple is homogenous."""
        return self._generics_manager.is_homogenous_tuple()

    def py__simple_getitem__(self, index: int) -> ValueSet:
        """Get item."""
        if self._is_homogenous():
            return self._generics_manager.get_index_and_execute(0)
        else:
            if isinstance(index, int):
                return self._generics_manager.get_index_and_execute(index)
            debug.dbg('The getitem type on Tuple was %s' % index)
            return NO_VALUES

    def py__iter__(self, contextualized_node=None) -> Iterator[Value]:
        """Get iterator."""
        if self._is_homogenous():
            yield LazyKnownValues(self._generics_manager.get_index_and_execute(0))
        else:
            for v in self._generics_manager.to_tuple():
                yield LazyKnownValues(v.execute_annotation())

    def py__getitem__(self, index_value_set: ValueSet, contextualized_node) -> ValueSet:
        """Get item."""
        if self._is_homogenous():
            return self._generics_manager.get_index_and_execute(0)
        return ValueSet.from_sets(self._generics_manager.to_tuple()).execute_annotation()

    def _get_wrapped_value(self) -> Value:
        """Get wrapped value."""
        tuple_, = self.inference_state.builtins_module.py__getattribute__('tuple').execute_annotation()
        return tuple_

    @property
    def name(self) -> ValueName:
        """Get name."""
        return self._wrapped_value.name

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, TypeVarClass]:
        """Infer type variables."""
        from jedi.inference.gradual.annotation import merge_pairwise_generics, merge_type_var_dicts
        value_set = value_set.filter(lambda x: x.py__name__().lower() == 'tuple')
        if self._is_homogenous():
            return self._class_value.get_generics()[0].infer_type_vars(value_set.merge_types_of_iterate())
        else:
            type_var_dict = {}
            for element in value_set:
                try:
                    method = element.get_annotated_class_object
                except AttributeError:
                    continue
                py_class = method()
                merge_type_var_dicts(type_var_dict, merge_pairwise_generics(self._class_value, py_class))
            return type_var_dict

class Generic(BaseTypingInstance):
    """A Generic class."""

class Protocol(BaseTypingInstance):
    """A Protocol class."""

class AnyClass(BaseTypingValue):
    """An AnyClass class."""

    def execute_annotation(self) -> ValueSet:
        """Execute annotation."""
        debug.warning('Used Any - returned no results')
        return NO_VALUES

class OverloadFunction(BaseTypingValue):
    """An OverloadFunction class."""

    @repack_with_argument_clinic('func, /')
    def py__call__(self, func_value_set: ValueSet) -> ValueSet:
        """Call function."""
        return func_value_set

class NewTypeFunction(ValueWrapper):
    """A NewTypeFunction class."""

    def py__call__(self, arguments: 'Arguments') -> ValueSet:
        """Call function."""
        ordered_args = arguments.unpack()
        next(ordered_args, (None, None))
        _, second_arg = next(ordered_args, (None, None))
        if second_arg is None:
            return NO_VALUES
        return ValueSet((NewType(self.inference_state, contextualized_node.context, contextualized_node.node, second_arg.infer()) for contextualized_node in arguments.get_calling_nodes()))

class NewType(Value):
    """A NewType class."""

    def __init__(self, inference_state: 'InferenceState', parent_context: 'Context', tree_node: 'TreeNode', type_value_set: ValueSet) -> None:
        """Initialize NewType."""
        super().__init__(inference_state, parent_context)
        self._type_value_set = type_value_set
        self.tree_node = tree_node

    def py__class__(self) -> Value:
        """Get class."""
        c, = self._type_value_set.py__class__()
        return c

    def py__call__(self, arguments: 'Arguments') -> ValueSet:
        """Call function."""
        return self._type_value_set.execute_annotation()

    @property
    def name(self) -> ValueName:
        """Get name."""
        from jedi.inference.compiled.value import CompiledValueName
        return CompiledValueName(self, 'NewType')

    def __repr__(self) -> str:
        """Get representation."""
        return '<NewType: %s>%s' % (self.tree_node, self._type_value_set)

class CastFunction(ValueWrapper):
    """A CastFunction class."""

    @repack_with_argument_clinic('type, object, /')
    def py__call__(self, type_value_set: ValueSet, object_value_set: ValueSet) -> ValueSet:
        """Call function."""
        return type_value_set.execute_annotation()

class TypedDictClass(BaseTypingValue):
    """A TypedDictClass class."""

class TypedDict(LazyValueWrapper):
    """A TypedDict class."""

    def __init__(self, definition_class: 'Value') -> None:
        """Initialize TypedDict."""
        self.inference_state = definition_class.inference_state
        self.parent_context = definition_class.parent_context
        self.tree_node = definition_class.tree_node
        self._definition_class = definition_class

    @property
    def name(self) -> ValueName:
        """Get name."""
        return ValueName(self, self.tree_node.name)

    def py__simple_getitem__(self, index: str) -> ValueSet:
        """Get item."""
        if isinstance(index, str):
            return ValueSet.from_sets((name.infer() for filter in self._definition_class.get_filters(is_instance=True) for name in filter.get(index)))
        return NO_VALUES

    def get_key_values(self) -> ValueSet:
        """Get key values."""
        filtered_values = itertools.chain.from_iterable((f.values() for f in self._definition_class.get_filters(is_instance=True)))
        return ValueSet({create_simple_object(self.inference_state, v.string_name) for v in filtered_values})

    def _get_wrapped_value(self) -> Value:
        """Get wrapped value."""
        d, = self.inference_state.builtins_module.py__getattribute__('dict')
        result, = d.execute_with_values()
        return result
