from abc import abstractproperty
from typing import Iterator, List, Optional, Set, Tuple, TypeVar, Union, Any, Dict, Generator, Iterable

from parso.tree import search_ancestor
from parso.python.tree import Name, PythonNode

from jedi import debug
from jedi import settings
from jedi.inference import compiled
from jedi.inference.compiled.value import CompiledValueFilter
from jedi.inference.helpers import values_from_qualified_names, is_big_annoying_library
from jedi.inference.filters import AbstractFilter, AnonymousFunctionExecutionFilter
from jedi.inference.names import ValueName, TreeNameDefinition, ParamName, NameWrapper
from jedi.inference.base_value import Value, NO_VALUES, ValueSet, iterator_to_value_set, ValueWrapper
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import ValuesArguments, TreeArgumentsWrapper
from jedi.inference.value.function import (
    FunctionValue, FunctionMixin, OverloadedFunctionValue,
    BaseFunctionExecutionContext, FunctionExecutionContext, FunctionNameInClass
)
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.dynamic_arrays import get_dynamic_array_instance
from jedi.parser_utils import function_is_staticmethod, function_is_classmethod

T = TypeVar('T')
ContextualizedNode = Any  # Replace with actual type if available


class InstanceExecutedParamName(ParamName):
    def __init__(self, instance: Value, function_value: FunctionValue, tree_name: Name) -> None:
        super().__init__(function_value, tree_name, arguments=None)
        self._instance: Value = instance

    def infer(self) -> ValueSet:
        return ValueSet([self._instance])

    def matches_signature(self) -> bool:
        return True


class AnonymousMethodExecutionFilter(AnonymousFunctionExecutionFilter):
    def __init__(self, instance: Value, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._instance: Value = instance

    def _convert_param(self, param: Any, name: Name) -> Union[InstanceExecutedParamName, Any]:
        if param.position_index == 0:
            if function_is_classmethod(self._function_value.tree_node):
                return InstanceExecutedParamName(
                    self._instance.py__class__(),
                    self._function_value,
                    name
                )
            elif not function_is_staticmethod(self._function_value.tree_node):
                return InstanceExecutedParamName(
                    self._instance,
                    self._function_value,
                    name
                )
        return super()._convert_param(param, name)


class AnonymousMethodExecutionContext(BaseFunctionExecutionContext):
    def __init__(self, instance: Value, value: FunctionValue) -> None:
        super().__init__(value)
        self.instance: Value = instance

    def get_filters(self, until_position: Optional[Tuple[int, int]] = None, origin_scope: Any = None) -> Iterator[AnonymousMethodExecutionFilter]:
        yield AnonymousMethodExecutionFilter(
            self.instance, self, self._value,
            until_position=until_position,
            origin_scope=origin_scope,
        )

    def get_param_names(self) -> List[ParamName]:
        param_names = list(self._value.get_param_names())
        param_names[0] = InstanceExecutedParamName(
            self.instance,
            self._value,
            param_names[0].tree_name
        )
        return param_names


class MethodExecutionContext(FunctionExecutionContext):
    def __init__(self, instance: Value, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.instance: Value = instance


class AbstractInstanceValue(Value):
    api_type = 'instance'

    def __init__(self, inference_state: Any, parent_context: Any, class_value: Value) -> None:
        super().__init__(inference_state, parent_context)
        self.class_value: Value = class_value

    def is_instance(self) -> bool:
        return True

    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        return self.class_value.get_qualified_names()

    def get_annotated_class_object(self) -> Value:
        return self.class_value

    def py__class__(self) -> Value:
        return self.class_value

    def py__bool__(self) -> Optional[bool]:
        return None

    @abstractproperty
    def name(self) -> Any:
        raise NotImplementedError

    def get_signatures(self) -> List[Any]:
        call_funcs = self.py__getattribute__('__call__').py__get__(self, self.class_value)
        return [s.bind(self) for s in call_funcs.get_signatures()]

    def get_function_slot_names(self, name: str) -> List[Any]:
        for filter in self.get_filters(include_self_names=False):
            names = filter.get(name)
            if names:
                return names
        return []

    def execute_function_slots(self, names: List[Any], *inferred_args: Any) -> ValueSet:
        return ValueSet.from_sets(
            name.infer().execute_with_values(*inferred_args)
            for name in names
        )

    def get_type_hint(self, add_class_info: bool = True) -> str:
        return self.py__name__()

    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: ContextualizedNode) -> ValueSet:
        names = self.get_function_slot_names('__getitem__')
        if not names:
            return super().py__getitem__(index_value_set, contextualized_node)
        args = ValuesArguments([index_value_set])
        return ValueSet.from_sets(name.infer().execute(args) for name in names)

    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[ValueSet, None, None]:
        iter_slot_names = self.get_function_slot_names('__iter__')
        if not iter_slot_names:
            return super().py__iter__(contextualized_node)

        def iterate() -> Generator[ValueSet, None, None]:
            for generator in self.execute_function_slots(iter_slot_names):
                yield from generator.py__next__(contextualized_node)
        return iterate()

    def __repr__(self) -> str:
        return "<%s of %s>" % (self.__class__.__name__, self.class_value)


class CompiledInstance(AbstractInstanceValue):
    def __init__(self, inference_state: Any, parent_context: Any, class_value: Value, arguments: Any) -> None:
        super().__init__(inference_state, parent_context, class_value)
        self._arguments: Any = arguments

    def get_filters(self, origin_scope: Any = None, include_self_names: bool = True) -> Iterator[CompiledInstanceClassFilter]:
        class_value = self.get_annotated_class_object()
        class_filters = class_value.get_filters(origin_scope=origin_scope, is_instance=True)
        for f in class_filters:
            yield CompiledInstanceClassFilter(self, f)

    @property
    def name(self) -> Any:
        return compiled.CompiledValueName(self, self.class_value.name.string_name)

    def is_stub(self) -> bool:
        return False


class _BaseTreeInstance(AbstractInstanceValue):
    @property
    def array_type(self) -> Optional[str]:
        name = self.class_value.py__name__()
        if name in ['list', 'set', 'dict'] and self.parent_context.get_root_context().is_builtins_module():
            return name
        return None

    @property
    def name(self) -> ValueName:
        return ValueName(self, self.class_value.name.tree_name)

    def get_filters(self, origin_scope: Any = None, include_self_names: bool = True) -> Iterator[Union[SelfAttributeFilter, InstanceClassFilter, CompiledInstanceClassFilter, Any]]:
        class_value = self.get_annotated_class_object()
        if include_self_names:
            for cls in class_value.py__mro__():
                if not cls.is_compiled():
                    yield SelfAttributeFilter(self, class_value, cls.as_context(), origin_scope)

        class_filters = class_value.get_filters(origin_scope=origin_scope, is_instance=True)
        for f in class_filters:
            if isinstance(f, ClassFilter):
                yield InstanceClassFilter(self, f)
            elif isinstance(f, CompiledValueFilter):
                yield CompiledInstanceClassFilter(self, f)
            else:
                yield f

    @inference_state_method_cache()
    def create_instance_context(self, class_context: Any, node: PythonNode) -> Any:
        new = node
        while True:
            func_node = new
            new = search_ancestor(new, 'funcdef', 'classdef')
            if class_context.tree_node is new:
                func = FunctionValue.from_context(class_context, func_node)
                bound_method = BoundMethod(self, class_context, func)
                if func_node.name.value == '__init__':
                    context = bound_method.as_context(self._arguments)
                else:
                    context = bound_method.as_context()
                break
        return context.create_context(node)

    def py__getattribute__alternatives(self, string_name: str) -> ValueSet:
        if self.is_stub():
            return NO_VALUES

        name = compiled.create_simple_object(self.inference_state, string_name)
        names = (self.get_function_slot_names('__getattr__')
                 or self.get_function_slot_names('__getattribute__'))
        return self.execute_function_slots(names, name)

    def py__next__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValues, None, None]:
        name = '__next__'
        next_slot_names = self.get_function_slot_names(name)
        if next_slot_names:
            yield LazyKnownValues(self.execute_function_slots(next_slot_names))
        else:
            debug.warning('Instance has no __next__ function in %s.', self)

    def py__call__(self, arguments: Any) -> ValueSet:
        names = self.get_function_slot_names('__call__')
        if not names:
            return super().py__call__(arguments)
        return ValueSet.from_sets(name.infer().execute(arguments) for name in names)

    def py__get__(self, instance: Optional[Value], class_value: Value) -> ValueSet:
        for cls in self.class_value.py__mro__():
            result = cls.py__get__on_class(self, instance, class_value)
            if result is not NotImplemented:
                return result

        names = self.get_function_slot_names('__get__')
        if names:
            if instance is None:
                instance = compiled.builtin_from_name(self.inference_state, 'None')
            return self.execute_function_slots(names, instance, class_value)
        else:
            return ValueSet([self])


class TreeInstance(_BaseTreeInstance):
    def __init__(self, inference_state: Any, parent_context: Any, class_value: Value, arguments: Any) -> None:
        if class_value.py__name__() in ['list', 'set'] and parent_context.get_root_context().is_builtins_module():
            if settings.dynamic_array_additions:
                arguments = get_dynamic_array_instance(self, arguments)

        super().__init__(inference_state, parent_context, class_value)
        self._arguments: Any = arguments
        self.tree_node: PythonNode = class_value.tree_node

    @inference_state_method_cache(default=None)
    def _get_annotated_class_object(self) -> Optional[Value]:
        from jedi.inference.gradual.annotation import py__annotations__, infer_type_vars_for_execution

        args = InstanceArguments(self, self._arguments)
        for signature in self.class_value.py__getattribute__('__init__').get_signatures():
            funcdef = signature.value.tree_node
            if funcdef is None or funcdef.type != 'funcdef' or not signature.matches_signature(args):
                continue
            bound_method = BoundMethod(self, self.class_value.as_context(), signature.value)
            all_annotations = py__annotations__(funcdef)
            type_var_dict = infer_type_vars_for_execution(bound_method, args, all_annotations)
            if type_var_dict:
                defined, = self.class_value.define_generics(
                    infer_type_vars_for_execution(signature.value, args, all_annotations),
                )
                debug.dbg('Inferred instance value as %s', defined, color='BLUE')
                return defined
        return None

    def get_annotated_class_object(self) -> Value:
        return self._get_annotated_class_object() or self.class_value

    def get_key_values(self) -> ValueSet:
        values = NO_VALUES
        if self.array_type == 'dict':
            for i, (key, instance) in enumerate(self._arguments.unpack()):
                if key is None and i == 0:
                    values |= ValueSet.from_sets(
                        v.get_key_values()
                        for v in instance.infer()
                        if v.array_type == 'dict'
                    )
                if key:
                    values |= ValueSet([compiled.create_simple_object(
                        self.inference_state,
                        key,
                    )])
        return values

    def py__simple_getitem__(self, index: str) -> ValueSet:
        if self.array_type == 'dict':
            for key, lazy_context in reversed(list(self._arguments.unpack())):
                if key is None:
                    values = ValueSet.from_sets(
                        dct_value.py__simple_getitem__(index)
                        for dct_value in lazy_context.infer()
                        if dct_value.array_type == 'dict'
                    )
                    if values:
                        return values
                else:
                    if key == index:
                        return lazy_context.infer()
        return super().py__simple_getitem__(index)

    def __repr__(self) -> str:
        return "<%s of %s(%s)>" % (self.__class__.__name__, self.class_value, self._arguments)


class AnonymousInstance(_BaseTreeInstance):
    _arguments: Any = None


class CompiledInstanceName(NameWrapper):
    @iterator_to_value_set
    def infer(self) -> Generator[Union[CompiledBoundMethod, Value], None, None]:
        for result_value in self._wrapped_name.infer():
            if result_value.api_type == 'function':
                yield CompiledBoundMethod(result_value)
            else:
                yield result_value


class CompiledInstanceClassFilter(AbstractFilter):
    def __init__(self, instance: Value, f: AbstractFilter) -> None:
        self._instance: Value = instance
        self._class_filter: AbstractFilter = f

    def get(self, name: str) -> List[CompiledInstanceName]:
        return self._convert(self._class_filter.get(name))

    def values(self) -> List[CompiledInstanceName]:
        return self._convert(self._class_filter.values())

    def _convert(self, names: List[Any]) -> List[CompiledInstanceName]:
        return [CompiledInstanceName(n) for n in names]


class BoundMethod(FunctionMixin, ValueWrapper):
    def __init__(self, instance: Value, class_context: Any, function: FunctionValue) -> None:
        super().__init__(function)
        self.instance: Value = instance
        self._class_context: Any = class_context

    def is_bound_method(self) -> bool:
        return True

    @property
    def name(self) -> FunctionNameInClass:
        return FunctionNameInClass(self._class_context, super().name)

    def py__class__(self) -> Value:
        c, = values_from_qualified_names(self.inference_state, 'types', 'MethodType')
        return c

    def _get_arguments(self, arguments: Any) -> InstanceArguments:
        assert arguments is not None
        return InstanceArguments(self.instance, arguments)

    def _as_context(self, arguments: Optional[Any] = None) -> Union[AnonymousMethodExecutionContext, MethodExecutionContext]:
        if arguments is None:
            return AnonymousMethodExecutionContext(self.instance, self)
        arguments = self._get_arguments(arguments)
        return MethodExecutionContext(self.instance, self, arguments)

    def py__call__(self, arguments: Any) -> ValueSet:
        if isinstance(self._wrapped_value, OverloadedFunctionValue):
            return self._wrapped_value.py__call__(self._get_arguments(arguments))
        function_execution = self.as_context(arguments)
        return function_execution.infer()

    def get_signature_functions(self) -> List[BoundMethod]:
        return [
            BoundMethod(self.instance, self._class_context, f)
            for f in self._wrapped_value.get_signature_functions()
        ]

    def get_signatures(self) -> List[Any]:
        return [sig.bind(self) for sig in super().get_signatures()]

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self._wrapped_value)


class CompiledBoundMethod(ValueWrapper):
    def is_bound_method(self) -> bool:
        return True

    def get_signatures(self) -> List[Any]:
        return [sig.bind(self) for sig in self._wrapped_value.get_signatures()]


class SelfName(TreeNameDefinition):
    def __init__(self, instance: Value, class_context: Any, tree_name: Name) -> None:
        self._instance: Value = instance
        self.class_context: Any = class_context
        self.tree_name: Name = tree_name

    @property
    def parent_context(self) -> Any:
        return self._instance.create_instance_context(self.class_context, self.tree_name)

    def get_defining_qualified_value(self) -> Value:
        return self._instance

    def infer(self) -> ValueSet:
        stmt = search_ancestor(self.tree_name, 'expr_stmt')
        if stmt is not None:
            if stmt.children[1].type == "annassign":
                from jedi.inference.gradual.annotation import infer_annotation
                values = infer_annotation(
                    self.parent_context, stmt.children[1].children[1]
                ).execute_annotation()
                if values:
                    return values
        return super().infer()


class LazyInstanceClassName(NameWrapper):
    def __init__(self, instance: Value, class_member_name: Any) -> None:
        super().__init__(class_member_name)
        self._instance: Value = instance

    @iterator_to_value_set
    def infer(self) ->