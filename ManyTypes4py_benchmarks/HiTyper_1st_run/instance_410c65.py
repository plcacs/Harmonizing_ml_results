from abc import abstractproperty
from parso.tree import search_ancestor
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
from jedi.inference.value.function import FunctionValue, FunctionMixin, OverloadedFunctionValue, BaseFunctionExecutionContext, FunctionExecutionContext, FunctionNameInClass
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.dynamic_arrays import get_dynamic_array_instance
from jedi.parser_utils import function_is_staticmethod, function_is_classmethod

class InstanceExecutedParamName(ParamName):

    def __init__(self, instance, function_value, tree_name) -> None:
        super().__init__(function_value, tree_name, arguments=None)
        self._instance = instance

    def infer(self) -> Union[str, bool, list[str]]:
        return ValueSet([self._instance])

    def matches_signature(self) -> bool:
        return True

class AnonymousMethodExecutionFilter(AnonymousFunctionExecutionFilter):

    def __init__(self, instance, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._instance = instance

    def _convert_param(self, param: Union[str, list[float]], name: Union[str, typing.Callable]) -> Union[InstanceExecutedParamName, str, dict[str, tuple[str]], T]:
        if param.position_index == 0:
            if function_is_classmethod(self._function_value.tree_node):
                return InstanceExecutedParamName(self._instance.py__class__(), self._function_value, name)
            elif not function_is_staticmethod(self._function_value.tree_node):
                return InstanceExecutedParamName(self._instance, self._function_value, name)
        return super()._convert_param(param, name)

class AnonymousMethodExecutionContext(BaseFunctionExecutionContext):

    def __init__(self, instance, value) -> None:
        super().__init__(value)
        self.instance = instance

    def get_filters(self, until_position=None, origin_scope: Union[None, bool, str]=None) -> Union[typing.Generator[SelfAttributeFilter], typing.Generator[InstanceClassFilter], typing.Generator[CompiledInstanceClassFilter], typing.Generator]:
        yield AnonymousMethodExecutionFilter(self.instance, self, self._value, until_position=until_position, origin_scope=origin_scope)

    def get_param_names(self) -> list[InstanceExecutedParamName]:
        param_names = list(self._value.get_param_names())
        param_names[0] = InstanceExecutedParamName(self.instance, self._value, param_names[0].tree_name)
        return param_names

class MethodExecutionContext(FunctionExecutionContext):

    def __init__(self, instance, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.instance = instance

class AbstractInstanceValue(Value):
    api_type = 'instance'

    def __init__(self, inference_state: Union[bool, typing.Mapping, None, str], parent_context: typing.Type, class_value: Union[str, typing.Type]) -> None:
        super().__init__(inference_state, parent_context)
        self.class_value = class_value

    def is_instance(self) -> bool:
        return True

    def get_qualified_names(self) -> Union[typing.Callable, str]:
        return self.class_value.get_qualified_names()

    def get_annotated_class_object(self) -> Union[str, typing.Type]:
        return self.class_value

    def py__class__(self):
        return self.class_value

    def py__bool__(self) -> None:
        return None

    @abstractproperty
    def name(self) -> ValueName:
        raise NotImplementedError

    def get_signatures(self) -> Union[str, None, bool]:
        call_funcs = self.py__getattribute__('__call__').py__get__(self, self.class_value)
        return [s.bind(self) for s in call_funcs.get_signatures()]

    def get_function_slot_names(self, name: Union[str, None]) -> list:
        for filter in self.get_filters(include_self_names=False):
            names = filter.get(name)
            if names:
                return names
        return []

    def execute_function_slots(self, names: str, *inferred_args):
        return ValueSet.from_sets((name.infer().execute_with_values(*inferred_args) for name in names))

    def get_type_hint(self, add_class_info: bool=True) -> str:
        return self.py__name__()

    def py__getitem__(self, index_value_set: Union[str, KspNativeControl, int], contextualized_node: Union[str, KspNativeControl, int]) -> Union[str, dict[str, tuple[str]], list[str]]:
        names = self.get_function_slot_names('__getitem__')
        if not names:
            return super().py__getitem__(index_value_set, contextualized_node)
        args = ValuesArguments([index_value_set])
        return ValueSet.from_sets((name.infer().execute(args) for name in names))

    def py__iter__(self, contextualized_node: Union[None, list[dict[str, typing.Any]]]=None) -> Union[str, int]:
        iter_slot_names = self.get_function_slot_names('__iter__')
        if not iter_slot_names:
            return super().py__iter__(contextualized_node)

        def iterate() -> typing.Generator:
            for generator in self.execute_function_slots(iter_slot_names):
                yield from generator.py__next__(contextualized_node)
        return iterate()

    def __repr__(self) -> typing.Text:
        return '<%s of %s>' % (self.__class__.__name__, self.class_value)

class CompiledInstance(AbstractInstanceValue):

    def __init__(self, inference_state: Union[bool, typing.Mapping, None, str], parent_context: typing.Type, class_value: Union[str, typing.Type], arguments: Union[str, None, int, typing.Mapping]) -> None:
        super().__init__(inference_state, parent_context, class_value)
        self._arguments = arguments

    def get_filters(self, origin_scope: Union[None, bool, str]=None, include_self_names: bool=True) -> Union[typing.Generator[SelfAttributeFilter], typing.Generator[InstanceClassFilter], typing.Generator[CompiledInstanceClassFilter], typing.Generator]:
        class_value = self.get_annotated_class_object()
        class_filters = class_value.get_filters(origin_scope=origin_scope, is_instance=True)
        for f in class_filters:
            yield CompiledInstanceClassFilter(self, f)

    @property
    def name(self) -> ValueName:
        return compiled.CompiledValueName(self, self.class_value.name.string_name)

    def is_stub(self) -> bool:
        return False

class _BaseTreeInstance(AbstractInstanceValue):

    @property
    def array_type(self) -> Union[str, tuple[str], None]:
        name = self.class_value.py__name__()
        if name in ['list', 'set', 'dict'] and self.parent_context.get_root_context().is_builtins_module():
            return name
        return None

    @property
    def name(self) -> ValueName:
        return ValueName(self, self.class_value.name.tree_name)

    def get_filters(self, origin_scope: Union[None, bool, str]=None, include_self_names: bool=True) -> Union[typing.Generator[SelfAttributeFilter], typing.Generator[InstanceClassFilter], typing.Generator[CompiledInstanceClassFilter], typing.Generator]:
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
    def create_instance_context(self, class_context: Any, node: dict) -> Union[dict[str, typing.Any], dict]:
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

    def py__getattribute__alternatives(self, string_name: str):
        """
        Since nothing was inferred, now check the __getattr__ and
        __getattribute__ methods. Stubs don't need to be checked, because
        they don't contain any logic.
        """
        if self.is_stub():
            return NO_VALUES
        name = compiled.create_simple_object(self.inference_state, string_name)
        if is_big_annoying_library(self.parent_context):
            return NO_VALUES
        names = self.get_function_slot_names('__getattr__') or self.get_function_slot_names('__getattribute__')
        return self.execute_function_slots(names, name)

    def py__next__(self, contextualized_node: Union[None, list[dict[str, typing.Any]]]=None) -> typing.Generator[LazyKnownValues]:
        name = u'__next__'
        next_slot_names = self.get_function_slot_names(name)
        if next_slot_names:
            yield LazyKnownValues(self.execute_function_slots(next_slot_names))
        else:
            debug.warning('Instance has no __next__ function in %s.', self)

    def py__call__(self, arguments: Union[str, typing.Type]) -> Union[bool, str]:
        names = self.get_function_slot_names('__call__')
        if not names:
            return super().py__call__(arguments)
        return ValueSet.from_sets((name.infer().execute(arguments) for name in names))

    def py__get__(self, instance: Union[typing.Any, None, typing.Type, typing.Mapping], class_value: typing.Type) -> Union[dict, typing.Generator[typing.Optional[typing.Any]], typing.MutableMapping, typing.MutableSequence, bool, float, str, ValueSet]:
        """
        obj may be None.
        """
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

    def __init__(self, inference_state: Union[bool, typing.Mapping, None, str], parent_context: typing.Type, class_value: Union[str, typing.Type], arguments: Union[str, None, int, typing.Mapping]) -> None:
        if class_value.py__name__() in ['list', 'set'] and parent_context.get_root_context().is_builtins_module():
            if settings.dynamic_array_additions:
                arguments = get_dynamic_array_instance(self, arguments)
        super().__init__(inference_state, parent_context, class_value)
        self._arguments = arguments
        self.tree_node = class_value.tree_node

    @inference_state_method_cache(default=None)
    def _get_annotated_class_object(self) -> Union[bool, qcore.helpers.MarkerObject, str, None]:
        from jedi.inference.gradual.annotation import py__annotations__, infer_type_vars_for_execution
        args = InstanceArguments(self, self._arguments)
        for signature in self.class_value.py__getattribute__('__init__').get_signatures():
            funcdef = signature.value.tree_node
            if funcdef is None or funcdef.type != 'funcdef' or (not signature.matches_signature(args)):
                continue
            bound_method = BoundMethod(self, self.class_value.as_context(), signature.value)
            all_annotations = py__annotations__(funcdef)
            type_var_dict = infer_type_vars_for_execution(bound_method, args, all_annotations)
            if type_var_dict:
                defined, = self.class_value.define_generics(infer_type_vars_for_execution(signature.value, args, all_annotations))
                debug.dbg('Inferred instance value as %s', defined, color='BLUE')
                return defined
        return None

    def get_annotated_class_object(self) -> Union[str, typing.Type]:
        return self._get_annotated_class_object() or self.class_value

    def get_key_values(self):
        values = NO_VALUES
        if self.array_type == 'dict':
            for i, (key, instance) in enumerate(self._arguments.unpack()):
                if key is None and i == 0:
                    values |= ValueSet.from_sets((v.get_key_values() for v in instance.infer() if v.array_type == 'dict'))
                if key:
                    values |= ValueSet([compiled.create_simple_object(self.inference_state, key)])
        return values

    def py__simple_getitem__(self, index: int) -> Union[typing.Type, str]:
        if self.array_type == 'dict':
            for key, lazy_context in reversed(list(self._arguments.unpack())):
                if key is None:
                    values = ValueSet.from_sets((dct_value.py__simple_getitem__(index) for dct_value in lazy_context.infer() if dct_value.array_type == 'dict'))
                    if values:
                        return values
                elif key == index:
                    return lazy_context.infer()
        return super().py__simple_getitem__(index)

    def __repr__(self) -> typing.Text:
        return '<%s of %s(%s)>' % (self.__class__.__name__, self.class_value, self._arguments)

class AnonymousInstance(_BaseTreeInstance):
    _arguments = None

class CompiledInstanceName(NameWrapper):

    @iterator_to_value_set
    def infer(self) -> Union[str, bool, list[str]]:
        for result_value in self._wrapped_name.infer():
            if result_value.api_type == 'function':
                yield CompiledBoundMethod(result_value)
            else:
                yield result_value

class CompiledInstanceClassFilter(AbstractFilter):

    def __init__(self, instance, f) -> None:
        self._instance = instance
        self._class_filter = f

    def get(self, name: Union[str, None]) -> Union[str, None]:
        return self._convert(self._class_filter.get(name))

    def values(self) -> Union[str, etl.names.TableName, dict, list]:
        return self._convert(self._class_filter.values())

    def _convert(self, names: Union[str, set[str]]) -> list[LazyInstanceClassName]:
        return [CompiledInstanceName(n) for n in names]

class BoundMethod(FunctionMixin, ValueWrapper):

    def __init__(self, instance, class_context, function) -> None:
        super().__init__(function)
        self.instance = instance
        self._class_context = class_context

    def is_bound_method(self) -> bool:
        return True

    @property
    def name(self) -> ValueName:
        return FunctionNameInClass(self._class_context, super().name)

    def py__class__(self):
        c, = values_from_qualified_names(self.inference_state, 'types', 'MethodType')
        return c

    def _get_arguments(self, arguments: Union[float, None, typing.Mapping]) -> InstanceArguments:
        assert arguments is not None
        return InstanceArguments(self.instance, arguments)

    def _as_context(self, arguments: Union[None, typing.Mapping, typing.Callable]=None) -> Union[AnonymousMethodExecutionContext, MethodExecutionContext]:
        if arguments is None:
            return AnonymousMethodExecutionContext(self.instance, self)
        arguments = self._get_arguments(arguments)
        return MethodExecutionContext(self.instance, self, arguments)

    def py__call__(self, arguments: Union[str, typing.Type]) -> Union[bool, str]:
        if isinstance(self._wrapped_value, OverloadedFunctionValue):
            return self._wrapped_value.py__call__(self._get_arguments(arguments))
        function_execution = self.as_context(arguments)
        return function_execution.infer()

    def get_signature_functions(self) -> list[BoundMethod]:
        return [BoundMethod(self.instance, self._class_context, f) for f in self._wrapped_value.get_signature_functions()]

    def get_signatures(self) -> Union[str, None, bool]:
        return [sig.bind(self) for sig in super().get_signatures()]

    def __repr__(self) -> typing.Text:
        return '<%s: %s>' % (self.__class__.__name__, self._wrapped_value)

class CompiledBoundMethod(ValueWrapper):

    def is_bound_method(self) -> bool:
        return True

    def get_signatures(self) -> Union[str, None, bool]:
        return [sig.bind(self) for sig in self._wrapped_value.get_signatures()]

class SelfName(TreeNameDefinition):
    """
    This name calculates the parent_context lazily.
    """

    def __init__(self, instance, class_context, tree_name) -> None:
        self._instance = instance
        self.class_context = class_context
        self.tree_name = tree_name

    @property
    def parent_context(self) -> Union[list, bool]:
        return self._instance.create_instance_context(self.class_context, self.tree_name)

    def get_defining_qualified_value(self):
        return self._instance

    def infer(self) -> Union[str, bool, list[str]]:
        stmt = search_ancestor(self.tree_name, 'expr_stmt')
        if stmt is not None:
            if stmt.children[1].type == 'annassign':
                from jedi.inference.gradual.annotation import infer_annotation
                values = infer_annotation(self.parent_context, stmt.children[1].children[1]).execute_annotation()
                if values:
                    return values
        return super().infer()

class LazyInstanceClassName(NameWrapper):

    def __init__(self, instance, class_member_name) -> None:
        super().__init__(class_member_name)
        self._instance = instance

    @iterator_to_value_set
    def infer(self) -> Union[str, bool, list[str]]:
        for result_value in self._wrapped_name.infer():
            yield from result_value.py__get__(self._instance, self._instance.py__class__())

    def get_signatures(self) -> Union[str, None, bool]:
        return self.infer().get_signatures()

    def get_defining_qualified_value(self):
        return self._instance

class InstanceClassFilter(AbstractFilter):
    """
    This filter is special in that it uses the class filter and wraps the
    resulting names in LazyInstanceClassName. The idea is that the class name
    filtering can be very flexible and always be reflected in instances.
    """

    def __init__(self, instance, class_filter) -> None:
        self._instance = instance
        self._class_filter = class_filter

    def get(self, name: Union[str, None]) -> Union[str, None]:
        return self._convert(self._class_filter.get(name))

    def values(self) -> Union[str, etl.names.TableName, dict, list]:
        return self._convert(self._class_filter.values())

    def _convert(self, names: Union[str, set[str]]) -> list[LazyInstanceClassName]:
        return [LazyInstanceClassName(self._instance, n) for n in names]

    def __repr__(self) -> typing.Text:
        return '<%s for %s>' % (self.__class__.__name__, self._class_filter)

class SelfAttributeFilter(ClassFilter):
    """
    This class basically filters all the use cases where `self.*` was assigned.
    """

    def __init__(self, instance, instance_class, node_context, origin_scope) -> None:
        super().__init__(class_value=instance_class, node_context=node_context, origin_scope=origin_scope, is_instance=True)
        self._instance = instance

    def _filter(self, names: Union[str, list[str], list]) -> str:
        start, end = (self._parser_scope.start_pos, self._parser_scope.end_pos)
        names = [n for n in names if start < n.start_pos < end]
        return self._filter_self_names(names)

    def _filter_self_names(self, names: str) -> typing.Generator[typing.Text]:
        for name in names:
            trailer = name.parent
            if trailer.type == 'trailer' and len(trailer.parent.children) == 2 and (trailer.children[0] == '.'):
                if name.is_definition() and self._access_possible(name):
                    if self._is_in_right_scope(trailer.parent.children[0], name):
                        yield name

    def _is_in_right_scope(self, self_name: Union[str, list[str]], name: Union[list[str], str, set[str]]) -> bool:
        self_context = self._node_context.create_context(self_name)
        names = self_context.goto(self_name, position=self_name.start_pos)
        return any((n.api_type == 'param' and n.tree_name.get_definition().position_index == 0 and (n.parent_context.tree_node is self._parser_scope) for n in names))

    def _convert_names(self, names: str) -> list[SelfName]:
        return [SelfName(self._instance, self._node_context, name) for name in names]

    def _check_flows(self, names: Union[str, set[str], list[str]]) -> Union[str, set[str], list[str]]:
        return names

class InstanceArguments(TreeArgumentsWrapper):

    def __init__(self, instance, arguments: Union[str, None, int, typing.Mapping]) -> None:
        super().__init__(arguments)
        self.instance = instance

    def unpack(self, func: Union[None, typing.Callable, bytes, bytearray]=None) -> Union[typing.Generator[tuple[typing.Optional[LazyKnownValue]]], typing.Generator]:
        yield (None, LazyKnownValue(self.instance))
        yield from self._wrapped_arguments.unpack(func)