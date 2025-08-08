from typing import Iterator, List, Union

class InstanceExecutedParamName(ParamName):
    def infer(self) -> ValueSet:
        ...

class AnonymousMethodExecutionFilter(AnonymousFunctionExecutionFilter):
    def _convert_param(self, param, name) -> Union[InstanceExecutedParamName, super()._convert_param(param, name)]:
        ...

class AnonymousMethodExecutionContext(BaseFunctionExecutionContext):
    def get_filters(self, until_position=None, origin_scope=None) -> Iterator[AnonymousMethodExecutionFilter]:
        ...

    def get_param_names(self) -> List[InstanceExecutedParamName]:
        ...

class MethodExecutionContext(FunctionExecutionContext):
    def __init__(self, instance, *args, **kwargs):
        ...

class AbstractInstanceValue(Value):
    def get_signatures(self) -> List[Signature]:
        ...

    def get_function_slot_names(self, name) -> List[NameWrapper]:
        ...

    def execute_function_slots(self, names, *inferred_args) -> ValueSet:
        ...

    def get_type_hint(self, add_class_info=True) -> str:
        ...

    def py__getitem__(self, index_value_set, contextualized_node) -> ValueSet:
        ...

    def py__iter__(self, contextualized_node=None) -> Iterator:
        ...

class CompiledInstance(AbstractInstanceValue):
    def get_filters(self, origin_scope=None, include_self_names=True) -> Iterator[CompiledInstanceClassFilter]:
        ...

class _BaseTreeInstance(AbstractInstanceValue):
    def py__getattribute__alternatives(self, string_name) -> ValueSet:
        ...

    def py__next__(self, contextualized_node=None) -> Iterator[LazyKnownValues]:
        ...

    def py__call__(self, arguments) -> ValueSet:
        ...

    def py__get__(self, instance, class_value) -> Union[ValueSet, Value]:
        ...

class TreeInstance(_BaseTreeInstance):
    def get_key_values(self) -> ValueSet:
        ...

    def py__simple_getitem__(self, index) -> Union[ValueSet, super().py__simple_getitem__(index)]:
        ...

class AnonymousInstance(_BaseTreeInstance):
    ...

class CompiledInstanceName(NameWrapper):
    def infer(self) -> Iterator[Union[CompiledBoundMethod, Value]]:
        ...

class CompiledInstanceClassFilter(AbstractFilter):
    def get(self, name) -> List[CompiledInstanceName]:
        ...

class BoundMethod(FunctionMixin, ValueWrapper):
    def py__call__(self, arguments) -> ValueSet:
        ...

class CompiledBoundMethod(ValueWrapper):
    ...

class SelfName(TreeNameDefinition):
    def infer(self) -> ValueSet:
        ...

class LazyInstanceClassName(NameWrapper):
    def infer(self) -> Iterator[Value]:
        ...

class InstanceClassFilter(AbstractFilter):
    ...

class SelfAttributeFilter(ClassFilter):
    ...

class InstanceArguments(TreeArgumentsWrapper):
    def unpack(self, func=None) -> Iterator[Tuple[Union[None, str], Union[LazyKnownValue, Value]]]:
        ...
