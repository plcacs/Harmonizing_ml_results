from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Dict,
    Type,
    TypeVar,
)
from jedi.inference.base_value import Value, ValueSet, LazyKnownValue
from jedi.inference.compiled.value import CompiledValueName
from jedi.inference.value.function import FunctionValue, OverloadedFunctionValue
from jedi.inference.value.klass import ClassValue
from jedi.inference.filters import AbstractFilter

T = TypeVar('T')

class InstanceExecutedParamName(ParamName):
    def infer(self) -> ValueSet:
        ...

class AnonymousMethodExecutionFilter(AnonymousFunctionExecutionFilter):
    def _convert_param(self, param: Any, name: str) -> Union[InstanceExecutedParamName, Any]:
        ...

class AnonymousMethodExecutionContext(BaseFunctionExecutionContext):
    def get_filters(self, until_position: Optional[int] = None, origin_scope: Optional[Any] = None) -> Iterable[AnonymousMethodExecutionFilter]:
        ...

    def get_param_names(self) -> List[InstanceExecutedParamName]:
        ...

class MethodExecutionContext(FunctionExecutionContext):
    ...

class AbstractInstanceValue(Value):
    api_type: str = 'instance'

    @property
    def name(self) -> str:
        ...

    def get_annotated_class_object(self) -> ClassValue:
        ...

    def py__class__(self) -> ClassValue:
        ...

    def py__bool__(self) -> Optional[bool]:
        ...

    def get_signatures(self) -> List[Any]:
        ...

    def get_function_slot_names(self, name: str) -> List[Any]:
        ...

    def execute_function_slots(self, names: List[Any], *inferred_args: Any) -> ValueSet:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...

    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: Any) -> ValueSet:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterator[Any]:
        ...

class CompiledInstance(AbstractInstanceValue):
    def get_filters(self, origin_scope: Optional[Any] = None, include_self_names: bool = True) -> Iterable[CompiledInstanceClassFilter]:
        ...

    @property
    def name(self) -> CompiledValueName:
        ...

class _BaseTreeInstance(AbstractInstanceValue):
    @property
    def array_type(self) -> Optional[str]:
        ...

    def get_filters(self, origin_scope: Optional[Any] = None, include_self_names: bool = True) -> Iterable[Union[SelfAttributeFilter, InstanceClassFilter, CompiledInstanceClassFilter, AbstractFilter]]:
        ...

    def create_instance_context(self, class_context: Any, node: Any) -> Any:
        ...

    def py__getattribute__alternatives(self, string_name: str) -> ValueSet:
        ...

    def py__next__(self, contextualized_node: Optional[Any] = None) -> Iterator[LazyKnownValue]:
        ...

    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

class TreeInstance(_BaseTreeInstance):
    def _get_annotated_class_object(self) -> Optional[ClassValue]:
        ...

    def get_annotated_class_object(self) -> ClassValue:
        ...

    def get_key_values(self) -> ValueSet:
        ...

    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...

class AnonymousInstance(_BaseTreeInstance):
    ...

class CompiledInstanceName(NameWrapper):
    def infer(self) -> Iterator[Value]:
        ...

class CompiledInstanceClassFilter(AbstractFilter):
    def get(self, name: str) -> List[CompiledInstanceName]:
        ...

    def values(self) -> List[CompiledInstanceName]:
        ...

class BoundMethod(FunctionMixin, ValueWrapper):
    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    def get_signature_functions(self) -> List[BoundMethod]:
        ...

    def get_signatures(self) -> List[Any]:
        ...

class CompiledBoundMethod(ValueWrapper):
    def get_signatures(self) -> List[Any]:
        ...

class SelfName(TreeNameDefinition):
    def infer(self) -> ValueSet:
        ...

class LazyInstanceClassName(NameWrapper):
    def infer(self) -> Iterator[Value]:
        ...

class InstanceClassFilter(AbstractFilter):
    def get(self, name: str) -> List[LazyInstanceClassName]:
        ...

    def values(self) -> List[LazyInstanceClassName]:
        ...

class SelfAttributeFilter(ClassFilter):
    def _filter(self, names: List[Any]) -> Iterable[Any]:
        ...

    def _filter_self_names(self, names: List[Any]) -> Iterable[Any]:
        ...

    def _convert_names(self, names: List[Any]) -> List[SelfName]:
        ...

class InstanceArguments(TreeArgumentsWrapper):
    def unpack(self, func: Optional[Callable] = None) -> Iterable[Tuple[Optional[Any], LazyKnownValue]]:
        ...