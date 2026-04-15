from abc import abstractproperty
from typing import (
    Iterator, Optional, Any, Union, List, Tuple, Dict, Set, 
    Generator, Iterable, TYPE_CHECKING
)
from parso.tree import NodeOrLeaf
from jedi.inference.base_value import (
    Value, NO_VALUES, ValueSet, ValueWrapper, 
    AbstractFilter, TreeNameDefinition, ParamName, NameWrapper
)
from jedi.inference.value.function import (
    FunctionValue, FunctionMixin, OverloadedFunctionValue, 
    BaseFunctionExecutionContext, FunctionExecutionContext, FunctionNameInClass
)
from jedi.inference.value.klass import ClassFilter
from jedi.inference.arguments import ValuesArguments, TreeArgumentsWrapper
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues
from jedi.inference.names import ValueName
from jedi.inference.compiled.value import CompiledValueFilter
from jedi.inference.compiled import CompiledValueName

if TYPE_CHECKING:
    from parso.python.tree import Function, Class, Name
    from jedi.inference import InferenceState
    from jedi.inference.context import Context
    from jedi.inference.gradual.annotation import TypeVarDict
    from jedi.inference.value import ClassValue
    from jedi.inference.signature import AbstractSignature
    from jedi.api.classes import Name as ApiName

class InstanceExecutedParamName(ParamName):
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        function_value: FunctionValue, 
        tree_name: "Name"
    ) -> None: ...
    def infer(self) -> ValueSet: ...
    def matches_signature(self) -> bool: ...

class AnonymousMethodExecutionFilter(AnonymousFunctionExecutionFilter):
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        *args: Any, 
        **kwargs: Any
    ) -> None: ...
    def _convert_param(
        self, 
        param: ParamName, 
        name: "Name"
    ) -> Union[InstanceExecutedParamName, ParamName]: ...

class AnonymousMethodExecutionContext(BaseFunctionExecutionContext):
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        value: FunctionValue
    ) -> None: ...
    def get_filters(
        self, 
        until_position: Optional[Tuple[int, int]] = None, 
        origin_scope: Optional["Context"] = None
    ) -> Iterator[AnonymousMethodExecutionFilter]: ...
    def get_param_names(self) -> List[Union[InstanceExecutedParamName, ParamName]]: ...

class MethodExecutionContext(FunctionExecutionContext):
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        *args: Any, 
        **kwargs: Any
    ) -> None: ...

class AbstractInstanceValue(Value):
    api_type: str = "instance"
    class_value: "ClassValue"
    
    def __init__(
        self, 
        inference_state: "InferenceState", 
        parent_context: "Context", 
        class_value: "ClassValue"
    ) -> None: ...
    def is_instance(self) -> bool: ...
    def get_qualified_names(self) -> Optional[Tuple[str, ...]]: ...
    def get_annotated_class_object(self) -> "ClassValue": ...
    def py__class__(self) -> "ClassValue": ...
    def py__bool__(self) -> Optional[bool]: ...
    @abstractproperty
    def name(self) -> Union[ValueName, CompiledValueName]: ...
    def get_signatures(self) -> List["AbstractSignature"]: ...
    def get_function_slot_names(self, name: str) -> List[NameWrapper]: ...
    def execute_function_slots(
        self, 
        names: List[NameWrapper], 
        *inferred_args: ValueSet
    ) -> ValueSet: ...
    def get_type_hint(self, add_class_info: bool = True) -> str: ...
    def py__getitem__(
        self, 
        index_value_set: ValueSet, 
        contextualized_node: Optional[NodeOrLeaf] = None
    ) -> ValueSet: ...
    def py__iter__(
        self, 
        contextualized_node: Optional[NodeOrLeaf] = None
    ) -> Iterator[ValueSet]: ...
    def py__getattribute__alternatives(self, string_name: str) -> ValueSet: ...
    def py__next__(
        self, 
        contextualized_node: Optional[NodeOrLeaf] = None
    ) -> Iterator[LazyKnownValues]: ...
    def py__call__(self, arguments: ValuesArguments) -> ValueSet: ...
    def py__get__(
        self, 
        instance: Optional["AbstractInstanceValue"], 
        class_value: "ClassValue"
    ) -> ValueSet: ...

class CompiledInstance(AbstractInstanceValue):
    _arguments: ValuesArguments
    
    def __init__(
        self, 
        inference_state: "InferenceState", 
        parent_context: "Context", 
        class_value: "ClassValue", 
        arguments: ValuesArguments
    ) -> None: ...
    def get_filters(
        self, 
        origin_scope: Optional["Context"] = None, 
        include_self_names: bool = True
    ) -> Iterator["CompiledInstanceClassFilter"]: ...
    @property
    def name(self) -> CompiledValueName: ...
    def is_stub(self) -> bool: ...

class _BaseTreeInstance(AbstractInstanceValue):
    @property
    def array_type(self) -> Optional[str]: ...
    @property
    def name(self) -> ValueName: ...
    def get_filters(
        self, 
        origin_scope: Optional["Context"] = None, 
        include_self_names: bool = True
    ) -> Iterator[Union["SelfAttributeFilter", "InstanceClassFilter", "CompiledInstanceClassFilter", AbstractFilter]]: ...
    def create_instance_context(
        self, 
        class_context: "Context", 
        node: NodeOrLeaf
    ) -> "Context": ...

class TreeInstance(_BaseTreeInstance):
    _arguments: ValuesArguments
    tree_node: Optional["Class"]
    
    def __init__(
        self, 
        inference_state: "InferenceState", 
        parent_context: "Context", 
        class_value: "ClassValue", 
        arguments: ValuesArguments
    ) -> None: ...
    def _get_annotated_class_object(self) -> Optional["ClassValue"]: ...
    def get_annotated_class_object(self) -> "ClassValue": ...
    def get_key_values(self) -> ValueSet: ...
    def py__simple_getitem__(self, index: str) -> ValueSet: ...

class AnonymousInstance(_BaseTreeInstance):
    _arguments: Optional[ValuesArguments] = None

class CompiledInstanceName(NameWrapper):
    def infer(self) -> ValueSet: ...

class CompiledInstanceClassFilter(AbstractFilter):
    def __init__(
        self, 
        instance: CompiledInstance, 
        f: CompiledValueFilter
    ) -> None: ...
    def get(self, name: str) -> List[CompiledInstanceName]: ...
    def values(self) -> List[CompiledInstanceName]: ...

class BoundMethod(FunctionMixin, ValueWrapper):
    instance: "AbstractInstanceValue"
    _class_context: "Context"
    
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        class_context: "Context", 
        function: FunctionValue
    ) -> None: ...
    def is_bound_method(self) -> bool: ...
    @property
    def name(self) -> FunctionNameInClass: ...
    def py__class__(self) -> Value: ...
    def _get_arguments(self, arguments: ValuesArguments) -> "InstanceArguments": ...
    def _as_context(
        self, 
        arguments: Optional[ValuesArguments] = None
    ) -> Union[AnonymousMethodExecutionContext, MethodExecutionContext]: ...
    def py__call__(self, arguments: ValuesArguments) -> ValueSet: ...
    def get_signature_functions(self) -> List["BoundMethod"]: ...
    def get_signatures(self) -> List["AbstractSignature"]: ...

class CompiledBoundMethod(ValueWrapper):
    def is_bound_method(self) -> bool: ...
    def get_signatures(self) -> List["AbstractSignature"]: ...

class SelfName(TreeNameDefinition):
    _instance: "AbstractInstanceValue"
    class_context: "Context"
    tree_name: "Name"
    
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        class_context: "Context", 
        tree_name: "Name"
    ) -> None: ...
    @property
    def parent_context(self) -> "Context": ...
    def get_defining_qualified_value(self) -> "AbstractInstanceValue": ...
    def infer(self) -> ValueSet: ...

class LazyInstanceClassName(NameWrapper):
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        class_member_name: NameWrapper
    ) -> None: ...
    def infer(self) -> ValueSet: ...
    def get_signatures(self) -> List["AbstractSignature"]: ...
    def get_defining_qualified_value(self) -> "AbstractInstanceValue": ...

class InstanceClassFilter(AbstractFilter):
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        class_filter: ClassFilter
    ) -> None: ...
    def get(self, name: str) -> List[LazyInstanceClassName]: ...
    def values(self) -> List[LazyInstanceClassName]: ...

class SelfAttributeFilter(ClassFilter):
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        instance_class: "ClassValue", 
        node_context: "Context", 
        origin_scope: Optional["Context"] = None
    ) -> None: ...
    def _filter(self, names: List["Name"]) -> Iterator["Name"]: ...
    def _filter_self_names(self, names: List["Name"]) -> Iterator["Name"]: ...
    def _is_in_right_scope(self, self_name: "Name", name: "Name") -> bool: ...
    def _convert_names(self, names: List["Name"]) -> List[SelfName]: ...
    def _check_flows(self, names: List["Name"]) -> List["Name"]: ...

class InstanceArguments(TreeArgumentsWrapper):
    instance: "AbstractInstanceValue"
    
    def __init__(
        self, 
        instance: "AbstractInstanceValue", 
        arguments: ValuesArguments
    ) -> None: ...
    def unpack(
        self, 
        func: Optional[FunctionValue] = None
    ) -> Iterator[Tuple[Optional[str], Union[LazyKnownValue, Any]]]: ...