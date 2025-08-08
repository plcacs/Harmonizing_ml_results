from typing import Optional, Tuple

class CheckAttribute:
    def __init__(self, check_name: Optional[str] = None):
        self.check_name = check_name

    def __call__(self, func):
        self.func = func
        if self.check_name is None:
            self.check_name = func.__name__[2:]
        return self

    def __get__(self, instance, owner):
        if instance is None:
            return self
        instance.access_handle.getattr_paths(self.check_name)
        return partial(self.func, instance)

class CompiledValue(Value):
    def __init__(self, inference_state, access_handle, parent_context=None):
        super().__init__(inference_state, parent_context)
        self.access_handle = access_handle

    def py__call__(self, arguments) -> ValueSet:
        ...

    @CheckAttribute()
    def py__class__(self) -> Value:
        ...

    @CheckAttribute()
    def py__mro__(self) -> Tuple[Value, ...]:
        ...

    @CheckAttribute()
    def py__bases__(self) -> Tuple[Value, ...]:
        ...

    def get_qualified_names(self) -> Tuple[str, ...]:
        ...

    def py__bool__(self) -> bool:
        ...

    def is_class(self) -> bool:
        ...

    def is_function(self) -> bool:
        ...

    def is_module(self) -> bool:
        ...

    def is_compiled(self) -> bool:
        ...

    def is_stub(self) -> bool:
        ...

    def is_instance(self) -> bool:
        ...

    def py__doc__(self) -> str:
        ...

    def get_param_names(self) -> Tuple[ParamNameInterface, ...]:
        ...

    def get_signatures(self) -> List[BuiltinSignature]:
        ...

    def get_filters(self, is_instance: bool = False, origin_scope=None) -> AbstractFilter:
        ...

    def py__simple_getitem__(self, index) -> ValueSet:
        ...

    def py__getitem__(self, index_value_set, contextualized_node) -> ValueSet:
        ...

    def py__iter__(self, contextualized_node=None):
        ...

    def py__name__(self) -> str:
        ...

    @property
    def name(self) -> CompiledValueName:
        ...

    def _execute_function(self, params) -> Generator[Value, None, None]:
        ...

    def get_safe_value(self, default=_sentinel):
        ...

    def execute_operation(self, other, operator) -> ValueSet:
        ...

    def execute_annotation(self) -> ValueSet:
        ...

    def negate(self) -> Value:
        ...

    def get_metaclasses(self) -> ValueSet:
        ...

    def _as_context(self) -> CompiledContext:
        ...

    @property
    def array_type(self) -> str:
        ...

    def get_key_values(self) -> List[Value]:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...

class CompiledModule(CompiledValue):
    def _as_context(self) -> CompiledModuleContext:
        ...

    def py__path__(self) -> str:
        ...

    def is_package(self) -> bool:
        ...

    @property
    def string_names(self) -> Tuple[str, ...]:
        ...

    def py__file__(self) -> str:

class CompiledName(AbstractNameDefinition):
    def __init__(self, inference_state, parent_value, name):
        ...

    def py__doc__(self) -> str:
        ...

    def _get_qualified_names(self) -> Tuple[str, ...]:
        ...

    def get_defining_qualified_value(self):
        ...

    @property
    def api_type(self) -> str:
        ...

    def infer(self) -> ValueSet:
        ...

class SignatureParamName(ParamNameInterface, AbstractNameDefinition):
    def __init__(self, compiled_value, signature_param):
        ...

    @property
    def string_name(self) -> str:
        ...

    def to_string(self) -> str:
        ...

    def get_kind(self) -> Parameter:
        ...

    def infer(self) -> ValueSet:
        ...

class UnresolvableParamName(ParamNameInterface, AbstractNameDefinition):
    def __init__(self, compiled_value, name, default):
        ...

    def get_kind(self) -> Parameter:
        ...

    def to_string(self) -> str:
        ...

    def infer(self) -> ValueSet:
        ...

class CompiledValueName(ValueNameMixin, AbstractNameDefinition):
    def __init__(self, value, name):
        ...

class EmptyCompiledName(AbstractNameDefinition):
    def __init__(self, inference_state, name):
        ...

    def infer(self) -> ValueSet:
        ...

class CompiledValueFilter(AbstractFilter):
    def __init__(self, inference_state, compiled_value, is_instance=False):
        ...

    def get(self, name) -> List[AbstractNameDefinition]:
        ...

    def values(self) -> List[AbstractNameDefinition]:
        ...

def _parse_function_doc(doc: str) -> Tuple[str, str]:
    ...

def create_from_name(inference_state, compiled_value, name) -> Value:
    ...

def create_from_access_path(inference_state, access_path) -> Value:
    ...

@_normalize_create_args
@inference_state_function_cache()
def create_cached_compiled_value(inference_state, access_handle, parent_context) -> Value:
    ...
