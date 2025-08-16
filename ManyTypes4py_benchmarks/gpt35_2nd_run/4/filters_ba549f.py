class AbstractFilter:
    _until_position: Type[None] = None

    @abstractmethod
    def get(self, name: str) -> List[str]:
        ...

    @abstractmethod
    def values(self) -> List[str]:
        ...

class FilterWrapper:

    def __init__(self, wrapped_filter: AbstractFilter):
        ...

    def wrap_names(self, names: List[str]) -> List[str]:
        ...

    def get(self, name: str) -> List[str]:
        ...

    def values(self) -> List[str]:
        ...

def _get_definition_names(parso_cache_node, used_names, name_key) -> List[str]:
    ...

class _AbstractUsedNamesFilter(AbstractFilter):
    name_class: Type[TreeNameDefinition]

    def __init__(self, parent_context, node_context=None):
        ...

    def get(self, name: str) -> List[str]:
        ...

    def _convert_names(self, names: List[str]) -> List[str]:
        ...

    def values(self) -> List[str]:
        ...

class ParserTreeFilter(_AbstractUsedNamesFilter):

    def __init__(self, parent_context, node_context=None, until_position=None, origin_scope=None):
        ...

    def _filter(self, names: List[str]) -> List[str]:
        ...

    def _is_name_reachable(self, name: str) -> bool:
        ...

    def _check_flows(self, names: List[str]) -> List[str]:
        ...

class _FunctionExecutionFilter(ParserTreeFilter):

    def __init__(self, parent_context, function_value, until_position, origin_scope):
        ...

    def _convert_param(self, param, name: str) -> None:
        ...

    def _convert_names(self, names: List[str]) -> List[str]:
        ...

class FunctionExecutionFilter(_FunctionExecutionFilter):

    def __init__(self, *args, arguments, **kwargs):
        ...

    def _convert_param(self, param, name: str) -> None:
        ...

class AnonymousFunctionExecutionFilter(_FunctionExecutionFilter):

    def _convert_param(self, param, name: str) -> None:
        ...

class GlobalNameFilter(_AbstractUsedNamesFilter):

    def get(self, name: str) -> List[str]:
        ...

    def _filter(self, names: List[str]) -> List[str]:
        ...

    def values(self) -> List[str]:
        ...

class DictFilter(AbstractFilter):

    def __init__(self, dct: MutableMapping[str, str]):
        ...

    def get(self, name: str) -> List[str]:
        ...

    def values(self) -> List[str]:
        ...

    def _convert(self, name: str, value: str) -> str:
        ...

class MergedFilter:

    def __init__(self, *filters: AbstractFilter):
        ...

    def get(self, name: str) -> List[str]:
        ...

    def values(self) -> List[str]:
        ...

class _BuiltinMappedMethod(ValueWrapper):

    def __init__(self, value: str, method: str, builtin_func: str):
        ...

    def py__call__(self, arguments: List[str]) -> str:
        ...

class SpecialMethodFilter(DictFilter):

    class SpecialMethodName(AbstractNameDefinition):

        def __init__(self, parent_context, string_name, callable_, builtin_value):
            ...

        def infer(self) -> ValueSet:
            ...

    def __init__(self, value: str, dct: MutableMapping[str, str], builtin_value: str):
        ...

    def _convert(self, name: str, value: str) -> str:
        ...

class _OverwriteMeta(type):

    def __init__(cls, name, bases, dct):
        ...

class _AttributeOverwriteMixin:

    def get_filters(self, *args, **kwargs):
        ...

class LazyAttributeOverwrite(_AttributeOverwriteMixin, LazyValueWrapper, metaclass=_OverwriteMeta):

    def __init__(self, inference_state):
        ...

class AttributeOverwrite(_AttributeOverwriteMixin, ValueWrapper, metaclass=_OverwriteMeta):
    ...

def publish_method(method_name: str):

    def decorator(func):
        ...

    return decorator
