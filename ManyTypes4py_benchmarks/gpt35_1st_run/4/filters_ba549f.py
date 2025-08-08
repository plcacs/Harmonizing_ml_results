class AbstractFilter:
    _until_position: Type[None] = None

    @abstractmethod
    def get(self, name: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def values(self) -> List[str]:
        raise NotImplementedError

class FilterWrapper:

    def __init__(self, wrapped_filter: AbstractFilter):
        self._wrapped_filter = wrapped_filter

    def wrap_names(self, names: List[str]) -> List[str]:
        return [self.name_wrapper_class(name) for name in names]

    def get(self, name: str) -> List[str]:
        return self.wrap_names(self._wrapped_filter.get(name))

    def values(self) -> List[str]:
        return self.wrap_names(self._wrapped_filter.values())

def _get_definition_names(parso_cache_node, used_names: MutableMapping, name_key: str) -> List[str]:

class _AbstractUsedNamesFilter(AbstractFilter):
    name_class: Type[TreeNameDefinition]

    def __init__(self, parent_context, node_context=None):
        self._node_context = node_context
        self._parser_scope = node_context.tree_node
        self._used_names = module_context.tree_node.get_used_names()
        self.parent_context = parent_context

    def get(self, name: str) -> List[str]:

    def _convert_names(self, names: List[str]) -> List[str]:

    def values(self) -> List[str]:

class ParserTreeFilter(_AbstractUsedNamesFilter):

    def __init__(self, parent_context, node_context=None, until_position=None, origin_scope=None):
        super().__init__(parent_context, node_context)
        self._origin_scope = origin_scope
        self._until_position = until_position

    def _filter(self, names: List[str]) -> List[str]:

    def _is_name_reachable(self, name: str) -> bool:

    def _check_flows(self, names: List[str]) -> List[str]:

class _FunctionExecutionFilter(ParserTreeFilter):

    def __init__(self, parent_context, function_value, until_position, origin_scope):
        super().__init__(parent_context, until_position=until_position, origin_scope=origin_scope)
        self._function_value = function_value

    def _convert_param(self, param, name) -> str:

    def _convert_names(self, names: List[str]) -> List[str]:

class FunctionExecutionFilter(_FunctionExecutionFilter):

    def __init__(self, *args, arguments, **kwargs):
        super().__init__(*args, **kwargs)
        self._arguments = arguments

    def _convert_param(self, param, name) -> str:

class AnonymousFunctionExecutionFilter(_FunctionExecutionFilter):

    def _convert_param(self, param, name) -> str:

class GlobalNameFilter(_AbstractUsedNamesFilter):

    def get(self, name: str) -> List[str]:

    def _filter(self, names: List[str]) -> List[str]:

    def values(self) -> List[str]:

class DictFilter(AbstractFilter):

    def __init__(self, dct: MutableMapping):

    def get(self, name: str) -> List[str]:

    def values(self) -> List[str]:

    def _convert(self, name: str, value) -> str:

class MergedFilter:

    def __init__(self, *filters: List[AbstractFilter]):

    def get(self, name: str) -> List[str]:

    def values(self) -> List[str]:

class _BuiltinMappedMethod(ValueWrapper):

    def __init__(self, value, method, builtin_func):

    def py__call__(self, arguments) -> str:

class SpecialMethodFilter(DictFilter):

    class SpecialMethodName(AbstractNameDefinition):

        def __init__(self, parent_context, string_name, callable_, builtin_value):

        def infer(self) -> ValueSet:

    def __init__(self, value, dct, builtin_value):

    def _convert(self, name: str, value) -> str:

class _OverwriteMeta(type):

class _AttributeOverwriteMixin:

    def get_filters(self, *args, **kwargs):

class LazyAttributeOverwrite(_AttributeOverwriteMixin, LazyValueWrapper, metaclass=_OverwriteMeta):

    def __init__(self, inference_state):

class AttributeOverwrite(_AttributeOverwriteMixin, ValueWrapper, metaclass=_OverwriteMeta):

def publish_method(method_name: str):

    def decorator(func):

        return func
