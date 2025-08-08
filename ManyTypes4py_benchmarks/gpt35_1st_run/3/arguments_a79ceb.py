from typing import Any, Iterable, Tuple, List, Dict

def try_iter_content(types: Iterable[Any], depth: int = 0) -> None:
    ...

def repack_with_argument_clinic(clinic_string: str) -> Any:
    ...

def iterate_argument_clinic(inference_state: Any, arguments: Any, clinic_string: str) -> Iterable[Any]:
    ...

def _parse_argument_clinic(string: str) -> Iterable[Tuple[str, bool, bool, int]]:
    ...

class _AbstractArgumentsMixin:
    def unpack(self, funcdef: Any = None) -> Any:
        ...

    def get_calling_nodes(self) -> List[Any]:
        ...

class AbstractArguments(_AbstractArgumentsMixin):
    context: Any = None
    argument_node: Any = None
    trailer: Any = None

    def unpack(self, funcdef: Any = None) -> Iterable[Tuple[None, Any]]:
        ...

class TreeArguments(AbstractArguments):
    def __init__(self, inference_state: Any, context: Any, argument_node: Any, trailer: Any = None) -> None:
        ...

    def unpack(self, funcdef: Any = None) -> Iterable[Tuple[None, Any]]:
        ...

    def _as_tree_tuple_objects(self) -> Iterable[Tuple[Any, Any, int]]:
        ...

    def iter_calling_names_with_star(self) -> Iterable[Any]:
        ...

    def get_calling_nodes(self) -> List[Any]:
        ...

class ValuesArguments(AbstractArguments):
    def __init__(self, values_list: List[Any]) -> None:
        ...

    def unpack(self, funcdef: Any = None) -> Iterable[Tuple[None, Any]]:
        ...

class TreeArgumentsWrapper(_AbstractArgumentsMixin):
    def __init__(self, arguments: Any) -> None:
        ...

    def unpack(self, func: Any = None) -> Any:
        ...

    def get_calling_nodes(self) -> List[Any]:
        ...
