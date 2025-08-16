from typing import List, Tuple, Set, Dict, Any

class InferenceState:
    def __init__(self, project: Any, environment: Any = None, script_path: Any = None) -> None:
        ...

    def import_module(self, import_names: Tuple[str], sys_path: Any = None, prefer_stubs: bool = True) -> Any:
        ...

    @staticmethod
    def execute(value: Any, arguments: Any) -> Any:
        ...

    @property
    def builtins_module(self) -> Any:
        ...

    @property
    def typing_module(self) -> Any:
        ...

    def reset_recursion_limitations(self) -> None:
        ...

    def get_sys_path(self, **kwargs: Any) -> Any:
        ...

    def infer(self, context: Any, name: Any) -> Any:
        ...

    def parse_and_get_code(self, code: str = None, path: str = None, use_latest_grammar: bool = False, file_io: Any = None, **kwargs: Any) -> Tuple[Any, str]:
        ...

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        ...
