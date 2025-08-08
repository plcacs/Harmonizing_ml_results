from typing import List, Tuple, Set, Union

class InferenceState:
    def __init__(self, project, environment=None, script_path=None) -> None:
        ...

    def import_module(self, import_names: Tuple[str], sys_path: Union[None, Tuple[str]] = None, prefer_stubs: bool = True) -> Tuple:
        ...

    @staticmethod
    def execute(value, arguments) -> ValueSet:
        ...

    @property
    def builtins_module(self) -> Tuple:
        ...

    @property
    def typing_module(self) -> Tuple:
        ...

    def reset_recursion_limitations(self) -> None:
        ...

    def get_sys_path(self, **kwargs) -> List[str]:
        ...

    def infer(self, context, name) -> ValueSet:
        ...

    def parse_and_get_code(self, code: Union[str, None] = None, path: Union[str, None] = None, use_latest_grammar: bool = False, file_io: Union[None, FileIO] = None, **kwargs) -> Tuple:
        ...

    def parse(self, *args, **kwargs):
        ...
