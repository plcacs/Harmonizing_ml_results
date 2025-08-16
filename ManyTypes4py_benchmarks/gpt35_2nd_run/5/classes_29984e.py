from typing import Optional, List, Tuple
import re
from jedi.inference.names import Name, BaseName

class BaseName:
    _mapping: dict[str, str] = {'posixpath': 'os.path', 'riscospath': 'os.path', 'ntpath': 'os.path', 'os2emxpath': 'os.path', 'macpath': 'os.path', 'genericpath': 'os.path', 'posix': 'os', '_io': 'io', '_functools': 'functools', '_collections': 'collections', '_socket': 'socket', '_sqlite3': 'sqlite3'}
    _tuple_mapping: dict[tuple[str], str] = {tuple(k.split('.')): 'argparse.ArgumentParser' for k, v in {'argparse._ActionsContainer': 'argparse.ArgumentParser'}.items()}

    def __init__(self, inference_state: Any, name: Name) -> None:
        self._inference_state = inference_state
        self._name = name
        self.is_keyword = isinstance(self._name, KeywordName)

    def _get_module_context(self) -> Any:
        return self._name.get_root_context()

    @property
    def module_path(self) -> Optional[str]:
        ...

    @property
    def name(self) -> Optional[str]:
        ...

    @property
    def type(self) -> str:
        ...

    @property
    def module_name(self) -> str:
        ...

    def in_builtin_module(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def full_name(self) -> Optional[str]:
        ...

    def is_stub(self) -> bool:
        ...

    @property
    def line(self) -> Optional[int]:
        ...

    @property
    def column(self) -> Optional[int]:
        ...

    def get_definition_start_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_definition_end_position(self) -> Optional[Tuple[int, int]]:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

class Completion(BaseName):
    def __init__(self, inference_state: Any, name: Name, stack: Any, like_name_length: int, is_fuzzy: bool, cached_name: Optional[str] = None) -> None:
        ...

    def _complete(self, like_name: bool) -> str:
        ...

    @property
    def complete(self) -> Optional[str]:
        ...

    @property
    def name_with_symbols(self) -> str:
        ...

    def docstring(self, raw: bool = False, fast: bool = True) -> str:
        ...

    def _get_docstring(self) -> str:
        ...

    def _get_docstring_signature(self) -> str:
        ...

    @property
    def type(self) -> str:
        ...

    def get_completion_prefix_length(self) -> int:
        ...

class Name(BaseName):
    def __init__(self, inference_state: Any, definition: Name) -> None:
        ...

    def defined_names(self) -> List[Name]:
        ...

    def is_definition(self) -> bool:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __ne__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

class BaseSignature(Name):
    def __init__(self, inference_state: Any, signature: Any) -> None:
        ...

    @property
    def params(self) -> List[ParamName]:
        ...

    def to_string(self) -> str:
        ...

class Signature(BaseSignature):
    def __init__(self, inference_state: Any, signature: Any, call_details: Any) -> None:
        ...

    @property
    def index(self) -> Optional[int]:
        ...

    @property
    def bracket_start(self) -> Tuple[int, int]:
        ...

class ParamName(Name):
    def infer_default(self) -> List[Name]:
        ...

    def infer_annotation(self, **kwargs) -> List[Name]:
        ...

    def to_string(self) -> str:
        ...

    @property
    def kind(self) -> Any:
        ...
