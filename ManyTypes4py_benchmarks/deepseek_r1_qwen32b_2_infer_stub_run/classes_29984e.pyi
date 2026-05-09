"""
Stub file for 'classes_29984e' module
"""

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Dict,
    Set,
    FrozenSet,
    Type,
    TypeVar,
    overload,
    TYPE_CHECKING,
)
from pathlib import Path
from jedi.inference.base_value import ValueSet, HasNoContext
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.names import ImportName, SubModuleName
from jedi.api.helpers import filter_follow_imports
import inspect

if TYPE_CHECKING:
    from jedi.inference.compiled.mixed import MixedName
    from jedi.inference.gradual.conversion import convert_names, convert_values
    from jedi.inference.base_value import ValueSet, HasNoContext
    from jedi.api import completion_cache
    from jedi.api.helpers import filter_follow_imports

class BaseName:
    _mapping: Dict[str, str]
    _tuple_mapping: Dict[Tuple[str, ...], str]

    def __init__(self, inference_state: Any, name: Any) -> None:
        ...

    @memoize_method
    def _get_module_context(self) -> Any:
        ...

    @property
    def module_path(self) -> Optional[Path]:
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

    def is_side_effect(self) -> bool:
        ...

    @debug.increase_indent_cm('goto on name')
    def goto(
        self,
        follow_imports: bool = False,
        follow_builtin_imports: bool = False,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List[Name]:
        ...

    @debug.increase_indent_cm('infer on name')
    def infer(
        self,
        only_stubs: bool = False,
        prefer_stubs: bool = False,
    ) -> List[Name]:
        ...

    def parent(self) -> Optional[Name]:
        ...

    def __repr__(self) -> str:
        ...

    def get_line_code(self, before: int = 0, after: int = 0) -> str:
        ...

    def _get_signatures(self, for_docstring: bool = False) -> List[Any]:
        ...

    def get_signatures(self) -> List[BaseSignature]:
        ...

    def execute(self) -> List[Name]:
        ...

    def get_type_hint(self) -> str:
        ...

class Completion(BaseName):
    def __init__(
        self,
        inference_state: Any,
        name: Any,
        stack: Any,
        like_name_length: int,
        is_fuzzy: bool,
        cached_name: Optional[Any] = None,
    ) -> None:
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

    def _get_cache(self) -> Tuple[str, str, str]:
        ...

    @property
    def type(self) -> str:
        ...

    def get_completion_prefix_length(self) -> int:
        ...

    def __repr__(self) -> str:
        ...

class Name(BaseName):
    def __init__(self, inference_state: Any, definition: Any) -> None:
        ...

    @memoize_method
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
    def __init__(
        self,
        inference_state: Any,
        signature: Any,
        call_details: Any,
    ) -> None:
        ...

    @property
    def index(self) -> Optional[int]:
        ...

    @property
    def bracket_start(self) -> Tuple[int, int]:
        ...

    def __repr__(self) -> str:
        ...

class ParamName(Name):
    def infer_default(self) -> List[Name]:
        ...

    def infer_annotation(self, **kwargs: Any) -> List[Name]:
        ...

    def to_string(self) -> str:
        ...

    @property
    def kind(self) -> inspect.Parameter.Kind:
        ...

def _sort_names_by_start_pos(names: List[Name]) -> List[Name]:
    ...

def defined_names(inference_state: Any, value: Any) -> List[Name]:
    ...

def _values_to_definitions(values: Any) -> List[Name]:
    ...