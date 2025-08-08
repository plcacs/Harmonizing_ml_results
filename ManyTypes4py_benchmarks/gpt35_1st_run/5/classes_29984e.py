from typing import Optional, List, Tuple
import re
from jedi.api import completion_cache

def defined_names(inference_state, value) -> List[Name]:
    ...

def _values_to_definitions(values) -> List[Name]:
    ...

class BaseName:
    ...

    def _get_module_context(self) -> 'parso.python.tree.Module':
        ...

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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def docstring(self, raw=False, fast=True) -> str:
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

    def get_line_code(self, before=0, after=0) -> str:
        ...

    def _get_signatures(self, for_docstring=False) -> List[Signature]:
        ...

    def get_signatures(self) -> List[BaseSignature]:
        ...

    def execute(self) -> List[Name]:
        ...

    def get_type_hint(self) -> str:
        ...

class Completion(BaseName):
    ...

    @property
    def complete(self) -> Optional[str]:
        ...

    @property
    def name_with_symbols(self) -> str:
        ...

    def docstring(self, raw=False, fast=True) -> str:
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
    ...

    def defined_names(self) -> List[Name]:
        ...

    def is_definition(self) -> bool:
        ...

class BaseSignature(Name):
    ...

    @property
    def params(self) -> List[ParamName]:
        ...

    def to_string(self) -> str:
        ...

class Signature(BaseSignature):
    ...

    @property
    def index(self) -> int:
        ...

    @property
    def bracket_start(self) -> Tuple[int, int]:
        ...

class ParamName(Name):
    ...

    def infer_default(self) -> List[Name]:
        ...

    def infer_annotation(self, **kwargs) -> List[Name]:
        ...

    def to_string(self) -> str:
        ...

    @property
    def kind(self) -> 'inspect.Parameter.kind':
        ...
