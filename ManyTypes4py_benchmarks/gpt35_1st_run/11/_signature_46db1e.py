from __future__ import annotations
import dataclasses
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable, Dict
from pydantic_core import PydanticUndefined
from ._utils import is_valid_identifier

if TYPE_CHECKING:
    from ..config import ExtraValues
    from ..fields import FieldInfo

class _HAS_DEFAULT_FACTORY_CLASS:

    def __repr__(self) -> str:
        return '<factory>'
_HAS_DEFAULT_FACTORY = _HAS_DEFAULT_FACTORY_CLASS()

def _field_name_for_signature(field_name: str, field_info: FieldInfo) -> str:
    ...

def _process_param_defaults(param: Parameter) -> Parameter:
    ...

def _generate_signature_parameters(init: Callable, fields: Dict[str, FieldInfo], populate_by_name: bool, extra: str) -> Dict[str, Parameter]:
    ...

def generate_pydantic_signature(init: Callable, fields: Dict[str, FieldInfo], populate_by_name: bool, extra: str, is_dataclass: bool = False) -> Signature:
    ...
