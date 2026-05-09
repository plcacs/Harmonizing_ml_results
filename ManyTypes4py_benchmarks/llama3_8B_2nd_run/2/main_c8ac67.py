from typing import TypeVar, Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from types import FunctionType
from inspect import Signature
from pydantic.v1.class_validators import ValidatorGroup, extract_root_validators, extract_validators, inherit_validators
from pydantic.v1.config import BaseConfig, Extra
from pydantic.v1.error_wrappers import ErrorWrapper, ValidationError
from pydantic.v1.errors import ConfigError, DictError, ExtraError, MissingError
from pydantic.v1.fields import MAPPING_LIKE_SHAPES, Field, ModelField, ModelPrivateAttr, PrivateAttr, Undefined, is_finalvar_with_default_val
from pydantic.v1.json import custom_pydantic_encoder, pydantic_encoder
from pydantic.v1.parse import Protocol, load_file, load_str_bytes
from pydantic.v1.schema import default_ref_template, model_schema
from pydantic.v1.types import PyObject, StrBytes
from pydantic.v1.typing import AnyCallable, get_args, get_origin, is_classvar, is_namedtuple, is_union, resolve_annotations, update_model_forward_refs
from pydantic.v1.utils import DUNDER_ATTRIBUTES, ROOT_KEY, ClassAttribute, GetterDict, Representation, ValueItems, generate_model_signature, is_valid_field, is_valid_private_name, lenient_issubclass, sequence_like, smart_deepcopy, unique_list, validate_field_name

_T = TypeVar('_T')

def validate_custom_root_type(fields: Mapping[str, Any]) -> None:
    if len(fields) > 1:
        raise ValueError(f'{ROOT_KEY} cannot be mixed with other fields')

def generate_hash_function(frozen: bool) -> Callable[[Any], int]:
    def hash_function(self_: Any) -> int:
        return hash(self_.__class__) + hash(tuple(self_.__dict__.values()))
    return hash_function if frozen else None

@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class ModelMetaclass(ABCMeta):
    # ... (rest of the code remains the same)
