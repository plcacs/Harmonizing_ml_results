from typing import Any, Callable, Coroutine, Dict, ForwardRef, List, Optional, Sequence, Tuple, Type, Union
import anyio
from fastapi import params
from fastapi._compat import PYDANTIC_V2, ErrorWrapper, ModelField, RequiredParam, Undefined, _regenerate_error_with_loc, copy_field_info, create_body_model, evaluate_forwardref, field_annotation_is_scalar, get_annotation_from_field_info, get_cached_model_fields, get_missing_field_error, is_bytes_field, is_bytes_sequence_field, is_scalar_field, is_scalar_sequence_field, is_sequence_field, is_uploadfile_or_nonable_uploadfile_annotation, is_uploadfile_sequence_annotation, lenient_issubclass, sequence_types, serialize_sequence_value, value_is_sequence
from fastapi.background import BackgroundTasks
from fastapi.concurrency import asynccontextmanager, contextmanager_in_threadpool
from fastapi.dependencies.models import Dependant, SecurityRequirement
from fastapi.logger import logger
from fastapi.security.base import SecurityBase
from fastapi.security.oauth2 import OAuth2, SecurityScopes
from fastapi.security.open_id_connect_url import OpenIdConnect
from fastapi.utils import create_model_field, get_path_param_names
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, Headers, ImmutableMultiDict, QueryParams, UploadFile
from starlette.requests import HTTPConnection, Request
from starlette.responses import Response
from starlette.websockets import WebSocket
from typing_extensions import Annotated, get_args, get_origin

multipart_not_installed_error: str = 'Form data requires "python-multipart" to be installed. \nYou can install "python-multipart" with: \n\npip install python-multipart\n'
multipart_incorrect_install_error: str = 'Form data requires "python-multipart" to be installed. It seems you installed "multipart" instead. \nYou can remove "multipart" with: \n\npip uninstall multipart\n\nAnd then install "python-multipart" with: \n\npip install python-multipart\n'

def ensure_multipart_is_installed() -> None:
    ...

def get_param_sub_dependant(*, param_name: str, depends: params.Depends, path: str, security_scopes: Optional[List[str]] = None) -> Dependant:
    ...

def get_parameterless_sub_dependant(*, depends: params.Depends, path: str) -> Dependant:
    ...

def get_sub_dependant(*, depends: params.Depends, dependency: Callable[..., Any], path: str, name: Optional[str] = None, security_scopes: Optional[List[str]] = None) -> Dependant:
    ...

CacheKey: Tuple[Optional[Callable[..., Any]], Tuple[str, ...]] = ...

def get_flat_dependant(dependant: Dependant, *, skip_repeats: bool = False, visited: Optional[List[CacheKey]] = None) -> Dependant:
    ...

def _get_flat_fields_from_params(fields: List[ModelField]) -> List[ModelField]:
    ...

def get_flat_params(dependant: Dependant) -> List[ModelField]:
    ...

def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    ...

def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
    ...

def get_typed_return_annotation(call: Callable[..., Any]) -> Any:
    ...

def get_dependant(*, path: str, call: Callable[..., Any], name: Optional[str] = None, security_scopes: Optional[List[str]] = None, use_cache: bool = True) -> Dependant:
    ...

def add_non_field_param_to_dependency(*, param_name: str, type_annotation: Any, dependant: Dependant) -> Optional[bool]:
    ...

@dataclass
class ParamDetails:
    ...

def analyze_param(*, param_name: str, annotation: Any, value: Any, is_path_param: bool) -> ParamDetails:
    ...

def add_param_to_fields(*, field: ModelField, dependant: Dependant) -> None:
    ...

def is_coroutine_callable(call: Callable[..., Any]) -> bool:
    ...

def is_async_gen_callable(call: Callable[..., Any]) -> bool:
    ...

def is_gen_callable(call: Callable[..., Any]) -> bool:
    ...

async def solve_generator(*, call: Callable[..., Any], stack: AsyncExitStack, sub_values: Dict[str, Any]) -> Any:
    ...

@dataclass
class SolvedDependency:
    ...

async def solve_dependencies(*, request: Request, dependant: Dependant, body: Any = None, background_tasks: Optional[BackgroundTasks] = None, response: Optional[Response] = None, dependency_overrides_provider: Any = None, dependency_cache: Optional[Dict[CacheKey, Any]] = None, async_exit_stack: AsyncExitStack, embed_body_fields: bool) -> SolvedDependency:
    ...

def _validate_value_with_model_field(*, field: ModelField, value: Any, values: Dict[str, Any], loc: Tuple[str, ...]) -> Tuple[Any, List[ErrorWrapper]]:
    ...

def _get_multidict_value(field: ModelField, values: Union[ImmutableMultiDict, Headers], alias: Optional[str] = None) -> Any:
    ...

def request_params_to_args(fields: List[ModelField], received_params: Union[ImmutableMultiDict, Headers]) -> Tuple[Dict[str, Any], List[ErrorWrapper]]:
    ...

async def _extract_form_body(body_fields: List[ModelField], received_body: FormData) -> Dict[str, Any]:
    ...

async def request_body_to_args(body_fields: List[ModelField], received_body: Any, embed_body_fields: bool) -> Tuple[Dict[str, Any], List[ErrorWrapper]]:
    ...

def get_body_field(*, flat_dependant: Dependant, name: str, embed_body_fields: bool) -> Optional[ModelField]:
    ...
