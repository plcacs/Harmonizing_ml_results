import inspect
from contextlib import AsyncExitStack, contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, ForwardRef, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast, Set, TypeVar
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

T = TypeVar('T')

multipart_not_installed_error: str = 'Form data requires "python-multipart" to be installed. \nYou can install "python-multipart" with: \n\npip install python-multipart\n'
multipart_incorrect_install_error: str = 'Form data requires "python-multipart" to be installed. It seems you installed "multipart" instead. \nYou can remove "multipart" with: \n\npip uninstall multipart\n\nAnd then install "python-multipart" with: \n\npip install python-multipart\n'

def ensure_multipart_is_installed() -> None:
    try:
        from python_multipart import __version__
        assert __version__ > '0.0.12'
    except (ImportError, AssertionError):
        try:
            from multipart import __version__
            assert __version__
            try:
                from multipart.multipart import parse_options_header
                assert parse_options_header
            except ImportError:
                logger.error(multipart_incorrect_install_error)
                raise RuntimeError(multipart_incorrect_install_error) from None
        except ImportError:
            logger.error(multipart_not_installed_error)
            raise RuntimeError(multipart_not_installed_error) from None

def get_param_sub_dependant(*, param_name: str, depends: params.Depends, path: str, security_scopes: Optional[List[str]] = None) -> Dependant:
    assert depends.dependency
    return get_sub_dependant(depends=depends, dependency=depends.dependency, path=path, name=param_name, security_scopes=security_scopes)

def get_parameterless_sub_dependant(*, depends: params.Depends, path: str) -> Dependant:
    assert callable(depends.dependency), 'A parameter-less dependency must have a callable dependency'
    return get_sub_dependant(depends=depends, dependency=depends.dependency, path=path)

def get_sub_dependant(*, depends: params.Depends, dependency: Callable[..., Any], path: str, name: Optional[str] = None, security_scopes: Optional[List[str]] = None) -> Dependant:
    security_requirement: Optional[SecurityRequirement] = None
    security_scopes = security_scopes or []
    if isinstance(depends, params.Security):
        dependency_scopes = depends.scopes
        security_scopes.extend(dependency_scopes)
    if isinstance(dependency, SecurityBase):
        use_scopes: List[str] = []
        if isinstance(dependency, (OAuth2, OpenIdConnect)):
            use_scopes = security_scopes
        security_requirement = SecurityRequirement(security_scheme=dependency, scopes=use_scopes)
    sub_dependant = get_dependant(path=path, call=dependency, name=name, security_scopes=security_scopes, use_cache=depends.use_cache)
    if security_requirement:
        sub_dependant.security_requirements.append(security_requirement)
    return sub_dependant

CacheKey = Tuple[Optional[Callable[..., Any]], Tuple[str, ...]]

def get_flat_dependant(dependant: Dependant, *, skip_repeats: bool = False, visited: Optional[List[CacheKey]] = None) -> Dependant:
    if visited is None:
        visited = []
    visited.append(dependant.cache_key)
    flat_dependant = Dependant(path_params=dependant.path_params.copy(), query_params=dependant.query_params.copy(), header_params=dependant.header_params.copy(), cookie_params=dependant.cookie_params.copy(), body_params=dependant.body_params.copy(), security_requirements=dependant.security_requirements.copy(), use_cache=dependant.use_cache, path=dependant.path)
    for sub_dependant in dependant.dependencies:
        if skip_repeats and sub_dependant.cache_key in visited:
            continue
        flat_sub = get_flat_dependant(sub_dependant, skip_repeats=skip_repeats, visited=visited)
        flat_dependant.path_params.extend(flat_sub.path_params)
        flat_dependant.query_params.extend(flat_sub.query_params)
        flat_dependant.header_params.extend(flat_sub.header_params)
        flat_dependant.cookie_params.extend(flat_sub.cookie_params)
        flat_dependant.body_params.extend(flat_sub.body_params)
        flat_dependant.security_requirements.extend(flat_sub.security_requirements)
    return flat_dependant

def _get_flat_fields_from_params(fields: List[ModelField]) -> List[ModelField]:
    if not fields:
        return fields
    first_field = fields[0]
    if len(fields) == 1 and lenient_issubclass(first_field.type_, BaseModel):
        fields_to_extract = get_cached_model_fields(first_field.type_)
        return fields_to_extract
    return fields

def get_flat_params(dependant: Dependant) -> List[ModelField]:
    flat_dependant = get_flat_dependant(dependant, skip_repeats=True)
    path_params = _get_flat_fields_from_params(flat_dependant.path_params)
    query_params = _get_flat_fields_from_params(flat_dependant.query_params)
    header_params = _get_flat_fields_from_params(flat_dependant.header_params)
    cookie_params = _get_flat_fields_from_params(flat_dependant.cookie_params)
    return path_params + query_params + header_params + cookie_params

def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    signature = inspect.signature(call)
    globalns = getattr(call, '__globals__', {})
    typed_params = [inspect.Parameter(name=param.name, kind=param.kind, default=param.default, annotation=get_typed_annotation(param.annotation, globalns)) for param in signature.parameters.values()]
    typed_signature = inspect.Signature(typed_params)
    return typed_signature

def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation

def get_typed_return_annotation(call: Callable[..., Any]) -> Optional[Any]:
    signature = inspect.signature(call)
    annotation = signature.return_annotation
    if annotation is inspect.Signature.empty:
        return None
    globalns = getattr(call, '__globals__', {})
    return get_typed_annotation(annotation, globalns)

def get_dependant(*, path: str, call: Callable[..., Any], name: Optional[str] = None, security_scopes: Optional[List[str]] = None, use_cache: bool = True) -> Dependant:
    path_param_names = get_path_param_names(path)
    endpoint_signature = get_typed_signature(call)
    signature_params = endpoint_signature.parameters
    dependant = Dependant(call=call, name=name, path=path, security_scopes=security_scopes, use_cache=use_cache)
    for param_name, param in signature_params.items():
        is_path_param = param_name in path_param_names
        param_details = analyze_param(param_name=param_name, annotation=param.annotation, value=param.default, is_path_param=is_path_param)
        if param_details.depends is not None:
            sub_dependant = get_param_sub_dependant(param_name=param_name, depends=param_details.depends, path=path, security_scopes=security_scopes)
            dependant.dependencies.append(sub_dependant)
            continue
        if add_non_field_param_to_dependency(param_name=param_name, type_annotation=param_details.type_annotation, dependant=dependant):
            assert param_details.field is None, f'Cannot specify multiple FastAPI annotations for {param_name!r}'
            continue
        assert param_details.field is not None
        if isinstance(param_details.field.field_info, params.Body):
            dependant.body_params.append(param_details.field)
        else:
            add_param_to_fields(field=param_details.field, dependant=dependant)
    return dependant

def add_non_field_param_to_dependency(*, param_name: str, type_annotation: Any, dependant: Dependant) -> Optional[bool]:
    if lenient_issubclass(type_annotation, Request):
        dependant.request_param_name = param_name
        return True
    elif lenient_issubclass(type_annotation, WebSocket):
        dependant.websocket_param_name = param_name
        return True
    elif lenient_issubclass(type_annotation, HTTPConnection):
        dependant.http_connection_param_name = param_name
        return True
    elif lenient_issubclass(type_annotation, Response):
        dependant.response_param_name = param_name
        return True
    elif lenient_issubclass(type_annotation, StarletteBackgroundTasks):
        dependant.background_tasks_param_name = param_name
        return True
    elif lenient_issubclass(type_annotation, SecurityScopes):
        dependant.security_scopes_param_name = param_name
        return True
    return None

@dataclass
class ParamDetails:
    type_annotation: Any
    depends: Optional[params.Depends]
    field: Optional[ModelField]

def analyze_param(*, param_name: str, annotation: Any, value: Any, is_path_param: bool) -> ParamDetails:
    field_info: Optional[FieldInfo] = None
    depends: Optional[params.Depends] = None
    type_annotation: Any = Any
    use_annotation: Any = Any
    if annotation is not inspect.Signature.empty:
        use_annotation = annotation
        type_annotation = annotation
    if get_origin(use_annotation) is Annotated:
        annotated_args = get_args(annotation)
        type_annotation = annotated_args[0]
        fastapi_annotations = [arg for arg in annotated_args[1:] if isinstance(arg, (FieldInfo, params.Depends))]
        fastapi_specific_annotations = [arg for arg in fastapi_annotations if isinstance(arg, (params.Param, params.Body, params.Depends))]
        if fastapi_specific_annotations:
            fastapi_annotation = fastapi_specific_annotations[-1]
        else:
            fastapi_annotation = None
        if isinstance(fastapi_annotation, FieldInfo):
            field_info = copy_field_info(field_info=fastapi_annotation, annotation=use_annotation)
            assert field_info.default is Undefined or field_info.default is RequiredParam, f'`{field_info.__class__.__name__}` default value cannot be set in `Annotated` for {param_name!r}. Set the default value with `=` instead.'
            if value is not inspect.Signature.empty:
                assert not is_path_param, 'Path parameters cannot have default values'
                field_info.default = value
            else:
                field_info.default = RequiredParam
        elif isinstance(fastapi_annotation, params.Depends):
            depends = fastapi_annotation
    if isinstance(value, params.Depends):
        assert depends is None, f'Cannot specify `Depends` in `Annotated` and default value together for {param_name!r}'
        assert field_info is None, f'Cannot specify a FastAPI annotation in `Annotated` and `Depends` as a default value together for {param_name!r}'
        depends = value
    elif isinstance(value, FieldInfo):
        assert field_info is None, f'Cannot specify FastAPI annotations in `Annotated` and default value together for {param_name!r}'
        field_info = value
        if PYDANTIC_V2:
            field_info.annotation = type_annotation
    if depends is not None and depends.dependency is None:
        depends = copy(depends)
        depends.dependency = type_annotation
    if lenient_issubclass(type_annotation, (Request, WebSocket, HTTPConnection, Response, StarletteBackgroundTasks, SecurityScopes)):
        assert depends is None, f'Cannot specify `Depends` for type {type_annotation!r}'
        assert field_info is None, f'Cannot specify FastAPI annotation for type {type_annotation!r}'
    elif field_info is None and depends is None:
        default_value = value if value is not inspect.Signature.empty else RequiredParam
        if is_path_param:
            field_info = params.Path(annotation=use_annotation)
        elif is_uploadfile_or_nonable_uploadfile_annotation(type_annotation) or is_uploadfile_sequence_annotation(type_annotation):
            field_info = params.File(annotation=use_annotation, default=default_value)
        elif not field_annotation_is_scalar(annotation=type_annotation):
            field_info = params.Body(annotation=use_annotation, default=default_value)
        else:
            field_info = params.Query(annotation=use_annotation, default=default_value)
    field: Optional[ModelField] = None
    if field_info is not None:
        if is_path_param:
            assert isinstance(field_info, params.Path), f'Cannot use `{field_info.__class__.__name__}` for path param {param_name!r}'
        elif isinstance(field_info, params.Param) and getattr(field_info, 'in_', None) is None:
            field_info.in_ = params.ParamTypes.query
        use_annotation_from_field_info = get_annotation_from_field_info(use_annotation, field_info, param_name)
        if isinstance(field_info, params.Form):
            ensure_multipart_is_installed()
        if not field_info.alias and getattr(field_info, 'convert_underscores', None):
            alias = param_name.replace('_', '-')
        else:
            alias = field_info.alias or param_name
        field_info.alias = alias
        field = create_model_field(name=param_name, type_=use_annotation_from_field_info, default=field_info.default, alias=alias, required=field_info.default in (RequiredParam, Undefined), field_info=field_info)
        if is_path_param:
            assert is_scalar_field(field=field), 'Path params must be of one of the supported types'
        elif isinstance(field_info, params.Query):
            assert is_scalar_field(field) or is_scalar_sequence_field(field) or (lenient_issubclass(field.type_, BaseModel) and getattr(field, 'shape', 1) == 1)
    return ParamDetails(type_annotation=type_annotation, depends=depends, field=field)

def add_param_to_fields(*, field: ModelField, dependant: Dependant) -> None:
    field_info = field.field_info
    field_info_in = getattr(field_info, 'in_', None)
    if field_info_in == params.ParamTypes.path:
        dependant.path_params.append(field)
    elif field_info_in == params.ParamTypes.query:
        dependant.query_params.append(field)
    elif field_info_in == params.ParamTypes.header:
        dependant.header_params.append(field)
    else:
        assert field_info_in == params.ParamTypes.cookie, f'non-body parameters must be in path, query, header or cookie: {field.name}'
        dependant.cookie_params.append(field)

def is_coroutine_callable(call: Callable[..., Any]) -> bool:
    if inspect.isroutine(call):
        return inspect.iscoroutinefunction(call)
    if inspect.isclass(call):
        return False
    dunder_call = getattr(call, '__call__', None)
    return inspect.iscoroutinefunction(dunder_call)

def is_async_gen_callable(call: Callable[..., Any]) -> bool:
    if inspect.isasyncgenfunction(call):
        return True
    dunder_call = getattr(call, '__call__', None)
    return inspect.isasyncgenfunction(dunder_call)

def is_gen_callable(call: Callable[..., Any]) -> bool:
    if inspect.isgeneratorfunction(call):
        return True
    dunder_call = getattr(call, '__call__', None)
    return inspect.isgeneratorfunction(dunder_call)

async def solve_generator(*, call: Callable[..., Any], stack: AsyncExitStack, sub_values: Dict[str, Any]) -> Any:
    if is_gen_callable(call):
        cm = contextmanager_in_threadpool(contextmanager(call)(**sub_values))
    elif is_async_gen_callable(call):
        cm = asynccontextmanager(call)(**sub_values)
    return await stack.enter_async_context(cm)

@dataclass
class SolvedDependency:
    values: Dict[str, Any]
    errors: List[ErrorWrapper]
    background_tasks: Optional[