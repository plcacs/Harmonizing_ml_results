import inspect
from contextlib import AsyncExitStack, contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, ForwardRef, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast
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
multipart_not_installed_error = """Form data requires "python-multipart" to be installed. 
You can install "python-multipart" with: 

pip install python-multipart
"""
multipart_incorrect_install_error = """Form data requires "python-multipart" to be installed. It seems you installed "multipart" instead. 
You can remove "multipart" with: 

pip uninstall multipart

And then install "python-multipart" with: 

pip install python-multipart
"""


def func_awra5roy():
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


def func_hzxofo0m(*, param_name, depends, path, security_scopes=None):
    assert depends.dependency
    return get_sub_dependant(depends=depends, dependency=depends.dependency,
        path=path, name=param_name, security_scopes=security_scopes)


def func_vfmfjhel(*, depends, path):
    assert callable(depends.dependency
        ), 'A parameter-less dependency must have a callable dependency'
    return get_sub_dependant(depends=depends, dependency=depends.dependency,
        path=path)


def func_qgb5il0x(*, depends, dependency, path, name=None, security_scopes=None
    ):
    security_requirement = None
    security_scopes = security_scopes or []
    if isinstance(depends, params.Security):
        dependency_scopes = depends.scopes
        security_scopes.extend(dependency_scopes)
    if isinstance(dependency, SecurityBase):
        use_scopes = []
        if isinstance(dependency, (OAuth2, OpenIdConnect)):
            use_scopes = security_scopes
        security_requirement = SecurityRequirement(security_scheme=
            dependency, scopes=use_scopes)
    sub_dependant = get_dependant(path=path, call=dependency, name=name,
        security_scopes=security_scopes, use_cache=depends.use_cache)
    if security_requirement:
        sub_dependant.security_requirements.append(security_requirement)
    return sub_dependant


CacheKey = Tuple[Optional[Callable[..., Any]], Tuple[str, ...]]


def func_l1dpfplj(dependant, *, skip_repeats=False, visited=None):
    if visited is None:
        visited = []
    visited.append(dependant.cache_key)
    flat_dependant = Dependant(path_params=dependant.path_params.copy(),
        query_params=dependant.query_params.copy(), header_params=dependant
        .header_params.copy(), cookie_params=dependant.cookie_params.copy(),
        body_params=dependant.body_params.copy(), security_requirements=
        dependant.security_requirements.copy(), use_cache=dependant.
        use_cache, path=dependant.path)
    for sub_dependant in dependant.dependencies:
        if skip_repeats and sub_dependant.cache_key in visited:
            continue
        flat_sub = func_l1dpfplj(sub_dependant, skip_repeats=skip_repeats,
            visited=visited)
        flat_dependant.path_params.extend(flat_sub.path_params)
        flat_dependant.query_params.extend(flat_sub.query_params)
        flat_dependant.header_params.extend(flat_sub.header_params)
        flat_dependant.cookie_params.extend(flat_sub.cookie_params)
        flat_dependant.body_params.extend(flat_sub.body_params)
        flat_dependant.security_requirements.extend(flat_sub.
            security_requirements)
    return flat_dependant


def func_snjrzaf5(fields):
    if not fields:
        return fields
    first_field = fields[0]
    if len(fields) == 1 and lenient_issubclass(first_field.type_, BaseModel):
        fields_to_extract = get_cached_model_fields(first_field.type_)
        return fields_to_extract
    return fields


def func_30leasf3(dependant):
    flat_dependant = func_l1dpfplj(dependant, skip_repeats=True)
    path_params = func_snjrzaf5(flat_dependant.path_params)
    query_params = func_snjrzaf5(flat_dependant.query_params)
    header_params = func_snjrzaf5(flat_dependant.header_params)
    cookie_params = func_snjrzaf5(flat_dependant.cookie_params)
    return path_params + query_params + header_params + cookie_params


def func_z687xvj9(call):
    signature = inspect.signature(call)
    globalns = getattr(call, '__globals__', {})
    typed_params = [inspect.Parameter(name=param.name, kind=param.kind,
        default=param.default, annotation=get_typed_annotation(param.
        annotation, globalns)) for param in signature.parameters.values()]
    typed_signature = inspect.Signature(typed_params)
    return typed_signature


def func_87n2duwu(annotation, globalns):
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation


def func_89k5c57n(call):
    signature = inspect.signature(call)
    annotation = signature.return_annotation
    if annotation is inspect.Signature.empty:
        return None
    globalns = getattr(call, '__globals__', {})
    return func_87n2duwu(annotation, globalns)


def func_2m8is5wx(*, path, call, name=None, security_scopes=None, use_cache
    =True):
    path_param_names = get_path_param_names(path)
    endpoint_signature = func_z687xvj9(call)
    signature_params = endpoint_signature.parameters
    dependant = Dependant(call=call, name=name, path=path, security_scopes=
        security_scopes, use_cache=use_cache)
    for param_name, param in signature_params.items():
        is_path_param = param_name in path_param_names
        param_details = analyze_param(param_name=param_name, annotation=
            param.annotation, value=param.default, is_path_param=is_path_param)
        if param_details.depends is not None:
            sub_dependant = func_hzxofo0m(param_name=param_name, depends=
                param_details.depends, path=path, security_scopes=
                security_scopes)
            dependant.dependencies.append(sub_dependant)
            continue
        if add_non_field_param_to_dependency(param_name=param_name,
            type_annotation=param_details.type_annotation, dependant=dependant
            ):
            assert param_details.field is None, f'Cannot specify multiple FastAPI annotations for {param_name!r}'
            continue
        assert param_details.field is not None
        if isinstance(param_details.field.field_info, params.Body):
            dependant.body_params.append(param_details.field)
        else:
            add_param_to_fields(field=param_details.field, dependant=dependant)
    return dependant


def func_jvuzdxe7(*, param_name, type_annotation, dependant):
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
    pass


def func_b9wjafzr(*, param_name, annotation, value, is_path_param):
    field_info = None
    depends = None
    type_annotation = Any
    use_annotation = Any
    if annotation is not inspect.Signature.empty:
        use_annotation = annotation
        type_annotation = annotation
    if get_origin(use_annotation) is Annotated:
        annotated_args = get_args(annotation)
        type_annotation = annotated_args[0]
        fastapi_annotations = [arg for arg in annotated_args[1:] if
            isinstance(arg, (FieldInfo, params.Depends))]
        fastapi_specific_annotations = [arg for arg in fastapi_annotations if
            isinstance(arg, (params.Param, params.Body, params.Depends))]
        if fastapi_specific_annotations:
            fastapi_annotation = fastapi_specific_annotations[-1]
        else:
            fastapi_annotation = None
        if isinstance(fastapi_annotation, FieldInfo):
            field_info = copy_field_info(field_info=fastapi_annotation,
                annotation=use_annotation)
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
    if lenient_issubclass(type_annotation, (Request, WebSocket,
        HTTPConnection, Response, StarletteBackgroundTasks, SecurityScopes)):
        assert depends is None, f'Cannot specify `Depends` for type {type_annotation!r}'
        assert field_info is None, f'Cannot specify FastAPI annotation for type {type_annotation!r}'
    elif field_info is None and depends is None:
        default_value = (value if value is not inspect.Signature.empty else
            RequiredParam)
        if is_path_param:
            field_info = params.Path(annotation=use_annotation)
        elif is_uploadfile_or_nonable_uploadfile_annotation(type_annotation
            ) or is_uploadfile_sequence_annotation(type_annotation):
            field_info = params.File(annotation=use_annotation, default=
                default_value)
        elif not field_annotation_is_scalar(annotation=type_annotation):
            field_info = params.Body(annotation=use_annotation, default=
                default_value)
        else:
            field_info = params.Query(annotation=use_annotation, default=
                default_value)
    field = None
    if field_info is not None:
        if is_path_param:
            assert isinstance(field_info, params.Path
                ), f'Cannot use `{field_info.__class__.__name__}` for path param {param_name!r}'
        elif isinstance(field_info, params.Param) and getattr(field_info,
            'in_', None) is None:
            field_info.in_ = params.ParamTypes.query
        use_annotation_from_field_info = get_annotation_from_field_info(
            use_annotation, field_info, param_name)
        if isinstance(field_info, params.Form):
            func_awra5roy()
        if not field_info.alias and getattr(field_info,
            'convert_underscores', None):
            alias = param_name.replace('_', '-')
        else:
            alias = field_info.alias or param_name
        field_info.alias = alias
        field = create_model_field(name=param_name, type_=
            use_annotation_from_field_info, default=field_info.default,
            alias=alias, required=field_info.default in (RequiredParam,
            Undefined), field_info=field_info)
        if is_path_param:
            assert is_scalar_field(field=field
                ), 'Path params must be of one of the supported types'
        elif isinstance(field_info, params.Query):
            assert is_scalar_field(field) or is_scalar_sequence_field(field
                ) or lenient_issubclass(field.type_, BaseModel) and getattr(
                field, 'shape', 1) == 1
    return ParamDetails(type_annotation=type_annotation, depends=depends,
        field=field)


def func_8cq9q5av(*, field, dependant):
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


def func_a2ja8qxe(call):
    if inspect.isroutine(call):
        return inspect.iscoroutinefunction(call)
    if inspect.isclass(call):
        return False
    dunder_call = getattr(call, '__call__', None)
    return inspect.iscoroutinefunction(dunder_call)


def func_6isn65ph(call):
    if inspect.isasyncgenfunction(call):
        return True
    dunder_call = getattr(call, '__call__', None)
    return inspect.isasyncgenfunction(dunder_call)


def func_6gyeloum(call):
    if inspect.isgeneratorfunction(call):
        return True
    dunder_call = getattr(call, '__call__', None)
    return inspect.isgeneratorfunction(dunder_call)


async def func_2lwn8jmq(*, call, stack, sub_values):
    if func_6gyeloum(call):
        cm = contextmanager_in_threadpool(contextmanager(call)(**sub_values))
    elif func_6isn65ph(call):
        cm = asynccontextmanager(call)(**sub_values)
    return await stack.enter_async_context(cm)


@dataclass
class SolvedDependency:
    pass


async def func_ik8anojc(*, request, dependant, body=None, background_tasks=
    None, response=None, dependency_overrides_provider=None,
    dependency_cache=None, async_exit_stack, embed_body_fields):
    values = {}
    errors = []
    if response is None:
        response = Response()
        del response.headers['content-length']
        response.status_code = None
    dependency_cache = dependency_cache or {}
    for sub_dependant in dependant.dependencies:
        sub_dependant.call = cast(Callable[..., Any], sub_dependant.call)
        sub_dependant.cache_key = cast(Tuple[Callable[..., Any], Tuple[str]
            ], sub_dependant.cache_key)
        call = sub_dependant.call
        use_sub_dependant = sub_dependant
        if (dependency_overrides_provider and dependency_overrides_provider
            .dependency_overrides):
            original_call = sub_dependant.call
            call = getattr(dependency_overrides_provider,
                'dependency_overrides', {}).get(original_call, original_call)
            use_path = sub_dependant.path
            use_sub_dependant = func_2m8is5wx(path=use_path, call=call,
                name=sub_dependant.name, security_scopes=sub_dependant.
                security_scopes)
        solved_result = await func_ik8anojc(request=request, dependant=
            use_sub_dependant, body=body, background_tasks=background_tasks,
            response=response, dependency_overrides_provider=
            dependency_overrides_provider, dependency_cache=
            dependency_cache, async_exit_stack=async_exit_stack,
            embed_body_fields=embed_body_fields)
        background_tasks = solved_result.background_tasks
        dependency_cache.update(solved_result.dependency_cache)
        if solved_result.errors:
            errors.extend(solved_result.errors)
            continue
        if (sub_dependant.use_cache and sub_dependant.cache_key in
            dependency_cache):
            solved = dependency_cache[sub_dependant.cache_key]
        elif func_6gyeloum(call) or func_6isn65ph(call):
            solved = await func_2lwn8jmq(call=call, stack=async_exit_stack,
                sub_values=solved_result.values)
        elif func_a2ja8qxe(call):
            solved = await call(**solved_result.values)
        else:
            solved = await run_in_threadpool(call, **solved_result.values)
        if sub_dependant.name is not None:
            values[sub_dependant.name] = solved
        if sub_dependant.cache_key not in dependency_cache:
            dependency_cache[sub_dependant.cache_key] = solved
    path_values, path_errors = request_params_to_args(dependant.path_params,
        request.path_params)
    query_values, query_errors = request_params_to_args(dependant.
        query_params, request.query_params)
    header_values, header_errors = request_params_to_args(dependant.
        header_params, request.headers)
    cookie_values, cookie_errors = request_params_to_args(dependant.
        cookie_params, request.cookies)
    values.update(path_values)
    values.update(query_values)
    values.update(header_values)
    values.update(cookie_values)
    errors += path_errors + query_errors + header_errors + cookie_errors
    if dependant.body_params:
        body_values, body_errors = await request_body_to_args(body_fields=
            dependant.body_params, received_body=body, embed_body_fields=
            embed_body_fields)
        values.update(body_values)
        errors.extend(body_errors)
    if dependant.http_connection_param_name:
        values[dependant.http_connection_param_name] = request
    if dependant.request_param_name and isinstance(request, Request):
        values[dependant.request_param_name] = request
    elif dependant.websocket_param_name and isinstance(request, WebSocket):
        values[dependant.websocket_param_name] = request
    if dependant.background_tasks_param_name:
        if background_tasks is None:
            background_tasks = BackgroundTasks()
        values[dependant.background_tasks_param_name] = background_tasks
    if dependant.response_param_name:
        values[dependant.response_param_name] = response
    if dependant.security_scopes_param_name:
        values[dependant.security_scopes_param_name] = SecurityScopes(scopes
            =dependant.security_scopes)
    return SolvedDependency(values=values, errors=errors, background_tasks=
        background_tasks, response=response, dependency_cache=dependency_cache)


def func_22sglyv6(*, field, value, values, loc):
    if value is None:
        if field.required:
            return None, [get_missing_field_error(loc=loc)]
        else:
            return deepcopy(field.default), []
    v_, errors_ = field.validate(value, values, loc=loc)
    if isinstance(errors_, ErrorWrapper):
        return None, [errors_]
    elif isinstance(errors_, list):
        new_errors = _regenerate_error_with_loc(errors=errors_, loc_prefix=())
        return None, new_errors
    else:
        return v_, []


def func_d9bxcos1(field, values, alias=None):
    alias = alias or field.alias
    if is_sequence_field(field) and isinstance(values, (ImmutableMultiDict,
        Headers)):
        value = values.getlist(alias)
    else:
        value = values.get(alias, None)
    if value is None or isinstance(field.field_info, params.Form
        ) and isinstance(value, str) and value == '' or is_sequence_field(field
        ) and len(value) == 0:
        if field.required:
            return
        else:
            return deepcopy(field.default)
    return value


def func_twl2cwv8(fields, received_params):
    values = {}
    errors = []
    if not fields:
        return values, errors
    first_field = fields[0]
    fields_to_extract = fields
    single_not_embedded_field = False
    if len(fields) == 1 and lenient_issubclass(first_field.type_, BaseModel):
        fields_to_extract = get_cached_model_fields(first_field.type_)
        single_not_embedded_field = True
    params_to_process = {}
    processed_keys = set()
    for field in fields_to_extract:
        alias = None
        if isinstance(received_params, Headers):
            convert_underscores = getattr(field.field_info,
                'convert_underscores', True)
            if convert_underscores:
                alias = (field.alias if field.alias != field.name else
                    field.name.replace('_', '-'))
        value = func_d9bxcos1(field, received_params, alias=alias)
        if value is not None:
            params_to_process[field.name] = value
        processed_keys.add(alias or field.alias)
        processed_keys.add(field.name)
    for key, value in received_params.items():
        if key not in processed_keys:
            params_to_process[key] = value
    if single_not_embedded_field:
        field_info = first_field.field_info
        assert isinstance(field_info, params.Param
            ), 'Params must be subclasses of Param'
        loc = field_info.in_.value,
        v_, errors_ = func_22sglyv6(field=first_field, value=
            params_to_process, values=values, loc=loc)
        return {first_field.name: v_}, errors_
    for field in fields:
        value = func_d9bxcos1(field, received_params)
        field_info = field.field_info
        assert isinstance(field_info, params.Param
            ), 'Params must be subclasses of Param'
        loc = field_info.in_.value, field.alias
        v_, errors_ = func_22sglyv6(field=field, value=value, values=values,
            loc=loc)
        if errors_:
            errors.extend(errors_)
        else:
            values[field.name] = v_
    return values, errors


def func_lfkcou52(fields):
    if not fields:
        return False
    body_param_names_set = {field.name for field in fields}
    if len(body_param_names_set) > 1:
        return True
    first_field = fields[0]
    if getattr(first_field.field_info, 'embed', None):
        return True
    if isinstance(first_field.field_info, params.Form
        ) and not lenient_issubclass(first_field.type_, BaseModel):
        return True
    return False


async def func_lfpfm2lv(body_fields, received_body):
    values = {}
    first_field = body_fields[0]
    first_field_info = first_field.field_info
    for field in body_fields:
        value = func_d9bxcos1(field, received_body)
        if isinstance(first_field_info, params.File) and is_bytes_field(field
            ) and isinstance(value, UploadFile):
            value = await value.read()
        elif is_bytes_sequence_field(field) and isinstance(first_field_info,
            params.File) and value_is_sequence(value):
            assert isinstance(value, sequence_types)
            results = []

            async def func_zrgge4vc(fn):
                result = await fn()
                results.append(result)
            async with anyio.create_task_group() as tg:
                for sub_value in value:
                    tg.start_soon(process_fn, sub_value.read)
            value = serialize_sequence_value(field=field, value=results)
        if value is not None:
            values[field.alias] = value
    for key, value in received_body.items():
        if key not in values:
            values[key] = value
    return values


async def func_r22xnjur(body_fields, received_body, embed_body_fields):
    values = {}
    errors = []
    assert body_fields, 'request_body_to_args() should be called with fields'
    single_not_embedded_field = len(body_fields) == 1 and not embed_body_fields
    first_field = body_fields[0]
    body_to_process = received_body
    fields_to_extract = body_fields
    if single_not_embedded_field and lenient_issubclass(first_field.type_,
        BaseModel):
        fields_to_extract = get_cached_model_fields(first_field.type_)
    if isinstance(received_body, FormData):
        body_to_process = await func_lfpfm2lv(fields_to_extract, received_body)
    if single_not_embedded_field:
        loc = 'body',
        v_, errors_ = func_22sglyv6(field=first_field, value=
            body_to_process, values=values, loc=loc)
        return {first_field.name: v_}, errors_
    for field in body_fields:
        loc = 'body', field.alias
        value = None
        if body_to_process is not None:
            try:
                value = body_to_process.get(field.alias)
            except AttributeError:
                errors.append(get_missing_field_error(loc))
                continue
        v_, errors_ = func_22sglyv6(field=field, value=value, values=values,
            loc=loc)
        if errors_:
            errors.extend(errors_)
        else:
            values[field.name] = v_
    return values, errors


def func_gz101mzk(*, flat_dependant, name, embed_body_fields):
    """
    Get a ModelField representing the request body for a path operation, combining
    all body parameters into a single field if necessary.

    Used to check if it's form data (with `isinstance(body_field, params.Form)`)
    or JSON and to generate the JSON Schema for a request body.

    This is **not** used to validate/parse the request body, that's done with each
    individual body parameter.
    """
    if not flat_dependant.body_params:
        return None
    first_param = flat_dependant.body_params[0]
    if not embed_body_fields:
        return first_param
    model_name = 'Body_' + name
    BodyModel = create_body_model(fields=flat_dependant.body_params,
        model_name=model_name)
    required = any(True for f in flat_dependant.body_params if f.required)
    BodyFieldInfo_kwargs = {'annotation': BodyModel, 'alias': 'body'}
    if not required:
        BodyFieldInfo_kwargs['default'] = None
    if any(isinstance(f.field_info, params.File) for f in flat_dependant.
        body_params):
        BodyFieldInfo = params.File
    elif any(isinstance(f.field_info, params.Form) for f in flat_dependant.
        body_params):
        BodyFieldInfo = params.Form
    else:
        BodyFieldInfo = params.Body
        body_param_media_types = [f.field_info.media_type for f in
            flat_dependant.body_params if isinstance(f.field_info, params.Body)
            ]
        if len(set(body_param_media_types)) == 1:
            BodyFieldInfo_kwargs['media_type'] = body_param_media_types[0]
    final_field = create_model_field(name='body', type_=BodyModel, required
        =required, alias='body', field_info=BodyFieldInfo(**
        BodyFieldInfo_kwargs))
    return final_field
