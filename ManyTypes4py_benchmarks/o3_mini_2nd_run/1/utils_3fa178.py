import http.client
import inspect
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

from fastapi import routing
from fastapi._compat import (
    GenerateJsonSchema,
    JsonSchemaValue,
    ModelField,
    Undefined,
    get_compat_model_name_map,
    get_definitions,
    get_schema_from_model_field,
    lenient_issubclass,
)
from fastapi.datastructures import DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import _get_flat_fields_from_params, get_flat_dependant, get_flat_params
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.constants import METHODS_WITH_BODY, REF_PREFIX, REF_TEMPLATE
from fastapi.openapi.models import OpenAPI
from fastapi.params import Body, ParamTypes
from fastapi.responses import Response
from fastapi.types import ModelNameMap
from fastapi.utils import deep_dict_update, generate_operation_id_for_path, is_body_allowed_for_status_code
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from typing_extensions import Literal

validation_error_definition: Dict[str, Any] = {
    "title": "ValidationError",
    "type": "object",
    "properties": {
        "loc": {
            "title": "Location",
            "type": "array",
            "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "msg": {"title": "Message", "type": "string"},
        "type": {"title": "Error Type", "type": "string"},
    },
    "required": ["loc", "msg", "type"],
}
validation_error_response_definition: Dict[str, Any] = {
    "title": "HTTPValidationError",
    "type": "object",
    "properties": {
        "detail": {
            "title": "Detail",
            "type": "array",
            "items": {"$ref": REF_PREFIX + "ValidationError"},
        }
    },
}
status_code_ranges: Dict[str, str] = {
    "1XX": "Information",
    "2XX": "Success",
    "3XX": "Redirection",
    "4XX": "Client Error",
    "5XX": "Server Error",
    "DEFAULT": "Default Response",
}


def get_openapi_security_definitions(
    flat_dependant: Dependant,
) -> Tuple[Dict[str, Any], List[Dict[str, List[str]]]]:
    security_definitions: Dict[str, Any] = {}
    operation_security: List[Dict[str, List[str]]] = []
    for security_requirement in flat_dependant.security_requirements:
        security_definition = jsonable_encoder(
            security_requirement.security_scheme.model, by_alias=True, exclude_none=True
        )
        security_name: str = security_requirement.security_scheme.scheme_name
        security_definitions[security_name] = security_definition
        operation_security.append({security_name: security_requirement.scopes})
    return security_definitions, operation_security


def _get_openapi_operation_parameters(
    *,
    dependant: Dependant,
    schema_generator: GenerateJsonSchema,
    model_name_map: ModelNameMap,
    field_mapping: Dict[str, Any],
    separate_input_output_schemas: bool = True,
) -> List[Dict[str, Any]]:
    parameters: List[Dict[str, Any]] = []
    flat_dependant: Dependant = get_flat_dependant(dependant, skip_repeats=True)
    path_params = _get_flat_fields_from_params(flat_dependant.path_params)
    query_params = _get_flat_fields_from_params(flat_dependant.query_params)
    header_params = _get_flat_fields_from_params(flat_dependant.header_params)
    cookie_params = _get_flat_fields_from_params(flat_dependant.cookie_params)
    parameter_groups: List[Tuple[ParamTypes, List[ModelField]]] = [
        (ParamTypes.path, path_params),
        (ParamTypes.query, query_params),
        (ParamTypes.header, header_params),
        (ParamTypes.cookie, cookie_params),
    ]
    for param_type, param_group in parameter_groups:
        for param in param_group:
            field_info = param.field_info
            if not getattr(field_info, "include_in_schema", True):
                continue
            param_schema: Dict[str, Any] = get_schema_from_model_field(
                field=param,
                schema_generator=schema_generator,
                model_name_map=model_name_map,
                field_mapping=field_mapping,
                separate_input_output_schemas=separate_input_output_schemas,
            )
            parameter: Dict[str, Any] = {
                "name": param.alias,
                "in": param_type.value,
                "required": param.required,
                "schema": param_schema,
            }
            if field_info.description:
                parameter["description"] = field_info.description
            openapi_examples = getattr(field_info, "openapi_examples", None)
            example = getattr(field_info, "example", None)
            if openapi_examples:
                parameter["examples"] = jsonable_encoder(openapi_examples)
            elif example != Undefined:
                parameter["example"] = jsonable_encoder(example)
            if getattr(field_info, "deprecated", None):
                parameter["deprecated"] = True
            parameters.append(parameter)
    return parameters


def get_openapi_operation_request_body(
    *,
    body_field: Optional[ModelField],
    schema_generator: GenerateJsonSchema,
    model_name_map: ModelNameMap,
    field_mapping: Dict[str, Any],
    separate_input_output_schemas: bool = True,
) -> Optional[Dict[str, Any]]:
    if not body_field:
        return None
    assert isinstance(body_field, ModelField)
    body_schema: Dict[str, Any] = get_schema_from_model_field(
        field=body_field,
        schema_generator=schema_generator,
        model_name_map=model_name_map,
        field_mapping=field_mapping,
        separate_input_output_schemas=separate_input_output_schemas,
    )
    field_info = cast(Body, body_field.field_info)
    request_media_type: str = field_info.media_type
    required: bool = body_field.required
    request_body_oai: Dict[str, Any] = {}
    if required:
        request_body_oai["required"] = required
    request_media_content: Dict[str, Any] = {"schema": body_schema}
    if field_info.openapi_examples:
        request_media_content["examples"] = jsonable_encoder(field_info.openapi_examples)
    elif field_info.example != Undefined:
        request_media_content["example"] = jsonable_encoder(field_info.example)
    request_body_oai["content"] = {request_media_type: request_media_content}
    return request_body_oai


def generate_operation_id(*, route: routing.APIRoute, method: str) -> str:
    warnings.warn(
        "fastapi.openapi.utils.generate_operation_id() was deprecated, it is not used internally, and will be removed soon",
        DeprecationWarning,
        stacklevel=2,
    )
    if route.operation_id:
        return route.operation_id
    path: str = route.path_format
    return generate_operation_id_for_path(name=route.name, path=path, method=method)


def generate_operation_summary(*, route: routing.APIRoute, method: str) -> str:
    if route.summary:
        return route.summary
    return route.name.replace("_", " ").title()


def get_openapi_operation_metadata(
    *,
    route: routing.APIRoute,
    method: str,
    operation_ids: Set[str],
) -> Dict[str, Any]:
    operation: Dict[str, Any] = {}
    if route.tags:
        operation["tags"] = route.tags
    operation["summary"] = generate_operation_summary(route=route, method=method)
    if route.description:
        operation["description"] = route.description
    operation_id: str = route.operation_id or route.unique_id
    if operation_id in operation_ids:
        message: str = f"Duplicate Operation ID {operation_id} for function " + f"{route.endpoint.__name__}"
        file_name: Optional[str] = getattr(route.endpoint, "__globals__", {}).get("__file__")
        if file_name:
            message += f" at {file_name}"
        warnings.warn(message, stacklevel=1)
    operation_ids.add(operation_id)
    operation["operationId"] = operation_id
    if route.deprecated:
        operation["deprecated"] = route.deprecated
    return operation


def get_openapi_path(
    *,
    route: routing.APIRoute,
    operation_ids: Set[str],
    schema_generator: GenerateJsonSchema,
    model_name_map: ModelNameMap,
    field_mapping: Dict[str, Any],
    separate_input_output_schemas: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    path: Dict[str, Any] = {}
    security_schemes: Dict[str, Any] = {}
    definitions: Dict[str, Any] = {}
    assert route.methods is not None, "Methods must be a list"
    if isinstance(route.response_class, DefaultPlaceholder):
        current_response_class = route.response_class.value
    else:
        current_response_class = route.response_class
    assert current_response_class, "A response class is needed to generate OpenAPI"
    route_response_media_type: str = current_response_class.media_type
    if route.include_in_schema:
        for method in route.methods:
            operation: Dict[str, Any] = get_openapi_operation_metadata(
                route=route, method=method, operation_ids=operation_ids
            )
            parameters: List[Dict[str, Any]] = []
            flat_dependant: Dependant = get_flat_dependant(route.dependant, skip_repeats=True)
            security_definitions, operation_security = get_openapi_security_definitions(flat_dependant=flat_dependant)
            if operation_security:
                operation.setdefault("security", []).extend(operation_security)
            if security_definitions:
                security_schemes.update(security_definitions)
            operation_parameters: List[Dict[str, Any]] = _get_openapi_operation_parameters(
                dependant=route.dependant,
                schema_generator=schema_generator,
                model_name_map=model_name_map,
                field_mapping=field_mapping,
                separate_input_output_schemas=separate_input_output_schemas,
            )
            parameters.extend(operation_parameters)
            if parameters:
                all_parameters: Dict[Tuple[str, str], Dict[str, Any]] = {
                    (param["in"], param["name"]): param for param in parameters
                }
                required_parameters: Dict[Tuple[str, str], Dict[str, Any]] = {
                    (param["in"], param["name"]): param for param in parameters if param.get("required")
                }
                all_parameters.update(required_parameters)
                operation["parameters"] = list(all_parameters.values())
            if method in METHODS_WITH_BODY:
                request_body_oai: Optional[Dict[str, Any]] = get_openapi_operation_request_body(
                    body_field=route.body_field,
                    schema_generator=schema_generator,
                    model_name_map=model_name_map,
                    field_mapping=field_mapping,
                    separate_input_output_schemas=separate_input_output_schemas,
                )
                if request_body_oai:
                    operation["requestBody"] = request_body_oai
            if route.callbacks:
                callbacks: Dict[str, Any] = {}
                for callback in route.callbacks:
                    if isinstance(callback, routing.APIRoute):
                        cb_path, cb_security_schemes, cb_definitions = get_openapi_path(
                            route=callback,
                            operation_ids=operation_ids,
                            schema_generator=schema_generator,
                            model_name_map=model_name_map,
                            field_mapping=field_mapping,
                            separate_input_output_schemas=separate_input_output_schemas,
                        )
                        callbacks[callback.name] = {callback.path: cb_path}
                operation["callbacks"] = callbacks
            if route.status_code is not None:
                status_code: str = str(route.status_code)
            else:
                response_signature = inspect.signature(current_response_class.__init__)
                status_code_param = response_signature.parameters.get("status_code")
                if status_code_param is not None:
                    if isinstance(status_code_param.default, int):
                        status_code = str(status_code_param.default)
                    else:
                        status_code = ""
                else:
                    status_code = ""
            operation.setdefault("responses", {}).setdefault(status_code, {})["description"] = route.response_description
            if route_response_media_type and is_body_allowed_for_status_code(route.status_code):
                response_schema: Dict[str, Any] = {"type": "string"}
                if lenient_issubclass(current_response_class, JSONResponse):
                    if route.response_field:
                        response_schema = get_schema_from_model_field(
                            field=route.response_field,
                            schema_generator=schema_generator,
                            model_name_map=model_name_map,
                            field_mapping=field_mapping,
                            separate_input_output_schemas=separate_input_output_schemas,
                        )
                    else:
                        response_schema = {}
                operation.setdefault("responses", {})\
                    .setdefault(status_code, {})\
                    .setdefault("content", {})\
                    .setdefault(route_response_media_type, {})["schema"] = response_schema
            if route.responses:
                operation_responses: Dict[str, Any] = operation.setdefault("responses", {})
                for additional_status_code, additional_response in route.responses.items():
                    process_response: Dict[str, Any] = additional_response.copy()
                    process_response.pop("model", None)
                    status_code_key: str = str(additional_status_code).upper()
                    if status_code_key == "DEFAULT":
                        status_code_key = "default"
                    openapi_response: Dict[str, Any] = operation_responses.setdefault(status_code_key, {})
                    assert isinstance(process_response, dict), "An additional response must be a dict"
                    field = route.response_fields.get(additional_status_code)
                    additional_field_schema: Optional[Dict[str, Any]] = None
                    if field:
                        additional_field_schema = get_schema_from_model_field(
                            field=field,
                            schema_generator=schema_generator,
                            model_name_map=model_name_map,
                            field_mapping=field_mapping,
                            separate_input_output_schemas=separate_input_output_schemas,
                        )
                        media_type: str = route_response_media_type or "application/json"
                        additional_schema: Dict[str, Any] = process_response.setdefault("content", {})\
                            .setdefault(media_type, {})\
                            .setdefault("schema", {})
                        deep_dict_update(additional_schema, additional_field_schema)
                    status_text: Optional[str] = status_code_ranges.get(str(additional_status_code).upper()) or \
                        http.client.responses.get(int(additional_status_code))
                    description: str = process_response.get("description") or \
                        openapi_response.get("description") or \
                        (status_text if status_text is not None else "Additional Response")
                    deep_dict_update(openapi_response, process_response)
                    openapi_response["description"] = description
            http422: str = str(HTTP_422_UNPROCESSABLE_ENTITY)
            all_route_params = get_flat_params(route.dependant)
            if (all_route_params or route.body_field) and (not any((status in operation["responses"] for status in [http422, "4XX", "default"]))):
                operation["responses"][http422] = {
                    "description": "Validation Error",
                    "content": {
                        "application/json": {"schema": {"$ref": REF_PREFIX + "HTTPValidationError"}}
                    },
                }
                if "ValidationError" not in definitions:
                    definitions.update(
                        {"ValidationError": validation_error_definition, "HTTPValidationError": validation_error_response_definition}
                    )
            if route.openapi_extra:
                deep_dict_update(operation, route.openapi_extra)
            path[method.lower()] = operation
    return path, security_schemes, definitions


def get_fields_from_routes(routes: Sequence[BaseRoute]) -> List[ModelField]:
    body_fields_from_routes: List[ModelField] = []
    responses_from_routes: List[ModelField] = []
    request_fields_from_routes: List[ModelField] = []
    callback_flat_models: List[ModelField] = []
    for route in routes:
        if getattr(route, "include_in_schema", None) and isinstance(route, routing.APIRoute):
            if route.body_field:
                assert isinstance(route.body_field, ModelField), "A request body must be a Pydantic Field"
                body_fields_from_routes.append(route.body_field)
            if route.response_field:
                responses_from_routes.append(route.response_field)
            if route.response_fields:
                responses_from_routes.extend(route.response_fields.values())
            if route.callbacks:
                callback_flat_models.extend(get_fields_from_routes(route.callbacks))
            params = get_flat_params(route.dependant)
            request_fields_from_routes.extend(params)
    flat_models: List[ModelField] = callback_flat_models + list(body_fields_from_routes + responses_from_routes + request_fields_from_routes)
    return flat_models


def get_openapi(
    *,
    title: str,
    version: str,
    openapi_version: str = "3.1.0",
    summary: Optional[str] = None,
    description: Optional[str] = None,
    routes: Sequence[BaseRoute],
    webhooks: Optional[Sequence[BaseRoute]] = None,
    tags: Optional[Any] = None,
    servers: Optional[Any] = None,
    terms_of_service: Optional[str] = None,
    contact: Optional[Dict[str, Any]] = None,
    license_info: Optional[Dict[str, Any]] = None,
    separate_input_output_schemas: bool = True,
) -> Any:
    info: Dict[str, Any] = {"title": title, "version": version}
    if summary:
        info["summary"] = summary
    if description:
        info["description"] = description
    if terms_of_service:
        info["termsOfService"] = terms_of_service
    if contact:
        info["contact"] = contact
    if license_info:
        info["license"] = license_info
    output: Dict[str, Any] = {"openapi": openapi_version, "info": info}
    if servers:
        output["servers"] = servers
    components: Dict[str, Any] = {}
    paths: Dict[str, Any] = {}
    webhook_paths: Dict[str, Any] = {}
    operation_ids: Set[str] = set()
    all_fields: List[ModelField] = get_fields_from_routes(list(routes or []) + list(webhooks or []))
    model_name_map: ModelNameMap = get_compat_model_name_map(all_fields)
    schema_generator: GenerateJsonSchema = GenerateJsonSchema(ref_template=REF_TEMPLATE)
    field_mapping, definitions = get_definitions(
        fields=all_fields,
        schema_generator=schema_generator,
        model_name_map=model_name_map,
        separate_input_output_schemas=separate_input_output_schemas,
    )
    for route in routes or []:
        if isinstance(route, routing.APIRoute):
            result = get_openapi_path(
                route=route,
                operation_ids=operation_ids,
                schema_generator=schema_generator,
                model_name_map=model_name_map,
                field_mapping=field_mapping,
                separate_input_output_schemas=separate_input_output_schemas,
            )
            if result:
                path, security_schemes, path_definitions = result
                if path:
                    paths.setdefault(route.path_format, {}).update(path)
                if security_schemes:
                    components.setdefault("securitySchemes", {}).update(security_schemes)
                if path_definitions:
                    definitions.update(path_definitions)
    for webhook in webhooks or []:
        if isinstance(webhook, routing.APIRoute):
            result = get_openapi_path(
                route=webhook,
                operation_ids=operation_ids,
                schema_generator=schema_generator,
                model_name_map=model_name_map,
                field_mapping=field_mapping,
                separate_input_output_schemas=separate_input_output_schemas,
            )
            if result:
                path, security_schemes, path_definitions = result
                if path:
                    webhook_paths.setdefault(webhook.path_format, {}).update(path)
                if security_schemes:
                    components.setdefault("securitySchemes", {}).update(security_schemes)
                if path_definitions:
                    definitions.update(path_definitions)
    if definitions:
        components["schemas"] = {k: definitions[k] for k in sorted(definitions)}
    if components:
        output["components"] = components
    output["paths"] = paths
    if webhook_paths:
        output["webhooks"] = webhook_paths
    if tags:
        output["tags"] = tags
    return jsonable_encoder(OpenAPI(**output), by_alias=True, exclude_none=True)