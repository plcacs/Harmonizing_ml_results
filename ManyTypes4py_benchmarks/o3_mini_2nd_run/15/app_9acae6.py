#!/usr/bin/env python3
"""Chalice app and routing code."""
import re
import sys
import os
import logging
import json
import traceback
import decimal
import base64
import copy
import functools
import datetime
from collections import defaultdict
from urllib.parse import unquote_plus
from collections.abc import Mapping, MutableMapping
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
__version__ = '1.31.4'
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from chalice.local import LambdaContext

_PARAMS = re.compile('{\\w+}')
MiddlewareFuncType = Callable[[Any, Callable[[Any], Any]], Any]
UserHandlerFuncType = Callable[..., Any]
HeadersType = Dict[str, Union[str, List[str]]]
_ANY_STRING = (str, bytes)


def handle_extra_types(obj: Any) -> Any:
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, MultiDict):
        return dict(obj)
    raise TypeError('Object of type %s is not JSON serializable' % obj.__class__.__name__)


def error_response(
    message: str,
    error_code: str,
    http_status_code: int,
    headers: Optional[HeadersType] = None
) -> "Response":
    body = {'Code': error_code, 'Message': message}
    response = Response(body=body, status_code=http_status_code, headers=headers)
    return response


def _matches_content_type(content_type: str, valid_content_types: List[str]) -> bool:
    content_type = content_type.lower()
    valid_content_types = [x.lower() for x in valid_content_types]
    return '*/*' in content_type or '*/*' in valid_content_types or _content_type_header_contains(content_type, valid_content_types)


def _content_type_header_contains(content_type_header: str, valid_content_types: List[str]) -> bool:
    content_type_header_parts = [p.strip() for p in re.split('[,;]', content_type_header)]
    valid_parts = set(valid_content_types).intersection(content_type_header_parts)
    return len(valid_parts) > 0


class ChaliceError(Exception):
    pass


class WebsocketDisconnectedError(ChaliceError):
    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id


class ChaliceViewError(ChaliceError):
    STATUS_CODE: int = 500


class ChaliceUnhandledError(ChaliceError):
    """This error is not caught from a Chalice view function.

    This exception is allowed to propagate from a view function so
    that middleware handlers can process the exception.
    """
    pass


class BadRequestError(ChaliceViewError):
    STATUS_CODE: int = 400


class UnauthorizedError(ChaliceViewError):
    STATUS_CODE: int = 401


class ForbiddenError(ChaliceViewError):
    STATUS_CODE: int = 403


class NotFoundError(ChaliceViewError):
    STATUS_CODE: int = 404


class MethodNotAllowedError(ChaliceViewError):
    STATUS_CODE: int = 405


class RequestTimeoutError(ChaliceViewError):
    STATUS_CODE: int = 408


class ConflictError(ChaliceViewError):
    STATUS_CODE: int = 409


class UnprocessableEntityError(ChaliceViewError):
    STATUS_CODE: int = 422


class TooManyRequestsError(ChaliceViewError):
    STATUS_CODE: int = 429


ALL_ERRORS: List[Type[ChaliceViewError]] = [
    ChaliceViewError,
    BadRequestError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError,
    MethodNotAllowedError,
    RequestTimeoutError,
    ConflictError,
    UnprocessableEntityError,
    TooManyRequestsError,
]


class MultiDict(MutableMapping[str, Any]):
    """A mapping of key to list of values.

    Accessing it in the usual way will return the last value in the list.
    Calling getlist will return a list of all the values associated with
    the same key.
    """

    def __init__(self, mapping: Optional[Dict[str, List[Any]]] = None) -> None:
        if mapping is None:
            mapping = {}
        self._dict: Dict[str, List[Any]] = mapping

    def __getitem__(self, k: str) -> Any:
        try:
            return self._dict[k][-1]
        except IndexError:
            raise KeyError(k)

    def __setitem__(self, k: str, v: Any) -> None:
        self._dict[k] = [v]

    def __delitem__(self, k: str) -> None:
        del self._dict[k]

    def getlist(self, k: str) -> List[Any]:
        return list(self._dict[k])

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __repr__(self) -> str:
        return 'MultiDict(%s)' % self._dict

    def __str__(self) -> str:
        return repr(self)


class CaseInsensitiveMapping(Mapping[str, Any]):
    """Case insensitive and read-only mapping."""

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None) -> None:
        mapping = mapping or {}
        self._dict = {k.lower(): v for k, v in mapping.items()}

    def __getitem__(self, key: str) -> Any:
        return self._dict[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return 'CaseInsensitiveMapping(%s)' % repr(self._dict)


class Authorizer(object):
    name: str = ''
    scopes: List[str] = []

    def to_swagger(self) -> Dict[str, Any]:
        raise NotImplementedError('to_swagger')

    def with_scopes(self, scopes: List[str]) -> "Authorizer":
        raise NotImplementedError('with_scopes')


class IAMAuthorizer(Authorizer):
    _AUTH_TYPE: str = 'aws_iam'

    def __init__(self) -> None:
        self.name = 'sigv4'
        self.scopes = []

    def to_swagger(self) -> Dict[str, Any]:
        return {
            'in': 'header',
            'type': 'apiKey',
            'name': 'Authorization',
            'x-amazon-apigateway-authtype': 'awsSigv4'
        }

    def with_scopes(self, scopes: List[str]) -> "Authorizer":
        raise NotImplementedError('with_scopes')


class CognitoUserPoolAuthorizer(Authorizer):
    _AUTH_TYPE: str = 'cognito_user_pools'

    def __init__(self, name: str, provider_arns: List[str], header: str = 'Authorization', scopes: Optional[List[str]] = None) -> None:
        self.name = name
        self._header = header
        if not isinstance(provider_arns, list):
            raise TypeError('provider_arns should be a list of ARNs, received: %s' % provider_arns)
        self._provider_arns = provider_arns
        self.scopes = scopes or []

    def to_swagger(self) -> Dict[str, Any]:
        return {
            'in': 'header',
            'type': 'apiKey',
            'name': self._header,
            'x-amazon-apigateway-authtype': self._AUTH_TYPE,
            'x-amazon-apigateway-authorizer': {'type': self._AUTH_TYPE, 'providerARNs': self._provider_arns}
        }

    def with_scopes(self, scopes: List[str]) -> "CognitoUserPoolAuthorizer":
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class CustomAuthorizer(Authorizer):
    _AUTH_TYPE: str = 'custom'

    def __init__(
        self,
        name: str,
        authorizer_uri: str,
        ttl_seconds: int = 300,
        header: str = 'Authorization',
        invoke_role_arn: Optional[str] = None,
        scopes: Optional[List[str]] = None
    ) -> None:
        self.name = name
        self._header = header
        self._authorizer_uri = authorizer_uri
        self._ttl_seconds = ttl_seconds
        self._invoke_role_arn = invoke_role_arn
        self.scopes = scopes or []

    def to_swagger(self) -> Dict[str, Any]:
        swagger = {
            'in': 'header',
            'type': 'apiKey',
            'name': self._header,
            'x-amazon-apigateway-authtype': self._AUTH_TYPE,
            'x-amazon-apigateway-authorizer': {
                'type': 'token',
                'authorizerUri': self._authorizer_uri,
                'authorizerResultTtlInSeconds': self._ttl_seconds
            }
        }
        if self._invoke_role_arn is not None:
            swagger['x-amazon-apigateway-authorizer']['authorizerCredentials'] = self._invoke_role_arn
        return swagger

    def with_scopes(self, scopes: List[str]) -> "CustomAuthorizer":
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class CORSConfig(object):
    """A cors configuration to attach to a route."""
    _REQUIRED_HEADERS: List[str] = ['Content-Type', 'X-Amz-Date', 'Authorization', 'X-Api-Key', 'X-Amz-Security-Token']

    def __init__(
        self,
        allow_origin: str = '*',
        allow_headers: Optional[List[str]] = None,
        expose_headers: Optional[List[str]] = None,
        max_age: Optional[int] = None,
        allow_credentials: Optional[bool] = None
    ) -> None:
        self.allow_origin = allow_origin
        if allow_headers is None:
            self._allow_headers = set(self._REQUIRED_HEADERS)
        else:
            self._allow_headers = set(list(allow_headers) + self._REQUIRED_HEADERS)
        if expose_headers is None:
            expose_headers = []
        self._expose_headers = expose_headers
        self._max_age = max_age
        self._allow_credentials = allow_credentials

    @property
    def allow_headers(self) -> str:
        return ','.join(sorted(self._allow_headers))

    def get_access_control_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            'Access-Control-Allow-Origin': self.allow_origin,
            'Access-Control-Allow-Headers': self.allow_headers
        }
        if self._expose_headers:
            headers.update({'Access-Control-Expose-Headers': ','.join(self._expose_headers)})
        if self._max_age is not None:
            headers.update({'Access-Control-Max-Age': str(self._max_age)})
        if self._allow_credentials is True:
            headers.update({'Access-Control-Allow-Credentials': 'true'})
        return headers

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.get_access_control_headers() == other.get_access_control_headers()
        return False


class Request(object):
    """The current request from API gateway."""
    _NON_SERIALIZED_ATTRS: List[str] = ['lambda_context']

    def __init__(self, event_dict: Dict[str, Any], lambda_context: Optional["LambdaContext"] = None) -> None:
        query_params = event_dict['multiValueQueryStringParameters']
        self.query_params: Optional[MultiDict] = None if query_params is None else MultiDict(query_params)
        self.headers = CaseInsensitiveMapping(event_dict['headers'])
        self.uri_params = event_dict['pathParameters']
        self.method = event_dict['requestContext']['httpMethod']
        self._is_base64_encoded = event_dict.get('isBase64Encoded', False)
        self._body = event_dict['body']
        self._json_body = None
        self._raw_body: bytes = b''
        self.context = event_dict['requestContext']
        self.stage_vars = event_dict['stageVariables']
        self.path = event_dict['requestContext']['resourcePath']
        self.lambda_context = lambda_context
        self._event_dict = event_dict

    def _base64decode(self, encoded: Union[bytes, str]) -> bytes:
        if not isinstance(encoded, bytes):
            encoded = encoded.encode('ascii')
        output = base64.b64decode(encoded)
        return output

    @property
    def raw_body(self) -> bytes:
        if not self._raw_body and self._body is not None:
            if self._is_base64_encoded:
                self._raw_body = self._base64decode(self._body)
            elif not isinstance(self._body, bytes):
                self._raw_body = self._body.encode('utf-8')
            else:
                self._raw_body = self._body
        return self._raw_body

    @property
    def json_body(self) -> Any:
        if self.headers.get('content-type', '').startswith('application/json'):
            if self._json_body is None:
                try:
                    self._json_body = json.loads(self.raw_body)
                except ValueError:
                    raise BadRequestError('Error Parsing JSON')
            return self._json_body

    def to_dict(self) -> Dict[str, Any]:
        copied = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in self._NON_SERIALIZED_ATTRS}
        copied['headers'] = dict(copied['headers'])
        if copied['query_params'] is not None:
            copied['query_params'] = dict(copied['query_params'])
        return copied

    def to_original_event(self) -> Dict[str, Any]:
        return self._event_dict


class Response(object):
    def __init__(self, body: Any, headers: Optional[HeadersType] = None, status_code: int = 200) -> None:
        self.body = body
        if headers is None:
            headers = {}
        self.headers = headers
        self.status_code = status_code

    def to_dict(self, binary_types: Optional[List[str]] = None) -> Dict[str, Any]:
        body = self.body
        if not isinstance(body, _ANY_STRING):
            body = json.dumps(body, separators=(',', ':'), default=handle_extra_types)
        single_headers, multi_headers = self._sort_headers(self.headers)
        response: Dict[str, Any] = {
            'headers': single_headers,
            'multiValueHeaders': multi_headers,
            'statusCode': self.status_code,
            'body': body
        }
        if binary_types is not None:
            self._b64encode_body_if_needed(response, binary_types)
        return response

    def _sort_headers(self, all_headers: HeadersType) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        multi_headers: Dict[str, List[str]] = {}
        single_headers: Dict[str, str] = {}
        for name, value in all_headers.items():
            if isinstance(value, list):
                multi_headers[name] = value
            else:
                single_headers[name] = value
        return (single_headers, multi_headers)

    def _b64encode_body_if_needed(self, response_dict: Dict[str, Any], binary_types: List[str]) -> None:
        response_headers = CaseInsensitiveMapping(response_dict['headers'])
        content_type = response_headers.get('content-type', '')
        body = response_dict['body']
        if _matches_content_type(content_type, binary_types):
            if _matches_content_type(content_type, ['application/json']) or not content_type:
                body = body if isinstance(body, bytes) else body.encode('utf-8')
            body = self._base64encode(body)
            response_dict['isBase64Encoded'] = True
        response_dict['body'] = body

    def _base64encode(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise ValueError('Expected bytes type for body with binary Content-Type. Got %s type body instead.' % type(data))
        data = base64.b64encode(data)
        return data.decode('ascii')


class RouteEntry(object):
    def __init__(
        self,
        view_function: UserHandlerFuncType,
        view_name: str,
        path: str,
        method: str,
        api_key_required: Optional[bool] = None,
        content_types: Optional[List[str]] = None,
        cors: Union[bool, CORSConfig] = False,
        authorizer: Optional[Authorizer] = None
    ) -> None:
        self.view_function = view_function
        self.view_name = view_name
        self.uri_pattern = path
        self.method = method
        self.api_key_required = api_key_required
        self.view_args = self._parse_view_args()
        self.content_types = content_types or []
        if cors is True:
            cors = CORSConfig()
        elif cors is False:
            cors = None
        self.cors = cors
        self.authorizer = authorizer

    def _parse_view_args(self) -> List[str]:
        if '{' not in self.uri_pattern:
            return []
        results = [r[1:-1] for r in _PARAMS.findall(self.uri_pattern)]
        return results

    def __eq__(self, other: Any) -> bool:
        return self.__dict__ == other.__dict__


class APIGateway(object):
    _DEFAULT_BINARY_TYPES: List[str] = [
        'application/octet-stream', 'application/x-tar', 'application/zip',
        'audio/basic', 'audio/ogg', 'audio/mp4', 'audio/mpeg', 'audio/wav', 'audio/webm',
        'image/png', 'image/jpg', 'image/jpeg', 'image/gif',
        'video/ogg', 'video/mpeg', 'video/webm'
    ]

    def __init__(self) -> None:
        self.binary_types: List[str] = self.default_binary_types
        self.cors: Union[bool, CORSConfig] = False

    @property
    def default_binary_types(self) -> List[str]:
        return list(self._DEFAULT_BINARY_TYPES)


class WebsocketAPI(object):
    _WEBSOCKET_ENDPOINT_TEMPLATE: str = 'https://{domain_name}/{stage}'
    _REGION_ENV_VARS: List[str] = ['AWS_REGION', 'AWS_DEFAULT_REGION']

    def __init__(self, env: Optional[Dict[str, str]] = None) -> None:
        self.session = None
        self._endpoint: Optional[str] = None
        self._client = None
        if env is None:
            self._env = os.environ
        else:
            self._env = env

    def configure(self, domain_name: str, stage: str) -> None:
        if self._endpoint is not None:
            return
        self._endpoint = self._WEBSOCKET_ENDPOINT_TEMPLATE.format(domain_name=domain_name, stage=stage)

    def configure_from_api_id(self, api_id: str, stage: str) -> None:
        if self._endpoint is not None:
            return
        region_name = self._get_region()
        if region_name.startswith('cn-'):
            domain_name_template = '{api_id}.execute-api.{region}.amazonaws.com.cn'
        else:
            domain_name_template = '{api_id}.execute-api.{region}.amazonaws.com'
        domain_name = domain_name_template.format(api_id=api_id, region=region_name)
        self.configure(domain_name, stage)

    def _get_region(self) -> str:
        for varname in self._REGION_ENV_VARS:
            if varname in self._env:
                return self._env[varname]
        if self.session is not None:
            region_name = self.session.region_name
            if region_name is not None:
                return region_name
        raise ValueError("Unable to retrieve the region name when configuring the websocket client.  Either set the 'AWS_REGION' environment variable or assign 'app.websocket_api.session' to a boto3 session.")

    def _get_client(self) -> Any:
        if self.session is None:
            raise ValueError('Assign app.websocket_api.session to a boto3 session before using the WebsocketAPI')
        if self._endpoint is None:
            raise ValueError('WebsocketAPI.configure must be called before using the WebsocketAPI')
        if self._client is None:
            self._client = self.session.client('apigatewaymanagementapi', endpoint_url=self._endpoint)
        return self._client

    def send(self, connection_id: str, message: Any) -> None:
        client = self._get_client()
        try:
            client.post_to_connection(ConnectionId=connection_id, Data=message)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)

    def close(self, connection_id: str) -> None:
        client = self._get_client()
        try:
            client.delete_connection(ConnectionId=connection_id)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)

    def info(self, connection_id: str) -> Any:
        client = self._get_client()
        try:
            return client.get_connection(ConnectionId=connection_id)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)


class DecoratorAPI(object):
    websocket_api: Optional[WebsocketAPI] = None

    def middleware(self, event_type: str = 'all') -> Callable[[UserHandlerFuncType], UserHandlerFuncType]:
        def _middleware_wrapper(func: UserHandlerFuncType) -> UserHandlerFuncType:
            self.register_middleware(func, event_type)
            return func
        return _middleware_wrapper

    def authorizer(self, ttl_seconds: Optional[int] = None, execution_role: Optional[str] = None, name: Optional[str] = None, header: str = 'Authorization') -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='authorizer',
            name=name,
            registration_kwargs={'ttl_seconds': ttl_seconds, 'execution_role': execution_role, 'header': header}
        )

    def on_s3_event(self, bucket: str, events: Optional[List[str]] = None, prefix: Optional[str] = None, suffix: Optional[str] = None, name: Optional[str] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_s3_event',
            name=name,
            registration_kwargs={'bucket': bucket, 'events': events, 'prefix': prefix, 'suffix': suffix}
        )

    def on_sns_message(self, topic: str, name: Optional[str] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_sns_message',
            name=name,
            registration_kwargs={'topic': topic}
        )

    def on_sqs_message(self, queue: Optional[str] = None, batch_size: int = 1, name: Optional[str] = None, queue_arn: Optional[str] = None, maximum_batching_window_in_seconds: int = 0, maximum_concurrency: Optional[int] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_sqs_message',
            name=name,
            registration_kwargs={
                'queue': queue,
                'queue_arn': queue_arn,
                'batch_size': batch_size,
                'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds,
                'maximum_concurrency': maximum_concurrency
            }
        )

    def on_cw_event(self, event_pattern: Any, name: Optional[str] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_cw_event',
            name=name,
            registration_kwargs={'event_pattern': event_pattern}
        )

    def schedule(self, expression: str, name: Optional[str] = None, description: str = '') -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='schedule',
            name=name,
            registration_kwargs={'expression': expression, 'description': description}
        )

    def on_kinesis_record(self, stream: str, batch_size: int = 100, starting_position: str = 'LATEST', name: Optional[str] = None, maximum_batching_window_in_seconds: int = 0) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_kinesis_record',
            name=name,
            registration_kwargs={
                'stream': stream,
                'batch_size': batch_size,
                'starting_position': starting_position,
                'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds
            }
        )

    def on_dynamodb_record(self, stream_arn: str, batch_size: int = 100, starting_position: str = 'LATEST', name: Optional[str] = None, maximum_batching_window_in_seconds: int = 0) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_dynamodb_record',
            name=name,
            registration_kwargs={
                'stream_arn': stream_arn,
                'batch_size': batch_size,
                'starting_position': starting_position,
                'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds
            }
        )

    def route(self, path: str, **kwargs: Any) -> Callable[[UserHandlerFuncType], Any]:
        name = kwargs.pop('name', None)
        return self._create_registration_function(
            handler_type='route',
            name=name,
            registration_kwargs={'path': path, 'kwargs': kwargs}
        )

    def lambda_function(self, name: Optional[str] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='lambda_function',
            name=name
        )

    def on_ws_connect(self, name: Optional[str] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_ws_connect',
            name=name,
            registration_kwargs={'route_key': '$connect'}
        )

    def on_ws_disconnect(self, name: Optional[str] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_ws_disconnect',
            name=name,
            registration_kwargs={'route_key': '$disconnect'}
        )

    def on_ws_message(self, name: Optional[str] = None) -> Callable[[UserHandlerFuncType], Any]:
        return self._create_registration_function(
            handler_type='on_ws_message',
            name=name,
            registration_kwargs={'route_key': '$default'}
        )

    def _create_registration_function(self, handler_type: str, name: Optional[str] = None, registration_kwargs: Optional[Dict[str, Any]] = None) -> Callable[[UserHandlerFuncType], Any]:
        def _register_handler(user_handler: UserHandlerFuncType) -> Any:
            handler_name = name
            if handler_name is None:
                handler_name = user_handler.__name__
            kwargs_inner: Dict[str, Any] = registration_kwargs if registration_kwargs is not None else {}
            wrapped = self._wrap_handler(handler_type, handler_name, user_handler)
            self._register_handler(handler_type, handler_name, user_handler, wrapped, kwargs_inner)
            return wrapped
        return _register_handler

    def _wrap_handler(self, handler_type: str, handler_name: str, user_handler: UserHandlerFuncType) -> Any:
        if handler_type in _EVENT_CLASSES:
            if handler_type == 'lambda_function':
                user_handler = PureLambdaWrapper(user_handler)
            return EventSourceHandler(user_handler, _EVENT_CLASSES[handler_type], middleware_handlers=list(self._get_middleware_handlers(_MIDDLEWARE_MAPPING[handler_type])))
        websocket_event_classes = ['on_ws_connect', 'on_ws_message', 'on_ws_disconnect']
        if self.websocket_api and handler_type in websocket_event_classes:
            return WebsocketEventSourceHandler(user_handler, WebsocketEvent, self.websocket_api, middleware_handlers=list(self._get_middleware_handlers('websocket')))
        if handler_type == 'authorizer':
            return ChaliceAuthorizer(handler_name, user_handler)
        return user_handler

    def _get_middleware_handlers(self, event_type: str) -> Iterable[Callable]:
        raise NotImplementedError('_get_middleware_handlers')

    def _register_handler(self, handler_type: str, name: Optional[str], user_handler: UserHandlerFuncType, wrapped_handler: Any, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError('_register_handler')

    def register_middleware(self, func: Callable[..., Any], event_type: str = 'all') -> None:
        raise NotImplementedError('register_middleware')


class _HandlerRegistration(object):
    def __init__(self) -> None:
        self.routes: Dict[str, Dict[str, RouteEntry]] = defaultdict(dict)
        self.websocket_handlers: Dict[str, Any] = {}
        self.builtin_auth_handlers: List[Any] = []
        self.event_sources: List[Any] = []
        self.pure_lambda_functions: List[Any] = []
        self.api = APIGateway()
        self.handler_map: Dict[str, Any] = {}
        self.middleware_handlers: List[Tuple[Callable, str]] = []

    def register_middleware(self, func: Callable[..., Any], event_type: str = 'all') -> None:
        self.middleware_handlers.append((func, event_type))

    def _do_register_handler(self, handler_type: str, name: str, user_handler: UserHandlerFuncType, wrapped_handler: Any, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        module_name = 'app'
        if options is not None:
            name_prefix = options.get('name_prefix')
            if name_prefix is not None:
                name = name_prefix + name
            url_prefix = options.get('url_prefix')
            if url_prefix is not None and handler_type == 'route':
                kwargs['url_prefix'] = url_prefix
            module_name = options['module_name']
        handler_string = '%s.%s' % (module_name, user_handler.__name__)
        getattr(self, '_register_%s' % handler_type)(name=name, user_handler=user_handler, handler_string=handler_string, kwargs=kwargs)
        self.handler_map[name] = wrapped_handler

    def _attach_websocket_handler(self, handler: Any) -> None:
        route_key = handler.route_key_handled
        decorator_name = {'$default': 'on_ws_message', '$connect': 'on_ws_connect', '$disconnect': 'on_ws_disconnect'}.get(route_key)
        if route_key in self.websocket_handlers:
            raise ValueError("Duplicate websocket handler: '%s'. There can only be one handler for each websocket decorator." % decorator_name)
        self.websocket_handlers[route_key] = handler

    def _register_on_ws_connect(self, *, name: str, user_handler: UserHandlerFuncType, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        wrapper = WebsocketConnectConfig(name=name, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)

    def _register_on_ws_message(self, *, name: str, user_handler: UserHandlerFuncType, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        route_key = kwargs['route_key']
        wrapper = WebsocketMessageConfig(name=name, route_key_handled=route_key, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)
        self.websocket_handlers[route_key] = wrapper

    def _register_on_ws_disconnect(self, *, name: str, user_handler: UserHandlerFuncType, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        wrapper = WebsocketDisconnectConfig(name=name, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)

    def _register_lambda_function(self, *, name: str, user_handler: UserHandlerFuncType, handler_string: str, **unused: Any) -> None:
        wrapper = LambdaFunction(func=user_handler, name=name, handler_string=handler_string)
        self.pure_lambda_functions.append(wrapper)

    def _register_on_s3_event(self, *, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        events = kwargs['events']
        if events is None:
            events = ['s3:ObjectCreated:*']
        s3_event = S3EventConfig(name=name, bucket=kwargs['bucket'], events=events, prefix=kwargs['prefix'], suffix=kwargs['suffix'], handler_string=handler_string)
        self.event_sources.append(s3_event)

    def _register_on_sns_message(self, *, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        sns_config = SNSEventConfig(name=name, handler_string=handler_string, topic=kwargs['topic'])
        self.event_sources.append(sns_config)

    def _register_on_sqs_message(self, *, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        queue = kwargs.get('queue')
        queue_arn = kwargs.get('queue_arn')
        if not queue and (not queue_arn):
            raise ValueError('Must provide either `queue` or `queue_arn` to the `on_sqs_message` decorator.')
        sqs_config = SQSEventConfig(
            name=name, handler_string=handler_string,
            queue=queue, queue_arn=queue_arn,
            batch_size=kwargs['batch_size'],
            maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds'],
            maximum_concurrency=kwargs['maximum_concurrency']
        )
        self.event_sources.append(sqs_config)

    def _register_on_kinesis_record(self, *, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        kinesis_config = KinesisEventConfig(
            name=name, handler_string=handler_string,
            stream=kwargs['stream'], batch_size=kwargs['batch_size'],
            starting_position=kwargs['starting_position'],
            maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds']
        )
        self.event_sources.append(kinesis_config)

    def _register_on_dynamodb_record(self, *, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        ddb_config = DynamoDBEventConfig(
            name=name, handler_string=handler_string,
            stream_arn=kwargs['stream_arn'], batch_size=kwargs['batch_size'],
            starting_position=kwargs['starting_position'],
            maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds']
        )
        self.event_sources.append(ddb_config)

    def _register_on_cw_event(self, *, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        event_source = CloudWatchEventConfig(name=name, event_pattern=kwargs['event_pattern'], handler_string=handler_string)
        self.event_sources.append(event_source)

    def _register_schedule(self, *, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        event_source = ScheduledEventConfig(name=name, schedule_expression=kwargs['expression'], description=kwargs['description'], handler_string=handler_string)
        self.event_sources.append(event_source)

    def _register_authorizer(self, *, name: str, handler_string: str, wrapped_handler: Any, kwargs: Dict[str, Any], **unused: Any) -> None:
        actual_kwargs = kwargs.copy()
        ttl_seconds = actual_kwargs.pop('ttl_seconds', None)
        execution_role = actual_kwargs.pop('execution_role', None)
        header = actual_kwargs.pop('header', None)
        if actual_kwargs:
            raise TypeError('TypeError: authorizer() got unexpected keyword arguments: %s' % ', '.join(list(actual_kwargs)))
        auth_config = BuiltinAuthConfig(name=name, handler_string=handler_string, ttl_seconds=ttl_seconds, execution_role=execution_role, header=header)
        wrapped_handler.config = auth_config
        self.builtin_auth_handlers.append(auth_config)

    def _register_route(self, *, name: str, user_handler: UserHandlerFuncType, kwargs: Dict[str, Any], **unused: Any) -> None:
        actual_kwargs = kwargs['kwargs']
        path = kwargs['path']
        url_prefix = kwargs.pop('url_prefix', None)
        if url_prefix is not None:
            path = '/'.join([url_prefix.rstrip('/'), path.strip('/')]).rstrip('/')
        methods = actual_kwargs.pop('methods', ['GET'])
        route_kwargs = {
            'authorizer': actual_kwargs.pop('authorizer', None),
            'api_key_required': actual_kwargs.pop('api_key_required', None),
            'content_types': actual_kwargs.pop('content_types', ['application/json']),
            'cors': actual_kwargs.pop('cors', self.api.cors)
        }
        if route_kwargs['cors'] is None:
            route_kwargs['cors'] = self.api.cors
        if not isinstance(route_kwargs['content_types'], list):
            raise ValueError('In view function "%s", the content_types value must be a list, not %s: %s' % (name, type(route_kwargs['content_types']), route_kwargs['content_types']))
        if actual_kwargs:
            raise TypeError('TypeError: route() got unexpected keyword arguments: %s' % ', '.join(list(actual_kwargs)))
        for method in methods:
            if method in self.routes[path]:
                raise ValueError('Duplicate method: \'%s\' detected for route: \'%s\'\nbetween view functions: "%s" and "%s". A specific method may only be specified once for a particular path.' % (method, path, self.routes[path][method].view_name, name))
            entry = RouteEntry(user_handler, name, path, method, **route_kwargs)
            self.routes[path][method] = entry


class Chalice(_HandlerRegistration, DecoratorAPI):
    FORMAT_STRING: str = '%(name)s - %(levelname)s - %(message)s'

    def __init__(self, app_name: str, debug: bool = False, configure_logs: bool = True, env: Optional[Dict[str, str]] = None) -> None:
        super(Chalice, self).__init__()
        self.app_name = app_name
        self.websocket_api = WebsocketAPI()
        self._debug = debug
        self.configure_logs = configure_logs
        self.log = logging.getLogger(self.app_name)
        if env is None:
            env = os.environ
        self._initialize(env)
        self.experimental_feature_flags: Set[str] = set()
        self._features_used: Set[str] = set()

    def _initialize(self, env: Dict[str, str]) -> None:
        if self.configure_logs:
            self._configure_logging()
        env['AWS_EXECUTION_ENV'] = '%s aws-chalice/%s' % (env.get('AWS_EXECUTION_ENV', 'AWS_Lambda'), __version__)

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value
        self._configure_log_level()

    def _configure_logging(self) -> None:
        if self._already_configured(self.log):
            return
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(self.FORMAT_STRING)
        handler.setFormatter(formatter)
        self.log.propagate = False
        self._configure_log_level()
        self.log.addHandler(handler)

    def _already_configured(self, log: logging.Logger) -> bool:
        if not log.handlers:
            return False
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream == sys.stdout:
                    return True
        return False

    def _configure_log_level(self) -> None:
        if self._debug:
            level = logging.DEBUG
        else:
            level = logging.ERROR
        self.log.setLevel(level)

    def register_blueprint(self, blueprint: "Blueprint", name_prefix: Optional[str] = None, url_prefix: Optional[str] = None) -> None:
        blueprint.register(self, options={'name_prefix': name_prefix, 'url_prefix': url_prefix})

    def _register_handler(self, handler_type: str, name: str, user_handler: UserHandlerFuncType, wrapped_handler: Any, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        self._do_register_handler(handler_type, name, user_handler, wrapped_handler, kwargs, options)

    def _register_on_ws_connect(self, *, name: str, user_handler: UserHandlerFuncType, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_on_ws_connect(name=name, user_handler=user_handler, handler_string=handler_string, kwargs=kwargs, **unused)

    def _register_on_ws_message(self, *, name: str, user_handler: UserHandlerFuncType, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_on_ws_message(name=name, user_handler=user_handler, handler_string=handler_string, kwargs=kwargs, **unused)

    def _register_on_ws_disconnect(self, *, name: str, user_handler: UserHandlerFuncType, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_on_ws_disconnect(name=name, user_handler=user_handler, handler_string=handler_string, kwargs=kwargs, **unused)

    def _get_middleware_handlers(self, event_type: str) -> Iterable[Callable]:
        return (func for func, filter_type in self.middleware_handlers if filter_type in [event_type, 'all'])

    def __call__(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        self.lambda_context = context
        handler = RestAPIEventHandler(self.routes, self.api, self.log, self.debug, middleware_handlers=list(self._get_middleware_handlers('http')))
        self.current_request = handler.create_request_object(event, context)
        return handler(event, context)


class BuiltinAuthConfig(object):
    def __init__(self, name: str, handler_string: str, ttl_seconds: Optional[int] = None, execution_role: Optional[str] = None, header: str = 'Authorization') -> None:
        self.name = name
        self.handler_string = handler_string
        self.ttl_seconds = ttl_seconds
        self.execution_role = execution_role
        self.header = header


class ChaliceAuthorizer(object):
    def __init__(self, name: str, func: Callable, scopes: Optional[List[str]] = None) -> None:
        self.name = name
        self.func = func
        self.scopes = scopes or []
        self.config: Optional[BuiltinAuthConfig] = None

    def __call__(self, event: Dict[str, Any], context: Any) -> Any:
        auth_request = self._transform_event(event)
        result = self.func(auth_request)
        if isinstance(result, AuthResponse):
            return result.to_dict(auth_request)
        return result

    def _transform_event(self, event: Dict[str, Any]) -> "AuthRequest":
        return AuthRequest(event['type'], event['authorizationToken'], event['methodArn'])

    def with_scopes(self, scopes: List[str]) -> "ChaliceAuthorizer":
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class AuthRequest(object):
    def __init__(self, auth_type: str, token: str, method_arn: str) -> None:
        self.auth_type = auth_type
        self.token = token
        self.method_arn = method_arn


class AuthResponse(object):
    ALL_HTTP_METHODS: List[str] = ['DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'POST', 'PUT', 'GET']

    def __init__(self, routes: List[Union['AuthRoute', str]], principal_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.routes = routes
        self.principal_id = principal_id
        if context is None:
            context = {}
        self.context = context

    def to_dict(self, request: AuthRequest) -> Dict[str, Any]:
        return {
            'context': self.context,
            'principalId': self.principal_id,
            'policyDocument': self._generate_policy(request)
        }

    def _generate_policy(self, request: AuthRequest) -> Dict[str, Any]:
        allowed_resources = self._generate_allowed_resources(request)
        return {
            'Version': '2012-10-17',
            'Statement': [{
                'Action': 'execute-api:Invoke',
                'Effect': 'Allow',
                'Resource': allowed_resources
            }]
        }

    def _generate_allowed_resources(self, request: AuthRequest) -> List[str]:
        allowed_resources: List[str] = []
        for route in self.routes:
            if isinstance(route, AuthRoute):
                methods = route.methods
                path = route.path
            elif route == '*':
                methods = ['*']
                path = '*'
            else:
                methods = ['*']
                path = route
            for method in methods:
                allowed_resources.append(self._generate_arn(path, request, method))
        return allowed_resources

    def _generate_arn(self, route: Union['AuthRoute', str], request: AuthRequest, method: str = '*') -> str:
        incoming_arn = request.method_arn
        arn_parts = incoming_arn.split(':', 5)
        allowed_resource = arn_parts[-1].split('/')[:2]
        allowed_resource.extend([method, route[1:]])
        last_arn_segment = '/'.join(allowed_resource)
        if route == '*':
            last_arn_segment += route
        arn_parts[-1] = last_arn_segment
        final_arn = ':'.join(arn_parts)
        return final_arn


class AuthRoute(object):
    def __init__(self, path: str, methods: List[str]) -> None:
        self.path = path
        self.methods = methods


class LambdaFunction(object):
    def __init__(self, func: Callable, name: str, handler_string: str) -> None:
        self.func = func
        self.name = name
        self.handler_string = handler_string

    def __call__(self, event: Any, context: Any) -> Any:
        return self.func(event, context)


class BaseEventSourceConfig(object):
    def __init__(self, name: str, handler_string: str) -> None:
        self.name = name
        self.handler_string = handler_string


class ScheduledEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, schedule_expression: str, description: str) -> None:
        super(ScheduledEventConfig, self).__init__(name, handler_string)
        self.schedule_expression = schedule_expression
        self.description = description


class CloudWatchEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, event_pattern: Any) -> None:
        super(CloudWatchEventConfig, self).__init__(name, handler_string)
        self.event_pattern = event_pattern


class ScheduleExpression(object):
    def to_string(self) -> str:
        raise NotImplementedError('to_string')


class Rate(ScheduleExpression):
    MINUTES: str = 'MINUTES'
    HOURS: str = 'HOURS'
    DAYS: str = 'DAYS'

    def __init__(self, value: int, unit: str) -> None:
        self.value = value
        self.unit = unit

    def to_string(self) -> str:
        unit = self.unit.lower()
        if self.value == 1:
            unit = unit[:-1]
        return 'rate(%s %s)' % (self.value, unit)


class Cron(ScheduleExpression):
    def __init__(self, minutes: str, hours: str, day_of_month: str, month: str, day_of_week: str, year: str) -> None:
        self.minutes = minutes
        self.hours = hours
        self.day_of_month = day_of_month
        self.month = month
        self.day_of_week = day_of_week
        self.year = year

    def to_string(self) -> str:
        return 'cron(%s %s %s %s %s %s)' % (self.minutes, self.hours, self.day_of_month, self.month, self.day_of_week, self.year)


class S3EventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, bucket: str, events: List[str], prefix: Optional[str], suffix: Optional[str], handler_string: str) -> None:
        super(S3EventConfig, self).__init__(name, handler_string)
        self.bucket = bucket
        self.events = events
        self.prefix = prefix
        self.suffix = suffix


class SNSEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, topic: str) -> None:
        super(SNSEventConfig, self).__init__(name, handler_string)
        self.topic = topic


class SQSEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, queue: Optional[str], queue_arn: Optional[str], batch_size: int, maximum_batching_window_in_seconds: int, maximum_concurrency: Optional[int]) -> None:
        super(SQSEventConfig, self).__init__(name, handler_string)
        self.queue = queue
        self.queue_arn = queue_arn
        self.batch_size = batch_size
        self.maximum_batching_window_in_seconds = maximum_batching_window_in_seconds
        self.maximum_concurrency = maximum_concurrency


class KinesisEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, stream: str, batch_size: int, starting_position: str, maximum_batching_window_in_seconds: int) -> None:
        super(KinesisEventConfig, self).__init__(name, handler_string)
        self.stream = stream
        self.batch_size = batch_size
        self.starting_position = starting_position
        self.maximum_batching_window_in_seconds = maximum_batching_window_in_seconds


class DynamoDBEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, stream_arn: str, batch_size: int, starting_position: str, maximum_batching_window_in_seconds: int) -> None:
        super(DynamoDBEventConfig, self).__init__(name, handler_string)
        self.stream_arn = stream_arn
        self.batch_size = batch_size
        self.starting_position = starting_position
        self.maximum_batching_window_in_seconds = maximum_batching_window_in_seconds


class WebsocketConnectConfig(BaseEventSourceConfig):
    CONNECT_ROUTE: str = '$connect'

    def __init__(self, name: str, handler_string: str, user_handler: Callable) -> None:
        super(WebsocketConnectConfig, self).__init__(name, handler_string)
        self.route_key_handled: str = self.CONNECT_ROUTE
        self.handler_function = user_handler


class WebsocketMessageConfig(BaseEventSourceConfig):
    def __init__(self, name: str, route_key_handled: str, handler_string: str, user_handler: Callable) -> None:
        super(WebsocketMessageConfig, self).__init__(name, handler_string)
        self.route_key_handled = route_key_handled
        self.handler_function = user_handler


class WebsocketDisconnectConfig(BaseEventSourceConfig):
    DISCONNECT_ROUTE: str = '$disconnect'

    def __init__(self, name: str, handler_string: str, user_handler: Callable) -> None:
        super(WebsocketDisconnectConfig, self).__init__(name, handler_string)
        self.route_key_handled = self.DISCONNECT_ROUTE
        self.handler_function = user_handler


class PureLambdaWrapper(object):
    def __init__(self, original_func: Callable) -> None:
        self._original_func = original_func

    def __call__(self, event: Any) -> Any:
        return self._original_func(event.to_dict(), event.context)


class MiddlewareHandler(object):
    def __init__(self, handler: Callable, next_handler: Callable) -> None:
        self.handler = handler
        self.next_handler = next_handler

    def __call__(self, request: Any) -> Any:
        return self.handler(request, self.next_handler)


class BaseLambdaHandler(object):
    def __call__(self, event: Any, context: Any) -> Any:
        pass

    def _build_middleware_handlers(self, handlers: List[Callable], original_handler: Callable) -> Callable:
        current = original_handler
        for handler in reversed(list(handlers)):
            current = MiddlewareHandler(handler=handler, next_handler=current)
        return current


class EventSourceHandler(BaseLambdaHandler):
    def __init__(self, func: Callable, event_class: Callable[[Dict[str, Any], Any], Any], middleware_handlers: Optional[List[Callable]] = None) -> None:
        self.func = func
        self.event_class = event_class
        if middleware_handlers is None:
            middleware_handlers = []
        self._middleware_handlers = middleware_handlers
        self.handler: Optional[Callable] = None

    @property
    def middleware_handlers(self) -> List[Callable]:
        return self._middleware_handlers

    @middleware_handlers.setter
    def middleware_handlers(self, value: List[Callable]) -> None:
        self._middleware_handlers = value

    def __call__(self, event: Dict[str, Any], context: Any) -> Any:
        event_obj = self.event_class(event, context)
        if self.handler is None:
            self.handler = self._build_middleware_handlers(self._middleware_handlers, original_handler=self.func)
        return self.handler(event_obj)


class WebsocketEventSourceHandler(EventSourceHandler):
    WEBSOCKET_API_RESPONSE: Dict[str, Any] = {'statusCode': 200}

    def __init__(self, func: Callable, event_class: Callable[[Dict[str, Any], Any], Any], websocket_api: WebsocketAPI, middleware_handlers: Optional[List[Callable]] = None) -> None:
        super(WebsocketEventSourceHandler, self).__init__(func, event_class, middleware_handlers)
        self.websocket_api = websocket_api

    def __call__(self, event: Dict[str, Any], context: Any) -> Any:
        self.websocket_api.configure_from_api_id(event['requestContext']['apiId'], event['requestContext']['stage'])
        response = super(WebsocketEventSourceHandler, self).__call__(event, context)
        data = None
        if isinstance(response, Response):
            data = response.to_dict()
        elif isinstance(response, dict):
            data = response
            if 'statusCode' not in data:
                data = {**self.WEBSOCKET_API_RESPONSE, **data}
        return data or self.WEBSOCKET_API_RESPONSE


class RestAPIEventHandler(BaseLambdaHandler):
    def __init__(self, route_table: Dict[str, Dict[str, RouteEntry]], api: APIGateway, log: logging.Logger, debug: bool, middleware_handlers: Optional[Iterable[Callable]] = None) -> None:
        self.routes = route_table
        self.api = api
        self.log = log
        self.debug = debug
        self.current_request: Optional[Request] = None
        self.lambda_context: Any = None
        if middleware_handlers is None:
            middleware_handlers = []
        self._middleware_handlers = middleware_handlers

    def _global_error_handler(self, event: Any, get_response: Callable[[Any], Any]) -> Any:
        try:
            return get_response(event)
        except Exception:
            return self._unhandled_exception_to_response()

    def create_request_object(self, event: Dict[str, Any], context: Any) -> Optional[Request]:
        resource_path = event.get('requestContext', {}).get('resourcePath')
        if resource_path is not None:
            self.current_request = Request(event, context)
            return self.current_request
        return None

    def __call__(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        def wrapped_event(request: Any) -> Any:
            return self._main_rest_api_handler(event, context)
        final_handler = self._build_middleware_handlers([self._global_error_handler] + list(self._middleware_handlers), original_handler=wrapped_event)
        response = final_handler(self.current_request)
        return response.to_dict(self.api.binary_types)

    def _main_rest_api_handler(self, event: Dict[str, Any], context: Any) -> Response:
        resource_path = event.get('requestContext', {}).get('resourcePath')
        if resource_path is None:
            return error_response(error_code='InternalServerError', message='Unknown request.', http_status_code=500)
        http_method = event['requestContext']['httpMethod']
        if http_method not in self.routes[resource_path]:
            allowed_methods = ', '.join(self.routes[resource_path].keys())
            return error_response(error_code='MethodNotAllowedError', message='Unsupported method: %s' % http_method, http_status_code=405, headers={'Allow': allowed_methods})
        route_entry = self.routes[resource_path][http_method]
        view_function = route_entry.view_function
        function_args = {name: event['pathParameters'][name] for name in route_entry.view_args}
        self.lambda_context = context
        cors_headers: Optional[Dict[str, str]] = None
        if self._cors_enabled_for_route(route_entry):
            cors_headers = self._get_cors_headers(route_entry.cors)
        if self.current_request and route_entry.content_types:
            content_type = self.current_request.headers.get('content-type', 'application/json')
            if not _matches_content_type(content_type, route_entry.content_types):
                return error_response(error_code='UnsupportedMediaType', message='Unsupported media type: %s' % content_type, http_status_code=415, headers=cors_headers)
        response = self._get_view_function_response(view_function, function_args)
        if cors_headers is not None:
            self._add_cors_headers(response, cors_headers)
        response_headers = CaseInsensitiveMapping(response.headers)
        if self.current_request and (not self._validate_binary_response(self.current_request.headers, response_headers)):
            content_type = response_headers.get('content-type', '')
            return error_response(error_code='BadRequest', message='Request did not specify an Accept header with %s, The response has a Content-Type of %s. If a response has a binary Content-Type then the request must specify an Accept header that matches.' % (content_type, content_type), http_status_code=400, headers=cors_headers)
        return response

    def _validate_binary_response(self, request_headers: CaseInsensitiveMapping, response_headers: CaseInsensitiveMapping) -> bool:
        request_accept_header = request_headers.get('accept')
        response_content_type = response_headers.get('content-type', 'application/json')
        response_is_binary = _matches_content_type(response_content_type, self.api.binary_types)
        expects_binary_response = False
        if request_accept_header is not None:
            expects_binary_response = _matches_content_type(request_accept_header, self.api.binary_types)
        if response_is_binary and (not expects_binary_response):
            return False
        return True

    def _get_view_function_response(self, view_function: Callable, function_args: Dict[str, Any]) -> Response:
        try:
            response = view_function(**function_args)
            if not isinstance(response, Response):
                response = Response(body=response)
            self._validate_response(response)
        except ChaliceUnhandledError:
            raise
        except ChaliceViewError as e:
            response = Response(body={'Code': e.__class__.__name__, 'Message': str(e)}, status_code=e.STATUS_CODE)
        except Exception:
            response = self._unhandled_exception_to_response()
        return response

    def _unhandled_exception_to_response(self) -> Response:
        headers: Dict[str, str] = {}
        path = getattr(self.current_request, 'path', 'unknown')
        self.log.error('Caught exception for path %s', path, exc_info=True)
        if self.debug:
            stack_trace = ''.join(traceback.format_exc())
            body = stack_trace
            headers['Content-Type'] = 'text/plain'
        else:
            body = {'Code': 'InternalServerError', 'Message': 'An internal server error occurred.'}
        response = Response(body=body, headers=headers, status_code=500)
        return response

    def _validate_response(self, response: Response) -> None:
        for header, value in response.headers.items():
            if '\n' in value:
                raise ChaliceError("Bad value for header '%s': %r" % (header, value))

    def _cors_enabled_for_route(self, route_entry: RouteEntry) -> bool:
        return route_entry.cors is not None

    def _get_cors_headers(self, cors: CORSConfig) -> Dict[str, str]:
        return cors.get_access_control_headers()

    def _add_cors_headers(self, response: Response, cors_headers: Dict[str, str]) -> None:
        for name, value in cors_headers.items():
            if name not in response.headers:
                response.headers[name] = value


class BaseLambdaEvent(object):
    def __init__(self, event_dict: Dict[str, Any], context: Any) -> None:
        self._event_dict = event_dict
        self.context = context
        self._extract_attributes(event_dict)

    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        raise NotImplementedError('_extract_attributes')

    def to_dict(self) -> Dict[str, Any]:
        return self._event_dict


class LambdaFunctionEvent(BaseLambdaEvent):
    def __init__(self, event_dict: Dict[str, Any], context: Any) -> None:
        self.event = event_dict
        self.context = context

    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return self.event


class CloudWatchEvent(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        self.version = event_dict['version']
        self.account = event_dict['account']
        self.region = event_dict['region']
        self.detail = event_dict['detail']
        self.detail_type = event_dict['detail-type']
        self.source = event_dict['source']
        self.time = event_dict['time']
        self.event_id = event_dict['id']
        self.resources = event_dict['resources']


class WebsocketEvent(BaseLambdaEvent):
    def __init__(self, event_dict: Dict[str, Any], context: Any) -> None:
        super(WebsocketEvent, self).__init__(event_dict, context)
        self._json_body = None

    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        request_context = event_dict['requestContext']
        self.domain_name = request_context['domainName']
        self.stage = request_context['stage']
        self.connection_id = request_context['connectionId']
        self.body = str(event_dict.get('body'))

    @property
    def json_body(self) -> Any:
        if self._json_body is None:
            try:
                self._json_body = json.loads(self.body)
            except ValueError:
                raise BadRequestError('Error Parsing JSON')
        return self._json_body


class SNSEvent(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        first_record = event_dict['Records'][0]
        self.message = first_record['Sns']['Message']
        self.subject = first_record['Sns']['Subject']
        self.message_attributes = first_record['Sns']['MessageAttributes']


class S3Event(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        s3 = event_dict['Records'][0]['s3']
        self.bucket = s3['bucket']['name']
        self.key = unquote_plus(s3['object']['key'])


class SQSEvent(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        pass

    def __iter__(self) -> Iterator["SQSRecord"]:
        for record in self._event_dict['Records']:
            yield SQSRecord(record, self.context)


class SQSRecord(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        self.body = event_dict['body']
        self.receipt_handle = event_dict['receiptHandle']


class KinesisEvent(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        pass

    def __iter__(self) -> Iterator["KinesisRecord"]:
        for record in self._event_dict['Records']:
            yield KinesisRecord(record, self.context)


class KinesisRecord(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        kinesis = event_dict['kinesis']
        encoded_payload = kinesis['data']
        self.data = base64.b64decode(encoded_payload)
        self.sequence_number = kinesis['sequenceNumber']
        self.partition_key = kinesis['partitionKey']
        self.schema_version = kinesis['kinesisSchemaVersion']
        self.timestamp = datetime.datetime.utcfromtimestamp(kinesis['approximateArrivalTimestamp'])


class DynamoDBEvent(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        pass

    def __iter__(self) -> Iterator["DynamoDBRecord"]:
        for record in self._event_dict['Records']:
            yield DynamoDBRecord(record, self.context)


class DynamoDBRecord(BaseLambdaEvent):
    def _extract_attributes(self, event_dict: Dict[str, Any]) -> None:
        dynamodb = event_dict['dynamodb']
        self.timestamp = datetime.datetime.utcfromtimestamp(dynamodb['ApproximateCreationDateTime'])
        self.keys = dynamodb.get('Keys')
        self.new_image = dynamodb.get('NewImage')
        self.old_image = dynamodb.get('OldImage')
        self.sequence_number = dynamodb['SequenceNumber']
        self.size_bytes = dynamodb['SizeBytes']
        self.stream_view_type = dynamodb['StreamViewType']
        self.aws_region = event_dict['awsRegion']
        self.event_id = event_dict['eventID']
        self.event_name = event_dict['eventName']
        self.event_source_arn = event_dict['eventSourceARN']

    @property
    def table_name(self) -> str:
        parts = self.event_source_arn.split(':', 5)
        if not len(parts) == 6:
            return ''
        full_name = parts[-1]
        name_parts = full_name.split('/')
        if len(name_parts) >= 2:
            return name_parts[1]
        return ''


class Blueprint(DecoratorAPI):
    def __init__(self, import_name: str) -> None:
        self._import_name = import_name
        self._deferred_registrations: List[Callable[[Chalice, Dict[str, Any]], None]] = []
        self._current_app: Optional[Chalice] = None
        self._lambda_context: Any = None

    @property
    def log(self) -> logging.Logger:
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.log if it's registered to an app.")
        return self._current_app.log

    @property
    def current_request(self) -> Request:
        if self._current_app is None or self._current_app.current_request is None:
            raise RuntimeError("Can only access Blueprint.current_request if it's registered to an app.")
        return self._current_app.current_request

    @property
    def current_app(self) -> Chalice:
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.current_app if it's registered to an app.")
        return self._current_app

    @property
    def lambda_context(self) -> Any:
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.lambda_context if it's registered to an app.")
        return self._current_app.lambda_context

    def register(self, app: Chalice, options: Dict[str, Any]) -> None:
        self._current_app = app
        all_options = options.copy()
        all_options['module_name'] = self._import_name
        for function in self._deferred_registrations:
            function(app, all_options)

    def register_middleware(self, func: Callable[..., Any], event_type: str = 'all') -> None:
        self._deferred_registrations.append(lambda app, options: app.register_middleware(func, event_type))

    def _register_handler(self, handler_type: str, name: str, user_handler: Callable, wrapped_handler: Callable, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        def _register_blueprint_handler(app: Chalice, options: Dict[str, Any]) -> None:
            if handler_type in _EVENT_CLASSES:
                wrapped_handler.middleware_handlers = list(app._get_middleware_handlers(_MIDDLEWARE_MAPPING[handler_type]))
            app._register_handler(handler_type, name, user_handler, wrapped_handler, kwargs, options)
        self._deferred_registrations.append(_register_blueprint_handler)

    def _get_middleware_handlers(self, event_type: str) -> List[Any]:
        return []


class ConvertToMiddleware(object):
    def __init__(self, lambda_wrapper: Callable) -> None:
        self._wrapper = lambda_wrapper

    def __call__(self, event: Any, get_response: Callable[[Any], Any]) -> Any:
        original_event, context = self._extract_original_param(event)

        @functools.wraps(self._wrapper)
        def wrapped(original_event: Dict[str, Any], context: Any) -> Any:
            return get_response(event)
        return self._wrapper(wrapped)(original_event, context)

    def _extract_original_param(self, event: Any) -> Tuple[Dict[str, Any], Any]:
        if isinstance(event, Request):
            return (event.to_original_event(), event.lambda_context)
        return (event.to_dict(), event.context)


_EVENT_CLASSES: Dict[str, Any] = {
    'on_s3_event': S3Event,
    'on_sns_message': SNSEvent,
    'on_sqs_message': SQSEvent,
    'on_cw_event': CloudWatchEvent,
    'on_kinesis_record': KinesisEvent,
    'on_dynamodb_record': DynamoDBEvent,
    'schedule': CloudWatchEvent,
    'lambda_function': LambdaFunctionEvent
}
_MIDDLEWARE_MAPPING: Dict[str, str] = {
    'on_s3_event': 's3',
    'on_sns_message': 'sns',
    'on_sqs_message': 'sqs',
    'on_cw_event': 'cloudwatch',
    'on_kinesis_record': 'kinesis',
    'on_dynamodb_record': 'dynamodb',
    'schedule': 'scheduled',
    'lambda_function': 'pure_lambda'
}