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
__version__ = '1.31.4'
from typing import List, Dict, Any, Optional, Sequence, Union, Callable, Set, Iterator, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from chalice.local import LambdaContext

_Params: re.Pattern = re.compile(r'{\w+}')
MiddlewareFuncType = Callable[[Any, Callable[[Any], Any]], Any]
UserHandlerFuncType = Callable[..., Any]
HeadersType = Dict[str, Union[str, List[str]]]
_ANY_STRING = (str, bytes)


def func_7v8ddinv(obj: Any) -> Any:
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, MultiDict):
        return dict(obj)
    raise TypeError('Object of type %s is not JSON serializable' % obj.__class__.__name__)


def func_jbgc6yig(message: str, error_code: str, http_status_code: int, headers: Optional[HeadersType] = None) -> "Response":
    body: Dict[str, Any] = {'Code': error_code, 'Message': message}
    response: Response = Response(body=body, status_code=http_status_code, headers=headers)
    return response


def func_cgh0i6nf(content_type: str, valid_content_types: List[str]) -> bool:
    content_type = content_type.lower()
    valid_content_types = [x.lower() for x in valid_content_types]
    return ('*/*' in content_type or '*/*' in valid_content_types or
            _content_type_header_contains(content_type, valid_content_types))


def _content_type_header_contains(content_type: str, valid_content_types: List[str]) -> bool:
    # A simple implementation assuming valid_content_types are substrings to check.
    for valid in valid_content_types:
        if valid in content_type:
            return True
    return False


def func_8i8y7mih(content_type_header: str, valid_content_types: List[str]) -> bool:
    content_type_header_parts: List[str] = [p.strip() for p in re.split('[,;]', content_type_header)]
    valid_parts: set = set(valid_content_types).intersection(content_type_header_parts)
    return len(valid_parts) > 0


class ChaliceError(Exception):
    pass


class WebsocketDisconnectedError(ChaliceError):
    def __init__(self, connection_id: Any) -> None:
        self.connection_id = connection_id


class ChaliceViewError(ChaliceError):
    STATUS_CODE: int = 500


class ChaliceUnhandledError(ChaliceError):
    """This error is not caught from a Chalice view function.
    
    This exception is allowed to propagate from a view function so
    that middleware handlers can process the exception.
    """
    

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


ALL_ERRORS: List[type] = [ChaliceViewError, BadRequestError, NotFoundError,
                            UnauthorizedError, ForbiddenError, MethodNotAllowedError,
                            RequestTimeoutError, ConflictError, UnprocessableEntityError,
                            TooManyRequestsError]


class MultiDict(MutableMapping):
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

    def func_1nkp63ev(self, k: str) -> List[Any]:
        return list(self._dict[k])

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __repr__(self) -> str:
        return 'MultiDict(%s)' % self._dict

    def __str__(self) -> str:
        return repr(self)


class CaseInsensitiveMapping(Mapping):
    """Case insensitive and read-only mapping."""

    def __init__(self, mapping: Optional[Dict[str, Any]]) -> None:
        mapping = mapping or {}
        self._dict: Dict[str, Any] = {k.lower(): v for k, v in mapping.items()}

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

    def func_5aepmyjj(self) -> Dict[str, Any]:
        raise NotImplementedError('to_swagger')

    def func_gv635333(self, scopes: List[str]) -> "Authorizer":
        raise NotImplementedError('with_scopes')


class IAMAuthorizer(Authorizer):
    _AUTH_TYPE: str = 'aws_iam'

    def __init__(self) -> None:
        self.name = 'sigv4'
        self.scopes = []

    def func_5aepmyjj(self) -> Dict[str, Any]:
        return {'in': 'header', 'type': 'apiKey', 'name': 'Authorization',
                'x-amazon-apigateway-authtype': 'awsSigv4'}

    def func_gv635333(self, scopes: List[str]) -> "Authorizer":
        raise NotImplementedError('with_scopes')


class CognitoUserPoolAuthorizer(Authorizer):
    _AUTH_TYPE: str = 'cognito_user_pools'

    def __init__(self, name: str, provider_arns: List[str], header: str = 'Authorization', scopes: Optional[List[str]] = None) -> None:
        self.name = name
        self._header = header
        if not isinstance(provider_arns, list):
            raise TypeError('provider_arns should be a list of ARNs, received: %s' % provider_arns)
        self._provider_arns: List[str] = provider_arns
        self.scopes = scopes or []

    def func_5aepmyjj(self) -> Dict[str, Any]:
        return {'in': 'header', 'type': 'apiKey', 'name': self._header,
                'x-amazon-apigateway-authtype': self._AUTH_TYPE,
                'x-amazon-apigateway-authorizer': {'type': self._AUTH_TYPE,
                                                   'providerARNs': self._provider_arns}}

    def func_gv635333(self, scopes: List[str]) -> "Authorizer":
        authorizer_with_scopes: CognitoUserPoolAuthorizer = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class CustomAuthorizer(Authorizer):
    _AUTH_TYPE: str = 'custom'

    def __init__(self, name: str, authorizer_uri: str, ttl_seconds: int = 300, header: str = 'Authorization', invoke_role_arn: Optional[str] = None, scopes: Optional[List[str]] = None) -> None:
        self.name = name
        self._header = header
        self._authorizer_uri = authorizer_uri
        self._ttl_seconds = ttl_seconds
        self._invoke_role_arn = invoke_role_arn
        self.scopes = scopes or []

    def func_5aepmyjj(self) -> Dict[str, Any]:
        swagger: Dict[str, Any] = {'in': 'header', 'type': 'apiKey', 'name': self._header,
                                   'x-amazon-apigateway-authtype': self._AUTH_TYPE,
                                   'x-amazon-apigateway-authorizer': {'type': 'token',
                                                                      'authorizerUri': self._authorizer_uri,
                                                                      'authorizerResultTtlInSeconds': self._ttl_seconds}}
        if self._invoke_role_arn is not None:
            swagger['x-amazon-apigateway-authorizer']['authorizerCredentials'] = self._invoke_role_arn
        return swagger

    def func_gv635333(self, scopes: List[str]) -> "Authorizer":
        authorizer_with_scopes: CustomAuthorizer = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class CORSConfig(object):
    """A cors configuration to attach to a route."""
    _REQUIRED_HEADERS: List[str] = ['Content-Type', 'X-Amz-Date', 'Authorization',
                                    'X-Api-Key', 'X-Amz-Security-Token']

    def __init__(self, allow_origin: str = '*', allow_headers: Optional[List[str]] = None, expose_headers: Optional[List[str]] = None, max_age: Optional[int] = None, allow_credentials: Optional[bool] = None) -> None:
        self.allow_origin: str = allow_origin
        if allow_headers is None:
            self._allow_headers: Set[str] = set(self._REQUIRED_HEADERS)
        else:
            self._allow_headers = set(list(allow_headers) + self._REQUIRED_HEADERS)
        if expose_headers is None:
            expose_headers = []
        self._expose_headers: List[str] = expose_headers
        self._max_age: Optional[int] = max_age
        self._allow_credentials: Optional[bool] = allow_credentials

    @property
    def func_6rluov3y(self) -> str:
        return ','.join(sorted(self._allow_headers))

    def func_nuiz31or(self) -> Dict[str, str]:
        headers: Dict[str, str] = {'Access-Control-Allow-Origin': self.allow_origin,
                                   'Access-Control-Allow-Headers': self.func_6rluov3y}
        if self._expose_headers:
            headers.update({'Access-Control-Expose-Headers': ','.join(self._expose_headers)})
        if self._max_age is not None:
            headers.update({'Access-Control-Max-Age': str(self._max_age)})
        if self._allow_credentials is True:
            headers.update({'Access-Control-Allow-Credentials': 'true'})
        return headers

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.get_access_control_headers() == other.get_access_control_headers()  # type: ignore
        return False

    def get_access_control_headers(self) -> Dict[str, str]:
        return self.func_nuiz31or()


class Request(object):
    """The current request from API gateway."""
    _NON_SERIALIZED_ATTRS: List[str] = ['lambda_context']

    def __init__(self, event_dict: Dict[str, Any], lambda_context: Optional[Any] = None) -> None:
        query_params: Optional[Dict[str, List[str]]] = event_dict.get('multiValueQueryStringParameters')
        self.query_params: Optional[MultiDict] = None if query_params is None else MultiDict(query_params)
        self.headers: CaseInsensitiveMapping = CaseInsensitiveMapping(event_dict.get('headers', {}))
        self.uri_params: Optional[Dict[str, Any]] = event_dict.get('pathParameters')
        self.method: str = event_dict.get('requestContext', {}).get('httpMethod', '')
        self._is_base64_encoded: bool = event_dict.get('isBase64Encoded', False)
        self._body: Optional[Any] = event_dict.get('body')
        self._json_body: Optional[Any] = None
        self._raw_body: bytes = b''
        self.context: Dict[str, Any] = event_dict.get('requestContext', {})
        self.stage_vars: Optional[Dict[str, Any]] = event_dict.get('stageVariables')
        self.path: str = event_dict.get('requestContext', {}).get('resourcePath', '')
        self.lambda_context: Optional[Any] = lambda_context
        self._event_dict: Dict[str, Any] = event_dict

    def func_jkxty10u(self, encoded: Union[str, bytes]) -> bytes:
        if not isinstance(encoded, bytes):
            encoded = encoded.encode('ascii')
        output: bytes = base64.b64decode(encoded)
        return output

    @property
    def func_xku2lr0w(self) -> bytes:
        if not self._raw_body and self._body is not None:
            if self._is_base64_encoded:
                self._raw_body = self.func_jkxty10u(self._body)
            elif not isinstance(self._body, bytes):
                self._raw_body = self._body.encode('utf-8')  # type: ignore
            else:
                self._raw_body = self._body  # type: ignore
        return self._raw_body

    @property
    def func_o1nc530n(self) -> Any:
        if self.headers.get('content-type', '').startswith('application/json'):
            if self._json_body is None:
                try:
                    self._json_body = json.loads(self.func_xku2lr0w.decode('utf-8'))
                except ValueError:
                    raise BadRequestError('Error Parsing JSON')
            return self._json_body

    def func_f1xsvgls(self) -> Dict[str, Any]:
        copied: Dict[str, Any] = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in self._NON_SERIALIZED_ATTRS}
        copied['headers'] = dict(copied['headers'])
        if copied.get('query_params') is not None:
            copied['query_params'] = dict(copied['query_params'])
        return copied

    def func_7tj7556g(self) -> Dict[str, Any]:
        return self._event_dict


class Response(object):
    def __init__(self, body: Any, headers: Optional[HeadersType] = None, status_code: int = 200) -> None:
        self.body: Any = body
        if headers is None:
            headers = {}
        self.headers: HeadersType = headers
        self.status_code: int = status_code

    def func_f1xsvgls(self, binary_types: Optional[List[str]] = None) -> Dict[str, Any]:
        body: Any = self.body
        if not isinstance(body, _ANY_STRING):
            body = json.dumps(body, separators=(',', ':'), default=func_7v8ddinv)
        single_headers, multi_headers = self.func_ky8ik2q0(self.headers)
        response: Dict[str, Any] = {'headers': single_headers, 'multiValueHeaders': multi_headers, 'statusCode': self.status_code, 'body': body}
        if binary_types is not None:
            self.func_36yxykc4(response, binary_types)
        return response

    def func_ky8ik2q0(self, all_headers: HeadersType) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        multi_headers: Dict[str, List[str]] = {}
        single_headers: Dict[str, str] = {}
        for name, value in all_headers.items():
            if isinstance(value, list):
                multi_headers[name] = value
            else:
                single_headers[name] = value
        return single_headers, multi_headers

    def func_36yxykc4(self, response_dict: Dict[str, Any], binary_types: List[str]) -> None:
        response_headers: CaseInsensitiveMapping = CaseInsensitiveMapping(response_dict.get('headers', {}))
        content_type: str = response_headers.get('content-type', '')
        body: Any = response_dict['body']
        if func_cgh0i6nf(content_type, binary_types):
            if func_cgh0i6nf(content_type, ['application/json']) or not content_type:
                body = body if isinstance(body, bytes) else body.encode('utf-8')
            body = self._base64encode(body)
            response_dict['isBase64Encoded'] = True
        response_dict['body'] = body

    def _base64encode(self, data: bytes) -> str:
        return base64.b64encode(data).decode('ascii')

    def func_4fy9phjy(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise ValueError('Expected bytes type for body with binary Content-Type. Got %s type body instead.' % type(data))
        data = base64.b64encode(data)
        return data.decode('ascii')


class RouteEntry(object):
    def __init__(self, view_function: UserHandlerFuncType, view_name: str, path: str, method: str, api_key_required: Optional[bool] = None, content_types: Optional[List[str]] = None, cors: Union[bool, CORSConfig] = False, authorizer: Optional[Any] = None) -> None:
        self.view_function: UserHandlerFuncType = view_function
        self.view_name: str = view_name
        self.uri_pattern: str = path
        self.method: str = method
        self.api_key_required: Optional[bool] = api_key_required
        self.view_args: List[str] = self._parse_view_args()
        self.content_types: List[str] = content_types if content_types is not None else []
        if cors is True:
            cors = CORSConfig()
        elif cors is False:
            cors = None
        self.cors: Optional[CORSConfig] = cors
        self.authorizer: Optional[Any] = authorizer

    def _parse_view_args(self) -> List[str]:
        return self.func_mklydfdc()

    def func_mklydfdc(self) -> List[str]:
        if '{' not in self.uri_pattern:
            return []
        results: List[str] = [r[1:-1] for r in _Params.findall(self.uri_pattern)]
        return results

    def __eq__(self, other: Any) -> bool:
        return self.__dict__ == other.__dict__


class APIGateway(object):
    _DEFAULT_BINARY_TYPES: List[str] = ['application/octet-stream',
                                        'application/x-tar', 'application/zip', 'audio/basic', 'audio/ogg',
                                        'audio/mp4', 'audio/mpeg', 'audio/wav', 'audio/webm', 'image/png',
                                        'image/jpg', 'image/jpeg', 'image/gif', 'video/ogg', 'video/mpeg',
                                        'video/webm']

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
        self.session: Optional[Any] = None
        self._endpoint: Optional[str] = None
        self._client: Optional[Any] = None
        if env is None:
            self._env: Dict[str, str] = os.environ
        else:
            self._env = env

    def func_ygcy8a2t(self, domain_name: str, stage: str) -> None:
        if self._endpoint is not None:
            return
        self._endpoint = self._WEBSOCKET_ENDPOINT_TEMPLATE.format(domain_name=domain_name, stage=stage)

    def func_wum7cakw(self, api_id: str, stage: str) -> None:
        if self._endpoint is not None:
            return
        region_name: str = self.func_lwroba3i()
        if region_name.startswith('cn-'):
            domain_name_template: str = '{api_id}.execute-api.{region}.amazonaws.com.cn'
        else:
            domain_name_template: str = '{api_id}.execute-api.{region}.amazonaws.com'
        domain_name: str = domain_name_template.format(api_id=api_id, region=region_name)
        self.configure(domain_name, stage)

    def configure(self, domain_name: str, stage: str) -> None:
        self._endpoint = self._WEBSOCKET_ENDPOINT_TEMPLATE.format(domain_name=domain_name, stage=stage)

    def func_lwroba3i(self) -> str:
        for varname in self._REGION_ENV_VARS:
            if varname in self._env:
                return self._env[varname]
        if self.session is not None:
            region_name: Optional[str] = self.session.region_name
            if region_name is not None:
                return region_name
        raise ValueError("Unable to retrieve the region name when configuring the websocket client.  Either set the 'AWS_REGION' environment variable or assign 'app.websocket_api.session' to a boto3 session.")

    def func_cd1u87kg(self) -> Any:
        if self.session is None:
            raise ValueError('Assign app.websocket_api.session to a boto3 session before using the WebsocketAPI')
        if self._endpoint is None:
            raise ValueError('WebsocketAPI.configure must be called before using the WebsocketAPI')
        if self._client is None:
            self._client = self.session.client('apigatewaymanagementapi', endpoint_url=self._endpoint)
        return self._client

    def func_t33jlkmk(self, connection_id: str, message: Union[str, bytes]) -> None:
        client: Any = self.func_cd1u87kg()
        try:
            client.post_to_connection(ConnectionId=connection_id, Data=message)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)

    def func_q2iec03k(self, connection_id: str) -> None:
        client: Any = self.func_cd1u87kg()
        try:
            client.delete_connection(ConnectionId=connection_id)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)

    def func_di4rt02f(self, connection_id: str) -> Any:
        client: Any = self.func_cd1u87kg()
        try:
            return client.get_connection(ConnectionId=connection_id)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)


class DecoratorAPI(object):
    websocket_api: Optional[WebsocketAPI] = None

    def func_v0mcmhir(self, event_type: str = 'all') -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _middleware_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register_middleware(func, event_type)
            return func
        return _middleware_wrapper

    def register_middleware(self, func: Callable[..., Any], event_type: str = 'all') -> None:
        raise NotImplementedError("register_middleware must be implemented by subclasses.")

    def func_k51ag724(self, ttl_seconds: Optional[int] = None, execution_role: Optional[str] = None, name: Optional[str] = None, header: str = 'Authorization') -> Callable:
        return self._create_registration_function(handler_type='authorizer',
                                                    name=name,
                                                    registration_kwargs={'ttl_seconds': ttl_seconds,
                                                                         'execution_role': execution_role,
                                                                         'header': header})

    def func_94u5685w(self, bucket: str, events: Optional[List[str]] = None, prefix: Optional[str] = None, suffix: Optional[str] = None, name: Optional[str] = None) -> Callable:
        return self._create_registration_function(handler_type='on_s3_event',
                                                    name=name,
                                                    registration_kwargs={'bucket': bucket,
                                                                         'events': events,
                                                                         'prefix': prefix,
                                                                         'suffix': suffix})

    def func_m5a0gxbu(self, topic: str, name: Optional[str] = None) -> Callable:
        return self._create_registration_function(handler_type='on_sns_message',
                                                    name=name,
                                                    registration_kwargs={'topic': topic})

    def func_9n5xc4jh(self, queue: Optional[str] = None, batch_size: int = 1, name: Optional[str] = None, queue_arn: Optional[str] = None, maximum_batching_window_in_seconds: int = 0, maximum_concurrency: Optional[int] = None) -> Callable:
        return self._create_registration_function(handler_type='on_sqs_message',
                                                    name=name,
                                                    registration_kwargs={'queue': queue,
                                                                         'queue_arn': queue_arn,
                                                                         'batch_size': batch_size,
                                                                         'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds,
                                                                         'maximum_concurrency': maximum_concurrency})

    def func_5o8xm957(self, event_pattern: Any, name: Optional[str] = None) -> Callable:
        return self._create_registration_function(handler_type='on_cw_event',
                                                    name=name,
                                                    registration_kwargs={'event_pattern': event_pattern})

    def func_z7ioyv1f(self, expression: Any, name: Optional[str] = None, description: str = '') -> Callable:
        return self._create_registration_function(handler_type='schedule',
                                                    name=name,
                                                    registration_kwargs={'expression': expression,
                                                                         'description': description})

    def func_fyu89xe4(self, stream: str, batch_size: int = 100, starting_position: str = 'LATEST', name: Optional[str] = None, maximum_batching_window_in_seconds: int = 0) -> Callable:
        return self._create_registration_function(handler_type='on_kinesis_record',
                                                    name=name,
                                                    registration_kwargs={'stream': stream,
                                                                         'batch_size': batch_size,
                                                                         'starting_position': starting_position,
                                                                         'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds})

    def func_p5wzreal(self, stream_arn: str, batch_size: int = 100, starting_position: str = 'LATEST', name: Optional[str] = None, maximum_batching_window_in_seconds: int = 0) -> Callable:
        return self._create_registration_function(handler_type='on_dynamodb_record',
                                                    name=name,
                                                    registration_kwargs={'stream_arn': stream_arn,
                                                                         'batch_size': batch_size,
                                                                         'starting_position': starting_position,
                                                                         'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds})

    def func_de9i4ols(self, path: str, **kwargs: Any) -> Callable:
        return self._create_registration_function(handler_type='route',
                                                    name=kwargs.pop('name', None),
                                                    registration_kwargs={'path': path, 'kwargs': kwargs})

    def func_3dg1vmn5(self, name: Optional[str] = None) -> Callable:
        return self._create_registration_function(handler_type='lambda_function', name=name)

    def func_4zhmzseu(self, name: Optional[str] = None) -> Callable:
        return self._create_registration_function(handler_type='on_ws_connect',
                                                    name=name,
                                                    registration_kwargs={'route_key': '$connect'})

    def func_jpb5qi5s(self, name: Optional[str] = None) -> Callable:
        return self._create_registration_function(handler_type='on_ws_disconnect',
                                                    name=name,
                                                    registration_kwargs={'route_key': '$disconnect'})

    def func_hwymwjgp(self, name: Optional[str] = None) -> Callable:
        return self._create_registration_function(handler_type='on_ws_message',
                                                    name=name,
                                                    registration_kwargs={'route_key': '$default'})

    def _create_registration_function(self, handler_type: str, name: Optional[str], registration_kwargs: Dict[str, Any]) -> Callable:
        def _register_handler(user_handler: Callable) -> Any:
            handler_name: str = name if name is not None else user_handler.__name__
            wrapped = self._wrap_handler(handler_type, handler_name, user_handler)
            self._register_handler(handler_type, handler_name, user_handler, wrapped, registration_kwargs)
            return wrapped
        return _register_handler

    def _wrap_handler(self, handler_type: str, handler_name: str, user_handler: Callable) -> Callable:
        # Placeholder for wrapping logic.
        return user_handler

    def _register_handler(self, handler_type: str, handler_name: str, user_handler: Callable, wrapped_handler: Callable, kwargs: Dict[str, Any]) -> None:
        raise NotImplementedError('_register_handler')


class _HandlerRegistration(object):

    def __init__(self) -> None:
        self.routes: Dict[str, Dict[str, RouteEntry]] = defaultdict(dict)
        self.websocket_handlers: Dict[str, Any] = {}
        self.builtin_auth_handlers: List[Any] = []
        self.event_sources: List[Any] = []
        self.pure_lambda_functions: List[Any] = []
        self.api: APIGateway = APIGateway()
        self.handler_map: Dict[str, Any] = {}
        self.middleware_handlers: List[Tuple[Callable, str]] = []

    def func_460y1b75(self, func: Callable, event_type: str = 'all') -> None:
        self.middleware_handlers.append((func, event_type))

    def func_kmma5k71(self, handler_type: str, name: str, user_handler: Callable, wrapped_handler: Callable, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        module_name: str = 'app'
        if options is not None:
            name_prefix: Optional[str] = options.get('name_prefix')
            if name_prefix is not None:
                name = name_prefix + name
            url_prefix: Optional[str] = options.get('url_prefix')
            if url_prefix is not None and handler_type == 'route':
                kwargs['url_prefix'] = url_prefix
            module_name = options['module_name']
        handler_string: str = '%s.%s' % (module_name, user_handler.__name__)
        getattr(self, '_register_%s' % handler_type)(name=name,
                                                       user_handler=user_handler,
                                                       handler_string=handler_string,
                                                       wrapped_handler=wrapped_handler,
                                                       kwargs=kwargs)
        self.handler_map[name] = wrapped_handler

    def func_7jtgy86z(self, handler: Any) -> None:
        route_key: str = handler.route_key_handled
        decorator_name: Optional[str] = {'$default': 'on_ws_message', '$connect': 'on_ws_connect', '$disconnect': 'on_ws_disconnect'}.get(route_key)
        if route_key in self.websocket_handlers:
            raise ValueError("Duplicate websocket handler: '%s'. There can only be one handler for each websocket decorator." % decorator_name)
        self.websocket_handlers[route_key] = handler

    def func_lx0u0s3m(self, name: str, user_handler: Callable, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        wrapper = WebsocketConnectConfig(name=name, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)

    def func_cpn5mr35(self, name: str, user_handler: Callable, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        route_key: str = kwargs['route_key']
        wrapper = WebsocketMessageConfig(name=name, route_key_handled=route_key, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)
        self.websocket_handlers[route_key] = wrapper

    def func_d70z3jua(self, name: str, user_handler: Callable, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        wrapper = WebsocketDisconnectConfig(name=name, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)

    def func_xrr0343w(self, name: str, user_handler: Callable, handler_string: str, **unused: Any) -> None:
        wrapper = LambdaFunction(func=user_handler, name=name, handler_string=handler_string)
        self.pure_lambda_functions.append(wrapper)

    def func_0858ha25(self, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        events: Optional[List[str]] = kwargs.get('events')
        if events is None:
            events = ['s3:ObjectCreated:*']
        s3_event = S3EventConfig(name=name, bucket=kwargs['bucket'], events=events, prefix=kwargs.get('prefix'), suffix=kwargs.get('suffix'), handler_string=handler_string)
        self.event_sources.append(s3_event)

    def func_tluu7j3j(self, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        sns_config = SNSEventConfig(name=name, handler_string=handler_string, topic=kwargs['topic'])
        self.event_sources.append(sns_config)

    def func_zzaifary(self, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        queue: Optional[str] = kwargs.get('queue')
        queue_arn: Optional[str] = kwargs.get('queue_arn')
        if not queue and not queue_arn:
            raise ValueError('Must provide either `queue` or `queue_arn` to the `on_sqs_message` decorator.')
        sqs_config = SQSEventConfig(name=name, handler_string=handler_string, queue=queue, queue_arn=queue_arn, batch_size=kwargs['batch_size'], maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds'], maximum_concurrency=kwargs.get('maximum_concurrency'))
        self.event_sources.append(sqs_config)

    def func_0hith5tm(self, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        kinesis_config = KinesisEventConfig(name=name, handler_string=handler_string, stream=kwargs['stream'], batch_size=kwargs['batch_size'], starting_position=kwargs['starting_position'], maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds'])
        self.event_sources.append(kinesis_config)

    def func_676tqwln(self, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        ddb_config = DynamoDBEventConfig(name=name, handler_string=handler_string, stream_arn=kwargs['stream_arn'], batch_size=kwargs['batch_size'], starting_position=kwargs['starting_position'], maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds'])
        self.event_sources.append(ddb_config)

    def func_jvt68hwp(self, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        event_source = CloudWatchEventConfig(name=name, event_pattern=kwargs['event_pattern'], handler_string=handler_string)
        self.event_sources.append(event_source)

    def func_p56f22t1(self, name: str, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        event_source = ScheduledEventConfig(name=name, schedule_expression=kwargs['expression'], description=kwargs['description'], handler_string=handler_string)
        self.event_sources.append(event_source)

    def func_smrf0wyr(self, name: str, wrapped_handler: Callable, kwargs: Dict[str, Any], **unused: Any) -> None:
        actual_kwargs: Dict[str, Any] = kwargs.copy()
        ttl_seconds: Optional[int] = actual_kwargs.pop('ttl_seconds', None)
        execution_role: Optional[str] = actual_kwargs.pop('execution_role', None)
        header: Optional[str] = actual_kwargs.pop('header', None)
        if actual_kwargs:
            raise TypeError('TypeError: authorizer() got unexpected keyword arguments: %s' % ', '.join(list(actual_kwargs)))
        auth_config = BuiltinAuthConfig(name=name, handler_string=wrapped_handler.__name__, ttl_seconds=ttl_seconds, execution_role=execution_role, header=header if header is not None else 'Authorization')
        wrapped_handler.config = auth_config  # type: ignore
        self.builtin_auth_handlers.append(auth_config)

    def func_shx6j18z(self, name: str, user_handler: Callable, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        actual_kwargs: Dict[str, Any] = kwargs['kwargs']
        path: str = kwargs['path']
        url_prefix: Optional[str] = kwargs.pop('url_prefix', None)
        if url_prefix is not None:
            path = '/'.join([url_prefix.rstrip('/'), path.strip('/')]).rstrip('/')
        methods: List[str] = actual_kwargs.pop('methods', ['GET'])
        route_kwargs: Dict[str, Any] = {'authorizer': actual_kwargs.pop('authorizer', None),
                                        'api_key_required': actual_kwargs.pop('api_key_required', None),
                                        'content_types': actual_kwargs.pop('content_types', ['application/json']),
                                        'cors': actual_kwargs.pop('cors', self.api.cors)}
        if route_kwargs['cors'] is None:
            route_kwargs['cors'] = self.api.cors
        if not isinstance(route_kwargs['content_types'], list):
            raise ValueError('In view function "%s", the content_types value must be a list, not %s: %s' % (name, type(route_kwargs['content_types']), route_kwargs['content_types']))
        if actual_kwargs:
            raise TypeError('TypeError: route() got unexpected keyword arguments: %s' % ', '.join(list(actual_kwargs)))
        for method in methods:
            if method in self.routes[path]:
                raise ValueError("""Duplicate method: '%s' detected for route: '%s'
between view functions: "%s" and "%s". A specific method may only be specified once for a particular path."""
                                 % (method, path, self.routes[path][method].view_name, name))
            entry: RouteEntry = RouteEntry(user_handler, name, path, method, **route_kwargs)
            self.routes[path][method] = entry

# Placeholder implementations for websocket handler attachment.
    def _attach_websocket_handler(self, handler: Any) -> None:
        # Implementation detail for attaching websocket handler.
        pass


class Chalice(_HandlerRegistration, DecoratorAPI):
    FORMAT_STRING: str = '%(name)s - %(levelname)s - %(message)s'

    def __init__(self, app_name: str, debug: bool = False, configure_logs: bool = True, env: Optional[Dict[str, str]] = None) -> None:
        super(Chalice, self).__init__()
        self.app_name: str = app_name
        self.websocket_api: WebsocketAPI = WebsocketAPI()
        self._debug: bool = debug
        self.configure_logs: bool = configure_logs
        self.log: logging.Logger = logging.getLogger(self.app_name)
        if env is None:
            env = os.environ
        self.func_u9x3vnkw(env)
        self.experimental_feature_flags: Set[str] = set()
        self._features_used: Set[str] = set()

    def func_u9x3vnkw(self, env: Dict[str, str]) -> None:
        if self.configure_logs:
            self._configure_logging()
        env['AWS_EXECUTION_ENV'] = '%s aws-chalice/%s' % (env.get('AWS_EXECUTION_ENV', 'AWS_Lambda'), __version__)

    def _configure_logging(self) -> None:
        self.func_1jeksm2w()

    @property
    def func_su0z65dv(self) -> bool:
        return self._debug

    @func_su0z65dv.setter
    def func_su0z65dv(self, value: bool) -> None:
        self._debug = value
        self._configure_log_level()

    def func_1jeksm2w(self) -> None:
        if self.func_enqqwzme(self.log):
            return
        handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        formatter: logging.Formatter = logging.Formatter(self.FORMAT_STRING)
        handler.setFormatter(formatter)
        self.log.propagate = False
        self._configure_log_level()
        self.log.addHandler(handler)

    def func_enqqwzme(self, log: logging.Logger) -> bool:
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

    def func_d5ka3h39(self, blueprint: "Blueprint", name_prefix: Optional[str] = None, url_prefix: Optional[str] = None) -> None:
        blueprint.func_jvov7e85(self, options={'name_prefix': name_prefix, 'url_prefix': url_prefix, 'module_name': blueprint._import_name})

    def _do_register_handler(self, handler_type: str, name: str, user_handler: Callable, wrapped_handler: Callable, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        # Delegate registration to parent's implementation.
        self._register_handler(handler_type, name, user_handler, wrapped_handler, kwargs, options)

    def func_d2tyxsl5(self, handler_type: str, name: str, user_handler: Callable, wrapped_handler: Callable, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        self._do_register_handler(handler_type, name, user_handler, wrapped_handler, kwargs, options)

    def _register_handler(self, handler_type: str, name: str, user_handler: Callable, wrapped_handler: Callable, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        # Placeholder for actual registration logic.
        pass

    def func_lx0u0s3m(self, name: str, user_handler: Callable, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_handler('on_ws_connect', name, user_handler, user_handler, kwargs, **unused)

    def func_cpn5mr35(self, name: str, user_handler: Callable, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_handler('on_ws_message', name, user_handler, user_handler, kwargs, **unused)

    def func_d70z3jua(self, name: str, user_handler: Callable, handler_string: str, kwargs: Dict[str, Any], **unused: Any) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_handler('on_ws_disconnect', name, user_handler, user_handler, kwargs, **unused)

    def _get_middleware_handlers(self, event_type: str) -> List[Callable]:
        return [func for func, filter_type in self.middleware_handlers if filter_type in [event_type, 'all']]

    def func_eeikzeqs(self, event_type: str) -> List[Callable]:
        return self._get_middleware_handlers(event_type)

    def __call__(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        self.lambda_context = context
        handler = RestAPIEventHandler(self.routes, self.api, self.log, self._debug, middleware_handlers=self._get_middleware_handlers('http'))
        self.current_request = handler.create_request_object(event, context)  # type: ignore
        response: Response = handler(event, context)
        return response.func_f1xsvgls(self.api.binary_types)


class BuiltinAuthConfig(object):
    def __init__(self, name: str, handler_string: str, ttl_seconds: Optional[int] = None, execution_role: Optional[str] = None, header: str = 'Authorization') -> None:
        self.name: str = name
        self.handler_string: str = handler_string
        self.ttl_seconds: Optional[int] = ttl_seconds
        self.execution_role: Optional[str] = execution_role
        self.header: str = header


class ChaliceAuthorizer(object):
    def __init__(self, name: str, func: Callable, scopes: Optional[List[str]] = None) -> None:
        self.name: str = name
        self.func: Callable = func
        self.scopes: List[str] = scopes or []
        self.config: Optional[BuiltinAuthConfig] = None

    def __call__(self, event: Dict[str, Any], context: Any) -> Any:
        auth_request: AuthRequest = self.func_qyiniaas(event)
        result: Any = self.func(auth_request)
        if isinstance(result, AuthResponse):
            return result.func_f1xsvgls(auth_request)
        return result

    def func_qyiniaas(self, event: Dict[str, Any]) -> "AuthRequest":
        return AuthRequest(event['type'], event['authorizationToken'], event['methodArn'])

    def func_gv635333(self, scopes: List[str]) -> "ChaliceAuthorizer":
        authorizer_with_scopes: ChaliceAuthorizer = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes


class AuthRequest(object):
    def __init__(self, auth_type: str, token: str, method_arn: str) -> None:
        self.auth_type: str = auth_type
        self.token: str = token
        self.method_arn: str = method_arn


class AuthResponse(object):
    ALL_HTTP_METHODS: List[str] = ['DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'POST', 'PUT', 'GET']

    def __init__(self, routes: List[Any], principal_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.routes: List[Any] = routes
        self.principal_id: str = principal_id
        if context is None:
            context = {}
        self.context: Dict[str, Any] = context

    def func_f1xsvgls(self, request: AuthRequest) -> Dict[str, Any]:
        return {'context': self.context, 'principalId': self.principal_id,
                'policyDocument': self._generate_policy(request)}

    def func_l0r7pq8z(self, request: AuthRequest) -> Dict[str, Any]:
        allowed_resources: List[str] = self._generate_allowed_resources(request)
        return {'Version': '2012-10-17', 'Statement': [{'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': allowed_resources}]}

    def func_uo81w7wz(self, request: AuthRequest) -> List[str]:
        allowed_resources: List[str] = []
        for route in self.routes:
            if isinstance(route, AuthRoute):
                methods: List[str] = route.methods
                path: str = route.path
            elif route == '*':
                methods = ['*']
                path = '*'
            else:
                methods = ['*']
                path = route
            for method in methods:
                allowed_resources.append(self._generate_arn(path, request, method))
        return allowed_resources

    def func_ubveyiys(self, route: str, request: AuthRequest, method: str = '*') -> str:
        incoming_arn: str = request.method_arn
        arn_parts: List[str] = incoming_arn.split(':', 5)
        allowed_resource: List[str] = arn_parts[-1].split('/')[:2]
        allowed_resource.extend([method, route[1:]])
        last_arn_segment: str = '/'.join(allowed_resource)
        if route == '*':
            last_arn_segment += route
        arn_parts[-1] = last_arn_segment
        final_arn: str = ':'.join(arn_parts)
        return final_arn

    def _generate_policy(self, request: AuthRequest) -> Dict[str, Any]:
        # Placeholder for policy generation logic.
        return {}

    def _generate_allowed_resources(self, request: AuthRequest) -> List[str]:
        # Placeholder for allowed resources generation.
        return []

    def _generate_arn(self, path: str, request: AuthRequest, method: str) -> str:
        return self.func_ubveyiys(path, request, method)


class AuthRoute(object):
    def __init__(self, path: str, methods: List[str]) -> None:
        self.path: str = path
        self.methods: List[str] = methods


class LambdaFunction(object):
    def __init__(self, func: Callable, name: str, handler_string: str) -> None:
        self.func: Callable = func
        self.name: str = name
        self.handler_string: str = handler_string

    def __call__(self, event: Dict[str, Any], context: Any) -> Any:
        return self.func(event, context)


class BaseEventSourceConfig(object):
    def __init__(self, name: str, handler_string: str) -> None:
        self.name: str = name
        self.handler_string: str = handler_string


class ScheduledEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, schedule_expression: Any, description: str) -> None:
        super(ScheduledEventConfig, self).__init__(name, handler_string)
        self.schedule_expression: Any = schedule_expression
        self.description: str = description


class CloudWatchEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, event_pattern: Any) -> None:
        super(CloudWatchEventConfig, self).__init__(name, handler_string)
        self.event_pattern: Any = event_pattern


class ScheduleExpression(object):
    def func_zj99nqpf(self) -> str:
        raise NotImplementedError('to_string')


class Rate(ScheduleExpression):
    MINUTES: str = 'MINUTES'
    HOURS: str = 'HOURS'
    DAYS: str = 'DAYS'

    def __init__(self, value: int, unit: str) -> None:
        self.value: int = value
        self.unit: str = unit

    def func_zj99nqpf(self) -> str:
        unit: str = self.unit.lower()
        if self.value == 1:
            unit = unit[:-1]
        return 'rate(%s %s)' % (self.value, unit)


class Cron(ScheduleExpression):
    def __init__(self, minutes: str, hours: str, day_of_month: str, month: str, day_of_week: str, year: str) -> None:
        self.minutes: str = minutes
        self.hours: str = hours
        self.day_of_month: str = day_of_month
        self.month: str = month
        self.day_of_week: str = day_of_week
        self.year: str = year

    def func_zj99nqpf(self) -> str:
        return 'cron(%s %s %s %s %s %s)' % (self.minutes, self.hours, self.day_of_month, self.month, self.day_of_week, self.year)


class S3EventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, bucket: str, events: List[str], prefix: Optional[str], suffix: Optional[str], handler_string: str) -> None:
        super(S3EventConfig, self).__init__(name, handler_string)
        self.bucket: str = bucket
        self.events: List[str] = events
        self.prefix: Optional[str] = prefix
        self.suffix: Optional[str] = suffix


class SNSEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, topic: str) -> None:
        super(SNSEventConfig, self).__init__(name, handler_string)
        self.topic: str = topic


class SQSEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, queue: Optional[str], queue_arn: Optional[str], batch_size: int, maximum_batching_window_in_seconds: int, maximum_concurrency: Optional[int]) -> None:
        super(SQSEventConfig, self).__init__(name, handler_string)
        self.queue: Optional[str] = queue
        self.queue_arn: Optional[str] = queue_arn
        self.batch_size: int = batch_size
        self.maximum_batching_window_in_seconds: int = maximum_batching_window_in_seconds
        self.maximum_concurrency: Optional[int] = maximum_concurrency


class KinesisEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, stream: str, batch_size: int, starting_position: str, maximum_batching_window_in_seconds: int) -> None:
        super(KinesisEventConfig, self).__init__(name, handler_string)
        self.stream: str = stream
        self.batch_size: int = batch_size
        self.starting_position: str = starting_position
        self.maximum_batching_window_in_seconds: int = maximum_batching_window_in_seconds


class DynamoDBEventConfig(BaseEventSourceConfig):
    def __init__(self, name: str, handler_string: str, stream_arn: str, batch_size: int, starting_position: str, maximum_batching_window_in_seconds: int) -> None:
        super(DynamoDBEventConfig, self).__init__(name, handler_string)
        self.stream_arn: str = stream_arn
        self.batch_size: int = batch_size
        self.starting_position: str = starting_position
        self.maximum_batching_window_in_seconds: int = maximum_batching_window_in_seconds


class WebsocketConnectConfig(BaseEventSourceConfig):
    CONNECT_ROUTE: str = '$connect'
    def __init__(self, name: str, handler_string: str, user_handler: Callable) -> None:
        super(WebsocketConnectConfig, self).__init__(name, handler_string)
        self.route_key_handled: str = self.CONNECT_ROUTE
        self.handler_function: Callable = user_handler


class WebsocketMessageConfig(BaseEventSourceConfig):
    def __init__(self, name: str, route_key_handled: str, handler_string: str, user_handler: Callable) -> None:
        super(WebsocketMessageConfig, self).__init__(name, handler_string)
        self.route_key_handled: str = route_key_handled
        self.handler_function: Callable = user_handler


class WebsocketDisconnectConfig(BaseEventSourceConfig):
    DISCONNECT_ROUTE: str = '$disconnect'
    def __init__(self, name: str, handler_string: str, user_handler: Callable) -> None:
        super(WebsocketDisconnectConfig, self).__init__(name, handler_string)
        self.route_key_handled: str = self.DISCONNECT_ROUTE
        self.handler_function: Callable = user_handler


class PureLambdaWrapper(object):
    def __init__(self, original_func: Callable) -> None:
        self._original_func: Callable = original_func

    def __call__(self, event: Any) -> Any:
        return self._original_func(event.to_dict(), event.context)


class MiddlewareHandler(object):
    def __init__(self, handler: Callable, next_handler: Callable) -> None:
        self.handler: Callable = handler
        self.next_handler: Callable = next_handler

    def __call__(self, request: Any) -> Any:
        return self.handler(request, self.next_handler)


class BaseLambdaHandler(object):
    def __call__(self, event: Any, context: Any) -> Any:
        pass

    def func_0awqlgli(self, handlers: List[Callable], original_handler: Callable) -> Callable:
        current: Callable = original_handler
        for handler in reversed(list(handlers)):
            current = MiddlewareHandler(handler=handler, next_handler=current)
        return current


class EventSourceHandler(BaseLambdaHandler):
    def __init__(self, func: Callable, event_class: Callable, middleware_handlers: Optional[List[Callable]] = None) -> None:
        self.func: Callable = func
        self.event_class: Callable = event_class
        if middleware_handlers is None:
            middleware_handlers = []
        self._middleware_handlers: List[Callable] = middleware_handlers
        self.handler: Optional[Callable] = None

    @property
    def func_pp7i98qf(self) -> List[Callable]:
        return self._middleware_handlers

    @func_pp7i98qf.setter
    def func_pp7i98qf(self, value: List[Callable]) -> None:
        self._middleware_handlers = value

    def __call__(self, event: Any, context: Any) -> Any:
        event_obj: Any = self.event_class(event, context)
        if self.handler is None:
            self.handler = self.func_0awqlgli(self._middleware_handlers, original_handler=self.func)
        return self.handler(event_obj)


class WebsocketEventSourceHandler(EventSourceHandler):
    WEBSOCKET_API_RESPONSE: Dict[str, int] = {'statusCode': 200}
    def __init__(self, func: Callable, event_class: Callable, websocket_api: WebsocketAPI, middleware_handlers: Optional[List[Callable]] = None) -> None:
        super(WebsocketEventSourceHandler, self).__init__(func, event_class, middleware_handlers)
        self.websocket_api: WebsocketAPI = websocket_api

    def __call__(self, event: Dict[str, Any], context: Any) -> Any:
        self.websocket_api.configure_from_api_id(event['requestContext']['apiId'], event['requestContext']['stage'])
        response: Any = super(WebsocketEventSourceHandler, self).__call__(event, context)
        data: Optional[Dict[str, Any]] = None
        if isinstance(response, Response):
            data = response.func_f1xsvgls()
        elif isinstance(response, dict):
            data = response
            if 'statusCode' not in data:
                data = {**self.WEBSOCKET_API_RESPONSE, **data}
        return data or self.WEBSOCKET_API_RESPONSE


class RestAPIEventHandler(BaseLambdaHandler):
    def __init__(self, route_table: Dict[str, Dict[str, RouteEntry]], api: APIGateway, log: logging.Logger, debug: bool, middleware_handlers: Optional[List[Callable]] = None) -> None:
        self.routes: Dict[str, Dict[str, RouteEntry]] = route_table
        self.api: APIGateway = api
        self.log: logging.Logger = log
        self.debug: bool = debug
        self.current_request: Optional[Request] = None
        self.lambda_context: Optional[Any] = None
        if middleware_handlers is None:
            middleware_handlers = []
        self._middleware_handlers: List[Callable] = middleware_handlers

    def func_qrlesjg5(self, event: Any, get_response: Callable) -> Any:
        try:
            return get_response(event)
        except Exception:
            return self._unhandled_exception_to_response()

    def func_38wk9dni(self, event: Dict[str, Any], context: Any) -> Optional[Request]:
        resource_path: Optional[str] = event.get('requestContext', {}).get('resourcePath')
        if resource_path is not None:
            self.current_request = Request(event, context)
            return self.current_request
        return None

    def __call__(self, event: Dict[str, Any], context: Any) -> Response:
        def wrapped_event(request: Request) -> Response:
            return self._main_rest_api_handler(event, context)
        final_handler: Callable = self.func_0awqlgli([self._global_error_handler] + list(self._middleware_handlers), original_handler=wrapped_event)
        response: Response = final_handler(self.current_request)  # type: ignore
        return response

    def _main_rest_api_handler(self, event: Dict[str, Any], context: Any) -> Response:
        resource_path: Optional[str] = event.get('requestContext', {}).get('resourcePath')
        if resource_path is None:
            return func_jbgc6yig(error_code='InternalServerError', message='Unknown request.', http_status_code=500)
        http_method: str = event['requestContext']['httpMethod']
        if http_method not in self.routes.get(resource_path, {}):
            allowed_methods: str = ', '.join(self.routes[resource_path].keys())
            return func_jbgc6yig(error_code='MethodNotAllowedError', message='Unsupported method: %s' % http_method, http_status_code=405, headers={'Allow': allowed_methods})
        route_entry: RouteEntry = self.routes[resource_path][http_method]
        view_function: Callable = route_entry.view_function
        function_args: Dict[str, Any] = {name: event['pathParameters'][name] for name in route_entry.view_args}  # type: ignore
        self.lambda_context = context
        cors_headers: Optional[Dict[str, str]] = None
        if self._cors_enabled_for_route(route_entry):
            cors_headers = self._get_cors_headers(route_entry.cors)  # type: ignore
        if self.current_request and route_entry.content_types:
            content_type: str = self.current_request.headers.get('content-type', 'application/json')
            if not func_cgh0i6nf(content_type, route_entry.content_types):
                return func_jbgc6yig(error_code='UnsupportedMediaType', message='Unsupported media type: %s' % content_type, http_status_code=415, headers=cors_headers)
        response: Response = self._get_view_function_response(view_function, function_args)
        if cors_headers is not None:
            self._add_cors_headers(response, cors_headers)
        response_headers: CaseInsensitiveMapping = CaseInsensitiveMapping(response.headers)
        if self.current_request and not self._validate_binary_response(self.current_request.headers, response_headers):
            content_type = response_headers.get('content-type', '')
            return func_jbgc6yig(error_code='BadRequest', message='Request did not specify an Accept header with %s, The response has a Content-Type of %s. If a response has a binary Content-Type then the request must specify an Accept header that matches.' % (content_type, content_type), http_status_code=400, headers=cors_headers)
        return response

    def _get_view_function_response(self, view_function: Callable, function_args: Dict[str, Any]) -> Response:
        try:
            result: Any = view_function(**function_args)
            if not isinstance(result, Response):
                result = Response(body=result)
            self._validate_response(result)
        except ChaliceUnhandledError:
            raise
        except ChaliceViewError as e:
            result = Response(body={'Code': e.__class__.__name__, 'Message': str(e)}, status_code=e.STATUS_CODE)
        except Exception:
            result = self._unhandled_exception_to_response()
        return result

    def _validate_response(self, response: Response) -> None:
        # Placeholder for response validation logic.
        pass

    def _global_error_handler(self, request: Request) -> Response:
        return self.func_y3b2zrat()

    def func_y3b2zrat(self) -> Response:
        headers: Dict[str, str] = {}
        path: str = getattr(self.current_request, 'path', 'unknown')
        self.log.error('Caught exception for path %s', path, exc_info=True)
        if self.debug:
            stack_trace: str = ''.join(traceback.format_exc())
            body: Any = stack_trace
            headers['Content-Type'] = 'text/plain'
        else:
            body = {'Code': 'InternalServerError', 'Message': 'An internal server error occurred.'}
        response: Response = Response(body=body, headers=headers, status_code=500)
        return response

    def _validate_binary_response(self, request_headers: Dict[str, Any], response_headers: CaseInsensitiveMapping) -> bool:
        request_accept_header: Optional[str] = request_headers.get('accept')
        response_content_type: str = response_headers.get('content-type', 'application/json')
        response_is_binary: bool = func_cgh0i6nf(response_content_type, self.api.binary_types)
        expects_binary_response: bool = False
        if request_accept_header is not None:
            expects_binary_response = func_cgh0i6nf(request_accept_header, self.api.binary_types)
        if response_is_binary and not expects_binary_response:
            return False
        return True

    def _cors_enabled_for_route(self, route_entry: RouteEntry) -> bool:
        return route_entry.cors is not None

    def _get_cors_headers(self, cors: CORSConfig) -> Dict[str, str]:
        return cors.get_access_control_headers()

    def _add_cors_headers(self, response: Response, cors_headers: Dict[str, str]) -> None:
        for name, value in cors_headers.items():
            if name not in response.headers:
                response.headers[name] = value

    def _unhandled_exception_to_response(self) -> Response:
        return func_jbgc6yig(message='Internal Server Error', error_code='InternalServerError', http_status_code=500)


class BaseLambdaEvent(object):
    def __init__(self, event_dict: Dict[str, Any], context: Any) -> None:
        self._event_dict: Dict[str, Any] = event_dict
        self.context: Any = context
        self.func_wsaqthlu(event_dict)

    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        raise NotImplementedError('_extract_attributes')

    def func_f1xsvgls(self) -> Dict[str, Any]:
        return self._event_dict


class LambdaFunctionEvent(BaseLambdaEvent):
    def __init__(self, event_dict: Dict[str, Any], context: Any) -> None:
        self.event: Dict[str, Any] = event_dict
        self.context: Any = context

    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        pass

    def func_f1xsvgls(self) -> Dict[str, Any]:
        return self.event


class CloudWatchEvent(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        self.version: str = event_dict['version']
        self.account: str = event_dict['account']
        self.region: str = event_dict['region']
        self.detail: Any = event_dict['detail']
        self.detail_type: str = event_dict['detail-type']
        self.source: str = event_dict['source']
        self.time: str = event_dict['time']
        self.event_id: str = event_dict['id']
        self.resources: List[str] = event_dict['resources']


class WebsocketEvent(BaseLambdaEvent):
    def __init__(self, event_dict: Dict[str, Any], context: Any) -> None:
        super(WebsocketEvent, self).__init__(event_dict, context)
        self._json_body: Optional[Any] = None

    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        request_context: Dict[str, Any] = event_dict['requestContext']
        self.domain_name: str = request_context['domainName']
        self.stage: str = request_context['stage']
        self.connection_id: str = request_context['connectionId']
        self.body: str = str(event_dict.get('body'))

    @property
    def func_o1nc530n(self) -> Any:
        if self._json_body is None:
            try:
                self._json_body = json.loads(self.body)
            except ValueError:
                raise BadRequestError('Error Parsing JSON')
        return self._json_body


class SNSEvent(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        first_record: Dict[str, Any] = event_dict['Records'][0]
        self.message: str = first_record['Sns']['Message']
        self.subject: str = first_record['Sns']['Subject']
        self.message_attributes: Dict[str, Any] = first_record['Sns']['MessageAttributes']


class S3Event(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        s3: Dict[str, Any] = event_dict['Records'][0]['s3']
        self.bucket: str = s3['bucket']['name']
        self.key: str = unquote_plus(s3['object']['key'])


class SQSEvent(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        pass

    def __iter__(self) -> Iterator["SQSRecord"]:
        for record in self._event_dict['Records']:
            yield SQSRecord(record, self.context)


class SQSRecord(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        self.body: str = event_dict['body']
        self.receipt_handle: str = event_dict['receiptHandle']


class KinesisEvent(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        pass

    def __iter__(self) -> Iterator["KinesisRecord"]:
        for record in self._event_dict['Records']:
            yield KinesisRecord(record, self.context)


class KinesisRecord(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        kinesis: Dict[str, Any] = event_dict['kinesis']
        encoded_payload: str = kinesis['data']
        self.data: bytes = base64.b64decode(encoded_payload)
        self.sequence_number: str = kinesis['sequenceNumber']
        self.partition_key: str = kinesis['partitionKey']
        self.schema_version: str = kinesis['kinesisSchemaVersion']
        self.timestamp: datetime.datetime = datetime.datetime.utcfromtimestamp(kinesis['approximateArrivalTimestamp'])


class DynamoDBEvent(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        pass

    def __iter__(self) -> Iterator["DynamoDBRecord"]:
        for record in self._event_dict['Records']:
            yield DynamoDBRecord(record, self.context)


class DynamoDBRecord(BaseLambdaEvent):
    def func_wsaqthlu(self, event_dict: Dict[str, Any]) -> None:
        dynamodb: Dict[str, Any] = event_dict['dynamodb']
        self.timestamp: datetime.datetime = datetime.datetime.utcfromtimestamp(dynamodb['ApproximateCreationDateTime'])
        self.keys: Optional[Dict[str, Any]] = dynamodb.get('Keys')
        self.new_image: Optional[Dict[str, Any]] = dynamodb.get('NewImage')
        self.old_image: Optional[Dict[str, Any]] = dynamodb.get('OldImage')
        self.sequence_number: str = dynamodb['SequenceNumber']
        self.size_bytes: int = dynamodb['SizeBytes']
        self.stream_view_type: str = dynamodb['StreamViewType']
        self.aws_region: str = event_dict['awsRegion']
        self.event_id: str = event_dict['eventID']
        self.event_name: str = event_dict['eventName']
        self.event_source_arn: str = event_dict['eventSourceARN']

    @property
    def func_mb3xm6yi(self) -> str:
        parts: List[str] = self.event_source_arn.split(':', 5)
        if not len(parts) == 6:
            return ''
        full_name: str = parts[-1]
        name_parts: List[str] = full_name.split('/')
        if len(name_parts) >= 2:
            return name_parts[1]
        return ''


class Blueprint(DecoratorAPI):
    def __init__(self, import_name: str) -> None:
        self._import_name: str = import_name
        self._deferred_registrations: List[Callable] = []
        self._current_app: Optional[Chalice] = None
        self._lambda_context: Optional[Any] = None

    @property
    def func_1dnfzrec(self) -> logging.Logger:
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.log if it's registered to an app.")
        return self._current_app.log

    @property
    def func_javv3wei(self) -> Request:
        if (self._current_app is None or self._current_app.current_request is None):
            raise RuntimeError("Can only access Blueprint.current_request if it's registered to an app.")
        return self._current_app.current_request  # type: ignore

    @property
    def func_4zge49ty(self) -> Chalice:
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.current_app if it's registered to an app.")
        return self._current_app

    @property
    def func_quo3iz0c(self) -> Any:
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.lambda_context if it's registered to an app.")
        return self._current_app.lambda_context

    def func_jvov7e85(self, app: Chalice, options: Dict[str, Any]) -> None:
        self._current_app = app
        all_options: Dict[str, Any] = options.copy()
        all_options['module_name'] = self._import_name
        for function in self._deferred_registrations:
            function(app, all_options)

    def func_460y1b75(self, func: Callable, event_type: str = 'all') -> None:
        self._deferred_registrations.append(lambda app, options: app.register_middleware(func, event_type))

    def func_d2tyxsl5(self, handler_type: str, name: Optional[str], user_handler: Callable, wrapped_handler: Callable, kwargs: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        def _register_blueprint_handler(app: Chalice, options: Dict[str, Any]) -> None:
            if handler_type in _EVENT_CLASSES:
                wrapped_handler.middleware_handlers = app._get_middleware_handlers(_MIDDLEWARE_MAPPING[handler_type])
            app._register_handler(handler_type, name if name is not None else user_handler.__name__, user_handler, wrapped_handler, kwargs, options)
        self._deferred_registrations.append(_register_blueprint_handler)

    def func_eeikzeqs(self, event_type: str) -> List[Callable]:
        return []


class ConvertToMiddleware(object):
    def __init__(self, lambda_wrapper: Callable) -> None:
        self._wrapper: Callable = lambda_wrapper

    def __call__(self, event: Any, get_response: Callable) -> Any:
        original_event, context = self.func_0uw2odw8(event)
        @functools.wraps(self._wrapper)
        def wrapped(original_event: Any, context: Any) -> Any:
            return get_response(event)
        return self._wrapper(wrapped)(original_event, context)

    def func_0uw2odw8(self, event: Any) -> Tuple[Any, Any]:
        if hasattr(event, 'to_original_event'):
            return event.to_original_event(), getattr(event, 'lambda_context', None)
        return event.to_dict(), event.context


_EVENT_CLASSES: Dict[str, Any] = {'on_s3_event': S3Event, 'on_sns_message': SNSEvent,
                                   'on_sqs_message': SQSEvent, 'on_cw_event': CloudWatchEvent,
                                   'on_kinesis_record': KinesisEvent, 'on_dynamodb_record': DynamoDBEvent,
                                   'schedule': CloudWatchEvent, 'lambda_function': LambdaFunctionEvent}
_MIDDLEWARE_MAPPING: Dict[str, str] = {'on_s3_event': 's3', 'on_sns_message': 'sns',
                                       'on_sqs_message': 'sqs', 'on_cw_event': 'cloudwatch',
                                       'on_kinesis_record': 'kinesis', 'on_dynamodb_record': 'dynamodb',
                                       'schedule': 'scheduled', 'lambda_function': 'pure_lambda'}
