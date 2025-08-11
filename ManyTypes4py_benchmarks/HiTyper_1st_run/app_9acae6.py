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
from collections.abc import Mapping
from collections.abc import MutableMapping
__version__ = '1.31.4'
from typing import List, Dict, Any, Optional, Sequence, Union, Callable, Set, Iterator, TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from chalice.local import LambdaContext
_PARAMS = re.compile('{\\w+}')
MiddlewareFuncType = Callable[[Any, Callable[[Any], Any]], Any]
UserHandlerFuncType = Callable[..., Any]
HeadersType = Dict[str, Union[str, List[str]]]
_ANY_STRING = (str, bytes)

def handle_extra_types(obj: Union[int, decimal.Decimal, Cash]) -> Union[float, str, dict[str, typing.Any]]:
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, MultiDict):
        return dict(obj)
    raise TypeError('Object of type %s is not JSON serializable' % obj.__class__.__name__)

def error_response(message: Union[str, int, None], error_code: Union[str, int, None], http_status_code: Union[int, str, dict], headers: Union[None, int, str, dict]=None) -> Response:
    body = {'Code': error_code, 'Message': message}
    response = Response(body=body, status_code=http_status_code, headers=headers)
    return response

def _matches_content_type(content_type: Union[str, bytes], valid_content_types: Union[str, dict]) -> bool:
    content_type = content_type.lower()
    valid_content_types = [x.lower() for x in valid_content_types]
    return '*/*' in content_type or '*/*' in valid_content_types or _content_type_header_contains(content_type, valid_content_types)

def _content_type_header_contains(content_type_header: Union[str, typing.Sequence[str], None], valid_content_types: str) -> bool:
    content_type_header_parts = [p.strip() for p in re.split('[,;]', content_type_header)]
    valid_parts = set(valid_content_types).intersection(content_type_header_parts)
    return len(valid_parts) > 0

class ChaliceError(Exception):
    pass

class WebsocketDisconnectedError(ChaliceError):

    def __init__(self, connection_id) -> None:
        self.connection_id = connection_id

class ChaliceViewError(ChaliceError):
    STATUS_CODE = 500

class ChaliceUnhandledError(ChaliceError):
    """This error is not caught from a Chalice view function.

    This exception is allowed to propagate from a view function so
    that middleware handlers can process the exception.
    """

class BadRequestError(ChaliceViewError):
    STATUS_CODE = 400

class UnauthorizedError(ChaliceViewError):
    STATUS_CODE = 401

class ForbiddenError(ChaliceViewError):
    STATUS_CODE = 403

class NotFoundError(ChaliceViewError):
    STATUS_CODE = 404

class MethodNotAllowedError(ChaliceViewError):
    STATUS_CODE = 405

class RequestTimeoutError(ChaliceViewError):
    STATUS_CODE = 408

class ConflictError(ChaliceViewError):
    STATUS_CODE = 409

class UnprocessableEntityError(ChaliceViewError):
    STATUS_CODE = 422

class TooManyRequestsError(ChaliceViewError):
    STATUS_CODE = 429
ALL_ERRORS = [ChaliceViewError, BadRequestError, NotFoundError, UnauthorizedError, ForbiddenError, MethodNotAllowedError, RequestTimeoutError, ConflictError, UnprocessableEntityError, TooManyRequestsError]

class MultiDict(MutableMapping):
    """A mapping of key to list of values.

    Accessing it in the usual way will return the last value in the list.
    Calling getlist will return a list of all the values associated with
    the same key.
    """

    def __init__(self, mapping) -> None:
        if mapping is None:
            mapping = {}
        self._dict = mapping

    def __getitem__(self, k: Union[str, int]):
        try:
            return self._dict[k][-1]
        except IndexError:
            raise KeyError(k)

    def __setitem__(self, k: Union[str, int], v: Union[str, int]) -> None:
        self._dict[k] = [v]

    def __delitem__(self, k: Union[str, int]) -> None:
        del self._dict[k]

    def getlist(self, k: Union[str, int]) -> list:
        return list(self._dict[k])

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> typing.Generator[SQSRecord]:
        return iter(self._dict)

    def __repr__(self) -> typing.Text:
        return 'MultiDict(%s)' % self._dict

    def __str__(self) -> str:
        return repr(self)

class CaseInsensitiveMapping(Mapping):
    """Case insensitive and read-only mapping."""

    def __init__(self, mapping) -> None:
        mapping = mapping or {}
        self._dict = {k.lower(): v for k, v in mapping.items()}

    def __getitem__(self, key):
        return self._dict[key.lower()]

    def __iter__(self) -> typing.Generator[SQSRecord]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> typing.Text:
        return 'CaseInsensitiveMapping(%s)' % repr(self._dict)

class Authorizer(object):
    name = ''
    scopes = []

    def to_swagger(self) -> dict[typing.Text, typing.Text]:
        raise NotImplementedError('to_swagger')

    def with_scopes(self, scopes: Union[typing.Mapping, dict, dict[str, typing.Any]]) -> None:
        raise NotImplementedError('with_scopes')

class IAMAuthorizer(Authorizer):
    _AUTH_TYPE = 'aws_iam'

    def __init__(self) -> None:
        self.name = 'sigv4'
        self.scopes = []

    def to_swagger(self) -> dict[typing.Text, typing.Text]:
        return {'in': 'header', 'type': 'apiKey', 'name': 'Authorization', 'x-amazon-apigateway-authtype': 'awsSigv4'}

    def with_scopes(self, scopes: Union[typing.Mapping, dict, dict[str, typing.Any]]) -> None:
        raise NotImplementedError('with_scopes')

class CognitoUserPoolAuthorizer(Authorizer):
    _AUTH_TYPE = 'cognito_user_pools'

    def __init__(self, name, provider_arns, header='Authorization', scopes=None) -> None:
        self.name = name
        self._header = header
        if not isinstance(provider_arns, list):
            raise TypeError('provider_arns should be a list of ARNs, received: %s' % provider_arns)
        self._provider_arns = provider_arns
        self.scopes = scopes or []

    def to_swagger(self) -> dict[typing.Text, typing.Text]:
        return {'in': 'header', 'type': 'apiKey', 'name': self._header, 'x-amazon-apigateway-authtype': self._AUTH_TYPE, 'x-amazon-apigateway-authorizer': {'type': self._AUTH_TYPE, 'providerARNs': self._provider_arns}}

    def with_scopes(self, scopes: Union[typing.Mapping, dict, dict[str, typing.Any]]) -> None:
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes

class CustomAuthorizer(Authorizer):
    _AUTH_TYPE = 'custom'

    def __init__(self, name, authorizer_uri, ttl_seconds=300, header='Authorization', invoke_role_arn=None, scopes=None) -> None:
        self.name = name
        self._header = header
        self._authorizer_uri = authorizer_uri
        self._ttl_seconds = ttl_seconds
        self._invoke_role_arn = invoke_role_arn
        self.scopes = scopes or []

    def to_swagger(self) -> dict[typing.Text, typing.Text]:
        swagger = {'in': 'header', 'type': 'apiKey', 'name': self._header, 'x-amazon-apigateway-authtype': self._AUTH_TYPE, 'x-amazon-apigateway-authorizer': {'type': 'token', 'authorizerUri': self._authorizer_uri, 'authorizerResultTtlInSeconds': self._ttl_seconds}}
        if self._invoke_role_arn is not None:
            swagger['x-amazon-apigateway-authorizer']['authorizerCredentials'] = self._invoke_role_arn
        return swagger

    def with_scopes(self, scopes: Union[typing.Mapping, dict, dict[str, typing.Any]]) -> None:
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes

class CORSConfig(object):
    """A cors configuration to attach to a route."""
    _REQUIRED_HEADERS = ['Content-Type', 'X-Amz-Date', 'Authorization', 'X-Api-Key', 'X-Amz-Security-Token']

    def __init__(self, allow_origin='*', allow_headers=None, expose_headers=None, max_age=None, allow_credentials=None) -> None:
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

    def get_access_control_headers(self) -> dict[typing.Text, str]:
        headers = {'Access-Control-Allow-Origin': self.allow_origin, 'Access-Control-Allow-Headers': self.allow_headers}
        if self._expose_headers:
            headers.update({'Access-Control-Expose-Headers': ','.join(self._expose_headers)})
        if self._max_age is not None:
            headers.update({'Access-Control-Max-Age': str(self._max_age)})
        if self._allow_credentials is True:
            headers.update({'Access-Control-Allow-Credentials': 'true'})
        return headers

    def __eq__(self, other: Union[dict, T]) -> bool:
        if isinstance(other, self.__class__):
            return self.get_access_control_headers() == other.get_access_control_headers()
        return False

class Request(object):
    """The current request from API gateway."""
    _NON_SERIALIZED_ATTRS = ['lambda_context']

    def __init__(self, event_dict, lambda_context=None) -> None:
        query_params = event_dict['multiValueQueryStringParameters']
        self.query_params = None if query_params is None else MultiDict(query_params)
        self.headers = CaseInsensitiveMapping(event_dict['headers'])
        self.uri_params = event_dict['pathParameters']
        self.method = event_dict['requestContext']['httpMethod']
        self._is_base64_encoded = event_dict.get('isBase64Encoded', False)
        self._body = event_dict['body']
        self._json_body = None
        self._raw_body = b''
        self.context = event_dict['requestContext']
        self.stage_vars = event_dict['stageVariables']
        self.path = event_dict['requestContext']['resourcePath']
        self.lambda_context = lambda_context
        self._event_dict = event_dict

    def _base64decode(self, encoded: str):
        if not isinstance(encoded, bytes):
            encoded = encoded.encode('ascii')
        output = base64.b64decode(encoded)
        return output

    @property
    def raw_body(self) -> Union[bytes, str]:
        if not self._raw_body and self._body is not None:
            if self._is_base64_encoded:
                self._raw_body = self._base64decode(self._body)
            elif not isinstance(self._body, bytes):
                self._raw_body = self._body.encode('utf-8')
            else:
                self._raw_body = self._body
        return self._raw_body

    @property
    def json_body(self) -> Union[str, dict[str, tuple[typing.Union[typing.Any,str]]], None]:
        if self.headers.get('content-type', '').startswith('application/json'):
            if self._json_body is None:
                try:
                    self._json_body = json.loads(self.raw_body)
                except ValueError:
                    raise BadRequestError('Error Parsing JSON')
            return self._json_body

    def to_dict(self) -> dict[typing.Text, typing.Union[str,bytes,dict[str, str]]]:
        copied = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in self._NON_SERIALIZED_ATTRS}
        copied['headers'] = dict(copied['headers'])
        if copied['query_params'] is not None:
            copied['query_params'] = dict(copied['query_params'])
        return copied

    def to_original_event(self):
        return self._event_dict

class Response(object):

    def __init__(self, body, headers=None, status_code=200) -> None:
        self.body = body
        if headers is None:
            headers = {}
        self.headers = headers
        self.status_code = status_code

    def to_dict(self, binary_types: Union[None, str, list[str]]=None) -> dict[typing.Text, typing.Union[str,bytes,dict[str, str]]]:
        body = self.body
        if not isinstance(body, _ANY_STRING):
            body = json.dumps(body, separators=(',', ':'), default=handle_extra_types)
        single_headers, multi_headers = self._sort_headers(self.headers)
        response = {'headers': single_headers, 'multiValueHeaders': multi_headers, 'statusCode': self.status_code, 'body': body}
        if binary_types is not None:
            self._b64encode_body_if_needed(response, binary_types)
        return response

    def _sort_headers(self, all_headers: Union[dict, dict[str, typing.Any]]) -> tuple[typing.Union[dict[tuple[typing.Union[str,typing.Any]], tuple[typing.Union[str,typing.Any]]],dict[tuple[typing.Union[str,typing.Any]], list]]]:
        multi_headers = {}
        single_headers = {}
        for name, value in all_headers.items():
            if isinstance(value, list):
                multi_headers[name] = value
            else:
                single_headers[name] = value
        return (single_headers, multi_headers)

    def _b64encode_body_if_needed(self, response_dict: Any, binary_types: Union[str, dict[str, typing.Any], T]) -> None:
        response_headers = CaseInsensitiveMapping(response_dict['headers'])
        content_type = response_headers.get('content-type', '')
        body = response_dict['body']
        if _matches_content_type(content_type, binary_types):
            if _matches_content_type(content_type, ['application/json']) or not content_type:
                body = body if isinstance(body, bytes) else body.encode('utf-8')
            body = self._base64encode(body)
            response_dict['isBase64Encoded'] = True
        response_dict['body'] = body

    def _base64encode(self, data: Union[bytes, str]):
        if not isinstance(data, bytes):
            raise ValueError('Expected bytes type for body with binary Content-Type. Got %s type body instead.' % type(data))
        data = base64.b64encode(data)
        return data.decode('ascii')

class RouteEntry(object):

    def __init__(self, view_function, view_name, path, method, api_key_required=None, content_types=None, cors=False, authorizer=None) -> None:
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

    def _parse_view_args(self) -> list:
        if '{' not in self.uri_pattern:
            return []
        results = [r[1:-1] for r in _PARAMS.findall(self.uri_pattern)]
        return results

    def __eq__(self, other: Union[dict, T]) -> bool:
        return self.__dict__ == other.__dict__

class APIGateway(object):
    _DEFAULT_BINARY_TYPES = ['application/octet-stream', 'application/x-tar', 'application/zip', 'audio/basic', 'audio/ogg', 'audio/mp4', 'audio/mpeg', 'audio/wav', 'audio/webm', 'image/png', 'image/jpg', 'image/jpeg', 'image/gif', 'video/ogg', 'video/mpeg', 'video/webm']

    def __init__(self) -> None:
        self.binary_types = self.default_binary_types
        self.cors = False

    @property
    def default_binary_types(self) -> list:
        return list(self._DEFAULT_BINARY_TYPES)

class WebsocketAPI(object):
    _WEBSOCKET_ENDPOINT_TEMPLATE = 'https://{domain_name}/{stage}'
    _REGION_ENV_VARS = ['AWS_REGION', 'AWS_DEFAULT_REGION']

    def __init__(self, env=None) -> None:
        self.session = None
        self._endpoint = None
        self._client = None
        if env is None:
            self._env = os.environ
        else:
            self._env = env

    def configure(self, domain_name: Union[str, None, int], stage: Union[str, None, int]) -> None:
        if self._endpoint is not None:
            return
        self._endpoint = self._WEBSOCKET_ENDPOINT_TEMPLATE.format(domain_name=domain_name, stage=stage)

    def configure_from_api_id(self, api_id: Union[str, int], stage: Union[str, None, int]) -> None:
        if self._endpoint is not None:
            return
        region_name = self._get_region()
        if region_name.startswith('cn-'):
            domain_name_template = '{api_id}.execute-api.{region}.amazonaws.com.cn'
        else:
            domain_name_template = '{api_id}.execute-api.{region}.amazonaws.com'
        domain_name = domain_name_template.format(api_id=api_id, region=region_name)
        self.configure(domain_name, stage)

    def _get_region(self) -> Union[str, list, list[str]]:
        for varname in self._REGION_ENV_VARS:
            if varname in self._env:
                return self._env[varname]
        if self.session is not None:
            region_name = self.session.region_name
            if region_name is not None:
                return region_name
        raise ValueError("Unable to retrieve the region name when configuring the websocket client.  Either set the 'AWS_REGION' environment variable or assign 'app.websocket_api.session' to a boto3 session.")

    def _get_client(self):
        if self.session is None:
            raise ValueError('Assign app.websocket_api.session to a boto3 session before using the WebsocketAPI')
        if self._endpoint is None:
            raise ValueError('WebsocketAPI.configure must be called before using the WebsocketAPI')
        if self._client is None:
            self._client = self.session.client('apigatewaymanagementapi', endpoint_url=self._endpoint)
        return self._client

    def send(self, connection_id: Union[str, int], message: Union[str, int]) -> None:
        client = self._get_client()
        try:
            client.post_to_connection(ConnectionId=connection_id, Data=message)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)

    def close(self, connection_id: Union[str, int]) -> None:
        client = self._get_client()
        try:
            client.delete_connection(ConnectionId=connection_id)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)

    def info(self, connection_id: Union[str, int, None]) -> dict:
        client = self._get_client()
        try:
            return client.get_connection(ConnectionId=connection_id)
        except client.exceptions.GoneException:
            raise WebsocketDisconnectedError(connection_id)

class DecoratorAPI(object):
    websocket_api = None

    def middleware(self, event_type: typing.Text='all'):

        def _middleware_wrapper(func: Any):
            self.register_middleware(func, event_type)
            return func
        return _middleware_wrapper

    def authorizer(self, ttl_seconds: Union[None, str, typing.Any, int]=None, execution_role: Union[None, str, typing.Any, int]=None, name: Union[None, str, typing.Any, int]=None, header: typing.Text='Authorization') -> Union[str, list[str], None, typing.Callable]:
        return self._create_registration_function(handler_type='authorizer', name=name, registration_kwargs={'ttl_seconds': ttl_seconds, 'execution_role': execution_role, 'header': header})

    def on_s3_event(self, bucket: Union[str, dict], events: Union[None, str, dict]=None, prefix: Union[None, str, dict]=None, suffix: Union[None, str, dict]=None, name: Union[None, str, dict]=None):
        return self._create_registration_function(handler_type='on_s3_event', name=name, registration_kwargs={'bucket': bucket, 'events': events, 'prefix': prefix, 'suffix': suffix})

    def on_sns_message(self, topic: str, name: Union[None, str]=None):
        return self._create_registration_function(handler_type='on_sns_message', name=name, registration_kwargs={'topic': topic})

    def on_sqs_message(self, queue: Union[None, int, datetime.date]=None, batch_size: int=1, name: Union[None, int, datetime.date]=None, queue_arn: Union[None, int, datetime.date]=None, maximum_batching_window_in_seconds: int=0, maximum_concurrency: Union[None, int, datetime.date]=None):
        return self._create_registration_function(handler_type='on_sqs_message', name=name, registration_kwargs={'queue': queue, 'queue_arn': queue_arn, 'batch_size': batch_size, 'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds, 'maximum_concurrency': maximum_concurrency})

    def on_cw_event(self, event_pattern: Union[str, list[str]], name: Union[None, str, list[str]]=None) -> bool:
        return self._create_registration_function(handler_type='on_cw_event', name=name, registration_kwargs={'event_pattern': event_pattern})

    def schedule(self, expression: Union[str, typing.Type, dict], name: Union[None, str, typing.Type, dict]=None, description: typing.Text='') -> Union[str, typing.Callable]:
        return self._create_registration_function(handler_type='schedule', name=name, registration_kwargs={'expression': expression, 'description': description})

    def on_kinesis_record(self, stream: Union[bool, typing.Callable, None], batch_size: int=100, starting_position: typing.Text='LATEST', name: Union[None, bool, typing.Callable]=None, maximum_batching_window_in_seconds: int=0):
        return self._create_registration_function(handler_type='on_kinesis_record', name=name, registration_kwargs={'stream': stream, 'batch_size': batch_size, 'starting_position': starting_position, 'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds})

    def on_dynamodb_record(self, stream_arn: int, batch_size: int=100, starting_position: typing.Text='LATEST', name: Union[None, int]=None, maximum_batching_window_in_seconds: int=0) -> bool:
        return self._create_registration_function(handler_type='on_dynamodb_record', name=name, registration_kwargs={'stream_arn': stream_arn, 'batch_size': batch_size, 'starting_position': starting_position, 'maximum_batching_window_in_seconds': maximum_batching_window_in_seconds})

    def route(self, path: str, **kwargs):
        return self._create_registration_function(handler_type='route', name=kwargs.pop('name', None), registration_kwargs={'path': path, 'kwargs': kwargs})

    def lambda_function(self, name: Union[None, str]=None) -> typing.Callable:
        return self._create_registration_function(handler_type='lambda_function', name=name)

    def on_ws_connect(self, name: Union[None, str, typing.Any]=None) -> bool:
        return self._create_registration_function(handler_type='on_ws_connect', name=name, registration_kwargs={'route_key': '$connect'})

    def on_ws_disconnect(self, name: Union[None, str, typing.Any]=None):
        return self._create_registration_function(handler_type='on_ws_disconnect', name=name, registration_kwargs={'route_key': '$disconnect'})

    def on_ws_message(self, name: Union[None, str, typing.Any]=None):
        return self._create_registration_function(handler_type='on_ws_message', name=name, registration_kwargs={'route_key': '$default'})

    def _create_registration_function(self, handler_type: Union[typing.Callable, str, int], name: Union[None, typing.Callable, str, int]=None, registration_kwargs: Union[None, typing.Callable, str, int]=None):

        def _register_handler(user_handler: str):
            handler_name = name
            if handler_name is None:
                handler_name = user_handler.__name__
            if registration_kwargs is not None:
                kwargs = registration_kwargs
            else:
                kwargs = {}
            wrapped = self._wrap_handler(handler_type, handler_name, user_handler)
            self._register_handler(handler_type, handler_name, user_handler, wrapped, kwargs)
            return wrapped
        return _register_handler

    def _wrap_handler(self, handler_type: Union[str, T], handler_name: Union[typing.Callable[..., collections.abc.Coroutine], str], user_handler: Union[str, T, typing.Callable, None]) -> Union[EventSourceHandler, WebsocketEventSourceHandler, ChaliceAuthorizer, str, T, typing.Callable, None, PureLambdaWrapper]:
        if handler_type in _EVENT_CLASSES:
            if handler_type == 'lambda_function':
                user_handler = PureLambdaWrapper(user_handler)
            return EventSourceHandler(user_handler, _EVENT_CLASSES[handler_type], middleware_handlers=self._get_middleware_handlers(event_type=_MIDDLEWARE_MAPPING[handler_type]))
        websocket_event_classes = ['on_ws_connect', 'on_ws_message', 'on_ws_disconnect']
        if self.websocket_api and handler_type in websocket_event_classes:
            return WebsocketEventSourceHandler(user_handler, WebsocketEvent, self.websocket_api, middleware_handlers=self._get_middleware_handlers(event_type='websocket'))
        if handler_type == 'authorizer':
            return ChaliceAuthorizer(handler_name, user_handler)
        return user_handler

    def _get_middleware_handlers(self, event_type: Union[str, dict[str, typing.Any], None, list]) -> None:
        raise NotImplementedError('_get_middleware_handlers')

    def _register_handler(self, handler_type, name, user_handler: str, wrapped_handler, kwargs, options=None):
        raise NotImplementedError('_register_handler')

    def register_middleware(self, func: Union[typing.Sized, str], event_type: typing.Text='all') -> None:
        raise NotImplementedError('register_middleware')

class _HandlerRegistration(object):

    def __init__(self) -> None:
        self.routes = defaultdict(dict)
        self.websocket_handlers = {}
        self.builtin_auth_handlers = []
        self.event_sources = []
        self.pure_lambda_functions = []
        self.api = APIGateway()
        self.handler_map = {}
        self.middleware_handlers = []

    def register_middleware(self, func: Union[typing.Sized, str], event_type: typing.Text='all') -> None:
        self.middleware_handlers.append((func, event_type))

    def _do_register_handler(self, handler_type: Union[str, Exception, None], name: str, user_handler: Union[str, None], wrapped_handler: Union[str, Exception, dict], kwargs: Any, options: Union[dict[str, typing.Any], None]=None) -> None:
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
        getattr(self, '_register_%s' % handler_type)(name=name, user_handler=user_handler, handler_string=handler_string, wrapped_handler=wrapped_handler, kwargs=kwargs)
        self.handler_map[name] = wrapped_handler

    def _attach_websocket_handler(self, handler: typing.Callable) -> None:
        route_key = handler.route_key_handled
        decorator_name = {'$default': 'on_ws_message', '$connect': 'on_ws_connect', '$disconnect': 'on_ws_disconnect'}.get(route_key)
        if route_key in self.websocket_handlers:
            raise ValueError("Duplicate websocket handler: '%s'. There can only be one handler for each websocket decorator." % decorator_name)
        self.websocket_handlers[route_key] = handler

    def _register_on_ws_connect(self, name: Union[str, BaseException], user_handler: Union[str, BaseException], handler_string: Union[str, BaseException], kwargs: Any, **unused) -> None:
        wrapper = WebsocketConnectConfig(name=name, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)

    def _register_on_ws_message(self, name: str, user_handler: str, handler_string: str, kwargs: Any, **unused) -> None:
        route_key = kwargs['route_key']
        wrapper = WebsocketMessageConfig(name=name, route_key_handled=route_key, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)
        self.websocket_handlers[route_key] = wrapper

    def _register_on_ws_disconnect(self, name: Union[str, dict[str, typing.Any], dict[str, str]], user_handler: Union[str, dict[str, typing.Any], dict[str, str]], handler_string: Union[str, dict[str, typing.Any], dict[str, str]], kwargs: Any, **unused) -> None:
        wrapper = WebsocketDisconnectConfig(name=name, handler_string=handler_string, user_handler=user_handler)
        self._attach_websocket_handler(wrapper)

    def _register_lambda_function(self, name: Union[str, dict, Exception, None], user_handler: Union[str, dict, Exception, None], handler_string: Union[str, dict, Exception, None], **unused) -> None:
        wrapper = LambdaFunction(func=user_handler, name=name, handler_string=handler_string)
        self.pure_lambda_functions.append(wrapper)

    def _register_on_s3_event(self, name: Union[str, dict[str, typing.Any]], handler_string: Union[str, dict[str, typing.Any]], kwargs: Any, **unused) -> None:
        events = kwargs['events']
        if events is None:
            events = ['s3:ObjectCreated:*']
        s3_event = S3EventConfig(name=name, bucket=kwargs['bucket'], events=events, prefix=kwargs['prefix'], suffix=kwargs['suffix'], handler_string=handler_string)
        self.event_sources.append(s3_event)

    def _register_on_sns_message(self, name: Union[str, dict[str, str]], handler_string: Union[str, dict[str, str]], kwargs: Any, **unused) -> None:
        sns_config = SNSEventConfig(name=name, handler_string=handler_string, topic=kwargs['topic'])
        self.event_sources.append(sns_config)

    def _register_on_sqs_message(self, name: Union[str, bytes, list[str], None], handler_string: Union[str, bytes, list[str], None], kwargs: Any, **unused) -> None:
        queue = kwargs.get('queue')
        queue_arn = kwargs.get('queue_arn')
        if not queue and (not queue_arn):
            raise ValueError('Must provide either `queue` or `queue_arn` to the `on_sqs_message` decorator.')
        sqs_config = SQSEventConfig(name=name, handler_string=handler_string, queue=queue, queue_arn=queue_arn, batch_size=kwargs['batch_size'], maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds'], maximum_concurrency=kwargs['maximum_concurrency'])
        self.event_sources.append(sqs_config)

    def _register_on_kinesis_record(self, name: Union[str, None, typing.Callable], handler_string: Union[str, None, typing.Callable], kwargs: Any, **unused) -> None:
        kinesis_config = KinesisEventConfig(name=name, handler_string=handler_string, stream=kwargs['stream'], batch_size=kwargs['batch_size'], starting_position=kwargs['starting_position'], maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds'])
        self.event_sources.append(kinesis_config)

    def _register_on_dynamodb_record(self, name: Union[str, bool, list[str]], handler_string: Union[str, bool, list[str]], kwargs: Any, **unused) -> None:
        ddb_config = DynamoDBEventConfig(name=name, handler_string=handler_string, stream_arn=kwargs['stream_arn'], batch_size=kwargs['batch_size'], starting_position=kwargs['starting_position'], maximum_batching_window_in_seconds=kwargs['maximum_batching_window_in_seconds'])
        self.event_sources.append(ddb_config)

    def _register_on_cw_event(self, name: Union[str, int], handler_string: Union[str, int], kwargs: Any, **unused) -> None:
        event_source = CloudWatchEventConfig(name=name, event_pattern=kwargs['event_pattern'], handler_string=handler_string)
        self.event_sources.append(event_source)

    def _register_schedule(self, name: Union[str, typing.Callable, dict[str, str]], handler_string: Union[str, typing.Callable, dict[str, str]], kwargs: Any, **unused) -> None:
        event_source = ScheduledEventConfig(name=name, schedule_expression=kwargs['expression'], description=kwargs['description'], handler_string=handler_string)
        self.event_sources.append(event_source)

    def _register_authorizer(self, name: Union[str, bool, None], handler_string: Union[str, bool, None], wrapped_handler: Union[bool, str, typing.Callable], kwargs: Any, **unused) -> None:
        actual_kwargs = kwargs.copy()
        ttl_seconds = actual_kwargs.pop('ttl_seconds', None)
        execution_role = actual_kwargs.pop('execution_role', None)
        header = actual_kwargs.pop('header', None)
        if actual_kwargs:
            raise TypeError('TypeError: authorizer() got unexpected keyword arguments: %s' % ', '.join(list(actual_kwargs)))
        auth_config = BuiltinAuthConfig(name=name, handler_string=handler_string, ttl_seconds=ttl_seconds, execution_role=execution_role, header=header)
        wrapped_handler.config = auth_config
        self.builtin_auth_handlers.append(auth_config)

    def _register_route(self, name: Union[str, bytes], user_handler: str, kwargs: Any, **unused) -> None:
        actual_kwargs = kwargs['kwargs']
        path = kwargs['path']
        url_prefix = kwargs.pop('url_prefix', None)
        if url_prefix is not None:
            path = '/'.join([url_prefix.rstrip('/'), path.strip('/')]).rstrip('/')
        methods = actual_kwargs.pop('methods', ['GET'])
        route_kwargs = {'authorizer': actual_kwargs.pop('authorizer', None), 'api_key_required': actual_kwargs.pop('api_key_required', None), 'content_types': actual_kwargs.pop('content_types', ['application/json']), 'cors': actual_kwargs.pop('cors', self.api.cors)}
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
    FORMAT_STRING = '%(name)s - %(levelname)s - %(message)s'

    def __init__(self, app_name, debug=False, configure_logs=True, env=None) -> None:
        super(Chalice, self).__init__()
        self.app_name = app_name
        self.websocket_api = WebsocketAPI()
        self._debug = debug
        self.configure_logs = configure_logs
        self.log = logging.getLogger(self.app_name)
        if env is None:
            env = os.environ
        self._initialize(env)
        self.experimental_feature_flags = set()
        self._features_used = set()

    def _initialize(self, env: Any) -> None:
        if self.configure_logs:
            self._configure_logging()
        env['AWS_EXECUTION_ENV'] = '%s aws-chalice/%s' % (env.get('AWS_EXECUTION_ENV', 'AWS_Lambda'), __version__)

    @property
    def debug(self) -> Union[bool, str]:
        return self._debug

    @debug.setter
    def debug(self, value) -> Union[bool, str]:
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

    def _already_configured(self, log: Union[dict, dict[str, str], logging.Logger, None]) -> bool:
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

    def register_blueprint(self, blueprint: Union[str, typing.Callable, int], name_prefix: Union[None, str, typing.Callable, int]=None, url_prefix: Union[None, str, typing.Callable, int]=None) -> None:
        blueprint.register(self, options={'name_prefix': name_prefix, 'url_prefix': url_prefix})

    def _register_handler(self, handler_type, name, user_handler: str, wrapped_handler, kwargs, options=None):
        self._do_register_handler(handler_type, name, user_handler, wrapped_handler, kwargs, options)

    def _register_on_ws_connect(self, name: Union[str, BaseException], user_handler: Union[str, BaseException], handler_string: Union[str, BaseException], kwargs: Any, **unused) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_on_ws_connect(name, user_handler, handler_string, kwargs, **unused)

    def _register_on_ws_message(self, name: str, user_handler: str, handler_string: str, kwargs: Any, **unused) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_on_ws_message(name, user_handler, handler_string, kwargs, **unused)

    def _register_on_ws_disconnect(self, name: Union[str, dict[str, typing.Any], dict[str, str]], user_handler: Union[str, dict[str, typing.Any], dict[str, str]], handler_string: Union[str, dict[str, typing.Any], dict[str, str]], kwargs: Any, **unused) -> None:
        self._features_used.add('WEBSOCKETS')
        super(Chalice, self)._register_on_ws_disconnect(name, user_handler, handler_string, kwargs, **unused)

    def _get_middleware_handlers(self, event_type: Union[str, dict[str, typing.Any], None, list]) -> None:
        return (func for func, filter_type in self.middleware_handlers if filter_type in [event_type, 'all'])

    def __call__(self, event: dict, context: Union[dict, str]) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        self.lambda_context = context
        handler = RestAPIEventHandler(self.routes, self.api, self.log, self.debug, middleware_handlers=self._get_middleware_handlers('http'))
        self.current_request = handler.create_request_object(event, context)
        return handler(event, context)

class BuiltinAuthConfig(object):

    def __init__(self, name, handler_string, ttl_seconds=None, execution_role=None, header='Authorization') -> None:
        self.name = name
        self.handler_string = handler_string
        self.ttl_seconds = ttl_seconds
        self.execution_role = execution_role
        self.header = header

class ChaliceAuthorizer(object):

    def __init__(self, name, func, scopes=None) -> None:
        self.name = name
        self.func = func
        self.scopes = scopes or []
        self.config = None

    def __call__(self, event: dict, context: Union[dict, str]) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        auth_request = self._transform_event(event)
        result = self.func(auth_request)
        if isinstance(result, AuthResponse):
            return result.to_dict(auth_request)
        return result

    def _transform_event(self, event: Union[dict, dict[str, typing.Any]]) -> AuthRequest:
        return AuthRequest(event['type'], event['authorizationToken'], event['methodArn'])

    def with_scopes(self, scopes: Union[typing.Mapping, dict, dict[str, typing.Any]]) -> None:
        authorizer_with_scopes = copy.deepcopy(self)
        authorizer_with_scopes.scopes = scopes
        return authorizer_with_scopes

class AuthRequest(object):

    def __init__(self, auth_type, token, method_arn) -> None:
        self.auth_type = auth_type
        self.token = token
        self.method_arn = method_arn

class AuthResponse(object):
    ALL_HTTP_METHODS = ['DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'POST', 'PUT', 'GET']

    def __init__(self, routes, principal_id, context=None) -> None:
        self.routes = routes
        self.principal_id = principal_id
        if context is None:
            context = {}
        self.context = context

    def to_dict(self, request) -> dict[typing.Text, typing.Union[str,bytes,dict[str, str]]]:
        return {'context': self.context, 'principalId': self.principal_id, 'policyDocument': self._generate_policy(request)}

    def _generate_policy(self, request: dict[str, typing.Any]) -> dict[typing.Text, typing.Union[typing.Text,list[dict[typing.Text, typing.Union[typing.Text,bool,set[str]]]]]]:
        allowed_resources = self._generate_allowed_resources(request)
        return {'Version': '2012-10-17', 'Statement': [{'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': allowed_resources}]}

    def _generate_allowed_resources(self, request: tuple[str]) -> list:
        allowed_resources = []
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

    def _generate_arn(self, route: str, request: Union[str, bytes], method: typing.Text='*') -> str:
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

    def __init__(self, path, methods) -> None:
        self.path = path
        self.methods = methods

class LambdaFunction(object):

    def __init__(self, func, name, handler_string) -> None:
        self.func = func
        self.name = name
        self.handler_string = handler_string

    def __call__(self, event: dict, context: Union[dict, str]) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        return self.func(event, context)

class BaseEventSourceConfig(object):

    def __init__(self, name, handler_string) -> None:
        self.name = name
        self.handler_string = handler_string

class ScheduledEventConfig(BaseEventSourceConfig):

    def __init__(self, name, handler_string, schedule_expression, description) -> None:
        super(ScheduledEventConfig, self).__init__(name, handler_string)
        self.schedule_expression = schedule_expression
        self.description = description

class CloudWatchEventConfig(BaseEventSourceConfig):

    def __init__(self, name, handler_string, event_pattern) -> None:
        super(CloudWatchEventConfig, self).__init__(name, handler_string)
        self.event_pattern = event_pattern

class ScheduleExpression(object):

    def to_string(self) -> None:
        raise NotImplementedError('to_string')

class Rate(ScheduleExpression):
    MINUTES = 'MINUTES'
    HOURS = 'HOURS'
    DAYS = 'DAYS'

    def __init__(self, value, unit) -> None:
        self.value = value
        self.unit = unit

    def to_string(self) -> None:
        unit = self.unit.lower()
        if self.value == 1:
            unit = unit[:-1]
        return 'rate(%s %s)' % (self.value, unit)

class Cron(ScheduleExpression):

    def __init__(self, minutes, hours, day_of_month, month, day_of_week, year) -> None:
        self.minutes = minutes
        self.hours = hours
        self.day_of_month = day_of_month
        self.month = month
        self.day_of_week = day_of_week
        self.year = year

    def to_string(self) -> None:
        return 'cron(%s %s %s %s %s %s)' % (self.minutes, self.hours, self.day_of_month, self.month, self.day_of_week, self.year)

class S3EventConfig(BaseEventSourceConfig):

    def __init__(self, name, bucket, events, prefix, suffix, handler_string) -> None:
        super(S3EventConfig, self).__init__(name, handler_string)
        self.bucket = bucket
        self.events = events
        self.prefix = prefix
        self.suffix = suffix

class SNSEventConfig(BaseEventSourceConfig):

    def __init__(self, name, handler_string, topic) -> None:
        super(SNSEventConfig, self).__init__(name, handler_string)
        self.topic = topic

class SQSEventConfig(BaseEventSourceConfig):

    def __init__(self, name, handler_string, queue, queue_arn, batch_size, maximum_batching_window_in_seconds, maximum_concurrency) -> None:
        super(SQSEventConfig, self).__init__(name, handler_string)
        self.queue = queue
        self.queue_arn = queue_arn
        self.batch_size = batch_size
        self.maximum_batching_window_in_seconds = maximum_batching_window_in_seconds
        self.maximum_concurrency = maximum_concurrency

class KinesisEventConfig(BaseEventSourceConfig):

    def __init__(self, name, handler_string, stream, batch_size, starting_position, maximum_batching_window_in_seconds) -> None:
        super(KinesisEventConfig, self).__init__(name, handler_string)
        self.stream = stream
        self.batch_size = batch_size
        self.starting_position = starting_position
        self.maximum_batching_window_in_seconds = maximum_batching_window_in_seconds

class DynamoDBEventConfig(BaseEventSourceConfig):

    def __init__(self, name, handler_string, stream_arn, batch_size, starting_position, maximum_batching_window_in_seconds) -> None:
        super(DynamoDBEventConfig, self).__init__(name, handler_string)
        self.stream_arn = stream_arn
        self.batch_size = batch_size
        self.starting_position = starting_position
        self.maximum_batching_window_in_seconds = maximum_batching_window_in_seconds

class WebsocketConnectConfig(BaseEventSourceConfig):
    CONNECT_ROUTE = '$connect'

    def __init__(self, name, handler_string, user_handler) -> None:
        super(WebsocketConnectConfig, self).__init__(name, handler_string)
        self.route_key_handled = self.CONNECT_ROUTE
        self.handler_function = user_handler

class WebsocketMessageConfig(BaseEventSourceConfig):

    def __init__(self, name, route_key_handled, handler_string, user_handler) -> None:
        super(WebsocketMessageConfig, self).__init__(name, handler_string)
        self.route_key_handled = route_key_handled
        self.handler_function = user_handler

class WebsocketDisconnectConfig(BaseEventSourceConfig):
    DISCONNECT_ROUTE = '$disconnect'

    def __init__(self, name, handler_string, user_handler) -> None:
        super(WebsocketDisconnectConfig, self).__init__(name, handler_string)
        self.route_key_handled = self.DISCONNECT_ROUTE
        self.handler_function = user_handler

class PureLambdaWrapper(object):

    def __init__(self, original_func) -> None:
        self._original_func = original_func

    def __call__(self, event: dict) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        return self._original_func(event.to_dict(), event.context)

class MiddlewareHandler(object):

    def __init__(self, handler, next_handler) -> None:
        self.handler = handler
        self.next_handler = next_handler

    def __call__(self, request) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        return self.handler(request, self.next_handler)

class BaseLambdaHandler(object):

    def __call__(self, event: dict, context: Union[dict, str]) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        pass

    def _build_middleware_handlers(self, handlers: list[typing.Callable], original_handler: typing.Callable) -> Union[MiddlewareHandler, typing.Callable]:
        current = original_handler
        for handler in reversed(list(handlers)):
            current = MiddlewareHandler(handler=handler, next_handler=current)
        return current

class EventSourceHandler(BaseLambdaHandler):

    def __init__(self, func, event_class, middleware_handlers=None) -> None:
        self.func = func
        self.event_class = event_class
        if middleware_handlers is None:
            middleware_handlers = []
        self._middleware_handlers = middleware_handlers
        self.handler = None

    @property
    def middleware_handlers(self) -> Union[dict[str, typing.Callable], logging.handlers.QueueHandler, int]:
        return self._middleware_handlers

    @middleware_handlers.setter
    def middleware_handlers(self, value) -> Union[dict[str, typing.Callable], logging.handlers.QueueHandler, int]:
        self._middleware_handlers = value

    def __call__(self, event: dict, context: Union[dict, str]) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        event_obj = self.event_class(event, context)
        if self.handler is None:
            self.handler = self._build_middleware_handlers(self._middleware_handlers, original_handler=self.func)
        return self.handler(event_obj)

class WebsocketEventSourceHandler(EventSourceHandler):
    WEBSOCKET_API_RESPONSE = {'statusCode': 200}

    def __init__(self, func, event_class, websocket_api, middleware_handlers=None) -> None:
        super(WebsocketEventSourceHandler, self).__init__(func, event_class, middleware_handlers)
        self.websocket_api = websocket_api

    def __call__(self, event: dict, context: Union[dict, str]) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
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

    def __init__(self, route_table, api, log, debug, middleware_handlers=None) -> None:
        self.routes = route_table
        self.api = api
        self.log = log
        self.debug = debug
        self.current_request = None
        self.lambda_context = None
        if middleware_handlers is None:
            middleware_handlers = []
        self._middleware_handlers = middleware_handlers

    def _global_error_handler(self, event: dict[str, typing.Any], get_response: dict[str, typing.Any]) -> str:
        try:
            return get_response(event)
        except Exception:
            return self._unhandled_exception_to_response()

    def create_request_object(self, event: dict, context: Union[dict, dict[str, typing.Any]]) -> Union[Request, None]:
        resource_path = event.get('requestContext', {}).get('resourcePath')
        if resource_path is not None:
            self.current_request = Request(event, context)
            return self.current_request
        return None

    def __call__(self, event: dict, context: Union[dict, str]) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:

        def wrapped_event(request: Any):
            return self._main_rest_api_handler(event, context)
        final_handler = self._build_middleware_handlers([self._global_error_handler] + list(self._middleware_handlers), original_handler=wrapped_event)
        response = final_handler(self.current_request)
        return response.to_dict(self.api.binary_types)

    def _main_rest_api_handler(self, event: Union[dict, str], context: Union[dict, dict[str, typing.Any]]) -> Union[tuple[typing.Union[str,dict[str, str]]], dict[str, typing.Any]]:
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
        cors_headers = None
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

    def _validate_binary_response(self, request_headers: Union[dict[str, str], dict], response_headers: Union[dict[str, str], dict]) -> bool:
        request_accept_header = request_headers.get('accept')
        response_content_type = response_headers.get('content-type', 'application/json')
        response_is_binary = _matches_content_type(response_content_type, self.api.binary_types)
        expects_binary_response = False
        if request_accept_header is not None:
            expects_binary_response = _matches_content_type(request_accept_header, self.api.binary_types)
        if response_is_binary and (not expects_binary_response):
            return False
        return True

    def _get_view_function_response(self, view_function: typing.Callable, function_args: typing.Callable) -> Response:
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
        headers = {}
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

    def _validate_response(self, response: Any) -> None:
        for header, value in response.headers.items():
            if '\n' in value:
                raise ChaliceError("Bad value for header '%s': %r" % (header, value))

    def _cors_enabled_for_route(self, route_entry: dict) -> bool:
        return route_entry.cors is not None

    def _get_cors_headers(self, cors: Union[str, dict, list[tuple[typing.Any]]]) -> dict:
        return cors.get_access_control_headers()

    def _add_cors_headers(self, response: Union[dict[str, typing.Any], dict], cors_headers: Union[dict[str, str], dict]) -> None:
        for name, value in cors_headers.items():
            if name not in response.headers:
                response.headers[name] = value

class BaseLambdaEvent(object):

    def __init__(self, event_dict, context) -> None:
        self._event_dict = event_dict
        self.context = context
        self._extract_attributes(event_dict)

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        raise NotImplementedError('_extract_attributes')

    def to_dict(self) -> dict[typing.Text, typing.Union[str,bytes,dict[str, str]]]:
        return self._event_dict

class LambdaFunctionEvent(BaseLambdaEvent):

    def __init__(self, event_dict, context) -> None:
        self.event = event_dict
        self.context = context

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        pass

    def to_dict(self) -> dict[typing.Text, typing.Union[str,bytes,dict[str, str]]]:
        return self.event

class CloudWatchEvent(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
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

    def __init__(self, event_dict, context) -> None:
        super(WebsocketEvent, self).__init__(event_dict, context)
        self._json_body = None

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        request_context = event_dict['requestContext']
        self.domain_name = request_context['domainName']
        self.stage = request_context['stage']
        self.connection_id = request_context['connectionId']
        self.body = str(event_dict.get('body'))

    @property
    def json_body(self) -> Union[str, dict[str, tuple[typing.Union[typing.Any,str]]], None]:
        if self._json_body is None:
            try:
                self._json_body = json.loads(self.body)
            except ValueError:
                raise BadRequestError('Error Parsing JSON')
        return self._json_body

class SNSEvent(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        first_record = event_dict['Records'][0]
        self.message = first_record['Sns']['Message']
        self.subject = first_record['Sns']['Subject']
        self.message_attributes = first_record['Sns']['MessageAttributes']

class S3Event(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        s3 = event_dict['Records'][0]['s3']
        self.bucket = s3['bucket']['name']
        self.key = unquote_plus(s3['object']['key'])

class SQSEvent(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        pass

    def __iter__(self) -> typing.Generator[SQSRecord]:
        for record in self._event_dict['Records']:
            yield SQSRecord(record, self.context)

class SQSRecord(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        self.body = event_dict['body']
        self.receipt_handle = event_dict['receiptHandle']

class KinesisEvent(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        pass

    def __iter__(self) -> typing.Generator[SQSRecord]:
        for record in self._event_dict['Records']:
            yield KinesisRecord(record, self.context)

class KinesisRecord(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        kinesis = event_dict['kinesis']
        encoded_payload = kinesis['data']
        self.data = base64.b64decode(encoded_payload)
        self.sequence_number = kinesis['sequenceNumber']
        self.partition_key = kinesis['partitionKey']
        self.schema_version = kinesis['kinesisSchemaVersion']
        self.timestamp = datetime.datetime.utcfromtimestamp(kinesis['approximateArrivalTimestamp'])

class DynamoDBEvent(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
        pass

    def __iter__(self) -> typing.Generator[SQSRecord]:
        for record in self._event_dict['Records']:
            yield DynamoDBRecord(record, self.context)

class DynamoDBRecord(BaseLambdaEvent):

    def _extract_attributes(self, event_dict: Union[dict, dict[str, typing.Any], dict[str, str]]) -> None:
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
    def table_name(self) -> Union[typing.Text, str]:
        parts = self.event_source_arn.split(':', 5)
        if not len(parts) == 6:
            return ''
        full_name = parts[-1]
        name_parts = full_name.split('/')
        if len(name_parts) >= 2:
            return name_parts[1]
        return ''

class Blueprint(DecoratorAPI):

    def __init__(self, import_name) -> None:
        self._import_name = import_name
        self._deferred_registrations = []
        self._current_app = None
        self._lambda_context = None

    @property
    def log(self):
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.log if it's registered to an app.")
        return self._current_app.log

    @property
    def current_request(self):
        if self._current_app is None or self._current_app.current_request is None:
            raise RuntimeError("Can only access Blueprint.current_request if it's registered to an app.")
        return self._current_app.current_request

    @property
    def current_app(self) -> None:
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.current_app if it's registered to an app.")
        return self._current_app

    @property
    def lambda_context(self):
        if self._current_app is None:
            raise RuntimeError("Can only access Blueprint.lambda_context if it's registered to an app.")
        return self._current_app.lambda_context

    def register(self, app: Any, options: Any) -> None:
        self._current_app = app
        all_options = options.copy()
        all_options['module_name'] = self._import_name
        for function in self._deferred_registrations:
            function(app, all_options)

    def register_middleware(self, func: Union[typing.Sized, str], event_type: typing.Text='all') -> None:
        self._deferred_registrations.append(lambda app, options: app.register_middleware(func, event_type))

    def _register_handler(self, handler_type, name, user_handler: str, wrapped_handler, kwargs, options=None):

        def _register_blueprint_handler(app: Any, options: Any) -> None:
            if handler_type in _EVENT_CLASSES:
                wrapped_handler.middleware_handlers = app._get_middleware_handlers(_MIDDLEWARE_MAPPING[handler_type])
            app._register_handler(handler_type, name, user_handler, wrapped_handler, kwargs, options)
        self._deferred_registrations.append(_register_blueprint_handler)

    def _get_middleware_handlers(self, event_type: Union[str, dict[str, typing.Any], None, list]) -> None:
        return []

class ConvertToMiddleware(object):

    def __init__(self, lambda_wrapper) -> None:
        self._wrapper = lambda_wrapper

    def __call__(self, event: dict, get_response) -> Union[None, dict, dict[, dict], dict[str, typing.Any], typing.Generator[typing.Optional[typing.Any]], typing.Mapping]:
        original_event, context = self._extract_original_param(event)

        @functools.wraps(self._wrapper)
        def wrapped(original_event: Any, context: Any):
            return get_response(event)
        return self._wrapper(wrapped)(original_event, context)

    def _extract_original_param(self, event: Union[dict, dict[str, typing.Any]]) -> tuple:
        if isinstance(event, Request):
            return (event.to_original_event(), event.lambda_context)
        return (event.to_dict(), event.context)
_EVENT_CLASSES = {'on_s3_event': S3Event, 'on_sns_message': SNSEvent, 'on_sqs_message': SQSEvent, 'on_cw_event': CloudWatchEvent, 'on_kinesis_record': KinesisEvent, 'on_dynamodb_record': DynamoDBEvent, 'schedule': CloudWatchEvent, 'lambda_function': LambdaFunctionEvent}
_MIDDLEWARE_MAPPING = {'on_s3_event': 's3', 'on_sns_message': 'sns', 'on_sqs_message': 'sqs', 'on_cw_event': 'cloudwatch', 'on_kinesis_record': 'kinesis', 'on_dynamodb_record': 'dynamodb', 'schedule': 'scheduled', 'lambda_function': 'pure_lambda'}