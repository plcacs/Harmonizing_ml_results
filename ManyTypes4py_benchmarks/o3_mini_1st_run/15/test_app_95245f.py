#!/usr/bin/env python3
from __future__ import annotations
import sys
import base64
import logging
import json
import gzip
import inspect
import collections
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
import pytest
from pytest import fixture
import hypothesis.strategies as st
from hypothesis import given, assume
import six
from chalice import app
from chalice import NotFoundError
from chalice.test import Client
from chalice.app import APIGateway, Request, Response, handle_extra_types, MultiDict, WebsocketEvent, BadRequestError, WebsocketDisconnectedError, WebsocketEventSourceHandler, ConvertToMiddleware, WebsocketAPI, ChaliceUnhandledError
from chalice import __version__ as chalice_version
from chalice.deploy.validate import ExperimentalFeatureError
from chalice.deploy.validate import validate_feature_flags

STR_MAP = st.dictionaries(st.text(), st.text())
STR_TO_LIST_MAP = st.dictionaries(st.text(), st.lists(elements=st.text(), min_size=1, max_size=5))
HTTP_METHOD = st.sampled_from(['GET', 'POST', 'PUT', 'PATCH', 'OPTIONS', 'HEAD', 'DELETE'])
PATHS = st.sampled_from(['/', '/foo/bar'])
HTTP_BODY = st.none() | st.text()
HTTP_REQUEST = st.fixed_dictionaries({
    'query_params': STR_TO_LIST_MAP,
    'headers': STR_MAP,
    'uri_params': STR_MAP,
    'method': HTTP_METHOD,
    'body': HTTP_BODY,
    'context': STR_MAP,
    'stage_vars': STR_MAP,
    'is_base64_encoded': st.booleans(),
    'path': PATHS})
HTTP_REQUEST = st.fixed_dictionaries({
    'multiValueQueryStringParameters': st.fixed_dictionaries({}),
    'headers': STR_MAP,
    'pathParameters': STR_MAP,
    'requestContext': st.fixed_dictionaries({'httpMethod': HTTP_METHOD, 'resourcePath': PATHS}),
    'body': HTTP_BODY,
    'stageVariables': STR_MAP,
    'isBase64Encoded': st.booleans()})
BINARY_TYPES = APIGateway().binary_types


class FakeLambdaContextIdentity:
    def __init__(self, cognito_identity_id: str, cognito_identity_pool_id: str) -> None:
        self.cognito_identity_id: str = cognito_identity_id
        self.cognito_identity_pool_id: str = cognito_identity_pool_id


class FakeLambdaContext:
    def __init__(self) -> None:
        self.function_name: str = 'test_name'
        self.function_version: str = 'version'
        self.invoked_function_arn: str = 'arn'
        self.memory_limit_in_mb: int = 256
        self.aws_request_id: str = 'id'
        self.log_group_name: str = 'log_group_name'
        self.log_stream_name: str = 'log_stream_name'
        self.identity: FakeLambdaContextIdentity = FakeLambdaContextIdentity('id', 'id_pool')
        self.client_context: Optional[Any] = None

    def get_remaining_time_in_millis(self) -> int:
        return 500

    def serialize(self) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {}
        serialized.update(vars(self))
        serialized['identity'] = vars(self.identity)
        return serialized


class FakeGoneException(Exception):
    pass


class FakeExceptionFactory:
    def __init__(self) -> None:
        self.GoneException = FakeGoneException


class FakeClient:
    def __init__(self, errors: Optional[List[Exception]] = None, infos: Optional[List[Any]] = None) -> None:
        if errors is None:
            errors = []
        if infos is None:
            infos = []
        self._errors: List[Exception] = errors
        self._infos: List[Any] = infos
        self.calls: Dict[str, List[tuple]] = collections.defaultdict(lambda: [])
        self.exceptions: FakeExceptionFactory = FakeExceptionFactory()

    def post_to_connection(self, ConnectionId: str, Data: Any) -> None:
        self._call('post_to_connection', ConnectionId, Data)

    def delete_connection(self, ConnectionId: str) -> None:
        self._call('close', ConnectionId)

    def get_connection(self, ConnectionId: str) -> Any:
        self._call('info', ConnectionId)
        if self._infos is not None:
            info = self._infos.pop()
            return info

    def _call(self, name: str, *args: Any) -> None:
        self.calls[name].append(tuple(args))
        if self._errors:
            error: Exception = self._errors.pop()
            raise error


class FakeSession:
    def __init__(self, client: Optional[FakeClient] = None, region_name: str = 'us-west-2') -> None:
        self.calls: List[tuple] = []
        self._client: Optional[FakeClient] = client
        self.region_name: str = region_name

    def client(self, name: str, endpoint_url: Optional[str] = None) -> Any:
        self.calls.append((name, endpoint_url))
        return self._client


@fixture
def view_function() -> Callable[[], Dict[str, str]]:
    def _func() -> Dict[str, str]:
        return {'hello': 'world'}
    return _func  # type: ignore


def create_request_with_content_type(content_type: str) -> Request:
    body: str = '{"json": "body"}'
    event: Dict[str, Any] = {
        'multiValueQueryStringParameters': '',
        'headers': {'Content-Type': content_type},
        'pathParameters': {},
        'requestContext': {'httpMethod': 'GET', 'resourcePath': '/'},
        'body': body,
        'stageVariables': {},
        'isBase64Encoded': False
    }
    return app.Request(event, FakeLambdaContext())


def assert_response_body_is(response: Dict[str, Any], body: Any) -> None:
    assert json.loads(response['body']) == body


def json_response_body(response: Dict[str, Any]) -> Any:
    return json.loads(response['body'])


def assert_requires_opt_in(app_obj: app.Chalice, flag: str) -> None:
    with pytest.raises(ExperimentalFeatureError):
        validate_feature_flags(app_obj)
    app_obj.experimental_feature_flags.add(flag)
    try:
        validate_feature_flags(app_obj)
    except ExperimentalFeatureError:
        raise AssertionError('Opting in to feature %s still raises an ExperimentalFeatureError.' % flag)


def websocket_handler_for_route(route: str, app_obj: app.Chalice) -> WebsocketEventSourceHandler:
    fn: Callable = app_obj.websocket_handlers[route].handler_function
    handler: WebsocketEventSourceHandler = WebsocketEventSourceHandler(fn, WebsocketEvent, app_obj.websocket_api)
    return handler


@fixture
def sample_app() -> app.Chalice:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/index', methods=['GET'])
    def index() -> Dict[str, str]:
        return {'hello': 'world'}

    @demo.route('/name/{name}', methods=['GET'])
    def name(name: str) -> Dict[str, str]:
        return {'provided-name': name}
    return demo


@fixture
def sample_app_with_cors() -> app.Chalice:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/image', methods=['POST'], cors=True, content_types=['image/gif'])
    def image() -> Dict[str, bool]:
        return {'image': True}
    return demo


@fixture
def sample_app_with_default_cors() -> app.Chalice:
    demo: app.Chalice = app.Chalice('demo-app')
    demo.api.cors = True

    @demo.route('/on', methods=['POST'], content_types=['image/gif'])
    def on() -> Dict[str, bool]:
        return {'image': True}

    @demo.route('/off', methods=['POST'], cors=False, content_types=['image/gif'])
    def off() -> Dict[str, bool]:
        return {'image': True}

    @demo.route('/default', methods=['POST'], cors=None, content_types=['image/gif'])
    def default() -> Dict[str, bool]:
        return {'image': True}
    return demo


@fixture
def sample_websocket_app() -> tuple[app.Chalice, List[tuple[str, WebsocketEvent]]]:
    demo: app.Chalice = app.Chalice('app-name')
    demo.websocket_api.session = FakeSession()
    calls: List[tuple[str, WebsocketEvent]] = []

    @demo.on_ws_connect()
    def connect(event: WebsocketEvent) -> None:
        demo.websocket_api.send(event.connection_id, 'connected')
        calls.append(('connect', event))

    @demo.on_ws_disconnect()
    def disconnect(event: WebsocketEvent) -> None:
        demo.websocket_api.send(event.connection_id, 'message')
        calls.append(('disconnect', event))

    @demo.on_ws_message()
    def message(event: WebsocketEvent) -> None:
        demo.websocket_api.send(event.connection_id, 'disconnected')
        calls.append(('default', event))
    return (demo, calls)


@fixture
def sample_middleware_app() -> app.Chalice:
    demo: app.Chalice = app.Chalice('app-name')
    demo.calls = []  # type: ignore

    @demo.middleware('all')
    def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
        demo.calls.append({'type': 'all', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('s3')
    def mymiddleware_s3(event: Any, get_response: Callable[[Any], Any]) -> Any:
        demo.calls.append({'type': 's3', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('sns')
    def mymiddleware_sns(event: Any, get_response: Callable[[Any], Any]) -> Any:
        demo.calls.append({'type': 'sns', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('http')
    def mymiddleware_http(event: Any, get_response: Callable[[Any], Any]) -> Any:
        demo.calls.append({'type': 'http', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('websocket')
    def mymiddleware_websocket(event: Any, get_response: Callable[[Any], Any]) -> Any:
        demo.calls.append({'type': 'websocket', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('pure_lambda')
    def mymiddleware_pure_lambda(event: Any, get_response: Callable[[Any], Any]) -> Any:
        demo.calls.append({'type': 'pure_lambda', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.route('/')
    def index() -> Dict:
        return {}

    @demo.on_s3_event(bucket='foo')
    def s3_handler(event: Any) -> None:
        pass

    @demo.on_sns_message(topic='foo')
    def sns_handler(event: Any) -> None:
        pass

    @demo.on_sqs_message(queue='foo')
    def sqs_handler(event: Any) -> None:
        pass

    @demo.lambda_function()
    def lambda_handler(event: Any, context: Any) -> None:
        pass

    @demo.on_ws_message()
    def ws_handler(event: WebsocketEvent) -> None:
        pass
    return demo


@fixture
def auth_request() -> app.AuthRequest:
    method_arn: str = 'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/needs/auth'
    request: app.AuthRequest = app.AuthRequest('TOKEN', 'authtoken', method_arn)
    return request


@pytest.mark.skipif(sys.version[0] == '2', reason='Test is irrelevant under python 2, since str and bytes are interchangeable.')
def test_invalid_binary_response_body_throws_value_error(sample_app: app.Chalice) -> None:
    response: Response = app.Response(status_code=200, body={'foo': 'bar'}, headers={'Content-Type': 'application/octet-stream'})
    with pytest.raises(ValueError):
        response.to_dict(sample_app.api.binary_types)


def test_invalid_JSON_response_body_throws_type_error(sample_app: app.Chalice) -> None:
    response: Response = app.Response(status_code=200, body={'foo': object()}, headers={'Content-Type': 'application/json'})
    with pytest.raises(TypeError):
        response.to_dict()


def test_can_encode_binary_body_as_base64(sample_app: app.Chalice) -> None:
    response: Response = app.Response(status_code=200, body=b'foobar', headers={'Content-Type': 'application/octet-stream'})
    encoded_response: Dict[str, Any] = response.to_dict(sample_app.api.binary_types)
    assert encoded_response['body'] == 'Zm9vYmFy'


def test_can_return_unicode_body(sample_app: app.Chalice) -> None:
    unicode_data: str = u'✓'
    response: Response = app.Response(status_code=200, body=unicode_data)
    encoded_response: Dict[str, Any] = response.to_dict()
    assert encoded_response['body'] == unicode_data


def test_can_encode_binary_body_with_header_charset(sample_app: app.Chalice) -> None:
    response: Response = app.Response(status_code=200, body=b'foobar', headers={'Content-Type': 'application/octet-stream; charset=binary'})
    encoded_response: Dict[str, Any] = response.to_dict(sample_app.api.binary_types)
    assert encoded_response['body'] == 'Zm9vYmFy'


def test_can_encode_binary_json(sample_app: app.Chalice) -> None:
    sample_app.api.binary_types.extend(['application/json'])
    response: Response = app.Response(status_code=200, body={'foo': 'bar'}, headers={'Content-Type': 'application/json'})
    encoded_response: Dict[str, Any] = response.to_dict(sample_app.api.binary_types)
    assert encoded_response['body'] == 'eyJmb28iOiJiYXIifQ=='


def test_wildcard_accepts_with_native_python_types_serializes_json(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    sample_app.api.binary_types = ['*/*']

    @sample_app.route('/py-dict')
    def py_dict() -> Dict[str, str]:
        return {'foo': 'bar'}
    event: Dict[str, Any] = create_event('/py-dict', 'GET', {})
    event['headers']['Accept'] = '*/*'
    response: Dict[str, Any] = sample_app(event, context=None)
    assert base64.b64decode(response['body']) == b'{"foo":"bar"}'
    assert response['isBase64Encoded']


def test_wildcard_accepts_with_response_class(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    sample_app.api.binary_types = ['*/*']

    @sample_app.route('/py-dict')
    def py_dict() -> Response:
        return Response(body=json.dumps({'foo': 'bar'}).encode('utf-8'), headers={'Content-Type': 'application/json'}, status_code=200)
    event: Dict[str, Any] = create_event('/py-dict', 'GET', {})
    event['headers']['Accept'] = '*/*'
    response: Dict[str, Any] = sample_app(event, context=None)
    assert base64.b64decode(response['body']) == b'{"foo": "bar"}'
    assert response['isBase64Encoded']


def test_can_parse_route_view_args() -> None:
    entry: app.RouteEntry = app.RouteEntry(lambda: {'foo': 'bar'}, 'view-name', '/foo/{bar}/baz/{qux}', method='GET')
    assert entry.view_args == ['bar', 'qux']


def test_can_route_single_view() -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Dict:
        return {}
    assert demo.routes['/index']['GET'] == app.RouteEntry(index_view, 'index_view', '/index', 'GET', content_types=['application/json'])


def test_can_handle_multiple_routes() -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Dict:
        return {}

    @demo.route('/other')
    def other_view() -> Dict:
        return {}
    assert len(demo.routes) == 2, demo.routes
    assert '/index' in demo.routes, demo.routes
    assert '/other' in demo.routes, demo.routes
    assert demo.routes['/index']['GET'].view_function == index_view
    assert demo.routes['/other']['GET'].view_function == other_view


def test_error_on_unknown_event(sample_app: app.Chalice) -> None:
    bad_event: Dict[str, Any] = {'random': 'event'}
    raw_response: Dict[str, Any] = sample_app(bad_event, context=None)
    assert raw_response['statusCode'] == 500
    assert json_response_body(raw_response)['Code'] == 'InternalServerError'


def test_can_route_api_call_to_view_function(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    response: Dict[str, Any] = sample_app(event, context=None)
    assert_response_body_is(response, {'hello': 'world'})


def test_can_call_to_dict_on_current_request(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    @sample_app.route('/todict')
    def todict() -> Dict[str, Any]:
        return sample_app.current_request.to_dict()
    event: Dict[str, Any] = create_event('/todict', 'GET', {})
    response: Dict[str, Any] = json_response_body(sample_app(event, context=None))
    assert isinstance(response, dict)
    assert response['method'] == 'GET'
    assert isinstance(json.loads(json.dumps(response)), dict)


def test_can_call_to_dict_on_request_with_querystring(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    @sample_app.route('/todict')
    def todict() -> Dict[str, Any]:
        return sample_app.current_request.to_dict()
    event: Dict[str, Any] = create_event('/todict', 'GET', {})
    event['multiValueQueryStringParameters'] = {'key': ['val1', 'val2'], 'key2': ['val']}
    response: Dict[str, Any] = json_response_body(sample_app(event, context=None))
    assert isinstance(response, dict)
    assert response['method'] == 'GET'
    assert response['query_params'] is not None
    assert response['query_params']['key'] == 'val2'
    assert response['query_params']['key2'] == 'val'
    assert isinstance(json.loads(json.dumps(response)), dict)


def test_request_to_dict_does_not_contain_internal_attrs(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    @sample_app.route('/todict')
    def todict() -> Dict[str, Any]:
        return sample_app.current_request.to_dict()
    event: Dict[str, Any] = create_event('/todict', 'GET', {})
    response: Dict[str, Any] = json_response_body(sample_app(event, context=None))
    internal_attrs: List[str] = [key for key in response if key.startswith('_')]
    assert not internal_attrs


def test_will_pass_captured_params_to_view(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    event: Dict[str, Any] = create_event('/name/{name}', 'GET', {'name': 'james'})
    response: Dict[str, Any] = sample_app(event, context=None)
    response = json_response_body(response)
    assert response == {'provided-name': 'james'}


def test_error_on_unsupported_method(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    event: Dict[str, Any] = create_event('/name/{name}', 'POST', {'name': 'james'})
    raw_response: Dict[str, Any] = sample_app(event, context=None)
    assert raw_response['statusCode'] == 405
    assert raw_response['headers']['Allow'] == 'GET'
    assert json_response_body(raw_response)['Code'] == 'MethodNotAllowedError'


def test_error_on_unsupported_method_gives_feedback_on_method(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    method: str = 'POST'
    event: Dict[str, Any] = create_event('/name/{name}', method, {'name': 'james'})
    raw_response: Dict[str, Any] = sample_app(event, context=None)
    assert 'POST' in json_response_body(raw_response)['Message']


def test_error_contains_cors_headers(sample_app_with_cors: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    event: Dict[str, Any] = create_event('/image', 'POST', {'not': 'image'})
    raw_response: Dict[str, Any] = sample_app_with_cors(event, context=None)
    assert raw_response['statusCode'] == 415
    assert 'Access-Control-Allow-Origin' in raw_response['headers']


class TestDefaultCORS:
    def test_cors_enabled(self, sample_app_with_default_cors: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
        event: Dict[str, Any] = create_event('/on', 'POST', {'not': 'image'})
        raw_response: Dict[str, Any] = sample_app_with_default_cors(event, context=None)
        assert raw_response['statusCode'] == 415
        assert 'Access-Control-Allow-Origin' in raw_response['headers']

    def test_cors_none(self, sample_app_with_default_cors: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
        event: Dict[str, Any] = create_event('/default', 'POST', {'not': 'image'})
        raw_response: Dict[str, Any] = sample_app_with_default_cors(event, context=None)
        assert raw_response['statusCode'] == 415
        assert 'Access-Control-Allow-Origin' in raw_response['headers']

    def test_cors_disabled(self, sample_app_with_default_cors: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
        event: Dict[str, Any] = create_event('/off', 'POST', {'not': 'image'})
        raw_response: Dict[str, Any] = sample_app_with_default_cors(event, context=None)
        assert raw_response['statusCode'] == 415
        assert 'Access-Control-Allow-Origin' not in raw_response['headers']


def test_can_access_context(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Dict[str, Any]:
        serialized: Dict[str, Any] = demo.lambda_context.serialize()
        return serialized
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    lambda_context: FakeLambdaContext = FakeLambdaContext()
    result: Dict[str, Any] = demo(event, lambda_context)
    result = json_response_body(result)
    serialized_lambda_context: Dict[str, Any] = lambda_context.serialize()
    assert result == serialized_lambda_context


def test_can_access_raw_body(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Dict[str, Any]:
        return {'rawbody': demo.current_request.raw_body.decode('utf-8')}
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    event['body'] = '{"hello": "world"}'
    result: Dict[str, Any] = demo(event, context=None)
    result = json_response_body(result)
    assert result == {'rawbody': '{"hello": "world"}'}


def test_raw_body_cache_returns_same_result(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Dict[str, Any]:
        return {'rawbody': demo.current_request.raw_body.decode('utf-8'),
                'rawbody2': demo.current_request.raw_body.decode('utf-8')}
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    event['base64-body'] = base64.b64encode(b'{"hello": "world"}').decode('ascii')
    result: Dict[str, Any] = demo(event, context=None)
    result = json_response_body(result)
    assert result['rawbody'] == result['rawbody2']


def test_can_have_views_of_same_route_but_different_methods(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index', methods=['GET'])
    def get_view() -> Dict[str, str]:
        return {'method': 'GET'}

    @demo.route('/index', methods=['PUT'])
    def put_view() -> Dict[str, str]:
        return {'method': 'PUT'}
    assert demo.routes['/index']['GET'].view_function == get_view
    assert demo.routes['/index']['PUT'].view_function == put_view
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    result: Dict[str, Any] = demo(event, context=None)
    assert json_response_body(result) == {'method': 'GET'}
    event = create_event('/index', 'PUT', {})
    result = demo(event, context=None)
    assert json_response_body(result) == {'method': 'PUT'}


def test_error_on_duplicate_route_methods() -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index', methods=['PUT'])
    def index_view() -> Dict[str, str]:
        return {'foo': 'bar'}
    with pytest.raises(ValueError):
        @demo.route('/index', methods=['PUT'])
        def index_view_dup() -> Dict[str, str]:
            return {'foo': 'bar'}


def test_json_body_available_with_right_content_type(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', methods=['POST'])
    def index() -> Any:
        return demo.current_request.json_body
    event: Dict[str, Any] = create_event('/', 'POST', {})
    event['body'] = json.dumps({'foo': 'bar'})
    result: Any = demo(event, context=None)
    result = json_response_body(result)
    assert result == {'foo': 'bar'}


def test_json_body_none_with_malformed_json(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', methods=['POST'])
    def index() -> Any:
        return demo.current_request.json_body
    event: Dict[str, Any] = create_event('/', 'POST', {})
    event['body'] = '{"foo": "bar"'
    result: Dict[str, Any] = demo(event, context=None)
    assert result['statusCode'] == 400
    assert json_response_body(result)['Code'] == 'BadRequestError'


def test_cant_access_json_body_with_wrong_content_type(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', methods=['POST'], content_types=['application/xml'])
    def index() -> Any:
        return (demo.current_request.json_body, demo.current_request.raw_body.decode('utf-8'))
    event: Dict[str, Any] = create_event('/', 'POST', {}, content_type='application/xml')
    event['body'] = '<Message>hello</Message>'
    response: Dict[str, Any] = json_response_body(demo(event, context=None))
    json_body, raw_body = response
    assert json_body is None
    assert raw_body == '<Message>hello</Message>'


def test_json_body_available_on_multiple_content_types(create_event_with_body: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', methods=['POST'], content_types=['application/xml', 'application/json'])
    def index() -> Any:
        return (demo.current_request.json_body, demo.current_request.raw_body.decode('utf-8'))
    event: Dict[str, Any] = create_event_with_body('<Message>hello</Message>', content_type='application/xml')
    response: Dict[str, Any] = json_response_body(demo(event, context=None))
    json_body, raw_body = response
    assert json_body is None
    assert raw_body == '<Message>hello</Message>'
    event = create_event_with_body({'foo': 'bar'}, content_type='application/json')
    response = json_response_body(demo(event, context=None))
    json_body, raw_body = response
    assert json_body == {'foo': 'bar'}
    assert raw_body == '{"foo": "bar"}'


def test_json_body_available_with_lowercase_content_type_key(create_event_with_body: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', methods=['POST'])
    def index() -> Any:
        return (demo.current_request.json_body, demo.current_request.raw_body.decode('utf-8'))
    event: Dict[str, Any] = create_event_with_body({'foo': 'bar'})
    del event['headers']['Content-Type']
    event['headers']['content-type'] = 'application/json'
    json_body, raw_body = json_response_body(demo(event, context=None))
    assert json_body == {'foo': 'bar'}
    assert raw_body == '{"foo": "bar"}'


def test_content_types_must_be_lists() -> None:
    demo: app.Chalice = app.Chalice('app-name')
    with pytest.raises(ValueError):
        @demo.route('/index', content_types='application/not-a-list')
        def index_post() -> Dict[str, str]:
            return {'foo': 'bar'}


def test_content_type_validation_raises_error_on_unknown_types(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', methods=['POST'], content_types=['application/xml'])
    def index() -> str:
        return 'success'
    bad_content_type: str = 'application/bad-xml'
    event: Dict[str, Any] = create_event('/', 'POST', {}, content_type=bad_content_type)
    event['body'] = 'Request body'
    json_response: Dict[str, Any] = json_response_body(demo(event, context=None))
    assert json_response['Code'] == 'UnsupportedMediaType'
    assert 'application/bad-xml' in json_response['Message']


def test_content_type_with_charset(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', content_types=['application/json'])
    def index() -> Dict[str, str]:
        return {'foo': 'bar'}
    event: Dict[str, Any] = create_event('/', 'GET', {}, 'application/json; charset=utf-8')
    response: Dict[str, Any] = json_response_body(demo(event, context=None))
    assert response == {'foo': 'bar'}


def test_can_return_response_object(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Response:
        return app.Response(status_code=200, body={'foo': 'bar'}, headers={'Content-Type': 'application/json', 'Set-Cookie': ['key=value', 'foo=bar']})
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    response: Dict[str, Any] = demo(event, context=None)
    assert response == {'statusCode': 200, 'body': '{"foo":"bar"}', 'headers': {'Content-Type': 'application/json'}, 'multiValueHeaders': {'Set-Cookie': ['key=value', 'foo=bar']}}


def test_headers_have_basic_validation(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Response:
        return app.Response(status_code=200, body='{}', headers={'Invalid-Header': 'foo\nbar'})
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    response: Dict[str, Any] = demo(event, context=None)
    assert response['statusCode'] == 500
    assert 'Invalid-Header' not in response['headers']
    assert json.loads(response['body'])['Code'] == 'InternalServerError'


def test_empty_headers_have_basic_validation(create_empty_header_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Response:
        return app.Response(status_code=200, body='{}', headers={})
    event: Dict[str, Any] = create_empty_header_event('/index', 'GET', {})
    response: Dict[str, Any] = demo(event, context=None)
    assert response['statusCode'] == 200


def test_no_content_type_is_still_allowed(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/', methods=['POST'], content_types=['application/json'])
    def index() -> Dict[str, bool]:
        return {'success': True}
    event: Dict[str, Any] = create_event('/', 'POST', {})
    del event['headers']['Content-Type']
    json_response: Dict[str, Any] = json_response_body(demo(event, context=None))
    assert json_response == {'success': True}


@pytest.mark.parametrize('content_type,accept', [
    ('application/octet-stream', 'application/octet-stream'),
    ('application/octet-stream', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'),
    ('image/gif', 'text/html,image/gif'),
    ('image/gif', 'text/html ,image/gif'),
    ('image/gif', 'text/html, image/gif'),
    ('image/gif', 'text/html;q=0.8, image/gif ;q=0.5'),
    ('image/gif', 'text/html,image/png'),
    ('image/png', 'text/html,image/gif')
])
def test_can_base64_encode_binary_multiple_media_types(create_event: Callable[..., Dict[str, Any]], content_type: str, accept: str) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/index')
    def index_view() -> Response:
        return app.Response(status_code=200, body=u'✓'.encode('utf-8'), headers={'Content-Type': content_type})
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    event['headers']['Accept'] = accept
    response: Dict[str, Any] = demo(event, context=None)
    assert response['statusCode'] == 200
    assert response['isBase64Encoded'] is True
    assert response['body'] == '4pyT'
    assert response['headers']['Content-Type'] == content_type


def test_can_return_text_even_with_binary_content_type_configured(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/index')
    def index_view() -> Response:
        return app.Response(status_code=200, body='Plain text', headers={'Content-Type': 'text/plain'})
    event: Dict[str, Any] = create_event('/index', 'GET', {})
    event['headers']['Accept'] = 'application/octet-stream'
    response: Dict[str, Any] = demo(event, context=None)
    assert response['statusCode'] == 200
    assert response['body'] == 'Plain text'
    assert response['headers']['Content-Type'] == 'text/plain'


def test_route_equality(view_function: Callable[[], Dict[str, str]]) -> None:
    a: app.RouteEntry = app.RouteEntry(view_function, view_name='myview', path='/', method='GET', api_key_required=True, content_types=['application/json'])
    b: app.RouteEntry = app.RouteEntry(view_function, view_name='myview', path='/', method='GET', api_key_required=True, content_types=['application/json'])
    assert a == b


def test_route_inequality(view_function: Callable[[], Dict[str, str]]) -> None:
    a: app.RouteEntry = app.RouteEntry(view_function, view_name='myview', path='/', method='GET', api_key_required=True, content_types=['application/json'])
    b: app.RouteEntry = app.RouteEntry(view_function, view_name='myview', path='/', method='GET', api_key_required=True, content_types=['application/xml'])
    assert not a == b


def test_exceptions_raised_as_chalice_errors(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    @sample_app.route('/error')
    def raise_error() -> None:
        raise TypeError('Raising arbitrary error, should never see.')
    event: Dict[str, Any] = create_event('/error', 'GET', {})
    raw_response: Dict[str, Any] = sample_app(event, context=None)
    response: Dict[str, Any] = json_response_body(raw_response)
    assert response['Code'] == 'InternalServerError'
    assert raw_response['statusCode'] == 500


def test_original_exception_raised_in_debug_mode(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    sample_app.debug = True

    @sample_app.route('/error')
    def raise_error() -> None:
        raise ValueError('You will see this error')
    event: Dict[str, Any] = create_event('/error', 'GET', {})
    response: Dict[str, Any] = sample_app(event, context=None)
    assert response['statusCode'] == 500
    assert 'ValueError' in response['body']
    assert 'You will see this error' in response['body']


def test_chalice_view_errors_propagate_in_non_debug_mode(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    @sample_app.route('/notfound')
    def notfound() -> None:
        raise NotFoundError('resource not found')
    event: Dict[str, Any] = create_event('/notfound', 'GET', {})
    raw_response: Dict[str, Any] = sample_app(event, context=None)
    assert raw_response['statusCode'] == 404
    assert json_response_body(raw_response)['Code'] == 'NotFoundError'


def test_chalice_view_errors_propagate_in_debug_mode(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    @sample_app.route('/notfound')
    def notfound() -> None:
        raise NotFoundError('resource not found')
    sample_app.debug = True
    event: Dict[str, Any] = create_event('/notfound', 'GET', {})
    raw_response: Dict[str, Any] = sample_app(event, context=None)
    assert raw_response['statusCode'] == 404
    assert json_response_body(raw_response)['Code'] == 'NotFoundError'


def test_case_insensitive_mapping() -> None:
    mapping: app.CaseInsensitiveMapping = app.CaseInsensitiveMapping({'HEADER': 'Value'})
    assert mapping['hEAdEr']
    assert mapping.get('hEAdEr')
    assert 'hEAdEr' in mapping
    assert repr({'header': 'Value'}) in repr(mapping)


def test_unknown_kwargs_raise_error(sample_app: app.Chalice, create_event: Callable[..., Dict[str, Any]]) -> None:
    with pytest.raises(TypeError):
        @sample_app.route('/foo', unknown_kwargs='foo')
        def badkwargs() -> None:
            pass


def test_name_kwargs_does_not_raise_error(sample_app: app.Chalice) -> None:
    try:
        @sample_app.route('/foo', name='foo')
        def name_kwarg() -> None:
            pass
    except TypeError:
        pytest.fail('route name kwarg should not raise TypeError.')


def test_default_logging_handlers_created() -> None:
    handlers_before: List[Any] = logging.getLogger('log_app').handlers[:]
    app.Chalice('log_app', configure_logs=True)
    handlers_after: List[Any] = logging.getLogger('log_app').handlers[:]
    new_handlers = set(handlers_after) - set(handlers_before)
    assert len(new_handlers) == 1


def test_default_logging_only_added_once() -> None:
    handlers_before: List[Any] = logging.getLogger('added_once').handlers[:]
    app.Chalice('added_once', configure_logs=True)
    app.Chalice('added_once', configure_logs=True)
    handlers_after: List[Any] = logging.getLogger('added_once').handlers[:]
    new_handlers = set(handlers_after) - set(handlers_before)
    assert len(new_handlers) == 1


def test_logs_can_be_disabled() -> None:
    handlers_before: List[Any] = logging.getLogger('log_app').handlers[:]
    app.Chalice('log_app', configure_logs=False)
    handlers_after: List[Any] = logging.getLogger('log_app').handlers[:]
    new_handlers = set(handlers_after) - set(handlers_before)
    assert len(new_handlers) == 0


@pytest.mark.parametrize('content_type,is_json', [
    ('application/json', True),
    ('application/json;charset=UTF-8', True),
    ('application/notjson', False)
])
def test_json_body_available_when_content_type_matches(content_type: str, is_json: bool) -> None:
    request: Request = create_request_with_content_type(content_type)
    if is_json:
        assert request.json_body == {'json': 'body'}
    else:
        assert request.json_body is None


def test_can_receive_binary_data(create_event_with_body: Callable[..., Dict[str, Any]]) -> None:
    content_type: str = 'application/octet-stream'
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/bincat', methods=['POST'], content_types=[content_type])
    def bincat() -> Response:
        raw_body: bytes = demo.current_request.raw_body
        return Response(raw_body, headers={'Content-Type': content_type}, status_code=200)
    body: str = 'L3UyNzEz'
    event: Dict[str, Any] = create_event_with_body(body, '/bincat', 'POST', content_type)
    event['headers']['Accept'] = content_type
    event['isBase64Encoded'] = True
    response: Dict[str, Any] = demo(event, context=None)
    assert response['statusCode'] == 200
    assert response['body'] == body


def test_cannot_receive_base64_string_with_binary_response(create_event_with_body: Callable[..., Dict[str, Any]]) -> None:
    content_type: str = 'application/octet-stream'
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/bincat', methods=['GET'], content_types=[content_type])
    def bincat() -> Response:
        return Response(status_code=200, body=u'✓'.encode('utf-8'), headers={'Content-Type': content_type})
    event: Dict[str, Any] = create_event_with_body('', '/bincat', 'GET', content_type)
    response: Dict[str, Any] = demo(event, context=None)
    assert response['statusCode'] == 400


def test_can_serialize_cognito_auth() -> None:
    auth: app.CognitoUserPoolAuthorizer = app.CognitoUserPoolAuthorizer('Name', provider_arns=['Foo'], header='Authorization')
    assert auth.to_swagger() == {'in': 'header', 'type': 'apiKey', 'name': 'Authorization', 'x-amazon-apigateway-authtype': 'cognito_user_pools', 'x-amazon-apigateway-authorizer': {'type': 'cognito_user_pools', 'providerARNs': ['Foo']}}


def test_can_serialize_iam_auth() -> None:
    auth: app.IAMAuthorizer = app.IAMAuthorizer()
    assert auth.to_swagger() == {'in': 'header', 'type': 'apiKey', 'name': 'Authorization', 'x-amazon-apigateway-authtype': 'awsSigv4'}


def test_typecheck_list_type() -> None:
    with pytest.raises(TypeError):
        app.CognitoUserPoolAuthorizer('Name', 'Authorization', provider_arns='foo')  # type: ignore


def test_can_serialize_custom_authorizer() -> None:
    auth: app.CustomAuthorizer = app.CustomAuthorizer('Name', 'myuri', ttl_seconds=10, header='NotAuth', invoke_role_arn='role-arn')
    assert auth.to_swagger() == {'in': 'header', 'type': 'apiKey', 'name': 'NotAuth', 'x-amazon-apigateway-authtype': 'custom', 'x-amazon-apigateway-authorizer': {'type': 'token', 'authorizerUri': 'myuri', 'authorizerResultTtlInSeconds': 10, 'authorizerCredentials': 'role-arn'}}


class TestCORSConfig:
    def test_eq(self) -> None:
        cors_config: app.CORSConfig = app.CORSConfig()
        other_cors_config: app.CORSConfig = app.CORSConfig()
        assert cors_config == other_cors_config

    def test_not_eq_different_type(self) -> None:
        cors_config: app.CORSConfig = app.CORSConfig()
        different_type_obj: Any = object()
        assert not cors_config == different_type_obj

    def test_not_eq_differing_configurations(self) -> None:
        cors_config: app.CORSConfig = app.CORSConfig()
        differing_cors_config: app.CORSConfig = app.CORSConfig(allow_origin='https://foo.example.com')
        assert cors_config != differing_cors_config

    def test_eq_non_default_configurations(self) -> None:
        custom_cors: app.CORSConfig = app.CORSConfig(allow_origin='https://foo.example.com', allow_headers=['X-Special-Header'], max_age=600, expose_headers=['X-Special-Header'], allow_credentials=True)
        same_custom_cors: app.CORSConfig = app.CORSConfig(allow_origin='https://foo.example.com', allow_headers=['X-Special-Header'], max_age=600, expose_headers=['X-Special-Header'], allow_credentials=True)
        assert custom_cors == same_custom_cors


def test_can_handle_builtin_auth() -> None:
    demo: app.Chalice = app.Chalice('builtin-auth')

    @demo.authorizer()
    def my_auth(auth_request: app.AuthRequest) -> None:
        pass

    @demo.route('/', authorizer=my_auth)
    def index_view() -> Dict:
        return {}
    assert len(demo.builtin_auth_handlers) == 1
    authorizer: app.BuiltinAuthConfig = demo.builtin_auth_handlers[0]
    assert isinstance(authorizer, app.BuiltinAuthConfig)
    assert authorizer.name == 'my_auth'
    assert authorizer.handler_string == 'app.my_auth'


def test_builtin_auth_can_transform_event() -> None:
    event: Dict[str, Any] = {'type': 'TOKEN', 'authorizationToken': 'authtoken', 'methodArn': 'arn:aws:execute-api:...:foo'}
    auth_app: app.Chalice = app.Chalice('builtin-auth')
    request: List[app.AuthRequest] = []

    @auth_app.authorizer()
    def builtin_auth(auth_request: app.AuthRequest) -> None:
        request.append(auth_request)
    builtin_auth(event, None)
    assert len(request) == 1
    transformed: app.AuthRequest = request[0]
    assert transformed.auth_type == 'TOKEN'
    assert transformed.token == 'authtoken'
    assert transformed.method_arn == 'arn:aws:execute-api:...:foo'


def test_can_return_auth_dict_directly() -> None:
    event: Dict[str, Any] = {'type': 'TOKEN', 'authorizationToken': 'authtoken', 'methodArn': 'arn:aws:execute-api:...:foo'}
    auth_app: app.Chalice = app.Chalice('builtin-auth')
    response: Dict[str, Any] = {'context': {'foo': 'bar'}, 'principalId': 'user', 'policyDocument': {'Version': '2012-10-17', 'Statement': []}}

    @auth_app.authorizer()
    def builtin_auth(auth_request: app.AuthRequest) -> Dict[str, Any]:
        return response
    actual: Dict[str, Any] = builtin_auth(event, None)
    assert actual == response


def test_can_specify_extra_auth_attributes() -> None:
    auth_app: app.Chalice = app.Chalice('builtin-auth')

    @auth_app.authorizer(ttl_seconds=10, execution_role='arn:my-role')
    def builtin_auth(auth_request: app.AuthRequest) -> None:
        pass
    handler: app.BuiltinAuthConfig = auth_app.builtin_auth_handlers[0]
    assert handler.ttl_seconds == 10
    assert handler.execution_role == 'arn:my-role'


def test_validation_raised_on_unknown_kwargs() -> None:
    auth_app: app.Chalice = app.Chalice('builtin-auth')
    with pytest.raises(TypeError):
        @auth_app.authorizer(this_is_an_unknown_kwarg=True)
        def builtin_auth(auth_request: app.AuthRequest) -> None:
            pass


def test_can_return_auth_response() -> None:
    event: Dict[str, Any] = {'type': 'TOKEN', 'authorizationToken': 'authtoken', 'methodArn': 'arn:aws:execute-api:us-west-2:1:id/dev/GET/a'}
    auth_app: app.Chalice = app.Chalice('builtin-auth')
    response: Dict[str, Any] = {
        'context': {},
        'principalId': 'principal',
        'policyDocument': {
            'Version': '2012-10-17',
            'Statement': [
                {'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': [event['methodArn'].replace('GET', '*')]}
            ]
        }
    }

    @auth_app.authorizer()
    def builtin_auth(auth_request: app.AuthRequest) -> app.AuthResponse:
        return app.AuthResponse(['/a'], 'principal')
    actual: Dict[str, Any] = builtin_auth(event, None)
    assert actual == response


def test_auth_response_with_colon_chars() -> None:
    event: Dict[str, Any] = {'type': 'TOKEN', 'authorizationToken': 'authtoken', 'methodArn': 'arn:aws:execute-api:us-west-2:1:id/api/GET/foo/a:b:c:d'}
    auth_app: app.Chalice = app.Chalice('builtin-auth')
    response: Dict[str, Any] = {
        'context': {},
        'principalId': 'principal',
        'policyDocument': {
            'Version': '2012-10-17',
            'Statement': [
                {'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': ['arn:aws:execute-api:us-west-2:1:id/api/*/foo/*']}
            ]
        }
    }

    @auth_app.authorizer()
    def builtin_auth(auth_request: app.AuthRequest) -> app.AuthResponse:
        return app.AuthResponse(['/foo/*'], 'principal')
    actual: Dict[str, Any] = builtin_auth(event, None)
    assert actual == response


def test_auth_response_serialization() -> None:
    method_arn: str = 'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/needs/auth'
    request: app.AuthRequest = app.AuthRequest('TOKEN', 'authtoken', method_arn)
    response: app.AuthResponse = app.AuthResponse(routes=['/needs/auth'], principal_id='foo')
    response_dict: Dict[str, Any] = response.to_dict(request)
    expected: List[str] = [method_arn.replace('GET', '*')]
    assert response_dict == {'policyDocument': {'Version': '2012-10-17', 'Statement': [{'Action': 'execute-api:Invoke', 'Resource': expected, 'Effect': 'Allow'}]}, 'context': {}, 'principalId': 'foo'}


def test_auth_response_can_include_context(auth_request: app.AuthRequest) -> None:
    response: app.AuthResponse = app.AuthResponse(['/foo'], 'principal', {'foo': 'bar'})
    serialized: Dict[str, Any] = response.to_dict(auth_request)
    assert serialized['context'] == {'foo': 'bar'}


def test_can_use_auth_routes_instead_of_strings(auth_request: app.AuthRequest) -> None:
    expected: List[str] = [
        'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/a',
        'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/a/b',
        'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/POST/a/b'
    ]
    response: app.AuthResponse = app.AuthResponse([app.AuthRoute('/a', ['GET']), app.AuthRoute('/a/b', ['GET', 'POST'])], 'principal')
    serialized: Dict[str, Any] = response.to_dict(auth_request)
    assert serialized['policyDocument'] == {'Version': '2012-10-17', 'Statement': [{'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': expected}]}


def test_auth_response_wildcard(auth_request: app.AuthRequest) -> None:
    response: app.AuthResponse = app.AuthResponse(routes=[app.AuthRoute(path='*', methods=['*'])], principal_id='user')
    serialized: Dict[str, Any] = response.to_dict(auth_request)
    assert serialized['policyDocument'] == {'Statement': [{'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': ['arn:aws:execute-api:us-west-2:123:rest-api-id/dev/*/*']}], 'Version': '2012-10-17'}


def test_auth_response_wildcard_string(auth_request: app.AuthRequest) -> None:
    response: app.AuthResponse = app.AuthResponse(routes=['*'], principal_id='user')
    serialized: Dict[str, Any] = response.to_dict(auth_request)
    assert serialized['policyDocument'] == {'Statement': [{'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': ['arn:aws:execute-api:us-west-2:123:rest-api-id/dev/*/*']}], 'Version': '2012-10-17'}


def test_can_mix_auth_routes_and_strings(auth_request: app.AuthRequest) -> None:
    expected: List[str] = [
        'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/*/a',
        'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/a/b'
    ]
    response: app.AuthResponse = app.AuthResponse(['/a', app.AuthRoute('/a/b', ['GET'])], 'principal')
    serialized: Dict[str, Any] = response.to_dict(auth_request)
    assert serialized['policyDocument'] == {'Version': '2012-10-17', 'Statement': [{'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': expected}]}


def test_root_resource(auth_request: app.AuthRequest) -> None:
    auth_request.method_arn = 'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/'
    expected: List[str] = ['arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/']
    response: app.AuthResponse = app.AuthResponse([app.AuthRoute('/', ['GET'])], 'principal')
    serialized: Dict[str, Any] = response.to_dict(auth_request)
    assert serialized['policyDocument'] == {'Version': '2012-10-17', 'Statement': [{'Action': 'execute-api:Invoke', 'Effect': 'Allow', 'Resource': expected}]}


def test_can_register_scheduled_event_with_str(sample_app: app.Chalice) -> None:
    @sample_app.schedule('rate(1 minute)')
    def foo(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    event_source: Any = sample_app.event_sources[0]
    assert event_source.name == 'foo'
    assert event_source.schedule_expression == 'rate(1 minute)'
    assert event_source.handler_string == 'app.foo'


def test_can_register_scheduled_event_with_rate(sample_app: app.Chalice) -> None:
    @sample_app.schedule(app.Rate(value=2, unit=app.Rate.HOURS))
    def foo(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    expression: Any = sample_app.event_sources[0].schedule_expression
    assert expression.value == 2
    assert expression.unit == app.Rate.HOURS


def test_can_register_scheduled_event_with_event(sample_app: app.Chalice) -> None:
    @sample_app.schedule(app.Cron(0, 10, '*', '*', '?', '*'))
    def foo(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    expression: Any = sample_app.event_sources[0].schedule_expression
    assert expression.minutes == 0
    assert expression.hours == 10
    assert expression.day_of_month == '*'
    assert expression.month == '*'
    assert expression.day_of_week == '?'
    assert expression.year == '*'


@pytest.mark.parametrize('value,unit,expected', [
    (1, app.Rate.MINUTES, 'rate(1 minute)'),
    (2, app.Rate.MINUTES, 'rate(2 minutes)'),
    (1, app.Rate.HOURS, 'rate(1 hour)'),
    (2, app.Rate.HOURS, 'rate(2 hours)'),
    (1, app.Rate.DAYS, 'rate(1 day)'),
    (2, app.Rate.DAYS, 'rate(2 days)')
])
def test_rule_object_converts_to_str(value: int, unit: str, expected: str) -> None:
    assert app.Rate(value=value, unit=unit).to_string() == expected


@pytest.mark.parametrize('minutes,hours,day_of_month,month,day_of_week,year,expected', [
    (0, 10, '*', '*', '?', '*', 'cron(0 10 * * ? *)'),
    (15, 12, '*', '*', '?', '*', 'cron(15 12 * * ? *)'),
    (0, 18, '?', '*', 'MON-FRI', '*', 'cron(0 18 ? * MON-FRI *)'),
    (0, 8, 1, '*', '?', '*', 'cron(0 8 1 * ? *)'),
    ('0/10', '*', '?', '*', 'MON-FRI', '*', 'cron(0/10 * ? * MON-FRI *)'),
    ('0/5', '8-17', '?', '*', 'MON-FRI', '*', 'cron(0/5 8-17 ? * MON-FRI *)'),
    (0, 9, '?', '*', '2#1', '*', 'cron(0 9 ? * 2#1 *)')
])
def test_cron_expression_converts_to_str(minutes: Union[int, str], hours: Union[int, str],
                                           day_of_month: Union[int, str], month: Union[int, str],
                                           day_of_week: Union[int, str], year: Union[int, str],
                                           expected: str) -> None:
    assert app.Cron(minutes=minutes, hours=hours, day_of_month=day_of_month, month=month, day_of_week=day_of_week, year=year).to_string() == expected


def test_can_map_schedule_event_dict_to_object(sample_app: app.Chalice) -> None:
    @sample_app.schedule('rate(1 hour)')
    def handler(event: Any) -> Any:
        return event
    lambda_event: Dict[str, Any] = {
        'version': '0',
        'account': '123456789012',
        'region': 'us-west-2',
        'detail': {},
        'detail-type': 'Scheduled Event',
        'source': 'aws.events',
        'time': '1970-01-01T00:00:00Z',
        'id': 'event-id',
        'resources': ['arn:aws:events:us-west-2:123456789012:rule/my-schedule']
    }
    event_object: Any = handler(lambda_event, context=None)
    assert event_object.version == '0'
    assert event_object.event_id == 'event-id'
    assert event_object.source == 'aws.events'
    assert event_object.account == '123456789012'
    assert event_object.time == '1970-01-01T00:00:00Z'
    assert event_object.region == 'us-west-2'
    assert event_object.resources == ['arn:aws:events:us-west-2:123456789012:rule/my-schedule']
    assert event_object.detail == {}
    assert event_object.detail_type == 'Scheduled Event'
    assert event_object.to_dict() == lambda_event


def test_can_create_cwe_event_handler(sample_app: app.Chalice) -> None:
    @sample_app.on_cw_event({'source': ['aws.ec2']})
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    event: Any = sample_app.event_sources[0]
    assert event.name == 'handler'
    assert event.event_pattern == {'source': ['aws.ec2']}
    assert event.handler_string == 'app.handler'


def test_can_map_cwe_event_dict_to_object(sample_app: app.Chalice) -> None:
    @sample_app.on_cw_event({'source': ['aws.ec2']})
    def handler(event: Any) -> Any:
        return event
    lambda_event: Dict[str, Any] = {
        'version': 0,
        'id': '7bf73129-1428-4cd3-a780-95db273d1602',
        'detail-type': 'EC2 Instance State-change Notification',
        'source': 'aws.ec2',
        'account': '123456789012',
        'time': '2015-11-11T21:29:54Z',
        'region': 'us-east-1',
        'resources': ['arn:aws:ec2:us-east-1:123456789012:instance/i-abcd1111'],
        'detail': {'instance-id': 'i-abcd1111', 'state': 'pending'}
    }
    event_object: Any = handler(lambda_event, context=None)
    assert event_object.detail_type == 'EC2 Instance State-change Notification'
    assert event_object.account == '123456789012'
    assert event_object.region == 'us-east-1'
    assert event_object.detail == {'instance-id': 'i-abcd1111', 'state': 'pending'}


def test_pure_lambda_function_direct_mapping(sample_app: app.Chalice) -> None:
    @sample_app.lambda_function()
    def handler(event: Any, context: Any) -> tuple[Any, Any]:
        return (event, context)
    return_value: tuple[Any, Any] = handler({'fake': 'event'}, {'fake': 'context'})
    assert return_value[0] == {'fake': 'event'}
    assert return_value[1] == {'fake': 'context'}


def test_pure_lambda_functions_are_registered_in_app(sample_app: app.Chalice) -> None:
    @sample_app.lambda_function()
    def handler(event: Any, context: Any) -> None:
        pass
    assert len(sample_app.pure_lambda_functions) == 1
    lambda_function: Any = sample_app.pure_lambda_functions[0]
    assert lambda_function.name == 'handler'
    assert lambda_function.handler_string == 'app.handler'


def test_aws_execution_env_set() -> None:
    env: Dict[str, str] = {'AWS_EXECUTION_ENV': 'AWS_Lambda_python2.7'}
    app.Chalice('app-name', env=env)
    assert env['AWS_EXECUTION_ENV'] == 'AWS_Lambda_python2.7 aws-chalice/%s' % chalice_version


def test_can_use_out_of_order_args(create_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')

    @demo.route('/{a}/{b}', methods=['GET'])
    def index(b: str, a: str) -> Dict[str, str]:
        return {'a': a, 'b': b}
    event: Dict[str, Any] = create_event('/{a}/{b}', 'GET', {'a': 'first', 'b': 'second'})
    response: Dict[str, Any] = demo(event, context=None)
    response = json_response_body(response)
    assert response == {'a': 'first', 'b': 'second'}


def test_ensure_debug_mode_is_false_by_default() -> None:
    test_app: app.Chalice = app.Chalice('logger-test-1')
    assert test_app.debug is False
    assert test_app.log.getEffectiveLevel() == logging.ERROR


def test_can_explicitly_set_debug_false_in_initializer() -> None:
    test_app: app.Chalice = app.Chalice('logger-test-2', debug=False)
    assert test_app.debug is False
    assert test_app.log.getEffectiveLevel() == logging.ERROR


def test_can_set_debug_mode_in_initialzier() -> None:
    test_app: app.Chalice = app.Chalice('logger-test-3', debug=True)
    assert test_app.debug is True
    assert test_app.log.getEffectiveLevel() == logging.DEBUG


def test_debug_mode_changes_log_level() -> None:
    test_app: app.Chalice = app.Chalice('logger-test-4', debug=False)
    test_app.debug = True
    assert test_app.debug is True
    assert test_app.log.getEffectiveLevel() == logging.DEBUG


def test_internal_exception_debug_false(capsys: Any, create_event: Callable[..., Dict[str, Any]]) -> None:
    test_app: app.Chalice = app.Chalice('logger-test-5', debug=False)

    @test_app.route('/error')
    def error() -> None:
        raise Exception('Something bad happened')
    event: Dict[str, Any] = create_event('/error', 'GET', {})
    test_app(event, context=None)
    out, err = capsys.readouterr()
    assert 'logger-test-5' in out
    assert 'Caught exception' in out
    assert 'Something bad happened' in out


def test_raw_body_is_none_if_body_is_none() -> None:
    event: Dict[str, Any] = {
        'body': None,
        'multiValueQueryStringParameters': '',
        'headers': {},
        'pathParameters': {},
        'requestContext': {'httpMethod': 'GET', 'resourcePath': '/'},
        'stageVariables': {},
        'isBase64Encoded': False
    }
    request: Request = app.Request(event, FakeLambdaContext())
    assert request.raw_body == b''


@given(http_request_event=HTTP_REQUEST)
def test_http_request_to_dict_is_json_serializable(http_request_event: Dict[str, Any]) -> None:
    is_base64_encoded: bool = http_request_event['isBase64Encoded']
    if is_base64_encoded:
        assume(http_request_event['body'] is not None)
        body: bytes = base64.b64encode(http_request_event['body'].encode('utf-8'))
        http_request_event['body'] = body.decode('ascii')
    request: Request = Request(http_request_event, FakeLambdaContext())
    assert isinstance(request.raw_body, bytes)
    request_dict: Dict[str, Any] = request.to_dict()
    assert json.dumps(request_dict, default=handle_extra_types)


@given(body=st.text(), headers=STR_MAP, status_code=st.integers(min_value=200, max_value=599))
def test_http_response_to_dict(body: str, headers: Dict[str, str], status_code: int) -> None:
    r: Response = Response(body=body, headers=headers, status_code=status_code)
    serialized: Dict[str, Any] = r.to_dict()
    assert 'headers' in serialized
    assert 'statusCode' in serialized
    assert 'body' in serialized
    assert isinstance(serialized['body'], six.string_types)


@given(body=st.binary(), content_type=st.sampled_from(BINARY_TYPES))
def test_handles_binary_responses(body: bytes, content_type: str) -> None:
    r: Response = Response(body=body, headers={'Content-Type': content_type})
    serialized: Dict[str, Any] = r.to_dict(BINARY_TYPES)
    assert serialized['isBase64Encoded']
    assert isinstance(serialized['body'], six.string_types)
    assert isinstance(base64.b64decode(serialized['body']), bytes)


def test_can_create_s3_event_handler(sample_app: app.Chalice) -> None:
    @sample_app.on_s3_event(bucket='mybucket')
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    event: Any = sample_app.event_sources[0]
    assert event.name == 'handler'
    assert event.bucket == 'mybucket'
    assert event.events == ['s3:ObjectCreated:*']
    assert event.handler_string == 'app.handler'


def test_can_map_to_s3_event_object(sample_app: app.Chalice) -> None:
    @sample_app.on_s3_event(bucket='mybucket')
    def handler(event: Any) -> Any:
        return event
    s3_event: Dict[str, Any] = {
        'Records': [{
            'awsRegion': 'us-west-2',
            'eventName': 'ObjectCreated:Put',
            'eventSource': 'aws:s3',
            'eventTime': '2018-05-22T04:41:23.823Z',
            'eventVersion': '2.0',
            'requestParameters': {'sourceIPAddress': '174.127.235.55'},
            'responseElements': {'x-amz-id-2': 'request-id-2', 'x-amz-request-id': 'request-id-1'},
            's3': {
                'bucket': {'arn': 'arn:aws:s3:::mybucket', 'name': 'mybucket', 'ownerIdentity': {'principalId': 'ABCD'}},
                'configurationId': 'config-id',
                'object': {'eTag': 'd41d8cd98f00b204e9800998ecf8427e', 'key': 'hello-world.txt', 'sequencer': '005B039F73C627CE8B', 'size': 0},
                's3SchemaVersion': '1.0'
            },
            'userIdentity': {'principalId': 'AWS:XYZ'}
        }]
    }
    lambda_context: FakeLambdaContext = FakeLambdaContext()
    actual_event: Any = handler(s3_event, context=lambda_context)
    assert actual_event.bucket == 'mybucket'
    assert actual_event.key == 'hello-world.txt'
    assert actual_event.to_dict() == s3_event


def test_s3_event_urldecodes_keys() -> None:
    s3_event: Dict[str, Any] = {'Records': [{
        's3': {
            'bucket': {'arn': 'arn:aws:s3:::mybucket', 'name': 'mybucket'},
            'object': {'key': 'file+with+spaces', 'sequencer': '005B039F73C627CE8B', 'size': 0}
        }
    }]}
    event: app.S3Event = app.S3Event(s3_event, FakeLambdaContext())
    assert event.key == 'file with spaces'
    assert event.to_dict() == s3_event


def test_s3_event_urldecodes_unicode_keys() -> None:
    s3_event: Dict[str, Any] = {'Records': [{
        's3': {
            'bucket': {'arn': 'arn:aws:s3:::mybucket', 'name': 'mybucket'},
            'object': {'key': '%E2%9C%93', 'sequencer': '005B039F73C627CE8B', 'size': 0}
        }
    }]}
    event: app.S3Event = app.S3Event(s3_event, FakeLambdaContext())
    assert event.key == u'✓'
    assert event.bucket == u'mybucket'
    assert event.to_dict() == s3_event


def test_can_create_sns_handler(sample_app: app.Chalice) -> None:
    @sample_app.on_sns_message(topic='MyTopic')
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    event: Any = sample_app.event_sources[0]
    assert event.name == 'handler'
    assert event.topic == 'MyTopic'
    assert event.handler_string == 'app.handler'


def test_can_map_sns_event(sample_app: app.Chalice) -> None:
    @sample_app.on_sns_message(topic='MyTopic')
    def handler(event: Any) -> Any:
        return event
    sns_event: Dict[str, Any] = {
        'Records': [{
            'EventSource': 'aws:sns',
            'EventSubscriptionArn': 'arn:subscription-arn',
            'EventVersion': '1.0',
            'Sns': {
                'Message': 'This is a raw message',
                'MessageAttributes': {'AttributeKey': {'Type': 'String', 'Value': 'AttributeValue'}},
                'MessageId': 'abcdefgh-51e4-5ae2-9964-b296c8d65d1a',
                'Signature': 'signature',
                'SignatureVersion': '1',
                'SigningCertUrl': 'https://sns.us-west-2.amazonaws.com/cert.pen',
                'Subject': 'ThisIsTheSubject',
                'Timestamp': '2018-06-26T19:41:38.695Z',
                'TopicArn': 'arn:aws:sns:us-west-2:12345:ConsoleTestTopic',
                'Type': 'Notification',
                'UnsubscribeUrl': 'https://unsubscribe-url/'
            }
        }]
    }
    lambda_context: FakeLambdaContext = FakeLambdaContext()
    actual_event: Any = handler(sns_event, context=lambda_context)
    assert actual_event.message == 'This is a raw message'
    assert actual_event.subject == 'ThisIsTheSubject'
    assert actual_event.to_dict() == sns_event
    assert actual_event.context == lambda_context


def test_can_create_sqs_handler(sample_app: app.Chalice) -> None:
    @sample_app.on_sqs_message(queue='MyQueue', batch_size=200)
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    event: Any = sample_app.event_sources[0]
    assert event.queue == 'MyQueue'
    assert event.batch_size == 200
    assert event.maximum_batching_window_in_seconds == 0
    assert event.handler_string == 'app.handler'


def test_can_set_sqs_handler_name(sample_app: app.Chalice) -> None:
    @sample_app.on_sqs_message(queue='MyQueue', name='sqs_handler')
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    event: Any = sample_app.event_sources[0]
    assert event.name == 'sqs_handler'


def test_can_set_sqs_handler_maximum_batching_window_in_seconds(sample_app: app.Chalice) -> None:
    @sample_app.on_sqs_message(queue='MyQueue', maximum_batching_window_in_seconds=60)
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    event: Any = sample_app.event_sources[0]
    assert event.maximum_batching_window_in_seconds == 60


def test_can_map_sqs_event(sample_app: app.Chalice) -> None:
    @sample_app.on_sqs_message(queue='queue-name')
    def handler(event: Any) -> Any:
        return event
    sqs_event: Dict[str, Any] = {
        'Records': [{
            'attributes': {
                'ApproximateFirstReceiveTimestamp': '1530576251596',
                'ApproximateReceiveCount': '1',
                'SenderId': 'sender-id',
                'SentTimestamp': '1530576251595'
            },
            'awsRegion': 'us-west-2',
            'body': 'queue message body',
            'eventSource': 'aws:sqs',
            'eventSourceARN': 'arn:aws:sqs:us-west-2:12345:queue-name',
            'md5OfBody': '754ac2f7a12df38320e0c5eafd060145',
            'messageAttributes': {},
            'messageId': 'message-id',
            'receiptHandle': 'receipt-handle'
        }]
    }
    lambda_context: FakeLambdaContext = FakeLambdaContext()
    actual_event: Any = handler(sqs_event, context=lambda_context)
    records: List[Any] = list(actual_event)
    assert len(records) == 1
    first_record: Any = records[0]
    assert first_record.body == 'queue message body'
    assert first_record.receipt_handle == 'receipt-handle'
    assert first_record.to_dict() == sqs_event['Records'][0]
    assert actual_event.to_dict() == sqs_event
    assert actual_event.context == lambda_context


def test_can_create_kinesis_handler(sample_app: app.Chalice) -> None:
    @sample_app.on_kinesis_record(stream='MyStream', batch_size=1, starting_position='TRIM_HORIZON')
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    config: Any = sample_app.event_sources[0]
    assert config.stream == 'MyStream'
    assert config.batch_size == 1
    assert config.starting_position == 'TRIM_HORIZON'
    assert config.maximum_batching_window_in_seconds == 0


def test_can_set_kinesis_handler_maximum_batching_window(sample_app: app.Chalice) -> None:
    @sample_app.on_kinesis_record(stream='MyStream', batch_size=1, starting_position='TRIM_HORIZON', maximum_batching_window_in_seconds=60)
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    config: Any = sample_app.event_sources[0]
    assert config.maximum_batching_window_in_seconds == 60


def test_can_map_kinesis_event(sample_app: app.Chalice) -> None:
    @sample_app.on_kinesis_record(stream='MyStream')
    def handler(event: Any) -> Any:
        return event
    kinesis_event: Dict[str, Any] = {
        'Records': [{
            'kinesis': {
                'kinesisSchemaVersion': '1.0',
                'partitionKey': '1',
                'sequenceNumber': '12345',
                'data': 'SGVsbG8sIHRoaXMgaXMgYSB0ZXN0Lg==',
                'approximateArrivalTimestamp': 1545084650.987
            },
            'eventSource': 'aws:kinesis',
            'eventVersion': '1.0',
            'eventID': 'shardId-000000000006:12345',
            'eventName': 'aws:kinesis:record',
            'invokeIdentityArn': 'arn:aws:iam::123:role/lambda-role',
            'awsRegion': 'us-east-2',
            'eventSourceARN': 'arn:aws:kinesis:us-east-2:123:stream/lambda-stream'
        }, {
            'kinesis': {
                'kinesisSchemaVersion': '1.0',
                'partitionKey': '1',
                'sequenceNumber': '12346',
                'data': 'VGhpcyBpcyBvbmx5IGEgdGVzdC4=',
                'approximateArrivalTimestamp': 1545084711.166
            },
            'eventSource': 'aws:kinesis',
            'eventVersion': '1.0',
            'eventID': 'shardId-000000000006:12346',
            'eventName': 'aws:kinesis:record',
            'invokeIdentityArn': 'arn:aws:iam::123:role/lambda-role',
            'awsRegion': 'us-east-2',
            'eventSourceARN': 'arn:aws:kinesis:us-east-2:123:stream/lambda-stream'
        }]
    }
    lambda_context: FakeLambdaContext = FakeLambdaContext()
    actual_event: Any = handler(kinesis_event, context=lambda_context)
    records: List[Any] = list(actual_event)
    assert len(records) == 2
    assert records[0].data == b'Hello, this is a test.'
    assert records[0].sequence_number == '12345'
    assert records[0].partition_key == '1'
    assert records[0].schema_version == '1.0'
    # Converting approximateArrivalTimestamp from float to datetime
    assert records[0].timestamp == datetime.fromtimestamp(1545084650.987)
    assert records[1].data == b'This is only a test.'


def test_can_create_ddb_handler(sample_app: app.Chalice) -> None:
    @sample_app.on_dynamodb_record(stream_arn='arn:aws:dynamodb:...:stream', batch_size=10, starting_position='TRIM_HORIZON')
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    config: Any = sample_app.event_sources[0]
    assert config.stream_arn == 'arn:aws:dynamodb:...:stream'
    assert config.batch_size == 10
    assert config.starting_position == 'TRIM_HORIZON'
    assert config.maximum_batching_window_in_seconds == 0


def test_can_set_ddb_handler_maximum_batching_window_in_seconds(sample_app: app.Chalice) -> None:
    @sample_app.on_dynamodb_record(stream_arn='arn:aws:...:stream', batch_size=10, starting_position='TRIM_HORIZON', maximum_batching_window_in_seconds=60)
    def handler(event: Any) -> None:
        pass
    assert len(sample_app.event_sources) == 1
    config: Any = sample_app.event_sources[0]
    assert config.maximum_batching_window_in_seconds == 60


def test_can_map_ddb_event(sample_app: app.Chalice) -> None:
    @sample_app.on_dynamodb_record(stream_arn='arn:aws:...:stream')
    def handler(event: Any) -> Any:
        return event
    ddb_event: Dict[str, Any] = {
        'Records': [{
            'awsRegion': 'us-west-2',
            'dynamodb': {
                'ApproximateCreationDateTime': 1601317140.0,
                'Keys': {'PK': {'S': 'foo'}, 'SK': {'S': 'bar'}},
                'NewImage': {'PK': {'S': 'foo'}, 'SK': {'S': 'bar'}},
                'SequenceNumber': '1700000000020701978607',
                'SizeBytes': 20,
                'StreamViewType': 'NEW_AND_OLD_IMAGES'
            },
            'eventID': 'da037887f71a88a1f6f4cfd149709d5a',
            'eventName': 'INSERT',
            'eventSource': 'aws:dynamodb',
            'eventSourceARN': 'arn:aws:dynamodb:us-west-2:12345:table/MyTable/stream/2020-09-28T16:49:14.209',
            'eventVersion': '1.1'
        }]
    }
    lambda_context: FakeLambdaContext = FakeLambdaContext()
    actual_event: Any = handler(ddb_event, context=lambda_context)
    records: List[Any] = list(actual_event)
    assert len(records) == 1
    # ApproximateCreationDateTime to datetime conversion may depends on implementation
    # For testing, we just check non-null
    assert records[0].timestamp is not None
    assert records[0].keys == {'PK': {'S': 'foo'}, 'SK': {'S': 'bar'}}
    assert records[0].new_image == {'PK': {'S': 'foo'}, 'SK': {'S': 'bar'}}
    assert records[0].old_image is None
    assert records[0].sequence_number == '1700000000020701978607'
    assert records[0].size_bytes == 20
    assert records[0].stream_view_type == 'NEW_AND_OLD_IMAGES'
    assert records[0].aws_region == 'us-west-2'
    assert records[0].event_id == 'da037887f71a88a1f6f4cfd149709d5a'
    assert records[0].event_name == 'INSERT'
    assert records[0].event_source_arn == 'arn:aws:dynamodb:us-west-2:12345:table/MyTable/stream/2020-09-28T16:49:14.209'
    assert records[0].table_name == 'MyTable'


def test_bytes_when_binary_type_is_application_json() -> app.Chalice:
    demo: app.Chalice = app.Chalice('demo-app')
    demo.api.binary_types.append('application/json')

    @demo.route('/compress_response')
    def index() -> Response:
        blob: bytes = json.dumps({'hello': 'world'}).encode('utf-8')
        payload: bytes = gzip.compress(blob)
        custom_headers: Dict[str, str] = {'Content-Type': 'application/json', 'Content-Encoding': 'gzip'}
        return Response(body=payload, status_code=200, headers=custom_headers)
    return demo


def test_can_register_blueprint_on_app() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('foo')

    @foo.route('/foo')
    def first() -> None:
        pass
    myapp.register_blueprint(foo)
    assert sorted(list(myapp.routes.keys())) == ['/foo']


def test_can_combine_multiple_blueprints_in_single_app() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('foo')
    bar: app.Blueprint = app.Blueprint('bar')

    @foo.route('/foo')
    def myfoo() -> None:
        pass

    @bar.route('/bar')
    def mybar() -> None:
        pass
    myapp.register_blueprint(foo)
    myapp.register_blueprint(bar)
    assert sorted(list(myapp.routes)) == ['/bar', '/foo']


def test_can_preserve_signature_on_blueprint() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('foo')

    @foo.lambda_function()
    def first(event: Any, context: Any) -> Dict[str, str]:
        return {'foo': 'bar'}
    myapp.register_blueprint(foo)
    assert first({}, None) == {'foo': 'bar'}


def test_doc_saved_on_route() -> None:
    myapp: app.Chalice = app.Chalice('myapp')

    @myapp.route('/')
    def index() -> None:
        """My index docstring."""
        pass
    assert index.__doc__ == 'My index docstring.'


def test_blueprint_docstring_is_preserved() -> None:
    foo: app.Blueprint = app.Blueprint('foo')

    @foo.route('/foo')
    def first() -> None:
        """Blueprint docstring."""
        pass
    assert first.__doc__ == 'Blueprint docstring.'


def test_can_mount_apis_at_url_prefix() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('foo')

    @foo.route('/foo')
    def myfoo() -> None:
        pass

    @foo.route('/bar')
    def mybar() -> None:
        pass
    myapp.register_blueprint(foo, url_prefix='/myprefix')
    assert list(sorted(myapp.routes)) == ['/myprefix/bar', '/myprefix/foo']


def test_can_mount_root_url_in_blueprint() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('foo')
    root: app.Blueprint = app.Blueprint('root')

    @root.route('/')
    def myroot() -> None:
        pass

    @foo.route('/')
    def myfoo() -> None:
        pass

    @foo.route('/bar')
    def mybar() -> None:
        pass
    myapp.register_blueprint(foo, url_prefix='/foo')
    myapp.register_blueprint(root)
    assert list(sorted(myapp.routes)) == ['/', '/foo', '/foo/bar']


def test_can_combine_lambda_functions_and_routes_in_blueprints() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @foo.route('/foo')
    def myfoo() -> None:
        pass

    @foo.lambda_function()
    def myfunction(event: Any, context: Any) -> None:
        pass
    myapp.register_blueprint(foo)
    assert len(myapp.pure_lambda_functions) == 1
    lambda_function: Any = myapp.pure_lambda_functions[0]
    assert lambda_function.name == 'myfunction'
    assert lambda_function.handler_string == 'app.chalicelib.blueprints.foo.myfunction'
    assert list(myapp.routes) == ['/foo']


def test_can_mount_lambda_functions_with_name_prefix() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @foo.lambda_function()
    def myfunction(event: Any, context: Any) -> Any:
        return event
    myapp.register_blueprint(foo, name_prefix='myprefix_')
    assert len(myapp.pure_lambda_functions) == 1
    lambda_function: Any = myapp.pure_lambda_functions[0]
    assert lambda_function.name == 'myprefix_myfunction'
    assert lambda_function.handler_string == 'app.chalicelib.blueprints.foo.myfunction'
    with Client(myapp) as c:
        response: Any = c.lambda_.invoke('myprefix_myfunction', {'foo': 'bar'})
    assert response.payload == {'foo': 'bar'}


def test_can_mount_event_sources_with_blueprint() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @foo.schedule('rate(5 minutes)')
    def myfunction(event: Any) -> Any:
        return event
    myapp.register_blueprint(foo, name_prefix='myprefix_')
    assert len(myapp.event_sources) == 1
    event_source: Any = myapp.event_sources[0]
    assert event_source.name == 'myprefix_myfunction'
    assert event_source.schedule_expression == 'rate(5 minutes)'
    assert event_source.handler_string == 'app.chalicelib.blueprints.foo.myfunction'


def test_can_mount_all_decorators_in_blueprint() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @foo.route('/foo')
    def routefoo() -> None:
        pass

    @foo.lambda_function(name='mylambdafunction')
    def mylambda(event: Any, context: Any) -> None:
        pass

    @foo.schedule('rate(5 minutes)')
    def bar(event: Any) -> None:
        pass

    @foo.on_s3_event('MyBucket')
    def on_s3(event: Any) -> None:
        pass

    @foo.on_sns_message('MyTopic')
    def on_sns(event: Any) -> None:
        pass

    @foo.on_sqs_message('MyQueue')
    def on_sqs(event: Any) -> None:
        pass
    myapp.register_blueprint(foo, name_prefix='myprefix_', url_prefix='/bar')
    event_sources: List[Any] = myapp.event_sources
    assert len(event_sources) == 4
    lambda_functions: List[Any] = myapp.pure_lambda_functions
    assert len(lambda_functions) == 1
    assert lambda_functions[0].name == 'myprefix_mylambdafunction'
    assert list(myapp.routes) == ['/bar/foo']


def test_can_call_current_request_on_blueprint_when_mounted(create_event: Callable[..., Dict[str, Any]]) -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    bp: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @bp.route('/todict')
    def todict() -> Dict[str, Any]:
        return bp.current_request.to_dict()
    myapp.register_blueprint(bp)
    event: Dict[str, Any] = create_event('/todict', 'GET', {})
    response: Dict[str, Any] = json_response_body(myapp(event, context=None))
    assert isinstance(response, dict)
    assert response['method'] == 'GET'


def test_can_call_current_app_on_blueprint_when_mounted(create_event: Callable[..., Dict[str, Any]]) -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    bp: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @bp.route('/appname')
    def appname() -> Dict[str, str]:
        return {'name': bp.current_app.app_name}
    myapp.register_blueprint(bp)
    event: Dict[str, Any] = create_event('/appname', 'GET', {})
    response: Dict[str, Any] = json_response_body(myapp(event, context=None))
    assert response == {'name': 'myapp'}


def test_can_call_lambda_context_on_blueprint_when_mounted(create_event: Callable[..., Dict[str, Any]]) -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    bp: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @bp.route('/context')
    def context() -> Any:
        return bp.lambda_context
    myapp.register_blueprint(bp)
    event: Dict[str, Any] = create_event('/context', 'GET', {})
    response: Dict[str, Any] = json_response_body(myapp(event, context={'context': 'foo'}))
    assert response == {'context': 'foo'}


def test_can_access_log_when_mounted(create_event: Callable[..., Dict[str, Any]]) -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    bp: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @bp.route('/log')
    def log_message() -> Dict:
        bp.log.info('test log message')
        return {}
    myapp.register_blueprint(bp)
    event: Dict[str, Any] = create_event('/log', 'GET', {})
    response: Dict[str, Any] = json_response_body(myapp(event, context={'context': 'foo'}))
    assert response == {}


def test_can_add_authorizer_with_url_prefix_and_routes() -> None:
    myapp: app.Chalice = app.Chalice('myapp')
    foo: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')

    @foo.authorizer()
    def myauth(event: Any) -> None:
        pass

    @foo.route('/foo', authorizer=myauth)
    def routefoo() -> None:
        pass
    myapp.register_blueprint(foo, url_prefix='/bar')
    assert len(myapp.builtin_auth_handlers) == 1
    authorizer: app.BuiltinAuthConfig = myapp.builtin_auth_handlers[0]
    assert isinstance(authorizer, app.BuiltinAuthConfig)
    assert authorizer.name == 'myauth'
    assert authorizer.handler_string == 'app.chalicelib.blueprints.foo.myauth'


def test_runtime_error_if_current_request_access_on_non_registered_blueprint() -> None:
    bp: app.Blueprint = app.Blueprint('app.chalicelib.blueprints.foo')
    with pytest.raises(RuntimeError):
        _ = bp.current_request  # type: ignore


def test_every_decorator_added_to_blueprint() -> None:
    def is_public_method(obj: Any) -> bool:
        return inspect.isfunction(obj) and (not obj.__name__.startswith('_'))
    public_api: List[tuple[str, Any]] = inspect.getmembers(app.DecoratorAPI, predicate=is_public_method)
    blueprint_api: List[str] = [i[0] for i in inspect.getmembers(app.Blueprint, predicate=is_public_method)]
    for method_name, _ in public_api:
        assert method_name in blueprint_api


@pytest.mark.parametrize('input_dict', [{}, {'key': []}])
def test_multidict_raises_keyerror(input_dict: Dict[str, List[Any]]) -> None:
    d: MultiDict = MultiDict(input_dict)
    with pytest.raises(KeyError):
        _ = d['key']


def test_multidict_pop_raises_del_error() -> None:
    d: MultiDict = MultiDict({})
    with pytest.raises(KeyError):
        del d['key']


def test_multidict_getlist_does_raise_keyerror() -> None:
    d: MultiDict = MultiDict({})
    with pytest.raises(KeyError):
        d.getlist('key')


@pytest.mark.parametrize('input_dict', [{'key': ['value']}, {'key': ['']}, {'key': ['value1', 'value2', 'value3']}, {'key': ['value1', 'value2', None]}])
def test_multidict_returns_lastvalue(input_dict: Dict[str, List[Any]]) -> None:
    d: MultiDict = MultiDict(input_dict)
    assert d['key'] == input_dict['key'][-1]


@pytest.mark.parametrize('input_dict', [{'key': ['value']}, {'key': ['']}, {'key': ['value1', 'value2', 'value3']}, {'key': ['value1', 'value2', None]}])
def test_multidict_returns_all_values(input_dict: Dict[str, List[Any]]) -> None:
    d: MultiDict = MultiDict(input_dict)
    assert d.getlist('key') == input_dict['key']


@pytest.mark.parametrize('input_dict', [{'key': ['value']}, {'key': ['']}, {'key': ['value1', 'value2', 'value3']}, {'key': ['value1', 'value2', None]}])
def test_multidict_list_wont_change_source(input_dict: Dict[str, List[Any]]) -> None:
    d: MultiDict = MultiDict(input_dict)
    dict_copy: Dict[str, List[Any]] = deepcopy(input_dict)
    d.getlist('key')[0] = 'othervalue'
    assert d.getlist('key') == dict_copy['key']


@pytest.mark.parametrize('input_dict,key,popped,leftover', [
    ({'key': ['value'], 'key2': [[]]}, 'key', 'value', {'key2': []}),
    ({'key': [''], 'key2': [[]]}, 'key', '', {'key2': []}),
    ({'key': ['value1', 'value2', 'value3'], 'key2': [[]]}, 'key', 'value3', {'key2': []})
])
def test_multidict_list_can_pop_value(input_dict: Dict[str, List[Any]], key: str, popped: Any, leftover: Dict[str, Any]) -> None:
    d: MultiDict = MultiDict(input_dict)
    pop_result: Any = d.pop(key)
    assert popped == pop_result
    assert leftover == {key: d[key] for key in d}


def test_multidict_assignment() -> None:
    d: MultiDict = MultiDict({})
    d['key'] = 'value'
    assert d['key'] == 'value'


def test_multidict_get_reassigned_value() -> None:
    d: MultiDict = MultiDict({})
    d['key'] = 'value'
    assert d['key'] == 'value'
    assert d.get('key') == 'value'
    assert d.getlist('key') == ['value']


def test_multidict_get_list_wraps_key() -> None:
    d: MultiDict = MultiDict({})
    d['key'] = ['value']
    assert d.getlist('key') == [['value']]


def test_multidict_repr() -> None:
    d: MultiDict = MultiDict({'foo': ['bar', 'baz'], 'buz': ['qux']})
    rep: str = repr(d)
    assert rep.startswith('MultiDict({')
    assert "'foo': ['bar', 'baz']" in rep
    assert "'buz': ['qux']" in rep


def test_multidict_str() -> None:
    d: MultiDict = MultiDict({'foo': ['bar', 'baz'], 'buz': ['qux']})
    rep: str = str(d)
    assert rep.startswith('MultiDict({')
    assert "'foo': ['bar', 'baz']" in rep
    assert "'buz': ['qux']" in rep


def test_can_configure_websockets(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]]) -> None:
    demo, _ = sample_websocket_app
    assert len(demo.websocket_handlers) == 3, demo.websocket_handlers
    assert '$connect' in demo.websocket_handlers, demo.websocket_handlers
    assert '$disconnect' in demo.websocket_handlers, demo.websocket_handlers
    assert '$default' in demo.websocket_handlers, demo.websocket_handlers


def test_websocket_event_json_body_available(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')
    called: Dict[str, bool] = {'wascalled': False}

    @demo.on_ws_message()
    def message(event: WebsocketEvent) -> None:
        called['wascalled'] = True
        assert event.json_body == {'foo': 'bar'}
        assert event.json_body == {'foo': 'bar'}
    event: Dict[str, Any] = create_websocket_event('$default', body='{"foo": "bar"}')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    assert called['wascalled'] is True


def test_websocket_event_json_body_can_raise_error(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('demo-app')
    called: Dict[str, bool] = {'wascalled': False}

    @demo.on_ws_message()
    def message(event: WebsocketEvent) -> None:
        called['wascalled'] = True
        with pytest.raises(BadRequestError):
            _ = event.json_body
    event: Dict[str, Any] = create_websocket_event('$default', body='{"foo": "bar"')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    assert called['wascalled'] is True


def test_can_route_websocket_connect_message(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$connect')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$connect', demo)
    response: Dict[str, Any] = handler(event, context=None)
    assert response == {'statusCode': 200}
    assert len(calls) == 1
    assert calls[0][0] == 'connect'
    event_obj: WebsocketEvent = calls[0][1]
    assert isinstance(event_obj, WebsocketEvent)
    assert event_obj.domain_name == 'abcd1234.execute-api.us-west-2.amazonaws.com'
    assert event_obj.stage == 'api'
    assert event_obj.connection_id == 'ABCD1234='


def test_can_route_websocket_connect_response_dict(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_connect()
    def connect(event: WebsocketEvent) -> Dict[str, Any]:
        return dict(headers={'Sec-WebSocket-Protocol': 'Test-Protocol'}, statusCode=200, body='Connected.')
    event: Dict[str, Any] = create_websocket_event('$connect')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$connect', demo)
    response: Dict[str, Any] = handler(event, context=None)
    assert response == {'headers': {'Sec-WebSocket-Protocol': 'Test-Protocol'}, 'statusCode': 200, 'body': 'Connected.'}


def test_can_route_websocket_connect_response_obj(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_connect()
    def connect(event: WebsocketEvent) -> Response:
        return Response('Connected.', status_code=200, headers={'Sec-WebSocket-Protocol': 'Test-Protocol'})
    event: Dict[str, Any] = create_websocket_event('$connect')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$connect', demo)
    response: Dict[str, Any] = handler(event, context=None)
    assert response == {'headers': {'Sec-WebSocket-Protocol': 'Test-Protocol'}, 'multiValueHeaders': {}, 'statusCode': 200, 'body': 'Connected.'}


def test_can_route_websocket_disconnect_message(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$disconnect')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$disconnect', demo)
    response: Dict[str, Any] = handler(event, context=None)
    assert response == {'statusCode': 200}
    assert len(calls) == 1
    assert calls[0][0] == 'disconnect'
    event_obj: WebsocketEvent = calls[0][1]
    assert isinstance(event_obj, WebsocketEvent)
    assert event_obj.domain_name == 'abcd1234.execute-api.us-west-2.amazonaws.com'
    assert event_obj.stage == 'api'
    assert event_obj.connection_id == 'ABCD1234='


def test_can_route_websocket_default_message(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    response: Dict[str, Any] = handler(event, context=None)
    assert response == {'statusCode': 200}
    assert len(calls) == 1
    assert calls[0][0] == 'default'
    event_obj: WebsocketEvent = calls[0][1]
    assert isinstance(event_obj, WebsocketEvent)
    assert event_obj.domain_name == 'abcd1234.execute-api.us-west-2.amazonaws.com'
    assert event_obj.stage == 'api'
    assert event_obj.connection_id == 'ABCD1234='
    assert event_obj.body == 'foo bar'


def test_can_configure_client_on_connect(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$connect')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$connect', demo)
    handler(event, context=None)
    assert demo.websocket_api.session.calls == [('apigatewaymanagementapi', 'https://abcd1234.execute-api.us-west-2.amazonaws.com/api')]


def test_can_configure_non_aws_partition_clients(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]], monkeypatch: Any) -> None:
    monkeypatch.setenv('AWS_REGION', 'cn-north-1')
    demo, _ = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$connect', endpoint='abcd1234.execute-api.cn-north-1.amazonaws.com.cn')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$connect', demo)
    handler(event, context=None)
    assert demo.websocket_api.session.calls == [('apigatewaymanagementapi', 'https://abcd1234.execute-api.cn-north-1.amazonaws.com.cn/api')]


def test_uses_api_id_not_domain_name(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$connect')
    event['requestContext']['domainName'] = 'api.custom-domain-name.com'
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$connect', demo)
    handler(event, context=None)
    assert demo.websocket_api.session.calls == [('apigatewaymanagementapi', 'https://abcd1234.execute-api.us-west-2.amazonaws.com/api')]


def test_fallsback_to_session_if_needed(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api = WebsocketAPI(env={})
    demo.websocket_api.session = FakeSession(client, region_name='us-east-2')
    event: Dict[str, Any] = create_websocket_event('$connect')
    event['requestContext']['domainName'] = 'api.custom-domain-name.com'
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$connect', demo)
    handler(event, context=None)
    assert demo.websocket_api.session.calls == [('apigatewaymanagementapi', 'https://abcd1234.execute-api.us-east-2.amazonaws.com/api')]


def test_can_configure_client_on_disconnect(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$disconnect')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$disconnect', demo)
    handler(event, context=None)
    assert demo.websocket_api.session.calls == [('apigatewaymanagementapi', 'https://abcd1234.execute-api.us-west-2.amazonaws.com/api')]


def test_can_configure_client_on_message(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    assert demo.websocket_api.session.calls == [('apigatewaymanagementapi', 'https://abcd1234.execute-api.us-west-2.amazonaws.com/api')]


def test_does_only_configure_client_once(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    handler(event, context=None)
    assert demo.websocket_api.session.calls == [('apigatewaymanagementapi', 'https://abcd1234.execute-api.us-west-2.amazonaws.com/api')]


def test_cannot_configure_client_without_session(sample_websocket_app: tuple[app.Chalice, List[tuple[str, WebsocketEvent]]], create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo, calls = sample_websocket_app
    demo.websocket_api.session = None
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    with pytest.raises(ValueError) as e:
        handler(event, context=None)
    assert str(e.value) == 'Assign app.websocket_api.session to a boto3 session before using the WebsocketAPI'


def test_cannot_send_websocket_message_without_configure() -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        demo.websocket_api.send('connection_id', event.body)
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    event_obj: WebsocketEvent = WebsocketEvent(event, None)
    handler: Callable = demo.websocket_handlers['$default'].handler_function
    with pytest.raises(ValueError) as e:
        handler(event_obj)
    assert str(e.value) == 'WebsocketAPI.configure must be called before using the WebsocketAPI'


def test_can_close_websocket_connection(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        demo.websocket_api.close('connection_id')
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    calls: List[tuple] = client.calls['close']
    assert len(calls) == 1
    call: tuple = calls[0]
    connection_id: str = call[0]
    assert connection_id == 'connection_id'


def test_close_does_fail_if_already_disconnected(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient(errors=[FakeGoneException])
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        with pytest.raises(WebsocketDisconnectedError) as e:
            demo.websocket_api.close('connection_id')
        assert e.value.connection_id == 'connection_id'
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    calls: List[tuple] = client.calls['close']
    assert len(calls) == 1
    call: tuple = calls[0]
    connection_id: str = call[0]
    assert connection_id == 'connection_id'


def test_info_does_fail_if_already_disconnected(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient(errors=[FakeGoneException])
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        with pytest.raises(WebsocketDisconnectedError) as e:
            demo.websocket_api.info('connection_id')
        assert e.value.connection_id == 'connection_id'
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    calls: List[tuple] = client.calls['info']
    assert len(calls) == 1
    call: tuple = calls[0]
    connection_id: str = call[0]
    assert connection_id == 'connection_id'


def test_can__about_websocket_connection(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient(infos=[{'foo': 'bar'}])
    demo.websocket_api.session = FakeSession(client)
    closure: Dict[str, Any] = {}

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        closure['info'] = demo.websocket_api.info('connection_id')
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    assert closure['info'] == {'foo': 'bar'}
    calls: List[tuple] = client.calls['info']
    assert len(calls) == 1
    call: tuple = calls[0]
    connection_id: str = call[0]
    assert connection_id == 'connection_id'


def test_can_send_websocket_message(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        demo.websocket_api.send('connection_id', event.body)
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)
    calls: List[tuple] = client.calls['post_to_connection']
    assert len(calls) == 1
    call: tuple = calls[0]
    connection_id: str = call[0]
    message: Any = call[1]
    assert connection_id == 'connection_id'
    assert message == 'foo bar'


def test_does_raise_on_send_to_bad_websocket(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient(errors=[FakeGoneException])
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        with pytest.raises(WebsocketDisconnectedError) as e:
            demo.websocket_api.send('connection_id', event.body)
        assert e.value.connection_id == 'connection_id'
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)


def test_does_reraise_on_websocket_send_error(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    class SomeOtherError(Exception):
        pass
    demo: app.Chalice = app.Chalice('app-name')
    fake_418_error: Exception = SomeOtherError()
    fake_418_error.response = {'ResponseMetadata': {'HTTPStatusCode': 418}}
    client: FakeClient = FakeClient(errors=[fake_418_error])
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        with pytest.raises(SomeOtherError):
            demo.websocket_api.send('connection_id', event.body)
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    handler: WebsocketEventSourceHandler = websocket_handler_for_route('$default', demo)
    handler(event, context=None)


def test_does_reraise_on_other_send_exception(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    fake_500_error: Exception = Exception()
    fake_500_error.response = {'ResponseMetadata': {'HTTPStatusCode': 500}}
    fake_500_error.key = 'foo'
    client: FakeClient = FakeClient(errors=[fake_500_error])
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        with pytest.raises(Exception) as e:
            demo.websocket_api.send('connection_id', event.body)
        assert e.value.key == 'foo'
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    demo(event, context=None)


def test_cannot_send_message_on_unconfigured_app() -> None:
    demo: app.Chalice = app.Chalice('app-name')
    demo.websocket_api.session = None
    with pytest.raises(ValueError) as e:
        demo.websocket_api.send('connection_id', 'body')
    assert str(e.value) == 'Assign app.websocket_api.session to a boto3 session before using the WebsocketAPI'


def test_cannot_re_register_websocket_handlers(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.on_ws_message()
    def message_handler(event: WebsocketEvent) -> None:
        pass
    with pytest.raises(ValueError) as e:
        @demo.on_ws_message()
        def message_handler_2(event: WebsocketEvent) -> None:
            pass
    assert str(e.value) == "Duplicate websocket handler: 'on_ws_message'. There can only be one handler for each websocket decorator."

    @demo.on_ws_connect()
    def connect_handler(event: WebsocketEvent) -> None:
        pass
    with pytest.raises(ValueError) as e:
        @demo.on_ws_connect()
        def conncet_handler_2(event: WebsocketEvent) -> None:
            pass
    assert str(e.value) == "Duplicate websocket handler: 'on_ws_connect'. There can only be one handler for each websocket decorator."

    @demo.on_ws_disconnect()
    def disconnect_handler(event: WebsocketEvent) -> None:
        pass
    with pytest.raises(ValueError) as e:
        @demo.on_ws_disconnect()
        def disconncet_handler_2(event: WebsocketEvent) -> None:
            pass
    assert str(e.value) == "Duplicate websocket handler: 'on_ws_disconnect'. There can only be one handler for each websocket decorator."


def test_can_parse_json_websocket_body(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message(event: WebsocketEvent) -> None:
        assert event.json_body == {'foo': 'bar'}
    event: Dict[str, Any] = create_websocket_event('$default', body='{"foo": "bar"}')
    demo(event, context=None)


def test_can_access_websocket_json_body_twice(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message(event: WebsocketEvent) -> None:
        assert event.json_body == {'foo': 'bar'}
        assert event.json_body == {'foo': 'bar'}
    event: Dict[str, Any] = create_websocket_event('$default', body='{"foo": "bar"}')
    demo(event, context=None)


def test_does_raise_on_invalid_json_wbsocket_body(create_websocket_event: Callable[..., Dict[str, Any]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')
    client: FakeClient = FakeClient()
    demo.websocket_api.session = FakeSession(client)

    @demo.on_ws_message()
    def message(event: WebsocketEvent) -> None:
        with pytest.raises(BadRequestError) as e:
            _ = event.json_body
        assert 'Error Parsing JSON' in str(e.value)
    event: Dict[str, Any] = create_websocket_event('$default', body='foo bar')
    demo(event, context=None)


class TestMiddleware:
    def test_middleware_basic_api(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.middleware('all')
        def myhandler(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler', 'bucket': event.bucket})
            return get_response(event)

        @demo.middleware('all')
        def myhandler2(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler2', 'bucket': event.bucket})
            return get_response(event)

        @demo.on_s3_event('mybucket')
        def handler(event: Any) -> Dict[str, Any]:
            called.append({'name': 'main', 'bucket': event.bucket})
            return {'bucket': event.bucket}
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('handler', c.events.generate_s3_event('mybucket', 'key'))
        assert response.payload == {'bucket': 'mybucket'}
        assert called == [{'name': 'myhandler', 'bucket': 'mybucket'}, {'name': 'myhandler2', 'bucket': 'mybucket'}, {'name': 'main', 'bucket': 'mybucket'}]

    def test_can_access_original_event_and_context_in_http(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.middleware('http')
        def myhandler(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'event': event})
            return get_response(event)

        @demo.route('/')
        def index() -> Dict[str, Any]:
            called.append({'url': '/'})
            return {'hello': 'world'}
        with Client(demo) as c:
            response: Any = c.http.get('/')
        assert response.json_body == {'hello': 'world'}
        actual_event: Any = called[0]['event']
        assert actual_event.path == '/'
        assert actual_event.lambda_context.function_name == 'api_handler'
        assert actual_event.to_original_event()['requestContext']['resourcePath'] == '/'

    def test_can_short_circuit_response(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.middleware('all')
        def myhandler(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler', 'bucket': event.bucket})
            return {'short-circuit': True}

        @demo.middleware('all')
        def myhandler2(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler2', 'bucket': event.bucket})
            return get_response(event)

        @demo.on_s3_event('mybucket')
        def handler(event: Any) -> Dict[str, Any]:
            called.append({'name': 'main', 'bucket': event.bucket})
            return {'bucket': event.bucket}
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('handler', c.events.generate_s3_event('mybucket', 'key'))
        assert response.payload == {'short-circuit': True}
        assert called == [{'name': 'myhandler', 'bucket': 'mybucket'}]

    def test_can_alter_response(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.middleware('all')
        def myhandler(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler', 'bucket': event.bucket})
            response: Dict[str, Any] = get_response(event)
            response['myhandler'] = True
            return response

        @demo.middleware('all')
        def myhandler2(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler2', 'bucket': event.bucket})
            response: Dict[str, Any] = get_response(event)
            response['myhandler2'] = True
            return response

        @demo.on_s3_event('mybucket')
        def handler(event: Any) -> Dict[str, Any]:
            called.append({'name': 'main', 'bucket': event.bucket})
            return {'bucket': event.bucket}
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('handler', c.events.generate_s3_event('mybucket', 'key'))
        assert response.payload == {'bucket': 'mybucket', 'myhandler': True, 'myhandler2': True}
        assert called == [{'name': 'myhandler', 'bucket': 'mybucket'}, {'name': 'myhandler2', 'bucket': 'mybucket'}, {'name': 'main', 'bucket': 'mybucket'}]

    def test_can_change_order_of_definitions(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.on_s3_event('mybucket')
        def handler(event: Any) -> Dict[str, Any]:
            called.append({'name': 'main', 'bucket': event.bucket})
            return {'bucket': event.bucket}

        @demo.middleware('all')
        def myhandler(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler', 'bucket': event.bucket})
            response: Dict[str, Any] = get_response(event)
            response['myhandler'] = True
            return response

        @demo.middleware('all')
        def myhandler2(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'myhandler2', 'bucket': event.bucket})
            response: Dict[str, Any] = get_response(event)
            response['myhandler2'] = True
            return response
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('handler', c.events.generate_s3_event('mybucket', 'key'))
        assert response.payload == {'bucket': 'mybucket', 'myhandler': True, 'myhandler2': True}
        assert called == [{'name': 'myhandler', 'bucket': 'mybucket'}, {'name': 'myhandler2', 'bucket': 'mybucket'}, {'name': 'main', 'bucket': 'mybucket'}]

    def test_can_use_middleware_for_pure_lambda(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.middleware('all')
        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'mymiddleware', 'event': event.to_dict()})
            return get_response(event)

        @demo.lambda_function()
        def myfunction(event: Any, context: Any) -> Dict[str, str]:
            called.append({'name': 'myfunction', 'event': event})
            return {'foo': 'bar'}
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('myfunction', {'input-event': True})
        assert response.payload == {'foo': 'bar'}
        assert called == [{'name': 'mymiddleware', 'event': {'input-event': True}}, {'name': 'myfunction', 'event': {'input-event': True}}]

    def test_can_use_for_websocket_handlers(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.middleware('all')
        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'mymiddleware', 'event': event.to_dict()})
            return get_response(event)

        @demo.on_ws_message()
        def myfunction(event: WebsocketEvent) -> Dict[str, Any]:
            called.append({'name': 'myfunction', 'event': event.to_dict()})
            return {'foo': 'bar'}
        with Client(demo) as c:
            event: Dict[str, Any] = {
                'requestContext': {
                    'domainName': 'example.com',
                    'stage': 'dev',
                    'connectionId': 'abcd',
                    'apiId': 'abcd1234'
                },
                'body': 'body'
            }
            response: Any = c.lambda_.invoke('myfunction', event)
        assert response.payload == {'foo': 'bar', 'statusCode': 200}
        assert called == [{'name': 'mymiddleware', 'event': event}, {'name': 'myfunction', 'event': event}]

    def test_can_use_rest_api_for_middleware(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        @demo.middleware('all')
        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'mymiddleware', 'method': event.method})
            response: Any = get_response(event)
            response.status_code = 201
            return response

        @demo.route('/')
        def index() -> Dict[str, bool]:
            called.append({'url': '/'})
            return {'index': True}

        @demo.route('/hello')
        def hello() -> Dict[str, bool]:
            called.append({'url': '/hello'})
            return {'hello': True}
        with Client(demo) as c:
            assert c.http.get('/').json_body == {'index': True}
            response: Any = c.http.get('/hello')
            assert response.json_body == {'hello': True}
            assert response.status_code == 201
        assert called == [{'name': 'mymiddleware', 'method': 'GET'}, {'url': '/'}, {'name': 'mymiddleware', 'method': 'GET'}, {'url': '/hello'}]

    def test_error_handler_rest_api_untouched(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')

        @demo.middleware('all')
        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            return get_response(event)

        @demo.route('/error')
        def index() -> None:
            raise NotFoundError('resource not found')
        with Client(demo) as c:
            response: Any = c.http.get('/error')
            assert response.status_code == 404
            assert response.json_body == {'Code': 'NotFoundError', 'Message': 'resource not found'}

    def test_unhandled_error_not_caught(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')

        @demo.middleware('all')
        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            try:
                return get_response(event)
            except ChaliceUnhandledError:
                return Response(body={'foo': 'bar'}, status_code=200)

        @demo.route('/error')
        def index() -> None:
            raise ChaliceUnhandledError('unhandled')
        with Client(demo) as c:
            response: Any = c.http.get('/error')
            assert response.statusCode == 200
            assert response.json_body == {'foo': 'bar'}

    def test_middleware_errors_result_in_500(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')

        @demo.middleware('all')
        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            raise Exception('Error from middleware.')

        @demo.route('/')
        def index() -> Dict:
            return {}
        with Client(demo) as c:
            response: Any = c.http.get('/')
            assert response.status_code == 500
            assert response.json_body['Code'] == 'InternalServerError'

    def test_can_filter_middleware_registration(self, sample_middleware_app: app.Chalice) -> None:
        with Client(sample_middleware_app) as c:
            c.http.get('/')
            assert sample_middleware_app.calls == [{'type': 'all', 'event': 'Request'}, {'type': 'http', 'event': 'Request'}]
            sample_middleware_app.calls[:] = []
            c.lambda_.invoke('s3_handler', c.events.generate_s3_event('bucket', 'key'))
            assert sample_middleware_app.calls == [{'type': 'all', 'event': 'S3Event'}, {'type': 's3', 'event': 'S3Event'}]
            sample_middleware_app.calls[:] = []
            c.lambda_.invoke('sns_handler', c.events.generate_sns_event('topic', 'message'))
            assert sample_middleware_app.calls == [{'type': 'all', 'event': 'SNSEvent'}, {'type': 'sns', 'event': 'SNSEvent'}]
            sample_middleware_app.calls[:] = []
            c.lambda_.invoke('sqs_handler', c.events.generate_sns_event('queue', 'message'))
            assert sample_middleware_app.calls == [{'type': 'all', 'event': 'SQSEvent'}]
            sample_middleware_app.calls[:] = []
            c.lambda_.invoke('lambda_handler', {})
            assert sample_middleware_app.calls == [{'type': 'all', 'event': 'LambdaFunctionEvent'}, {'type': 'pure_lambda', 'event': 'LambdaFunctionEvent'}]
            sample_middleware_app.calls[:] = []
            c.lambda_.invoke('ws_handler', {
                'requestContext': {'domainName': 'example.com', 'stage': 'dev', 'connectionId': 'abcd', 'apiId': 'abcd1234'},
                'body': 'body'
            })
            assert sample_middleware_app.calls == [{'type': 'all', 'event': 'WebsocketEvent'}, {'type': 'websocket', 'event': 'WebsocketEvent'}]

    def test_can_register_middleware_on_blueprints(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        bp: app.Blueprint = app.Blueprint('bpmiddleware')
        called: List[Dict[str, Any]] = []

        @demo.middleware('all')
        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'fromapp', 'bucket': event.bucket})
            return get_response(event)

        @bp.middleware('all')
        def bp_middleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'frombp', 'bucket': event.bucket})
            return get_response(event)

        @bp.on_s3_event('mybucket')
        def bp_handler(event: Any) -> Dict[str, Any]:
            called.append({'name': 'bp_handler', 'bucket': event.bucket})
            return {'bucket': event.bucket}

        @bp.route('/')
        def index() -> None:
            pass

        @demo.on_s3_event('mybucket')
        def handler(event: Any) -> Dict[str, Any]:
            called.append({'name': 'main', 'bucket': event.bucket})
            return {'bucket': event.bucket}
        demo.register_blueprint(bp)
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('handler', c.events.generate_s3_event('mybucket', 'key'))
            assert response.payload == {'bucket': 'mybucket'}
            assert called == [{'name': 'fromapp', 'bucket': 'mybucket'}, {'name': 'frombp', 'bucket': 'mybucket'}, {'name': 'main', 'bucket': 'mybucket'}]
            called[:] = []
            response = c.lambda_.invoke('bp_handler', c.events.generate_s3_event('mybucket', 'key'))
            assert response.payload == {'bucket': 'mybucket'}
            assert called == [{'name': 'fromapp', 'bucket': 'mybucket'}, {'name': 'frombp', 'bucket': 'mybucket'}, {'name': 'bp_handler', 'bucket': 'mybucket'}]

    def test_blueprint_gets_middlware_added(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        bp: app.Blueprint = app.Blueprint('bpmiddleware')
        called: List[Dict[str, Any]] = []

        @bp.middleware('all')
        def bp_middleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'frombp', 'bucket': 'mybucket'})
            return get_response(event)

        @demo.on_s3_event('mybucket')
        def handler(event: Any) -> Dict[str, Any]:
            called.append({'name': 'main', 'bucket': event.bucket})
            return {'bucket': event.bucket}
        demo.register_blueprint(bp)
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('handler', c.events.generate_s3_event('mybucket', 'key'))
        assert response.payload == {'bucket': 'mybucket'}
        assert called == [{'name': 'frombp', 'bucket': 'mybucket'}, {'name': 'main', 'bucket': 'mybucket'}]

    def test_can_register_middleware_without_decorator(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        def mymiddleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'mymiddleware', 'event': event.to_dict()})
            return get_response(event)

        @demo.lambda_function()
        def myfunction(event: Any, context: Any) -> Dict[str, str]:
            called.append({'name': 'myfunction', 'event': event})
            return {'foo': 'bar'}
        demo.register_middleware(mymiddleware, 'all')
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('myfunction', {'input-event': True})
        assert response.payload == {'foo': 'bar'}
        assert called == [{'name': 'mymiddleware', 'event': {'input-event': True}}, {'name': 'myfunction', 'event': {'input-event': True}}]

    def test_can_convert_existing_lambda_decorator_to_middleware(self) -> None:
        demo: app.Chalice = app.Chalice('app-name')
        called: List[Dict[str, Any]] = []

        def mydecorator(func: Callable) -> Callable:
            def _wrapped(event: Any, context: Any) -> Any:
                called.append({'name': 'wrapped', 'event': event})
                return func(event, context)
            return _wrapped

        @demo.middleware('all')
        def second_middleware(event: Any, get_response: Callable[[Any], Any]) -> Any:
            called.append({'name': 'second', 'event': event.to_dict()})
            return get_response(event)

        @demo.lambda_function()
        def myfunction(event: Any, context: Any) -> Dict[str, str]:
            called.append({'name': 'myfunction', 'event': event})
            return {'foo': 'bar'}
        demo.register_middleware(ConvertToMiddleware(mydecorator))
        with Client(demo) as c:
            response: Any = c.lambda_.invoke('myfunction', {'input-event': True})
        assert response.payload == {'foo': 'bar'}
        assert called == [{'name': 'second', 'event': {'input-event': True}}, {'name': 'wrapped', 'event': {'input-event': True}}, {'name': 'myfunction', 'event': {'input-event': True}}]


# End of annotated code.
