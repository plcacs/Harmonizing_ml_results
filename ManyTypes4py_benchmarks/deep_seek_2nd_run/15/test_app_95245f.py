import sys
import base64
import logging
import json
import gzip
import inspect
import collections
from copy import deepcopy
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    TypeVar,
    Tuple,
    Set,
    DefaultDict,
    Iterator,
    cast,
    overload,
)
from typing_extensions import Literal
import pytest
from pytest import fixture
import hypothesis.strategies as st
from hypothesis import given, assume
import six
from chalice import app
from chalice import NotFoundError
from chalice.test import Client
from chalice.app import (
    APIGateway,
    Request,
    Response,
    handle_extra_types,
    MultiDict,
    WebsocketEvent,
    BadRequestError,
    WebsocketDisconnectedError,
    WebsocketEventSourceHandler,
    ConvertToMiddleware,
    WebsocketAPI,
    ChaliceUnhandledError,
)
from chalice import __version__ as chalice_version
from chalice.deploy.validate import ExperimentalFeatureError
from chalice.deploy.validate import validate_feature_flags

T = TypeVar('T')
STR_MAP: st.SearchStrategy[Dict[str, str]] = st.dictionaries(st.text(), st.text())
STR_TO_LIST_MAP: st.SearchStrategy[Dict[str, List[str]]] = st.dictionaries(
    st.text(), st.lists(elements=st.text(), min_size=1, max_size=5)
HTTP_METHOD: st.SearchStrategy[str] = st.sampled_from(
    ['GET', 'POST', 'PUT', 'PATCH', 'OPTIONS', 'HEAD', 'DELETE'])
PATHS: st.SearchStrategy[str] = st.sampled_from(['/', '/foo/bar'])
HTTP_BODY: st.SearchStrategy[Optional[str]] = st.none() | st.text()
HTTP_REQUEST: st.SearchStrategy[Dict[str, Any]] = st.fixed_dictionaries({
    'query_params': STR_TO_LIST_MAP,
    'headers': STR_MAP,
    'uri_params': STR_MAP,
    'method': HTTP_METHOD,
    'body': HTTP_BODY,
    'context': STR_MAP,
    'stage_vars': STR_MAP,
    'is_base64_encoded': st.booleans(),
    'path': PATHS,
})
HTTP_REQUEST = st.fixed_dictionaries({
    'multiValueQueryStringParameters': st.fixed_dictionaries({}),
    'headers': STR_MAP,
    'pathParameters': STR_MAP,
    'requestContext': st.fixed_dictionaries({
        'httpMethod': HTTP_METHOD,
        'resourcePath': PATHS,
    }),
    'body': HTTP_BODY,
    'stageVariables': STR_MAP,
    'isBase64Encoded': st.booleans(),
})
BINARY_TYPES: List[str] = APIGateway().binary_types


class FakeLambdaContextIdentity:
    def __init__(self, cognito_identity_id: str, cognito_identity_pool_id: str) -> None:
        self.cognito_identity_id = cognito_identity_id
        self.cognito_identity_pool_id = cognito_identity_pool_id


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
        self.GoneException: Type[FakeGoneException] = FakeGoneException


class FakeClient:
    def __init__(
        self,
        errors: Optional[List[Exception]] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if errors is None:
            errors = []
        if infos is None:
            infos = []
        self._errors: List[Exception] = errors
        self._infos: List[Dict[str, Any]] = infos
        self.calls: DefaultDict[str, List[Tuple[Any, ...]]] = collections.defaultdict(
            lambda: [])
        self.exceptions: FakeExceptionFactory = FakeExceptionFactory()

    def post_to_connection(self, ConnectionId: str, Data: str) -> None:
        self._call('post_to_connection', ConnectionId, Data)

    def delete_connection(self, ConnectionId: str) -> None:
        self._call('close', ConnectionId)

    def get_connection(self, ConnectionId: str) -> Optional[Dict[str, Any]]:
        self._call('info', ConnectionId)
        if self._infos is not None:
            info = self._infos.pop()
            return info

    def _call(self, name: str, *args: Any) -> None:
        self.calls[name].append(tuple(args))
        if self._errors:
            error = self._errors.pop()
            raise error


class FakeSession:
    def __init__(
        self,
        client: Optional[FakeClient] = None,
        region_name: str = 'us-west-2',
    ) -> None:
        self.calls: List[Tuple[str, Optional[str]]] = []
        self._client: Optional[FakeClient] = client
        self.region_name: str = region_name

    def client(self, name: str, endpoint_url: Optional[str] = None) -> Optional[FakeClient]:
        self.calls.append((name, endpoint_url))
        return self._client


@pytest.fixture
def view_function() -> Callable[[], Dict[str, str]]:
    def _func() -> Dict[str, str]:
        return {'hello': 'world'}
    return _func


def create_request_with_content_type(content_type: str) -> Request:
    body = '{"json": "body"}'
    event = {
        'multiValueQueryStringParameters': '',
        'headers': {'Content-Type': content_type},
        'pathParameters': {},
        'requestContext': {'httpMethod': 'GET', 'resourcePath': '/'},
        'body': body,
        'stageVariables': {},
        'isBase64Encoded': False,
    }
    return app.Request(event, FakeLambdaContext())


def assert_response_body_is(response: Dict[str, Any], body: Dict[str, Any]) -> None:
    assert json.loads(response['body']) == body


def json_response_body(response: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(response['body'])


def assert_requires_opt_in(app: app.Chalice, flag: str) -> None:
    with pytest.raises(ExperimentalFeatureError):
        validate_feature_flags(app)
    app.experimental_feature_flags.add(flag)
    try:
        validate_feature_flags(app)
    except ExperimentalFeatureError:
        raise AssertionError(
            'Opting in to feature %s still raises an ExperimentalFeatureError.' % flag)


def websocket_handler_for_route(
    route: str,
    app: app.Chalice,
) -> WebsocketEventSourceHandler:
    fn = app.websocket_handlers[route].handler_function
    handler = WebsocketEventSourceHandler(fn, WebsocketEvent, app.websocket_api)
    return handler


@fixture
def sample_app() -> app.Chalice:
    demo = app.Chalice('demo-app')

    @demo.route('/index', methods=['GET'])
    def index() -> Dict[str, str]:
        return {'hello': 'world'}

    @demo.route('/name/{name}', methods=['GET'])
    def name(name: str) -> Dict[str, str]:
        return {'provided-name': name}
    return demo


@fixture
def sample_app_with_cors() -> app.Chalice:
    demo = app.Chalice('demo-app')

    @demo.route('/image', methods=['POST'], cors=True, content_types=['image/gif'])
    def image() -> Dict[str, bool]:
        return {'image': True}
    return demo


@fixture
def sample_app_with_default_cors() -> app.Chalice:
    demo = app.Chalice('demo-app')
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
def sample_websocket_app() -> Tuple[app.Chalice, List[Tuple[str, WebsocketEvent]]]:
    demo = app.Chalice('app-name')
    demo.websocket_api.session = FakeSession()
    calls: List[Tuple[str, WebsocketEvent]] = []

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
    demo = app.Chalice('app-name')
    demo.calls = []

    @demo.middleware('all')
    def mymiddleware(event: Any, get_response: Callable) -> Any:
        demo.calls.append({'type': 'all', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('s3')
    def mymiddleware_s3(event: Any, get_response: Callable) -> Any:
        demo.calls.append({'type': 's3', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('sns')
    def mymiddleware_sns(event: Any, get_response: Callable) -> Any:
        demo.calls.append({'type': 'sns', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('http')
    def mymiddleware_http(event: Any, get_response: Callable) -> Any:
        demo.calls.append({'type': 'http', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('websocket')
    def mymiddleware_websocket(event: Any, get_response: Callable) -> Any:
        demo.calls.append({'type': 'websocket', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.middleware('pure_lambda')
    def mymiddleware_pure_lambda(event: Any, get_response: Callable) -> Any:
        demo.calls.append({'type': 'pure_lambda', 'event': event.__class__.__name__})
        return get_response(event)

    @demo.route('/')
    def index() -> Dict[str, Any]:
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
    def ws_handler(event: Any) -> None:
        pass
    return demo


@fixture
def auth_request() -> app.AuthRequest:
    method_arn = 'arn:aws:execute-api:us-west-2:123:rest-api-id/dev/GET/needs/auth'
    request = app.AuthRequest('TOKEN', 'authtoken', method_arn)
    return request


@pytest.mark.skipif(
    sys.version[0] == '2',
    reason='Test is irrelevant under python 2, since str and bytes are interchangeable.')
def test_invalid_binary_response_body_throws_value_error(sample_app: app.Chalice) -> None:
    response = app.Response(
        status_code=200,
        body={'foo': 'bar'},
        headers={'Content-Type': 'application/octet-stream'},
    )
    with pytest.raises(ValueError):
        response.to_dict(sample_app.api.binary_types)


def test_invalid_JSON_response_body_throws_type_error(sample_app: app.Chalice) -> None:
    response = app.Response(
        status_code=200,
        body={'foo': object()},
        headers={'Content-Type': 'application/json'},
    )
    with pytest.raises(TypeError):
        response.to_dict()


def test_can_encode_binary_body_as_base64(sample_app: app.Chalice) -> None:
    response = app.Response(
        status_code=200,
        body=b'foobar',
        headers={'Content-Type': 'application/octet-stream'},
    )
    encoded_response = response.to_dict(sample_app.api.binary_types)
    assert encoded_response['body'] == 'Zm9vYmFy'


def test_can_return_unicode_body(sample_app: app.Chalice) -> None:
    unicode_data = u'✓'
    response = app.Response(status_code=200, body=unicode_data)
    encoded_response = response.to_dict()
    assert encoded_response['body'] == unicode_data


def test_can_encode_binary_body_with_header_charset(sample_app: app.Chalice) -> None:
    response = app.Response(
        status_code=200,
        body=b'foobar',
        headers={'Content-Type': 'application/octet-stream; charset=binary'},
    )
    encoded_response = response.to_dict(sample_app.api.binary_types)
    assert encoded_response['body'] == 'Zm9vYmFy'


def test_can_encode_binary_json(sample_app: app.Chalice) -> None:
    sample_app.api.binary_types.extend(['application/json'])
    response = app.Response(
        status_code=200,
        body={'foo': 'bar'},
        headers={'Content-Type': 'application/json'},
    )
    encoded_response = response.to_dict(sample_app.api.binary_types)
    assert encoded_response['body'] == 'eyJmb28iOiJiYXIifQ=='


def test_wildcard_accepts_with_native_python_types_serializes_json(
    sample_app: app.Chalice,
    create_event: Callable,
) -> None:
    sample_app.api.binary_types = ['*/*']

    @sample_app.route('/py-dict')
    def py_dict() -> Dict[str, str]:
        return {'foo': 'bar'}
    event = create_event('/py-dict', 'GET', {})
    event['headers']['Accept'] = '*/*'
    response = sample_app(event, context=None)
    assert base64.b64decode(response['body']) == b'{"foo":"bar"}'
    assert response['isBase64Encoded']


def test_wildcard_accepts_with_response_class(
    sample_app: app.Chalice,
    create_event: Callable,
) -> None:
    sample_app.api.binary_types = ['*/*']

    @sample_app.route('/py-dict')
    def py_dict() -> Response:
        return Response(
            body=json.dumps({'foo': 'bar'}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            status_code=200,
        )
    event = create_event('/py-dict', 'GET', {})
    event['headers']['Accept'] = '*/*'
    response = sample_app(event, context=None)
    assert base64.b64decode(response['body']) == b'{"foo": "bar"}'
    assert response['isBase64Encoded']


def test_can_parse_route_view_args() -> None:
    entry = app.RouteEntry(
        lambda: {'foo': 'bar'},
        'view-name',
        '/foo/{bar}/baz/{qux}',
        method='GET',
    )
    assert entry.view_args == ['bar', 'qux']


def test_can_route_single_view() -> None:
    demo = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Dict[str, Any]:
        return {}
    assert demo.routes['/index']['GET'] == app.RouteEntry(
        index_view,
        'index_view',
        '/index',
        'GET',
        content_types=['application/json'],
    )


def test_can_handle_multiple_routes() -> None:
    demo = app.Chalice('app-name')

    @demo.route('/index')
    def index_view() -> Dict[str, Any]:
        return {}

    @demo.route('/other')
    def other_view() -> Dict[str, Any]:
        return {}
    assert len(demo.routes) == 2, demo.routes
    assert '/index' in demo.routes, demo.routes
    assert '/other' in demo.routes, demo.routes
    assert demo.routes['/index']['GET'].view_function == index_view
    assert demo.routes['/other']['GET'].view_function == other_view


def test_error_on_unknown_event(sample_app: app.Chalice) -> None:
    bad_event = {'random': 'event'}
    raw_response = sample_app(bad_event, context=None)
    assert raw_response['statusCode'] == 500
    assert json_response_body(raw_response)['Code'] == 'InternalServer