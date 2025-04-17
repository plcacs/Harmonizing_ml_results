import json
import os
import time
import shutil
import uuid
from unittest import mock
from typing import Any, Callable, Dict, Optional

import botocore.session
import pytest
import requests
import websocket

from chalice.cli.factory import CLIFactory
from chalice.utils import OSUtils, UI
from chalice.deploy.deployer import ChaliceDeploymentError
from chalice.config import DeployedResources


CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR: str = os.path.join(CURRENT_DIR, 'testapp')
APP_FILE: str = os.path.join(PROJECT_DIR, 'app.py')
RANDOM_APP_NAME: str = f'smoketest-{str(uuid.uuid4())[:13]}'


T = Callable[..., Any]

def retry(max_attempts: int, delay: float) -> Callable[[T], T]:
    def _create_wrapped_retry_function(function: T) -> T:
        def _wrapped_with_retry(*args: Any, **kwargs: Any) -> Any:
            for _ in range(max_attempts):
                result = function(*args, **kwargs)
                if result is not None:
                    return result
                time.sleep(delay)
            raise RuntimeError(
                "Exhausted max retries of %s for function: %s"
                % (max_attempts, function)
            )
        return _wrapped_with_retry  # type: ignore
    return _create_wrapped_retry_function


class InternalServerError(Exception):
    pass


class SmokeTestApplication:
    # Number of seconds to wait after redeploy before starting
    # to poll for successful 200.
    _REDEPLOY_SLEEP: int = 30
    # Seconds to wait between poll attempts after redeploy.
    _POLLING_DELAY: int = 5
    # Number of successful wait attempts before we consider the app
    # stabilized.
    _NUM_SUCCESS: int = 3

    def __init__(
        self,
        deployed_values: Dict[str, Any],
        stage_name: str,
        app_name: str,
        app_dir: str,
        region: str
    ) -> None:
        self._deployed_resources: DeployedResources = DeployedResources(deployed_values)
        self.stage_name: str = stage_name
        self.app_name: str = app_name
        # The name of the tmpdir where the app is copied.
        self.app_dir: str = app_dir
        self._has_redeployed: bool = False
        self._region: str = region

    @property
    def url(self) -> str:
        return (
            "https://{rest_api_id}.execute-api.{region}.amazonaws.com/"
            "{api_gateway_stage}".format(
                rest_api_id=self.rest_api_id,
                region=self._region,
                api_gateway_stage='api'
            )
        )

    @property
    def rest_api_id(self) -> str:
        return self._deployed_resources.resource_values('rest_api')['rest_api_id']

    @property
    def websocket_api_id(self) -> str:
        return self._deployed_resources.resource_values('websocket_api')['websocket_api_id']

    @property
    def websocket_connect_url(self) -> str:
        return (
            "wss://{websocket_api_id}.execute-api.{region}.amazonaws.com/"
            "{api_gateway_stage}".format(
                websocket_api_id=self.websocket_api_id,
                region=self._region,
                api_gateway_stage='api',
            )
        )

    @retry(max_attempts=10, delay=5)
    def get_json(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            return self._get_json(url)
        except requests.exceptions.HTTPError:
            pass

    def _get_json(self, url: str) -> Dict[str, Any]:
        if not url.startswith('/'):
            url = '/' + url
        response = requests.get(self.url + url)
        response.raise_for_status()
        return response.json()

    @retry(max_attempts=10, delay=5)
    def get_response(self, url: str, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        try:
            return self._send_request('GET', url, headers=headers)
        except InternalServerError:
            pass

    def _send_request(
        self,
        http_method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None
    ) -> requests.Response:
        kwargs: Dict[str, Any] = {}
        if headers is not None:
            kwargs['headers'] = headers
        if data is not None:
            kwargs['data'] = data
        response = requests.request(http_method, self.url + url, **kwargs)
        if response.status_code >= 500:
            raise InternalServerError()
        return response

    @retry(max_attempts=10, delay=5)
    def post_response(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None
    ) -> requests.Response:
        try:
            return self._send_request('POST', url, headers=headers, data=data)
        except InternalServerError:
            pass

    @retry(max_attempts=10, delay=5)
    def put_response(self, url: str) -> requests.Response:
        try:
            return self._send_request('PUT', url)
        except InternalServerError:
            pass

    @retry(max_attempts=10, delay=5)
    def options_response(self, url: str) -> requests.Response:
        try:
            return self._send_request('OPTIONS', url)
        except InternalServerError:
            pass

    def redeploy_once(self) -> None:
        # Redeploy the application once.  If a redeploy
        # has already happened, this function is a noop.
        if self._has_redeployed:
            return
        new_file: str = os.path.join(self.app_dir, 'app-redeploy.py')
        original_app_py: str = os.path.join(self.app_dir, 'app.py')
        shutil.move(original_app_py, original_app_py + '.bak')
        shutil.copy(new_file, original_app_py)
        application: SmokeTestApplication = _deploy_app(self.app_dir)
        self._has_redeployed = True
        # Give it settling time before running more tests.
        time.sleep(self._REDEPLOY_SLEEP)
        for _ in range(self._NUM_SUCCESS):
            self._wait_for_stablize()
            time.sleep(self._POLLING_DELAY)

    def _wait_for_stablize(self) -> Optional[Dict[str, Any]]:
        # After a deployment we sometimes need to wait for
        # API Gateway to propagate all of its changes.
        # We're going to give it num_attempts to give us a
        # 200 response before failing.
        return self.get_json('/')


@pytest.fixture
def apig_client() -> botocore.client.BaseClient:
    s = botocore.session.get_session()
    return s.create_client('apigateway')


@pytest.fixture(scope='module')
def smoke_test_app(tmpdir_factory: pytest.TempdirFactory) -> SmokeTestApplication:
    # We can't use the monkeypatch fixture here because this is a module scope
    # fixture and monkeypatch is a function scoped fixture.
    os.environ['APP_NAME'] = RANDOM_APP_NAME
    tmpdir: str = str(tmpdir_factory.mktemp(RANDOM_APP_NAME))
    OSUtils().copytree(PROJECT_DIR, tmpdir)
    _inject_app_name(tmpdir)
    application: SmokeTestApplication = _deploy_app(tmpdir)
    yield application
    _delete_app(application, tmpdir)
    os.environ.pop('APP_NAME')


def _inject_app_name(dirname: str) -> None:
    config_filename: str = os.path.join(dirname, '.chalice', 'config.json')
    with open(config_filename) as f:
        data: Dict[str, Any] = json.load(f)
    data['app_name'] = RANDOM_APP_NAME
    data['stages']['dev']['environment_variables']['APP_NAME'] = RANDOM_APP_NAME
    with open(config_filename, 'w') as f:
        f.write(json.dumps(data, indent=2))


def _deploy_app(temp_dirname: str) -> SmokeTestApplication:
    factory: CLIFactory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(
        chalice_stage_name='dev',
        autogen_policy=True
    )
    session: botocore.session.Session = factory.create_botocore_session()
    deployer = factory.create_default_deployer(session, config, UI())
    region: str = session.get_config_variable('region')  # type: ignore
    deployed: Dict[str, Any] = _deploy_with_retries(deployer, config)
    application = SmokeTestApplication(
        region=region,
        deployed_values=deployed,
        stage_name='dev',
        app_name=RANDOM_APP_NAME,
        app_dir=temp_dirname,
    )
    return application


@retry(max_attempts=10, delay=20)
def _deploy_with_retries(deployer: Any, config: Any) -> Dict[str, Any]:
    try:
        deployed_stages: Dict[str, Any] = deployer.deploy(config, 'dev')
        return deployed_stages
    except ChaliceDeploymentError as e:
        # API Gateway aggressively throttles deployments.
        # If we run into this case, we just wait and try
        # again.
        error_code: Optional[str] = _get_error_code_from_exception(e)
        if error_code != 'TooManyRequestsException':
            raise
    return {}  # To satisfy return type, though logically unreachable


def _get_error_code_from_exception(exception: ChaliceDeploymentError) -> Optional[str]:
    error_response: Optional[Dict[str, Any]] = getattr(exception.original_error, 'response', None)
    if error_response is None:
        return None
    return error_response.get('Error', {}).get('Code')


def _delete_app(application: SmokeTestApplication, temp_dirname: str) -> None:
    factory: CLIFactory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(
        chalice_stage_name='dev',
        autogen_policy=True
    )
    session: botocore.session.Session = factory.create_botocore_session()
    deployer = factory.create_deletion_deployer(session, UI())
    _deploy_with_retries(deployer, config)


def test_returns_simple_response(smoke_test_app: SmokeTestApplication) -> None:
    assert smoke_test_app.get_json('/') == {'hello': 'world'}


def test_can_have_nested_routes(smoke_test_app: SmokeTestApplication) -> None:
    assert smoke_test_app.get_json('/a/b/c/d/e/f/g') == {'nested': True}


def test_supports_path_params(smoke_test_app: SmokeTestApplication) -> None:
    assert smoke_test_app.get_json('/path/foo') == {'path': 'foo'}
    assert smoke_test_app.get_json('/path/bar') == {'path': 'bar'}


def test_path_params_mapped_in_api(
    smoke_test_app: SmokeTestApplication,
    apig_client: botocore.client.BaseClient
) -> None:
    # Use the API Gateway API to ensure that path parameters
    # are modeled as such.  Otherwise this will break
    # SDK generation and any future features that depend
    # on params.  We could try to verify the generated
    # javascript SDK looks ok.  Instead we're going to
    # query the resources we've created in API gateway
    # and make sure requestParameters are present.
    rest_api_id: str = smoke_test_app.rest_api_id
    response = apig_client.get_export(
        restApiId=rest_api_id,
        stageName='api',
        exportType='swagger'
    )
    swagger_doc: Dict[str, Any] = json.loads(response['body'].read())
    route_config: Dict[str, Any] = swagger_doc['paths']['/path/{name}']['get']
    assert route_config.get('parameters', {}) == [
        {'name': 'name', 'in': 'path', 'required': True, 'type': 'string'},
    ]


def test_single_doc_mapped_in_api(
    smoke_test_app: SmokeTestApplication,
    apig_client: botocore.client.BaseClient
) -> None:
    # We'll use the same API Gateway technique as in
    # test_path_params_mapped_in_api()
    rest_api_id: str = smoke_test_app.rest_api_id
    doc_parts = apig_client.get_documentation_parts(
        restApiId=rest_api_id,
        type='METHOD',
        path='/singledoc'
    )
    doc_props: Dict[str, Any] = json.loads(doc_parts['items'][0]['properties'])
    assert 'summary' in doc_props
    assert 'description' not in doc_props
    assert doc_props['summary'] == 'Single line docstring.'


def test_multi_doc_mapped_in_api(
    smoke_test_app: SmokeTestApplication,
    apig_client: botocore.client.BaseClient
) -> None:
    # We'll use the same API Gateway technique as in
    # test_path_params_mapped_in_api()
    rest_api_id: str = smoke_test_app.rest_api_id
    doc_parts = apig_client.get_documentation_parts(
        restApiId=rest_api_id,
        type='METHOD',
        path='/multidoc'
    )
    doc_props: Dict[str, Any] = json.loads(doc_parts['items'][0]['properties'])
    assert 'summary' in doc_props
    assert 'description' in doc_props
    assert doc_props['summary'] == 'Multi-line docstring.'
    assert doc_props['description'] == 'And here is another line.'


@retry(max_attempts=18, delay=10)
def _get_resource_id(
    apig_client: botocore.client.BaseClient,
    rest_api_id: str,
    path: str
) -> Optional[str]:
    # This is the resource id for the '/path/{name}'
    # route.  As far as I know this is the best way to get
    # this id.
    matches: List[Dict[str, Any]] = [
        resource for resource in
        apig_client.get_resources(restApiId=rest_api_id)['items']
        if resource['path'] == path
    ]
    if matches:
        return matches[0]['id']
    return None


def test_supports_post(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.post_response('/post')
    response.raise_for_status()
    assert response.json() == {'success': True}
    with pytest.raises(requests.HTTPError):
        # Only POST is supported.
        response = smoke_test_app.get_response('/post')
        response.raise_for_status()


def test_supports_put(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.put_response('/put')
    response.raise_for_status()
    assert response.json() == {'success': True}
    with pytest.raises(requests.HTTPError):
        # Only PUT is supported.
        response = smoke_test_app.get_response('/put')
        response.raise_for_status()


def test_supports_shared_routes(smoke_test_app: SmokeTestApplication) -> None:
    response: Optional[Dict[str, Any]] = smoke_test_app.get_json('/shared')
    assert response == {'method': 'GET'}
    response_post: requests.Response = smoke_test_app.post_response('/shared')
    assert response_post.json() == {'method': 'POST'}


def test_can_read_json_body_on_post(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.post_response(
        '/jsonpost',
        data=json.dumps({'hello': 'world'}),
        headers={'Content-Type': 'application/json'}
    )
    response.raise_for_status()
    assert response.json() == {'json_body': {'hello': 'world'}}


def test_can_raise_bad_request(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.get_response('/badrequest')
    assert response.status_code == 400
    assert response.json()['Code'] == 'BadRequestError'
    assert response.json()['Message'] == 'BadRequestError: Bad request.'


def test_can_raise_not_found(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.get_response('/notfound')
    assert response.status_code == 404
    assert response.json()['Code'] == 'NotFoundError'


def test_unexpected_error_raises_500_in_prod_mode(smoke_test_app: SmokeTestApplication) -> None:
    # Can't use smoke_test_app.get_response() because we're explicitly
    # testing for a 500.
    response: requests.Response = requests.get(smoke_test_app.url + '/arbitrary-error')
    assert response.status_code == 500
    assert response.json()['Code'] == 'InternalServerError'
    assert 'internal server error' in response.json()['Message']


def test_can_route_multiple_methods_in_one_view(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.get_response('/multimethod')
    response.raise_for_status()
    assert response.json()['method'] == 'GET'

    response_post: requests.Response = smoke_test_app.post_response('/multimethod')
    response_post.raise_for_status()
    assert response_post.json()['method'] == 'POST'


def test_form_encoded_content_type(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.post_response(
        '/formencoded',
        data={'foo': 'bar'}
    )
    response.raise_for_status()
    assert response.json() == {'parsed': {'foo': ['bar']}}


def test_can_round_trip_binary(smoke_test_app: SmokeTestApplication) -> None:
    # xde xed xbe xef will fail unicode decoding because xbe is an invalid
    # start byte in utf-8.
    bin_data: bytes = b'\xDE\xAD\xBE\xEF'
    response: requests.Response = smoke_test_app.post_response(
        '/binary',
        headers={
            'Content-Type': 'application/octet-stream',
            'Accept': 'application/octet-stream'
        },
        data=bin_data
    )
    response.raise_for_status()
    assert response.content == bin_data


def test_can_round_trip_binary_custom_content_type(smoke_test_app: SmokeTestApplication) -> None:
    bin_data: bytes = b'\xDE\xAD\xBE\xEF'
    response: requests.Response = smoke_test_app.post_response(
        '/custom-binary',
        headers={
            'Content-Type': 'application/binary',
            'Accept': 'application/binary'
        },
        data=bin_data
    )
    assert response.content == bin_data


def test_can_return_default_binary_data_to_a_browser(smoke_test_app: SmokeTestApplication) -> None:
    base64encoded_response: bytes = b'3q2+7w=='
    accept: str = 'text/html,application/xhtml+xml;q=0.9,image/webp,*/*;q=0.8'
    response: requests.Response = smoke_test_app.get_response(
        '/get-binary',
        headers={'Accept': accept}
    )
    response.raise_for_status()
    assert response.content == base64encoded_response


def _assert_contains_access_control_allow_methods(
    headers: Dict[str, str],
    methods: List[str]
) -> None:
    actual_methods: List[str] = headers['Access-Control-Allow-Methods'].split(',')
    assert sorted(methods) == sorted(actual_methods), (
        'The expected allowed methods does not match the actual allowed '
        'methods for CORS.'
    )


def test_can_support_cors(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.get_response('/cors')
    response.raise_for_status()
    assert response.headers['Access-Control-Allow-Origin'] == '*'

    # Should also have injected an OPTIONs request.
    response_options: requests.Response = smoke_test_app.options_response('/cors')
    response_options.raise_for_status()
    headers: Dict[str, str] = response_options.headers
    assert headers['Access-Control-Allow-Origin'] == '*'
    assert headers['Access-Control-Allow-Headers'] == (
        'Authorization,Content-Type,X-Amz-Date,X-Amz-Security-Token,'
        'X-Api-Key'
    )
    _assert_contains_access_control_allow_methods(
        headers, ['GET', 'POST', 'PUT', 'OPTIONS']
    )


def test_can_support_custom_cors(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.get_response('/custom_cors')
    response.raise_for_status()
    expected_allow_origin: str = 'https://foo.example.com'
    assert response.headers['Access-Control-Allow-Origin'] == expected_allow_origin

    # Should also have injected an OPTIONs request.
    response_options: requests.Response = smoke_test_app.options_response('/custom_cors')
    response_options.raise_for_status()
    headers: Dict[str, str] = response_options.headers
    assert headers['Access-Control-Allow-Origin'] == expected_allow_origin
    assert headers['Access-Control-Allow-Headers'] == (
        'Authorization,Content-Type,X-Amz-Date,X-Amz-Security-Token,'
        'X-Api-Key,X-Special-Header'
    )
    _assert_contains_access_control_allow_methods(
        headers, ['GET', 'POST', 'PUT', 'OPTIONS']
    )
    assert headers['Access-Control-Max-Age'] == '600'
    assert headers['Access-Control-Expose-Headers'] == 'X-Special-Header'
    assert headers['Access-Control-Allow-Credentials'] == 'true'


def test_to_dict_is_also_json_serializable(smoke_test_app: SmokeTestApplication) -> None:
    assert 'headers' in smoke_test_app.get_json('/todict')


def test_multifile_support(smoke_test_app: SmokeTestApplication) -> None:
    response: Optional[Dict[str, Any]] = smoke_test_app.get_json('/multifile')
    assert response == {'message': 'success'}


def test_custom_response(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.get_response('/custom-response')
    response.raise_for_status()
    # Custom header
    assert response.headers['Content-Type'] == 'text/plain'
    # Multi headers
    assert response.headers['Set-Cookie'] == 'key=value, foo=bar'
    # Custom status code
    assert response.status_code == 204


def test_api_key_required_fails_with_no_key(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.get_response('/api-key-required')
    # Request should fail because we're not providing
    # an API key.
    assert response.status_code == 403


def test_can_handle_charset(smoke_test_app: SmokeTestApplication) -> None:
    # Should pass content type validation even with charset specified.
    response: requests.Response = smoke_test_app.get_response(
        '/json-only',
        headers={'Content-Type': 'application/json; charset=utf-8'}
    )
    assert response.status_code == 200


def test_can_use_builtin_custom_auth(smoke_test_app: SmokeTestApplication) -> None:
    url: str = '/builtin-auth'
    # First time without an Auth header, we should fail.
    response: requests.Response = smoke_test_app.get_response(url)
    assert response.status_code == 401
    # Now with the proper auth header, things should work.
    response_auth: requests.Response = smoke_test_app.get_response(
        url, headers={'Authorization': 'yes'}
    )
    assert response_auth.status_code == 200
    context: Dict[str, Any] = response_auth.json()['context']
    assert 'authorizer' in context
    # The keyval context we added should also be in the authorizer
    # dict.
    assert context['authorizer']['foo'] == 'bar'


def test_can_use_shared_auth(smoke_test_app: SmokeTestApplication) -> None:
    response_get: requests.Response = smoke_test_app.get_response('/fake-profile')
    # GETs are allowed
    assert response_get.status_code == 200
    # However, POSTs require auth.
    # This has the same auth config as /builtin-auth,
    # so we're testing the auth handler can be shared.
    response_post: requests.Response = smoke_test_app.post_response('/fake-profile')
    assert response_post.status_code == 401
    response_post_auth: requests.Response = smoke_test_app.post_response(
        '/fake-profile',
        headers={'Authorization': 'yes'}
    )
    assert response_post_auth.status_code == 200
    context: Dict[str, Any] = response_post_auth.json()['context']
    assert 'authorizer' in context
    assert context['authorizer']['foo'] == 'bar'


def test_empty_raw_body(smoke_test_app: SmokeTestApplication) -> None:
    response: requests.Response = smoke_test_app.post_response('/repr-raw-body')
    response.raise_for_status()
    assert response.json() == {'repr-raw-body': ''}


def test_websocket_lifecycle(smoke_test_app: SmokeTestApplication) -> None:
    ws: websocket.WebSocket = websocket.create_connection(smoke_test_app.websocket_connect_url)
    ws.send("Hello, World 1")
    ws.recv()
    ws.close()
    ws = websocket.create_connection(smoke_test_app.websocket_connect_url)
    ws.send("Hello, World 2")
    second_response: Any = json.loads(ws.recv())
    ws.close()

    expected_second_response: List[Any] = [
        [mock.ANY, 'Hello, World 1'],
        [mock.ANY, 'Hello, World 2']
    ]
    assert expected_second_response == second_response
    assert second_response[0][0] != second_response[1][0]


@pytest.mark.on_redeploy
def test_redeploy_no_change_view(smoke_test_app: SmokeTestApplication) -> None:
    smoke_test_app.redeploy_once()
    assert smoke_test_app.get_json('/') == {'hello': 'world'}


@pytest.mark.on_redeploy
def test_redeploy_changed_function(smoke_test_app: SmokeTestApplication) -> None:
    smoke_test_app.redeploy_once()
    assert smoke_test_app.get_json('/a/b/c/d/e/f/g') == {
        'redeployed': True
    }


@pytest.mark.on_redeploy
def test_redeploy_new_function(smoke_test_app: SmokeTestApplication) -> None:
    smoke_test_app.redeploy_once()
    assert smoke_test_app.get_json('/redeploy') == {'success': True}


@pytest.mark.on_redeploy
def test_redeploy_change_route_info(smoke_test_app: SmokeTestApplication) -> None:
    smoke_test_app.redeploy_once()
    # POST is no longer allowed:
    response_post: requests.Response = smoke_test_app.post_response('/multimethod')
    assert response_post.status_code == 403
    # But PUT is now allowed in the redeployed app.py
    response_put: requests.Response = smoke_test_app.put_response('/multimethod')
    assert response_put.status_code == 200


@pytest.mark.on_redeploy
def test_redeploy_view_deleted(smoke_test_app: SmokeTestApplication) -> None:
    smoke_test_app.redeploy_once()
    response: requests.Response = smoke_test_app.get_response('/path/foo')
    # Request should fail because it's not in the redeployed
    # app.py
    assert response.status_code == 403
