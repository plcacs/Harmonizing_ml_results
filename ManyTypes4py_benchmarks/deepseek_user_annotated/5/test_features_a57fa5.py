import json
import os
import time
import shutil
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, cast
from unittest import mock

import botocore.session
import pytest
import requests
import websocket
from botocore.client import BaseClient
from chalice.cli.factory import CLIFactory
from chalice.utils import OSUtils, UI
from chalice.deploy.deployer import ChaliceDeploymentError
from chalice.config import DeployedResources


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(CURRENT_DIR, 'testapp')
APP_FILE = os.path.join(PROJECT_DIR, 'app.py')
RANDOM_APP_NAME = 'smoketest-%s' % str(uuid.uuid4())[:13]


T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def retry(max_attempts: int, delay: int) -> Callable[[F], F]:
    def _create_wrapped_retry_function(function: F) -> F:
        def _wrapped_with_retry(*args: Any, **kwargs: Any) -> Any:
            for _ in range(max_attempts):
                result = function(*args, **kwargs)
                if result is not None:
                    return result
                time.sleep(delay)
            raise RuntimeError("Exhausted max retries of %s for function: %s"
                               % (max_attempts, function))
        return cast(F, _wrapped_with_retry)
    return _create_wrapped_retry_function


class InternalServerError(Exception):
    pass


class SmokeTestApplication(object):
    _REDEPLOY_SLEEP = 30
    _POLLING_DELAY = 5
    _NUM_SUCCESS = 3

    def __init__(
        self,
        deployed_values: Dict[str, Any],
        stage_name: str,
        app_name: str,
        app_dir: str,
        region: str
    ) -> None:
        self._deployed_resources = DeployedResources(deployed_values)
        self.stage_name = stage_name
        self.app_name = app_name
        self.app_dir = app_dir
        self._has_redeployed = False
        self._region = region

    @property
    def url(self) -> str:
        return (
            "https://{rest_api_id}.execute-api.{region}.amazonaws.com/"
            "{api_gateway_stage}".format(rest_api_id=self.rest_api_id,
                                         region=self._region,
                                         api_gateway_stage='api')
        )

    @property
    def rest_api_id(self) -> str:
        return self._deployed_resources.resource_values(
            'rest_api')['rest_api_id']

    @property
    def websocket_api_id(self) -> str:
        return self._deployed_resources.resource_values(
            'websocket_api')['websocket_api_id']

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
            return None

    def _get_json(self, url: str) -> Dict[str, Any]:
        if not url.startswith('/'):
            url = '/' + url
        response = requests.get(self.url + url)
        response.raise_for_status()
        return response.json()

    @retry(max_attempts=10, delay=5)
    def get_response(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
        try:
            return self._send_request('GET', url, headers=headers)
        except InternalServerError:
            return None

    def _send_request(
        self,
        http_method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict[str, Any], str, bytes]] = None
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
        data: Optional[Union[Dict[str, Any], str, bytes]] = None
    ) -> Optional[requests.Response]:
        try:
            return self._send_request('POST', url, headers=headers, data=data)
        except InternalServerError:
            return None

    @retry(max_attempts=10, delay=5)
    def put_response(self, url: str) -> Optional[requests.Response]:
        try:
            return self._send_request('PUT', url)
        except InternalServerError:
            return None

    @retry(max_attempts=10, delay=5)
    def options_response(self, url: str) -> Optional[requests.Response]:
        try:
            return self._send_request('OPTIONS', url)
        except InternalServerError:
            return None

    def redeploy_once(self) -> None:
        if self._has_redeployed:
            return
        new_file = os.path.join(self.app_dir, 'app-redeploy.py')
        original_app_py = os.path.join(self.app_dir, 'app.py')
        shutil.move(original_app_py, original_app_py + '.bak')
        shutil.copy(new_file, original_app_py)
        _deploy_app(self.app_dir)
        self._has_redeployed = True
        time.sleep(self._REDEPLOY_SLEEP)
        for _ in range(self._NUM_SUCCESS):
            self._wait_for_stablize()
            time.sleep(self._POLLING_DELAY)

    def _wait_for_stablize(self) -> Dict[str, Any]:
        return self.get_json('/')


@pytest.fixture
def apig_client() -> BaseClient:
    s = botocore.session.get_session()
    return s.create_client('apigateway')


@pytest.fixture(scope='module')
def smoke_test_app(tmpdir_factory: Any) -> SmokeTestApplication:
    os.environ['APP_NAME'] = RANDOM_APP_NAME
    tmpdir = str(tmpdir_factory.mktemp(RANDOM_APP_NAME))
    OSUtils().copytree(PROJECT_DIR, tmpdir)
    _inject_app_name(tmpdir)
    application = _deploy_app(tmpdir)
    yield application
    _delete_app(application, tmpdir)
    os.environ.pop('APP_NAME')


def _inject_app_name(dirname: str) -> None:
    config_filename = os.path.join(dirname, '.chalice', 'config.json')
    with open(config_filename) as f:
        data = json.load(f)
    data['app_name'] = RANDOM_APP_NAME
    data['stages']['dev']['environment_variables']['APP_NAME'] = \
        RANDOM_APP_NAME
    with open(config_filename, 'w') as f:
        f.write(json.dumps(data, indent=2))


def _deploy_app(temp_dirname: str) -> SmokeTestApplication:
    factory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(
        chalice_stage_name='dev',
        autogen_policy=True
    )
    session = factory.create_botocore_session()
    d = factory.create_default_deployer(session, config, UI())
    region = session.get_config_variable('region')
    deployed = _deploy_with_retries(d, config)
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
        deployed_stages = deployer.deploy(config, 'dev')
        return deployed_stages
    except ChaliceDeploymentError as e:
        error_code = _get_error_code_from_exception(e)
        if error_code != 'TooManyRequestsException':
            raise
        return {}


def _get_error_code_from_exception(exception: Exception) -> Optional[str]:
    error_response = getattr(exception.original_error, 'response', None)
    if error_response is None:
        return None
    return error_response.get('Error', {}).get('Code')


def _delete_app(application: SmokeTestApplication, temp_dirname: str) -> None:
    factory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(
        chalice_stage_name='dev',
        autogen_policy=True
    )
    session = factory.create_botocore_session()
    d = factory.create_deletion_deployer(session, UI())
    _deploy_with_retries(d, config)


def test_returns_simple_response(smoke_test_app: SmokeTestApplication) -> None:
    assert smoke_test_app.get_json('/') == {'hello': 'world'}


def test_can_have_nested_routes(smoke_test_app: SmokeTestApplication) -> None:
    assert smoke_test_app.get_json('/a/b/c/d/e/f/g') == {'nested': True}


def test_supports_path_params(smoke_test_app: SmokeTestApplication) -> None:
    assert smoke_test_app.get_json('/path/foo') == {'path': 'foo'}
    assert smoke_test_app.get_json('/path/bar') == {'path': 'bar'}


def test_path_params_mapped_in_api(
    smoke_test_app: SmokeTestApplication,
    apig_client: BaseClient
) -> None:
    rest_api_id = smoke_test_app.rest_api_id
    response = apig_client.get_export(restApiId=rest_api_id,
                                      stageName='api',
                                      exportType='swagger')
    swagger_doc = json.loads(response['body'].read())
    route_config = swagger_doc['paths']['/path/{name}']['get']
    assert route_config.get('parameters', {}) == [
        {'name': 'name', 'in': 'path', 'required': True, 'type': 'string'},
    ]


def test_single_doc_mapped_in_api(
    smoke_test_app: SmokeTestApplication,
    apig_client: BaseClient
) -> None:
    rest_api_id = smoke_test_app.rest_api_id
    doc_parts = apig_client.get_documentation_parts(
        restApiId=rest_api_id,
        type='METHOD',
        path='/singledoc'
    )
    doc_props = json.loads(doc_parts['items'][0]['properties'])
    assert 'summary' in doc_props
    assert 'description' not in doc_props
    assert doc_props['summary'] == 'Single line docstring.'


def test_multi_doc_mapped_in_api(
    smoke_test_app: SmokeTestApplication,
    apig_client: BaseClient
) -> None:
    rest_api_id = smoke_test_app.rest_api_id
    doc_parts = apig_client.get_documentation_parts(
        restApiId=rest_api_id,
        type='METHOD',
        path='/multidoc'
    )
    doc_props = json.loads(doc_parts['items'][0]['properties'])
    assert 'summary' in doc_props
    assert 'description' in doc_props
    assert doc_props['summary'] == 'Multi-line docstring.'
    assert doc_props['description'] == 'And here is another line.'


@retry(max_attempts=18, delay=10)
def _get_resource_id(
    apig_client: BaseClient,
    rest_api_id: str,
    path: str
) -> Optional[str]:
    matches = [
        resource for resource in
        apig_client.get_resources(restApiId=rest_api_id)['items']
        if resource['path'] == path
    ]
    if matches:
        return matches[0]['id']
    return None


def test_supports_post(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.post_response('/post')
    assert response is not None
    response.raise_for_status()
    assert response.json() == {'success': True}
    with pytest.raises(requests.HTTPError):
        response = smoke_test_app.get_response('/post')
        assert response is not None
        response.raise_for_status()


def test_supports_put(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.put_response('/put')
    assert response is not None
    response.raise_for_status()
    assert response.json() == {'success': True}
    with pytest.raises(requests.HTTPError):
        response = smoke_test_app.get_response('/put')
        assert response is not None
        response.raise_for_status()


def test_supports_shared_routes(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.get_json('/shared')
    assert response == {'method': 'GET'}
    response = smoke_test_app.post_response('/shared')
    assert response is not None
    assert response.json() == {'method': 'POST'}


def test_can_read_json_body_on_post(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.post_response(
        '/jsonpost', data=json.dumps({'hello': 'world'}),
        headers={'Content-Type': 'application/json'})
    assert response is not None
    response.raise_for_status()
    assert response.json() == {'json_body': {'hello': 'world'}}


def test_can_raise_bad_request(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.get_response('/badrequest')
    assert response is not None
    assert response.status_code == 400
    assert response.json()['Code'] == 'BadRequestError'
    assert response.json()['Message'] == 'BadRequestError: Bad request.'


def test_can_raise_not_found(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.get_response('/notfound')
    assert response is not None
    assert response.status_code == 404
    assert response.json()['Code'] == 'NotFoundError'


def test_unexpected_error_raises_500_in_prod_mode(smoke_test_app: SmokeTestApplication) -> None:
    response = requests.get(smoke_test_app.url + '/arbitrary-error')
    assert response.status_code == 500
    assert response.json()['Code'] == 'InternalServerError'
    assert 'internal server error' in response.json()['Message']


def test_can_route_multiple_methods_in_one_view(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.get_response('/multimethod')
    assert response is not None
    response.raise_for_status()
    assert response.json()['method'] == 'GET'

    response = smoke_test_app.post_response('/multimethod')
    assert response is not None
    response.raise_for_status()
    assert response.json()['method'] == 'POST'


def test_form_encoded_content_type(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.post_response('/formencoded',
                                            data={'foo': 'bar'})
    assert response is not None
    response.raise_for_status()
    assert response.json() == {'parsed': {'foo': ['bar']}}


def test_can_round_trip_binary(smoke_test_app: SmokeTestApplication) -> None:
    bin_data = b'\xDE\xAD\xBE\xEF'
    response = smoke_test_app.post_response(
        '/binary',
        headers={'Content-Type': 'application/octet-stream',
                 'Accept': 'application/octet-stream'},
        data=bin_data)
    assert response is not None
    response.raise_for_status()
    assert response.content == bin_data


def test_can_round_trip_binary_custom_content_type(smoke_test_app: SmokeTestApplication) -> None:
    bin_data = b'\xDE\xAD\xBE\xEF'
    response = smoke_test_app.post_response(
        '/custom-binary',
        headers={'Content-Type': 'application/binary',
                 'Accept': 'application/binary'},
        data=bin_data
    )
    assert response is not None
    assert response.content == bin_data


def test_can_return_default_binary_data_to_a_browser(smoke_test_app: SmokeTestApplication) -> None:
    base64encoded_response = b'3q2+7w=='
    accept = 'text/html,application/xhtml+xml;q=0.9,image/webp,*/*;q=0.8'
    response = smoke_test_app.get_response(
        '/get-binary', headers={'Accept': accept})
    assert response is not None
    response.raise_for_status()
    assert response.content == base64encoded_response


def _assert_contains_access_control_allow_methods(
    headers: Dict[str, str],
    methods: List[str]
) -> None:
    actual_methods = headers['Access-Control-Allow-Methods'].split(',')
    assert sorted(methods) == sorted(actual_methods), (
        'The expected allowed methods does not match the actual allowed '
        'methods for CORS.')


def test_can_support_cors(smoke_test_app: SmokeTestApplication) -> None:
    response = smoke_test_app.get_response('/cors')
    assert response is not None
    response.raise_for_status()
    assert response.headers['Access-Control-Allow-Origin'] == '*'

    response = smoke_test_app.options_response('/cors')
    assert response is not None
    response.raise_for_status()
    headers = response.headers
    assert headers['Access-Control-Allow-Origin'] == '*'
    assert headers['Access-Control-Allow-Headers'] == (
        'Authorization,Content-Type,X-Am