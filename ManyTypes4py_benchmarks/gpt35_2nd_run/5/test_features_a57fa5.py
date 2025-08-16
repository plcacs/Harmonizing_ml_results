from typing import Dict, Any, Union, List

class SmokeTestApplication:
    _REDEPLOY_SLEEP: int = 30
    _POLLING_DELAY: int = 5
    _NUM_SUCCESS: int = 3

    def __init__(self, deployed_values: Dict[str, Any], stage_name: str, app_name: str, app_dir: str, region: str) -> None:
        self._deployed_resources = DeployedResources(deployed_values)
        self.stage_name: str = stage_name
        self.app_name: str = app_name
        self.app_dir: str = app_dir
        self._has_redeployed: bool = False
        self._region: str = region

    @property
    def url(self) -> str:
        return 'https://{rest_api_id}.execute-api.{region}.amazonaws.com/{api_gateway_stage}'.format(rest_api_id=self.rest_api_id, region=self._region, api_gateway_stage='api')

    @property
    def rest_api_id(self) -> str:
        return self._deployed_resources.resource_values('rest_api')['rest_api_id']

    @property
    def websocket_api_id(self) -> str:
        return self._deployed_resources.resource_values('websocket_api')['websocket_api_id']

    @property
    def websocket_connect_url(self) -> str:
        return 'wss://{websocket_api_id}.execute-api.{region}.amazonaws.com/{api_gateway_stage}'.format(websocket_api_id=self.websocket_api_id, region=self._region, api_gateway_stage='api')

    @retry(max_attempts=10, delay=5)
    def get_json(self, url: str) -> Union[Dict[str, Any], None]:
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
    def get_response(self, url: str, headers: Dict[str, str] = None) -> Union[requests.Response, None]:
        try:
            return self._send_request('GET', url, headers=headers)
        except InternalServerError:
            pass

    def _send_request(self, http_method: str, url: str, headers: Dict[str, str] = None, data: Any = None) -> requests.Response:
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
    def post_response(self, url: str, headers: Dict[str, str] = None, data: Any = None) -> Union[requests.Response, None]:
        try:
            return self._send_request('POST', url, headers=headers, data=data)
        except InternalServerError:
            pass

    @retry(max_attempts=10, delay=5)
    def put_response(self, url: str) -> Union[requests.Response, None]:
        try:
            return self._send_request('PUT', url)
        except InternalServerError:
            pass

    @retry(max_attempts=10, delay=5)
    def options_response(self, url: str) -> Union[requests.Response, None]:
        try:
            return self._send_request('OPTIONS', url)
        except InternalServerError:
            pass

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

    def _wait_for_stablize(self) -> Union[Dict[str, Any], None]:
        return self.get_json('/')

def _inject_app_name(dirname: str) -> None:
    ...

def _deploy_app(temp_dirname: str) -> SmokeTestApplication:
    ...

@retry(max_attempts=10, delay=20)
def _deploy_with_retries(deployer: Any, config: Any) -> Union[Dict[str, Any], None]:
    ...

def _get_error_code_from_exception(exception: Exception) -> Union[str, None]:
    ...

def _delete_app(application: SmokeTestApplication, temp_dirname: str) -> None:
    ...

def _assert_contains_access_control_allow_methods(headers: Dict[str, str], methods: List[str]) -> None:
    ...

def _get_resource_id(apig_client: Any, rest_api_id: str, path: str) -> Union[str, None]:
    ...
