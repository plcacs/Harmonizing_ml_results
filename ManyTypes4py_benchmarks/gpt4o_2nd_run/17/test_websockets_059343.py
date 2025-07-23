import os
import json
import uuid
import threading
import shutil
import time
import pytest
import websocket
from typing import Callable, Any, Optional, List, Tuple, Generator
from chalice.cli.factory import CLIFactory
from chalice.utils import OSUtils, UI
from chalice.deploy.deployer import ChaliceDeploymentError
from chalice.config import DeployedResources

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(CURRENT_DIR, 'testwebsocketapp')
APP_FILE = os.path.join(PROJECT_DIR, 'app.py')
RANDOM_APP_NAME = 'smoketest-%s' % str(uuid.uuid4())[:13]

def retry(max_attempts: int, delay: int) -> Callable:
    def _create_wrapped_retry_function(function: Callable) -> Callable:
        def _wrapped_with_retry(*args: Any, **kwargs: Any) -> Any:
            for _ in range(max_attempts):
                result = function(*args, **kwargs)
                if result is not None:
                    return result
                time.sleep(delay)
            raise RuntimeError('Exhausted max retries of %s for function: %s' % (max_attempts, function))
        return _wrapped_with_retry
    return _create_wrapped_retry_function

def _create_ws_connection(url: str, attempts: int = 5, delay: int = 5) -> Optional[websocket.WebSocket]:
    for _ in range(attempts):
        try:
            ws = websocket.create_connection(url)
            return ws
        except websocket.WebSocketBadStatusException:
            time.sleep(delay)
    return None

def _inject_app_name(dirname: str) -> None:
    config_filename = os.path.join(dirname, '.chalice', 'config.json')
    with open(config_filename) as f:
        data = json.load(f)
    data['app_name'] = RANDOM_APP_NAME
    data['stages']['dev']['environment_variables']['APP_NAME'] = RANDOM_APP_NAME
    with open(config_filename, 'w') as f:
        f.write(json.dumps(data, indent=2))

def _deploy_app(temp_dirname: str) -> 'SmokeTestApplication':
    factory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(chalice_stage_name='dev', autogen_policy=True)
    session = factory.create_botocore_session()
    d = factory.create_default_deployer(session, config, UI())
    region = session.get_config_variable('region')
    deployed = _deploy_with_retries(d, config)
    application = SmokeTestApplication(region=region, deployed_values=deployed, stage_name='dev', app_name=RANDOM_APP_NAME, app_dir=temp_dirname)
    return application

@retry(max_attempts=10, delay=20)
def _deploy_with_retries(deployer: Any, config: Any) -> Any:
    try:
        deployed_stages = deployer.deploy(config, 'dev')
        return deployed_stages
    except ChaliceDeploymentError as e:
        error_code = _get_error_code_from_exception(e)
        if error_code != 'TooManyRequestsException':
            raise

def _get_error_code_from_exception(exception: ChaliceDeploymentError) -> Optional[str]:
    error_response = getattr(exception.original_error, 'response', None)
    if error_response is None:
        return None
    return error_response.get('Error', {}).get('Code')

def _delete_app(application: 'SmokeTestApplication', temp_dirname: str) -> None:
    factory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(chalice_stage_name='dev', autogen_policy=True)
    session = factory.create_botocore_session()
    d = factory.create_deletion_deployer(session, UI())
    _deploy_with_retries(d, config)

class SmokeTestApplication(object):
    _REDEPLOY_SLEEP = 20
    _POLLING_DELAY = 5

    def __init__(self, deployed_values: Any, stage_name: str, app_name: str, app_dir: str, region: str) -> None:
        self._deployed_resources = DeployedResources(deployed_values)
        self.stage_name = stage_name
        self.app_name = app_name
        self.app_dir = app_dir
        self._has_redeployed = False
        self._region = region

    @property
    def websocket_api_id(self) -> str:
        return self._deployed_resources.resource_values('websocket_api')['websocket_api_id']

    @property
    def websocket_connect_url(self) -> str:
        return 'wss://{websocket_api_id}.execute-api.{region}.amazonaws.com/{api_gateway_stage}'.format(websocket_api_id=self.websocket_api_id, region=self._region, api_gateway_stage='api')

    @property
    def websocket_message_handler_arn(self) -> str:
        return self._deployed_resources.resource_values('websocket_message')['lambda_arn']

    @property
    def region(self) -> str:
        return self._region

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

@pytest.fixture
def smoke_test_app_ws(tmpdir_factory: Any) -> Generator['SmokeTestApplication', None, None]:
    os.environ['APP_NAME'] = RANDOM_APP_NAME
    tmpdir = str(tmpdir_factory.mktemp(RANDOM_APP_NAME))
    _create_dynamodb_table(RANDOM_APP_NAME, tmpdir)
    OSUtils().copytree(PROJECT_DIR, tmpdir)
    _inject_app_name(tmpdir)
    application = _deploy_app(tmpdir)
    yield application
    _delete_app(application, tmpdir)
    _delete_dynamodb_table(RANDOM_APP_NAME, tmpdir)
    os.environ.pop('APP_NAME')

def _create_dynamodb_table(table_name: str, temp_dirname: str) -> None:
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    ddb.create_table(TableName=table_name, AttributeDefinitions=[{'AttributeName': 'entry', 'AttributeType': 'N'}], KeySchema=[{'AttributeName': 'entry', 'KeyType': 'HASH'}], ProvisionedThroughput={'ReadCapacityUnits': 50, 'WriteCapacityUnits': 50})

def _delete_dynamodb_table(table_name: str, temp_dirname: str) -> None:
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    ddb.delete_table(TableName=table_name)

class Task(threading.Thread):

    def __init__(self, action: Callable, delay: float = 0.05) -> None:
        threading.Thread.__init__(self)
        self._action = action
        self._done = threading.Event()
        self._delay = delay

    def run(self) -> None:
        while not self._done.is_set():
            self._action()
            time.sleep(self._delay)

    def stop(self) -> None:
        self._done.set()

def counter() -> Generator[int, None, None]:
    """Generator of sequential increasing numbers"""
    count = 1
    while True:
        yield count
        count += 1

class CountingMessageSender(object):
    """Class to send values from a counter over a websocket."""

    def __init__(self, ws: websocket.WebSocket, counter: Generator[int, None, None]) -> None:
        self._ws = ws
        self._counter = counter
        self._last_sent = None

    def send(self) -> None:
        value = next(self._counter)
        self._ws.send('%s' % value)
        self._last_sent = value

    @property
    def last_sent(self) -> Optional[int]:
        return self._last_sent

def get_numbers_from_dynamodb(temp_dirname: str) -> List[int]:
    """Get numbers from DynamoDB in the format written by testwebsocketapp."""
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    paginator = ddb.get_paginator('scan')
    numbers = sorted([int(item['entry']['N']) for page in paginator.paginate(TableName=RANDOM_APP_NAME, ConsistentRead=True) for item in page['Items']])
    return numbers

def get_errors_from_dynamodb(temp_dirname: str) -> Optional[str]:
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    item = ddb.get_item(TableName=RANDOM_APP_NAME, Key={'entry': {'N': '-9999'}})
    if 'Item' not in item:
        return None
    return item['Item']['errormsg']['S']

def find_skips_in_seq(numbers: List[int]) -> List[Tuple[int, int]]:
    """Find non-sequential gaps in a sequence of numbers

    :type numbers: Iterable of ints
    :param numbers: Iterable to check for gaps

    :returns: List of tuples with the gaps in the format
        [(start_of_gap, end_of_gap, ...)]. If the list is empty then there
        are no gaps.
    """
    last = numbers[0] - 1
    skips = []
    for elem in numbers:
        if elem != last + 1:
            skips.append((last, elem))
        last = elem
    return skips

def test_websocket_redployment_does_not_lose_messages(smoke_test_app_ws: SmokeTestApplication) -> None:
    ws = _create_ws_connection(smoke_test_app_ws.websocket_connect_url)
    counter_generator = counter()
    sender = CountingMessageSender(ws, counter_generator)
    ping_endpoint = Task(sender.send)
    ping_endpoint.start()
    smoke_test_app_ws.redeploy_once()
    time.sleep(1)
    ping_endpoint.stop()
    errors = get_errors_from_dynamodb(smoke_test_app_ws.app_dir)
    assert errors is None
    numbers = get_numbers_from_dynamodb(smoke_test_app_ws.app_dir)
    assert 1 in numbers
    assert sender.last_sent in numbers
    skips = find_skips_in_seq(numbers)
    assert skips == []
