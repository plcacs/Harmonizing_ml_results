import os
import json
import uuid
import threading
import shutil
import time
import pytest
import websocket
from chalice.cli.factory import CLIFactory
from chalice.utils import OSUtils, UI
from chalice.deploy.deployer import ChaliceDeploymentError
from chalice.config import DeployedResources

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(CURRENT_DIR, 'testwebsocketapp')
APP_FILE = os.path.join(PROJECT_DIR, 'app.py')
RANDOM_APP_NAME: str = 'smoketest-%s' % str(uuid.uuid4())[:13]

def retry(max_attempts: int, delay: float) -> callable:
    """Retry a function up to max_attempts times with a delay between each attempt.

    :param max_attempts: The maximum number of times to retry
    :param delay: The delay between each attempt in seconds
    :returns: A wrapped function that retries the original function
    """
    def _create_wrapped_retry_function(function: callable) -> callable:
        def _wrapped_with_retry(*args: any, **kwargs: any) -> any:
            for _ in range(max_attempts):
                result = function(*args, **kwargs)
                if result is not None:
                    return result
                time.sleep(delay)
            raise RuntimeError('Exhausted max retries of %s for function: %s' % (max_attempts, function))
        return _wrapped_with_retry
    return _create_wrapped_retry_function

def _create_ws_connection(url: str, attempts: int = 5, delay: float = 5) -> websocket.WebSocket:
    """Create a WebSocket connection with retries.

    :param url: The URL of the WebSocket connection
    :param attempts: The number of times to retry
    :param delay: The delay between each attempt in seconds
    :returns: A WebSocket object
    """
    for _ in range(attempts):
        try:
            ws = websocket.create_connection(url)
            return ws
        except websocket.WebSocketBadStatusException:
            time.sleep(delay)

def _inject_app_name(dirname: str) -> None:
    """Inject a random app name into the chalice config.

    :param dirname: The directory of the chalice config
    """
    config_filename = os.path.join(dirname, '.chalice', 'config.json')
    with open(config_filename) as f:
        data = json.load(f)
    data['app_name'] = RANDOM_APP_NAME
    data['stages']['dev']['environment_variables']['APP_NAME'] = RANDOM_APP_NAME
    with open(config_filename, 'w') as f:
        f.write(json.dumps(data, indent=2))

def _deploy_app(temp_dirname: str) -> SmokeTestApplication:
    """Deploy the chalice app.

    :param temp_dirname: The temporary directory to deploy the app
    :returns: A SmokeTestApplication object
    """
    factory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(chalice_stage_name='dev', autogen_policy=True)
    session = factory.create_botocore_session()
    d = factory.create_default_deployer(session, config, UI())
    region = session.get_config_variable('region')
    deployed = _deploy_with_retries(d, config)
    application = SmokeTestApplication(region=region, deployed_values=deployed, stage_name='dev', app_name=RANDOM_APP_NAME, app_dir=temp_dirname)
    return application

@retry(max_attempts=10, delay=20)
def _deploy_with_retries(deployer: callable, config: any) -> any:
    """Deploy the chalice app with retries.

    :param deployer: The deployer function
    :param config: The config object
    :returns: The deployed stages
    """
    try:
        deployed_stages = deployer.deploy(config, 'dev')
        return deployed_stages
    except ChaliceDeploymentError as e:
        error_code = _get_error_code_from_exception(e)
        if error_code != 'TooManyRequestsException':
            raise

def _get_error_code_from_exception(exception: ChaliceDeploymentError) -> str:
    """Get the error code from a ChaliceDeploymentError.

    :param exception: The ChaliceDeploymentError
    :returns: The error code
    """
    error_response = getattr(exception.original_error, 'response', None)
    if error_response is None:
        return None
    return error_response.get('Error', {}).get('Code')

def _delete_app(application: SmokeTestApplication, temp_dirname: str) -> None:
    """Delete the chalice app.

    :param application: The SmokeTestApplication object
    :param temp_dirname: The temporary directory to delete the app
    """
    factory = CLIFactory(temp_dirname)
    config = factory.create_config_obj(chalice_stage_name='dev', autogen_policy=True)
    session = factory.create_botocore_session()
    d = factory.create_deletion_deployer(session, UI())
    _deploy_with_retries(d, config)

class SmokeTestApplication(object):
    _REDEPLOY_SLEEP: float = 20
    _POLLING_DELAY: float = 5

    def __init__(self, deployed_values: any, stage_name: str, app_name: str, app_dir: str, region: str) -> None:
        """Initialize the SmokeTestApplication object.

        :param deployed_values: The deployed values
        :param stage_name: The stage name
        :param app_name: The app name
        :param app_dir: The app directory
        :param region: The region
        """
        self._deployed_resources = DeployedResources(deployed_values)
        self.stage_name = stage_name
        self.app_name = app_name
        self.app_dir = app_dir
        self._has_redeployed = False
        self._region = region

    @property
    def websocket_api_id(self) -> str:
        """Get the WebSocket API ID.

        :returns: The WebSocket API ID
        """
        return self._deployed_resources.resource_values('websocket_api')['websocket_api_id']

    @property
    def websocket_connect_url(self) -> str:
        """Get the WebSocket connect URL.

        :returns: The WebSocket connect URL
        """
        return 'wss://{websocket_api_id}.execute-api.{region}.amazonaws.com/{api_gateway_stage}'.format(websocket_api_id=self.websocket_api_id, region=self._region, api_gateway_stage='api')

    @property
    def websocket_message_handler_arn(self) -> str:
        """Get the WebSocket message handler ARN.

        :returns: The WebSocket message handler ARN
        """
        return self._deployed_resources.resource_values('websocket_message')['lambda_arn']

    @property
    def region(self) -> str:
        """Get the region.

        :returns: The region
        """
        return self._region

    def redeploy_once(self) -> None:
        """Redeploy the app once.

        :returns: None
        """
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
def smoke_test_app_ws(tmpdir_factory: callable) -> SmokeTestApplication:
    """Create a SmokeTestApplication object.

    :param tmpdir_factory: The temporary directory factory
    :returns: A SmokeTestApplication object
    """
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
    """Create a DynamoDB table.

    :param table_name: The table name
    :param temp_dirname: The temporary directory
    """
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    ddb.create_table(TableName=table_name, AttributeDefinitions=[{'AttributeName': 'entry', 'AttributeType': 'N'}], KeySchema=[{'AttributeName': 'entry', 'KeyType': 'HASH'}], ProvisionedThroughput={'ReadCapacityUnits': 50, 'WriteCapacityUnits': 50})

def _delete_dynamodb_table(table_name: str, temp_dirname: str) -> None:
    """Delete a DynamoDB table.

    :param table_name: The table name
    :param temp_dirname: The temporary directory
    """
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    ddb.delete_table(TableName=table_name)

class Task(threading.Thread):
    """A task that runs in a separate thread."""

    def __init__(self, action: callable, delay: float = 0.05) -> None:
        """Initialize the task.

        :param action: The action to perform
        :param delay: The delay between each action
        """
        threading.Thread.__init__(self)
        self._action = action
        self._done = threading.Event()
        self._delay = delay

    def run(self) -> None:
        """Run the task.

        :returns: None
        """
        while not self._done.is_set():
            self._action()
            time.sleep(self._delay)

    def stop(self) -> None:
        """Stop the task.

        :returns: None
        """
        self._done.set()

def counter() -> int:
    """A generator of sequential increasing numbers.

    :returns: A generator of sequential increasing numbers
    """
    count = 1
    while True:
        yield count
        count += 1

class CountingMessageSender(object):
    """A class to send values from a counter over a WebSocket."""

    def __init__(self, ws: websocket.WebSocket, counter: callable) -> None:
        """Initialize the sender.

        :param ws: The WebSocket object
        :param counter: The counter generator
        """
        self._ws = ws
        self._counter = counter
        self._last_sent = None

    def send(self) -> None:
        """Send a message over the WebSocket.

        :returns: None
        """
        value = next(self._counter)
        self._ws.send('%s' % value)
        self._last_sent = value

    @property
    def last_sent(self) -> int:
        """Get the last sent value.

        :returns: The last sent value
        """
        return self._last_sent

def get_numbers_from_dynamodb(temp_dirname: str) -> list[int]:
    """Get numbers from DynamoDB.

    :param temp_dirname: The temporary directory
    :returns: A list of numbers
    """
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    paginator = ddb.get_paginator('scan')
    numbers = sorted([int(item['entry']['N']) for page in paginator.paginate(TableName=RANDOM_APP_NAME, ConsistentRead=True) for item in page['Items']])
    return numbers

def get_errors_from_dynamodb(temp_dirname: str) -> str:
    """Get errors from DynamoDB.

    :param temp_dirname: The temporary directory
    :returns: A string representing the error
    """
    factory = CLIFactory(temp_dirname)
    session = factory.create_botocore_session()
    ddb = session.create_client('dynamodb')
    item = ddb.get_item(TableName=RANDOM_APP_NAME, Key={'entry': {'N': '-9999'}})
    if 'Item' not in item:
        return None
    return item['Item']['errormsg']['S']

def find_skips_in_seq(numbers: list[int]) -> list[tuple[int, int]]:
    """Find non-sequential gaps in a sequence of numbers.

    :param numbers: A list of numbers
    :returns: A list of tuples representing the gaps
    """
    last = numbers[0] - 1
    skips = []
    for elem in numbers:
        if elem != last + 1:
            skips.append((last, elem))
        last = elem
    return skips

def test_websocket_redployment_does_not_lose_messages(smoke_test_app_ws: SmokeTestApplication) -> None:
    """Test that redeploying the app does not lose messages.

    :param smoke_test_app_ws: A SmokeTestApplication object
    """
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
