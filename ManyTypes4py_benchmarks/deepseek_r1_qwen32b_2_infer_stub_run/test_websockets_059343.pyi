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
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Generator,
    Union,
    Dict,
    Iterable,
    Any,
    AnyStr,
    cast,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Type,
    Set,
    FrozenSet,
    Text,
    BinaryIO,
    TextIO,
    overload,
    NoReturn,
    NewType,
    NamedTuple,
    Sequence,
    Iterable,
    Iterator,
    Any,
    AnyStr,
    cast,
    TypeVar,
    Type,
    ClassVar,
    overload,
    final,
    Protocol,
    runtime_checkable,
    Literal,
    get_origin,
    get_args,
    get_type_hints,
    ForwardRef,
    ParamSpec,
    Concatenate,
    TypeAlias,
    TypedDict,
    NotRequired,
    Required,
    Final,
    ClassVar,
    Any,
    AnyStr,
    cast,
    TypeVar,
    Type,
    ClassVar,
    overload,
    final,
    Protocol,
    runtime_checkable,
    Literal,
    get_origin,
    get_args,
    get_type_hints,
    ForwardRef,
    ParamSpec,
    Concatenate,
    TypeAlias,
    TypedDict,
    NotRequired,
    Required,
    Final,
    ClassVar,
    Any,
    AnyStr,
    cast,
    TypeVar,
    Type,
    ClassVar,
    overload,
    final,
    Protocol,
    runtime_checkable,
    Literal,
    get_origin,
    get_args,
    get_type_hints,
    ForwardRef,
    ParamSpec,
    Concatenate,
    TypeAlias,
    TypedDict,
    NotRequired,
    Required,
    Final,
    ClassVar,
)

if TYPE_CHECKING:
    from . import *

CURRENT_DIR: str
PROJECT_DIR: str
APP_FILE: str
RANDOM_APP_NAME: str

def retry(max_attempts: int, delay: float) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

def _create_ws_connection(url: str, attempts: int = 5, delay: int = 5) -> websocket.WebSocket:
    ...

def _inject_app_name(dirname: str) -> None:
    ...

def _deploy_app(temp_dirname: str) -> SmokeTestApplication:
    ...

@retry(max_attempts=10, delay=20)
def _deploy_with_retries(deployer: Any, config: Any) -> DeployedResources:
    ...

def _get_error_code_from_exception(exception: ChaliceDeploymentError) -> Optional[str]:
    ...

def _delete_app(application: SmokeTestApplication, temp_dirname: str) -> None:
    ...

class SmokeTestApplication:
    _REDEPLOY_SLEEP: int
    _POLLING_DELAY: int

    def __init__(self, deployed_values: Any, stage_name: str, app_name: str, app_dir: str, region: str) -> None:
        ...

    @property
    def websocket_api_id(self) -> str:
        ...

    @property
    def websocket_connect_url(self) -> str:
        ...

    @property
    def websocket_message_handler_arn(self) -> str:
        ...

    @property
    def region(self) -> str:
        ...

    def redeploy_once(self) -> None:
        ...

@pytest.fixture
def smoke_test_app_ws(tmpdir_factory: Any) -> Generator[SmokeTestApplication, None, None]:
    ...

def _create_dynamodb_table(table_name: str, temp_dirname: str) -> None:
    ...

def _delete_dynamodb_table(table_name: str, temp_dirname: str) -> None:
    ...

class Task(threading.Thread):
    def __init__(self, action: Callable[[], None], delay: float = 0.05) -> None:
        ...

    def run(self) -> None:
        ...

    def stop(self) -> None:
        ...

def counter() -> Generator[int, None, None]:
    ...

class CountingMessageSender:
    def __init__(self, ws: websocket.WebSocket, counter: Generator[int, None, None]) -> None:
        ...

    def send(self) -> None:
        ...

    @property
    def last_sent(self) -> int:
        ...

def get_numbers_from_dynamodb(temp_dirname: str) -> List[int]:
    ...

def get_errors_from_dynamodb(temp_dirname: str) -> Optional[str]:
    ...

def find_skips_in_seq(numbers: Iterable[int]) -> List[Tuple[int, ...]]:
    ...

def test_websocket_redployment_does_not_lose_messages(smoke_test_app_ws: SmokeTestApplication) -> None:
    ...