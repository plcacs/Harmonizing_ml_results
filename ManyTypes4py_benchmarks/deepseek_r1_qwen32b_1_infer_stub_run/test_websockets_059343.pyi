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
from chalice.deploy.deployer import ChaliceDeploymentError, DeployedResources
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

CURRENT_DIR: str = ...
PROJECT_DIR: str = ...
APP_FILE: str = ...
RANDOM_APP_NAME: str = ...

def retry(max_attempts: int, delay: int) -> Callable:
    ...

def _create_ws_connection(url: str, attempts: int = ..., delay: int = ...) -> websocket.WebSocket:
    ...

def _inject_app_name(dirname: str) -> None:
    ...

def _deploy_app(temp_dirname: str) -> 'SmokeTestApplication':
    ...

@retry(max_attempts=10, delay=20)
def _deploy_with_retries(deployer: Any, config: Any) -> DeployedResources:
    ...

def _get_error_code_from_exception(exception: Exception) -> Optional[str]:
    ...

def _delete_app(application: 'SmokeTestApplication', temp_dirname: str) -> None:
    ...

class SmokeTestApplication:
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

class Task(threading.Thread):
    def __init__(self, action: Callable, delay: float = ...) -> None:
        ...

    def run(self) -> None:
        ...

    def stop(self) -> None:
        ...

class CountingMessageSender:
    def __init__(self, ws: websocket.WebSocket, counter: Any) -> None:
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

def find_skips_in_seq(numbers: List[int]) -> List[Tuple[int, int]]:
    ...

def test_websocket_redployment_does_not_lose_messages(smoke_test_app_ws: 'SmokeTestApplication') -> None:
    ...