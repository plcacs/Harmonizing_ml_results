from __future__ import annotations
import base64
import boto3
import contextlib
import contextvars
import importlib
import ipaddress
import json
import shlex
import sys
from copy import deepcopy
from functools import partial
from textwrap import dedent
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional
import anyio
import anyio.to_thread

class ElasticContainerServicePushProvisioner:
    """
    An infrastructure provisioner for ECS push work pools.
    """

    def __init__(self) -> None:
        self._console: Console = Console()

    @property
    def console(self) -> Console:
        return self._console

    @console.setter
    def console(self, value: Console) -> None:
        self._console = value

    async def _prompt_boto3_installation(self) -> None:
        global boto3
        await anyio.to_thread.run_process([shlex.quote(sys.executable), '-m', 'pip', 'install', 'boto3'])
        boto3 = importlib.import_module('boto3')

    @staticmethod
    def is_boto3_installed() -> bool:
        """
        Check if boto3 is installed.
        """
        try:
            importlib.import_module('boto3')
            return True
        except ModuleNotFoundError:
            return False

    def _generate_resources(self,