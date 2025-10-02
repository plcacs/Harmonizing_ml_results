from typing import Any, List, Tuple
import logging
import pytest
from collections import namedtuple
from dynaconf.validator import Validator
from kedro.framework.hooks.manager import _register_hooks
from kedro.framework.project import _ProjectSettings
from kedro.framework.session import KedroSession

MockDistInfo = namedtuple('MockDistInfo', ['project_name', 'version'])

def test_assert_register_hooks(project_hooks: Any, mock_session: Any) -> None:
    ...

def test_calling_register_hooks_twice(project_hooks: Any, mock_session: Any) -> None:
    ...

def test_hooks_registered_when_session_created(mocker: Any, request: Any, caplog: Any, project_hooks: Any, num_plugins: int) -> None:
    ...

def test_disabling_auto_discovered_hooks(mocker: Any, caplog: Any, tmp_path: Any, mock_package_name: Any, naughty_plugin: Any, good_plugin: Any) -> None:
    ...
