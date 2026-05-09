import logging
import re
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec
import pytest
import toml
import yaml
from omegaconf import OmegaConf
from kedro import __version__ as kedro_version
from kedro.config import AbstractConfigLoader, OmegaConfigLoader
from kedro.framework.cli.utils import _split_params
from kedro.framework.context import KedroContext
from kedro.framework.project import LOGGING, ValidationError, Validator, _HasSharedParentClassValidator, _ProjectSettings
from kedro.framework.session import KedroSession
from kedro.framework.session.session import KedroSessionError
from kedro.framework.session.store import BaseSessionStore
from kedro.utils import _has_rich_handler
from kedro.runner import SequentialRunner, ThreadRunner

_FAKE_PROJECT_NAME: str = 'fake_project'
_FAKE_PIPELINE_NAME: str = 'fake_pipeline'

class BadStore:
    """Store class that doesn't subclass `BaseSessionStore`, for testing only."""
    pass

class BadConfigLoader:
    """ConfigLoader class that doesn't subclass `AbstractConfigLoader`, for testing only."""
    pass

ATTRS_ATTRIBUTE: str = '__attrs_attrs__'
NEW_TYPING: bool = sys.version_info[:3] >= (3, 7, 0)

def create_attrs_autospec(spec: Any, spec_set: bool = True) -> Any:
    """Creates a mock of an attr class (creates mocks recursively on all attributes).
    https://github.com/python-attrs/attrs/issues/462#issuecomment-1134656377

    :param spec: the spec to mock
    :param spec_set: if True, AttributeError will be raised if an attribute that is not in the spec is set.
    """
    if not hasattr(spec, ATTRS_ATTRIBUTE):
        raise TypeError(f'{spec!r} is not an attrs class')
    mock = create_autospec(spec, spec_set=spec_set)
    for attribute in getattr(spec, ATTRS_ATTRIBUTE):
        attribute_type = attribute.type
        if NEW_TYPING:
            while hasattr(attribute_type, '__origin__'):
                attribute_type = attribute_type.__origin__
        if hasattr(attribute_type, ATTRS_ATTRIBUTE):
            mock_attribute = create_attrs_autospec(attribute_type, spec_set)
        else:
            mock_attribute = create_autospec(attribute_type, spec_set=spec_set)
        object.__setattr__(mock, attribute.name, mock_attribute)
    return mock

@pytest.fixture
def mock_runner(mocker: Any) -> Any:
    mock_runner = mocker.patch('kedro.runner.sequential_runner.SequentialRunner', autospec=True)
    mock_runner.__name__ = 'MockRunner'
    return mock_runner

@pytest.fixture
def mock_thread_runner(mocker: Any) -> Any:
    mock_runner = mocker.patch('kedro.runner.thread_runner.ThreadRunner', autospec=True)
    mock_runner.__name__ = 'MockThreadRunner'
    return mock_runner

@pytest.fixture
def mock_context_class(mocker: Any) -> Any:
    mock_cls = create_attrs_autospec(KedroContext)
    return mocker.patch('kedro.framework.context.KedroContext', autospec=True, return_value=mock_cls)

def _mock_imported_settings_paths(mocker: Any, mock_settings: Any) -> Any:
    for path in ['kedro.framework.project.settings', 'kedro.framework.session.session.settings']:
        mocker.patch(path, mock_settings)
    return mock_settings

@pytest.fixture
def mock_settings(mocker: Any) -> Any:
    return _mock_imported_settings_paths(mocker, _ProjectSettings())

@pytest.fixture
def mock_settings_context_class(mocker: Any, mock_context_class: Any) -> Any:
    class MockSettings(_ProjectSettings):
        _CONTEXT_CLASS = Validator('CONTEXT_CLASS', default=lambda *_: mock_context_class)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_custom_context_class(mocker: Any) -> Any:
    class MyContext(KedroContext):
        pass

    class MockSettings(_ProjectSettings):
        _CONTEXT_CLASS = Validator('CONTEXT_CLASS', default=lambda *_: MyContext)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_custom_config_loader_class(mocker: Any) -> Any:
    class MyConfigLoader(AbstractConfigLoader):
        pass

    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_CLASS = _HasSharedParentClassValidator('CONFIG_LOADER_CLASS', default=lambda *_: MyConfigLoader)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_omega_config_loader_class(mocker: Any) -> Any:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_CLASS = _HasSharedParentClassValidator('CONFIG_LOADER_CLASS', default=lambda *_: OmegaConfigLoader)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_config_loader_args(mocker: Any) -> Any:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_ARGS = Validator('CONFIG_LOADER_ARGS', default={'config_patterns': {'spark': ['spark/*']}})
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_config_loader_args_env(mocker: Any) -> Any:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_ARGS = Validator('CONFIG_LOADER_ARGS', default={'base_env': 'something_new'})
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_file_bad_config_loader_class(tmpdir: Any) -> Any:
    mock_settings_file = tmpdir.join('mock_settings_file.py')
    mock_settings_file.write(textwrap.dedent(f'\n            from {__name__} import BadConfigLoader\n            CONFIG_LOADER_CLASS = BadConfigLoader\n            '))
    return mock_settings_file

@pytest.fixture
def mock_settings_file_bad_session_store_class(tmpdir: Any) -> Any:
    mock_settings_file = tmpdir.join('mock_settings_file.py')
    mock_settings_file.write(textwrap.dedent(f'\n            from {__name__} import BadStore\n            SESSION_STORE_CLASS = BadStore\n            '))
    return mock_settings_file

@pytest.fixture
def mock_settings_bad_session_store_args(mocker: Any) -> Any:
    class MockSettings(_ProjectSettings):
        _SESSION_STORE_ARGS = Validator('SESSION_STORE_ARGS', default={'wrong_arg': 'O_o'})
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def fake_session_id(mocker: Any) -> str:
    session_id = 'fake_session_id'
    mocker.patch('