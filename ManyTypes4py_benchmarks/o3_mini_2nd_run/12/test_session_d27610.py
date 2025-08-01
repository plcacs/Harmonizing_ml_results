#!/usr/bin/env python
import logging
import re
import subprocess
import sys
import textwrap
from collections.abc import Mapping, Iterator
from pathlib import Path
from typing import Any, Optional, Dict, Union
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

_FAKE_PROJECT_NAME: str = 'fake_project'
_FAKE_PIPELINE_NAME: str = 'fake_pipeline'

class BadStore:
    """
    Store class that doesn't subclass `BaseSessionStore`, for testing only.
    """
    pass

class BadConfigLoader:
    """
    ConfigLoader class that doesn't subclass `AbstractConfigLoader`, for testing only.
    """
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
    mock_runner_obj = mocker.patch('kedro.runner.sequential_runner.SequentialRunner', autospec=True)
    mock_runner_obj.__name__ = 'MockRunner'
    return mock_runner_obj

@pytest.fixture
def mock_thread_runner(mocker: Any) -> Any:
    mock_runner_obj = mocker.patch('kedro.runner.thread_runner.ThreadRunner', autospec=True)
    mock_runner_obj.__name__ = 'MockThreadRunner`'
    return mock_runner_obj

@pytest.fixture
def mock_context_class(mocker: Any) -> Any:
    mock_cls = create_attrs_autospec(KedroContext)
    return mocker.patch('kedro.framework.context.KedroContext', autospec=True, return_value=mock_cls)

def _mock_imported_settings_paths(mocker: Any, mock_settings: _ProjectSettings) -> _ProjectSettings:
    for path in ['kedro.framework.project.settings', 'kedro.framework.session.session.settings']:
        mocker.patch(path, mock_settings)
    return mock_settings

@pytest.fixture
def mock_settings(mocker: Any) -> _ProjectSettings:
    return _mock_imported_settings_paths(mocker, _ProjectSettings())

@pytest.fixture
def mock_settings_context_class(mocker: Any, mock_context_class: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _CONTEXT_CLASS = Validator('CONTEXT_CLASS', default=lambda *args, **kwargs: mock_context_class)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_custom_context_class(mocker: Any) -> _ProjectSettings:
    class MyContext(KedroContext):
        pass

    class MockSettings(_ProjectSettings):
        _CONTEXT_CLASS = Validator('CONTEXT_CLASS', default=lambda *args, **kwargs: MyContext)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_custom_config_loader_class(mocker: Any) -> _ProjectSettings:
    class MyConfigLoader(AbstractConfigLoader):
        pass

    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_CLASS = _HasSharedParentClassValidator('CONFIG_LOADER_CLASS', default=lambda *args, **kwargs: MyConfigLoader)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_omega_config_loader_class(mocker: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_CLASS = _HasSharedParentClassValidator('CONFIG_LOADER_CLASS', default=lambda *args, **kwargs: OmegaConfigLoader)
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_config_loader_args(mocker: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_ARGS = Validator('CONFIG_LOADER_ARGS', default={'config_patterns': {'spark': ['spark/*']}})
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_config_loader_args_env(mocker: Any) -> _ProjectSettings:
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
def mock_settings_bad_session_store_args(mocker: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _SESSION_STORE_ARGS = Validator('SESSION_STORE_ARGS', default={'wrong_arg': 'O_o'})
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_settings_uncaught_session_store_exception(mocker: Any) -> Any:
    class MockSettings(_ProjectSettings):
        _SESSION_STORE_ARGS = Validator('SESSION_STORE_ARGS', default={'path': 'path'})
    _mock_imported_settings_paths(mocker, MockSettings())
    return mocker.patch.object(BaseSessionStore, '__init__', side_effect=Exception('Fake'))

@pytest.fixture
def fake_session_id(mocker: Any) -> str:
    session_id: str = 'fake_session_id'
    mocker.patch('kedro.framework.session.session.generate_timestamp', return_value=session_id)
    return session_id

@pytest.fixture
def fake_project(tmp_path: Path, mock_package_name: str) -> Path:
    fake_project_dir: Path = Path(tmp_path) / 'fake_project'
    (fake_project_dir / 'src').mkdir(parents=True)
    pyproject_toml_path: Path = fake_project_dir / 'pyproject.toml'
    payload: Dict[str, Any] = {
        'tool': {
            'kedro': {
                'kedro_init_version': kedro_version,
                'project_name': _FAKE_PROJECT_NAME,
                'package_name': mock_package_name
            }
        }
    }
    toml_str: str = toml.dumps(payload)
    pyproject_toml_path.write_text(toml_str, encoding='utf-8')
    (fake_project_dir / 'conf' / 'base').mkdir(parents=True)
    (fake_project_dir / 'conf' / 'local').mkdir()
    return fake_project_dir

@pytest.fixture
def fake_username(mocker: Any) -> str:
    username: str = 'user1'
    mocker.patch('kedro.framework.session.session.getpass.getuser', return_value=username)
    return username

class FakeException(Exception):
    """Fake exception class for testing purposes"""
    pass

SESSION_LOGGER_NAME: str = 'kedro.framework.session.session'
STORE_LOGGER_NAME: str = 'kedro.framework.session.store'

class TestKedroSession:
    @pytest.mark.usefixtures('mock_settings_context_class')
    @pytest.mark.parametrize('env', [None, 'env1'])
    @pytest.mark.parametrize('extra_params', [None, {'key': 'val'}])
    def test_create(self, fake_project: Path, mock_context_class: Any, fake_session_id: str, 
                    mocker: Any, env: Optional[str], extra_params: Optional[Dict[str, Any]], 
                    fake_username: str) -> None:
        mock_click_ctx: Any = mocker.patch('click.get_current_context').return_value
        mocker.patch('sys.argv', ['kedro', 'run', '--params=x'])
        session: KedroSession = KedroSession.create(fake_project, env=env, extra_params=extra_params)
        expected_cli_data: Dict[str, Any] = {
            'args': mock_click_ctx.args,
            'params': mock_click_ctx.params,
            'command_name': mock_click_ctx.command.name,
            'command_path': 'kedro run --params=x'
        }
        expected_store: Dict[str, Any] = {
            'project_path': fake_project,
            'session_id': fake_session_id,
            'cli': expected_cli_data
        }
        if env:
            expected_store['env'] = env
        if extra_params:
            expected_store['extra_params'] = extra_params
        expected_store['username'] = fake_username
        mocker.patch('kedro_telemetry.plugin._check_for_telemetry_consent', return_value=False)
        assert session.store == expected_store
        assert session.load_context() is mock_context_class.return_value
        assert isinstance(session._get_config_loader(), OmegaConfigLoader)

    @pytest.mark.usefixtures('mock_settings')
    def test_create_multiple_sessions(self, fake_project: Path) -> None:
        with KedroSession.create(fake_project):
            with KedroSession.create(fake_project):
                pass

    @pytest.mark.usefixtures('mock_settings_context_class')
    def test_create_no_env_extra_params(self, fake_project: Path, mock_context_class: Any, fake_session_id: str, 
                                        mocker: Any, fake_username: str) -> None:
        mock_click_ctx: Any = mocker.patch('click.get_current_context').return_value
        mocker.patch('sys.argv', ['kedro', 'run', '--params=x'])
        session: KedroSession = KedroSession.create(fake_project)
        expected_cli_data: Dict[str, Any] = {
            'args': mock_click_ctx.args,
            'params': mock_click_ctx.params,
            'command_name': mock_click_ctx.command.name,
            'command_path': 'kedro run --params=x'
        }
        expected_store: Dict[str, Any] = {
            'project_path': fake_project,
            'session_id': fake_session_id,
            'cli': expected_cli_data,
            'username': fake_username
        }
        mocker.patch('kedro_telemetry.plugin._check_for_telemetry_consent', return_value=False)
        assert session.store == expected_store
        assert session.load_context() is mock_context_class.return_value
        assert isinstance(session._get_config_loader(), OmegaConfigLoader)

    @pytest.mark.usefixtures('mock_settings')
    def test_load_context_with_envvar(self, fake_project: Path, monkeypatch: Any, mocker: Any) -> None:
        monkeypatch.setenv('KEDRO_ENV', 'my_fake_env')
        session: KedroSession = KedroSession.create(fake_project)
        result: KedroContext = session.load_context()
        assert isinstance(result, KedroContext)
        assert result.__class__.__name__ == 'KedroContext'
        assert result.env == 'my_fake_env'

    @pytest.mark.usefixtures('mock_settings')
    def test_load_config_loader_with_envvar(self, fake_project: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv('KEDRO_ENV', 'my_fake_env')
        session: KedroSession = KedroSession.create(fake_project)
        result: OmegaConfigLoader = session._get_config_loader()
        assert isinstance(result, OmegaConfigLoader)
        assert result.__class__.__name__ == 'OmegaConfigLoader'
        assert result.env == 'my_fake_env'

    @pytest.mark.usefixtures('mock_settings_custom_context_class')
    def test_load_context_custom_context_class(self, fake_project: Path) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: KedroContext = session.load_context()
        assert isinstance(result, KedroContext)
        assert result.__class__.__name__ == 'MyContext'

    @pytest.mark.usefixtures('mock_settings_custom_config_loader_class')
    def test_load_config_loader_custom_config_loader_class(self, fake_project: Path) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: AbstractConfigLoader = session._get_config_loader()
        assert isinstance(result, AbstractConfigLoader)
        assert result.__class__.__name__ == 'MyConfigLoader'

    @pytest.mark.usefixtures('mock_settings_config_loader_args')
    def test_load_config_loader_args(self, fake_project: Path, mocker: Any) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: OmegaConfigLoader = session._get_config_loader()
        assert isinstance(result, OmegaConfigLoader)
        assert result.config_patterns['catalog'] == ['catalog*', 'catalog*/**', '**/catalog*']
        assert result.config_patterns['spark'] == ['spark/*']
        mocker.patch('kedro.config.OmegaConfigLoader.__getitem__', return_value=['spark/*'])
        assert result['spark'] == ['spark/*']

    @pytest.mark.usefixtures('mock_settings_config_loader_args')
    def test_config_loader_args_no_env_overwrites_env(self, fake_project: Path, mocker: Any) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: OmegaConfigLoader = session._get_config_loader()
        assert isinstance(result, OmegaConfigLoader)
        assert result.base_env == ''
        assert result.default_run_env == ''

    @pytest.mark.usefixtures('mock_settings_config_loader_args_env')
    def test_config_loader_args_overwrite_env(self, fake_project: Path, mocker: Any) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: OmegaConfigLoader = session._get_config_loader()
        assert isinstance(result, OmegaConfigLoader)
        assert result.base_env == 'something_new'
        assert result.default_run_env == ''

    def test_broken_config_loader(self, mock_settings_file_bad_config_loader_class: Any) -> None:
        pattern: str = "Invalid value 'tests.framework.session.test_session.BadConfigLoader' received for setting 'CONFIG_LOADER_CLASS'. It must be a subclass of 'kedro.config.abstract_config.AbstractConfigLoader'."
        mock_settings_obj = _ProjectSettings(settings_file=str(mock_settings_file_bad_config_loader_class))
        with pytest.raises(ValidationError, match=re.escape(pattern)):
            _ = mock_settings_obj.CONFIG_LOADER_CLASS

    def test_logging_is_not_reconfigure(self, fake_project: Path, caplog: Any, mocker: Any) -> None:
        caplog.set_level(logging.DEBUG, logger='kedro')
        mock_logging: Any = mocker.patch.object(LOGGING, 'configure')
        session: KedroSession = KedroSession.create(fake_project)
        session.close()
        mock_logging.assert_not_called()

    @pytest.mark.usefixtures('mock_settings_context_class')
    def test_default_store(self, fake_project: Path, fake_session_id: str, caplog: Any) -> None:
        caplog.set_level(logging.DEBUG, logger='kedro')
        session: KedroSession = KedroSession.create(fake_project)
        assert isinstance(session.store, dict)
        assert session._store.__class__ is BaseSessionStore
        assert session._store._path == (fake_project / 'sessions').as_posix()
        assert session._store._session_id == fake_session_id
        session.close()
        expected_log_messages = ["'read()' not implemented for 'BaseSessionStore'. Assuming empty store.",
                                 "'save()' not implemented for 'BaseSessionStore'. Skipping the step."]
        actual_log_messages = [rec.getMessage() for rec in caplog.records if rec.name == STORE_LOGGER_NAME and rec.levelno == logging.DEBUG]
        assert actual_log_messages == expected_log_messages

    def test_wrong_store_type(self, mock_settings_file_bad_session_store_class: Any) -> None:
        pattern: str = "Invalid value 'tests.framework.session.test_session.BadStore' received for setting 'SESSION_STORE_CLASS'. It must be a subclass of 'kedro.framework.session.store.BaseSessionStore'."
        mock_settings_obj = _ProjectSettings(settings_file=str(mock_settings_file_bad_session_store_class))
        with pytest.raises(ValidationError, match=re.escape(pattern)):
            _ = mock_settings_obj.SESSION_STORE_CLASS

    @pytest.mark.usefixtures('mock_settings_bad_session_store_args')
    def test_wrong_store_args(self, fake_project: Path) -> None:
        classpath: str = f'{BaseSessionStore.__module__}.{BaseSessionStore.__qualname__}'
        pattern: str = f"Store config must only contain arguments valid for the constructor of '{classpath}'."
        with pytest.raises(ValueError, match=re.escape(pattern)):
            KedroSession.create(fake_project)

    def test_store_uncaught_error(self, fake_project: Path, fake_session_id: str, 
                                  mock_settings_uncaught_session_store_exception: Any) -> None:
        classpath: str = f'{BaseSessionStore.__module__}.{BaseSessionStore.__qualname__}'
        pattern: str = f"Failed to instantiate session store of type '{classpath}'."
        with pytest.raises(ValueError, match=re.escape(pattern)):
            KedroSession.create(fake_project)
        mock_settings_uncaught_session_store_exception.assert_called_once_with(path='path', session_id=fake_session_id)

    @pytest.mark.usefixtures('mock_settings')
    @pytest.mark.parametrize('fake_git_status', ['dirty', ''])
    @pytest.mark.parametrize('fake_commit_hash', ['fake_commit_hash'])
    def test_git_describe(self, fake_project: Path, fake_commit_hash: str, fake_git_status: str, mocker: Any) -> None:
        mocker.patch('subprocess.check_output', side_effect=[fake_commit_hash.encode(), fake_git_status.encode()])
        session: KedroSession = KedroSession.create(fake_project)
        expected_git_info: Dict[str, Any] = {'commit_sha': fake_commit_hash, 'dirty': bool(fake_git_status)}
        assert session.store['git'] == expected_git_info

    @pytest.mark.usefixtures('mock_settings')
    @pytest.mark.parametrize('exception', [subprocess.CalledProcessError(1, 'fake command'), FileNotFoundError, NotADirectoryError])
    def test_git_describe_error(self, fake_project: Path, exception: Exception, mocker: Any, caplog: Any) -> None:
        caplog.set_level(logging.DEBUG, logger='kedro')
        mocker.patch('subprocess.check_output', side_effect=exception)
        session: KedroSession = KedroSession.create(fake_project)
        assert 'git' not in session.store
        expected_log_message: str = f'Unable to git describe {fake_project}'
        actual_log_messages = [rec.getMessage() for rec in caplog.records if rec.name == SESSION_LOGGER_NAME and rec.levelno == logging.DEBUG]
        assert expected_log_message in actual_log_messages

    def test_get_username_error(self, fake_project: Path, mocker: Any, caplog: Any) -> None:
        caplog.set_level(logging.DEBUG, logger='kedro')
        mocker.patch('subprocess.check_output')
        mocker.patch('getpass.getuser', side_effect=FakeException('getuser error'))
        session: KedroSession = KedroSession.create(fake_project)
        assert 'username' not in session.store
        expected_log_messages = ['Unable to get username. Full exception: getuser error']
        actual_log_messages = [rec.getMessage() for rec in caplog.records if rec.name == SESSION_LOGGER_NAME and rec.levelno == logging.DEBUG]
        assert actual_log_messages == expected_log_messages

    @pytest.mark.usefixtures('mock_settings')
    def test_log_error(self, fake_project: Path) -> None:
        with pytest.raises(FakeException), KedroSession.create(fake_project) as session:
            raise FakeException
        exception: Dict[str, Any] = session.store['exception']
        assert exception['type'] == 'tests.framework.session.test_session.FakeException'
        assert not exception['value']
        assert any(('raise FakeException' in tb_line for tb_line in exception['traceback']))

    @pytest.mark.usefixtures('mock_settings_context_class')
    @pytest.mark.parametrize('fake_pipeline_name', [None, _FAKE_PIPELINE_NAME])
    def test_run(self, fake_project: Path, fake_session_id: str, fake_pipeline_name: Optional[str], 
                 mock_context_class: Any, mock_runner: Any, mocker: Any) -> None:
        mock_hook: Any = mocker.patch('kedro.framework.session.session._create_hook_manager').return_value.hook
        mock_pipelines: Any = mocker.patch('kedro.framework.session.session.pipelines', return_value={_FAKE_PIPELINE_NAME: mocker.Mock(), '__default__': mocker.Mock()})
        mock_context: Any = mock_context_class.return_value
        mock_catalog: Any = mock_context._get_catalog.return_value
        mock_runner.__name__ = 'SequentialRunner'
        mock_pipeline: Any = mock_pipelines.__getitem__.return_value.filter.return_value
        with KedroSession.create(fake_project) as session:
            session.run(runner=mock_runner, pipeline_name=fake_pipeline_name)
        record_data: Dict[str, Any] = {
            'session_id': fake_session_id,
            'project_path': fake_project.as_posix(),
            'env': mock_context.env,
            'kedro_version': kedro_version,
            'tags': None,
            'from_nodes': None,
            'to_nodes': None,
            'node_names': None,
            'from_inputs': None,
            'to_outputs': None,
            'load_versions': None,
            'extra_params': {},
            'pipeline_name': fake_pipeline_name,
            'namespace': None,
            'runner': mock_runner.__name__
        }
        mock_hook.before_pipeline_run.assert_called_once_with(run_params=record_data, pipeline=mock_pipeline, catalog=mock_catalog)
        mock_runner.run.assert_called_once_with(mock_pipeline, mock_catalog, session._hook_manager, fake_session_id)
        mock_hook.after_pipeline_run.assert_called_once_with(run_params=record_data, run_result=mock_runner.run.return_value, pipeline=mock_pipeline, catalog=mock_catalog)

    @pytest.mark.usefixtures('mock_settings_context_class')
    @pytest.mark.parametrize('fake_pipeline_name', [None, _FAKE_PIPELINE_NAME])
    @pytest.mark.parametrize('match_pattern', [True, False])
    def test_run_thread_runner(self, fake_project: Path, fake_session_id: str, fake_pipeline_name: Optional[str], 
                               mock_context_class: Any, mock_thread_runner: Any, mocker: Any, match_pattern: bool) -> None:
        mock_hook: Any = mocker.patch('kedro.framework.session.session._create_hook_manager').return_value.hook
        ds_mock: Any = mocker.Mock(**{'datasets.return_value': ['ds_1', 'ds_2']})
        filter_mock: Any = mocker.Mock(**{'filter.return_value': ds_mock})
        pipelines_ret: Dict[str, Any] = {_FAKE_PIPELINE_NAME: filter_mock, '__default__': filter_mock}
        mocker.patch('kedro.framework.session.session.pipelines', pipelines_ret)
        mocker.patch('kedro.io.data_catalog.CatalogConfigResolver.match_pattern', return_value=match_pattern)
        with KedroSession.create(fake_project) as session:
            session.run(runner=mock_thread_runner, pipeline_name=fake_pipeline_name)
        mock_context: Any = mock_context_class.return_value
        record_data: Dict[str, Any] = {
            'session_id': fake_session_id,
            'project_path': fake_project.as_posix(),
            'env': mock_context.env,
            'kedro_version': kedro_version,
            'tags': None,
            'from_nodes': None,
            'to_nodes': None,
            'node_names': None,
            'from_inputs': None,
            'to_outputs': None,
            'load_versions': None,
            'extra_params': {},
            'pipeline_name': fake_pipeline_name,
            'namespace': None,
            'runner': mock_thread_runner.__name__
        }
        mock_catalog: Any = mock_context._get_catalog.return_value
        mock_pipeline: Any = filter_mock.filter()
        mock_hook.before_pipeline_run.assert_called_once_with(run_params=record_data, pipeline=mock_pipeline, catalog=mock_catalog)
        mock_thread_runner.run.assert_called_once_with(mock_pipeline, mock_catalog, session._hook_manager, fake_session_id)
        mock_hook.after_pipeline_run.assert_called_once_with(run_params=record_data, run_result=mock_thread_runner.run.return_value, pipeline=mock_pipeline, catalog=mock_catalog)

    @pytest.mark.usefixtures('mock_settings_context_class')
    @pytest.mark.parametrize('fake_pipeline_name', [None, _FAKE_PIPELINE_NAME])
    def test_run_multiple_times(self, fake_project: Path, fake_session_id: str, fake_pipeline_name: Optional[str], 
                                mock_context_class: Any, mock_runner: Any, mocker: Any) -> None:
        mock_hook: Any = mocker.patch('kedro.framework.session.session._create_hook_manager').return_value.hook
        mock_pipelines: Any = mocker.patch('kedro.framework.session.session.pipelines', return_value={_FAKE_PIPELINE_NAME: mocker.Mock(), '__default__': mocker.Mock()})
        mock_context: Any = mock_context_class.return_value
        mock_catalog: Any = mock_context._get_catalog.return_value
        mock_pipeline: Any = mock_pipelines.__getitem__.return_value.filter.return_value
        message: str = 'A run has already been completed as part of the active KedroSession. KedroSession has a 1-1 mapping with runs, and thus only one run should be executed per session.'
        with pytest.raises(Exception, match=message):
            with KedroSession.create(fake_project) as session:
                session.run(runner=mock_runner, pipeline_name=fake_pipeline_name)
                session.run(runner=mock_runner, pipeline_name=fake_pipeline_name)
        record_data: Dict[str, Any] = {
            'session_id': fake_session_id,
            'project_path': fake_project.as_posix(),
            'env': mock_context.env,
            'kedro_version': kedro_version,
            'tags': None,
            'from_nodes': None,
            'to_nodes': None,
            'node_names': None,
            'from_inputs': None,
            'to_outputs': None,
            'load_versions': None,
            'extra_params': {},
            'pipeline_name': fake_pipeline_name,
            'namespace': None,
            'runner': mock_runner.__name__
        }
        mock_hook.before_pipeline_run.assert_called_once_with(run_params=record_data, pipeline=mock_pipeline, catalog=mock_catalog)
        mock_runner.run.assert_called_once_with(mock_pipeline, mock_catalog, session._hook_manager, fake_session_id)
        mock_hook.after_pipeline_run.assert_called_once_with(run_params=record_data, run_result=mock_runner.run.return_value, pipeline=mock_pipeline, catalog=mock_catalog)

    @pytest.mark.usefixtures('mock_settings_context_class')
    def test_run_non_existent_pipeline(self, fake_project: Path, mock_runner: Any, mocker: Any) -> None:
        pattern: str = "Failed to find the pipeline named 'doesnotexist'. It needs to be generated and returned by the 'register_pipelines' function."
        mocker.patch('kedro_telemetry.plugin._check_for_telemetry_consent', return_value=False)
        with pytest.raises(ValueError, match=re.escape(pattern)):
            with KedroSession.create(fake_project) as session:
                session.run(runner=mock_runner, pipeline_name='doesnotexist')

    @pytest.mark.usefixtures('mock_settings_context_class')
    @pytest.mark.parametrize('fake_pipeline_name', [None, _FAKE_PIPELINE_NAME])
    def test_run_exception(self, fake_project: Path, fake_session_id: str, fake_pipeline_name: Optional[str], 
                           mock_context_class: Any, mock_runner: Any, mocker: Any) -> None:
        mock_hook: Any = mocker.patch('kedro.framework.session.session._create_hook_manager').return_value.hook
        mock_pipelines: Any = mocker.patch('kedro.framework.session.session.pipelines', return_value={_FAKE_PIPELINE_NAME: mocker.Mock(), '__default__': mocker.Mock()})
        mock_context: Any = mock_context_class.return_value
        mock_catalog: Any = mock_context._get_catalog.return_value
        error: Exception = FakeException('You shall not pass!')
        mock_runner.run.side_effect = error
        mock_pipeline: Any = mock_pipelines.__getitem__.return_value.filter.return_value
        with pytest.raises(FakeException), KedroSession.create(fake_project) as session:
            session.run(runner=mock_runner, pipeline_name=fake_pipeline_name)
        record_data: Dict[str, Any] = {
            'session_id': fake_session_id,
            'project_path': fake_project.as_posix(),
            'env': mock_context.env,
            'kedro_version': kedro_version,
            'tags': None,
            'from_nodes': None,
            'to_nodes': None,
            'node_names': None,
            'from_inputs': None,
            'to_outputs': None,
            'load_versions': None,
            'extra_params': {},
            'pipeline_name': fake_pipeline_name,
            'namespace': None,
            'runner': mock_runner.__name__
        }
        mock_hook.on_pipeline_error.assert_called_once_with(error=error, run_params=record_data, pipeline=mock_pipeline, catalog=mock_catalog)
        mock_hook.after_pipeline_run.assert_not_called()
        exception: Dict[str, Any] = session.store['exception']
        assert exception['type'] == 'tests.framework.session.test_session.FakeException'
        assert exception['value'] == 'You shall not pass!'
        assert exception['traceback']

    @pytest.mark.usefixtures('mock_settings_context_class')
    @pytest.mark.parametrize('fake_pipeline_name', [None, _FAKE_PIPELINE_NAME])
    def test_run_broken_pipeline_multiple_times(self, fake_project: Path, fake_session_id: str, fake_pipeline_name: Optional[str], 
                                                  mock_context_class: Any, mock_runner: Any, mocker: Any) -> None:
        mock_hook: Any = mocker.patch('kedro.framework.session.session._create_hook_manager').return_value.hook
        mock_pipelines: Any = mocker.patch('kedro.framework.session.session.pipelines', return_value={_FAKE_PIPELINE_NAME: mocker.Mock(), '__default__': mocker.Mock()})
        mock_context: Any = mock_context_class.return_value
        mock_catalog: Any = mock_context._get_catalog.return_value
        session: KedroSession = KedroSession.create(fake_project)
        broken_runner: Any = mocker.patch('kedro.runner.SequentialRunner', autospec=True)
        broken_runner.__name__ = 'BrokenRunner'
        error: Exception = FakeException('You shall not pass!')
        broken_runner.run.side_effect = error
        mock_pipeline: Any = mock_pipelines.__getitem__.return_value.filter.return_value
        with pytest.raises(FakeException):
            session.run(runner=broken_runner, pipeline_name=fake_pipeline_name)
        record_data: Dict[str, Any] = {
            'session_id': fake_session_id,
            'project_path': fake_project.as_posix(),
            'env': mock_context.env,
            'kedro_version': kedro_version,
            'tags': None,
            'from_nodes': None,
            'to_nodes': None,
            'node_names': None,
            'from_inputs': None,
            'to_outputs': None,
            'load_versions': None,
            'extra_params': {},
            'pipeline_name': fake_pipeline_name,
            'namespace': None,
            'runner': broken_runner.__name__
        }
        mock_hook.on_pipeline_error.assert_called_once_with(error=error, run_params=record_data, pipeline=mock_pipeline, catalog=mock_catalog)
        mock_hook.after_pipeline_run.assert_not_called()
        fixed_runner: Any = mock_runner
        session.run(runner=fixed_runner, pipeline_name=fake_pipeline_name)
        fixed_runner.run.assert_called_once_with(mock_pipeline, mock_catalog, session._hook_manager, fake_session_id)
        record_data['runner'] = 'MockRunner'
        mock_hook.after_pipeline_run.assert_called_once_with(run_params=record_data, run_result=fixed_runner.run.return_value, pipeline=mock_pipeline, catalog=mock_catalog)

    @pytest.mark.usefixtures('mock_settings_context_class')
    def test_session_raise_error_with_invalid_runner_instance(self, fake_project: Path, mocker: Any) -> None:
        mocker.patch('kedro.framework.session.session.pipelines', return_value={'__default__': mocker.Mock()})
        mock_runner_class: Any = mocker.patch('kedro.runner.SequentialRunner')
        mocker.patch('kedro_telemetry.plugin._check_for_telemetry_consent', return_value=False)
        session: KedroSession = KedroSession.create(fake_project)
        with pytest.raises(KedroSessionError, match='KedroSession expect an instance of Runner instead of a class.'):
            session.run(runner=mock_runner_class)

    def test_logging_rich_markup(self, fake_project: Path) -> None:
        KedroSession.create(fake_project)
        assert _has_rich_handler()

@pytest.fixture
def fake_project_with_logging_file_handler(fake_project: Path) -> Path:
    logging_config: Dict[str, Any] = {'version': 1, 'handlers': {'info_file_handler': {'filename': 'logs/info.log'}}}
    logging_yml: Path = fake_project / 'conf' / 'logging.yml'
    logging_yml.write_text(yaml.dump(logging_config))
    return fake_project

def get_all_values(mapping: Mapping[Any, Any]) -> Iterator[Any]:
    for value in mapping.values():
        yield value
        if isinstance(value, Mapping):
            yield from get_all_values(value)

@pytest.mark.parametrize('params', ['a=1,b.c=2', 'a=1,b=2,c=3', ''])
def test_no_DictConfig_in_store(params: str, fake_project: Path) -> None:
    extra_params: Dict[str, Any] = _split_params(None, None, params)
    session: KedroSession = KedroSession.create(fake_project, extra_params=extra_params)
    assert not any((OmegaConf.is_config(value) for value in get_all_values(session._store)))
