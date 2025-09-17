#!/usr/bin/env python3
import logging
from collections import namedtuple
from pathlib import Path
from typing import Any, List, Tuple

import pytest
from dynaconf.validator import Validator
from kedro.framework.hooks.manager import _register_hooks
from kedro.framework.project import _ProjectSettings
from kedro.framework.session import KedroSession
from tests.framework.session.conftest import _mock_imported_settings_paths
from pytest import FixtureRequest
from pytest_mock import MockerFixture

MockDistInfo = namedtuple('MockDistInfo', ['project_name', 'version'])

@pytest.fixture
def naughty_plugin() -> MockDistInfo:
    return MockDistInfo('test-project-a', '0.1')

@pytest.fixture
def good_plugin() -> MockDistInfo:
    return MockDistInfo('test-project-b', '0.2')

@pytest.fixture
def mock_settings_with_disabled_hooks(mocker: MockerFixture, project_hooks: Any, naughty_plugin: MockDistInfo) -> None:

    class MockSettings(_ProjectSettings):
        _HOOKS = Validator('HOOKS', default=(project_hooks,))
        _DISABLE_HOOKS_FOR_PLUGINS = Validator(
            'DISABLE_HOOKS_FOR_PLUGINS', default=(naughty_plugin.project_name,)
        )

    _mock_imported_settings_paths(mocker, MockSettings())

class TestSessionHookManager:
    """Test the process of registering hooks with the hook manager in a session."""

    @pytest.mark.nologreset
    def test_assert_register_hooks(self, project_hooks: Any, mock_session: Any) -> None:
        hook_manager: Any = mock_session._hook_manager
        assert hook_manager.is_registered(project_hooks)

    @pytest.mark.usefixtures('mock_session')
    @pytest.mark.nologreset
    def test_calling_register_hooks_twice(self, project_hooks: Any, mock_session: Any) -> None:
        """Calling hook registration multiple times should not raise"""
        hook_manager: Any = mock_session._hook_manager
        assert hook_manager.is_registered(project_hooks)
        _register_hooks(hook_manager, (project_hooks,))
        _register_hooks(hook_manager, (project_hooks,))
        assert hook_manager.is_registered(project_hooks)

    @pytest.mark.parametrize('num_plugins', [0, 1])
    @pytest.mark.nologreset
    def test_hooks_registered_when_session_created(
        self,
        mocker: MockerFixture,
        request: FixtureRequest,
        caplog: Any,
        project_hooks: Any,
        num_plugins: int
    ) -> None:
        caplog.set_level(logging.DEBUG, logger='kedro')
        load_setuptools_entrypoints = mocker.patch(
            'pluggy._manager.PluginManager.load_setuptools_entrypoints', return_value=num_plugins
        )
        distinfo: List[Tuple[Any, MockDistInfo]] = [
            ('plugin_obj_1', MockDistInfo('test-project-a', '0.1'))
        ]
        list_distinfo_mock = mocker.patch(
            'pluggy._manager.PluginManager.list_plugin_distinfo', return_value=distinfo
        )
        session: Any = request.getfixturevalue('mock_session')
        hook_manager: Any = session._hook_manager
        assert hook_manager.is_registered(project_hooks)
        load_setuptools_entrypoints.assert_called_once_with('kedro.hooks')
        list_distinfo_mock.assert_called_once_with()
        if num_plugins:
            log_messages: List[str] = [record.getMessage() for record in caplog.records]
            plugin: str = f'{distinfo[0][1].project_name}-{distinfo[0][1].version}'
            expected_msg: str = f'Registered hooks from {num_plugins} installed plugin(s): {plugin}'
            assert expected_msg in log_messages

    @pytest.mark.usefixtures('mock_settings_with_disabled_hooks')
    @pytest.mark.nologreset
    def test_disabling_auto_discovered_hooks(
        self,
        mocker: MockerFixture,
        caplog: Any,
        tmp_path: Path,
        mock_package_name: Any,
        naughty_plugin: MockDistInfo,
        good_plugin: MockDistInfo
    ) -> None:
        caplog.set_level(logging.DEBUG, logger='kedro')
        distinfo: List[Tuple[Any, MockDistInfo]] = [
            ('plugin_obj_1', naughty_plugin),
            ('plugin_obj_2', good_plugin)
        ]
        mocked_distinfo = mocker.patch(
            'pluggy._manager.PluginManager.list_plugin_distinfo', return_value=distinfo
        )
        mocker.patch(
            'pluggy._manager.PluginManager.load_setuptools_entrypoints', return_value=len(distinfo)
        )
        unregister_mock = mocker.patch('pluggy._manager.PluginManager.unregister')
        KedroSession.create(tmp_path, extra_params={'params:key': 'value'})
        mocked_distinfo.assert_called_once_with()
        unregister_mock.assert_called_once_with(plugin=distinfo[0][0])
        log_messages: List[str] = [record.getMessage() for record in caplog.records]
        expected_msg_registered: str = (
            f'Registered hooks from 1 installed plugin(s): {good_plugin.project_name}-{good_plugin.version}'
        )
        assert expected_msg_registered in log_messages
        expected_msg_disabled: str = (
            f'Hooks are disabled for plugin(s): {naughty_plugin.project_name}-{naughty_plugin.version}'
        )
        assert expected_msg_disabled in log_messages
