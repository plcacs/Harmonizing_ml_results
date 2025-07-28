#!/usr/bin/env python3
"""Test Home Assistant package util methods."""
import asyncio
import logging
import os
import sys
from collections.abc import Generator
from importlib.metadata import metadata
from subprocess import PIPE
from typing import Any

from unittest.mock import MagicMock, Mock, call, patch

import pytest
from homeassistant.util import package

RESOURCE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))
TEST_NEW_REQ: str = 'pyhelloworld3==1.0.0'
TEST_ZIP_REQ: str = f'file://{RESOURCE_DIR}/pyhelloworld3.zip#{TEST_NEW_REQ}'


@pytest.fixture
def mock_sys() -> Generator[Any, None, Any]:
    """Mock sys."""
    with patch('homeassistant.util.package.sys', spec=object) as sys_mock:
        sys_mock.executable = 'python3'
        yield sys_mock


@pytest.fixture
def deps_dir() -> str:
    """Return path to deps directory."""
    return os.path.abspath('/deps_dir')


@pytest.fixture
def lib_dir(deps_dir: str) -> str:
    """Return path to lib directory."""
    return os.path.join(deps_dir, 'lib_dir')


@pytest.fixture
def mock_popen(lib_dir: str) -> Generator[Any, None, Any]:
    """Return a Popen mock."""
    with patch('homeassistant.util.package.Popen') as popen_mock:
        popen_mock.return_value.__enter__ = popen_mock
        popen_mock.return_value.communicate.return_value = (bytes(lib_dir, 'utf-8'), b'error')
        popen_mock.return_value.returncode = 0
        yield popen_mock


@pytest.fixture
def mock_env_copy() -> Generator[Any, None, Any]:
    """Mock os.environ.copy."""
    with patch('homeassistant.util.package.os.environ.copy') as env_copy:
        env_copy.return_value = {}
        yield env_copy


@pytest.fixture
def mock_venv() -> Generator[Any, None, Any]:
    """Mock homeassistant.util.package.is_virtual_env."""
    with patch('homeassistant.util.package.is_virtual_env') as mock:
        mock.return_value = True
        yield mock


def mock_async_subprocess() -> MagicMock:
    """Return an async Popen mock."""
    async_popen: MagicMock = MagicMock()

    async def communicate(input: Any = None) -> Any:
        """Communicate mock."""
        stdout: bytes = bytes('/deps_dir/lib_dir', 'utf-8')
        return (stdout, None)
    async_popen.communicate = communicate  # type: ignore
    return async_popen


@pytest.mark.usefixtures('mock_venv')
def test_install(mock_popen: Any, mock_env_copy: Any, mock_sys: Any) -> None:
    """Test an install attempt on a package that doesn't exist."""
    env: dict[str, str] = mock_env_copy()
    assert package.install_package(TEST_NEW_REQ, False)
    assert mock_popen.call_count == 2
    expected_call = call(
        [mock_sys.executable, '-m', 'uv', 'pip', 'install', '--quiet', TEST_NEW_REQ, '--index-strategy', 'unsafe-first-match'],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
        close_fds=False
    )
    assert mock_popen.mock_calls[0] == expected_call
    assert mock_popen.return_value.communicate.call_count == 1


@pytest.mark.usefixtures('mock_venv')
def test_install_with_timeout(mock_popen: Any, mock_env_copy: Any, mock_sys: Any) -> None:
    """Test an install attempt on a package that doesn't exist with a timeout set."""
    env: dict[str, str] = mock_env_copy()
    assert package.install_package(TEST_NEW_REQ, False, timeout=10)
    assert mock_popen.call_count == 2
    env['HTTP_TIMEOUT'] = '10'
    expected_call = call(
        [mock_sys.executable, '-m', 'uv', 'pip', 'install', '--quiet', TEST_NEW_REQ, '--index-strategy', 'unsafe-first-match'],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
        close_fds=False
    )
    assert mock_popen.mock_calls[0] == expected_call
    assert mock_popen.return_value.communicate.call_count == 1


@pytest.mark.usefixtures('mock_venv')
def test_install_upgrade(mock_popen: Any, mock_env_copy: Any, mock_sys: Any) -> None:
    """Test an upgrade attempt on a package."""
    env: dict[str, str] = mock_env_copy()
    assert package.install_package(TEST_NEW_REQ)
    assert mock_popen.call_count == 2
    expected_call = call(
        [mock_sys.executable, '-m', 'uv', 'pip', 'install', '--quiet', TEST_NEW_REQ,
         '--index-strategy', 'unsafe-first-match', '--upgrade'],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
        close_fds=False
    )
    assert mock_popen.mock_calls[0] == expected_call
    assert mock_popen.return_value.communicate.call_count == 1


@pytest.mark.parametrize('is_venv', [True, False])
def test_install_target(mock_sys: Any, mock_popen: Any, mock_env_copy: Any, mock_venv: Any, is_venv: bool) -> None:
    """Test an install with a target."""
    target: str = 'target_folder'
    env: dict[str, str] = mock_env_copy()
    abs_target: str = os.path.abspath(target)
    env['PYTHONUSERBASE'] = abs_target
    mock_venv.return_value = is_venv
    mock_sys.platform = 'linux'
    args: list[str] = [
        mock_sys.executable,
        '-m', 'uv', 'pip', 'install', '--quiet', TEST_NEW_REQ,
        '--index-strategy', 'unsafe-first-match', '--target', abs_target
    ]
    assert package.install_package(TEST_NEW_REQ, False, target=target)
    assert mock_popen.call_count == 2
    expected_call = call(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env, close_fds=False)
    assert mock_popen.mock_calls[0] == expected_call
    assert mock_popen.return_value.communicate.call_count == 1


@pytest.mark.parametrize(
    ('in_venv', 'additional_env_vars'),
    [
        (True, {}),
        (False, {'UV_SYSTEM_PYTHON': 'true'}),
        (False, {'UV_PYTHON': 'python3'}),
        (False, {'UV_SYSTEM_PYTHON': 'true', 'UV_PYTHON': 'python3'})
    ],
    ids=['in_venv', 'UV_SYSTEM_PYTHON', 'UV_PYTHON', 'UV_SYSTEM_PYTHON and UV_PYTHON']
)
def test_install_pip_compatibility_no_workaround(
    mock_sys: Any, mock_popen: Any, mock_env_copy: Any, mock_venv: Any,
    in_venv: bool, additional_env_vars: dict[str, str]
) -> None:
    """Test install will not use pip fallback."""
    env: dict[str, str] = mock_env_copy()
    env.update(additional_env_vars)
    mock_venv.return_value = in_venv
    mock_sys.platform = 'linux'
    args: list[str] = [
        mock_sys.executable,
        '-m', 'uv', 'pip', 'install', '--quiet', TEST_NEW_REQ, '--index-strategy', 'unsafe-first-match'
    ]
    assert package.install_package(TEST_NEW_REQ, False)
    assert mock_popen.call_count == 2
    expected_call = call(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env, close_fds=False)
    assert mock_popen.mock_calls[0] == expected_call
    assert mock_popen.return_value.communicate.call_count == 1


def test_install_pip_compatibility_use_workaround(mock_sys: Any, mock_popen: Any, mock_env_copy: Any, mock_venv: Any) -> None:
    """Test install will use pip compatibility fallback."""
    env: dict[str, str] = mock_env_copy()
    mock_venv.return_value = False
    mock_sys.platform = 'linux'
    python: str = 'python3'
    mock_sys.executable = python
    site_dir: str = '/site_dir'
    args: list[str] = [
        mock_sys.executable,
        '-m', 'uv', 'pip', 'install', '--quiet', TEST_NEW_REQ, '--index-strategy', 'unsafe-first-match',
        '--python', python, '--target', site_dir
    ]
    with patch('homeassistant.util.package.site', autospec=True) as site_mock:
        site_mock.getusersitepackages.return_value = site_dir
        assert package.install_package(TEST_NEW_REQ, False)
    assert mock_popen.call_count == 2
    expected_call = call(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env, close_fds=False)
    assert mock_popen.mock_calls[0] == expected_call
    assert mock_popen.return_value.communicate.call_count == 1


@pytest.mark.usefixtures('mock_sys', 'mock_venv')
def test_install_error(caplog: Any, mock_popen: Any) -> None:
    """Test an install that errors out."""
    caplog.set_level(logging.WARNING)
    mock_popen.return_value.returncode = 1
    assert not package.install_package(TEST_NEW_REQ)
    assert len(caplog.records) == 1
    for record in caplog.records:
        assert record.levelname == 'ERROR'


@pytest.mark.usefixtures('mock_venv')
def test_install_constraint(mock_popen: Any, mock_env_copy: Any, mock_sys: Any) -> None:
    """Test install with constraint file on not installed package."""
    env: dict[str, str] = mock_env_copy()
    constraints: str = 'constraints_file.txt'
    assert package.install_package(TEST_NEW_REQ, False, constraints=constraints)
    assert mock_popen.call_count == 2
    expected_call = call(
        [mock_sys.executable, '-m', 'uv', 'pip', 'install', '--quiet', TEST_NEW_REQ,
         '--index-strategy', 'unsafe-first-match', '--constraint', constraints],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
        close_fds=False
    )
    assert mock_popen.mock_calls[0] == expected_call
    assert mock_popen.return_value.communicate.call_count == 1


async def test_async_get_user_site(mock_env_copy: Any) -> None:
    """Test async get user site directory."""
    deps_dir_str: str = '/deps_dir'
    env: dict[str, str] = mock_env_copy()
    env['PYTHONUSERBASE'] = os.path.abspath(deps_dir_str)
    args: list[str] = [sys.executable, '-m', 'site', '--user-site']
    with patch(
        'homeassistant.util.package.asyncio.create_subprocess_exec',
        return_value=mock_async_subprocess()
    ) as popen_mock:
        ret: str = await package.async_get_user_site(deps_dir_str)
    assert popen_mock.call_count == 1
    expected_call = call(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
        close_fds=False
    )
    assert popen_mock.call_args == expected_call
    assert ret == os.path.join(deps_dir_str, 'lib_dir')


def test_check_package_global(caplog: Any) -> None:
    """Test for an installed package."""
    pkg = metadata('homeassistant')
    installed_package: str = pkg['name']
    installed_version: str = pkg['version']
    assert package.is_installed(installed_package)
    assert package.is_installed(f'{installed_package}=={installed_version}')
    assert package.is_installed(f'{installed_package}>={installed_version}')
    assert package.is_installed(f'{installed_package}<={installed_version}')
    assert not package.is_installed(f'{installed_package}<{installed_version}')
    assert package.is_installed('-1 invalid_package') is False
    assert "Invalid requirement '-1 invalid_package'" in caplog.text


def test_check_package_fragment(caplog: Any) -> None:
    """Test for an installed package with a fragment."""
    assert not package.is_installed(TEST_ZIP_REQ)
    assert package.is_installed('git+https://github.com/pypa/pip#pip>=1')
    assert not package.is_installed('git+https://github.com/pypa/pip#-1 invalid')
    assert "Invalid requirement 'git+https://github.com/pypa/pip#-1 invalid'" in caplog.text


def test_get_is_installed() -> None:
    """Test is_installed can parse complex requirements."""
    pkg = metadata('homeassistant')
    installed_package: str = pkg['name']
    installed_version: str = pkg['version']
    assert package.is_installed(installed_package)
    assert package.is_installed(f'{installed_package}=={installed_version}')
    assert package.is_installed(f'{installed_package}>={installed_version}')
    assert package.is_installed(f'{installed_package}<={installed_version}')
    assert not package.is_installed(f'{installed_package}<{installed_version}')


def test_check_package_previous_failed_install() -> None:
    """Test for when a previously install package failed and left cruft behind."""
    pkg = metadata('homeassistant')
    installed_package: str = pkg['name']
    installed_version: str = pkg['version']
    with patch('homeassistant.util.package.version', return_value=None):
        assert not package.is_installed(installed_package)
        assert not package.is_installed(f'{installed_package}=={installed_version}')


@pytest.mark.parametrize('dockerenv', [True, False], ids=['dockerenv', 'not_dockerenv'])
@pytest.mark.parametrize('containerenv', [True, False], ids=['containerenv', 'not_containerenv'])
@pytest.mark.parametrize('kubernetes_service_host', [True, False], ids=['kubernetes', 'not_kubernetes'])
@pytest.mark.parametrize('is_official_image', [True, False], ids=['official_image', 'not_official_image'])
async def test_is_docker_env(
    dockerenv: bool,
    containerenv: bool,
    kubernetes_service_host: bool,
    is_official_image: bool
) -> None:
    """Test is_docker_env."""
    def new_path_mock(path: str) -> Any:
        mock: Any = Mock()
        if path == '/.dockerenv':
            mock.exists.return_value = dockerenv
        elif path == '/run/.containerenv':
            mock.exists.return_value = containerenv
        return mock

    env: dict[str, str] = {}
    if kubernetes_service_host:
        env['KUBERNETES_SERVICE_HOST'] = 'True'
    package.is_docker_env.cache_clear()
    with patch('homeassistant.util.package.Path', side_effect=new_path_mock), \
         patch('homeassistant.util.package.is_official_image', return_value=is_official_image), \
         patch.dict(os.environ, env):
        expected: bool = any([dockerenv, containerenv, kubernetes_service_host, is_official_image])
        assert package.is_docker_env() is expected
