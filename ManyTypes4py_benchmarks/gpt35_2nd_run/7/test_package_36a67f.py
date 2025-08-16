from typing import Generator
import asyncio
import logging
import os
from subprocess import PIPE
import sys
from unittest.mock import MagicMock, Mock, call, patch
import pytest
from homeassistant.util import package

RESOURCE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))
TEST_NEW_REQ: str = 'pyhelloworld3==1.0.0'
TEST_ZIP_REQ: str = f'file://{RESOURCE_DIR}/pyhelloworld3.zip#{TEST_NEW_REQ}'

def mock_async_subprocess() -> MagicMock:
    async_popen: MagicMock = MagicMock()

    async def communicate(input=None) -> tuple:
        stdout: bytes = bytes('/deps_dir/lib_dir', 'utf-8')
        return (stdout, None)
    async_popen.communicate = communicate
    return async_popen

def new_path_mock(path: str) -> Mock:
    mock: Mock = Mock()
    if path == '/.dockerenv':
        mock.exists.return_value = dockerenv
    elif path == '/run/.containerenv':
        mock.exists.return_value = containerenv
    return mock

async def test_async_get_user_site(mock_env_copy: MagicMock) -> None:
    deps_dir: str = '/deps_dir'
    env: dict = mock_env_copy()
    env['PYTHONUSERBASE'] = os.path.abspath(deps_dir)
    args: list = [sys.executable, '-m', 'site', '--user-site']
    with patch('homeassistant.util.package.asyncio.create_subprocess_exec', return_value=mock_async_subprocess()) as popen_mock:
        ret: str = await package.async_get_user_site(deps_dir)
    assert popen_mock.call_count == 1
    assert popen_mock.call_args == call(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=env, close_fds=False)
    assert ret == os.path.join(deps_dir, 'lib_dir')

def test_check_package_global(caplog: MagicMock) -> None:
    pkg: dict = metadata('homeassistant')
    installed_package: str = pkg['name']
    installed_version: str = pkg['version']
    assert package.is_installed(installed_package)
    assert package.is_installed(f'{installed_package}=={installed_version}')
    assert package.is_installed(f'{installed_package}>={installed_version}')
    assert package.is_installed(f'{installed_package}<={installed_version}')
    assert not package.is_installed(f'{installed_package}<{installed_version}')
    assert "Invalid requirement '-1 invalid_package'" in caplog.text

def test_check_package_fragment(caplog: MagicMock) -> None:
    assert not package.is_installed(TEST_ZIP_REQ)
    assert package.is_installed('git+https://github.com/pypa/pip#pip>=1')
    assert not package.is_installed('git+https://github.com/pypa/pip#-1 invalid')
    assert "Invalid requirement 'git+https://github.com/pypa/pip#-1 invalid'" in caplog.text

def test_get_is_installed() -> None:
    pkg: dict = metadata('homeassistant')
    installed_package: str = pkg['name']
    installed_version: str = pkg['version']
    assert package.is_installed(installed_package)
    assert package.is_installed(f'{installed_package}=={installed_version}')
    assert package.is_installed(f'{installed_package}>={installed_version}')
    assert package.is_installed(f'{installed_package}<={installed_version}')
    assert not package.is_installed(f'{installed_package}<{installed_version}')

def test_check_package_previous_failed_install() -> None:
    pkg: dict = metadata('homeassistant')
    installed_package: str = pkg['name']
    installed_version: str = pkg['version']
    with patch('homeassistant.util.package.version', return_value=None):
        assert not package.is_installed(installed_package)
        assert not package.is_installed(f'{installed_package}=={installed_version}')

async def test_is_docker_env(dockerenv: bool, containerenv: bool, kubernetes_service_host: bool, is_official_image: bool) -> None:
    env: dict = {}
    if kubernetes_service_host:
        env['KUBERNETES_SERVICE_HOST'] = 'True'
    package.is_docker_env.cache_clear()
    with patch('homeassistant.util.package.Path', side_effect=new_path_mock), patch('homeassistant.util.package.is_official_image', return_value=is_official_image), patch.dict(os.environ, env):
        assert package.is_docker_env() is any([dockerenv, containerenv, kubernetes_service_host, is_official_image])
