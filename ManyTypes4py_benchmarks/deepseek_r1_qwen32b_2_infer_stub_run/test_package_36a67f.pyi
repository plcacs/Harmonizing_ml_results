"""Test Home Assistant package util methods."""

import asyncio
from collections.abc import Generator, Iterable
from importlib.metadata import PackageMetadata
import logging
import os
from subprocess import CompletedProcess
import sys
from unittest.mock import MagicMock, Mock, call
import pytest
from homeassistant.util import package

RESOURCE_DIR: str = ...

TEST_NEW_REQ: str = ...
TEST_ZIP_REQ: str = ...

@pytest.fixture
def mock_sys() -> MagicMock[object]:
    ...

@pytest.fixture
def deps_dir() -> str:
    ...

@pytest.fixture
def lib_dir(deps_dir: str) -> str:
    ...

@pytest.fixture
def mock_popen(lib_dir: str) -> MagicMock[asyncio.subprocess.Process]:
    ...

@pytest.fixture
def mock_env_copy() -> MagicMock[dict[str, str]]:
    ...

@pytest.fixture
def mock_venv() -> MagicMock[bool]:
    ...

def mock_async_subprocess() -> MagicMock[asyncio.subprocess.Process]:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install(mock_popen: MagicMock[asyncio.subprocess.Process], mock_env_copy: MagicMock[dict[str, str]], mock_sys: MagicMock[object]) -> None:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install_with_timeout(mock_popen: MagicMock[asyncio.subprocess.Process], mock_env_copy: MagicMock[dict[str, str]], mock_sys: MagicMock[object]) -> None:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install_upgrade(mock_popen: MagicMock[asyncio.subprocess.Process], mock_env_copy: MagicMock[dict[str, str]], mock_sys: MagicMock[object]) -> None:
    ...

@pytest.mark.parametrize('is_venv', [True, False])
def test_install_target(mock_sys: MagicMock[object], mock_popen: MagicMock[asyncio.subprocess.Process], mock_env_copy: MagicMock[dict[str, str]], mock_venv: MagicMock[bool], is_venv: bool) -> None:
    ...

@pytest.mark.parametrize(('in_venv', 'additional_env_vars'), [(True, {}), (False, {'HTTP_TIMEOUT': '10'})], ids=['in_venv', 'UV_SYSTEM_PYTHON'])
def test_install_pip_compatibility_no_workaround(mock_sys: MagicMock[object], mock_popen: MagicMock[asyncio.subprocess.Process], mock_env_copy: MagicMock[dict[str, str]], mock_venv: MagicMock[bool], in_venv: bool, additional_env_vars: dict[str, str]) -> None:
    ...

def test_install_pip_compatibility_use_workaround(mock_sys: MagicMock[object], mock_popen: MagicMock[asyncio.subprocess.Process], mock_env_copy: MagicMock[dict[str, str]], mock_venv: MagicMock[bool]) -> None:
    ...

@pytest.mark.usefixtures('mock_sys', 'mock_venv')
def test_install_error(caplog: pytest.LogCaptureFixture, mock_popen: MagicMock[asyncio.subprocess.Process]) -> None:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install_constraint(mock_popen: MagicMock[asyncio.subprocess.Process], mock_env_copy: MagicMock[dict[str, str]], mock_sys: MagicMock[object]) -> None:
    ...

async def test_async_get_user_site(mock_env_copy: MagicMock[dict[str, str]]) -> None:
    ...

def test_check_package_global(caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_check_package_fragment(caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_get_is_installed() -> None:
    ...

def test_check_package_previous_failed_install() -> None:
    ...

@pytest.mark.parametrize('dockerenv', [True, False], ids=['dockerenv', 'not_dockerenv'])
@pytest.mark.parametrize('containerenv', [True, False], ids=['containerenv', 'not_containerenv'])
@pytest.mark.parametrize('kubernetes_service_host', [True, False], ids=['kubernetes', 'not_kubernetes'])
@pytest.mark.parametrize('is_official_image', [True, False], ids=['official_image', 'not_official_image'])
async def test_is_docker_env(dockerenv: bool, containerenv: bool, kubernetes_service_host: bool, is_official_image: bool) -> None:
    ...

def install_package(req: str, upgrade: bool = ..., constraints: Optional[str] = ..., timeout: Optional[int] = ...) -> bool:
    ...

def is_installed(req: str, version: Optional[str] = ...) -> bool:
    ...

def async_get_user_site(deps_dir: str) -> Awaitable[str]:
    ...

def is_docker_env() -> bool:
    ...