"""Test Home Assistant package util methods."""

import asyncio
from collections.abc import Generator
from importlib.metadata import metadata
import logging
import os
from subprocess import PIPE
import sys
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    call,
    patch,
)
import pytest
from homeassistant.util import package

RESOURCE_DIR: str = ...
TEST_NEW_REQ: str = ...
TEST_ZIP_REQ: str = ...

@pytest.fixture
def mock_sys() -> Generator[MagicMock[object], None, None]:
    ...

@pytest.fixture
def deps_dir() -> str:
    ...

@pytest.fixture
def lib_dir(deps_dir: str) -> str:
    ...

@pytest.fixture
def mock_popen(lib_dir: str) -> Generator[MagicMock, None, None]:
    ...

@pytest.fixture
def mock_env_copy() -> Generator[dict, None, None]:
    ...

@pytest.fixture
def mock_venv() -> Generator[bool, None, None]:
    ...

def mock_async_subprocess() -> AsyncMock:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install(mock_popen: MagicMock, mock_env_copy: dict, mock_sys: MagicMock) -> None:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install_with_timeout(mock_popen: MagicMock, mock_env_copy: dict, mock_sys: MagicMock) -> None:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install_upgrade(mock_popen: MagicMock, mock_env_copy: dict, mock_sys: MagicMock) -> None:
    ...

@pytest.mark.parametrize('is_venv', [True, False])
def test_install_target(mock_sys: MagicMock, mock_popen: MagicMock, mock_env_copy: dict, mock_venv: bool, is_venv: bool) -> None:
    ...

@pytest.mark.parametrize(('in_venv', 'additional_env_vars'), [(True, {}), (False, {'HTTP_TIMEOUT': str}), (False, {'UV_SYSTEM_PYTHON': str}), (False, {'HTTP_TIMEOUT': str, 'UV_SYSTEM_PYTHON': str})], ids=['in_venv', 'UV_SYSTEM_PYTHON', 'UV_PYTHON', 'UV_SYSTEM_PYTHON and UV_PYTHON'])
def test_install_pip_compatibility_no_workaround(mock_sys: MagicMock, mock_popen: MagicMock, mock_env_copy: dict, mock_venv: bool, in_venv: bool, additional_env_vars: dict) -> None:
    ...

def test_install_pip_compatibility_use_workaround(mock_sys: MagicMock, mock_popen: MagicMock, mock_env_copy: dict, mock_venv: bool) -> None:
    ...

@pytest.mark.usefixtures('mock_sys', 'mock_venv')
def test_install_error(caplog: pytest.LogCaptureFixture, mock_popen: MagicMock) -> None:
    ...

@pytest.mark.usefixtures('mock_venv')
def test_install_constraint(mock_popen: MagicMock, mock_env_copy: dict, mock_sys: MagicMock) -> None:
    ...

async def test_async_get_user_site(mock_env_copy: dict) -> None:
    ...

def test_check_package_global(caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_check_package_fragment(caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_get_is_installed() -> None:
    ...

def test_check_package_previous_failed_install(caplog: pytest.LogCaptureFixture) -> None:
    ...

@pytest.mark.parametrize('dockerenv', [True, False], ids=['dockerenv', 'not_dockerenv'])
@pytest.mark.parametrize('containerenv', [True, False], ids=['containerenv', 'not_containerenv'])
@pytest.mark.parametrize('kubernetes_service_host', [True, False], ids=['kubernetes', 'not_kubernetes'])
@pytest.mark.parametrize('is_official_image', [True, False], ids=['official_image', 'not_official_image'])
async def test_is_docker_env(dockerenv: bool, containerenv: bool, kubernetes_service_host: bool, is_official_image: bool) -> None:
    ...