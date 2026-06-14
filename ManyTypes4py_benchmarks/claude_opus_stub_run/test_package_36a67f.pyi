import asyncio
from collections.abc import Generator
import logging
import os
from subprocess import PIPE
import sys
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from homeassistant.util import package

RESOURCE_DIR: str
TEST_NEW_REQ: str
TEST_ZIP_REQ: str

@pytest.fixture
def mock_sys() -> Generator[MagicMock]:
    """Mock sys."""
    ...

@pytest.fixture
def deps_dir() -> str:
    """Return path to deps directory."""
    ...

@pytest.fixture
def lib_dir(deps_dir: str) -> str:
    """Return path to lib directory."""
    ...

@pytest.fixture
def mock_popen(lib_dir: str) -> Generator[MagicMock]:
    """Return a Popen mock."""
    ...

@pytest.fixture
def mock_env_copy() -> Generator[MagicMock]:
    """Mock os.environ.copy."""
    ...

@pytest.fixture
def mock_venv() -> Generator[MagicMock]:
    """Mock homeassistant.util.package.is_virtual_env."""
    ...

def mock_async_subprocess() -> MagicMock:
    """Return an async Popen mock."""
    ...

def test_install(mock_popen: MagicMock, mock_env_copy: MagicMock, mock_sys: MagicMock) -> None:
    """Test an install attempt on a package that doesn't exist."""
    ...

def test_install_with_timeout(mock_popen: MagicMock, mock_env_copy: MagicMock, mock_sys: MagicMock) -> None:
    """Test an install attempt on a package that doesn't exist with a timeout set."""
    ...

def test_install_upgrade(mock_popen: MagicMock, mock_env_copy: MagicMock, mock_sys: MagicMock) -> None:
    """Test an upgrade attempt on a package."""
    ...

def test_install_target(mock_sys: MagicMock, mock_popen: MagicMock, mock_env_copy: MagicMock, mock_venv: MagicMock, is_venv: bool) -> None:
    """Test an install with a target."""
    ...

def test_install_pip_compatibility_no_workaround(mock_sys: MagicMock, mock_popen: MagicMock, mock_env_copy: MagicMock, mock_venv: MagicMock, in_venv: bool, additional_env_vars: dict[str, str]) -> None:
    """Test install will not use pip fallback."""
    ...

def test_install_pip_compatibility_use_workaround(mock_sys: MagicMock, mock_popen: MagicMock, mock_env_copy: MagicMock, mock_venv: MagicMock) -> None:
    """Test install will use pip compatibility fallback."""
    ...

def test_install_error(caplog: pytest.LogCaptureFixture, mock_popen: MagicMock) -> None:
    """Test an install that errors out."""
    ...

def test_install_constraint(mock_popen: MagicMock, mock_env_copy: MagicMock, mock_sys: MagicMock) -> None:
    """Test install with constraint file on not installed package."""
    ...

async def test_async_get_user_site(mock_env_copy: MagicMock) -> None:
    """Test async get user site directory."""
    ...

def test_check_package_global(caplog: pytest.LogCaptureFixture) -> None:
    """Test for an installed package."""
    ...

def test_check_package_fragment(caplog: pytest.LogCaptureFixture) -> None:
    """Test for an installed package with a fragment."""
    ...

def test_get_is_installed() -> None:
    """Test is_installed can parse complex requirements."""
    ...

def test_check_package_previous_failed_install() -> None:
    """Test for when a previously install package failed and left cruft behind."""
    ...

async def test_is_docker_env(dockerenv: bool, containerenv: bool, kubernetes_service_host: bool, is_official_image: bool) -> None:
    """Test is_docker_env."""
    ...