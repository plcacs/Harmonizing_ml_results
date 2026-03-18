```python
import asyncio.subprocess
import os
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock

import pytest

def install_package(
    pkg: str,
    upgrade: bool = True,
    *,
    target: str | None = None,
    constraints: str | None = None,
    timeout: int | None = None,
    find_links: str | None = None,
    no_deps: bool = False,
    user: bool = False,
    extra_index_url: str | None = None,
    pre: bool = False,
) -> bool: ...

async def async_get_user_site(deps_dir: str) -> str: ...

def is_installed(pkg: str) -> bool: ...

def is_virtual_env() -> bool: ...

def is_docker_env() -> bool: ...

def is_official_image() -> bool: ...

RESOURCE_DIR: str = ...
TEST_NEW_REQ: str = ...
TEST_ZIP_REQ: str = ...

@pytest.fixture
def mock_sys() -> Generator[Mock, Any, None]: ...

@pytest.fixture
def deps_dir() -> str: ...

@pytest.fixture
def lib_dir(deps_dir: str) -> str: ...

@pytest.fixture
def mock_popen(lib_dir: str) -> Generator[Mock, Any, None]: ...

@pytest.fixture
def mock_env_copy() -> Generator[Mock, Any, None]: ...

@pytest.fixture
def mock_venv() -> Generator[Mock, Any, None]: ...

def mock_async_subprocess() -> Mock: ...
```