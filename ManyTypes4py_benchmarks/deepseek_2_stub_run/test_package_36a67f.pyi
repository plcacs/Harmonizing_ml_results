```python
import asyncio.subprocess
import os
from collections.abc import Generator
from typing import Any

def install_package(
    pkg: str,
    upgrade: bool = ...,
    *,
    timeout: int | None = ...,
    target: str | None = ...,
    constraints: str | None = ...,
    find_links: str | None = ...,
    no_index: bool = ...,
    extra_index_url: str | None = ...,
    pre: bool = ...,
) -> bool: ...

async def async_get_user_site(deps_dir: str) -> str: ...

def is_installed(package: str) -> bool: ...

def is_virtual_env() -> bool: ...

def is_docker_env() -> bool: ...

def is_official_image() -> bool: ...

RESOURCE_DIR: str = ...
TEST_NEW_REQ: str = ...
TEST_ZIP_REQ: str = ...
```