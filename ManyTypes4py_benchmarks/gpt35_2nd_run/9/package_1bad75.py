from __future__ import annotations
import asyncio
from functools import cache
from importlib.metadata import PackageNotFoundError, version
import logging
import os
from pathlib import Path
import site
from subprocess import PIPE, Popen
import sys
from urllib.parse import urlparse
from packaging.requirements import InvalidRequirement, Requirement
from .system_info import is_official_image

_LOGGER: logging.Logger = logging.getLogger(__name__)

def is_virtual_env() -> bool:
    return getattr(sys, 'base_prefix', sys.prefix) != sys.prefix or hasattr(sys, 'real_prefix')

@cache
def is_docker_env() -> bool:
    return Path('/.dockerenv').exists() or Path('/run/.containerenv').exists() or 'KUBERNETES_SERVICE_HOST' in os.environ or is_official_image()

def get_installed_versions(specifiers: set[str]) -> set[str]:
    return {specifier for specifier in specifiers if is_installed(specifier)}

def is_installed(requirement_str: str) -> bool:
    try:
        req = Requirement(requirement_str)
    except InvalidRequirement:
        if '#' not in requirement_str:
            _LOGGER.error("Invalid requirement '%s'", requirement_str)
            return False
        try:
            req = Requirement(urlparse(requirement_str).fragment)
        except InvalidRequirement:
            _LOGGER.error("Invalid requirement '%s'", requirement_str)
            return False
    try:
        if (installed_version := version(req.name)) is None:
            _LOGGER.error('Installed version for %s resolved to None', req.name)
            return False
        return req.specifier.contains(installed_version, prereleases=True)
    except PackageNotFoundError:
        return False

_UV_ENV_PYTHON_VARS: tuple[str, str] = ('UV_SYSTEM_PYTHON', 'UV_PYTHON')

def install_package(package: str, upgrade: bool = True, target: str | None = None, constraints: str | None = None, timeout: int | None = None) -> bool:
    _LOGGER.info('Attempting install of %s', package)
    env = os.environ.copy()
    args = [sys.executable, '-m', 'uv', 'pip', 'install', '--quiet', package, '--index-strategy', 'unsafe-first-match']
    if timeout:
        env['HTTP_TIMEOUT'] = str(timeout)
    if upgrade:
        args.append('--upgrade')
    if constraints is not None:
        args += ['--constraint', constraints]
    if target:
        abs_target = os.path.abspath(target)
        args += ['--target', abs_target]
    elif not is_virtual_env() and (not any((var in env for var in _UV_ENV_PYTHON_VARS))) and (abs_target := site.getusersitepackages()):
        args += ['--python', sys.executable, '--target', abs_target]
    _LOGGER.debug('Running uv pip command: args=%s', args)
    with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env, close_fds=False) as process:
        _, stderr = process.communicate()
        if process.returncode != 0:
            _LOGGER.error('Unable to install package %s: %s', package, stderr.decode('utf-8').lstrip().strip())
            return False
    return True

async def async_get_user_site(deps_dir: str) -> str:
    env = os.environ.copy()
    env['PYTHONUSERBASE'] = os.path.abspath(deps_dir)
    args = [sys.executable, '-m', 'site', '--user-site']
    process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=env, close_fds=False)
    stdout, _ = await process.communicate()
    return stdout.decode().strip()
