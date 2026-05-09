from __future__ import annotations
import sys
import hashlib
import inspect
import re
import subprocess
import logging
import functools
from email.parser import FeedParser
from email.message import Message
from zipfile import ZipFile
from typing import Any, Set, List, Optional, Tuple, Iterable, Callable, Dict, MutableMapping
from typing import Iterator
from chalice.compat import pip_import_string
from chalice.compat import pip_no_compile_c_env_vars
from chalice.compat import pip_no_compile_c_shim
from chalice.utils import OSUtils
from chalice.utils import UI
from chalice.constants import MISSING_DEPENDENCIES_TEMPLATE
import chalice
from chalice import app
from chalice.packager import StrMap, OptStrMap, EnvVars, OptStr, OptBytes
from chalice.packager import LambdaDeploymentPackager, AppOnlyDeploymentPackager, LayerDeploymentPackager
from chalice.packager import DependencyBuilder, Package, SDistMetadataFetcher, SubprocessPip, PipRunner

class InvalidSourceDistributionNameError(Exception):
    pass

class MissingDependencyError(Exception):
    def __init__(self, missing: Set[Package]):
        self.missing = missing

class NoSuchPackageError(Exception):
    def __init__(self, package_name: str):
        super().__init__(f'Could not satisfy the requirement: {package_name}')

class PackageDownloadError(Exception):
    pass

class EmptyPackageError(Exception):
    pass

class UnsupportedPackageError(Exception):
    def __init__(self, package_name: str):
        super().__init__(f'Unable to retrieve name/version for package: {package_name}')

class BaseLambdaDeploymentPackager:
    _CHALICE_LIB_DIR: str
    _VENDOR_DIR: str
    _RUNTIME_TO_ABI: Dict[str, str]

    def __init__(self, osutils: OSUtils, dependency_builder: DependencyBuilder, ui: UI):
        self._osutils: OSUtils
        self._dependency_builder: DependencyBuilder
        self._ui: UI

    def create_deployment_package(self, project_dir: str, python_version: str) -> str:
        # ...

class LambdaDeploymentPackager(BaseLambdaDeploymentPackager):
    # ...

class AppOnlyDeploymentPackager(BaseLambdaDeploymentPackager):
    # ...

class LayerDeploymentPackager(BaseLambdaDeploymentPackager):
    # ...

class DependencyBuilder:
    _ADDITIONAL_COMPATIBLE_PLATFORM: Set[str]
    _MANYLINUX_LEGACY_MAP: Dict[str, str]
    _RUNTIME_GLIBC: Dict[str, Tuple[int, int]]
    _DEFAULT_GLIBC: Tuple[int, int]
    _COMPATIBLE_PACKAGE_WHITELIST: Set[str]

    def __init__(self, osutils: OSUtils, pip_runner: PipRunner = None):
        self._osutils: OSUtils
        self._pip: PipRunner

    def _is_compatible_wheel_filename(self, abi: str, filename: str) -> bool:
        # ...

    def _download_dependencies(self, abi: str, directory: str, requirements_filename: str) -> Tuple[Set[Package], Set[Package]]:
        # ...

    def _install_wheels(self, src_dir: str, dst_dir: str, wheels: Set[Package]) -> None:
        # ...

class Package:
    def __init__(self, directory: str, filename: str, osutils: OSUtils = None):
        self._directory: str
        self._filename: str
        self._osutils: OSUtils

    def __str__(self) -> str:
        # ...

class SDistMetadataFetcher:
    _SETUPTOOLS_SHIM: str

    def __init__(self, osutils: OSUtils = None):
        self._osutils: OSUtils

    def _parse_pkg_info_file(self, filepath: str) -> Message:
        # ...

    def get_package_name_and_version(self, sdist_path: str) -> Tuple[str, str]:
        # ...

class SubprocessPip:
    def __init__(self, osutils: OSUtils = None, import_string: str = None):
        self._osutils: OSUtils
        self._import_string: str

   