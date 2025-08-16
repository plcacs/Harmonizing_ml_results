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
from typing import Any, Set, List, Optional, Tuple, Iterable, Callable, Iterator, Dict, MutableMapping
from chalice.compat import pip_import_string
from chalice.compat import pip_no_compile_c_env_vars
from chalice.compat import pip_no_compile_c_shim
from chalice.utils import OSUtils
from chalice.utils import UI
from chalice.constants import MISSING_DEPENDENCIES_TEMPLATE
import chalice
from chalice import app

StrMap: Dict[str, Any] = {}
OptStrMap: Optional[StrMap] = None
EnvVars: MutableMapping = {}
OptStr: Optional[str] = None
OptBytes: Optional[bytes] = None
logger: logging.Logger = logging.getLogger(__name__)

class InvalidSourceDistributionNameError(Exception):
    pass

class MissingDependencyError(Exception):
    def __init__(self, missing: Set[str]) -> None:
        self.missing: Set[str] = missing

class NoSuchPackageError(Exception):
    def __init__(self, package_name: str) -> None:
        super(NoSuchPackageError, self).__init__('Could not satisfy the requirement: %s' % package_name)

class PackageDownloadError(Exception):
    pass

class EmptyPackageError(Exception):
    pass

class UnsupportedPackageError(Exception):
    def __init__(self, package_name: str) -> None:
        super(UnsupportedPackageError, self).__init__('Unable to retrieve name/version for package: %s' % package_name)

class BaseLambdaDeploymentPackager:
    _CHALICE_LIB_DIR: str = 'chalicelib'
    _VENDOR_DIR: str = 'vendor'
    _RUNTIME_TO_ABI: Dict[str, str] = {'python3.8': 'cp38', 'python3.9': 'cp39', 'python3.10': 'cp310', 'python3.11': 'cp311', 'python3.12': 'cp312'}

    def __init__(self, osutils: OSUtils, dependency_builder: DependencyBuilder, ui: UI) -> None:
        self._osutils: OSUtils = osutils
        self._dependency_builder: DependencyBuilder = dependency_builder
        self._ui: UI = ui

    def create_deployment_package(self, project_dir: str, python_version: str) -> None:
        raise NotImplementedError('create_deployment_package')

    def _get_requirements_filename(self, project_dir: str) -> str:
        return self._osutils.joinpath(project_dir, 'requirements.txt')

    def _add_vendor_files(self, zipped: ZipFile, dirname: str, prefix: str = '') -> None:
        if not self._osutils.directory_exists(dirname):
            return
        prefix_len: int = len(dirname) + 1
        for root, _, filenames in self._osutils.walk(dirname, followlinks=True):
            for filename in filenames:
                full_path: str = self._osutils.joinpath(root, filename)
                zip_path: str = full_path[prefix_len:]
                if prefix:
                    zip_path = self._osutils.joinpath(prefix, zip_path)
                zipped.write(full_path, zip_path)

    def deployment_package_filename(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version)

    def _deployment_package_filename(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        requirements_filename: str = self._get_requirements_filename(project_dir)
        hash_contents: str = self._hash_project_dir(requirements_filename, self._osutils.joinpath(project_dir, self._VENDOR_DIR), project_dir)
        filename: str = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename: str = self._osutils.joinpath(project_dir, '.chalice', 'deployments', filename)
        return deployment_package_filename

    def _add_py_deps(self, zip_fileobj: ZipFile, deps_dir: str, prefix: str = '') -> None:
        prefix_len: int = len(deps_dir) + 1
        for root, dirnames, filenames in self._osutils.walk(deps_dir):
            if root == deps_dir and 'chalice' in dirnames:
                dirnames.remove('chalice')
            for filename in filenames:
                full_path: str = self._osutils.joinpath(root, filename)
                zip_path: str = full_path[prefix_len:]
                if prefix:
                    zip_path = self._osutils.joinpath(prefix, zip_path)
                zip_fileobj.write(full_path, zip_path)

    def _add_app_files(self, zip_fileobj: ZipFile, project_dir: str) -> None:
        for full_path, zip_path in self._iter_app_filenames(project_dir):
            zip_fileobj.write(full_path, zip_path)

    def _iter_app_filenames(self, project_dir: str) -> Iterable[Tuple[str, str]]:
        chalice_router: str = inspect.getfile(app)
        if chalice_router.endswith('.pyc'):
            chalice_router = chalice_router[:-1]
        yield (chalice_router, 'chalice/app.py')
        chalice_init: str = inspect.getfile(chalice)
        if chalice_init.endswith('.pyc'):
            chalice_init = chalice_init[:-1]
        yield (chalice_init, 'chalice/__init__.py')
        yield (self._osutils.joinpath(project_dir, 'app.py'), 'app.py')
        yield from self._iter_chalice_lib_if_needed(project_dir)

    def _hash_project_dir(self, requirements_filename: str, vendor_dir: str, project_dir: str) -> str:
        if not self._osutils.file_exists(requirements_filename):
            contents: bytes = b''
        else:
            contents: bytes = cast(bytes, self._osutils.get_file_contents(requirements_filename, binary=True))
        h: hashlib.md5 = hashlib.md5(contents)
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader: Callable = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        return h.hexdigest()

    def _hash_vendor_dir(self, vendor_dir: str, md5: hashlib.md5) -> None:
        for rootdir, _, filenames in self._osutils.walk(vendor_dir, followlinks=True):
            for filename in filenames:
                fullpath: str = self._osutils.joinpath(rootdir, filename)
                with self._osutils.open(fullpath, 'rb') as f:
                    reader: Callable = functools.partial(f.read, 1024 * 1024)
                    for chunk in iter(reader, b''):
                        md5.update(chunk)

    def inject_latest_app(self, deployment_package_filename: str, project_dir: str) -> None:
        self._ui.write('Regen deployment package.\n')
        tmpzip: str = deployment_package_filename + '.tmp.zip'
        with self._osutils.open_zip(deployment_package_filename, 'r') as inzip:
            with self._osutils.open_zip(tmpzip, 'w', self._osutils.ZIP_DEFLATED) as outzip:
                for el in inzip.infolist():
                    if self._needs_latest_version(el.filename):
                        continue
                    contents: bytes = inzip.read(el.filename)
                    outzip.writestr(el, contents)
                self._add_app_files(outzip, project_dir)
        self._osutils.move(tmpzip, deployment_package_filename)

    def _needs_latest_version(self, filename: str) -> bool:
        return filename == 'app.py' or filename.startswith(('chalicelib/', 'chalice/'))

    def _iter_chalice_lib_if_needed(self, project_dir: str) -> Iterable[Tuple[str, str]]:
        libdir: str = self._osutils.joinpath(project_dir, self._CHALICE_LIB_DIR)
        if self._osutils.directory_exists(libdir):
            for rootdir, _, filenames in self._osutils.walk(libdir):
                for filename in filenames:
                    fullpath: str = self._osutils.joinpath(rootdir, filename)
                    zip_path: str = self._osutils.joinpath(self._CHALICE_LIB_DIR, fullpath[len(libdir) + 1:])
                    yield (fullpath, zip_path)

    def _create_output_dir_if_needed(self, package_filename: str) -> None:
        dirname: str = self._osutils.dirname(self._osutils.abspath(package_filename))
        if not self._osutils.directory_exists(dirname):
            self._osutils.makedirs(dirname)

    def _build_python_dependencies(self, python_version: str, requirements_filepath: str, site_packages_dir: str) -> None:
        try:
            abi: str = self._RUNTIME_TO_ABI[python_version]
            self._dependency_builder.build_site_packages(abi, requirements_filepath, site_packages_dir)
        except MissingDependencyError as e:
            missing_packages: str = '\n'.join([p.identifier for p in e.missing])
            self._ui.write(MISSING_DEPENDENCIES_TEMPLATE % missing_packages)

class LambdaDeploymentPackager(BaseLambdaDeploymentPackager):

    def create_deployment_package(self, project_dir: str, python_version: str) -> None:
        msg: str = 'Creating deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename: str = self.deployment_package_filename(project_dir, python_version)
        if self._osutils.file_exists(package_filename):
            self._ui.write('Reusing existing deployment package.\n')
            return package_filename
        self._create_output_dir_if_needed(package_filename)
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath: str = self._get_requirements_filename(project_dir)
            self._build_python_dependencies(python_version, requirements_filepath, site_packages_dir=tmpdir)
            with self._osutils.open_zip(package_filename, 'w', self._osutils.ZIP_DEFLATED) as z:
                self._add_py_deps(z, deps_dir=tmpdir)
                self._add_app_files(z, project_dir)
                self._add_vendor_files(z, self._osutils.joinpath(project_dir, self._VENDOR_DIR))
        return package_filename

class AppOnlyDeploymentPackager(BaseLambdaDeploymentPackager):

    def create_deployment_package(self, project_dir: str, python_version: str) -> None:
        msg: str = 'Creating app deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename: str = self.deployment_package_filename(project_dir, python_version)
        if self._osutils.file_exists(package_filename):
            self._ui.write('  Reusing existing app deployment package.\n')
            return package_filename
        self._create_output_dir_if_needed(package_filename)
        with self._osutils.open_zip(package_filename, 'w', self._osutils.ZIP_DEFLATED) as z:
            self._add_app_files(z, project_dir)
        return package_filename

    def deployment_package_filename(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version, prefix='appcode-')

    def _deployment_package_filename(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        h: hashlib.md5 = hashlib.md5(b'')
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader: Callable = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        digest: str = h.hexdigest()
        filename: str = '%s%s-%s.zip' % (prefix, digest, python_version)
        deployment_package_filename: str = self._osutils.joinpath(project_dir, '.chalice', 'deployments', filename)
        return deployment_package_filename

class LayerDeploymentPackager(BaseLambdaDeploymentPackager):
    _PREFIX: str = 'python/lib/%s/site-packages'

    def create_deployment_package(self, project_dir: str, python_version: str) -> None:
        msg: str = 'Creating shared layer deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename: str = self.deployment_package_filename(project_dir, python_version)
        self._create_output_dir_if_needed(package_filename)
        if self._osutils.file_exists(package_filename):
            self._ui.write('  Reusing existing shared layer deployment package.\n')
            return package_filename
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath: str = self._get_requirements_filename(project_dir)
            self._build_python_dependencies(python_version, requirements_filepath, site_packages_dir=tmpdir)
            with self._osutils.open_zip(package_filename, 'w', self._osutils.ZIP_DEFLATED) as z:
                prefix: str = self._PREFIX % python_version
                self._add_py_deps(z, deps_dir=tmpdir, prefix=prefix)
                self._add_vendor_files(z, self._osutils.joinpath(project_dir, self._VENDOR_DIR), prefix=prefix)
        self._check_valid_package(package_filename)
        return package_filename

    def _check_valid_package(self, package_filename: str) -> None:
        with self._osutils.open_zip(package_filename, 'r', self._osutils.ZIP_DEFLATED) as z:
            total_size: int = sum((f.file_size for f in z.infolist()))
            if not total_size > 0:
                self._osutils.remove_file(package_filename)
                raise EmptyPackageError(package_filename)

    def deployment_package_filename(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version, prefix='managed-layer-')

    def _deployment_package_filename(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        requirements_filename: str = self._get_requirements_filename(project_dir)
        if not self._osutils.file_exists(requirements_filename):
            contents: bytes = b''
        else:
            contents: bytes = cast(bytes, self._osutils.get_file_contents(requirements_filename, binary=True))
        h: hashlib.md5 = hashlib.md5(contents)
        vendor_dir: str = self._osutils.joinpath(project_dir, self._VENDOR_DIR)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        hash_contents: str = h.hexdigest()
        filename: str = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename: str = self._osutils.joinpath(project_dir, '.chalice', 'deployments', filename)
        return deployment_package_filename

class DependencyBuilder:
    def __init__(self, osutils: OSUtils, pip_runner: Optional[PipRunner] = None) -> None:
        self._osutils: OSUtils = osutils
        if pip_runner is None:
            pip_runner = PipRunner(SubprocessPip(osutils))
        self._pip: PipRunner = pip_runner

    def _is_compatible_wheel_filename(self, expected_abi: str, filename: str) -> bool:
        pass

    def _is_compatible_platform_tag(self, expected_abi: str, platform: str) -> bool:
        pass

    def _iter_all_compatibility_tags(self, wheel: str) -> Iterable[Tuple[str, str, str]]:
        pass

    def _has_at_least_one_package(self, filename: str) -> bool:
        pass

    def _download_all_dependencies(self, requirements_filename: str, directory: str) -> Set[Package]:
        pass

    def _download_binary_wheels(self, abi: str, packages: Set[str], directory: str) -> None:
        pass

    def _download_sdists(self, packages: Set[str], directory: str) -> None:
        pass

    def _find_sdists(self, directory: str) -> Set[Package]:
        pass

    def _build_sdists(self, sdists: Set[Package], directory: str, compile_c: bool) -> None:
        pass

    def _categorize_wheel_files(self, abi: str, directory: str) -> Tuple[Set[Package], Set[Package]]:
        pass

    def _categorize_deps(self, abi: str, deps: Set[Package]) -> Tuple[Set[Package], Set[Package], Set[Package]]:
        pass

    def _download_dependencies(self, abi: str, directory: str, requirements_filename: str) -> Tuple[Set[Package], Set[Package]]:
        pass

    def _apply_wheel_whitelist(self, compatible_wheels: Set[Package], incompatible_wheels: Set[Package]) -> Tuple[Set[Package], Set[Package]]:
        pass

    def _install_purelib_and_platlib(self, wheel: Package, root: str) -> None:
        pass

    def _install_wheels(self, src_dir: str, dst_dir: str, wheels: Set[Package]) -> None:
        pass

    def build_site_packages(self, abi: str, requirements_filepath: str, target_directory: str) -> None:
        pass

class Package:
    def __init__(self, directory: str, filename: str, osutils: Optional[OSUtils] = None) -> None:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def data_dir(self) -> str:
        pass

    def matches_data_dir(self, dirname: str) -> bool:
        pass

    @property
    def identifier(self) -> str:
        pass

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

    def __eq__(self, other: Any) -> bool:
        pass

    def __hash__(self) -> int:
        pass

    def _calculate_name_and_version(self) -> Tuple[str, str]:
        pass

    def _normalize_name(self, name: str) -> str:
        pass

class SDistMetadataFetcher:
    def __init__(self, osutils: Optional[OSUtils] = None) -> None:
        pass

    def _parse_pkg_info_file(self, filepath: str) -> Message:
        pass

    def _get_pkg_info_filepath(self, package_dir: str) -> str:
        pass

    def _unpack_sdist_into_dir(self, sdist_path: str, unpack_dir: str) -> str:
        pass

    def get_package_name_and_version(self, sdist_path: str) -> Tuple[str, str]:
        pass

class SubprocessPip:
    def __init__(self, osutils: Optional[OSUtils] = None, import_string: Optional[str] = None) -> None:
        pass

    def