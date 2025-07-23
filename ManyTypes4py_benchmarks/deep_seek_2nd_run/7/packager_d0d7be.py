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
from typing import Any, Set, List, Optional, Tuple, Iterable, Callable
from typing import Iterator, Union, BinaryIO, IO
from typing import Dict, MutableMapping, cast
from chalice.compat import pip_import_string
from chalice.compat import pip_no_compile_c_env_vars
from chalice.compat import pip_no_compile_c_shim
from chalice.utils import OSUtils
from chalice.utils import UI
from chalice.constants import MISSING_DEPENDENCIES_TEMPLATE
import chalice
from chalice import app
from types import TracebackType

StrMap = Dict[str, Any]
OptStrMap = Optional[StrMap]
EnvVars = MutableMapping[str, str]
OptStr = Optional[str]
OptBytes = Optional[bytes]
logger = logging.getLogger(__name__)

class InvalidSourceDistributionNameError(Exception):
    pass

class MissingDependencyError(Exception):
    """Raised when some dependencies could not be packaged for any reason."""

    def __init__(self, missing: Set[Package]) -> None:
        self.missing = missing

class NoSuchPackageError(Exception):
    """Raised when a package name or version could not be found."""

    def __init__(self, package_name: str) -> None:
        super(NoSuchPackageError, self).__init__('Could not satisfy the requirement: %s' % package_name)

class PackageDownloadError(Exception):
    """Generic networking error during a package download."""

class EmptyPackageError(Exception):
    """A deployment package cannot be an empty zip file."""

class UnsupportedPackageError(Exception):
    """Unable to parse package metadata."""

    def __init__(self, package_name: str) -> None:
        super(UnsupportedPackageError, self).__init__('Unable to retrieve name/version for package: %s' % package_name)

class BaseLambdaDeploymentPackager(object):
    _CHALICE_LIB_DIR: str = 'chalicelib'
    _VENDOR_DIR: str = 'vendor'
    _RUNTIME_TO_ABI: Dict[str, str] = {'python3.8': 'cp38', 'python3.9': 'cp39', 'python3.10': 'cp310', 'python3.11': 'cp311', 'python3.12': 'cp312'}

    def __init__(self, osutils: OSUtils, dependency_builder: DependencyBuilder, ui: UI) -> None:
        self._osutils = osutils
        self._dependency_builder = dependency_builder
        self._ui = ui

    def create_deployment_package(self, project_dir: str, python_version: str) -> str:
        raise NotImplementedError('create_deployment_package')

    def _get_requirements_filename(self, project_dir: str) -> str:
        return self._osutils.joinpath(project_dir, 'requirements.txt')

    def _add_vendor_files(self, zipped: ZipFile, dirname: str, prefix: str = '') -> None:
        if not self._osutils.directory_exists(dirname):
            return
        prefix_len = len(dirname) + 1
        for root, _, filenames in self._osutils.walk(dirname, followlinks=True):
            for filename in filenames:
                full_path = self._osutils.joinpath(root, filename)
                zip_path = full_path[prefix_len:]
                if prefix:
                    zip_path = self._osutils.joinpath(prefix, zip_path)
                zipped.write(full_path, zip_path)

    def deployment_package_filename(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version)

    def _deployment_package_filename(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        requirements_filename = self._get_requirements_filename(project_dir)
        hash_contents = self._hash_project_dir(requirements_filename, self._osutils.joinpath(project_dir, self._VENDOR_DIR), project_dir)
        filename = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename = self._osutils.joinpath(project_dir, '.chalice', 'deployments', filename)
        return deployment_package_filename

    def _add_py_deps(self, zip_fileobj: ZipFile, deps_dir: str, prefix: str = '') -> None:
        prefix_len = len(deps_dir) + 1
        for root, dirnames, filenames in self._osutils.walk(deps_dir):
            if root == deps_dir and 'chalice' in dirnames:
                dirnames.remove('chalice')
            for filename in filenames:
                full_path = self._osutils.joinpath(root, filename)
                zip_path = full_path[prefix_len:]
                if prefix:
                    zip_path = self._osutils.joinpath(prefix, zip_path)
                zip_fileobj.write(full_path, zip_path)

    def _add_app_files(self, zip_fileobj: ZipFile, project_dir: str) -> None:
        for full_path, zip_path in self._iter_app_filenames(project_dir):
            zip_fileobj.write(full_path, zip_path)

    def _iter_app_filenames(self, project_dir: str) -> Iterator[Tuple[str, str]]:
        chalice_router = inspect.getfile(app)
        if chalice_router.endswith('.pyc'):
            chalice_router = chalice_router[:-1]
        yield (chalice_router, 'chalice/app.py')
        chalice_init = inspect.getfile(chalice)
        if chalice_init.endswith('.pyc'):
            chalice_init = chalice_init[:-1]
        yield (chalice_init, 'chalice/__init__.py')
        yield (self._osutils.joinpath(project_dir, 'app.py'), 'app.py')
        yield from self._iter_chalice_lib_if_needed(project_dir)

    def _hash_project_dir(self, requirements_filename: str, vendor_dir: str, project_dir: str) -> str:
        if not self._osutils.file_exists(requirements_filename):
            contents = b''
        else:
            contents = cast(bytes, self._osutils.get_file_contents(requirements_filename, binary=True))
        h = hashlib.md5(contents)
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        return h.hexdigest()

    def _hash_vendor_dir(self, vendor_dir: str, md5: Any) -> None:
        for rootdir, _, filenames in self._osutils.walk(vendor_dir, followlinks=True):
            for filename in filenames:
                fullpath = self._osutils.joinpath(rootdir, filename)
                with self._osutils.open(fullpath, 'rb') as f:
                    reader = functools.partial(f.read, 1024 * 1024)
                    for chunk in iter(reader, b''):
                        md5.update(chunk)

    def inject_latest_app(self, deployment_package_filename: str, project_dir: str) -> None:
        """Inject latest version of chalice app into a zip package.

        This method takes a pre-created deployment package and injects
        in the latest chalice app code.  This is useful in the case where
        you have no new package deps but have updated your chalice app code.

        :type deployment_package_filename: str
        :param deployment_package_filename: The zipfile of the
            preexisting deployment package.

        :type project_dir: str
        :param project_dir: Path to chalice project dir.

        """
        self._ui.write('Regen deployment package.\n')
        tmpzip = deployment_package_filename + '.tmp.zip'
        with self._osutils.open_zip(deployment_package_filename, 'r') as inzip:
            with self._osutils.open_zip(tmpzip, 'w', self._osutils.ZIP_DEFLATED) as outzip:
                for el in inzip.infolist():
                    if self._needs_latest_version(el.filename):
                        continue
                    contents = inzip.read(el.filename)
                    outzip.writestr(el, contents)
                self._add_app_files(outzip, project_dir)
        self._osutils.move(tmpzip, deployment_package_filename)

    def _needs_latest_version(self, filename: str) -> bool:
        return filename == 'app.py' or filename.startswith(('chalicelib/', 'chalice/'))

    def _iter_chalice_lib_if_needed(self, project_dir: str) -> Iterator[Tuple[str, str]]:
        libdir = self._osutils.joinpath(project_dir, self._CHALICE_LIB_DIR)
        if self._osutils.directory_exists(libdir):
            for rootdir, _, filenames in self._osutils.walk(libdir):
                for filename in filenames:
                    fullpath = self._osutils.joinpath(rootdir, filename)
                    zip_path = self._osutils.joinpath(self._CHALICE_LIB_DIR, fullpath[len(libdir) + 1:])
                    yield (fullpath, zip_path)

    def _create_output_dir_if_needed(self, package_filename: str) -> None:
        dirname = self._osutils.dirname(self._osutils.abspath(package_filename))
        if not self._osutils.directory_exists(dirname):
            self._osutils.makedirs(dirname)

    def _build_python_dependencies(self, python_version: str, requirements_filepath: str, site_packages_dir: str) -> None:
        try:
            abi = self._RUNTIME_TO_ABI[python_version]
            self._dependency_builder.build_site_packages(abi, requirements_filepath, site_packages_dir)
        except MissingDependencyError as e:
            missing_packages = '\n'.join([p.identifier for p in e.missing])
            self._ui.write(MISSING_DEPENDENCIES_TEMPLATE % missing_packages)

class LambdaDeploymentPackager(BaseLambdaDeploymentPackager):

    def create_deployment_package(self, project_dir: str, python_version: str) -> str:
        msg = 'Creating deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename = self.deployment_package_filename(project_dir, python_version)
        if self._osutils.file_exists(package_filename):
            self._ui.write('Reusing existing deployment package.\n')
            return package_filename
        self._create_output_dir_if_needed(package_filename)
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath = self._get_requirements_filename(project_dir)
            self._build_python_dependencies(python_version, requirements_filepath, site_packages_dir=tmpdir)
            with self._osutils.open_zip(package_filename, 'w', self._osutils.ZIP_DEFLATED) as z:
                self._add_py_deps(z, deps_dir=tmpdir)
                self._add_app_files(z, project_dir)
                self._add_vendor_files(z, self._osutils.joinpath(project_dir, self._VENDOR_DIR))
        return package_filename

class AppOnlyDeploymentPackager(BaseLambdaDeploymentPackager):

    def create_deployment_package(self, project_dir: str, python_version: str) -> str:
        msg = 'Creating app deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename = self.deployment_package_filename(project_dir, python_version)
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
        h = hashlib.md5(b'')
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        digest = h.hexdigest()
        filename = '%s%s-%s.zip' % (prefix, digest, python_version)
        deployment_package_filename = self._osutils.joinpath(project_dir, '.chalice', 'deployments', filename)
        return deployment_package_filename

class LayerDeploymentPackager(BaseLambdaDeploymentPackager):
    _PREFIX: str = 'python/lib/%s/site-packages'

    def create_deployment_package(self, project_dir: str, python_version: str) -> str:
        msg = 'Creating shared layer deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename = self.deployment_package_filename(project_dir, python_version)
        self._create_output_dir_if_needed(package_filename)
        if self._osutils.file_exists(package_filename):
            self._ui.write('  Reusing existing shared layer deployment package.\n')
            return package_filename
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath = self._get_requirements_filename(project_dir)
            self._build_python_dependencies(python_version, requirements_filepath, site_packages_dir=tmpdir)
            with self._osutils.open_zip(package_filename, 'w', self._osutils.ZIP_DEFLATED) as z:
                prefix = self._PREFIX % python_version
                self._add_py_deps(z, deps_dir=tmpdir, prefix=prefix)
                self._add_vendor_files(z, self._osutils.joinpath(project_dir, self._VENDOR_DIR), prefix=prefix)
        self._check_valid_package(package_filename)
        return package_filename

    def _check_valid_package(self, package_filename: str) -> None:
        with self._osutils.open_zip(package_filename, 'r', self._osutils.ZIP_DEFLATED) as z:
            total_size = sum((f.file_size for f in z.infolist()))
            if not total_size > 0:
                self._osutils.remove_file(package_filename)
                raise EmptyPackageError(package_filename)

    def deployment_package_filename(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version, prefix='managed-layer-')

    def _deployment_package_filename(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        requirements_filename = self._get_requirements_filename(project_dir)
        if not self._osutils.file_exists(requirements_filename):
            contents = b''
        else:
            contents = cast(bytes, self._osutils.get_file_contents(requirements_filename, binary=True))
        h = hashlib.md5(contents)
        vendor_dir = self._osutils.joinpath(project_dir, self._VENDOR_DIR)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        hash_contents = h.hexdigest()
        filename = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename = self._osutils.joinpath(project_dir, '.chalice', 'deployments', filename)
        return deployment_package_filename

class DependencyBuilder(object):
    """Build site-packages by manually downloading and unpacking wheels.

    Pip is used to download all the dependency sdists. Then wheels that are
    compatible with lambda are downloaded. Any source packages that do not
    have a matching wheel file are built into a wheel and that file is checked
    for compatibility with the lambda python runtime environment.

    All compatible wheels that are downloaded/built this way are unpacked
    into a site-packages directory, to be included in the bundle by the
    packager.
    """
    _ADDITIONAL_COMPATIBLE_PLATFORM: Set[str] = {'any', 'linux_x86_64'}
    _MANYLINUX_LEGACY_MAP: Dict[str, str] = {'manylinux1_x86_64': 'manylinux_2_5_x86_64', 'manylinux2010_x86_64': 'manylinux_2_12_x86_64', 'manylinux2014_x86_64': 'manylinux_2_17_x86_64'}
    _RUNTIME_GLIBC: Dict[str, Tuple[int, int]] = {'cp27mu': (2, 17), 'cp36m': (2, 17), 'cp37m': (2, 17), 'cp38': (2, 26), 'cp310': (2, 26), 'cp311': (2, 26), 'cp312': (2, 26)}
    _DEFAULT_GLIBC: Tuple[int, int] = (2, 17)
    _COMPATIBLE_PACKAGE_WHITELIST: Set[str] = {'sqlalchemy', 'pyyaml', 'pyrsistent'}

    def __init__(self, osutils: OSUtils, pip_runner: Optional[PipRunner] = None) -> None:
        self._osutils = osutils
        if pip_runner is None:
            pip_runner = PipRunner(SubprocessPip(osutils))
        self._pip = pip_runner

    def _is_compatible_wheel_filename(self, expected_abi: str