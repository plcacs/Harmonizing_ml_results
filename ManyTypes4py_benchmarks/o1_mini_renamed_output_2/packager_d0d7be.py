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
from typing import Any, Set, List, Optional, Tuple, Iterable, Callable, Iterator, Dict, MutableMapping, cast
from chalice.compat import pip_import_string
from chalice.compat import pip_no_compile_c_env_vars
from chalice.compat import pip_no_compile_c_shim
from chalice.utils import OSUtils
from chalice.utils import UI
from chalice.constants import MISSING_DEPENDENCIES_TEMPLATE
import chalice
from chalice import app

StrMap = Dict[str, Any]
OptStrMap = Optional[StrMap]
EnvVars = MutableMapping
OptStr = Optional[str]
OptBytes = Optional[bytes]
logger = logging.getLogger(__name__)


class InvalidSourceDistributionNameError(Exception):
    pass


class MissingDependencyError(Exception):
    """Raised when some dependencies could not be packaged for any reason."""

    def __init__(self, missing: Any) -> None:
        self.missing = missing


class NoSuchPackageError(Exception):
    """Raised when a package name or version could not be found."""

    def __init__(self, package_name: str) -> None:
        super(NoSuchPackageError, self).__init__(
            'Could not satisfy the requirement: %s' % package_name)


class PackageDownloadError(Exception):
    """Generic networking error during a package download."""
    pass


class EmptyPackageError(Exception):
    """A deployment package cannot be an empty zip file."""
    pass


class UnsupportedPackageError(Exception):
    """Unable to parse package metadata."""

    def __init__(self, package_name: str) -> None:
        super(UnsupportedPackageError, self).__init__(
            'Unable to retrieve name/version for package: %s' % package_name)


class BaseLambdaDeploymentPackager:
    _CHALICE_LIB_DIR: str = 'chalicelib'
    _VENDOR_DIR: str = 'vendor'
    _RUNTIME_TO_ABI: Dict[str, str] = {
        'python3.8': 'cp38', 
        'python3.9': 'cp39',
        'python3.10': 'cp310', 
        'python3.11': 'cp311', 
        'python3.12': 'cp312'
    }

    def __init__(self, osutils: OSUtils, dependency_builder: DependencyBuilder, ui: UI) -> None:
        self._osutils: OSUtils = osutils
        self._dependency_builder: DependencyBuilder = dependency_builder
        self._ui: UI = ui

    def func_ceu6ued4(self, project_dir: str, python_version: str) -> Any:
        raise NotImplementedError('create_deployment_package')

    def func_kh0f0cgp(self, project_dir: str) -> str:
        return self._osutils.joinpath(project_dir, 'requirements.txt')

    def func_t3nyriwr(self, zipped: ZipFile, dirname: str, prefix: str = '') -> None:
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

    def func_32zdtec8(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version)

    def func_sra9ju6u(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        requirements_filename: str = self._get_requirements_filename(project_dir)
        hash_contents: str = self._hash_project_dir(
            requirements_filename, 
            self._osutils.joinpath(project_dir, self._VENDOR_DIR), 
            project_dir
        )
        filename: str = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename: str = self._osutils.joinpath(
            project_dir,
            '.chalice', 
            'deployments', 
            filename
        )
        return deployment_package_filename

    def func_4pwjgz8d(self, zip_fileobj: ZipFile, deps_dir: str, prefix: str = '') -> None:
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

    def func_xxn09sa1(self, zip_fileobj: ZipFile, project_dir: str) -> None:
        for full_path, zip_path in self._iter_app_filenames(project_dir):
            zip_fileobj.write(full_path, zip_path)

    def func_u736xen7(self, project_dir: str) -> Iterator[Tuple[str, str]]:
        chalice_router: str = inspect.getfile(app)
        if chalice_router.endswith('.pyc'):
            chalice_router = chalice_router[:-1]
        yield chalice_router, 'chalice/app.py'
        chalice_init: str = inspect.getfile(chalice)
        if chalice_init.endswith('.pyc'):
            chalice_init = chalice_init[:-1]
        yield chalice_init, 'chalice/__init__.py'
        yield self._osutils.joinpath(project_dir, 'app.py'), 'app.py'
        yield from self._iter_chalice_lib_if_needed(project_dir)

    def func_ifz9jvwm(self, requirements_filename: str, vendor_dir: str, project_dir: str) -> str:
        if not self._osutils.file_exists(requirements_filename):
            contents: bytes = b''
        else:
            contents = cast(bytes, self._osutils.get_file_contents(
                requirements_filename, binary=True))
        h: hashlib._hashlib.HASH = hashlib.md5(contents)
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        return h.hexdigest()

    def func_pos8i7r1(self, vendor_dir: str, md5: hashlib._hashlib.HASH) -> None:
        for rootdir, _, filenames in self._osutils.walk(vendor_dir, followlinks=True):
            for filename in filenames:
                fullpath: str = self._osutils.joinpath(rootdir, filename)
                with self._osutils.open(fullpath, 'rb') as f:
                    reader = functools.partial(f.read, 1024 * 1024)
                    for chunk in iter(reader, b''):
                        md5.update(chunk)

    def func_fktefsts(self, deployment_package_filename: str, project_dir: str) -> None:
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

    def func_8ic50shz(self, filename: str) -> bool:
        return filename == 'app.py' or filename.startswith(('chalicelib/', 'chalice/'))

    def func_wxztma6g(self, project_dir: str) -> Iterator[Tuple[str, str]]:
        libdir: str = self._osutils.joinpath(project_dir, self._CHALICE_LIB_DIR)
        if self._osutils.directory_exists(libdir):
            for rootdir, _, filenames in self._osutils.walk(libdir):
                for filename in filenames:
                    fullpath: str = self._osutils.joinpath(rootdir, filename)
                    zip_path: str = self._osutils.joinpath(
                        self._CHALICE_LIB_DIR,
                        fullpath[len(libdir) + 1:]
                    )
                    yield fullpath, zip_path

    def func_qv77aull(self, package_filename: str) -> None:
        dirname: str = self._osutils.dirname(self._osutils.abspath(package_filename))
        if not self._osutils.directory_exists(dirname):
            self._osutils.makedirs(dirname)

    def func_501tq1ed(self, python_version: str, requirements_filepath: str, site_packages_dir: str) -> None:
        try:
            abi: str = self._RUNTIME_TO_ABI[python_version]
            self._dependency_builder.build_site_packages(abi,
                requirements_filepath, site_packages_dir)
        except MissingDependencyError as e:
            missing_packages: str = '\n'.join([p.identifier for p in e.missing])
            self._ui.write(MISSING_DEPENDENCIES_TEMPLATE % missing_packages)


class LambdaDeploymentPackager(BaseLambdaDeploymentPackager):

    def func_ceu6ued4(self, project_dir: str, python_version: str) -> str:
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
            self._build_python_dependencies(
                python_version,
                requirements_filepath, 
                site_packages_dir=tmpdir
            )
            with self._osutils.open_zip(package_filename, 'w', self._osutils.ZIP_DEFLATED) as z:
                self._add_py_deps(z, deps_dir=tmpdir)
                self._add_app_files(z, project_dir)
                self._add_vendor_files(z, self._osutils.joinpath(
                    project_dir, self._VENDOR_DIR))
        return package_filename


class AppOnlyDeploymentPackager(BaseLambdaDeploymentPackager):

    def func_ceu6ued4(self, project_dir: str, python_version: str) -> str:
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

    def func_32zdtec8(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version, prefix='appcode-')

    def func_sra9ju6u(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        h: hashlib._hashlib.HASH = hashlib.md5(b'')
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        digest: str = h.hexdigest()
        filename: str = '%s%s-%s.zip' % (prefix, digest, python_version)
        deployment_package_filename: str = self._osutils.joinpath(project_dir,
            '.chalice', 'deployments', filename)
        return deployment_package_filename


class LayerDeploymentPackager(BaseLambdaDeploymentPackager):
    _PREFIX: str = 'python/lib/%s/site-packages'

    def func_ceu6ued4(self, project_dir: str, python_version: str) -> str:
        msg: str = 'Creating shared layer deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename: str = self.deployment_package_filename(project_dir, python_version)
        self._create_output_dir_if_needed(package_filename)
        if self._osutils.file_exists(package_filename):
            self._ui.write(
                '  Reusing existing shared layer deployment package.\n')
            return package_filename
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath: str = self._get_requirements_filename(project_dir)
            self._build_python_dependencies(
                python_version,
                requirements_filepath, 
                site_packages_dir=tmpdir
            )
            with self._osutils.open_zip(package_filename, 'w', self._osutils.ZIP_DEFLATED) as z:
                prefix: str = self._PREFIX % python_version
                self._add_py_deps(z, deps_dir=tmpdir, prefix=prefix)
                self._add_vendor_files(z, self._osutils.joinpath(
                    project_dir, self._VENDOR_DIR), prefix=prefix)
        self._check_valid_package(package_filename)
        return package_filename

    def func_hp9voaqv(self, package_filename: str) -> None:
        with self._osutils.open_zip(package_filename, 'r', self._osutils.ZIP_DEFLATED) as z:
            total_size: int = sum(f.file_size for f in z.infolist())
            if not total_size > 0:
                self._osutils.remove_file(package_filename)
                raise EmptyPackageError(package_filename)

    def func_32zdtec8(self, project_dir: str, python_version: str) -> str:
        return self._deployment_package_filename(project_dir, python_version, prefix='managed-layer-')

    def func_sra9ju6u(self, project_dir: str, python_version: str, prefix: str = '') -> str:
        requirements_filename: str = self._get_requirements_filename(project_dir)
        if not self._osutils.file_exists(requirements_filename):
            contents: bytes = b''
        else:
            contents = cast(bytes, self._osutils.get_file_contents(
                requirements_filename, binary=True))
        h: hashlib._hashlib.HASH = hashlib.md5(contents)
        vendor_dir: str = self._osutils.joinpath(project_dir, self._VENDOR_DIR)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        hash_contents: str = h.hexdigest()
        filename: str = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename: str = self._osutils.joinpath(project_dir,
            '.chalice', 'deployments', filename)
        return deployment_package_filename


class DependencyBuilder:
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
    _MANYLINUX_LEGACY_MAP: Dict[str, str] = {
        'manylinux1_x86_64': 'manylinux_2_5_x86_64',
        'manylinux2010_x86_64': 'manylinux_2_12_x86_64',
        'manylinux2014_x86_64': 'manylinux_2_17_x86_64'
    }
    _RUNTIME_GLIBC: Dict[str, Tuple[int, int]] = {
        'cp27mu': (2, 17), 
        'cp36m': (2, 17), 
        'cp37m': (2, 17),
        'cp38': (2, 26), 
        'cp310': (2, 26), 
        'cp311': (2, 26), 
        'cp312': (2, 26)
    }
    _DEFAULT_GLIBC: Tuple[int, int] = (2, 17)
    _COMPATIBLE_PACKAGE_WHITELIST: Set[str] = {'sqlalchemy', 'pyyaml', 'pyrsistent'}

    def __init__(self, osutils: OSUtils, pip_runner: Optional[PipRunner] = None) -> None:
        self._osutils: OSUtils = osutils
        if pip_runner is None:
            pip_runner = PipRunner(SubprocessPip(osutils))
        self._pip: PipRunner = pip_runner

    def func_ho4qku00(self, expected_abi: str, filename: str) -> bool:
        wheel: str = filename[:-4]
        all_compatibility_tags = self._iter_all_compatibility_tags(wheel)
        for implementation, abi, platform in all_compatibility_tags:
            if not self.func_aflj15l6(expected_abi, platform):
                continue
            if abi == 'none':
                return True
            prefix_version: str = implementation[:3]
            expected_abis: List[str] = [expected_abi]
            if prefix_version == 'cp3':
                expected_abis.append('abi3')
            if abi in expected_abis:
                return True
        return False

    def func_aflj15l6(self, expected_abi: str, platform: str) -> bool:
        if platform in self._ADDITIONAL_COMPATIBLE_PLATFORM:
            logger.debug('Found compatible platform tag: %s', platform)
            return True
        elif platform.startswith('manylinux'):
            perennial_tag: str = self._MANYLINUX_LEGACY_MAP.get(platform, platform)
            m: Optional[re.Match] = re.match('manylinux_([0-9]+)_([0-9]+)_(.*)', perennial_tag)
            if m is None:
                return False
            tag_major, tag_minor = [int(x) for x in m.groups()[:2]]
            runtime_major, runtime_minor = self._RUNTIME_GLIBC.get(expected_abi, self._DEFAULT_GLIBC)
            if (tag_major, tag_minor) <= (runtime_major, runtime_minor):
                logger.debug(
                    'Tag glibc (%s, %s) is compatible with runtime glibc (%s, %s)',
                    tag_major, tag_minor, runtime_major, runtime_minor
                )
                return True
        return False

    def func_hvruqo0g(self, wheel: str) -> Iterator[Tuple[str, str, str]]:
        implementation_tag, abi_tag, platform_tag = wheel.split('-')[-3:]
        for implementation in implementation_tag.split('.'):
            for abi in abi_tag.split('.'):
                for platform in platform_tag.split('.'):
                    yield implementation, abi, platform

    def func_w0l7bjfv(self, filename: str) -> bool:
        if not self._osutils.file_exists(filename):
            return False
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return True
        return False

    def func_se6djnh0(self, requirements_filename: str, directory: str) -> Set[Package]:
        self._pip.download_all_dependencies(requirements_filename, directory)
        deps: Set[Package] = {Package(directory, filename) for filename in self._osutils.get_directory_contents(directory)}
        logger.debug('Full dependency closure: %s', deps)
        return deps

    def func_bxjp0fio(self, abi: str, packages: List[Package], directory: str) -> None:
        logger.debug('Downloading manylinux wheels: %s', packages)
        self._pip.download_manylinux_wheels(abi, [pkg.identifier for pkg in packages], directory)

    def func_huerumnr(self, packages: List[Package], directory: str) -> None:
        logger.debug('Downloading missing sdists: %s', packages)
        self._pip.download_sdists([pkg.identifier for pkg in packages], directory)

    def func_u49qpu20(self, directory: str) -> Set[Package]:
        packages: List[Package] = [Package(directory, filename) for filename in self._osutils.get_directory_contents(directory)]
        sdists: Set[Package] = {package for package in packages if package.dist_type == 'sdist'}
        return sdists

    def func_4h9hlclf(self, sdists: Set[Package], directory: str, compile_c: bool = True) -> None:
        logger.debug('Build missing wheels from sdists (C compiling %s): %s', compile_c, sdists)
        for sdist in sdists:
            path_to_sdist: str = self._osutils.joinpath(directory, sdist.filename)
            self._pip.build_wheel(path_to_sdist, directory, compile_c)

    def func_d7rwop8n(self, abi: str, directory: str) -> Tuple[Set[Package], Set[Package]]:
        final_wheels: List[Package] = [
            Package(directory, filename) 
            for filename in self._osutils.get_directory_contents(directory) 
            if filename.endswith('.whl')
        ]
        compatible_wheels: Set[Package] = set()
        incompatible_wheels: Set[Package] = set()
        for wheel in final_wheels:
            if self._is_compatible_wheel_filename(abi, wheel.filename):
                compatible_wheels.add(wheel)
            else:
                incompatible_wheels.add(wheel)
        return compatible_wheels, incompatible_wheels

    def func_5b3t7km3(self, abi: str, deps: Set[Package]) -> Tuple[Set[Package], Set[Package], Set[Package]]:
        compatible_wheels: Set[Package] = set()
        incompatible_wheels: Set[Package] = set()
        sdists: Set[Package] = set()
        for package in deps:
            if package.dist_type == 'sdist':
                sdists.add(package)
            elif self._is_compatible_wheel_filename(abi, package.filename):
                compatible_wheels.add(package)
            else:
                incompatible_wheels.add(package)
        return sdists, compatible_wheels, incompatible_wheels

    def func_rpv34wk6(self, abi: str, directory: str, requirements_filename: str) -> Tuple[Set[Package], Set[Package]]:
        deps: Set[Package] = self._download_all_dependencies(requirements_filename, directory)
        sdists, compatible_wheels, incompatible_wheels = self._categorize_deps(abi, deps)
        logger.debug('Compatible wheels for Lambda: %s', compatible_wheels)
        logger.debug('Initial incompatible wheels for Lambda: %s', incompatible_wheels | sdists)
        missing_wheels: Set[Package] = sdists.union(incompatible_wheels)
        self._download_binary_wheels(abi, missing_wheels, directory)
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(abi, directory)
        incompatible_wheels -= compatible_wheels
        missing_sdists: Set[Package] = incompatible_wheels - sdists
        self._download_sdists(missing_sdists, directory)
        sdists = self._find_sdists(directory)
        logger.debug('compatible wheels after second download pass: %s', compatible_wheels)
        missing_wheels = sdists - compatible_wheels
        self._build_sdists(missing_wheels, directory, compile_c=True)
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(abi, directory)
        logger.debug('compatible after building wheels (C compiling): %s', compatible_wheels)
        missing_wheels = sdists - compatible_wheels
        self._build_sdists(missing_wheels, directory, compile_c=False)
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(abi, directory)
        logger.debug('compatible after building wheels (no C compiling): %s', compatible_wheels)
        compatible_wheels, incompatible_wheels = self._apply_wheel_whitelist(compatible_wheels, incompatible_wheels)
        missing_wheels = deps - compatible_wheels
        logger.debug('Final compatible: %s', compatible_wheels)
        logger.debug('Final incompatible: %s', incompatible_wheels)
        logger.debug('Final missing wheels: %s', missing_wheels)
        return compatible_wheels, missing_wheels

    def func_ic91w52k(self, compatible_wheels: Set[Package], incompatible_wheels: Set[Package]) -> Tuple[Set[Package], Set[Package]]:
        compatible_wheels = set(compatible_wheels)
        actual_incompatible_wheels: Set[Package] = set()
        for missing_package in incompatible_wheels:
            if missing_package.name in self._COMPATIBLE_PACKAGE_WHITELIST:
                compatible_wheels.add(missing_package)
            else:
                actual_incompatible_wheels.add(missing_package)
        return compatible_wheels, actual_incompatible_wheels

    def func_og7js3hz(self, wheel: Package, root: str) -> None:
        dirnames: List[str] = self._osutils.get_directory_contents(root)
        for dirname in dirnames:
            if wheel.matches_data_dir(dirname):
                data_dir: str = self._osutils.joinpath(root, dirname)
                break
        else:
            return
        unpack_dirs: Set[str] = {'purelib', 'platlib'}
        data_contents: List[str] = self._osutils.get_directory_contents(data_dir)
        for content_name in data_contents:
            if content_name in unpack_dirs:
                source: str = self._osutils.joinpath(data_dir, content_name)
                self._osutils.copytree(source, root)
                self._osutils.rmtree(source)

    def func_x0b1gawx(self, src_dir: str, dst_dir: str, wheels: Set[Package]) -> None:
        if self._osutils.directory_exists(dst_dir):
            self._osutils.rmtree(dst_dir)
        self._osutils.makedirs(dst_dir)
        for wheel in wheels:
            zipfile_path: str = self._osutils.joinpath(src_dir, wheel.filename)
            self._osutils.extract_zipfile(zipfile_path, dst_dir)
            self._install_purelib_and_platlib(wheel, dst_dir)

    def func_awcol3hx(self, abi: str, requirements_filepath: str, target_directory: str) -> None:
        if self._has_at_least_one_package(requirements_filepath):
            with self._osutils.tempdir() as tempdir:
                wheels, packages_without_wheels = self._download_dependencies(
                    abi, tempdir, requirements_filepath
                )
                self._install_wheels(tempdir, target_directory, wheels)
            if packages_without_wheels:
                raise MissingDependencyError(packages_without_wheels)


class Package:
    """A class to represent a package downloaded but not yet installed."""

    def __init__(self, directory: str, filename: str, osutils: Optional[OSUtils] = None) -> None:
        self.dist_type: str = 'wheel' if filename.endswith('.whl') else 'sdist'
        self._directory: str = directory
        self.filename: str = filename
        if osutils is None:
            osutils = OSUtils()
        self._osutils: OSUtils = osutils
        self._name: str
        self._version: str = self._calculate_name_and_version()

    @property
    def func_l8br6fuz(self) -> str:
        return self._name

    @property
    def func_qwdotjx1(self) -> str:
        return f'{self._name}-{self._version}.data'

    def func_v3y2idmg(self, dirname: str) -> bool:
        """Check if a directory name matches the data_dir of a package.

        This will normalize the directory name and perform a case-insensitive
        match against the package name's data dir.

        """
        if not self.dist_type == 'wheel' or '-' not in dirname:
            return False
        name, version = dirname.split('-')[:2]
        comparison_data_dir: str = f'{self._normalize_name(name)}-{version}'
        return self.data_dir == comparison_data_dir

    @property
    def func_4ggpyzp3(self) -> str:
        return f'{self._name}=={self._version}'

    def __str__(self) -> str:
        return f'{self.identifier}({self.dist_type})'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Package):
            return False
        return self.identifier == other.identifier

    def __hash__(self) -> int:
        return hash(self.identifier)

    def func_6ckvzd54(self) -> Tuple[str, str]:
        if self.dist_type == 'wheel':
            name, version = self.filename.split('-')[:2]
        else:
            info_fetcher = SDistMetadataFetcher(osutils=self._osutils)
            sdist_path: str = self._osutils.joinpath(self._directory, self.filename)
            name, version = info_fetcher.get_package_name_and_version(sdist_path)
        normalized_name: str = self._normalize_name(name)
        return normalized_name, version

    def func_faxg210t(self, name: str) -> str:
        return re.sub('[-_.]+', '-', name).lower()


class SDistMetadataFetcher:
    """This is the "correct" way to get name and version from an sdist."""
    _SETUPTOOLS_SHIM: str = (
        "import setuptools, tokenize;__file__=%r;f=getattr(tokenize, 'open', open)(__file__);"
        "code=f.read().replace('\\r\\n', '\\n');f.close();exec(compile(code, __file__, 'exec'))"
    )

    def __init__(self, osutils: Optional[OSUtils] = None) -> None:
        if osutils is None:
            osutils = OSUtils()
        self._osutils: OSUtils = osutils

    def func_z9jwayhu(self, filepath: str) -> Message:
        data: str = self._osutils.get_file_contents(filepath, binary=False)
        parser: FeedParser = FeedParser()
        parser.feed(data)
        return parser.close()

    def func_nvcznxci(self, package_dir: str) -> str:
        setup_py: str = self._osutils.joinpath(package_dir, 'setup.py')
        script: str = self._SETUPTOOLS_SHIM % setup_py
        cmd: List[str] = [
            sys.executable, '-c', script, '--no-user-cfg', 'egg_info',
            '--egg-base', 'egg-info'
        ]
        egg_info_dir: str = self._osutils.joinpath(package_dir, 'egg-info')
        self._osutils.makedirs(egg_info_dir)
        p: subprocess.Popen = subprocess.Popen(
            cmd, cwd=package_dir, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, stderr: Tuple[bytes, bytes] = p.communicate()
        if p.returncode != 0:
            logger.debug(
                'Non zero rc (%s) from the setup.py egg_info command: %s',
                p.returncode, stderr
            )
        info_contents: List[str] = self._osutils.get_directory_contents(egg_info_dir)
        if info_contents:
            pkg_info_path: str = self._osutils.joinpath(egg_info_dir, info_contents[0], 'PKG-INFO')
        else:
            logger.debug(
                'Using fallback location for PKG-INFO file in package directory: %s',
                package_dir
            )
            pkg_info_path: str = self._osutils.joinpath(package_dir, 'PKG-INFO')
        if not self._osutils.file_exists(pkg_info_path):
            raise UnsupportedPackageError(self._osutils.basename(package_dir))
        return pkg_info_path

    def func_3f4crera(self, sdist_path: str, unpack_dir: str) -> str:
        if sdist_path.endswith('.zip'):
            self._osutils.extract_zipfile(sdist_path, unpack_dir)
        elif sdist_path.endswith(('.tar.gz', '.tar.bz2')):
            self._osutils.extract_tarfile(sdist_path, unpack_dir)
        else:
            raise InvalidSourceDistributionNameError(sdist_path)
        contents: List[str] = self._osutils.get_directory_contents(unpack_dir)
        return self._osutils.joinpath(unpack_dir, contents[0])

    def func_o4bv84l8(self, sdist_path: str) -> Tuple[str, str]:
        with self._osutils.tempdir() as tempdir:
            package_dir: str = self._unpack_sdist_into_dir(sdist_path, tempdir)
            pkg_info_filepath: str = self._get_pkg_info_filepath(package_dir)
            metadata: Message = self._parse_pkg_info_file(pkg_info_filepath)
            name: str = metadata['Name']
            version: str = metadata['Version']
        return name, version


class SubprocessPip:
    """Wrapper around calling pip through a subprocess."""

    def __init__(self, osutils: Optional[OSUtils] = None, import_string: Optional[str] = None) -> None:
        if osutils is None:
            osutils = OSUtils()
        self._osutils: OSUtils = osutils
        if import_string is None:
            import_string = pip_import_string()
        self._import_string: str = import_string

    def func_9bbdfpp9(
        self, 
        args: List[str], 
        env_vars: Optional[EnvVars] = None, 
        shim: Optional[str] = None
    ) -> Tuple[int, bytes, bytes]:
        if env_vars is None:
            env_vars = self._osutils.environ()
        if shim is None:
            shim = ''
        python_exe: str = sys.executable
        run_pip: str = f'import sys; {self._import_string}; sys.exit(main({args}))'
        exec_string: str = f'{shim}{run_pip}'
        invoke_pip: List[str] = [python_exe, '-c', exec_string]
        p: subprocess.Popen = self._osutils.popen(
            invoke_pip, stdout=self._osutils.pipe,
            stderr=self._osutils.pipe, env=env_vars
        )
        out, err = p.communicate()
        rc: int = p.returncode
        return rc, out, err


class PipRunner:
    """Wrapper around pip calls used by chalice."""
    _LINK_IS_DIR_PATTERN: str = (
        'Processing (.+?)\n  Link is a directory, ignoring download_dir'
    )

    def __init__(self, pip: SubprocessPip, osutils: Optional[OSUtils] = None) -> None:
        if osutils is None:
            osutils = OSUtils()
        self._wrapped_pip: SubprocessPip = pip
        self._osutils: OSUtils = osutils

    def func_ggprypbp(
        self, 
        command: str, 
        args: List[str], 
        env_vars: Optional[EnvVars] = None, 
        shim: Optional[str] = None
    ) -> Tuple[int, bytes, bytes]:
        """Execute a pip command with the given arguments."""
        main_args: List[str] = [command] + args
        logger.debug('calling pip %s', ' '.join(main_args))
        rc, out, err = self._wrapped_pip.func_9bbdfpp9(main_args, env_vars=env_vars, shim=shim)
        return rc, out, err

    def func_kq9muiw8(
        self, 
        wheel: str, 
        directory: str, 
        compile_c: bool = True
    ) -> None:
        """Build an sdist into a wheel file."""
        arguments: List[str] = ['--no-deps', '--wheel-dir', directory, wheel]
        env_vars: EnvVars = self._osutils.environ()
        shim: str = ''
        if not compile_c:
            env_vars.update(pip_no_compile_c_env_vars)
            shim = pip_no_compile_c_shim
        self._execute('wheel', arguments, env_vars=env_vars, shim=shim)

    def func_5qwdsy8c(self, requirements_filename: str, directory: str) -> None:
        """Download all dependencies as sdist or wheel."""
        arguments: List[str] = ['-r', requirements_filename, '--dest', directory]
        rc, out, err = self._execute('download', arguments)
        if rc != 0:
            if err is None:
                err = b'Unknown error'
            error: str = err.decode()
            match: Optional[re.Match] = re.search(
                'Could not find a version that satisfies the requirement ([^\\s]+)',
                error
            )
            if match:
                package_name: str = match.group(1)
                raise NoSuchPackageError(str(package_name))
            raise PackageDownloadError(error)
        stdout: str = out.decode()
        matches = re.finditer(self._LINK_IS_DIR_PATTERN, stdout)
        for match in matches:
            wheel_package_path: str = str(match.group(1))
            self.build_wheel(wheel_package_path, directory)

    def func_anlf57y6(self, abi: str, packages: List[str], directory: str) -> None:
        """Download wheel files for manylinux for all the given packages."""
        for package in packages:
            arguments: List[str] = [
                '--only-binary=:all:', '--no-deps', '--platform',
                'manylinux2014_x86_64', '--implementation', 'cp', '--abi',
                abi, '--dest', directory, package
            ]
            self._execute('download', arguments)

    def func_495rqpva(self, packages: List[str], directory: str) -> None:
        for package in packages:
            arguments: List[str] = [
                '--no-binary=:all:', '--no-deps', '--dest',
                directory, package
            ]
            self._execute('download', arguments)

    def _execute(
        self, 
        command: str, 
        args: List[str], 
        env_vars: Optional[EnvVars] = None, 
        shim: Optional[str] = None
    ) -> Tuple[int, bytes, bytes]:
        return self.func_ggprypbp(command, args, env_vars, shim)
