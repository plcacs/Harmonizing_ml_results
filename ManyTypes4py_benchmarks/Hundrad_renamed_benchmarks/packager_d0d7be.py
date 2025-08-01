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
from typing import Iterator
from typing import Dict, MutableMapping, cast
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

    def __init__(self, missing):
        self.missing = missing


class NoSuchPackageError(Exception):
    """Raised when a package name or version could not be found."""

    def __init__(self, package_name):
        super(NoSuchPackageError, self).__init__(
            'Could not satisfy the requirement: %s' % package_name)


class PackageDownloadError(Exception):
    """Generic networking error during a package download."""


class EmptyPackageError(Exception):
    """A deployment package cannot be an empty zip file."""


class UnsupportedPackageError(Exception):
    """Unable to parse package metadata."""

    def __init__(self, package_name):
        super(UnsupportedPackageError, self).__init__(
            'Unable to retrieve name/version for package: %s' % package_name)


class BaseLambdaDeploymentPackager(object):
    _CHALICE_LIB_DIR = 'chalicelib'
    _VENDOR_DIR = 'vendor'
    _RUNTIME_TO_ABI = {'python3.8': 'cp38', 'python3.9': 'cp39',
        'python3.10': 'cp310', 'python3.11': 'cp311', 'python3.12': 'cp312'}

    def __init__(self, osutils, dependency_builder, ui):
        self._osutils = osutils
        self._dependency_builder = dependency_builder
        self._ui = ui

    def func_61btvbom(self, project_dir, python_version):
        raise NotImplementedError('create_deployment_package')

    def func_im2sqyl5(self, project_dir):
        return self._osutils.joinpath(project_dir, 'requirements.txt')

    def func_6hq286zf(self, zipped, dirname, prefix=''):
        if not self._osutils.directory_exists(dirname):
            return
        prefix_len = len(dirname) + 1
        for root, _, filenames in self._osutils.walk(dirname, followlinks=True
            ):
            for filename in filenames:
                full_path = self._osutils.joinpath(root, filename)
                zip_path = full_path[prefix_len:]
                if prefix:
                    zip_path = self._osutils.joinpath(prefix, zip_path)
                zipped.write(full_path, zip_path)

    def func_wtyyp0ez(self, project_dir, python_version):
        return self._deployment_package_filename(project_dir, python_version)

    def func_ybgjq7hz(self, project_dir, python_version, prefix=''):
        requirements_filename = self._get_requirements_filename(project_dir)
        hash_contents = self._hash_project_dir(requirements_filename, self.
            _osutils.joinpath(project_dir, self._VENDOR_DIR), project_dir)
        filename = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename = self._osutils.joinpath(project_dir,
            '.chalice', 'deployments', filename)
        return deployment_package_filename

    def func_3xpuozob(self, zip_fileobj, deps_dir, prefix=''):
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

    def func_3d590ubg(self, zip_fileobj, project_dir):
        for full_path, zip_path in self._iter_app_filenames(project_dir):
            zip_fileobj.write(full_path, zip_path)

    def func_dyzesrn1(self, project_dir):
        chalice_router = inspect.getfile(app)
        if chalice_router.endswith('.pyc'):
            chalice_router = chalice_router[:-1]
        yield chalice_router, 'chalice/app.py'
        chalice_init = inspect.getfile(chalice)
        if chalice_init.endswith('.pyc'):
            chalice_init = chalice_init[:-1]
        yield chalice_init, 'chalice/__init__.py'
        yield self._osutils.joinpath(project_dir, 'app.py'), 'app.py'
        yield from self._iter_chalice_lib_if_needed(project_dir)

    def func_4bmi1ppy(self, requirements_filename, vendor_dir, project_dir):
        if not self._osutils.file_exists(requirements_filename):
            contents = b''
        else:
            contents = cast(bytes, self._osutils.get_file_contents(
                requirements_filename, binary=True))
        h = hashlib.md5(contents)
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        return h.hexdigest()

    def func_ewf8b28a(self, vendor_dir, md5):
        for rootdir, _, filenames in self._osutils.walk(vendor_dir,
            followlinks=True):
            for filename in filenames:
                fullpath = self._osutils.joinpath(rootdir, filename)
                with self._osutils.open(fullpath, 'rb') as f:
                    reader = functools.partial(f.read, 1024 * 1024)
                    for chunk in iter(reader, b''):
                        md5.update(chunk)

    def func_6vmzc74s(self, deployment_package_filename, project_dir):
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
            with self._osutils.open_zip(tmpzip, 'w', self._osutils.ZIP_DEFLATED
                ) as outzip:
                for el in inzip.infolist():
                    if self._needs_latest_version(el.filename):
                        continue
                    contents = inzip.read(el.filename)
                    outzip.writestr(el, contents)
                self._add_app_files(outzip, project_dir)
        self._osutils.move(tmpzip, deployment_package_filename)

    def func_9h6bkrpn(self, filename):
        return filename == 'app.py' or filename.startswith(('chalicelib/',
            'chalice/'))

    def func_7iqpvcy6(self, project_dir):
        libdir = self._osutils.joinpath(project_dir, self._CHALICE_LIB_DIR)
        if self._osutils.directory_exists(libdir):
            for rootdir, _, filenames in self._osutils.walk(libdir):
                for filename in filenames:
                    fullpath = self._osutils.joinpath(rootdir, filename)
                    zip_path = self._osutils.joinpath(self._CHALICE_LIB_DIR,
                        fullpath[len(libdir) + 1:])
                    yield fullpath, zip_path

    def func_4jqig29b(self, package_filename):
        dirname = self._osutils.dirname(self._osutils.abspath(package_filename)
            )
        if not self._osutils.directory_exists(dirname):
            self._osutils.makedirs(dirname)

    def func_04nbfpav(self, python_version, requirements_filepath,
        site_packages_dir):
        try:
            abi = self._RUNTIME_TO_ABI[python_version]
            self._dependency_builder.build_site_packages(abi,
                requirements_filepath, site_packages_dir)
        except MissingDependencyError as e:
            missing_packages = '\n'.join([p.identifier for p in e.missing])
            self._ui.write(MISSING_DEPENDENCIES_TEMPLATE % missing_packages)


class LambdaDeploymentPackager(BaseLambdaDeploymentPackager):

    def func_61btvbom(self, project_dir, python_version):
        msg = 'Creating deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename = self.deployment_package_filename(project_dir,
            python_version)
        if self._osutils.file_exists(package_filename):
            self._ui.write('Reusing existing deployment package.\n')
            return package_filename
        self._create_output_dir_if_needed(package_filename)
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath = self._get_requirements_filename(project_dir
                )
            self._build_python_dependencies(python_version,
                requirements_filepath, site_packages_dir=tmpdir)
            with self._osutils.open_zip(package_filename, 'w', self.
                _osutils.ZIP_DEFLATED) as z:
                self._add_py_deps(z, deps_dir=tmpdir)
                self._add_app_files(z, project_dir)
                self._add_vendor_files(z, self._osutils.joinpath(
                    project_dir, self._VENDOR_DIR))
        return package_filename


class AppOnlyDeploymentPackager(BaseLambdaDeploymentPackager):

    def func_61btvbom(self, project_dir, python_version):
        msg = 'Creating app deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename = self.deployment_package_filename(project_dir,
            python_version)
        if self._osutils.file_exists(package_filename):
            self._ui.write('  Reusing existing app deployment package.\n')
            return package_filename
        self._create_output_dir_if_needed(package_filename)
        with self._osutils.open_zip(package_filename, 'w', self._osutils.
            ZIP_DEFLATED) as z:
            self._add_app_files(z, project_dir)
        return package_filename

    def func_wtyyp0ez(self, project_dir, python_version):
        return self._deployment_package_filename(project_dir,
            python_version, prefix='appcode-')

    def func_ybgjq7hz(self, project_dir, python_version, prefix=''):
        h = hashlib.md5(b'')
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        digest = h.hexdigest()
        filename = '%s%s-%s.zip' % (prefix, digest, python_version)
        deployment_package_filename = self._osutils.joinpath(project_dir,
            '.chalice', 'deployments', filename)
        return deployment_package_filename


class LayerDeploymentPackager(BaseLambdaDeploymentPackager):
    _PREFIX = 'python/lib/%s/site-packages'

    def func_61btvbom(self, project_dir, python_version):
        msg = 'Creating shared layer deployment package.'
        self._ui.write('%s\n' % msg)
        logger.debug(msg)
        package_filename = self.deployment_package_filename(project_dir,
            python_version)
        self._create_output_dir_if_needed(package_filename)
        if self._osutils.file_exists(package_filename):
            self._ui.write(
                '  Reusing existing shared layer deployment package.\n')
            return package_filename
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath = self._get_requirements_filename(project_dir
                )
            self._build_python_dependencies(python_version,
                requirements_filepath, site_packages_dir=tmpdir)
            with self._osutils.open_zip(package_filename, 'w', self.
                _osutils.ZIP_DEFLATED) as z:
                prefix = self._PREFIX % python_version
                self._add_py_deps(z, deps_dir=tmpdir, prefix=prefix)
                self._add_vendor_files(z, self._osutils.joinpath(
                    project_dir, self._VENDOR_DIR), prefix=prefix)
        self._check_valid_package(package_filename)
        return package_filename

    def func_c14uw6xv(self, package_filename):
        with self._osutils.open_zip(package_filename, 'r', self._osutils.
            ZIP_DEFLATED) as z:
            total_size = sum(f.file_size for f in z.infolist())
            if not total_size > 0:
                self._osutils.remove_file(package_filename)
                raise EmptyPackageError(package_filename)

    def func_wtyyp0ez(self, project_dir, python_version):
        return self._deployment_package_filename(project_dir,
            python_version, prefix='managed-layer-')

    def func_ybgjq7hz(self, project_dir, python_version, prefix=''):
        requirements_filename = self._get_requirements_filename(project_dir)
        if not self._osutils.file_exists(requirements_filename):
            contents = b''
        else:
            contents = cast(bytes, self._osutils.get_file_contents(
                requirements_filename, binary=True))
        h = hashlib.md5(contents)
        vendor_dir = self._osutils.joinpath(project_dir, self._VENDOR_DIR)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        hash_contents = h.hexdigest()
        filename = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename = self._osutils.joinpath(project_dir,
            '.chalice', 'deployments', filename)
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
    _ADDITIONAL_COMPATIBLE_PLATFORM = {'any', 'linux_x86_64'}
    _MANYLINUX_LEGACY_MAP = {'manylinux1_x86_64': 'manylinux_2_5_x86_64',
        'manylinux2010_x86_64': 'manylinux_2_12_x86_64',
        'manylinux2014_x86_64': 'manylinux_2_17_x86_64'}
    _RUNTIME_GLIBC = {'cp27mu': (2, 17), 'cp36m': (2, 17), 'cp37m': (2, 17),
        'cp38': (2, 26), 'cp310': (2, 26), 'cp311': (2, 26), 'cp312': (2, 26)}
    _DEFAULT_GLIBC = 2, 17
    _COMPATIBLE_PACKAGE_WHITELIST = {'sqlalchemy', 'pyyaml', 'pyrsistent'}

    def __init__(self, osutils, pip_runner=None):
        self._osutils = osutils
        if pip_runner is None:
            pip_runner = PipRunner(SubprocessPip(osutils))
        self._pip = pip_runner

    def func_9ztp3itn(self, expected_abi, filename):
        wheel = filename[:-4]
        all_compatibility_tags = self._iter_all_compatibility_tags(wheel)
        for implementation, abi, platform in all_compatibility_tags:
            if not self._is_compatible_platform_tag(expected_abi, platform):
                continue
            if abi == 'none':
                return True
            prefix_version = implementation[:3]
            expected_abis = [expected_abi]
            if prefix_version == 'cp3':
                expected_abis.append('abi3')
            if abi in expected_abis:
                return True
        return False

    def func_1kqmez9q(self, expected_abi, platform):
        if platform in self._ADDITIONAL_COMPATIBLE_PLATFORM:
            logger.debug('Found compatible platform tag: %s', platform)
            return True
        elif platform.startswith('manylinux'):
            perennial_tag = self._MANYLINUX_LEGACY_MAP.get(platform, platform)
            m = re.match('manylinux_([0-9]+)_([0-9]+)_(.*)', perennial_tag)
            if m is None:
                return False
            tag_major, tag_minor = [int(x) for x in m.groups()[:2]]
            runtime_major, runtime_minor = self._RUNTIME_GLIBC.get(expected_abi
                , self._DEFAULT_GLIBC)
            if (tag_major, tag_minor) <= (runtime_major, runtime_minor):
                logger.debug(
                    'Tag glibc (%s, %s) is compatible with runtime glibc (%s, %s)'
                    , tag_major, tag_minor, runtime_major, runtime_minor)
                return True
        return False

    def func_091p1wm1(self, wheel):
        implementation_tag, abi_tag, platform_tag = wheel.split('-')[-3:]
        for implementation in implementation_tag.split('.'):
            for abi in abi_tag.split('.'):
                for platform in platform_tag.split('.'):
                    yield implementation, abi, platform

    def func_24wxx5kb(self, filename):
        if not self._osutils.file_exists(filename):
            return False
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return True
        return False

    def func_um9i11xy(self, requirements_filename, directory):
        self._pip.download_all_dependencies(requirements_filename, directory)
        deps = {Package(directory, filename) for filename in self._osutils.
            get_directory_contents(directory)}
        logger.debug('Full dependency closure: %s', deps)
        return deps

    def func_rg8pn8em(self, abi, packages, directory):
        logger.debug('Downloading manylinux wheels: %s', packages)
        self._pip.download_manylinux_wheels(abi, [pkg.identifier for pkg in
            packages], directory)

    def func_1c08dt2q(self, packages, directory):
        logger.debug('Downloading missing sdists: %s', packages)
        self._pip.download_sdists([pkg.identifier for pkg in packages],
            directory)

    def func_uvykgb3w(self, directory):
        packages = [Package(directory, filename) for filename in self.
            _osutils.get_directory_contents(directory)]
        sdists = {package for package in packages if package.dist_type ==
            'sdist'}
        return sdists

    def func_bkkq40a2(self, sdists, directory, compile_c=True):
        logger.debug('Build missing wheels from sdists (C compiling %s): %s',
            compile_c, sdists)
        for sdist in sdists:
            path_to_sdist = self._osutils.joinpath(directory, sdist.filename)
            self._pip.build_wheel(path_to_sdist, directory, compile_c)

    def func_o4zh9jfz(self, abi, directory):
        final_wheels = [Package(directory, filename) for filename in self.
            _osutils.get_directory_contents(directory) if filename.endswith
            ('.whl')]
        compatible_wheels, incompatible_wheels = set(), set()
        for wheel in final_wheels:
            if self._is_compatible_wheel_filename(abi, wheel.filename):
                compatible_wheels.add(wheel)
            else:
                incompatible_wheels.add(wheel)
        return compatible_wheels, incompatible_wheels

    def func_gf10qshk(self, abi, deps):
        compatible_wheels = set()
        incompatible_wheels = set()
        sdists = set()
        for package in deps:
            if package.dist_type == 'sdist':
                sdists.add(package)
            elif self._is_compatible_wheel_filename(abi, package.filename):
                compatible_wheels.add(package)
            else:
                incompatible_wheels.add(package)
        return sdists, compatible_wheels, incompatible_wheels

    def func_dsswmsc5(self, abi, directory, requirements_filename):
        deps = self._download_all_dependencies(requirements_filename, directory
            )
        sdists, compatible_wheels, incompatible_wheels = self._categorize_deps(
            abi, deps)
        logger.debug('Compatible wheels for Lambda: %s', compatible_wheels)
        logger.debug('Initial incompatible wheels for Lambda: %s', 
            incompatible_wheels | sdists)
        missing_wheels = sdists.union(incompatible_wheels)
        self._download_binary_wheels(abi, missing_wheels, directory)
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(
            abi, directory)
        incompatible_wheels -= compatible_wheels
        missing_sdists = incompatible_wheels - sdists
        self._download_sdists(missing_sdists, directory)
        sdists = self._find_sdists(directory)
        logger.debug('compatible wheels after second download pass: %s',
            compatible_wheels)
        missing_wheels = sdists - compatible_wheels
        self._build_sdists(missing_wheels, directory, compile_c=True)
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(
            abi, directory)
        logger.debug('compatible after building wheels (C compiling): %s',
            compatible_wheels)
        missing_wheels = sdists - compatible_wheels
        self._build_sdists(missing_wheels, directory, compile_c=False)
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(
            abi, directory)
        logger.debug('compatible after building wheels (no C compiling): %s',
            compatible_wheels)
        compatible_wheels, incompatible_wheels = self._apply_wheel_whitelist(
            compatible_wheels, incompatible_wheels)
        missing_wheels = deps - compatible_wheels
        logger.debug('Final compatible: %s', compatible_wheels)
        logger.debug('Final incompatible: %s', incompatible_wheels)
        logger.debug('Final missing wheels: %s', missing_wheels)
        return compatible_wheels, missing_wheels

    def func_8wc3k89n(self, compatible_wheels, incompatible_wheels):
        compatible_wheels = set(compatible_wheels)
        actual_incompatible_wheels = set()
        for missing_package in incompatible_wheels:
            if missing_package.name in self._COMPATIBLE_PACKAGE_WHITELIST:
                compatible_wheels.add(missing_package)
            else:
                actual_incompatible_wheels.add(missing_package)
        return compatible_wheels, actual_incompatible_wheels

    def func_gxgplu06(self, wheel, root):
        dirnames = self._osutils.get_directory_contents(root)
        for dirname in dirnames:
            if wheel.matches_data_dir(dirname):
                data_dir = self._osutils.joinpath(root, dirname)
                break
        else:
            return
        unpack_dirs = {'purelib', 'platlib'}
        data_contents = self._osutils.get_directory_contents(data_dir)
        for content_name in data_contents:
            if content_name in unpack_dirs:
                source = self._osutils.joinpath(data_dir, content_name)
                self._osutils.copytree(source, root)
                self._osutils.rmtree(source)

    def func_6pvki0sd(self, src_dir, dst_dir, wheels):
        if self._osutils.directory_exists(dst_dir):
            self._osutils.rmtree(dst_dir)
        self._osutils.makedirs(dst_dir)
        for wheel in wheels:
            zipfile_path = self._osutils.joinpath(src_dir, wheel.filename)
            self._osutils.extract_zipfile(zipfile_path, dst_dir)
            self._install_purelib_and_platlib(wheel, dst_dir)

    def func_kaf9qqi6(self, abi, requirements_filepath, target_directory):
        if self._has_at_least_one_package(requirements_filepath):
            with self._osutils.tempdir() as tempdir:
                wheels, packages_without_wheels = self._download_dependencies(
                    abi, tempdir, requirements_filepath)
                self._install_wheels(tempdir, target_directory, wheels)
            if packages_without_wheels:
                raise MissingDependencyError(packages_without_wheels)


class Package(object):
    """A class to represent a package downloaded but not yet installed."""

    def __init__(self, directory, filename, osutils=None):
        self.dist_type = 'wheel' if filename.endswith('.whl') else 'sdist'
        self._directory = directory
        self.filename = filename
        if osutils is None:
            osutils = OSUtils()
        self._osutils = osutils
        self._name, self._version = self._calculate_name_and_version()

    @property
    def func_wytg3ki2(self):
        return self._name

    @property
    def func_j4n35d30(self):
        return '%s-%s.data' % (self._name, self._version)

    def func_9mmrpbhd(self, dirname):
        """Check if a directory name matches the data_dir of a package.

        This will normalize the directory name and perform a case-insensitive
        match against the package name's data dir.

        """
        if not self.dist_type == 'wheel' or '-' not in dirname:
            return False
        name, version = dirname.split('-')[:2]
        comparison_data_dir = '%s-%s' % (self._normalize_name(name), version)
        return self.data_dir == comparison_data_dir

    @property
    def func_9yhzlvsz(self):
        return '%s==%s' % (self._name, self._version)

    def __str__(self):
        return '%s(%s)' % (self.identifier, self.dist_type)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Package):
            return False
        return self.identifier == other.identifier

    def __hash__(self):
        return hash(self.identifier)

    def func_lzpbe3gw(self):
        if self.dist_type == 'wheel':
            name, version = self.filename.split('-')[:2]
        else:
            info_fetcher = SDistMetadataFetcher(osutils=self._osutils)
            sdist_path = self._osutils.joinpath(self._directory, self.filename)
            name, version = info_fetcher.get_package_name_and_version(
                sdist_path)
        normalized_name = self._normalize_name(name)
        return normalized_name, version

    def func_cfnxsben(self, name):
        return re.sub('[-_.]+', '-', name).lower()


class SDistMetadataFetcher(object):
    """This is the "correct" way to get name and version from an sdist."""
    _SETUPTOOLS_SHIM = (
        "import setuptools, tokenize;__file__=%r;f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\\r\\n', '\\n');f.close();exec(compile(code, __file__, 'exec'))"
        )

    def __init__(self, osutils=None):
        if osutils is None:
            osutils = OSUtils()
        self._osutils = osutils

    def func_xioue8vq(self, filepath):
        data = self._osutils.get_file_contents(filepath, binary=False)
        parser = FeedParser()
        parser.feed(data)
        return parser.close()

    def func_f1rphxy7(self, package_dir):
        setup_py = self._osutils.joinpath(package_dir, 'setup.py')
        script = self._SETUPTOOLS_SHIM % setup_py
        cmd = [sys.executable, '-c', script, '--no-user-cfg', 'egg_info',
            '--egg-base', 'egg-info']
        egg_info_dir = self._osutils.joinpath(package_dir, 'egg-info')
        self._osutils.makedirs(egg_info_dir)
        p = subprocess.Popen(cmd, cwd=package_dir, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        _, stderr = p.communicate()
        if p.returncode != 0:
            logger.debug(
                'Non zero rc (%s) from the setup.py egg_info command: %s',
                p.returncode, stderr)
        info_contents = self._osutils.get_directory_contents(egg_info_dir)
        if info_contents:
            pkg_info_path = self._osutils.joinpath(egg_info_dir,
                info_contents[0], 'PKG-INFO')
        else:
            logger.debug(
                'Using fallback location for PKG-INFO file in package directory: %s'
                , package_dir)
            pkg_info_path = self._osutils.joinpath(package_dir, 'PKG-INFO')
        if not self._osutils.file_exists(pkg_info_path):
            raise UnsupportedPackageError(self._osutils.basename(package_dir))
        return pkg_info_path

    def func_8b4jswgy(self, sdist_path, unpack_dir):
        if sdist_path.endswith('.zip'):
            self._osutils.extract_zipfile(sdist_path, unpack_dir)
        elif sdist_path.endswith(('.tar.gz', '.tar.bz2')):
            self._osutils.extract_tarfile(sdist_path, unpack_dir)
        else:
            raise InvalidSourceDistributionNameError(sdist_path)
        contents = self._osutils.get_directory_contents(unpack_dir)
        return self._osutils.joinpath(unpack_dir, contents[0])

    def func_fuueicot(self, sdist_path):
        with self._osutils.tempdir() as tempdir:
            package_dir = self._unpack_sdist_into_dir(sdist_path, tempdir)
            pkg_info_filepath = self._get_pkg_info_filepath(package_dir)
            metadata = self._parse_pkg_info_file(pkg_info_filepath)
            name = metadata['Name']
            version = metadata['Version']
        return name, version


class SubprocessPip(object):
    """Wrapper around calling pip through a subprocess."""

    def __init__(self, osutils=None, import_string=None):
        if osutils is None:
            osutils = OSUtils()
        self._osutils = osutils
        if import_string is None:
            import_string = pip_import_string()
        self._import_string = import_string

    def func_e92x3xi4(self, args, env_vars=None, shim=None):
        if env_vars is None:
            env_vars = self._osutils.environ()
        if shim is None:
            shim = ''
        python_exe = sys.executable
        run_pip = 'import sys; %s; sys.exit(main(%s))' % (self.
            _import_string, args)
        exec_string = '%s%s' % (shim, run_pip)
        invoke_pip = [python_exe, '-c', exec_string]
        p = self._osutils.popen(invoke_pip, stdout=self._osutils.pipe,
            stderr=self._osutils.pipe, env=env_vars)
        out, err = p.communicate()
        rc = p.returncode
        return rc, out, err


class PipRunner(object):
    """Wrapper around pip calls used by chalice."""
    _LINK_IS_DIR_PATTERN = (
        'Processing (.+?)\n  Link is a directory, ignoring download_dir')

    def __init__(self, pip, osutils=None):
        if osutils is None:
            osutils = OSUtils()
        self._wrapped_pip = pip
        self._osutils = osutils

    def func_3rq8ipju(self, command, args, env_vars=None, shim=None):
        """Execute a pip command with the given arguments."""
        main_args = [command] + args
        logger.debug('calling pip %s', ' '.join(main_args))
        rc, out, err = self._wrapped_pip.main(main_args, env_vars=env_vars,
            shim=shim)
        return rc, out, err

    def func_5f1u72n7(self, wheel, directory, compile_c=True):
        """Build an sdist into a wheel file."""
        arguments = ['--no-deps', '--wheel-dir', directory, wheel]
        env_vars = self._osutils.environ()
        shim = ''
        if not compile_c:
            env_vars.update(pip_no_compile_c_env_vars)
            shim = pip_no_compile_c_shim
        self._execute('wheel', arguments, env_vars=env_vars, shim=shim)

    def func_4j515eug(self, requirements_filename, directory):
        """Download all dependencies as sdist or wheel."""
        arguments = ['-r', requirements_filename, '--dest', directory]
        rc, out, err = self._execute('download', arguments)
        if rc != 0:
            if err is None:
                err = b'Unknown error'
            error = err.decode()
            match = re.search(
                'Could not find a version that satisfies the requirement ([^\\s]+)'
                , error)
            if match:
                package_name = match.group(1)
                raise NoSuchPackageError(str(package_name))
            raise PackageDownloadError(error)
        stdout = out.decode()
        matches = re.finditer(self._LINK_IS_DIR_PATTERN, stdout)
        for match in matches:
            wheel_package_path = str(match.group(1))
            self.build_wheel(wheel_package_path, directory)

    def func_bv8j20dh(self, abi, packages, directory):
        """Download wheel files for manylinux for all the given packages."""
        for package in packages:
            arguments = ['--only-binary=:all:', '--no-deps', '--platform',
                'manylinux2014_x86_64', '--implementation', 'cp', '--abi',
                abi, '--dest', directory, package]
            self._execute('download', arguments)

    def func_52c5zbfs(self, packages, directory):
        for package in packages:
            arguments = ['--no-binary=:all:', '--no-deps', '--dest',
                directory, package]
            self._execute('download', arguments)
