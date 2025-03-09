# pylint: disable=too-many-lines
from __future__ import annotations
import sys
import hashlib
import inspect
import re
import subprocess
import logging
import functools
from email.parser import FeedParser
from email.message import Message  # noqa
from zipfile import ZipFile  # noqa

from typing import Any, Set, List, Optional, Tuple, Iterable, Callable  # noqa
from typing import Iterator  # noqa
from typing import Dict, MutableMapping, cast  # noqa
from chalice.compat import pip_import_string
from chalice.compat import pip_no_compile_c_env_vars
from chalice.compat import pip_no_compile_c_shim
from chalice.utils import OSUtils
from chalice.utils import UI  # noqa
from chalice.constants import MISSING_DEPENDENCIES_TEMPLATE

import chalice
from chalice import app


StrMap = Dict[str, Any]
OptStrMap = Optional[StrMap]
EnvVars = MutableMapping[str, str]
OptStr = Optional[str]
OptBytes = Optional[bytes]

logger: logging.Logger = logging.getLogger(__name__)


class InvalidSourceDistributionNameError(Exception):
    pass


class MissingDependencyError(Exception):
    """Raised when some dependencies could not be packaged for any reason."""

    def __init__(self, missing: Set[Package]) -> None:
        self.missing: Set[Package] = missing


class NoSuchPackageError(Exception):
    """Raised when a package name or version could not be found."""

    def __init__(self, package_name: str) -> None:
        super(NoSuchPackageError, self).__init__(
            'Could not satisfy the requirement: %s' % package_name
        )


class PackageDownloadError(Exception):
    """Generic networking error during a package download."""


class EmptyPackageError(Exception):
    """A deployment package cannot be an empty zip file."""


class UnsupportedPackageError(Exception):
    """Unable to parse package metadata."""

    def __init__(self, package_name: str) -> None:
        super(UnsupportedPackageError, self).__init__(
            'Unable to retrieve name/version for package: %s' % package_name
        )


class BaseLambdaDeploymentPackager(object):
    _CHALICE_LIB_DIR: str = 'chalicelib'
    _VENDOR_DIR: str = 'vendor'

    _RUNTIME_TO_ABI: Dict[str, str] = {
        'python3.8': 'cp38',
        'python3.9': 'cp39',
        'python3.10': 'cp310',
        'python3.11': 'cp311',
        'python3.12': 'cp312',
    }

    def __init__(
        self, osutils: OSUtils, dependency_builder: DependencyBuilder, ui: UI
    ) -> None:
        self._osutils: OSUtils = osutils
        self._dependency_builder: DependencyBuilder = dependency_builder
        self._ui: UI = ui

    def create_deployment_package(
        self, project_dir: str, python_version: str
    ) -> str:
        raise NotImplementedError("create_deployment_package")

    def _get_requirements_filename(self, project_dir: str) -> str:
        # Gets the path to a requirements.txt file out of a project dir path
        return self._osutils.joinpath(project_dir, 'requirements.txt')

    def _add_vendor_files(
        self, zipped: ZipFile, dirname: str, prefix: str = ''
    ) -> None:
        if not self._osutils.directory_exists(dirname):
            return
        prefix_len: int = len(dirname) + 1
        for root, _, filenames in self._osutils.walk(
            dirname, followlinks=True
        ):
            for filename in filenames:
                full_path: str = self._osutils.joinpath(root, filename)
                zip_path: str = full_path[prefix_len:]
                if prefix:
                    zip_path = self._osutils.joinpath(prefix, zip_path)
                zipped.write(full_path, zip_path)

    def deployment_package_filename(
        self, project_dir: str, python_version: str
    ) -> str:
        # Computes the name of the deployment package zipfile
        # based on a hash of the requirements file.
        # This is done so that we only "pip install -r requirements.txt"
        # when we know there's new dependencies we need to install.
        # The python version these depedencies were downloaded for is appended
        # to the end of the filename since the the dependencies may not change
        # but if the python version changes then the dependencies need to be
        # re-downloaded since they will not be compatible.
        return self._deployment_package_filename(project_dir, python_version)

    def _deployment_package_filename(
        self, project_dir: str, python_version: str, prefix: str = ''
    ) -> str:
        requirements_filename: str = self._get_requirements_filename(project_dir)
        hash_contents: str = self._hash_project_dir(
            requirements_filename,
            self._osutils.joinpath(project_dir, self._VENDOR_DIR),
            project_dir,
        )
        filename: str = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename: str = self._osutils.joinpath(
            project_dir, '.chalice', 'deployments', filename
        )
        return deployment_package_filename

    def _add_py_deps(
        self, zip_fileobj: ZipFile, deps_dir: str, prefix: str = ''
    ) -> None:
        prefix_len: int = len(deps_dir) + 1
        for root, dirnames, filenames in self._osutils.walk(deps_dir):
            if root == deps_dir and 'chalice' in dirnames:
                # Don't include any chalice deps.  We cherry pick
                # what we want to include in _add_app_files.
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

    def _iter_app_filenames(
        self, project_dir: str
    ) -> Iterator[Tuple[str, str]]:
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

    def _hash_project_dir(
        self, requirements_filename: str, vendor_dir: str, project_dir: str
    ) -> str:
        if not self._osutils.file_exists(requirements_filename):
            contents: bytes = b''
        else:
            contents = cast(
                bytes,
                self._osutils.get_file_contents(
                    requirements_filename, binary=True
                ),
            )
        h: Any = hashlib.md5(contents)
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader: Callable[[int], bytes] = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        return h.hexdigest()

    def _hash_vendor_dir(self, vendor_dir: str, md5: Any) -> None:
        for rootdir, _, filenames in self._osutils.walk(
            vendor_dir, followlinks=True
        ):
            for filename in filenames:
                fullpath: str = self._osutils.joinpath(rootdir, filename)
                with self._osutils.open(fullpath, 'rb') as f:
                    # Not actually an issue, but pylint will complain
                    # about the f var being used in the lambda function
                    # is being used in a loop.  This is ok because
                    # we're immediately using the lambda function.
                    # Also binding it as a default argument fixes
                    # pylint, but mypy will complain that it can't
                    # infer the types.  So the compromise here is to
                    # just write it the idiomatic way and have pylint
                    # ignore this warning.
                    reader: Callable[[int], bytes] = functools.partial(f.read, 1024 * 1024)
                    for chunk in iter(reader, b''):
                        md5.update(chunk)

    def inject_latest_app(
        self, deployment_package_filename: str, project_dir: str
    ) -> None:
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
        # Use the premade zip file and replace the app.py file
        # with the latest version.  Python's zipfile does not have
        # a way to do this efficiently so we need to create a new
        # zip file that has all the same stuff except for the new
        # app file.
        self._ui.write("Regen deployment package.\n")
        tmpzip: str = deployment_package_filename + '.tmp.zip'

        with self._osutils.open_zip(deployment_package_filename, 'r') as inzip:
            with self._osutils.open_zip(
                tmpzip, 'w', self._osutils.ZIP_DEFLATED
            ) as outzip:
                for el in inzip.infolist():
                    if self._needs_latest_version(el.filename):
                        continue
                    contents: bytes = inzip.read(el.filename)
                    outzip.writestr(el, contents)
                # Then at the end, add back the app.py, chalicelib,
                # and runtime files.
                self._add_app_files(outzip, project_dir)
        self._osutils.move(tmpzip, deployment_package_filename)

    def _needs_latest_version(self, filename: str) -> bool:
        return filename == 'app.py' or filename.startswith(
            ('chalicelib/', 'chalice/')
        )

    def _iter_chalice_lib_if_needed(
        self, project_dir: str
    ) -> Iterator[Tuple[str, str]]:
        libdir: str = self._osutils.joinpath(project_dir, self._CHALICE_LIB_DIR)
        if self._osutils.directory_exists(libdir):
            for rootdir, _, filenames in self._osutils.walk(libdir):
                for filename in filenames:
                    fullpath: str = self._osutils.joinpath(rootdir, filename)
                    zip_path: str = self._osutils.joinpath(
                        self._CHALICE_LIB_DIR, fullpath[len(libdir) + 1:]
                    )
                    yield (fullpath, zip_path)

    def _create_output_dir_if_needed(self, package_filename: str) -> None:
        dirname: str = self._osutils.dirname(
            self._osutils.abspath(package_filename)
        )
        if not self._osutils.directory_exists(dirname):
            self._osutils.makedirs(dirname)

    def _build_python_dependencies(
        self,
        python_version: str,
        requirements_filepath: str,
        site_packages_dir: str,
    ) -> None:
        try:
            abi: str = self._RUNTIME_TO_ABI[python_version]
            self._dependency_builder.build_site_packages(
                abi, requirements_filepath, site_packages_dir
            )
        except MissingDependencyError as e:
            missing_packages: str = '\n'.join([p.identifier for p in e.missing])
            self._ui.write(MISSING_DEPENDENCIES_TEMPLATE % missing_packages)


class LambdaDeploymentPackager(BaseLambdaDeploymentPackager):
    def create_deployment_package(
        self, project_dir: str, python_version: str
    ) -> str:
        msg: str = "Creating deployment package."
        self._ui.write("%s\n" % msg)
        logger.debug(msg)
        package_filename: str = self.deployment_package_filename(
            project_dir, python_version
        )
        if self._osutils.file_exists(package_filename):
            self._ui.write("Reusing existing deployment package.\n")
            return package_filename
        self._create_output_dir_if_needed(package_filename)
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath: str = self._get_requirements_filename(
                project_dir
            )
            self._build_python_dependencies(
                python_version, requirements_filepath, site_packages_dir=tmpdir
            )
            with self._osutils.open_zip(
                package_filename, 'w', self._osutils.ZIP_DEFLATED
            ) as z:
                self._add_py_deps(z, deps_dir=tmpdir)
                self._add_app_files(z, project_dir)
                self._add_vendor_files(
                    z, self._osutils.joinpath(project_dir, self._VENDOR_DIR)
                )
        return package_filename


class AppOnlyDeploymentPackager(BaseLambdaDeploymentPackager):
    def create_deployment_package(
        self, project_dir: str, python_version: str
    ) -> str:
        msg: str = "Creating app deployment package."
        self._ui.write("%s\n" % msg)
        logger.debug(msg)
        package_filename: str = self.deployment_package_filename(
            project_dir, python_version
        )
        if self._osutils.file_exists(package_filename):
            self._ui.write("  Reusing existing app deployment package.\n")
            return package_filename
        self._create_output_dir_if_needed(package_filename)
        with self._osutils.open_zip(
            package_filename, 'w', self._osutils.ZIP_DEFLATED
        ) as z:
            self._add_app_files(z, project_dir)
        return package_filename

    def deployment_package_filename(
        self, project_dir: str, python_version: str
    ) -> str:
        return self._deployment_package_filename(
            project_dir, python_version, prefix='appcode-'
        )

    def _deployment_package_filename(
        self, project_dir: str, python_version: str, prefix: str = ''
    ) -> str:
        h: Any = hashlib.md5(b'')
        for filename, _ in self._iter_app_filenames(project_dir):
            with self._osutils.open(filename, 'rb') as f:
                reader: Callable[[int], bytes] = functools.partial(f.read, 1024 * 1024)
                for chunk in iter(reader, b''):
                    h.update(chunk)
        digest: str = h.hexdigest()
        filename: str = '%s%s-%s.zip' % (prefix, digest, python_version)
        deployment_package_filename: str = self._osutils.joinpath(
            project_dir, '.chalice', 'deployments', filename
        )
        return deployment_package_filename


class LayerDeploymentPackager(BaseLambdaDeploymentPackager):

    # A Lambda layer will unzip into the /opt directory instead of
    # the current working directory of the function.  This means
    # in order for our python dependencies to work we need.
    _PREFIX: str = 'python/lib/%s/site-packages'

    def create_deployment_package(
        self, project_dir: str, python_version: str
    ) -> str:
        msg: str = "Creating shared layer deployment package."
        self._ui.write("%s\n" % msg)
        logger.debug(msg)
        package_filename: str = self.deployment_package_filename(
            project_dir, python_version
        )
        self._create_output_dir_if_needed(package_filename)
        if self._osutils.file_exists(package_filename):
            self._ui.write(
                "  Reusing existing shared layer deployment package.\n"
            )
            return package_filename
        with self._osutils.tempdir() as tmpdir:
            requirements_filepath: str = self._get_requirements_filename(
                project_dir
            )
            self._build_python_dependencies(
                python_version, requirements_filepath, site_packages_dir=tmpdir
            )
            with self._osutils.open_zip(
                package_filename, 'w', self._osutils.ZIP_DEFLATED
            ) as z:
                prefix: str = self._PREFIX % python_version
                self._add_py_deps(z, deps_dir=tmpdir, prefix=prefix)
                self._add_vendor_files(
                    z,
                    self._osutils.joinpath(project_dir, self._VENDOR_DIR),
                    prefix=prefix,
                )
        self._check_valid_package(package_filename)
        return package_filename

    def _check_valid_package(self, package_filename: str) -> None:
        # Lambda does not allow empty deployment packages, so if there are no
        # requirements.txt deps or anything in vendor/, we need to let the
        # call know that we couldn't generate a useful deployment package
        # for them.  The caller will then need make the appropriate adjustments
        # i.e remove that LambdaLayer model from the app.
        with self._osutils.open_zip(
            package_filename, 'r', self._osutils.ZIP_DEFLATED
        ) as z:
            total_size: int = sum(f.file_size for f in z.infolist())
            # We have to check the total archive size, Lambda will still error
            # out if you have a zip file with all empty files.  It's not enough
            # to check if the zipfile is empty.
            if not total_size > 0:
                # We want to make sure we remove any deployment packages we
                # know are invalid, we don't want them being used as cache
                # hits in subsequent create_deployment_package() requests.
                self._osutils.remove_file(package_filename)
                raise EmptyPackageError(package_filename)

    def deployment_package_filename(
        self, project_dir: str, python_version: str
    ) -> str:
        return self._deployment_package_filename(
            project_dir, python_version, prefix='managed-layer-'
        )

    def _deployment_package_filename(
        self, project_dir: str, python_version: str, prefix: str = ''
    ) -> str:
        requirements_filename: str = self._get_requirements_filename(project_dir)
        if not self._osutils.file_exists(requirements_filename):
            contents: bytes = b''
        else:
            contents = cast(
                bytes,
                self._osutils.get_file_contents(
                    requirements_filename, binary=True
                ),
            )
        h: Any = hashlib.md5(contents)
        vendor_dir: str = self._osutils.joinpath(project_dir, self._VENDOR_DIR)
        if self._osutils.directory_exists(vendor_dir):
            self._hash_vendor_dir(vendor_dir, h)
        hash_contents: str = h.hexdigest()
        filename: str = '%s%s-%s.zip' % (prefix, hash_contents, python_version)
        deployment_package_filename: str = self._osutils.joinpath(
            project_dir, '.chalice', 'deployments', filename
        )
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
    _MANYLINUX_LEGACY_MAP: Dict[str, str] = {
        'manylinux1_x86_64': 'manylinux_2_5_x86_64',
        'manylinux2010_x86_64': 'manylinux_2_12_x86_64',
        'manylinux2014_x86_64': 'manylinux_2_17_x86_64',
    }

    # Mapping of abi to glibc version in Lambda runtime.
    _RUNTIME_GLIBC: Dict[str, Tuple[int, int]] = {
        'cp27mu': (2, 17),
        'cp36m': (2, 17),
        'cp37m': (2, 17),
        'cp38': (2, 26),
        'cp310': (2, 26),
        'cp311': (2, 26),
        'cp312': (2, 26),
    }
    # Fallback version if we're on an unknown python version
    # not in _RUNTIME_GLIBC.
    # Unlikely to hit this case.
    _DEFAULT_GLIBC: Tuple[int, int] = (2, 17)

    _COMPATIBLE_PACKAGE_WHITELIST: Set[str] = {
        'sqlalchemy',
        'pyyaml',
        'pyrsistent',
    }

    def __init__(
        self, osutils: OSUtils, pip_runner: Optional[PipRunner] = None
    ) -> None:
        self._osutils: OSUtils = osutils
        if pip_runner is None:
            pip_runner = PipRunner(SubprocessPip(osutils))
        self._pip: PipRunner = pip_runner

    def _is_compatible_wheel_filename(
        self, expected_abi: str, filename: str
    ) -> bool:
        wheel: str = filename[:-4]
        all_compatibility_tags: Iterator[Tuple[str, str, str]] = self._iter_all_compatibility_tags(wheel)
        for implementation, abi, platform in all_compatibility_tags:
            # Verify platform is compatible
            if not self._is_compatible_platform_tag(expected_abi, platform):
                continue
            # Verify that the ABI is compatible with lambda. Either none or the
            # correct type for the python version cp27mu for py27 and cp36m for
            # py36.
            if abi == 'none':
                return True
            prefix_version: str = implementation[:3]
            expected_abis: List[str] = [expected_abi]
            if prefix_version == 'cp3':
                # Deploying python 3 function which means we can accept the
                # version specific abi, or we can accept the CPython 3 stable
                # ABI of 'abi3'.
                expected_abis.append('abi3')
            if abi in expected_abis:
                return True
        return False

    def _is_compatible_platform_tag(
        self, expected_abi: str, platform: str
    ) -> bool:
        # From PEP 600, the new manylinux tag is
        # manylinux_${GLIBCMAJOR}_${GLIBCMINOR}_${ARCH}
        # e.g. manylinux_2_17_x86_64.
        # To check if the wheel is compatible, we first need to map any of the
        # legacy manylinux formats to the new perennial format.
        # Then we verify that the glibc version is compatible with the version
        # on the Lambda runtime (from _RUNTIME_GLIBC).
        if platform in self._ADDITIONAL_COMPATIBLE_PLATFORM:
            logger.debug("Found compatible platform tag: %s", platform)
            return True
        elif platform.startswith('manylinux'):
            # This is roughly based on the "Package Installers" section from
            # PEP 600.
            perennial_tag: str = self._MANYLINUX_LEGACY_MAP.get(platform, platform)
            m: Optional[re.Match] = re.match("manylinux_([0-9]+)_([0-9]+)_(.*)", perennial_tag)
            if m is None:
                return False
            tag_major, tag_minor = [int(x) for x in m.groups()[:2]]
            runtime_major, runtime_minor = self._RUNTIME_GLIBC.get(
                expected_abi, self._DEFAULT_GLIBC
            )
            if (tag_major, tag_minor) <= (runtime_major, runtime_minor):
                logger.debug(
                    "Tag glibc (%s, %s) is compatible with "
                    "runtime glibc (%s, %s)",
                    tag_major,
                    tag_minor,
                    runtime_major,
                    runtime_minor,
                )
                return True
        return False

    def _iter_all_compatibility_tags(
        self, wheel: str
    ) -> Iterator[Tuple[str, str, str]]:
        # From PEP 425, section "Compressed Tag Sets"
        #
        # "To allow for compact filenames of bdists that work with more than
        # one compatibility tag triple, each tag in a filename can instead be a
        # '.'-separated, sorted, set of tags."
        #
        # So this means that we have to iterate over all the possible
        # compatibility tag tuples and check if the wheel is compatible.
        # We just need any of the combinations to match for us to consider the
        # wheel compatible.
        implementation_tag, abi_tag, platform_tag = wheel.split('-')[-3:]
        for implementation in implementation_tag.split('.'):
            for abi in abi_tag.split('.'):
                for platform in platform_tag.split('.'):
                    yield (implementation, abi, platform)

    def _has_at_least_one_package(self, filename: str) -> bool:
        if not self._osutils.file_exists(filename):
            return False
        with open(filename, 'r') as f:
            # This is meant to be a best effort attempt.
            # This can return True and still have no packages
            # actually being specified, but those aren't common
            # cases.
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return True
        return False

    def _download_all_dependencies(
        self, requirements_filename: str, directory: str
    ) -> Set[Package]:
        # Download dependencies prefering wheel files but falling back to
        # raw source dependences to get the transitive closure over
        # the dependency graph. Return the set of all package objects
        # which will serve as the master list of dependencies needed to deploy
        # successfully.
        self._pip.download_all_dependencies(requirements_filename, directory)
        deps: Set[Package] = {
            Package(directory, filename)
            for filename in self._osutils.get_directory_contents(directory)
        }
        logger.debug("Full dependency closure: %s", deps)
        return deps

    def _download_binary_wheels(
        self, abi: str, packages: Set[Package], directory: str
    ) -> None:
        # Try to get binary wheels for each package that isn't compatible.
        logger.debug("Downloading manylinux wheels: %s", packages)
        self._pip.download_manylinux_wheels(
            abi, [pkg.identifier for pkg in packages], directory
        )

    def _download_sdists(self, packages: Set[Package], directory: str) -> None:
        logger.debug("Downloading missing sdists: %s", packages)
        self._pip.download_sdists(
            [pkg.identifier for pkg in packages], directory
        )

    def _find_sdists(self, directory: str) -> Set[Package]:
        packages: List[Package] = [
            Package(directory, filename)
            for filename in self._osutils.get_directory_contents(directory)
        ]
        sdists: Set[Package] = {
            package for package in packages if package.dist_type == 'sdist'
        }
        return sdists

    def _build_sdists(
        self, sdists: Set[Package], directory: str, compile_c: bool = True
    ) -> None:
        logger.debug(
            "Build missing wheels from sdists (C compiling %s): %s",
            compile_c,
            sdists,
        )
        for sdist in sdists:
            path_to_sdist: str = self._osutils.joinpath(directory, sdist.filename)
            self._pip.build_wheel(path_to_sdist, directory, compile_c)

    def _categorize_wheel_files(
        self, abi: str, directory: str
    ) -> Tuple[Set[Package], Set[Package]]:
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

    def _categorize_deps(self, abi: str, deps: Set[Package]) -> Any:
        compatible_wheels: Set[Package] = set()
        incompatible_wheels: Set[Package] = set()
        sdists: Set[Package] = set()
        for package in deps:
            if package.dist_type == 'sdist':
                sdists.add(package)
            else:
                if self._is_compatible_wheel_filename(abi, package.filename):
                    compatible_wheels.add(package)
                else:
                    incompatible_wheels.add(package)
        return sdists, compatible_wheels, incompatible_wheels

    def _download_dependencies(
        self, abi: str, directory: str, requirements_filename: str
    ) -> Tuple[Set[Package], Set[Package]]:
        # Download all dependencies we can, letting pip choose what to
        # download.
        # deps should represent the best effort we can make to gather all the
        # dependencies.
        deps: Set[Package] = self._download_all_dependencies(
            requirements_filename, directory
        )
        # Sort the downloaded packages into three categories:
        # - sdists (Pip could not get a wheel so it gave us an sdist)
        # - lambda compatible wheel files
        # - lambda incompatible wheel files
        # Pip will give us a wheel when it can, but some distributions do not
        # ship with wheels at all in which case we will have an sdist for it.
        # In some cases a platform specific wheel file may be availble so pip
        # will have downloaded that, if our platform does not match the
        # platform lambda runs on (linux_x86_64/manylinux) then the downloaded
        # wheel file may not be compatible with lambda. Pure python wheels
        # still will be compatible because they have no platform dependencies.
        sdists, compatible_wheels, incompatible_wheels = self._categorize_deps(
            abi, deps
        )
        logger.debug("Compatible wheels for Lambda: %s", compatible_wheels)
        logger.debug(
            "Initial incompatible wheels for Lambda: %s",
            incompatible_wheels | sdists,
        )

        # Next we need to go through the downloaded packages and pick out any
        # dependencies that do not have a compatible wheel file downloaded.
        # For these packages we need to explicitly try to download a
        # compatible wheel file.
        missing_wheels: Set[Package] = sdists.union(incompatible_wheels)
        self._download_binary_wheels(abi, missing_wheels, directory)

        # Re-count the wheel files after the second download pass. Anything
        # that has an sdist but not a valid wheel file is still not going to
        # work on lambda and we must now try and build the sdist into a wheel
        # file ourselves.
        # There also may be the case where no sdist was ever downloaded. For
        # example if we are on MacOS, and the package in question has a mac
        # compatible wheel file but no linux ones, we will only have an
        # incompatible wheel file and no sdist. So we need to get any missing
        # sdists before we can build them.
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(
            abi, directory
        )
        # The self._download_binary_wheels() can now introduce duplicate
        # entries.  For example, if we download a macOS whl at first but
        # then we're able to download a manylinux1 wheel, we'll now have
        # two wheels for the package, so we have to remove any compatible
        # wheels from our set of incompatible wheels.
        incompatible_wheels -= compatible_wheels
        missing_sdists: Set[Package] = incompatible_wheels - sdists
        self._download_sdists(missing_sdists, directory)
        sdists: Set[Package] = self._find_sdists(directory)
        logger.debug(
            "compatible wheels after second download pass: %s",
            compatible_wheels,
        )
        missing_wheels: Set[Package] = sdists - compatible_wheels
        self._build_sdists(missing_wheels, directory, compile_c=True)

        # There is still the case where the package had optional C dependencies
        # for speedups. In this case the wheel file will have built above with
        # the C dependencies if it managed to find a C compiler. If we are on
        # an incompatible architecture this means the wheel file generated will
        # not be compatible. If we categorize our files once more and find that
        # there are missing dependencies we can try our last ditch effort of
        # building the package and trying to sever its ability to find a C
        # compiler.
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(
            abi, directory
        )
        logger.debug(
            "compatible after building wheels (C compiling): %s",
            compatible_wheels,
        )
        missing_wheels: Set[Package] = sdists - compatible_wheels
        self._build_sdists(missing_wheels, directory, compile_c=False)

        # Final pass to find the compatible wheel files and see if there are
        # any unmet dependencies left over. At this point there is nothing we
        # can do about any missing wheel files. We tried downloading a
        # compatible version directly and building from source.
        compatible_wheels, incompatible_wheels = self._categorize_wheel_files(
            abi, directory
        )
        logger.debug(
            "compatible after building wheels (no C compiling):