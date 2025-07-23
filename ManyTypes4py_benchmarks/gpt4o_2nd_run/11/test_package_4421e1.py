import os
import sys
import stat
import uuid
import fnmatch
from zipfile import ZipFile
import hashlib
from contextlib import contextmanager
from click.testing import CliRunner
import pytest
from chalice import cli
from chalice.cli import factory
from chalice.cli.newproj import create_new_project_skeleton
from chalice.deploy.packager import NoSuchPackageError
from typing import Generator, List, Tuple

PY_VERSION: Tuple[int, int] = sys.version_info[:2]
VERSION_CUTOFF: Tuple[int, int] = (3, 11)
PACKAGES_TO_TEST: dict = {
    'pandas': {'version': '2.2.0', 'legacy_version': '1.5.3', 'contents': ['pandas/*__init__.py', 'pandas/*cpython-*-x86_64-linux-gnu.so']},
    'SQLAlchemy': {'version': '2.0.27', 'legacy_version': '1.4.47', 'contents': ['sqlalchemy/__init__.py', 'sqlalchemy/*cpython-*-x86_64-linux-gnu.so']},
    'numpy': {'version': '1.26.4', 'legacy_version': '1.23.3', 'contents': ['numpy/__init__.py', 'numpy/*cpython-*-x86_64-linux-gnu.so']},
    'cryptography': {'version': '42.0.4', 'legacy_version': '39.0.0', 'contents': ['cryptography/__init__.py', 'cryptography/*.so']},
    'Jinja2': {'version': '3.1.3', 'legacy_version': '2.11.2', 'contents': ['jinja2/__init__.py']},
    'Mako': {'version': '1.3.2', 'legacy_version': '1.1.3', 'contents': ['mako/__init__.py']},
    'MarkupSafe': {'version': '2.1.5', 'legacy_version': '1.1.1', 'contents': ['markupsafe/__init__.py']},
    'scipy': {'version': '1.12.0', 'legacy_version': '1.10.1', 'contents': ['scipy/__init__.py', 'scipy/cluster/_hierarchy.cpython-*-x86_64-linux-gnu.so']},
    'cffi': {'version': '1.16.0', 'legacy_version': '1.15.1', 'contents': ['_cffi_backend.cpython-*-x86_64-linux-gnu.so']},
    'pygit2': {'version': '1.14.1', 'legacy_version': '1.10.1', 'contents': ['pygit2/_pygit2.cpython-*-x86_64-linux-gnu.so']},
    'pyrsistent': {'version': '0.20.0', 'legacy_version': '0.17.3', 'contents': ['pyrsistent/__init__.py']}
}

@contextmanager
def cd(path: str) -> Generator[None, None, None]:
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()

@pytest.fixture
def app_skeleton(tmpdir, runner: CliRunner) -> str:
    project_name = 'deployment-integ-test'
    with cd(str(tmpdir)):
        create_new_project_skeleton(project_name)
    return str(tmpdir.join(project_name))

def _get_random_package_name() -> str:
    return 'foobar-%s' % str(uuid.uuid4())[:8]

def _get_package_install_test_cases() -> List[Tuple[str, List[str]]]:
    testcases = []
    if PY_VERSION <= VERSION_CUTOFF:
        version_key = 'legacy_version'
    else:
        version_key = 'version'
    for package, config in PACKAGES_TO_TEST.items():
        package_version = f'{package}=={config[version_key]}'
        testcases.append((package_version, config['contents']))
    return testcases

@pytest.mark.skipif(not os.environ.get('CHALICE_TEST_EXTENDED_PACKAGING'), reason='Set CHALICE_TEST_EXTENDED_PACKAGING for extended packaging tests.')
@pytest.mark.parametrize('package,contents', _get_package_install_test_cases())
def test_package_install_smoke_tests(package: str, contents: List[str], runner: CliRunner, app_skeleton: str) -> None:
    assert_can_package_dependency(runner, app_skeleton, package, contents)

def assert_can_package_dependency(runner: CliRunner, app_skeleton: str, package: str, contents: List[str]) -> None:
    req = os.path.join(app_skeleton, 'requirements.txt')
    with open(req, 'w') as f:
        f.write('%s\n' % package)
    cli_factory = factory.CLIFactory(app_skeleton)
    package_output_location = os.path.join(app_skeleton, 'pkg')
    result = runner.invoke(cli.package, [package_output_location], obj={'project_dir': app_skeleton, 'debug': False, 'factory': cli_factory})
    if result.exit_code != 0:
        raise AssertionError(f'Non-zero RC when packaging {package}') from result.exception
    assert result.exit_code == 0
    assert result.output.strip() == 'Creating deployment package.'
    package_path = os.path.join(app_skeleton, 'pkg', 'deployment.zip')
    package_file = ZipFile(package_path)
    package_content = package_file.namelist()
    for content in contents:
        assert any((fnmatch.fnmatch(filename, content) for filename in package_content)), 'No match found for %s' % content

class TestPackage(object):

    def test_can_package_with_dashes_in_name(self, runner: CliRunner, app_skeleton: str, no_local_config) -> None:
        assert_can_package_dependency(runner, app_skeleton, 'googleapis-common-protos==1.5.2', contents=['google/api/__init__.py'])

    def test_can_package_simplejson(self, runner: CliRunner, app_skeleton: str, no_local_config) -> None:
        assert_can_package_dependency(runner, app_skeleton, 'simplejson==3.17.0', contents=['simplejson/__init__.py'])

    def test_can_package_sqlalchemy(self, runner: CliRunner, app_skeleton: str, no_local_config) -> None:
        assert_can_package_dependency(runner, app_skeleton, 'SQLAlchemy==1.3.13', contents=['sqlalchemy/__init__.py'])

    def test_can_package_pandas(self, runner: CliRunner, app_skeleton: str, no_local_config) -> None:
        version = '2.2.0' if sys.version_info[1] >= 10 else '2.0.3'
        assert_can_package_dependency(runner, app_skeleton, 'pandas==' + version, contents=['pandas/_libs/__init__.py'])

    def test_does_not_package_bad_requirements_file(self, runner: CliRunner, app_skeleton: str) -> None:
        req = os.path.join(app_skeleton, 'requirements.txt')
        package = _get_random_package_name()
        with open(req, 'w') as f:
            f.write('%s\n' % package)
        cli_factory = factory.CLIFactory(app_skeleton)
        result = runner.invoke(cli.package, ['package'], obj={'project_dir': app_skeleton, 'debug': False, 'factory': cli_factory})
        assert result.exception is not None
        ex = result.exception
        assert isinstance(ex, NoSuchPackageError)
        assert str(ex) == 'Could not satisfy the requirement: %s' % package

    def test_packaging_requirements_keeps_same_hash(self, runner: CliRunner, app_skeleton: str, no_local_config) -> None:
        req = os.path.join(app_skeleton, 'requirements.txt')
        package = 'botocore==1.12.202'
        with open(req, 'w') as f:
            f.write('%s\n' % package)
        cli_factory = factory.CLIFactory(app_skeleton)
        package_output_location = os.path.join(app_skeleton, 'pkg')
        self._run_package_cmd(package_output_location, app_skeleton, cli_factory, runner)
        original_checksum = self._calculate_checksum(package_output_location)
        self._run_package_cmd(package_output_location, app_skeleton, cli_factory, runner)
        new_checksum = self._calculate_checksum(package_output_location)
        assert original_checksum == new_checksum

    def test_preserves_executable_permissions(self, runner: CliRunner, app_skeleton: str, no_local_config) -> None:
        vendor = os.path.join(app_skeleton, 'vendor')
        os.makedirs(vendor)
        executable_file = os.path.join(vendor, 'myscript.sh')
        with open(executable_file, 'w') as f:
            f.write('#!/bin/bash\necho foo\n')
        os.chmod(executable_file, 493)
        cli_factory = factory.CLIFactory(app_skeleton)
        package_output_location = os.path.join(app_skeleton, 'pkg')
        self._run_package_cmd(package_output_location, app_skeleton, cli_factory, runner)
        self._verify_file_is_executable(package_output_location, 'myscript.sh')
        original_checksum = self._calculate_checksum(package_output_location)
        self._run_package_cmd(package_output_location, app_skeleton, cli_factory, runner)
        new_checksum = self._calculate_checksum(package_output_location)
        assert original_checksum == new_checksum

    def _calculate_checksum(self, package_output_location: str) -> str:
        zip_filename = os.path.join(package_output_location, 'deployment.zip')
        with open(zip_filename, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _run_package_cmd(self, package_output_location: str, app_skeleton: str, cli_factory: factory.CLIFactory, runner: CliRunner, expected_exit_code: int = 0) -> None:
        result = runner.invoke(cli.package, [package_output_location], obj={'project_dir': app_skeleton, 'debug': False, 'factory': cli_factory})
        assert result.exit_code == expected_exit_code
        return result

    def _verify_file_is_executable(self, package_output_location: str, filename: str) -> None:
        zip_filename = os.path.join(package_output_location, 'deployment.zip')
        with ZipFile(zip_filename) as zip:
            zipinfo = zip.getinfo(filename)
            assert zipinfo.external_attr >> 16 & stat.S_IXUSR
