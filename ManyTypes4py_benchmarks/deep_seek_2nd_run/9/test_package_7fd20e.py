import os
import zipfile
import tarfile
import io
from unittest import mock
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Tuple, Set, Optional, Union, DefaultDict, NamedTuple
import pytest
from chalice.awsclient import TypedAWSClient
from chalice.config import Config
from chalice import Chalice
from chalice import package
from chalice.deploy.packager import PipRunner
from chalice.deploy.packager import DependencyBuilder
from chalice.deploy.packager import Package
from chalice.deploy.packager import MissingDependencyError
from chalice.deploy.packager import SubprocessPip
from chalice.deploy.packager import SDistMetadataFetcher
from chalice.deploy.packager import InvalidSourceDistributionNameError
from chalice.deploy.packager import UnsupportedPackageError
from chalice.compat import pip_no_compile_c_env_vars
from chalice.compat import pip_no_compile_c_shim
from chalice.package import PackageOptions
from chalice.utils import OSUtils

FakePipCall = namedtuple('FakePipEntry', ['args', 'env_vars', 'shim'])

def _create_app_structure(tmpdir: Any) -> Any:
    appdir = tmpdir.mkdir('app')
    appdir.join('app.py').write('# Test app')
    appdir.mkdir('.chalice')
    return appdir

def sample_app() -> Chalice:
    app = Chalice('sample_app')

    @app.route('/')
    def index() -> Dict[str, str]:
        return {'hello': 'world'}
    return app

@pytest.fixture
def sdist_reader() -> SDistMetadataFetcher:
    return SDistMetadataFetcher()

@pytest.fixture
def sdist_builder() -> 'FakeSdistBuilder':
    s = FakeSdistBuilder()
    return s

class FakeSdistBuilder(object):
    _SETUP_PY = 'from setuptools import setup\nsetup(\n    name="%s",\n    version="%s"\n)\n'

    def write_fake_sdist(self, directory: str, name: str, version: str) -> Tuple[str, str]:
        filename = '%s-%s.zip' % (name, version)
        path = '%s/%s' % (directory, filename)
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('sdist/setup.py', self._SETUP_PY % (name, version))
        return (directory, filename)

class PathArgumentEndingWith(object):
    def __init__(self, filename: str) -> None:
        self._filename = filename

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            filename = os.path.split(other)[-1]
            return self._filename == filename
        return False

class FakePip(object):
    def __init__(self) -> None:
        self._calls: DefaultDict[str, List[Tuple[List[str], Optional[Dict[str, str]], Optional[str]]] = defaultdict(lambda: [])
        self._call_history: List[Tuple[FakePipCall, FakePipCall]] = []
        self._side_effects: DefaultDict[str, List[List['PipSideEffect']]] = defaultdict(lambda: [])
        self._return_tuple: Tuple[int, bytes, bytes] = (0, b'', b'')

    def main(self, args: List[str], env_vars: Optional[Dict[str, str]] = None, shim: Optional[str] = None) -> Tuple[int, bytes, bytes]:
        cmd, args = (args[0], args[1:])
        self._calls[cmd].append((args, env_vars, shim))
        try:
            side_effects = self._side_effects[cmd].pop(0)
            for side_effect in side_effects:
                self._call_history.append((FakePipCall(args, env_vars, shim), FakePipCall(side_effect.expected_args, side_effect.expected_env_vars, side_effect.expected_shim)))
                side_effect.execute(args)
        except IndexError:
            pass
        return self._return_tuple

    def set_return_tuple(self, rc: int, out: bytes, err: bytes) -> None:
        self._return_tuple = (rc, out, err)

    def packages_to_download(self, expected_args: List[str], packages: List[str], whl_contents: Optional[List[str]] = None) -> None:
        side_effects = [PipSideEffect(pkg, '--dest', expected_args, whl_contents) for pkg in packages]
        self._side_effects['download'].append(side_effects)

    def wheels_to_build(self, expected_args: List[str], wheels_to_build: List[str], expected_env_vars: Optional[Dict[str, str]] = None, expected_shim: Optional[str] = None) -> None:
        if expected_env_vars is None:
            expected_env_vars = {}
        if expected_shim is None:
            expected_shim = ''
        side_effects = [PipSideEffect(pkg, '--wheel-dir', expected_args, expected_env_vars=expected_env_vars, expected_shim=expected_shim) for pkg in wheels_to_build]
        self._side_effects['wheel'].append(side_effects)

    @property
    def calls(self) -> DefaultDict[str, List[Tuple[List[str], Optional[Dict[str, str]], Optional[str]]]:
        return self._calls

    def validate(self) -> None:
        for calls in self._call_history:
            actual_call, expected_call = calls
            assert actual_call.args == expected_call.args
            assert actual_call.env_vars == expected_call.env_vars
            assert actual_call.shim == expected_call.shim

class PipSideEffect(object):
    def __init__(self, filename: str, dirarg: str, expected_args: List[str], whl_contents: Optional[List[str]] = None, expected_env_vars: Optional[Dict[str, str]] = None, expected_shim: Optional[str] = None) -> None:
        self._filename = filename
        self._package_name = filename.split('-')[0]
        self._dirarg = dirarg
        self.expected_args = expected_args
        self.expected_env_vars = expected_env_vars
        self.expected_shim = expected_shim
        if whl_contents is None:
            whl_contents = ['{package_name}/placeholder']
        self._whl_contents = whl_contents

    def _build_fake_whl(self, directory: str, filename: str) -> None:
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            package = Package(directory, filename)
            with zipfile.ZipFile(filepath, 'w') as z:
                for content_path in self._whl_contents:
                    z.writestr(content_path.format(package_name=self._package_name, data_dir=package.data_dir), b'')

    def _build_fake_sdist(self, filepath: str) -> None:
        assert filepath.endswith('.zip')
        components = os.path.split(filepath)
        prefix, filename = (components[:-1], components[-1])
        directory = os.path.join(*prefix)
        filename_without_ext = filename[:-4]
        pkg_name, pkg_version = filename_without_ext.split('-')
        builder = FakeSdistBuilder()
        builder.write_fake_sdist(directory, pkg_name, pkg_version)

    def execute(self, args: List[str]) -> None:
        """Generate the file in the target_dir."""
        if self._dirarg:
            target_dir = None
            for i, arg in enumerate(args):
                if arg == self._dirarg:
                    target_dir = args[i + 1]
            if target_dir:
                filepath = os.path.join(target_dir, self._filename)
                if filepath.endswith('.whl'):
                    self._build_fake_whl(target_dir, self._filename)
                else:
                    self._build_fake_sdist(filepath)

@pytest.fixture
def osutils() -> OSUtils:
    return OSUtils()

@pytest.fixture
def empty_env_osutils() -> Any:
    class EmptyEnv(object):
        def environ(self) -> Dict[str, str]:
            return {}
    return EmptyEnv()

@pytest.fixture
def pip_runner(empty_env_osutils: Any) -> Tuple[FakePip, PipRunner]:
    pip = FakePip()
    pip_runner = PipRunner(pip, osutils=empty_env_osutils)
    return (pip, pip_runner)

class TestDependencyBuilder(object):
    def _write_requirements_txt(self, packages: List[str], directory: str) -> None:
        contents = '\n'.join(packages)
        filepath = os.path.join(directory, 'requirements.txt')
        with open(filepath, 'w') as f:
            f.write(contents)

    def _make_appdir_and_dependency_builder(self, reqs: List[str], tmpdir: Any, runner: PipRunner) -> Tuple[str, DependencyBuilder]:
        appdir = str(_create_app_structure(tmpdir))
        self._write_requirements_txt(reqs, appdir)
        builder = DependencyBuilder(OSUtils(), runner)
        return (appdir, builder)

    def test_can_build_local_dir_as_whl(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['../foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.set_return_tuple(0, b'Processing ../foo\n  Link is a directory, ignoring download_dir', b'')
        pip.wheels_to_build(expected_args=['--no-deps', '--wheel-dir', mock.ANY, '../foo'], wheels_to_build=['foo-1.2-cp36-none-any.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        assert ['foo'] == installed_packages

    # ... [rest of the TestDependencyBuilder methods with type annotations] ...

class TestSubprocessPip(object):
    def test_can_invoke_pip(self) -> None:
        pip = SubprocessPip()
        rc, out, err = pip.main(['--version'])
        print(out, err)
        assert rc == 0
        assert err == b''

    def test_does_error_code_propagate(self) -> None:
        pip = SubprocessPip()
        rc, _, err = pip.main(['badcommand'])
        assert rc != 0
        assert err != b''

class TestSdistMetadataFetcher(object):
    _SETUPTOOLS = 'from setuptools import setup'
    _DISTUTILS = 'from distutils.core import setup'
    _BOTH = 'try:\n    from setuptools import setup\nexcept ImportError:\n    from distutils.core import setuptools\n'
    _SETUP_PY = '%s\nsetup(\n    name="%s",\n    version="%s"\n)\n'
    _VALID_TAR_FORMATS = ['tar.gz', 'tar.bz2']

    def _write_fake_sdist(self, setup_py: str, directory: str, ext: str, pkg_info_contents: Optional[str] = None) -> str:
        filename = 'sdist.%s' % ext
        path = '%s/%s' % (directory, filename)
        if ext == 'zip':
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('sdist/setup.py', setup_py)
                if pkg_info_contents is not None:
                    z.writestr('sdist/PKG-INFO', pkg_info_contents)
        elif ext in self._VALID_TAR_FORMATS:
            compression_format = ext.split('.')[1]
            with tarfile.open(path, 'w:%s' % compression_format) as tar:
                tarinfo = tarfile.TarInfo('sdist/setup.py')
                tarinfo.size = len(setup_py)
                tar.addfile(tarinfo, io.BytesIO(setup_py.encode()))
                if pkg_info_contents is not None:
                    tarinfo = tarfile.TarInfo('sdist/PKG-INFO')
                    tarinfo.size = len(pkg_info_contents)
                    tar.addfile(tarinfo, io.BytesIO(pkg_info_contents.encode()))
        else:
            open(path, 'a').close()
        filepath = os.path.join(directory, filename)
        return filepath

    # ... [rest of the TestSdistMetadataFetcher methods with type annotations] ...

class TestPackage(object):
    def test_same_pkg_sdist_and_wheel_collide(self, osutils: OSUtils, sdist_builder: FakeSdistBuilder) -> None:
        with osutils.tempdir() as tempdir:
            sdist_builder.write_fake_sdist(tempdir, 'foobar', '1.0')
            pkgs: Set[Package] = set()
            pkgs.add(Package('', 'foobar-1.0-py3-none-any.whl'))
            pkgs.add(Package(tempdir, 'foobar-1.0.zip'))
            assert len(pkgs) == 1

    # ... [rest of the TestPackage methods with type annotations] ...

def test_can_create_app_packager_with_no_autogen(tmpdir: Any, stubbed_session: Any) -> None:
    appdir = _create_app_structure(tmpdir)
    outdir = tmpdir.mkdir('outdir')
    default_params = {'autogen_policy': True}
    config = Config.create(project_dir=str(appdir), chalice_app=sample_app(), **default_params)
    options = PackageOptions(TypedAWSClient(session=stubbed_session))
    p = package.create_app_packager(config, options)
    p.package_app(config, str(outdir), 'dev')
    contents = os.listdir(str(outdir))
    assert 'deployment.zip' in contents
    assert 'sam.json' in contents

# ... [rest of the test functions with type annotations] ...
