#!/usr/bin/env python3
from __future__ import annotations
import os
import zipfile
import tarfile
import io
from unittest import mock
from collections import defaultdict, namedtuple
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
from typing import Any, Dict, List, Tuple, Optional

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
def sdist_builder() -> FakeSdistBuilder:
    s = FakeSdistBuilder()
    return s

class FakeSdistBuilder:
    _SETUP_PY: str = 'from setuptools import setup\nsetup(\n    name="%s",\n    version="%s"\n)\n'

    def write_fake_sdist(self, directory: str, name: str, version: str) -> Tuple[str, str]:
        filename = '%s-%s.zip' % (name, version)
        path = '%s/%s' % (directory, filename)
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('sdist/setup.py', self._SETUP_PY % (name, version))
        return (directory, filename)

class PathArgumentEndingWith:
    def __init__(self, filename: str) -> None:
        self._filename = filename

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            filename = os.path.split(other)[-1]
            return self._filename == filename
        return False

class FakePip:
    def __init__(self) -> None:
        self._calls: Dict[str, List[Tuple[List[str], Optional[Dict[str, str]], Optional[str]]]] = defaultdict(lambda: [])
        self._call_history: List[Tuple[FakePipCall, FakePipCall]] = []
        self._side_effects: Dict[str, List[List[PipSideEffect]]] = defaultdict(lambda: [])
        self._return_tuple: Tuple[int, bytes, bytes] = (0, b'', b'')

    def main(self, args: List[str], env_vars: Optional[Dict[str, str]] = None, shim: Optional[str] = None) -> Tuple[int, bytes, bytes]:
        cmd, args_ = (args[0], args[1:])
        self._calls[cmd].append((args_, env_vars, shim))
        try:
            side_effects: List[PipSideEffect] = self._side_effects[cmd].pop(0)
            for side_effect in side_effects:
                self._call_history.append((FakePipCall(args_, env_vars, shim), 
                                             FakePipCall(side_effect.expected_args, side_effect.expected_env_vars, side_effect.expected_shim)))
                side_effect.execute(args)
        except IndexError:
            pass
        return self._return_tuple

    def set_return_tuple(self, rc: int, out: bytes, err: bytes) -> None:
        self._return_tuple = (rc, out, err)

    def packages_to_download(self, expected_args: List[Any], packages: List[str], whl_contents: Optional[List[str]] = None) -> None:
        side_effects = [PipSideEffect(pkg, '--dest', expected_args, whl_contents) for pkg in packages]
        self._side_effects['download'].append(side_effects)

    def wheels_to_build(self, expected_args: List[Any], wheels_to_build: List[str], expected_env_vars: Optional[Dict[str, str]] = None, expected_shim: Optional[str] = None) -> None:
        if expected_env_vars is None:
            expected_env_vars = {}
        if expected_shim is None:
            expected_shim = ''
        side_effects = [PipSideEffect(pkg, '--wheel-dir', expected_args, expected_env_vars=expected_env_vars, expected_shim=expected_shim) for pkg in wheels_to_build]
        self._side_effects['wheel'].append(side_effects)

    @property
    def calls(self) -> Dict[str, List[Tuple[List[str], Optional[Dict[str, str]], Optional[str]]]]:
        return self._calls

    def validate(self) -> None:
        for actual_call, expected_call in self._call_history:
            assert actual_call.args == expected_call.args
            assert actual_call.env_vars == expected_call.env_vars
            assert actual_call.shim == expected_call.shim

class PipSideEffect:
    def __init__(self, filename: str, dirarg: str, expected_args: List[Any], whl_contents: Optional[List[str]] = None, 
                 expected_env_vars: Optional[Dict[str, str]] = None, expected_shim: Optional[str] = None) -> None:
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
            package_obj = Package(directory, filename)
            with zipfile.ZipFile(filepath, 'w') as z:
                for content_path in self._whl_contents:
                    z.writestr(content_path.format(package_name=self._package_name, data_dir=package_obj.data_dir), b'')

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
        if self._dirarg:
            target_dir: Optional[str] = None
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
    class EmptyEnv:
        def environ(self) -> Dict[Any, Any]:
            return {}
    return EmptyEnv()

@pytest.fixture
def pip_runner(empty_env_osutils: Any) -> Tuple[FakePip, PipRunner]:
    pip = FakePip()
    pip_runner_obj = PipRunner(pip, osutils=empty_env_osutils)
    return (pip, pip_runner_obj)

class TestDependencyBuilder:
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

    def test_can_get_sdist_if_missing_initially(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=['foo-1.2-cp36-cp36m-macosx_10_6_intel.whl'])
        pip.packages_to_download(expected_args=['--only-binary=:all:', '--no-deps', '--platform', 'manylinux2014_x86_64', '--implementation', 'cp', '--abi', 'cp36m', '--dest', mock.ANY, 'foo==1.2'], packages=[])
        pip.packages_to_download(expected_args=['--no-binary=:all:', '--no-deps', '--dest', mock.ANY, 'foo==1.2'], packages=['foo-1.2.zip'])
        pip.wheels_to_build(expected_args=['--no-deps', '--wheel-dir', mock.ANY, PathArgumentEndingWith('foo-1.2.zip')], wheels_to_build=['foo-1.2-cp36-none-any.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_get_whls_all_manylinux(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=['foo-1.2-cp36-cp36m-manylinux1_x86_64.whl', 'bar-1.2-cp36-cp36m-manylinux1_x86_64.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_support_new_wheel_tags(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['numpy']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=['numpy-1.20.3-cp37-cp37m-manylinux_2_12_x86_64.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp37m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_support_compressed_tags(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['numpy']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=['numpy-1.20.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp37m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_use_abi3_whl_for_any_python3(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar', 'baz', 'qux']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=[
            'foo-1.2-cp33-abi3-manylinux1_x86_64.whl',
            'bar-1.2-cp34-abi3-manylinux1_x86_64.whl',
            'baz-1.2-cp35-abi3-manylinux1_x86_64.whl',
            'qux-1.2-cp36-abi3-manylinux1_x86_64.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_expand_purelib_whl(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=[
            'foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['foo-1.2.data/purelib/foo/'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_normalize_dirname_for_purelib_whl(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=[
            'foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['Foo-1.2.data/purelib/foo/'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_expand_platlib_whl(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=[
            'foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['Foo-1.2.data/platlib/foo/'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_expand_platlib_and_purelib(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=[
            'foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['foo-1.2.data/platlib/foo/', 'foo-1.2.data/purelib/bar/'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_does_ignore_data(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['foo/placeholder', 'foo-1.2.data/data/bar/']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages
        assert 'bar' not in installed_packages

    def test_does_ignore_include(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['foo/placeholder', 'foo.1.2.data/includes/bar/']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages
        assert 'bar' not in installed_packages

    def test_does_ignore_scripts(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['{package_name}/placeholder', '{data_dir}/scripts/bar/placeholder']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages
        assert 'bar' not in installed_packages

    def test_can_expand_platlib_and_platlib_and_root(self, tmpdir: Any, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar', 'baz']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.2-cp36-cp36m-manylinux1_x86_64.whl'],
            whl_contents=['{package_name}/placeholder', '{data_dir}/platlib/bar/placeholder', '{data_dir}/purelib/baz/placeholder']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_get_whls_mixed_compat(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar', 'baz']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.0-cp36-none-any.whl', 'bar-1.2-cp36-cp36m-manylinux1_x86_64.whl', 'baz-1.5-cp36-cp36m-linux_x86_64.whl']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_can_get_py27_whls(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar', 'baz']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.0-cp27-none-any.whl', 'bar-1.2-cp27-none-manylinux1_x86_64.whl', 'baz-1.5-cp27-cp27mu-linux_x86_64.whl']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp27mu', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_does_fail_on_invalid_local_package(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['../foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.set_return_tuple(0, b'Processing ../foo\n  Link is a directory, ignoring download_dir', b'')
        pip.wheels_to_build(expected_args=['--no-deps', '--wheel-dir', mock.ANY, '../foo'], wheels_to_build=['foo-1.2-cp36-cp36m-macosx_10_6_intel.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        with pytest.raises(MissingDependencyError) as e:
            builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        missing_packages = list(e.value.missing)
        pip.validate()
        assert len(missing_packages) == 1
        assert missing_packages[0].identifier == 'foo==1.2'
        assert len(installed_packages) == 0

    def test_does_fail_on_narrow_py27_unicode(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['baz']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=['baz-1.5-cp27-cp27m-linux_x86_64.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        with pytest.raises(MissingDependencyError) as e:
            builder.build_site_packages('cp27mu', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        missing_packages = list(e.value.missing)
        pip.validate()
        assert len(missing_packages) == 1
        assert missing_packages[0].identifier == 'baz==1.5'
        assert len(installed_packages) == 0

    def test_does_fail_on_python_1_whl(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['baz']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=['baz-1.5-cp14-cp14m-linux_x86_64.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        with pytest.raises(MissingDependencyError) as e:
            builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        missing_packages = list(e.value.missing)
        pip.validate()
        assert len(missing_packages) == 1
        assert missing_packages[0].identifier == 'baz==1.5'
        assert len(installed_packages) == 0

    def test_can_replace_incompat_whl(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.0-cp36-none-any.whl', 'bar-1.2-cp36-cp36m-macosx_10_6_intel.whl']
        )
        pip.packages_to_download(
            expected_args=['--only-binary=:all:', '--no-deps', '--platform', 'manylinux2014_x86_64', '--implementation', 'cp', '--abi', 'cp36m', '--dest', mock.ANY, 'bar==1.2'],
            packages=['bar-1.2-cp36-cp36m-manylinux1_x86_64.whl']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    @pytest.mark.parametrize('package,package_filename', [('sqlalchemy', 'SQLAlchemy'), ('pyyaml', 'PyYAML')])
    def test_whitelist_sqlalchemy(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner], package: str, package_filename: str) -> None:
        reqs = ['%s==1.1.18' % package]
        abi = 'cp36m'
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['%s-1.1.18-cp36-cp36m-macosx_10_11_x86_64.whl' % package_filename]
        )
        pip.packages_to_download(
            expected_args=['--only-binary=:all:', '--no-deps', '--platform', 'manylinux2014_x86_64', '--implementation', 'cp', '--abi', abi, '--dest', mock.ANY, '%s==1.1.18' % package],
            packages=['%s-1.1.18-cp36-cp36m-macosx_10_11_x86_64.whl' % package_filename]
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages(abi, requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        assert installed_packages == [package_filename]

    def test_can_build_sdist(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.2.zip', 'bar-1.2-cp36-cp36m-manylinux1_x86_64.whl']
        )
        pip.wheels_to_build(expected_args=['--no-deps', '--wheel-dir', mock.ANY, PathArgumentEndingWith('foo-1.2.zip')], wheels_to_build=['foo-1.2-cp36-none-any.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        for req in reqs:
            assert req in installed_packages

    def test_build_sdist_makes_incompatible_whl(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.2.zip', 'bar-1.2-cp36-cp36m-manylinux1_x86_64.whl']
        )
        pip.wheels_to_build(expected_args=['--no-deps', '--wheel-dir', mock.ANY, PathArgumentEndingWith('foo-1.2.zip')], wheels_to_build=['foo-1.2-cp36-cp36m-macosx_10_6_intel.whl'])
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        with pytest.raises(MissingDependencyError) as e:
            builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        missing_packages = list(e.value.missing)
        pip.validate()
        assert len(missing_packages) == 1
        assert missing_packages[0].identifier == 'foo==1.2'
        assert installed_packages == ['bar']

    def test_can_build_package_with_optional_c_speedups_and_no_wheel(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo']
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(expected_args=['-r', requirements_file, '--dest', mock.ANY], packages=['foo-1.2.zip'])
        pip.wheels_to_build(expected_args=['--no-deps', '--wheel-dir', mock.ANY, PathArgumentEndingWith('foo-1.2.zip')], wheels_to_build=['foo-1.2-cp36-cp36m-macosx_10_6_intel.whl'])
        pip.wheels_to_build(
            expected_args=['--no-deps', '--wheel-dir', mock.ANY, PathArgumentEndingWith('foo-1.2.zip')],
            expected_env_vars=pip_no_compile_c_env_vars,
            expected_shim=pip_no_compile_c_shim,
            wheels_to_build=['foo-1.2-cp36-none-any.whl']
        )
        site_packages = os.path.join(appdir, '.chalice.', 'site-packages')
        builder.build_site_packages('cp36m', requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        pip.validate()
        assert installed_packages == ['foo']

    def test_build_into_existing_dir_with_preinstalled_packages(self, tmpdir: Any, osutils: OSUtils, pip_runner: Tuple[FakePip, PipRunner]) -> None:
        reqs = ['foo', 'bar']
        abi = 'cp36m'
        pip, runner = pip_runner
        appdir, builder = self._make_appdir_and_dependency_builder(reqs, tmpdir, runner)
        requirements_file = os.path.join(appdir, 'requirements.txt')
        pip.packages_to_download(
            expected_args=['-r', requirements_file, '--dest', mock.ANY],
            packages=['foo-1.2.zip', 'bar-1.2-cp36-cp36m-manylinux1_x86_64.whl']
        )
        pip.packages_to_download(
            expected_args=['--only-binary=:all:', '--no-deps', '--platform', 'manylinux2014_x86_64', '--implementation', 'cp', '--abi', abi, '--dest', mock.ANY, 'foo==1.2'],
            packages=['foo-1.2-cp36-cp36m-macosx_10_6_intel.whl']
        )
        site_packages = os.path.join(appdir, '.chalice', 'site-packages')
        foo_dir = os.path.join(site_packages, 'foo')
        os.makedirs(foo_dir)
        bar_dir = os.path.join(site_packages, 'bar')
        os.makedirs(bar_dir)
        with pytest.raises(MissingDependencyError) as e:
            builder.build_site_packages(abi, requirements_file, site_packages)
        installed_packages = os.listdir(site_packages)
        missing_packages = list(e.value.missing)
        pip.validate()
        assert len(missing_packages) == 1
        assert missing_packages[0].identifier == 'foo==1.2'
        assert installed_packages == ['bar']

def test_can_create_app_packager_with_no_autogen(tmpdir: Any, stubbed_session: Any) -> None:
    appdir = _create_app_structure(tmpdir)
    outdir = tmpdir.mkdir('outdir')
    default_params: Dict[str, Any] = {'autogen_policy': True}
    config = Config.create(project_dir=str(appdir), chalice_app=sample_app(), **default_params)
    options = PackageOptions(TypedAWSClient(session=stubbed_session))
    p = package.create_app_packager(config, options)
    p.package_app(config, str(outdir), 'dev')
    contents = os.listdir(str(outdir))
    assert 'deployment.zip' in contents
    assert 'sam.json' in contents

def test_can_create_app_packager_with_yaml_extention(tmpdir: Any, stubbed_session: Any) -> None:
    appdir = _create_app_structure(tmpdir)
    outdir = tmpdir.mkdir('outdir')
    default_params: Dict[str, Any] = {'autogen_policy': True}
    extras_file = tmpdir.join('extras.yaml')
    extras_file.write('foo: bar')
    config = Config.create(project_dir=str(appdir), chalice_app=sample_app(), **default_params)
    options = PackageOptions(TypedAWSClient(session=stubbed_session))
    p = package.create_app_packager(config, options, merge_template=str(extras_file))
    p.package_app(config, str(outdir), 'dev')
    contents = os.listdir(str(outdir))
    assert 'deployment.zip' in contents
    assert 'sam.yaml' in contents

def test_can_specify_yaml_output(tmpdir: Any, stubbed_session: Any) -> None:
    appdir = _create_app_structure(tmpdir)
    outdir = tmpdir.mkdir('outdir')
    default_params: Dict[str, Any] = {'autogen_policy': True}
    config = Config.create(project_dir=str(appdir), chalice_app=sample_app(), **default_params)
    options = PackageOptions(TypedAWSClient(session=stubbed_session))
    p = package.create_app_packager(config, options, template_format='yaml')
    p.package_app(config, str(outdir), 'dev')
    contents = os.listdir(str(outdir))
    assert 'deployment.zip' in contents
    assert 'sam.yaml' in contents

def test_will_create_outdir_if_needed(tmpdir: Any, stubbed_session: Any) -> None:
    appdir = _create_app_structure(tmpdir)
    outdir = str(appdir.join('outdir'))
    default_params: Dict[str, Any] = {'autogen_policy': True}
    config = Config.create(project_dir=str(appdir), chalice_app=sample_app(), **default_params)
    options = PackageOptions(TypedAWSClient(session=stubbed_session))
    p = package.create_app_packager(config, options)
    p.package_app(config, str(outdir), 'dev')
    contents = os.listdir(str(outdir))
    assert 'deployment.zip' in contents
    assert 'sam.json' in contents

def test_includes_layer_package_with_sam(tmpdir: Any, stubbed_session: Any) -> None:
    appdir = _create_app_structure(tmpdir)
    appdir.mkdir('vendor').join('hello').write('hello\n')
    outdir = str(appdir.join('outdir'))
    default_params: Dict[str, Any] = {'autogen_policy': True}
    config = Config.create(project_dir=str(appdir), chalice_app=sample_app(), automatic_layer=True, **default_params)
    options = PackageOptions(TypedAWSClient(session=stubbed_session))
    p = package.create_app_packager(config, options)
    p.package_app(config, str(outdir), 'dev')
    contents = os.listdir(str(outdir))
    assert 'deployment.zip' in contents
    assert 'layer-deployment.zip' in contents
    assert 'sam.json' in contents

def test_includes_layer_package_with_terraform(tmpdir: Any, stubbed_session: Any) -> None:
    appdir = _create_app_structure(tmpdir)
    appdir.mkdir('vendor').join('hello').write('hello\n')
    outdir = str(appdir.join('outdir'))
    default_params: Dict[str, Any] = {'autogen_policy': True}
    config = Config.create(project_dir=str(appdir), chalice_app=sample_app(), automatic_layer=True, **default_params)
    options = PackageOptions(TypedAWSClient(session=stubbed_session))
    p = package.create_app_packager(config, options, package_format='terraform')
    p.package_app(config, str(outdir), 'dev')
    contents = os.listdir(str(outdir))
    assert 'deployment.zip' in contents
    assert 'layer-deployment.zip' in contents
    assert 'chalice.tf.json' in contents

class TestSubprocessPip:
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

class TestSdistMetadataFetcher:
    _SETUPTOOLS: str = 'from setuptools import setup'
    _DISTUTILS: str = 'from distutils.core import setup'
    _BOTH: str = 'try:\n    from setuptools import setup\nexcept ImportError:\n    from distutils.core import setuptools\n'
    _SETUP_PY: str = '%s\nsetup(\n    name="%s",\n    version="%s"\n)\n'
    _VALID_TAR_FORMATS: List[str] = ['tar.gz', 'tar.bz2']

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

    def test_setup_tar_gz(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._SETUPTOOLS, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'tar.gz')
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_setup_tar_bz2(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._SETUPTOOLS, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'tar.bz2')
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_setup_zip(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._SETUPTOOLS, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'zip')
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_distutil_tar_gz(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._DISTUTILS, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'tar.gz')
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_distutil_tar_bz2(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._DISTUTILS, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'tar.bz2')
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_distutil_zip(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._DISTUTILS, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'zip')
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_both_zip(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._BOTH, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'zip')
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_bad_format(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % (self._BOTH, 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'tar.gz2')
            with pytest.raises(InvalidSourceDistributionNameError):
                _ = sdist_reader.get_package_name_and_version(filepath)

    def test_cant_get_egg_info_filename(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        bad_setup_py = self._SETUP_PY % ('import some_build_dependency', 'foo', '1.0')
        pkg_info_file = 'Name: foo\nVersion: 1.0\n'
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(bad_setup_py, tempdir, 'zip', pkg_info_file)
            name, version = sdist_reader.get_package_name_and_version(filepath)
        assert name == 'foo'
        assert version == '1.0'

    def test_pkg_info_fallback_fails_raises_error(self, osutils: OSUtils, sdist_reader: SDistMetadataFetcher) -> None:
        setup_py = self._SETUP_PY % ('import build_time_dependency', 'foo', '1.0')
        with osutils.tempdir() as tempdir:
            filepath = self._write_fake_sdist(setup_py, tempdir, 'tar.gz')
            with pytest.raises(UnsupportedPackageError):
                sdist_reader.get_package_name_and_version(filepath)

class TestPackage:
    def test_same_pkg_sdist_and_wheel_collide(self, osutils: OSUtils, sdist_builder: FakeSdistBuilder) -> None:
        with osutils.tempdir() as tempdir:
            sdist_builder.write_fake_sdist(tempdir, 'foobar', '1.0')
            pkgs = set()
            pkgs.add(Package('', 'foobar-1.0-py3-none-any.whl'))
            pkgs.add(Package(tempdir, 'foobar-1.0.zip'))
            assert len(pkgs) == 1

    def test_ensure_sdist_name_normalized_for_comparison(self, osutils: OSUtils, sdist_builder: FakeSdistBuilder) -> None:
        with osutils.tempdir() as tempdir:
            sdist_builder.write_fake_sdist(tempdir, 'Foobar', '1.0')
            pkgs = set()
            pkgs.add(Package('', 'foobar-1.0-py3-none-any.whl'))
            pkgs.add(Package(tempdir, 'Foobar-1.0.zip'))
            assert len(pkgs) == 1

    def test_ensure_wheel_name_normalized_for_comparison(self, osutils: OSUtils, sdist_builder: FakeSdistBuilder) -> None:
        with osutils.tempdir() as tempdir:
            sdist_builder.write_fake_sdist(tempdir, 'foobar', '1.0')
            pkgs = set()
            pkgs.add(Package('', 'Foobar-1.0-py3-none-any.whl'))
            pkgs.add(Package(tempdir, 'foobar-1.0.zip'))
            assert len(pkgs) == 1
