#!/usr/bin/env python3
from typing import Any, Dict, List, Optional, Type
import unittest
from argparse import Namespace
from copy import deepcopy
from unittest import mock

import dbt.deps
import dbt.exceptions
from dbt.clients.registry import is_compatible_version
from dbt.config.project import PartialProject
from dbt.config.renderer import DbtProjectYamlRenderer
from dbt.contracts.project import GitPackage, LocalPackage, PackageConfig, RegistryPackage, TarballPackage
from dbt.deps.git import GitUnpinnedPackage
from dbt.deps.local import LocalPinnedPackage, LocalUnpinnedPackage
from dbt.deps.registry import RegistryUnpinnedPackage
from dbt.deps.resolver import resolve_packages
from dbt.deps.tarball import TarballUnpinnedPackage
from dbt.flags import set_from_args
from dbt.version import get_installed_version
from dbt_common.dataclass_schema import ValidationError
from dbt_common.semver import VersionSpecifier

set_from_args(Namespace(WARN_ERROR=False), None)


class TestLocalPackage(unittest.TestCase):

    def test_init(self) -> None:
        a_contract: LocalPackage = LocalPackage.from_dict({'local': '/path/to/package'})
        self.assertEqual(a_contract.local, '/path/to/package')
        a: LocalUnpinnedPackage = LocalUnpinnedPackage.from_contract(a_contract)
        self.assertEqual(a.local, '/path/to/package')
        a_pinned: LocalPinnedPackage = a.resolved()
        self.assertEqual(a_pinned.local, '/path/to/package')
        self.assertEqual(str(a_pinned), '/path/to/package')


class TestTarballPackage(unittest.TestCase):

    class MockMetadata:
        name: str = 'mock_metadata_name'

    @mock.patch('dbt.config.project.PartialProject.from_project_root')
    @mock.patch('os.listdir')
    @mock.patch('dbt.deps.tarball.get_downloads_path')
    @mock.patch('dbt_common.clients.system.untar_package')
    @mock.patch('dbt_common.clients.system.download')
    def test_fetch_metadata(self, 
                            mock_download: mock.MagicMock, 
                            mock_untar_package: mock.MagicMock, 
                            mock_get_downloads_path: mock.MagicMock, 
                            mock_listdir: mock.MagicMock, 
                            mock_from_project_root: mock.MagicMock
                           ) -> None:
        mock_listdir.return_value = ['one_directory/']
        mock_get_downloads_path.return_value = 'downloads_path'
        mock_from_project_root.return_value = object()
        dict_well_formed_contract: Dict[str, Any] = {
            'tarball': 'http://example.com/invalid_url@/package.tar.gz', 
            'name': 'my_package'
        }
        a_contract: TarballPackage = TarballPackage.from_dict(dict_well_formed_contract)
        a: TarballUnpinnedPackage = TarballUnpinnedPackage.from_contract(a_contract)
        a_pinned: Any = a.resolved()
        with mock.patch.object(PartialProject, 'from_project_root', return_value=PartialProject):
            with mock.patch.object(PartialProject, 'render_package_metadata', return_value=self.MockMetadata):
                metadata: Any = a_pinned.fetch_metadata('', DbtProjectYamlRenderer())
        self.assertEqual(metadata, self.MockMetadata)
        mock_download.assert_called_once_with(
            'http://example.com/invalid_url@/package.tar.gz', 'downloads_path/my_package'
        )
        mock_untar_package.assert_called_once_with(
            'downloads_path/my_package', 'downloads_path/my_package_untarred', 'my_package'
        )

    @mock.patch('dbt.config.project.PartialProject.from_project_root')
    @mock.patch('os.listdir')
    @mock.patch('dbt.deps.tarball.get_downloads_path')
    @mock.patch('dbt_common.clients.system.untar_package')
    @mock.patch('dbt_common.clients.system.download')
    def test_fetch_metadata_fails_on_incorrect_tar_folder_structure(self, 
                                                                    mock_download: mock.MagicMock, 
                                                                    mock_untar_package: mock.MagicMock, 
                                                                    mock_get_downloads_path: mock.MagicMock, 
                                                                    mock_listdir: mock.MagicMock, 
                                                                    mock_from_project_root: mock.MagicMock
                                                                   ) -> None:
        mock_listdir.return_value = ['one_directory/', 'another_directory/']
        mock_get_downloads_path.return_value = 'downloads_path'
        mock_from_project_root.return_value = object()
        dict_well_formed_contract: Dict[str, Any] = {
            'tarball': 'http://example.com/invalid_url@/package.tar.gz', 
            'name': 'my_package'
        }
        a_contract: TarballPackage = TarballPackage.from_dict(dict_well_formed_contract)
        a: TarballUnpinnedPackage = TarballUnpinnedPackage.from_contract(a_contract)
        a_pinned: Any = a.resolved()
        with mock.patch.object(PartialProject, 'from_project_root', return_value=PartialProject):
            with mock.patch.object(PartialProject, 'render_package_metadata', return_value=self.MockMetadata):
                with self.assertRaises(dbt.exceptions.DependencyError):
                    a_pinned.fetch_metadata('', DbtProjectYamlRenderer())

    @mock.patch('dbt.deps.tarball.get_downloads_path')
    def test_tarball_package_contract(self, mock_get_downloads_path: mock.MagicMock) -> None:
        dict_well_formed_contract: Dict[str, Any] = {
            'tarball': 'http://example.com/invalid_url@/package.tar.gz', 
            'name': 'my_cool_package'
        }
        a_contract: TarballPackage = TarballPackage.from_dict(dict_well_formed_contract)
        self.assertEqual(a_contract.tarball, 'http://example.com/invalid_url@/package.tar.gz')
        self.assertEqual(a_contract.name, 'my_cool_package')
        a: TarballUnpinnedPackage = TarballUnpinnedPackage.from_contract(a_contract)
        self.assertEqual(a.tarball, 'http://example.com/invalid_url@/package.tar.gz')
        self.assertEqual(a.package, 'my_cool_package')
        a_pinned: Any = a.resolved()
        self.assertEqual(a_pinned.source_type(), 'tarball')
        a_pinned_dict: Dict[str, Any] = a_pinned.to_dict()
        self.assertEqual(a_pinned_dict, {
            'tarball': 'http://example.com/invalid_url@/package.tar.gz', 
            'name': 'my_cool_package'
        })

    @mock.patch('dbt.deps.tarball.get_downloads_path')
    def test_tarball_pinned_package_contract_with_unrendered(self, mock_get_downloads_path: mock.MagicMock) -> None:
        contract: TarballPackage = TarballPackage(
            tarball='http://example.com/invalid_url@/package.tar.gz', 
            name='my_cool_package', 
            unrendered={'tarball': 'tarball_unrendered'}
        )
        tarball_unpinned_package: TarballUnpinnedPackage = TarballUnpinnedPackage.from_contract(contract)
        self.assertEqual(tarball_unpinned_package.tarball, 'http://example.com/invalid_url@/package.tar.gz')
        self.assertEqual(tarball_unpinned_package.package, 'my_cool_package')
        self.assertEqual(tarball_unpinned_package.tarball_unrendered, 'tarball_unrendered')
        tarball_pinned_package: Any = tarball_unpinned_package.resolved()
        tarball_unpinned_package_dict: Dict[str, Any] = tarball_pinned_package.to_dict()
        self.assertEqual(tarball_unpinned_package_dict, {
            'tarball': 'tarball_unrendered', 
            'name': 'my_cool_package'
        })

    def test_tarball_package_contract_fails_on_no_name(self) -> None:
        from mashumaro.exceptions import MissingField
        a_contract: Dict[str, Any] = {'tarball': 'http://example.com'}
        with self.assertRaises(MissingField):
            TarballPackage.from_dict(a_contract)


class TestGitPackage(unittest.TestCase):

    def test_init(self) -> None:
        a_contract: GitPackage = GitPackage.from_dict({'git': 'http://example.com', 'revision': '0.0.1'})
        self.assertEqual(a_contract.git, 'http://example.com')
        self.assertEqual(a_contract.revision, '0.0.1')
        self.assertIs(a_contract.warn_unpinned, None)
        a: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(a_contract)
        self.assertEqual(a.git, 'http://example.com')
        self.assertEqual(a.revisions, ['0.0.1'])
        self.assertIs(a.warn_unpinned, True)
        a_pinned: Any = a.resolved()
        self.assertEqual(a_pinned.name, 'http://example.com')
        self.assertEqual(a_pinned.get_version(), '0.0.1')
        self.assertEqual(a_pinned.source_type(), 'git')
        self.assertIs(a_pinned.warn_unpinned, True)
        a_pinned_dict: Dict[str, Any] = a_pinned.to_dict()
        self.assertEqual(a_pinned_dict, {'git': 'http://example.com', 'revision': '0.0.1'})

    def test_init_with_unrendered(self) -> None:
        contract: GitPackage = GitPackage(git='http://example.com', revision='0.0.1', unrendered={'git': 'git_unrendered'})
        git_unpinned_package: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(contract)
        self.assertEqual(git_unpinned_package.git, 'http://example.com')
        self.assertEqual(git_unpinned_package.revisions, ['0.0.1'])
        self.assertEqual(git_unpinned_package.git_unrendered, 'git_unrendered')
        git_pinned_package: Any = git_unpinned_package.resolved()
        git_pinned_package_dict: Dict[str, Any] = git_pinned_package.to_dict()
        self.assertEqual(git_pinned_package_dict, {'git': 'git_unrendered', 'revision': '0.0.1'})

    @mock.patch('shutil.copytree')
    @mock.patch('dbt.deps.local.system.make_symlink')
    @mock.patch('dbt.deps.local.LocalPinnedPackage.get_installation_path')
    @mock.patch('dbt.deps.local.LocalPinnedPackage.resolve_path')
    def test_deps_install(self, 
                          mock_resolve_path: mock.MagicMock, 
                          mock_get_installation_path: mock.MagicMock, 
                          mock_symlink: mock.MagicMock, 
                          mock_shutil: mock.MagicMock
                         ) -> None:
        mock_resolve_path.return_value = '/tmp/source'
        mock_get_installation_path.return_value = '/tmp/dest'
        mock_symlink.side_effect = OSError('Install deps symlink error')
        LocalPinnedPackage('local').install('dummy', 'dummy')
        self.assertEqual(mock_shutil.call_count, 1)
        mock_shutil.assert_called_once_with('/tmp/source', '/tmp/dest')

    def test_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            GitPackage.validate({'git': 'http://example.com', 'version': '0.0.1'})

    def test_resolve_ok(self) -> None:
        a_contract: GitPackage = GitPackage.from_dict({'git': 'http://example.com', 'revision': '0.0.1'})
        b_contract: GitPackage = GitPackage.from_dict({'git': 'http://example.com', 'revision': '0.0.1', 'warn-unpinned': False})
        d_contract: GitPackage = GitPackage.from_dict({'git': 'http://example.com', 'revision': '0.0.1', 'subdirectory': 'foo-bar'})
        a: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(a_contract)
        b: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(b_contract)
        c: GitUnpinnedPackage = a.incorporate(b)
        d: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(d_contract)
        self.assertTrue(a.warn_unpinned)
        self.assertFalse(b.warn_unpinned)
        self.assertTrue(d.warn_unpinned)
        c_pinned: Any = c.resolved()
        self.assertEqual(c_pinned.name, 'http://example.com')
        self.assertEqual(c_pinned.get_version(), '0.0.1')
        self.assertEqual(c_pinned.source_type(), 'git')
        self.assertFalse(c_pinned.warn_unpinned)
        d_pinned: Any = d.resolved()
        self.assertEqual(d_pinned.name, 'http://example.com/foo-bar')
        self.assertEqual(d_pinned.get_version(), '0.0.1')
        self.assertEqual(d_pinned.source_type(), 'git')
        self.assertEqual(d_pinned.subdirectory, 'foo-bar')

    def test_resolve_fail(self) -> None:
        a_contract: GitPackage = GitPackage.from_dict({'git': 'http://example.com', 'revision': '0.0.1'})
        b_contract: GitPackage = GitPackage.from_dict({'git': 'http://example.com', 'revision': '0.0.2'})
        a: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(a_contract)
        b: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(b_contract)
        c: GitUnpinnedPackage = a.incorporate(b)
        self.assertEqual(c.git, 'http://example.com')
        self.assertEqual(c.revisions, ['0.0.1', '0.0.2'])
        with self.assertRaises(dbt.exceptions.DependencyError):
            c.resolved()

    def test_default_revision(self) -> None:
        a_contract: GitPackage = GitPackage.from_dict({'git': 'http://example.com'})
        self.assertEqual(a_contract.revision, None)
        self.assertIs(a_contract.warn_unpinned, None)
        a: GitUnpinnedPackage = GitUnpinnedPackage.from_contract(a_contract)
        self.assertEqual(a.git, 'http://example.com')
        self.assertEqual(a.revisions, [])
        self.assertIs(a.warn_unpinned, True)
        a_pinned: Any = a.resolved()
        self.assertEqual(a_pinned.name, 'http://example.com')
        self.assertEqual(a_pinned.get_version(), 'HEAD')
        self.assertEqual(a_pinned.source_type(), 'git')
        self.assertIs(a_pinned.warn_unpinned, True)


class TestHubPackage(unittest.TestCase):

    def setUp(self) -> None:
        self.patcher: Any = mock.patch('dbt.deps.registry.registry')
        self.registry: Any = self.patcher.start()
        self.index_cached = self.registry.index_cached
        self.get_compatible_versions = self.registry.get_compatible_versions
        self.package_version = self.registry.package_version
        self.index_cached.return_value = ['dbt-labs-test/a']
        self.get_compatible_versions.return_value = ['0.1.2', '0.1.3', '0.1.4a1']
        self.package_version.return_value = {
            'id': 'dbt-labs-test/a/0.1.2',
            'name': 'a',
            'version': '0.1.2',
            'packages': [],
            '_source': {'blahblah': 'asdfas'},
            'downloads': {'tarball': 'https://example.com/invalid-url!', 'extra': 'field'},
            'newfield': ['another', 'value']
        }

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_init(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='0.1.2')
        self.assertEqual(a_contract.package, 'dbt-labs-test/a')
        self.assertEqual(a_contract.version, '0.1.2')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        self.assertEqual(a.package, 'dbt-labs-test/a')
        self.assertEqual(a.versions, [VersionSpecifier(build=None, major='0', matcher='=', minor='1', patch='2', prerelease=None)])
        a_pinned: Any = a.resolved()
        self.assertEqual(a_contract.package, 'dbt-labs-test/a')
        self.assertEqual(a_contract.version, '0.1.2')
        self.assertEqual(a_pinned.source_type(), 'hub')

    def test_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            RegistryPackage.validate({'package': 'namespace/name', 'key': 'invalid'})

    def test_resolve_ok(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='0.1.2')
        b_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='0.1.2')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        b: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(b_contract)
        c: RegistryUnpinnedPackage = a.incorporate(b)
        self.assertEqual(c.package, 'dbt-labs-test/a')
        self.assertEqual(c.versions, [
            VersionSpecifier(build=None, major='0', matcher='=', minor='1', patch='2', prerelease=None),
            VersionSpecifier(build=None, major='0', matcher='=', minor='1', patch='2', prerelease=None)
        ])
        c_pinned: Any = c.resolved()
        self.assertEqual(c_pinned.package, 'dbt-labs-test/a')
        self.assertEqual(c_pinned.version, '0.1.2')
        self.assertEqual(c_pinned.source_type(), 'hub')

    def test_resolve_missing_package(self) -> None:
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(
            RegistryPackage(package='dbt-labs-test/b', version='0.1.2')
        )
        with self.assertRaises(dbt.exceptions.DependencyError) as exc:
            a.resolved()
        msg: str = 'Package dbt-labs-test/b was not found in the package index'
        self.assertEqual(msg, str(exc.exception))

    def test_resolve_missing_version(self) -> None:
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(
            RegistryPackage(package='dbt-labs-test/a', version='0.1.4')
        )
        with self.assertRaises(dbt.exceptions.DependencyError) as exc:
            a.resolved()
        msg: str = (
            "Could not find a matching compatible version for package dbt-labs-test/a\n"
            "  Requested range: =0.1.4, =0.1.4\n"
            "  Compatible versions: ['0.1.2', '0.1.3']\n"
        )
        self.assertIn(msg, str(exc.exception))

    def test_resolve_conflict(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='0.1.2')
        b_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='0.1.3')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        b: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(b_contract)
        c: RegistryUnpinnedPackage = a.incorporate(b)
        with self.assertRaises(dbt.exceptions.DependencyError) as exc:
            c.resolved()
        msg: str = "Version error for package dbt-labs-test/a: Could not find a satisfactory version from options: ['=0.1.2', '=0.1.3']"
        self.assertEqual(msg, str(exc.exception))

    def test_resolve_ranges(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='0.1.2')
        b_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='<0.1.4')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        b: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(b_contract)
        c: RegistryUnpinnedPackage = a.incorporate(b)
        self.assertEqual(c.package, 'dbt-labs-test/a')
        self.assertEqual(c.versions, [
            VersionSpecifier(build=None, major='0', matcher='=', minor='1', patch='2', prerelease=None),
            VersionSpecifier(build=None, major='0', matcher='<', minor='1', patch='4', prerelease=None)
        ])
        c_pinned: Any = c.resolved()
        self.assertEqual(c_pinned.package, 'dbt-labs-test/a')
        self.assertEqual(c_pinned.version, '0.1.2')
        self.assertEqual(c_pinned.source_type(), 'hub')

    def test_resolve_ranges_install_prerelease_default_false(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='>0.1.2')
        b_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='<0.1.5')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        b: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(b_contract)
        c: RegistryUnpinnedPackage = a.incorporate(b)
        self.assertEqual(c.package, 'dbt-labs-test/a')
        self.assertEqual(c.versions, [
            VersionSpecifier(build=None, major='0', matcher='>', minor='1', patch='2', prerelease=None),
            VersionSpecifier(build=None, major='0', matcher='<', minor='1', patch='5', prerelease=None)
        ])
        c_pinned: Any = c.resolved()
        self.assertEqual(c_pinned.package, 'dbt-labs-test/a')
        self.assertEqual(c_pinned.version, '0.1.3')
        self.assertEqual(c_pinned.source_type(), 'hub')

    def test_resolve_ranges_install_prerelease_true(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='>0.1.2', install_prerelease=True)
        b_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='<0.1.5')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        b: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(b_contract)
        c: RegistryUnpinnedPackage = a.incorporate(b)
        self.assertEqual(c.package, 'dbt-labs-test/a')
        self.assertEqual(c.versions, [
            VersionSpecifier(build=None, major='0', matcher='>', minor='1', patch='2', prerelease=None),
            VersionSpecifier(build=None, major='0', matcher='<', minor='1', patch='5', prerelease=None)
        ])
        c_pinned: Any = c.resolved()
        self.assertEqual(c_pinned.package, 'dbt-labs-test/a')
        self.assertEqual(c_pinned.version, '0.1.4a1')
        self.assertEqual(c_pinned.source_type(), 'hub')

    def test_get_version_latest_prelease_true(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='>0.1.0', install_prerelease=True)
        b_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='<0.1.4')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        b: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(b_contract)
        c: RegistryUnpinnedPackage = a.incorporate(b)
        self.assertEqual(c.package, 'dbt-labs-test/a')
        self.assertEqual(c.versions, [
            VersionSpecifier(build=None, major='0', matcher='>', minor='1', patch='0', prerelease=None),
            VersionSpecifier(build=None, major='0', matcher='<', minor='1', patch='4', prerelease=None)
        ])
        c_pinned: Any = c.resolved()
        self.assertEqual(c_pinned.package, 'dbt-labs-test/a')
        self.assertEqual(c_pinned.version, '0.1.3')
        self.assertEqual(c_pinned.get_version_latest(), '0.1.4a1')
        self.assertEqual(c_pinned.source_type(), 'hub')

    def test_get_version_latest_prelease_false(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='>0.1.0', install_prerelease=False)
        b_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='<0.1.4')
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        b: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(b_contract)
        c: RegistryUnpinnedPackage = a.incorporate(b)
        self.assertEqual(c.package, 'dbt-labs-test/a')
        self.assertEqual(c.versions, [
            VersionSpecifier(build=None, major='0', matcher='>', minor='1', patch='0', prerelease=None),
            VersionSpecifier(build=None, major='0', matcher='<', minor='1', patch='4', prerelease=None)
        ])
        c_pinned: Any = c.resolved()
        self.assertEqual(c_pinned.package, 'dbt-labs-test/a')
        self.assertEqual(c_pinned.version, '0.1.3')
        self.assertEqual(c_pinned.get_version_latest(), '0.1.3')
        self.assertEqual(c_pinned.source_type(), 'hub')

    def test_get_version_prerelease_explicitly_requested(self) -> None:
        a_contract: RegistryPackage = RegistryPackage(package='dbt-labs-test/a', version='0.1.4a1', install_prerelease=None)
        a: RegistryUnpinnedPackage = RegistryUnpinnedPackage.from_contract(a_contract)
        self.assertEqual(a.package, 'dbt-labs-test/a')
        self.assertEqual(a.versions, [
            VersionSpecifier(build=None, major='0', matcher='=', minor='1', patch='4', prerelease='a1')
        ])
        a_pinned: Any = a.resolved()
        self.assertEqual(a_pinned.package, 'dbt-labs-test/a')
        self.assertEqual(a_pinned.version, '0.1.4a1')
        self.assertEqual(a_pinned.get_version_latest(), '0.1.4a1')
        self.assertEqual(a_pinned.source_type(), 'hub')


class MockRegistry:

    def __init__(self, packages: Dict[str, Any]) -> None:
        self.packages: Dict[str, Any] = packages

    def index_cached(self, registry_base_url: Optional[str] = None) -> List[str]:
        return sorted(self.packages)

    def package(self, package_name: str, registry_base_url: Optional[str] = None) -> List[Any]:
        try:
            pkg = self.packages[package_name]
        except KeyError:
            return []
        return pkg

    def get_compatible_versions(self, package_name: str, dbt_version: Any, should_version_check: bool) -> List[str]:
        packages = self.package(package_name)
        return [pkg_version for pkg_version, info in packages.items() if is_compatible_version(info, dbt_version)]

    def package_version(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        try:
            return self.packages[name][version]
        except KeyError:
            return None


class TestPackageSpec(unittest.TestCase):

    def setUp(self) -> None:
        dbt_version: Any = get_installed_version()
        next_version: Any = deepcopy(dbt_version)
        next_version.minor = str(int(next_version.minor) + 1)
        next_version.prerelease = None
        require_next_version: str = '>' + next_version.to_version_string()
        self.patcher: Any = mock.patch('dbt.deps.registry.registry')
        self.registry: Any = self.patcher.start()
        self.mock_registry: MockRegistry = MockRegistry(packages={
            'dbt-labs-test/a': {
                '0.1.2': {
                    'id': 'dbt-labs-test/a/0.1.2',
                    'name': 'a',
                    'version': '0.1.2',
                    'packages': [],
                    '_source': {'blahblah': 'asdfas'},
                    'downloads': {'tarball': 'https://example.com/invalid-url!', 'extra': 'field'},
                    'newfield': ['another', 'value']
                },
                '0.1.3': {
                    'id': 'dbt-labs-test/a/0.1.3',
                    'name': 'a',
                    'version': '0.1.3',
                    'packages': [],
                    '_source': {'blahblah': 'asdfas'},
                    'downloads': {'tarball': 'https://example.com/invalid-url!', 'extra': 'field'},
                    'newfield': ['another', 'value']
                },
                '0.1.4a1': {
                    'id': 'dbt-labs-test/a/0.1.3a1',
                    'name': 'a',
                    'version': '0.1.4a1',
                    'packages': [],
                    '_source': {'blahblah': 'asdfas'},
                    'downloads': {'tarball': 'https://example.com/invalid-url!', 'extra': 'field'},
                    'newfield': ['another', 'value']
                },
                '0.2.0': {
                    'id': 'dbt-labs-test/a/0.2.0',
                    'name': 'a',
                    'version': '0.2.0',
                    'packages': [],
                    '_source': {'blahblah': 'asdfas'},
                    'require_dbt_version': require_next_version,
                    'downloads': {'tarball': 'https://example.com/invalid-url!', 'extra': 'field'},
                    'newfield': ['another', 'value']
                }
            },
            'dbt-labs-test/b': {
                '0.2.1': {
                    'id': 'dbt-labs-test/b/0.2.1',
                    'name': 'b',
                    'version': '0.2.1',
                    'packages': [{'package': 'dbt-labs-test/a', 'version': '>=0.1.3'}],
                    '_source': {'blahblah': 'asdfas'},
                    'downloads': {'tarball': 'https://example.com/invalid-url!', 'extra': 'field'},
                    'newfield': ['another', 'value']
                }
            }
        })
        self.registry.index_cached.side_effect = self.mock_registry.index_cached
        self.registry.get_compatible_versions.side_effect = self.mock_registry.get_compatible_versions
        self.registry.package_version.side_effect = self.mock_registry.package_version

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_dependency_resolution(self) -> None:
        package_config: PackageConfig = PackageConfig.from_dict({
            'packages': [
                {'package': 'dbt-labs-test/a', 'version': '>0.1.2'},
                {'package': 'dbt-labs-test/b', 'version': '0.2.1'}
            ]
        })
        resolved: List[Any] = resolve_packages(package_config.packages, mock.MagicMock(project_name='test'), {})
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0].name, 'dbt-labs-test/a')
        self.assertEqual(resolved[0].version, '0.1.3')
        self.assertEqual(resolved[1].name, 'dbt-labs-test/b')
        self.assertEqual(resolved[1].version, '0.2.1')

    def test_private_package_raise_error(self) -> None:
        package_config: PackageConfig = PackageConfig.from_dict({
            'packages': [{'private': 'dbt-labs-test/a', 'subdirectory': 'foo-bar'}]
        })
        with self.assertRaisesRegex(dbt.exceptions.DependencyError, 'Cannot resolve private package'):
            resolve_packages(package_config.packages, mock.MagicMock(project_name='test'), {})

    def test_dependency_resolution_allow_prerelease(self) -> None:
        package_config: PackageConfig = PackageConfig.from_dict({
            'packages': [
                {'package': 'dbt-labs-test/a', 'version': '>0.1.2', 'install_prerelease': True},
                {'package': 'dbt-labs-test/b', 'version': '0.2.1'}
            ]
        })
        resolved: List[Any] = resolve_packages(package_config.packages, mock.MagicMock(project_name='test'), {})
        self.assertEqual(resolved[0].name, 'dbt-labs-test/a')
        self.assertEqual(resolved[0].version, '0.1.4a1')

    def test_validation_error_when_version_is_missing_from_package_config(self) -> None:
        packages_data: Dict[str, Any] = {'packages': [{'package': 'dbt-labs-test/b', 'version': None}]}
        with self.assertRaises(ValidationError) as exc:
            PackageConfig.validate(data=packages_data)
        msg: str = 'dbt-labs-test/b is missing the version. When installing from the Hub package index, version is a required property'
        self.assertIn(msg, str(exc.exception))

    def test_validation_error_when_namespace_is_missing_from_package_config(self) -> None:
        packages_data: Dict[str, Any] = {'packages': [{'package': 'dbt-labs', 'version': '1.0.0'}]}
        with self.assertRaises(ValidationError) as exc:
            PackageConfig.validate(data=packages_data)
        msg: str = 'dbt-labs was not found in the package index. Packages on the index require a namespace, e.g dbt-labs/dbt_utils'
        self.assertIn(msg, str(exc.exception))
