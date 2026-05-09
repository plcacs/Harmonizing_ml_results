import unittest
from argparse import Namespace
from copy import deepcopy
from unittest import mock
from unittest.mock import MagicMock
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

class TestLocalPackage(unittest.TestCase):
    def test_init(self) -> None:
        ...

class TestTarballPackage(unittest.TestCase):
    class MockMetadata:
        name: str
    @mock.patch('dbt.config.project.PartialProject.from_project_root')
    @mock.patch('os.listdir')
    @mock.patch('dbt.deps.tarball.get_downloads_path')
    @mock.patch('dbt_common.clients.system.untar_package')
    @mock.patch('dbt_common.clients.system.download')
    def test_fetch_metadata(self, mock_download: MagicMock, mock_untar_package: MagicMock, mock_get_downloads_path: MagicMock, mock_listdir: MagicMock, mock_from_project_root: MagicMock) -> None:
        ...
    @mock.patch('dbt.config.project.PartialProject.from_project_root')
    @mock.patch('os.listdir')
    @mock.patch('dbt.deps.tarball.get_downloads_path')
    @mock.patch('dbt_common.clients.system.untar_package')
    @mock.patch('dbt_common.clients.system.download')
    def test_fetch_metadata_fails_on_incorrect_tar_folder_structure(self, mock_download: MagicMock, mock_untar_package: MagicMock, mock_get_downloads_path: MagicMock, mock_listdir: MagicMock, mock_from_project_root: MagicMock) -> None:
        ...
    @mock.patch('dbt.deps.tarball.get_downloads_path')
    def test_tarball_package_contract(self, mock_get_downloads_path: MagicMock) -> None:
        ...
    @mock.patch('dbt.deps.tarball.get_downloads_path')
    def test_tarball_pinned_package_contract_with_unrendered(self, mock_get_downloads_path: MagicMock) -> None:
        ...
    def test_tarball_package_contract_fails_on_no_name(self) -> None:
        ...

class TestGitPackage(unittest.TestCase):
    def test_init(self) -> None:
        ...
    def test_init_with_unrendered(self) -> None:
        ...
    @mock.patch('shutil.copytree')
    @mock.patch('dbt.deps.local.system.make_symlink')
    @mock.patch('dbt.deps.local.LocalPinnedPackage.get_installation_path')
    @mock.patch('dbt.deps.local.LocalPinnedPackage.resolve_path')
    def test_deps_install(self, mock_resolve_path: MagicMock, mock_get_installation_path: MagicMock, mock_symlink: MagicMock, mock_shutil: MagicMock) -> None:
        ...
    def test_invalid(self) -> None:
        ...
    def test_resolve_ok(self) -> None:
        ...
    def test_resolve_fail(self) -> None:
        ...
    def test_default_revision(self) -> None:
        ...

class TestHubPackage(unittest.TestCase):
    def setUp(self) -> None:
        ...
    def tearDown(self) -> None:
        ...
    def test_init(self) -> None:
        ...
    def test_invalid(self) -> None:
        ...
    def test_resolve_ok(self) -> None:
        ...
    def test_resolve_missing_package(self) -> None:
        ...
    def test_resolve_missing_version(self) -> None:
        ...
    def test_resolve_conflict(self) -> None:
        ...
    def test_resolve_ranges(self) -> None:
        ...
    def test_resolve_ranges_install_prerelease_default_false(self) -> None:
        ...
    def test_resolve_ranges_install_prerelease_true(self) -> None:
        ...
    def test_get_version_latest_prelease_true(self) -> None:
        ...
    def test_get_version_latest_prelease_false(self) -> None:
        ...
    def test_get_version_prerelease_explicitly_requested(self) -> None:
        ...

class TestPackageSpec(unittest.TestCase):
    def setUp(self) -> None:
        ...
    def tearDown(self) -> None:
        ...
    def test_dependency_resolution(self) -> None:
        ...
    def test_private_package_raise_error(self) -> None:
        ...
    def test_dependency_resolution_allow_prerelease(self) -> None:
        ...
    def test_validation_error_when_version_is_missing_from_package_config(self) -> None:
        ...
    def test_validation_error_when_namespace_is_missing_from_package_config(self) -> None:
        ...