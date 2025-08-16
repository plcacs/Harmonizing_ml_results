import unittest
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
import pytest
from flask import g
from sqlalchemy.orm.session import make_transient
from tests.integration_tests.fixtures.energy_dashboard import load_energy_table_with_slice, load_energy_table_data
from tests.integration_tests.test_app import app
from superset.commands.dashboard.importers.v0 import decode_dashboards
from superset import db, security_manager
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.commands.dashboard.importers.v0 import import_chart, import_dashboard
from superset.commands.dataset.importers.v0 import import_dataset
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.utils.core import DatasourceType, get_example_default_schema
from superset.utils.database import get_example_database
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME
from tests.integration_tests.fixtures.world_bank_dashboard import load_world_bank_dashboard_with_slices, load_world_bank_data

def delete_imports() -> None:
    ...

@pytest.fixture(autouse=True, scope='module')
def clean_imports() -> None:
    ...

class TestImportExport(SupersetTestCase):
    ...

    def create_slice(self, name: str, ds_id: int = None, id: int = None, db_name: str = 'examples', table_name: str = 'wb_health_population', schema: str = None) -> Slice:
        ...

    def create_dashboard(self, title: str, id: int = 0, slcs: List[Slice] = []) -> Dashboard:
        ...

    def create_table(self, name: str, schema: str = None, id: int = 0, cols_names: List[str] = [], metric_names: List[str] = []) -> SqlaTable:
        ...

    def get_slice(self, slc_id: int) -> Slice:
        ...

    def get_slice_by_name(self, name: str) -> Slice:
        ...

    def get_dash(self, dash_id: int) -> Dashboard:
        ...

    def assert_dash_equals(self, expected_dash: Dashboard, actual_dash: Dashboard, check_position: bool = True, check_slugs: bool = True) -> None:
        ...

    def assert_table_equals(self, expected_ds: SqlaTable, actual_ds: SqlaTable) -> None:
        ...

    def assert_datasource_equals(self, expected_ds: SqlaTable, actual_ds: SqlaTable) -> None:
        ...

    def assert_slice_equals(self, expected_slc: Slice, actual_slc: Slice) -> None:
        ...

    def assert_only_exported_slc_fields(self, expected_dash: Dashboard, actual_dash: Dashboard) -> None:
        ...

    @unittest.skip('Schema needs to be updated')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_export_1_dashboard(self) -> None:
        ...

    @unittest.skip('Schema needs to be updated')
    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices', 'load_birth_names_dashboard_with_slices')
    def test_export_2_dashboards(self) -> None:
        ...

    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_import_1_slice(self) -> None:
        ...

    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_import_2_slices_for_same_table(self) -> None:
        ...

    def test_import_slices_override(self) -> None:
        ...

    def test_import_empty_dashboard(self) -> None:
        ...

    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_import_dashboard_1_slice(self) -> None:
        ...

    @pytest.mark.usefixtures('load_energy_table_with_slice')
    def test_import_dashboard_2_slices(self) -> None:
        ...

    @pytest.mark.usefixtures('load_energy_table_with_slice')
    def test_import_override_dashboard_2_slices(self) -> None:
        ...

    def test_import_new_dashboard_slice_reset_ownership(self) -> None:
        ...

    @pytest.mark.skip
    def test_import_override_dashboard_slice_reset_ownership(self) -> None:
        ...

    def _create_dashboard_for_import(self, id_: int = 10100) -> Dashboard:
        ...

    def test_import_table_no_metadata(self) -> None:
        ...

    def test_import_table_1_col_1_met(self) -> None:
        ...

    def test_import_table_2_col_2_met(self) -> None:
        ...

    def test_import_table_override(self) -> None:
        ...

    def test_import_table_override_identical(self) -> None:
        ...
