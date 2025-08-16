import unittest
from uuid import uuid4
import yaml
from tests.integration_tests.test_app import app
from superset import db
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.utils.database import get_example_database
from superset.utils import json
from .base_tests import SupersetTestCase
from typing import Tuple, Any, Dict

DBREF: str = 'dict_import__export_test'
NAME_PREFIX: str = 'dict_'
ID_PREFIX: int = 20000

class TestDictImportExport(SupersetTestCase):
    def delete_imports(cls) -> None:
    def setUpClass(cls) -> None:
    def tearDownClass(cls) -> None:
    def create_table(self, name: str, schema: str = None, id: int = 0, cols_names: List[str] = [], cols_uuids: List[uuid4] = None, metric_names: List[str] = []) -> Tuple[SqlaTable, Dict[str, Any]]:
    def yaml_compare(self, obj_1: Any, obj_2: Any) -> None:
    def assert_table_equals(self, expected_ds: SqlaTable, actual_ds: SqlaTable) -> None:
    def assert_datasource_equals(self, expected_ds: SqlaTable, actual_ds: SqlaTable) -> None:
    def test_import_table_no_metadata(self) -> None:
    def test_import_table_1_col_1_met(self) -> None:
    def test_import_table_2_col_2_met(self) -> None:
    def test_import_table_override_append(self) -> None:
    def test_import_table_override_sync(self) -> None:
    def test_import_table_override_identical(self) -> None:
