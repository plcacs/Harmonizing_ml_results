"""Unit tests for Superset"""
import unittest
from uuid import uuid4
import yaml
from typing import Optional, List, Dict, Any, Tuple

from tests.integration_tests.test_app import app
from superset import db
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.utils.database import get_example_database
from superset.utils import json
from .base_tests import SupersetTestCase

DBREF = 'dict_import__export_test'
NAME_PREFIX = 'dict_'
ID_PREFIX = 20000


class TestDictImportExport(SupersetTestCase):
    """Testing export import functionality for dashboards"""

    @classmethod
    def delete_imports(cls) -> None:
        with app.app_context():
            for table in db.session.query(SqlaTable):
                if DBREF in table.params_dict:
                    db.session.delete(table)
            db.session.commit()

    @classmethod
    def setUpClass(cls) -> None:
        cls.delete_imports()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.delete_imports()

    def create_table(
        self,
        name: str,
        schema: Optional[str] = None,
        id: int = 0,
        cols_names: List[str] = [],
        cols_uuids: Optional[List[Optional[str]]] = None,
        metric_names: List[str] = []
    ) -> Tuple[SqlaTable, Dict[str, Any]]:
        database_name = 'main'
        name = f'{NAME_PREFIX}{name}'
        params: Dict[str, Any] = {DBREF: id, 'database_name': database_name}
        if cols_uuids is None:
            cols_uuids = [None] * len(cols_names)
        dict_rep: Dict[str, Any] = {
            'database_id': get_example_database().id,
            'table_name': name,
            'schema': schema,
            'id': id,
            'params': json.dumps(params),
            'columns': [
                {'column_name': c, 'uuid': u}
                for c, u in zip(cols_names, cols_uuids, strict=False)
            ],
            'metrics': [
                {'metric_name': c, 'expression': ''}
                for c in metric_names
            ],
        }
        table = SqlaTable(
            id=id,
            schema=schema,
            table_name=name,
            params=json.dumps(params)
        )
        for col_name, uuid in zip(cols_names, cols_uuids, strict=False):
            table.columns.append(TableColumn(column_name=col_name, uuid=uuid))
        for metric_name in metric_names:
            table.metrics.append(SqlMetric(metric_name=metric_name, expression=''))
        return table, dict_rep

    def yaml_compare(self, obj_1: Any, obj_2: Any) -> None:
        obj_1_str = yaml.safe_dump(obj_1, default_flow_style=False)
        obj_2_str = yaml.safe_dump(obj_2, default_flow_style=False)
        assert obj_1_str == obj_2_str

    def assert_table_equals(self, expected_ds: SqlaTable, actual_ds: SqlaTable) -> None:
        assert expected_ds.table_name == actual_ds.table_name
        assert expected_ds.main_dttm_col == actual_ds.main_dttm_col
        assert expected_ds.schema == actual_ds.schema
        assert len(expected_ds.metrics) == len(actual_ds.metrics)
        assert len(expected_ds.columns) == len(actual_ds.columns)
        assert {c.column_name for c in expected_ds.columns} == {c.column_name for c in actual_ds.columns}
        assert {m.metric_name for m in expected_ds.metrics} == {m.metric_name for m in actual_ds.metrics}

    def assert_datasource_equals(self, expected_ds: Any, actual_ds: Any) -> None:
        assert expected_ds.datasource_name == actual_ds.datasource_name
        assert expected_ds.main_dttm_col == actual_ds.main_dttm_col
        assert len(expected_ds.metrics) == len(actual_ds.metrics)
        assert len(expected_ds.columns) == len(actual_ds.columns)
        assert {c.column_name for c in expected_ds.columns} == {c.column_name for c in actual_ds.columns}
        assert {m.metric_name for m in expected_ds.metrics} == {m.metric_name for m in actual_ds.metrics}

    def test_import_table_no_metadata(self) -> None:
        table, dict_table = self.create_table('pure_table', id=ID_PREFIX + 1)
        new_table: SqlaTable = SqlaTable.import_from_dict(dict_table)
        db.session.commit()
        imported_id: int = new_table.id
        imported: SqlaTable = self.get_table_by_id(imported_id)
        self.assert_table_equals(table, imported)
        self.yaml_compare(table.export_to_dict(), imported.export_to_dict())

    def test_import_table_1_col_1_met(self) -> None:
        table, dict_table = self.create_table(
            'table_1_col_1_met',
            id=ID_PREFIX + 2,
            cols_names=['col1'],
            cols_uuids=[str(uuid4())],
            metric_names=['metric1']
        )
        imported_table: SqlaTable = SqlaTable.import_from_dict(dict_table)
        db.session.commit()
        imported: SqlaTable = self.get_table_by_id(imported_table.id)
        self.assert_table_equals(table, imported)
        assert {DBREF: ID_PREFIX + 2, 'database_name': 'main'} == json.loads(imported.params)
        self.yaml_compare(table.export_to_dict(), imported.export_to_dict())

    def test_import_table_2_col_2_met(self) -> None:
        table, dict_table = self.create_table(
            'table_2_col_2_met',
            id=ID_PREFIX + 3,
            cols_names=['c1', 'c2'],
            cols_uuids=[str(uuid4()), str(uuid4())],
            metric_names=['m1', 'm2']
        )
        imported_table: SqlaTable = SqlaTable.import_from_dict(dict_table)
        db.session.commit()
        imported: SqlaTable = self.get_table_by_id(imported_table.id)
        self.assert_table_equals(table, imported)
        self.yaml_compare(table.export_to_dict(), imported.export_to_dict())

    def test_import_table_override_append(self) -> None:
        table, dict_table = self.create_table(
            'table_override',
            id=ID_PREFIX + 3,
            cols_names=['col1'],
            metric_names=['m1']
        )
        imported_table: SqlaTable = SqlaTable.import_from_dict(dict_table)
        db.session.commit()
        table_over, dict_table_over = self.create_table(
            'table_override',
            id=ID_PREFIX + 3,
            cols_names=['new_col1', 'col2', 'col3'],
            metric_names=['new_metric1']
        )
        imported_over_table: SqlaTable = SqlaTable.import_from_dict(dict_table_over)
        db.session.commit()
        imported_over: SqlaTable = self.get_table_by_id(imported_over_table.id)
        assert imported_table.id == imported_over.id
        expected_table, _ = self.create_table(
            'table_override',
            id=ID_PREFIX + 3,
            metric_names=['new_metric1', 'm1'],
            cols_names=['col1', 'new_col1', 'col2', 'col3'],
            cols_uuids=[col.uuid for col in imported_over.columns]
        )
        self.assert_table_equals(expected_table, imported_over)
        self.yaml_compare(expected_table.export_to_dict(), imported_over.export_to_dict())

    def test_import_table_override_sync(self) -> None:
        table, dict_table = self.create_table(
            'table_override',
            id=ID_PREFIX + 3,
            cols_names=['col1'],
            metric_names=['m1']
        )
        imported_table: SqlaTable = SqlaTable.import_from_dict(dict_table)
        db.session.commit()
        table_over, dict_table_over = self.create_table(
            'table_override',
            id=ID_PREFIX + 3,
            cols_names=['new_col1', 'col2', 'col3'],
            metric_names=['new_metric1']
        )
        imported_over_table: SqlaTable = SqlaTable.import_from_dict(
            dict_rep=dict_table_over,
            sync=['metrics', 'columns']
        )
        db.session.commit()
        imported_over: SqlaTable = self.get_table_by_id(imported_over_table.id)
        assert imported_table.id == imported_over.id
        expected_table, _ = self.create_table(
            'table_override',
            id=ID_PREFIX + 3,
            metric_names=['new_metric1'],
            cols_names=['new_col1', 'col2', 'col3'],
            cols_uuids=[col.uuid for col in imported_over.columns]
        )
        self.assert_table_equals(expected_table, imported_over)
        self.yaml_compare(expected_table.export_to_dict(), imported_over.export_to_dict())

    def test_import_table_override_identical(self) -> None:
        table, dict_table = self.create_table(
            'copy_cat',
            id=ID_PREFIX + 4,
            cols_names=['new_col1', 'col2', 'col3'],
            metric_names=['new_metric1']
        )
        imported_table: SqlaTable = SqlaTable.import_from_dict(dict_table)
        db.session.commit()
        copy_table, dict_copy_table = self.create_table(
            'copy_cat',
            id=ID_PREFIX + 4,
            cols_names=['new_col1', 'col2', 'col3'],
            metric_names=['new_metric1']
        )
        imported_copy_table: SqlaTable = SqlaTable.import_from_dict(dict_copy_table)
        db.session.commit()
        assert imported_table.id == imported_copy_table.id
        self.assert_table_equals(copy_table, self.get_table_by_id(imported_table.id))
        self.yaml_compare(imported_copy_table.export_to_dict(), imported_table.export_to_dict())


if __name__ == '__main__':
    unittest.main()
