#!/usr/bin/env python3
"""Unit tests for Superset"""
import unittest
from typing import Optional, List, Generator, Any
import pytest
from flask import g
from sqlalchemy.orm.session import make_transient
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.energy_dashboard import load_energy_table_with_slice, load_energy_table_data
from tests.integration_tests.test_app import app
from superset.commands.dashboard.importers.v0 import decode_dashboards, import_chart, import_dashboard
from superset import db, security_manager
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
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
    with app.app_context():
        for slc in db.session.query(Slice):
            if 'remote_id' in slc.params_dict:
                db.session.delete(slc)
        for dash in db.session.query(Dashboard):
            if 'remote_id' in dash.params_dict:
                db.session.delete(dash)
        for table in db.session.query(SqlaTable):
            if 'remote_id' in table.params_dict:
                db.session.delete(table)
        db.session.commit()


@pytest.fixture(autouse=True, scope='module')
def clean_imports() -> Generator[None, None, None]:
    yield
    delete_imports()


class TestImportExport(SupersetTestCase):
    """Testing export import functionality for dashboards"""

    def create_slice(
        self,
        name: str,
        ds_id: Optional[int] = None,
        id: Optional[int] = None,
        db_name: str = 'examples',
        table_name: str = 'wb_health_population',
        schema: Optional[str] = None
    ) -> Slice:
        params: dict = {
            'num_period_compare': '10',
            'remote_id': id,
            'datasource_name': table_name,
            'database_name': db_name,
            'schema': schema,
            'metrics': ['sum__signup_attempt_email', 'sum__signup_attempt_facebook']
        }
        if table_name and (not ds_id):
            table: Optional[SqlaTable] = self.get_table(schema=schema, name=table_name)
            if table:
                ds_id = table.id
        return Slice(
            slice_name=name,
            datasource_type=DatasourceType.TABLE,
            viz_type='bubble',
            params=json.dumps(params),
            datasource_id=ds_id,
            id=id
        )

    def create_dashboard(
        self,
        title: str,
        id: int = 0,
        slcs: List[Slice] = []
    ) -> Dashboard:
        json_metadata: dict = {'remote_id': id}
        return Dashboard(
            id=id,
            dashboard_title=title,
            slices=slcs,
            position_json='{"size_y": 2, "size_x": 2}',
            slug=f'{title.lower()}_imported',
            json_metadata=json.dumps(json_metadata),
            published=False
        )

    def create_table(
        self,
        name: str,
        schema: Optional[str] = None,
        id: int = 0,
        cols_names: List[str] = [],
        metric_names: List[str] = []
    ) -> SqlaTable:
        params: dict = {'remote_id': id, 'database_name': 'examples'}
        table: SqlaTable = SqlaTable(id=id, schema=schema, table_name=name, params=json.dumps(params))
        for col_name in cols_names:
            table.columns.append(TableColumn(column_name=col_name))
        for metric_name in metric_names:
            table.metrics.append(SqlMetric(metric_name=metric_name, expression=''))
        return table

    def get_slice(self, slc_id: int) -> Optional[Slice]:
        return db.session.query(Slice).filter_by(id=slc_id).first()

    def get_slice_by_name(self, name: str) -> Optional[Slice]:
        return db.session.query(Slice).filter_by(slice_name=name).first()

    def get_dash(self, dash_id: int) -> Optional[Dashboard]:
        return db.session.query(Dashboard).filter_by(id=dash_id).first()

    def assert_dash_equals(
        self,
        expected_dash: Dashboard,
        actual_dash: Dashboard,
        check_position: bool = True,
        check_slugs: bool = True
    ) -> None:
        if check_slugs:
            assert expected_dash.slug == actual_dash.slug
        assert expected_dash.dashboard_title == actual_dash.dashboard_title
        assert len(expected_dash.slices) == len(actual_dash.slices)
        expected_slices: List[Slice] = sorted(expected_dash.slices, key=lambda s: s.slice_name or '')
        actual_slices: List[Slice] = sorted(actual_dash.slices, key=lambda s: s.slice_name or '')
        for e_slc, a_slc in zip(expected_slices, actual_slices, strict=False):
            self.assert_slice_equals(e_slc, a_slc)
        if check_position:
            assert expected_dash.position_json == actual_dash.position_json

    def assert_table_equals(
        self,
        expected_ds: SqlaTable,
        actual_ds: SqlaTable
    ) -> None:
        assert expected_ds.table_name == actual_ds.table_name
        assert expected_ds.main_dttm_col == actual_ds.main_dttm_col
        assert expected_ds.schema == actual_ds.schema
        assert len(expected_ds.metrics) == len(actual_ds.metrics)
        assert len(expected_ds.columns) == len(actual_ds.columns)
        assert {c.column_name for c in expected_ds.columns} == {c.column_name for c in actual_ds.columns}
        assert {m.metric_name for m in expected_ds.metrics} == {m.metric_name for m in actual_ds.metrics}

    def assert_datasource_equals(
        self,
        expected_ds: Any,
        actual_ds: Any
    ) -> None:
        assert expected_ds.datasource_name == actual_ds.datasource_name
        assert expected_ds.main_dttm_col == actual_ds.main_dttm_col
        assert len(expected_ds.metrics) == len(actual_ds.metrics)
        assert len(expected_ds.columns) == len(actual_ds.columns)
        assert {c.column_name for c in expected_ds.columns} == {c.column_name for c in actual_ds.columns}
        assert {m.metric_name for m in expected_ds.metrics} == {m.metric_name for m in actual_ds.metrics}

    def assert_slice_equals(
        self,
        expected_slc: Slice,
        actual_slc: Slice
    ) -> None:
        expected_slc_name: str = expected_slc.slice_name or ''
        actual_slc_name: str = actual_slc.slice_name or ''
        assert expected_slc_name == actual_slc_name
        assert expected_slc.datasource_type == actual_slc.datasource_type
        assert expected_slc.viz_type == actual_slc.viz_type
        exp_params: dict = json.loads(expected_slc.params)
        actual_params: dict = json.loads(actual_slc.params)
        diff_params_keys = ('schema', 'database_name', 'datasource_name', 'remote_id', 'import_time')
        for k in diff_params_keys:
            if k in actual_params:
                actual_params.pop(k)
            if k in exp_params:
                exp_params.pop(k)
        assert exp_params == actual_params

    def assert_only_exported_slc_fields(
        self,
        expected_dash: Dashboard,
        actual_dash: Dashboard
    ) -> None:
        """only exported json has this params
        imported/created dashboard has relationships to other models instead
        """
        expected_slices: List[Slice] = sorted(expected_dash.slices, key=lambda s: s.slice_name or '')
        actual_slices: List[Slice] = sorted(actual_dash.slices, key=lambda s: s.slice_name or '')
        for e_slc, a_slc in zip(expected_slices, actual_slices, strict=False):
            params: dict = a_slc.params_dict
            assert e_slc.datasource.name == params['datasource_name']
            assert e_slc.datasource.schema == params['schema']
            assert e_slc.datasource.database.name == params['database_name']

    @unittest.skip('Schema needs to be updated')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_export_1_dashboard(self) -> None:
        self.login(ADMIN_USERNAME)
        birth_dash: Dashboard = self.get_dash_by_slug('births')  # type: ignore
        id_: int = birth_dash.id
        export_dash_url: str = f'/dashboard/export_dashboards_form?id={id_}&action=go'
        resp = self.client.get(export_dash_url)
        exported_dashboards: List[Any] = json.loads(resp.data.decode('utf-8'), object_hook=decode_dashboards)['dashboards']
        birth_dash = self.get_dash_by_slug('births')  # type: ignore
        self.assert_only_exported_slc_fields(birth_dash, exported_dashboards[0])
        self.assert_dash_equals(birth_dash, exported_dashboards[0])
        assert id_ == json.loads(exported_dashboards[0].json_metadata, object_hook=decode_dashboards)['remote_id']
        exported_tables: List[Any] = json.loads(resp.data.decode('utf-8'), object_hook=decode_dashboards)['datasources']
        assert 1 == len(exported_tables)
        self.assert_table_equals(self.get_table(name='birth_names'), exported_tables[0])  # type: ignore

    @unittest.skip('Schema needs to be updated')
    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices', 'load_birth_names_dashboard_with_slices')
    def test_export_2_dashboards(self) -> None:
        self.login(ADMIN_USERNAME)
        birth_dash: Dashboard = self.get_dash_by_slug('births')  # type: ignore
        world_health_dash: Dashboard = self.get_dash_by_slug('world_health')  # type: ignore
        export_dash_url: str = '/dashboard/export_dashboards_form?id={}&id={}&action=go'.format(birth_dash.id, world_health_dash.id)
        resp = self.client.get(export_dash_url)
        resp_data: dict = json.loads(resp.data.decode('utf-8'), object_hook=decode_dashboards)
        exported_dashboards: List[Any] = sorted(resp_data.get('dashboards'), key=lambda d: d.dashboard_title)
        assert 2 == len(exported_dashboards)
        birth_dash = self.get_dash_by_slug('births')  # type: ignore
        self.assert_only_exported_slc_fields(birth_dash, exported_dashboards[0])
        self.assert_dash_equals(birth_dash, exported_dashboards[0])
        assert birth_dash.id == json.loads(exported_dashboards[0].json_metadata)['remote_id']
        world_health_dash = self.get_dash_by_slug('world_health')  # type: ignore
        self.assert_only_exported_slc_fields(world_health_dash, exported_dashboards[1])
        self.assert_dash_equals(world_health_dash, exported_dashboards[1])
        assert world_health_dash.id == json.loads(exported_dashboards[1].json_metadata)['remote_id']
        exported_tables: List[Any] = sorted(resp_data.get('datasources'), key=lambda t: t.table_name)
        assert 2 == len(exported_tables)
        self.assert_table_equals(self.get_table(name='birth_names'), exported_tables[0])  # type: ignore
        self.assert_table_equals(self.get_table(name='wb_health_population'), exported_tables[1])  # type: ignore

    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_import_1_slice(self) -> None:
        expected_slice: Slice = self.create_slice('Import Me', id=10001, schema=get_example_default_schema())
        slc_id: int = import_chart(expected_slice, None, import_time=1989)  # type: ignore
        slc: Optional[Slice] = self.get_slice(slc_id)
        assert slc is not None
        assert slc.datasource.perm == slc.perm
        self.assert_slice_equals(expected_slice, slc)
        table_id: int = self.get_table(name='wb_health_population').id  # type: ignore
        assert table_id == self.get_slice(slc_id).datasource_id

    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_import_2_slices_for_same_table(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        table_id: int = self.get_table(name='wb_health_population').id  # type: ignore
        slc_1: Slice = self.create_slice('Import Me 1', ds_id=table_id, id=10002, schema=schema)
        slc_id_1: int = import_chart(slc_1, None)  # type: ignore
        slc_2: Slice = self.create_slice('Import Me 2', ds_id=table_id, id=10003, schema=schema)
        slc_id_2: int = import_chart(slc_2, None)  # type: ignore
        imported_slc_1: Optional[Slice] = self.get_slice(slc_id_1)
        imported_slc_2: Optional[Slice] = self.get_slice(slc_id_2)
        assert imported_slc_1 is not None and imported_slc_2 is not None
        assert table_id == imported_slc_1.datasource_id
        self.assert_slice_equals(slc_1, imported_slc_1)
        assert imported_slc_1.datasource.perm == imported_slc_1.perm
        assert table_id == imported_slc_2.datasource_id
        self.assert_slice_equals(slc_2, imported_slc_2)
        assert imported_slc_2.datasource.perm == imported_slc_2.perm

    def test_import_slices_override(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        slc: Slice = self.create_slice('Import Me New', id=10005, schema=schema)
        slc_1_id: int = import_chart(slc, None, import_time=1990)  # type: ignore
        imported_slc_1: Optional[Slice] = self.get_slice(slc_1_id)
        assert imported_slc_1 is not None
        slc.slice_name = 'Import Me New'
        slc_2: Slice = self.create_slice('Import Me New', id=10005, schema=schema)
        slc_2_id: int = import_chart(slc_2, imported_slc_1, import_time=1990)  # type: ignore
        assert slc_1_id == slc_2_id
        imported_slc_2: Optional[Slice] = self.get_slice(slc_2_id)
        assert imported_slc_2 is not None
        self.assert_slice_equals(slc, imported_slc_2)

    def test_import_empty_dashboard(self) -> None:
        empty_dash: Dashboard = self.create_dashboard('empty_dashboard', id=10001)
        imported_dash_id: int = import_dashboard(empty_dash, import_time=1989)  # type: ignore
        imported_dash: Optional[Dashboard] = self.get_dash(imported_dash_id)
        assert imported_dash is not None
        self.assert_dash_equals(empty_dash, imported_dash, check_position=False)

    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_import_dashboard_1_slice(self) -> None:
        slc: Slice = self.create_slice('health_slc', id=10006, schema=get_example_default_schema())
        dash_with_1_slice: Dashboard = self.create_dashboard('dash_with_1_slice', slcs=[slc], id=10002)
        dash_with_1_slice.position_json = '\n            {{"DASHBOARD_VERSION_KEY": "v2",\n              "DASHBOARD_CHART_TYPE-{0}": {{\n                "type": "CHART",\n                "id": {0},\n                "children": [],\n                "meta": {{\n                  "width": 4,\n                  "height": 50,\n                  "chartId": {0}\n                }}\n              }}\n            }}\n        '.format(slc.id)
        imported_dash_id: int = import_dashboard(dash_with_1_slice, import_time=1990)  # type: ignore
        imported_dash: Optional[Dashboard] = self.get_dash(imported_dash_id)
        assert imported_dash is not None
        expected_dash: Dashboard = self.create_dashboard('dash_with_1_slice', slcs=[slc], id=10002)
        make_transient(expected_dash)
        self.assert_dash_equals(expected_dash, imported_dash, check_position=False, check_slugs=False)
        assert {'remote_id': 10002, 'import_time': 1990, 'native_filter_configuration': []} == json.loads(imported_dash.json_metadata)
        expected_position: dict = dash_with_1_slice.position
        meta: dict = expected_position['DASHBOARD_CHART_TYPE-10006']['meta']
        meta['chartId'] = imported_dash.slices[0].id
        assert expected_position == imported_dash.position

    @pytest.mark.usefixtures('load_energy_table_with_slice')
    def test_import_dashboard_2_slices(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        e_slc: Slice = self.create_slice('e_slc', id=10007, table_name='energy_usage', schema=schema)
        b_slc: Slice = self.create_slice('b_slc', id=10008, table_name='birth_names', schema=schema)
        dash_with_2_slices: Dashboard = self.create_dashboard('dash_with_2_slices', slcs=[e_slc, b_slc], id=10003)
        dash_with_2_slices.json_metadata = json.dumps({
            'remote_id': 10003,
            'expanded_slices': {f'{e_slc.id}': True, f'{b_slc.id}': False},
            'filter_scopes': {str(e_slc.id): {'region': {'scope': ['ROOT_ID'], 'immune': [b_slc.id]}}}
        })
        imported_dash_id: int = import_dashboard(dash_with_2_slices, import_time=1991)  # type: ignore
        imported_dash: Optional[Dashboard] = self.get_dash(imported_dash_id)
        assert imported_dash is not None
        expected_dash: Dashboard = self.create_dashboard('dash_with_2_slices', slcs=[e_slc, b_slc], id=10003)
        make_transient(expected_dash)
        self.assert_dash_equals(imported_dash, expected_dash, check_position=False, check_slugs=False)
        i_e_slc: Optional[Slice] = self.get_slice_by_name('e_slc')
        i_b_slc: Optional[Slice] = self.get_slice_by_name('b_slc')
        assert i_e_slc is not None and i_b_slc is not None
        expected_json_metadata: dict = {
            'remote_id': 10003,
            'import_time': 1991,
            'expanded_slices': {f'{i_e_slc.id}': True, f'{i_b_slc.id}': False},
            'native_filter_configuration': []
        }
        assert expected_json_metadata == json.loads(imported_dash.json_metadata)

    @pytest.mark.usefixtures('load_energy_table_with_slice')
    def test_import_override_dashboard_2_slices(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        e_slc: Slice = self.create_slice('e_slc', id=10009, table_name='energy_usage', schema=schema)
        b_slc: Slice = self.create_slice('b_slc', id=10010, table_name='birth_names', schema=schema)
        dash_to_import: Dashboard = self.create_dashboard('override_dashboard', slcs=[e_slc, b_slc], id=10004)
        imported_dash_id_1: int = import_dashboard(dash_to_import, import_time=1992)  # type: ignore
        e_slc = self.create_slice('e_slc', id=10009, table_name='energy_usage', schema=schema)
        b_slc = self.create_slice('b_slc', id=10010, table_name='birth_names', schema=schema)
        c_slc: Slice = self.create_slice('c_slc', id=10011, table_name='birth_names', schema=schema)
        dash_to_import_override: Dashboard = self.create_dashboard('override_dashboard_new', slcs=[e_slc, b_slc, c_slc], id=10004)
        imported_dash_id_2: int = import_dashboard(dash_to_import_override, import_time=1992)  # type: ignore
        assert imported_dash_id_1 == imported_dash_id_2
        expected_dash: Dashboard = self.create_dashboard('override_dashboard_new', slcs=[e_slc, b_slc, c_slc], id=10004)
        make_transient(expected_dash)
        imported_dash: Optional[Dashboard] = self.get_dash(imported_dash_id_2)
        assert imported_dash is not None
        self.assert_dash_equals(expected_dash, imported_dash, check_position=False, check_slugs=False)
        assert {'remote_id': 10004, 'import_time': 1992, 'native_filter_configuration': []} == json.loads(imported_dash.json_metadata)

    def test_import_new_dashboard_slice_reset_ownership(self) -> None:
        admin_user: Any = security_manager.find_user(username='admin')
        assert admin_user
        gamma_user: Any = security_manager.find_user(username='gamma')
        assert gamma_user
        g.user = gamma_user
        dash_with_1_slice: Dashboard = self._create_dashboard_for_import(id_=10200)
        dash_with_1_slice.created_by = admin_user
        dash_with_1_slice.changed_by = admin_user
        dash_with_1_slice.owners = [admin_user]
        imported_dash_id: int = import_dashboard(dash_with_1_slice)  # type: ignore
        imported_dash: Optional[Dashboard] = self.get_dash(imported_dash_id)
        assert imported_dash is not None
        assert imported_dash.created_by == gamma_user
        assert imported_dash.changed_by == gamma_user
        assert imported_dash.owners == [gamma_user]
        imported_slc: Slice = imported_dash.slices[0]
        assert imported_slc.created_by == gamma_user
        assert imported_slc.changed_by == gamma_user
        assert imported_slc.owners == [gamma_user]

    @pytest.mark.skip
    def test_import_override_dashboard_slice_reset_ownership(self) -> None:
        admin_user: Any = security_manager.find_user(username='admin')
        assert admin_user
        gamma_user: Any = security_manager.find_user(username='gamma')
        assert gamma_user
        g.user = gamma_user
        dash_with_1_slice: Dashboard = self._create_dashboard_for_import(id_=10300)
        imported_dash_id: int = import_dashboard(dash_with_1_slice)  # type: ignore
        imported_dash: Optional[Dashboard] = self.get_dash(imported_dash_id)
        assert imported_dash is not None
        assert imported_dash.created_by == gamma_user
        assert imported_dash.changed_by == gamma_user
        assert imported_dash.owners == [gamma_user]
        imported_slc: Slice = imported_dash.slices[0]
        assert imported_slc.created_by == gamma_user
        assert imported_slc.changed_by == gamma_user
        assert imported_slc.owners == [gamma_user]
        g.user = admin_user
        dash_with_1_slice = self._create_dashboard_for_import(id_=10300)
        imported_dash_id = import_dashboard(dash_with_1_slice)  # type: ignore
        imported_dash = self.get_dash(imported_dash_id)
        assert imported_dash is not None
        assert imported_dash.created_by == gamma_user
        assert imported_dash.changed_by == gamma_user
        assert imported_dash.owners == [gamma_user]
        imported_slc = imported_dash.slices[0]
        assert imported_slc.created_by == gamma_user
        assert imported_slc.changed_by == gamma_user
        assert imported_slc.owners == [gamma_user]

    def _create_dashboard_for_import(self, id_: int) -> Dashboard:
        slc: Slice = self.create_slice('health_slc' + str(id_), id=id_ + 1, schema=get_example_default_schema())
        dash_with_1_slice: Dashboard = self.create_dashboard('dash_with_1_slice' + str(id_), slcs=[slc], id=id_ + 2)
        dash_with_1_slice.position_json = '\n                {{"DASHBOARD_VERSION_KEY": "v2",\n                "DASHBOARD_CHART_TYPE-{0}": {{\n                    "type": "CHART",\n                    "id": {0},\n                    "children": [],\n                    "meta": {{\n                    "width": 4,\n                    "height": 50,\n                    "chartId": {0}\n                    }}\n                }}\n                }}\n            '.format(slc.id)
        return dash_with_1_slice

    def test_import_table_no_metadata(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        db_id: int = get_example_database().id
        table: SqlaTable = self.create_table('pure_table', id=10001, schema=schema)
        imported_id: int = import_dataset(table, db_id, import_time=1989)  # type: ignore
        imported: Optional[SqlaTable] = self.get_table_by_id(imported_id)
        assert imported is not None
        self.assert_table_equals(table, imported)

    def test_import_table_1_col_1_met(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        table: SqlaTable = self.create_table('table_1_col_1_met', id=10002, cols_names=['col1'], metric_names=['metric1'], schema=schema)
        db_id: int = get_example_database().id
        imported_id: int = import_dataset(table, db_id, import_time=1990)  # type: ignore
        imported: Optional[SqlaTable] = self.get_table_by_id(imported_id)
        assert imported is not None
        self.assert_table_equals(table, imported)
        assert {'remote_id': 10002, 'import_time': 1990, 'database_name': 'examples'} == json.loads(imported.params)

    def test_import_table_2_col_2_met(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        table: SqlaTable = self.create_table('table_2_col_2_met', id=10003, cols_names=['c1', 'c2'], metric_names=['m1', 'm2'], schema=schema)
        db_id: int = get_example_database().id
        imported_id: int = import_dataset(table, db_id, import_time=1991)  # type: ignore
        imported: Optional[SqlaTable] = self.get_table_by_id(imported_id)
        assert imported is not None
        self.assert_table_equals(table, imported)

    def test_import_table_override(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        table: SqlaTable = self.create_table('table_override', id=10003, cols_names=['col1'], metric_names=['m1'], schema=schema)
        db_id: int = get_example_database().id
        imported_id: int = import_dataset(table, db_id, import_time=1991)  # type: ignore
        table_over: SqlaTable = self.create_table('table_override', id=10003, cols_names=['new_col1', 'col2', 'col3'], metric_names=['new_metric1'], schema=schema)
        imported_over_id: int = import_dataset(table_over, db_id, import_time=1992)  # type: ignore
        imported_over: Optional[SqlaTable] = self.get_table_by_id(imported_over_id)
        assert imported_over is not None
        assert imported_id == imported_over.id
        expected_table: SqlaTable = self.create_table('table_override', id=10003, metric_names=['new_metric1', 'm1'], cols_names=['col1', 'new_col1', 'col2', 'col3'], schema=schema)
        self.assert_table_equals(expected_table, imported_over)

    def test_import_table_override_identical(self) -> None:
        schema: Optional[str] = get_example_default_schema()
        table: SqlaTable = self.create_table('copy_cat', id=10004, cols_names=['new_col1', 'col2', 'col3'], metric_names=['new_metric1'], schema=schema)
        db_id: int = get_example_database().id
        imported_id: int = import_dataset(table, db_id, import_time=1993)  # type: ignore
        copy_table: SqlaTable = self.create_table('copy_cat', id=10004, cols_names=['new_col1', 'col2', 'col3'], metric_names=['new_metric1'], schema=schema)
        imported_id_copy: int = import_dataset(copy_table, db_id, import_time=1994)  # type: ignore
        assert imported_id == imported_id_copy
        imported: Optional[SqlaTable] = self.get_table_by_id(imported_id)
        assert imported is not None
        self.assert_table_equals(copy_table, imported)


if __name__ == '__main__':
    unittest.main()