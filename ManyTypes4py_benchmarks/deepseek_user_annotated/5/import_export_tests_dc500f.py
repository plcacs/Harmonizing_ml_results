# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# isort:skip_file
"""Unit tests for Superset"""

import unittest
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,  # noqa: F401
    load_birth_names_data,  # noqa: F401
)

import pytest
from flask import g
from sqlalchemy.orm.session import make_transient

from tests.integration_tests.fixtures.energy_dashboard import (
    load_energy_table_with_slice,  # noqa: F401
    load_energy_table_data,  # noqa: F401
)
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
from tests.integration_tests.fixtures.world_bank_dashboard import (
    load_world_bank_dashboard_with_slices,  # noqa: F401
    load_world_bank_data,  # noqa: F401
)


def delete_imports() -> None:
    with app.app_context():
        # Imported data clean up
        for slc in db.session.query(Slice):
            if "remote_id" in slc.params_dict:
                db.session.delete(slc)
        for dash in db.session.query(Dashboard):
            if "remote_id" in dash.params_dict:
                db.session.delete(dash)
        for table in db.session.query(SqlaTable):
            if "remote_id" in table.params_dict:
                db.session.delete(table)
        db.session.commit()


@pytest.fixture(autouse=True, scope="module")
def clean_imports() -> None:
    yield
    delete_imports()


class TestImportExport(SupersetTestCase):
    """Testing export import functionality for dashboards"""

    def create_slice(
        self,
        name: str,
        ds_id: Optional[int] = None,
        id: Optional[int] = None,
        db_name: str = "examples",
        table_name: str = "wb_health_population",
        schema: Optional[str] = None,
    ) -> Slice:
        params = {
            "num_period_compare": "10",
            "remote_id": id,
            "datasource_name": table_name,
            "database_name": db_name,
            "schema": schema,
            # Test for trailing commas
            "metrics": ["sum__signup_attempt_email", "sum__signup_attempt_facebook"],
        }

        if table_name and not ds_id:
            table = self.get_table(schema=schema, name=table_name)
            if table:
                ds_id = table.id

        return Slice(
            slice_name=name,
            datasource_type=DatasourceType.TABLE,
            viz_type="bubble",
            params=json.dumps(params),
            datasource_id=ds_id,
            id=id,
        )

    def create_dashboard(
        self, title: str, id: int = 0, slcs: List[Slice] = []
    ) -> Dashboard:
        json_metadata = {"remote_id": id}
        return Dashboard(
            id=id,
            dashboard_title=title,
            slices=slcs,
            position_json='{"size_y": 2, "size_x": 2}',
            slug=f"{title.lower()}_imported",
            json_metadata=json.dumps(json_metadata),
            published=False,
        )

    def create_table(
        self,
        name: str,
        schema: Optional[str] = None,
        id: int = 0,
        cols_names: List[str] = [],
        metric_names: List[str] = [],
    ) -> SqlaTable:
        params = {"remote_id": id, "database_name": "examples"}
        table = SqlaTable(
            id=id,
            schema=schema,
            table_name=name,
            params=json.dumps(params),
        )
        for col_name in cols_names:
            table.columns.append(TableColumn(column_name=col_name))
        for metric_name in metric_names:
            table.metrics.append(SqlMetric(metric_name=metric_name, expression=""))
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
        check_slugs: bool = True,
    ) -> None:
        if check_slugs:
            assert expected_dash.slug == actual_dash.slug
        assert expected_dash.dashboard_title == actual_dash.dashboard_title
        assert len(expected_dash.slices) == len(actual_dash.slices)
        expected_slices = sorted(expected_dash.slices, key=lambda s: s.slice_name or "")
        actual_slices = sorted(actual_dash.slices, key=lambda s: s.slice_name or "")
        for e_slc, a_slc in zip(expected_slices, actual_slices, strict=False):
            self.assert_slice_equals(e_slc, a_slc)
        if check_position:
            assert expected_dash.position_json == actual_dash.position_json

    def assert_table_equals(self, expected_ds: SqlaTable, actual_ds: SqlaTable) -> None:
        assert expected_ds.table_name == actual_ds.table_name
        assert expected_ds.main_dttm_col == actual_ds.main_dttm_col
        assert expected_ds.schema == actual_ds.schema
        assert len(expected_ds.metrics) == len(actual_ds.metrics)
        assert len(expected_ds.columns) == len(actual_ds.columns)
        assert {c.column_name for c in expected_ds.columns} == {
            c.column_name for c in actual_ds.columns
        }
        assert {m.metric_name for m in expected_ds.metrics} == {
            m.metric_name for m in actual_ds.metrics
        }

    def assert_datasource_equals(
        self, expected_ds: SqlaTable, actual_ds: SqlaTable
    ) -> None:
        assert expected_ds.datasource_name == actual_ds.datasource_name
        assert expected_ds.main_dttm_col == actual_ds.main_dttm_col
        assert len(expected_ds.metrics) == len(actual_ds.metrics)
        assert len(expected_ds.columns) == len(actual_ds.columns)
        assert {c.column_name for c in expected_ds.columns} == {
            c.column_name for c in actual_ds.columns
        }
        assert {m.metric_name for m in expected_ds.metrics} == {
            m.metric_name for m in actual_ds.metrics
        }

    def assert_slice_equals(self, expected_slc: Slice, actual_slc: Slice) -> None:
        expected_slc_name = expected_slc.slice_name or ""
        actual_slc_name = actual_slc.slice_name or ""
        assert expected_slc_name == actual_slc_name
        assert expected_slc.datasource_type == actual_slc.datasource_type
        assert expected_slc.viz_type == actual_slc.viz_type
        exp_params = json.loads(expected_slc.params)
        actual_params = json.loads(actual_slc.params)
        diff_params_keys = (
            "schema",
            "database_name",
            "datasource_name",
            "remote_id",
            "import_time",
        )
        for k in diff_params_keys:
            if k in actual_params:
                actual_params.pop(k)
            if k in exp_params:
                exp_params.pop(k)
        assert exp_params == actual_params

    def assert_only_exported_slc_fields(
        self, expected_dash: Dashboard, actual_dash: Dashboard
    ) -> None:
        expected_slices = sorted(expected_dash.slices, key=lambda s: s.slice_name or "")
        actual_slices = sorted(actual_dash.slices, key=lambda s: s.slice_name or "")
        for e_slc, a_slc in zip(expected_slices, actual_slices, strict=False):
            params = a_slc.params_dict
            assert e_slc.datasource.name == params["datasource_name"]
            assert e_slc.datasource.schema == params["schema"]
            assert e_slc.datasource.database.name == params["database_name"]

    @unittest.skip("Schema needs to be updated")
    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_export_1_dashboard(self) -> None:
        self.login(ADMIN_USERNAME)
        birth_dash = self.get_dash_by_slug("births")
        id_ = birth_dash.id
        export_dash_url = f"/dashboard/export_dashboards_form?id={id_}&action=go"
        resp = self.client.get(export_dash_url)
        exported_dashboards = json.loads(
            resp.data.decode("utf-8"), object_hook=decode_dashboards
        )["dashboards"]

        birth_dash = self.get_dash_by_slug("births")
        self.assert_only_exported_slc_fields(birth_dash, exported_dashboards[0])
        self.assert_dash_equals(birth_dash, exported_dashboards[0])
        assert (
            id_
            == json.loads(
                exported_dashboards[0].json_metadata, object_hook=decode_dashboards
            )["remote_id"]
        )

        exported_tables = json.loads(
            resp.data.decode("utf-8"), object_hook=decode_dashboards
        )["datasources"]
        assert 1 == len(exported_tables)
        self.assert_table_equals(self.get_table(name="birth_names"), exported_tables[0])

    @unittest.skip("Schema needs to be updated")
    @pytest.mark.usefixtures(
        "load_world_bank_dashboard_with_slices",
        "load_birth_names_dashboard_with_slices",
    )
    def test_export_2_dashboards(self) -> None:
        self.login(ADMIN_USERNAME)
        birth_dash = self.get_dash_by_slug("births")
        world_health_dash = self.get_dash_by_slug("world_health")
        export_dash_url = (
            "/dashboard/export_dashboards_form?id={}&id={}&action=go".format(
                birth_dash.id, world_health_dash.id
            )
        )
        resp = self.client.get(export_dash_url)
        resp_data = json.loads(resp.data.decode("utf-8"), object_hook=decode_dashboards)
        exported_dashboards = sorted(
            resp_data.get("dashboards"), key=lambda d: d.dashboard_title
        )
        assert 2 == len(exported_dashboards)

        birth_dash = self.get_dash_by_slug("births")
        self.assert_only_exported_slc_fields(birth_dash, exported_dashboards[0])
        self.assert_dash_equals(birth_dash, exported_dashboards[0])
        assert (
            birth_dash.id
            == json.loads(exported_dashboards[0].json_metadata)["remote_id"]
        )

        world_health_dash = self.get_dash_by_slug("world_health")
        self.assert_only_exported_slc_fields(world_health_dash, exported_dashboards[1])
        self.assert_dash_equals(world_health_dash, exported_dashboards[1])
        assert (
            world_health_dash.id
            == json.loads(exported_dashboards[1].json_metadata)["remote_id"]
        )

        exported_tables = sorted(
            resp_data.get("datasources"), key=lambda t: t.table_name
        )
        assert 2 == len(exported_tables)
        self.assert_table_equals(self.get_table(name="birth_names"), exported_tables[0])
        self.assert_table_equals(
            self.get_table(name="wb_health_population"), exported_tables[1]
        )

    @pytest.mark.usefixtures("load_world_bank_dashboard_with_slices")
    def test_import_1_slice(self) -> None:
        expected_slice = self.create_slice(
            "Import Me", id=10001, schema=get_example_default_schema()
        )
        slc_id = import_chart(expected_slice, None, import_time=1989)
        slc = self.get_slice(slc_id)
        assert slc.datasource.perm == slc.perm
        self.assert_slice_equals(expected_slice, slc)

        table_id = self.get_table(name="wb_health_population").id
        assert table_id == self.get_slice(slc_id).datasource_id

    @pytest.mark.usefixtures("load_world_bank_dashboard_with_slices")
    def test_import_2_slices_for_same_table(self) -> None:
        schema = get_example_default_schema()
        table_id = self.get_table(name="wb_health_population").id
        slc_1 = self.create_slice(
            "Import Me 1", ds_id=table_id, id=10002, schema=schema
        )
        slc_id_1 = import_chart(slc_1, None)
        slc_2 = self.create_slice(
            "Import Me 2", ds_id=table_id, id=10003, schema=schema
        )
        slc_id_2 = import_chart(slc_2, None)

        imported_slc_1 = self.get_slice(slc_id_1)
        imported_slc_2 = self.get_slice(slc_id_2)
        assert table_id == imported_slc_1.datasource_id
        self.assert_slice_equals(slc_1, imported_slc_1)
        assert imported_slc_1.datasource.perm == imported_slc_1.perm

        assert table_id == imported_slc_2.datasource_id
        self.assert_slice_equals(slc_2, imported_slc_2)
        assert imported_slc_2.datasource.perm == imported_slc_2.perm

    def test_import_slices_override(self) -> None:
        schema = get_example_default_schema()
        slc = self.create_slice("Import Me New", id=10005, schema=schema)
        slc_1_id = import_chart(slc, None, import_time=1990)
        slc.slice_name = "Import Me New"
        imported_slc_1 = self.get_slice(slc_1_id)
        slc_2 = self.create_slice("Import Me New", id=10005, schema=schema)
        slc_2_id = import_chart(slc_2, imported_slc_1, import_time=1990)
        assert slc_1_id == slc_2_id
        imported_slc_2 = self.get_slice(slc_2_id)
        self.assert_slice_equals(slc, imported_slc_2)

    def test_import_empty_dashboard(self) -> None:
        empty_dash = self.create_dashboard("empty_dashboard", id=10001)
        imported_dash_id = import_dashboard(empty_dash, import_time=1989)
        imported_dash = self.get_dash(imported_dash_id)
        self.assert_dash_equals(empty_dash, imported_dash, check_position=False)

    @pytest.mark.usefixtures("load_world_bank_dashboard_with_slices")
    def test_import_dashboard_1_slice(self) -> None:
        slc = self.create_slice(
            "health_slc", id=10006, schema=get_example_default_schema()
        )
        dash_with_1_slice = self.create_dashboard(
            "dash_with_1_slice", slcs=[slc], id=10002
        )
        dash_with_1_slice.position_json = """
            {{"DASHBOARD_VERSION_KEY": "v2",
              "DASHBOARD_CHART_TYPE-{0}": {{
                "type": "CHART",
                "id": {0},
                "children": [],
                "meta": {{
                  "width": 4,
                  "height": 50,
                  "chartId": {0}
                }}
              }}
            }}
        """.format(slc.id)
        imported_dash_id = import_dashboard(dash_with_1_slice, import_time=1990)
        imported_dash = self.get_dash(imported_dash_id)

        expected_dash = self.create_dashboard("dash_with_1_slice",