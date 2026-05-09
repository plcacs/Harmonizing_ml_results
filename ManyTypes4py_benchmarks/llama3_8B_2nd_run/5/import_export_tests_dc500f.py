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
def clean_imports() -> None:
    yield
    delete_imports()

class TestImportExport(SupersetTestCase):
    """Testing export import functionality for dashboards"""

    def create_slice(self, name: str, ds_id: int | None, id: int | None, db_name: str = 'examples', table_name: str = 'wb_health_population', schema: str | None = None) -> Slice:
        params = {'num_period_compare': '10', 'remote_id': id, 'datasource_name': table_name, 'database_name': db_name, 'schema': schema, 'metrics': ['sum__signup_attempt_email', 'sum__signup_attempt_facebook']}
        if table_name and (not ds_id):
            table = self.get_table(schema=schema, name=table_name)
            if table:
                ds_id = table.id
        return Slice(slice_name=name, datasource_type=DatasourceType.TABLE, viz_type='bubble', params=json.dumps(params), datasource_id=ds_id, id=id)

    def create_dashboard(self, title: str, id: int = 0, slcs: list[Slice] = []) -> Dashboard:
        json_metadata = {'remote_id': id}
        return Dashboard(id=id, dashboard_title=title, slices=slcs, position_json='{"size_y": 2, "size_x": 2}', slug=f'{title.lower()}_imported', json_metadata=json.dumps(json_metadata), published=False)

    def create_table(self, name: str, schema: str | None = None, id: int = 0, cols_names: list[str] = [], metric_names: list[str] = []) -> SqlaTable:
        params = {'remote_id': id, 'database_name': 'examples'}
        table = SqlaTable(id=id, schema=schema, table_name=name, params=json.dumps(params))
        for col_name in cols_names:
            table.columns.append(TableColumn(column_name=col_name))
        for metric_name in metric_names:
            table.metrics.append(SqlMetric(metric_name=metric_name, expression=''))
        return table

    # ... rest of the code
