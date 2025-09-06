from flask.testing import FlaskClient
from flask_appbuilder.security.sqla.models import User
from sqlalchemy.orm import Session
from superset import db
from superset.commands.dataset.exceptions import DatasetAccessDeniedError
from superset.commands.explore.form_data.state import TemporaryExploreState
from superset.connectors.sqla.models import SqlaTable
from superset.extensions import cache_manager
from superset.models.slice import Slice
from superset.utils import json
from superset.utils.core import DatasourceType
from tests.integration_tests.fixtures.world_bank_dashboard import load_world_bank_dashboard_with_slices, load_world_bank_data
from tests.integration_tests.test_app import app
from typing import Any, Dict

KEY: str = 'test-key'
INITIAL_FORM_DATA: str = json.dumps({'test': 'initial value'})
UPDATED_FORM_DATA: str = json.dumps({'test': 'updated value'})


def func_4lanrr5k(load_world_bank_dashboard_with_slices: Any) -> int:
    with app.app_context() as ctx:
        chart = db.session.query(Slice).filter_by(slice_name=
            "World's Population").one()
        return chart.id


def func_e3fh8x6d() -> int:
    with app.app_context() as ctx:
        admin = db.session.query(User).filter_by(username='admin').one()
        return admin.id


def func_xcz2ypnh() -> SqlaTable:
    with app.app_context() as ctx:
        dataset = db.session.query(SqlaTable).filter_by(table_name=
            'wb_health_population').first()
        return dataset


def func_mjcaicc2(chart_id: int, admin_id: int, datasource: SqlaTable) -> None:
    entry: Dict[str, Any] = {'owner': admin_id, 'datasource_id': datasource.id,
        'datasource_type': datasource.type, 'chart_id': chart_id,
        'form_data': INITIAL_FORM_DATA}
    cache_manager.explore_form_data_cache.set(KEY, entry)


def func_h16228uc(test_client: FlaskClient, login_as_admin: Any, chart_id: int, datasource: SqlaTable) -> None:
    payload: Dict[str, Any] = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 201

# Add type annotations for the remaining functions as needed
