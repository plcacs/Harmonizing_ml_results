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

KEY: str = 'test-key'
INITIAL_FORM_DATA: str = json.dumps({'test': 'initial value'})
UPDATED_FORM_DATA: str = json.dumps({'test': 'updated value'})

def test_post(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_bad_request_non_string(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_bad_request_non_json_string(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_access_denied(test_client: FlaskClient, login_as: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_same_key_for_same_context(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_different_key_for_different_context(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_same_key_for_same_tab_id(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_different_key_for_different_tab_id(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_post_different_key_for_no_tab_id(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_same_key_for_same_tab_id(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_different_key_for_different_tab_id(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_different_key_for_no_tab_id(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_bad_request(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_bad_request_non_string(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_bad_request_non_json_string(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_access_denied(test_client: FlaskClient, login_as: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_put_not_owner(test_client: FlaskClient, login_as: callable, chart_id: int, datasource: SqlaTable) -> None:
def test_get_key_not_found(test_client: FlaskClient, login_as_admin: callable) -> None:
def test_get(test_client: FlaskClient, login_as_admin: callable) -> None:
def test_get_access_denied(test_client: FlaskClient, login_as: callable) -> None:
def test_get_dataset_access_denied(mock_can_access_datasource: callable, test_client: FlaskClient, login_as_admin: callable) -> None:
def test_delete(test_client: FlaskClient, login_as_admin: callable) -> None:
def test_delete_access_denied(test_client: FlaskClient, login_as: callable) -> None:
def test_delete_not_owner(test_client: FlaskClient, login_as_admin: callable, chart_id: int, datasource: SqlaTable, admin_id: int) -> None:
