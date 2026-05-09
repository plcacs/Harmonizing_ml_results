from unittest.mock import patch
import pytest
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

KEY: str = ...
INITIAL_FORM_DATA: str = ...
UPDATED_FORM_DATA: str = ...

@pytest.fixture
def chart_id(load_world_bank_dashboard_with_slices) -> int:
    ...

@pytest.fixture
def admin_id() -> int:
    ...

@pytest.fixture
def datasource() -> SqlaTable:
    ...

@pytest.fixture(autouse=True)
def cache(chart_id: int, admin_id: int, datasource: SqlaTable) -> None:
    ...

def test_post(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_bad_request_non_string(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_bad_request_non_json_string(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_access_denied(test_client: pytest.fixture, login_as: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_same_key_for_same_context(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_different_key_for_different_context(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_same_key_for_same_tab_id(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_different_key_for_different_tab_id(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_post_different_key_for_no_tab_id(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_same_key_for_same_tab_id(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_different_key_for_different_tab_id(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_different_key_for_no_tab_id(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_bad_request(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_bad_request_non_string(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_bad_request_non_json_string(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_access_denied(test_client: pytest.fixture, login_as: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_put_not_owner(test_client: pytest.fixture, login_as: pytest.fixture, chart_id: int, datasource: SqlaTable) -> None:
    ...

def test_get_key_not_found(test_client: pytest.fixture, login_as_admin: pytest.fixture) -> None:
    ...

def test_get(test_client: pytest.fixture, login_as_admin: pytest.fixture) -> None:
    ...

def test_get_access_denied(test_client: pytest.fixture, login_as: pytest.fixture) -> None:
    ...

@patch('superset.security.SupersetSecurityManager.can_access_datasource')
def test_get_dataset_access_denied(mock_can_access_datasource: pytest.fixture, test_client: pytest.fixture, login_as_admin: pytest.fixture) -> None:
    ...

def test_delete(test_client: pytest.fixture, login_as_admin: pytest.fixture) -> None:
    ...

def test_delete_access_denied(test_client: pytest.fixture, login_as: pytest.fixture) -> None:
    ...

def test_delete_not_owner(test_client: pytest.fixture, login_as_admin: pytest.fixture, chart_id: int, datasource: SqlaTable, admin_id: int) -> None:
    ...