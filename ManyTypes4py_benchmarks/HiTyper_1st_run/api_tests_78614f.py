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
KEY = 'test-key'
INITIAL_FORM_DATA = json.dumps({'test': 'initial value'})
UPDATED_FORM_DATA = json.dumps({'test': 'updated value'})

@pytest.fixture
def chart_id(load_world_bank_dashboard_with_slices: Union[bool, list[dict[str, typing.Any]]]):
    with app.app_context() as ctx:
        chart = db.session.query(Slice).filter_by(slice_name="World's Population").one()
        return chart.id

@pytest.fixture
def admin_id():
    with app.app_context() as ctx:
        admin = db.session.query(User).filter_by(username='admin').one()
        return admin.id

@pytest.fixture
def datasource() -> Union[str, crux.models.dataseDataset, supersemodels.slice.Slice]:
    with app.app_context() as ctx:
        dataset = db.session.query(SqlaTable).filter_by(table_name='wb_health_population').first()
        return dataset

@pytest.fixture(autouse=True)
def cache(chart_id: str, admin_id: str, datasource: str) -> None:
    entry = {'owner': admin_id, 'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    cache_manager.explore_form_data_cache.set(KEY, entry)

def test_post(test_client: core.models.Recipe, login_as_admin: Union[str, typing.Callable[..., None], list[int]], chart_id: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], sqlalchemy.Table], datasource: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], sqlalchemy.Table]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 201

def test_post_bad_request_non_string(test_client: requests.Session, login_as_admin: Union[typing.Callable[..., None], str, list[int]], chart_id: Union[str, sqlalchemy.orm.Session], datasource: Union[str, sqlalchemy.orm.Session]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': 1234}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 400

def test_post_bad_request_non_json_string(test_client: Union[requests.Session, str, tests.clienMDMClient], login_as_admin: Union[list[int], list[tuple[typing.Union[int,str]]], typing.Callable[..., None]], chart_id: Union[str, sqlalchemy.orm.Session], datasource: Union[str, sqlalchemy.orm.Session]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': 'foo'}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 400

def test_post_access_denied(test_client: Any, login_as: Union[str, list[tuple[typing.Union[int,str]]], typing.Callable[..., None]], chart_id: Union[str, sqlalchemy.orm.session.Session], datasource: Union[str, sqlalchemy.orm.session.Session]) -> None:
    login_as('gamma')
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 403

def test_post_same_key_for_same_context(test_client: requests.Session, login_as_admin: Union[str, typing.Callable[..., None], list[int]], chart_id: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], sqlalchemy.Table], datasource: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], sqlalchemy.Table]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key == second_key

def test_post_different_key_for_different_context(test_client: Union[requests.Session, str], login_as_admin: Union[list[int], sqlalchemy.orm.session.Session.Transaction, int], chart_id: Union[sqlalchemy.orm.Session, fal.models.Season, sqlalchemy.Table], datasource: Union[sqlalchemy.orm.Session, models.Dashboard, fal.models.Season]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'form_data': json.dumps({'test': 'initial value'})}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key

def test_post_same_key_for_same_tab_id(test_client: Union[requests.Session, str, tests.clienMDMClient], login_as_admin: Union[int, list[tuple[typing.Union[int,str]]], list[int]], chart_id: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], models.Course], datasource: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], models.Course]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': json.dumps({'test': 'initial value'})}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key == second_key

def test_post_different_key_for_different_tab_id(test_client: Union[str, requests.Session, tests.clienMDMClient], login_as_admin: Union[int, list[int], list[tuple[typing.Union[int,str]]]], chart_id: Union[recidiviz.persistence.database.session.Session, fal.models.Season, list[recidiviz.persistence.database.schema.state.schema.StatePerson]], datasource: Union[recidiviz.persistence.database.session.Session, fal.models.Season, list[recidiviz.persistence.database.schema.state.schema.StatePerson]]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': json.dumps({'test': 'initial value'})}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data?tab_id=2', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key

def test_post_different_key_for_no_tab_id(test_client: Union[requests.Session, str, tests.clienMDMClient], login_as_admin: Union[list[tuple[typing.Union[int,str]]], int, list[int]], chart_id: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], fal.models.Season], datasource: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], fal.models.Season]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key

def test_put(test_client: Any, login_as_admin: Union[str, list[int], list[tuple[typing.Union[int,str]]]], chart_id: Union[sqlalchemy.orm.Session, str], datasource: Union[sqlalchemy.orm.Session, str]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 200

def test_put_same_key_for_same_tab_id(test_client: Union[requests.Session, tests.clienMDMClient, str], login_as_admin: Union[list[tuple[typing.Union[int,str]]], list[int], grouper.models.service_accounServiceAccount], chart_id: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson]], datasource: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson]]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key == second_key

def test_put_different_key_for_different_tab_id(test_client: requests.Session, login_as_admin: Union[int, list[int], list[tuple[typing.Union[int,str]]]], chart_id: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], fal.models.Season], datasource: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], fal.models.Season]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=2', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key

def test_put_different_key_for_no_tab_id(test_client: Union[requests.Session, str, tests.clienMDMClient], login_as_admin: Union[list[tuple[typing.Union[int,str]]], list[int], int], chart_id: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], fal.models.Season], datasource: Union[recidiviz.persistence.database.session.Session, list[recidiviz.persistence.database.schema.state.schema.StatePerson], fal.models.Season]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key

def test_put_bad_request(test_client: Any, login_as_admin: Union[typing.Callable[..., None], str, list[int]], chart_id: Union[str, sqlalchemy.orm.Session], datasource: Union[str, sqlalchemy.orm.Session]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': 1234}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 400

def test_put_bad_request_non_string(test_client: Any, login_as_admin: Union[str, list[tuple[typing.Union[int,str]]], typing.Callable[..., None]], chart_id: Union[str, sqlalchemy.orm.Session], datasource: Union[str, sqlalchemy.orm.Session]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': 1234}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 400

def test_put_bad_request_non_json_string(test_client: str, login_as_admin: Union[list[tuple[typing.Union[int,str]]], list[int], int], chart_id: Union[str, sqlalchemy.orm.Session], datasource: Union[str, sqlalchemy.orm.Session]) -> None:
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': 'foo'}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 400

def test_put_access_denied(test_client: str, login_as: Union[str, list[tuple[typing.Union[int,str]]], typing.Callable[..., None]], chart_id: Union[str, sqlalchemy.orm.session.Session, sqlalchemy.Table], datasource: Union[str, sqlalchemy.orm.session.Session, sqlalchemy.Table]) -> None:
    login_as('gamma')
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 403

def test_put_not_owner(test_client: str, login_as: Union[str, list[tuple[typing.Union[int,str]]], typing.Callable[..., None]], chart_id: Union[str, sqlalchemy.orm.session.Session, sqlalchemy.engine.Engine], datasource: Union[str, sqlalchemy.orm.session.Session, sqlalchemy.engine.Engine]) -> None:
    login_as('gamma')
    payload = {'datasource_id': datasource.id, 'datasource_type': datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 403

def test_get_key_not_found(test_client: Any, login_as_admin: bool) -> None:
    resp = test_client.get(f'api/v1/explore/form_data/unknown-key')
    assert resp.status_code == 404

def test_get(test_client: Any, login_as_admin: bool) -> None:
    resp = test_client.get(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 200
    data = json.loads(resp.data.decode('utf-8'))
    assert INITIAL_FORM_DATA == data.get('form_data')

def test_get_access_denied(test_client: Any, login_as: Any) -> None:
    login_as('gamma')
    resp = test_client.get(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 403

@patch('superset.security.SupersetSecurityManager.can_access_datasource')
def test_get_dataset_access_denied(mock_can_access_datasource: Any, test_client: Any, login_as_admin: Any) -> None:
    mock_can_access_datasource.side_effect = DatasetAccessDeniedError()
    resp = test_client.get(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 403

def test_delete(test_client: User, login_as_admin: bool) -> None:
    resp = test_client.delete(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 200

def test_delete_access_denied(test_client: Any, login_as: Any) -> None:
    login_as('gamma')
    resp = test_client.delete(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 403

def test_delete_not_owner(test_client: Union[str, None, models.user.User], login_as_admin: Union[typing.Callable[..., None], str], chart_id: Union[sqlalchemy.orm.session.Session, app.models.farm.Farm], datasource: Union[sqlalchemy.orm.session.Session, app.models.farm.Farm], admin_id: Union[int, sqlalchemy.orm.Session, core.Pipeline]) -> None:
    another_key = 'another_key'
    another_owner = admin_id + 1
    entry = {'owner': another_owner, 'datasource_id': datasource.id, 'datasource_type': DatasourceType(datasource.type), 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    cache_manager.explore_form_data_cache.set(another_key, entry)
    resp = test_client.delete(f'api/v1/explore/form_data/{another_key}')
    assert resp.status_code == 403