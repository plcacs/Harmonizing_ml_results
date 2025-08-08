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
def func_4lanrr5k(load_world_bank_dashboard_with_slices):
    with app.app_context() as ctx:
        chart = db.session.query(Slice).filter_by(slice_name=
            "World's Population").one()
        return chart.id


@pytest.fixture
def func_e3fh8x6d():
    with app.app_context() as ctx:
        admin = db.session.query(User).filter_by(username='admin').one()
        return admin.id


@pytest.fixture
def func_xcz2ypnh():
    with app.app_context() as ctx:
        dataset = db.session.query(SqlaTable).filter_by(table_name=
            'wb_health_population').first()
        return dataset


@pytest.fixture(autouse=True)
def func_mjcaicc2(chart_id, admin_id, datasource):
    entry = {'owner': admin_id, 'datasource_id': datasource.id,
        'datasource_type': datasource.type, 'chart_id': chart_id,
        'form_data': INITIAL_FORM_DATA}
    cache_manager.explore_form_data_cache.set(KEY, entry)


def func_h16228uc(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 201


def func_b4hhhybw(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': 1234}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 400


def func_e11p6qrp(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': 'foo'}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 400


def func_hvlq8prx(test_client, login_as, chart_id, datasource):
    login_as('gamma')
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    assert resp.status_code == 403


def func_qnahp785(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key == second_key


def func_cyh7aeix(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'form_data': json.dumps({'test': 'initial value'})}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key


def func_ihx5nof4(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': json.dumps({
        'test': 'initial value'})}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key == second_key


def func_oelo1up4(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': json.dumps({
        'test': 'initial value'})}
    resp = test_client.post('api/v1/explore/form_data?tab_id=1', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data?tab_id=2', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key


def func_9gogaxvi(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': INITIAL_FORM_DATA}
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.post('api/v1/explore/form_data', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key


def func_b9ew2wg6(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 200


def func_59q3ci40(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=1', json
        =payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=1', json
        =payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key == second_key


def func_wmu43t78(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=1', json
        =payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}?tab_id=2', json
        =payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key


def func_nhkbc07g(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    first_key = data.get('key')
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    data = json.loads(resp.data.decode('utf-8'))
    second_key = data.get('key')
    assert first_key != second_key


def func_chlcnbs3(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': 1234}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 400


def func_4gs53568(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': 1234}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 400


def func_lyk8hxoh(test_client, login_as_admin, chart_id, datasource):
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': 'foo'}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 400


def func_9gxjr1to(test_client, login_as, chart_id, datasource):
    login_as('gamma')
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 403


def func_sqs6yzrx(test_client, login_as, chart_id, datasource):
    login_as('gamma')
    payload = {'datasource_id': datasource.id, 'datasource_type':
        datasource.type, 'chart_id': chart_id, 'form_data': UPDATED_FORM_DATA}
    resp = test_client.put(f'api/v1/explore/form_data/{KEY}', json=payload)
    assert resp.status_code == 403


def func_gitagz36(test_client, login_as_admin):
    resp = test_client.get(f'api/v1/explore/form_data/unknown-key')
    assert resp.status_code == 404


def func_tjmso8ro(test_client, login_as_admin):
    resp = test_client.get(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 200
    data = json.loads(resp.data.decode('utf-8'))
    assert INITIAL_FORM_DATA == data.get('form_data')


def func_ov91jh15(test_client, login_as):
    login_as('gamma')
    resp = test_client.get(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 403


@patch('superset.security.SupersetSecurityManager.can_access_datasource')
def func_95zdj0rb(mock_can_access_datasource, test_client, login_as_admin):
    mock_can_access_datasource.side_effect = DatasetAccessDeniedError()
    resp = test_client.get(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 403


def func_qa3tf9o8(test_client, login_as_admin):
    resp = test_client.delete(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 200


def func_xruu7uab(test_client, login_as):
    login_as('gamma')
    resp = test_client.delete(f'api/v1/explore/form_data/{KEY}')
    assert resp.status_code == 403


def func_2qiyfxil(test_client, login_as_admin, chart_id, datasource, admin_id):
    another_key = 'another_key'
    another_owner = admin_id + 1
    entry = {'owner': another_owner, 'datasource_id': datasource.id,
        'datasource_type': DatasourceType(datasource.type), 'chart_id':
        chart_id, 'form_data': INITIAL_FORM_DATA}
    cache_manager.explore_form_data_cache.set(another_key, entry)
    resp = test_client.delete(f'api/v1/explore/form_data/{another_key}')
    assert resp.status_code == 403
