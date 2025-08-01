from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from unittest.mock import ANY, patch
from flask_appbuilder.security.sqla.models import User
import prison
from superset import db
from superset.models.core import Log
from superset.views.log.api import LogRestApi
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.conftest import with_feature_flags
from tests.integration_tests.constants import ADMIN_USERNAME, ALPHA_USERNAME, GAMMA_USERNAME
from tests.integration_tests.dashboard_utils import create_dashboard
from tests.integration_tests.test_app import app

EXPECTED_COLUMNS: List[str] = ['action', 'dashboard_id', 'dttm', 'duration_ms', 'json', 'referrer', 'slice_id', 'user', 'user_id']


class TestLogApi(SupersetTestCase):

    def insert_log(
        self, 
        action: str, 
        user: User, 
        dashboard_id: int = 0, 
        slice_id: int = 0, 
        json: str = '', 
        duration_ms: int = 0
    ) -> Log:
        log: Log = Log(
            action=action, 
            user=user, 
            dashboard_id=dashboard_id, 
            slice_id=slice_id, 
            json=json, 
            duration_ms=duration_ms
        )
        db.session.add(log)
        db.session.commit()
        return log

    def test_not_enabled(self) -> None:
        with patch.object(LogRestApi, 'is_enabled', return_value=False):
            admin_user: User = self.get_user('admin')
            self.insert_log('some_action', admin_user)
            self.login(ADMIN_USERNAME)
            arguments: Dict[str, Any] = {'filters': [{'col': 'action', 'opr': 'sw', 'value': 'some_'}]}
            uri: str = f'api/v1/log/?q={prison.dumps(arguments)}'
            rv = self.client.get(uri)
            assert rv.status_code == 404

    def test_get_list(self) -> None:
        """
        Log API: Test get list
        """
        admin_user: User = self.get_user('admin')
        log: Log = self.insert_log('some_action', admin_user)
        self.login(ADMIN_USERNAME)
        arguments: Dict[str, Any] = {'filters': [{'col': 'action', 'opr': 'sw', 'value': 'some_'}]}
        uri: str = f'api/v1/log/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert list(response['result'][0].keys()) == EXPECTED_COLUMNS
        assert response['result'][0]['action'] == 'some_action'
        assert response['result'][0]['user'] == {'username': 'admin'}
        db.session.delete(log)
        db.session.commit()

    def test_get_list_not_allowed(self) -> None:
        """
        Log API: Test get list
        """
        admin_user: User = self.get_user('admin')
        log: Log = self.insert_log('action', admin_user)
        self.login(GAMMA_USERNAME)
        uri: str = 'api/v1/log/'
        rv = self.client.get(uri)
        assert rv.status_code == 403
        self.login(ALPHA_USERNAME)
        rv = self.client.get(uri)
        assert rv.status_code == 403
        db.session.delete(log)
        db.session.commit()

    def test_get_item(self) -> None:
        """
        Log API: Test get item
        """
        admin_user: User = self.get_user('admin')
        log: Log = self.insert_log('some_action', admin_user)
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/log/{log.id}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert list(response['result'].keys()) == EXPECTED_COLUMNS
        assert response['result']['action'] == 'some_action'
        assert response['result']['user'] == {'username': 'admin'}
        db.session.delete(log)
        db.session.commit()

    def test_delete_log(self) -> None:
        """
        Log API: Test delete (does not exist)
        """
        admin_user: User = self.get_user('admin')
        log: Log = self.insert_log('action', admin_user)
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/log/{log.id}'
        rv = self.client.delete(uri)
        assert rv.status_code == 405
        db.session.delete(log)
        db.session.commit()

    def test_update_log(self) -> None:
        """
        Log API: Test update (does not exist)
        """
        admin_user: User = self.get_user('admin')
        log: Log = self.insert_log('action', admin_user)
        self.login(ADMIN_USERNAME)
        log_data: Dict[str, Any] = {'action': 'some_action'}
        uri: str = f'api/v1/log/{log.id}'
        rv = self.client.put(uri, json=log_data)
        assert rv.status_code == 405
        db.session.delete(log)
        db.session.commit()

    def test_get_recent_activity(self) -> None:
        """
        Log API: Test recent activity endpoint
        """
        admin_user: User = self.get_user('admin')
        self.login(ADMIN_USERNAME)
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        log1: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        uri: str = f'api/v1/log/recent_activity/'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        db.session.delete(log1)
        db.session.delete(log2)
        db.session.delete(dash)
        db.session.commit()
        expected_result: Dict[str, Any] = {
            'result': [{
                'action': 'dashboard',
                'item_type': 'dashboard',
                'item_url': '/superset/dashboard/dash_slug/',
                'item_title': 'dash_title',
                'time': ANY,
                'time_delta_humanized': ANY
            }]
        }
        assert response == expected_result

    def test_get_recent_activity_actions_filter(self) -> None:
        """
        Log API: Test recent activity actions argument
        """
        admin_user: User = self.get_user('admin')
        self.login(ADMIN_USERNAME)
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        log: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2: Log = self.insert_log('explore', admin_user, dashboard_id=dash.id)
        arguments: Dict[str, Any] = {'actions': ['dashboard']}
        uri: str = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        db.session.delete(log)
        db.session.delete(log2)
        db.session.delete(dash)
        db.session.commit()
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert len(response['result']) == 1

    def test_get_recent_activity_distinct_false(self) -> None:
        """
        Log API: Test recent activity when distinct is false
        """
        db.session.query(Log).delete(synchronize_session=False)
        db.session.commit()
        admin_user: User = self.get_user('admin')
        self.login(ADMIN_USERNAME)
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        log: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        arguments: Dict[str, Any] = {'distinct': False}
        uri: str = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        db.session.delete(log)
        db.session.delete(log2)
        db.session.delete(dash)
        db.session.commit()
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        assert len(response['result']) == 2

    def test_get_recent_activity_pagination(self) -> None:
        """
        Log API: Test recent activity pagination arguments
        """
        admin_user: User = self.get_user('admin')
        self.login(ADMIN_USERNAME)
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        dash2 = create_dashboard('dash2_slug', 'dash2_title', '{}', [])
        dash3 = create_dashboard('dash3_slug', 'dash3_title', '{}', [])
        log: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash2.id)
        log3: Log = self.insert_log('dashboard', admin_user, dashboard_id=dash3.id)
        now: datetime = datetime.now()
        log3.dttm = now
        log2.dttm = now - timedelta(days=1)
        log.dttm = now - timedelta(days=2)
        arguments: Dict[str, Any] = {'page': 0, 'page_size': 2}
        uri: str = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode('utf-8'))
        expected_result_page_0: Dict[str, Any] = {
            'result': [
                {
                    'action': 'dashboard',
                    'item_type': 'dashboard',
                    'item_url': '/superset/dashboard/dash3_slug/',
                    'item_title': 'dash3_title',
                    'time': ANY,
                    'time_delta_humanized': ANY
                },
                {
                    'action': 'dashboard',
                    'item_type': 'dashboard',
                    'item_url': '/superset/dashboard/dash2_slug/',
                    'item_title': 'dash2_title',
                    'time': ANY,
                    'time_delta_humanized': ANY
                }
            ]
        }
        assert response == expected_result_page_0

        arguments = {'page': 1, 'page_size': 2}
        uri = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        db.session.delete(log)
        db.session.delete(log2)
        db.session.delete(log3)
        db.session.delete(dash)
        db.session.delete(dash2)
        db.session.delete(dash3)
        db.session.commit()
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        expected_result_page_1: Dict[str, Any] = {
            'result': [
                {
                    'action': 'dashboard',
                    'item_type': 'dashboard',
                    'item_url': '/superset/dashboard/dash_slug/',
                    'item_title': 'dash_title',
                    'time': ANY,
                    'time_delta_humanized': ANY
                }
            ]
        }
        assert response == expected_result_page_1