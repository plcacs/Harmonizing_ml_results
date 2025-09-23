from datetime import datetime, timedelta
from typing import Generator, List, Any
from unittest import mock
import random
import string

import pytest
import prison
from sqlalchemy.sql import func
import tests.integration_tests.test_app
from superset import db, security_manager
from superset.common.db_query_status import QueryStatus
from superset.models.core import Database
from superset.utils.database import get_example_database, get_main_database
from superset.utils import json
from superset.models.sql_lab import Query
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_SQLLAB_USERNAME

QUERIES_FIXTURE_COUNT: int = 10


class TestQueryApi(SupersetTestCase):

    def insert_query(
        self,
        database_id: int,
        user_id: int,
        client_id: str,
        sql: str = '',
        select_sql: str = '',
        executed_sql: str = '',
        limit: int = 100,
        progress: int = 100,
        rows: int = 100,
        tab_name: str = '',
        status: str = 'success',
        changed_on: datetime = datetime(2020, 1, 1)
    ) -> Query:
        database: Database = db.session.query(Database).get(database_id)
        user: Any = db.session.query(security_manager.user_model).get(user_id)
        query: Query = Query(
            database=database,
            user=user,
            client_id=client_id,
            sql=sql,
            select_sql=select_sql,
            executed_sql=executed_sql,
            limit=limit,
            progress=progress,
            rows=rows,
            tab_name=tab_name,
            status=status,
            changed_on=changed_on,
        )
        db.session.add(query)
        db.session.commit()
        return query

    @pytest.fixture
    def create_queries(self) -> Generator[List[Query], None, None]:
        with self.create_app().app_context():
            queries: List[Query] = []
            admin_id: int = self.get_user('admin').id
            alpha_id: int = self.get_user('alpha').id
            example_database_id: int = get_example_database().id
            main_database_id: int = get_main_database().id
            for cx in range(QUERIES_FIXTURE_COUNT - 1):
                queries.append(
                    self.insert_query(
                        example_database_id,
                        admin_id,
                        self.get_random_string(),
                        sql=f'SELECT col1, col2 from table{cx}',
                        rows=cx,
                        status=QueryStatus.SUCCESS if cx % 2 == 0 else QueryStatus.RUNNING,
                    )
                )
            queries.append(
                self.insert_query(
                    main_database_id,
                    alpha_id,
                    self.get_random_string(),
                    sql=f'SELECT col1, col2 from table{QUERIES_FIXTURE_COUNT}',
                    rows=QUERIES_FIXTURE_COUNT,
                    status=QueryStatus.SUCCESS,
                )
            )
            yield queries
            for query in queries:
                db.session.delete(query)
            db.session.commit()

    @staticmethod
    def get_random_string(length: int = 10) -> str:
        letters: str = string.ascii_letters
        return ''.join((random.choice(letters) for _ in range(length)))

    def test_get_query(self) -> None:
        """
        Query API: Test get query
        """
        admin: Any = self.get_user('admin')
        client_id: str = self.get_random_string()
        example_db: Database = get_example_database()
        query: Query = self.insert_query(
            example_db.id,
            admin.id,
            client_id,
            sql='SELECT col1, col2 from table1',
            select_sql='SELECT col1, col2 from table1',
            executed_sql='SELECT col1, col2 from table1 LIMIT 100',
        )
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/query/{query.id}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        expected_result: dict = {
            'database': {'id': example_db.id},
            'client_id': client_id,
            'end_result_backend_time': None,
            'error_message': None,
            'executed_sql': 'SELECT col1, col2 from table1 LIMIT 100',
            'limit': 100,
            'progress': 100,
            'results_key': None,
            'rows': 100,
            'schema': None,
            'select_as_cta': None,
            'select_as_cta_used': False,
            'select_sql': 'SELECT col1, col2 from table1',
            'sql': 'SELECT col1, col2 from table1',
            'sql_editor_id': None,
            'status': 'success',
            'tab_name': '',
            'tmp_schema_name': None,
            'tmp_table_name': None,
            'tracking_url': None,
        }
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert 'changed_on' in data['result']
        for key, value in data['result'].items():
            if key not in ('changed_on', 'end_time', 'start_running_time', 'start_time', 'id'):
                assert value == expected_result[key]
        db.session.delete(query)
        db.session.commit()

    def test_get_query_not_found(self) -> None:
        """
        Query API: Test get query not found
        """
        admin: Any = self.get_user('admin')
        client_id: str = self.get_random_string()
        query: Query = self.insert_query(get_example_database().id, admin.id, client_id)
        max_id: int = db.session.query(func.max(Query.id)).scalar()
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/query/{max_id + 1}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 404
        db.session.delete(query)
        db.session.commit()

    def test_get_query_no_data_access(self) -> None:
        """
        Query API: Test get query without data access
        """
        gamma1: Any = self.create_user('gamma_1', 'password', 'Gamma', email='gamma1@superset.org')
        gamma2: Any = self.create_user('gamma_2', 'password', 'Gamma', email='gamma2@superset.org')
        sqllab_role: Any = self.get_role('sql_lab')
        gamma1.roles.append(sqllab_role)
        gamma2.roles.append(sqllab_role)
        gamma1_client_id: str = self.get_random_string()
        gamma2_client_id: str = self.get_random_string()
        query_gamma1: Query = self.insert_query(get_example_database().id, gamma1.id, gamma1_client_id)
        query_gamma2: Query = self.insert_query(get_example_database().id, gamma2.id, gamma2_client_id)
        self.login(username='gamma_1', password='password')
        uri: str = f'api/v1/query/{query_gamma2.id}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 404
        uri = f'api/v1/query/{query_gamma1.id}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        self.logout()
        self.login(username='gamma_2', password='password')
        uri = f'api/v1/query/{query_gamma1.id}'
        rv = self.client.get(uri)
        assert rv.status_code == 404
        uri = f'api/v1/query/{query_gamma2.id}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        self.logout()
        self.login(ADMIN_USERNAME)
        uri = f'api/v1/query/{query_gamma1.id}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        uri = f'api/v1/query/{query_gamma2.id}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        db.session.delete(query_gamma1)
        db.session.delete(query_gamma2)
        db.session.delete(gamma1)
        db.session.delete(gamma2)
        db.session.commit()

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query(self) -> None:
        """
        Query API: Test get list query
        """
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/query/'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == QUERIES_FIXTURE_COUNT
        assert sorted(list(data['result'][0].keys())) == [
            'changed_on',
            'database',
            'end_time',
            'executed_sql',
            'id',
            'rows',
            'schema',
            'sql',
            'sql_tables',
            'start_time',
            'status',
            'tab_name',
            'tmp_table_name',
            'tracking_url',
            'user',
        ]
        assert sorted(list(data['result'][0]['user'].keys())) == ['first_name', 'id', 'last_name']
        assert list(data['result'][0]['database'].keys()) == ['database_name']

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_sql(self) -> None:
        """
        Query API: Test get list query filter
        """
        self.login(ADMIN_USERNAME)
        arguments: dict = {'filters': [{'col': 'sql', 'opr': 'ct', 'value': 'table2'}]}
        uri: str = f'api/v1/query/?q={prison.dumps(arguments)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == 1

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_database(self) -> None:
        """
        Query API: Test get list query filter database
        """
        self.login(ADMIN_USERNAME)
        database_id: int = get_main_database().id
        arguments: dict = {'filters': [{'col': 'database', 'opr': 'rel_o_m', 'value': database_id}]}
        uri: str = f'api/v1/query/?q={prison.dumps(arguments)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == 1

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_user(self) -> None:
        """
        Query API: Test get list query filter user
        """
        self.login(ADMIN_USERNAME)
        alpha_id: int = self.get_user('alpha').id
        arguments: dict = {'filters': [{'col': 'user', 'opr': 'rel_o_m', 'value': alpha_id}]}
        uri: str = f'api/v1/query/?q={prison.dumps(arguments)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == 1

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_changed_on(self) -> None:
        """
        Query API: Test get list query filter changed_on
        """
        self.login(ADMIN_USERNAME)
        arguments: dict = {
            'filters': [
                {'col': 'changed_on', 'opr': 'lt', 'value': '2020-02-01T00:00:00Z'},
                {'col': 'changed_on', 'opr': 'gt', 'value': '2019-12-30T00:00:00Z'},
            ]
        }
        uri: str = f'api/v1/query/?q={prison.dumps(arguments)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == QUERIES_FIXTURE_COUNT

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_order(self) -> None:
        """
        Query API: Test get list query filter changed_on
        """
        self.login(ADMIN_USERNAME)
        order_columns: List[str] = [
            'changed_on',
            'database.database_name',
            'rows',
            'schema',
            'sql',
            'tab_name',
            'user.first_name',
        ]
        for order_column in order_columns:
            arguments: dict = {'order_column': order_column, 'order_direction': 'asc'}
            uri: str = f'api/v1/query/?q={prison.dumps(arguments)}'
            rv: Any = self.client.get(uri)
            assert rv.status_code == 200

    def test_get_list_query_no_data_access(self) -> None:
        """
        Query API: Test get queries no data access
        """
        admin: Any = self.get_user('admin')
        client_id: str = self.get_random_string()
        query: Query = self.insert_query(get_example_database().id, admin.id, client_id, sql='SELECT col1, col2 from table1')
        self.login(GAMMA_SQLLAB_USERNAME)
        arguments: dict = {'filters': [{'col': 'sql', 'opr': 'sw', 'value': 'SELECT col1'}]}
        uri: str = f'api/v1/query/?q={prison.dumps(arguments)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == 0
        db.session.delete(query)
        db.session.commit()

    def test_get_updated_since(self) -> None:
        """
        Query API: Test get queries updated since timestamp
        """
        now: datetime = datetime.utcnow()
        client_id: str = self.get_random_string()
        admin: Any = self.get_user('admin')
        example_db: Database = get_example_database()
        old_query: Query = self.insert_query(
            example_db.id,
            admin.id,
            self.get_random_string(),
            sql='SELECT col1, col2 from table1',
            select_sql='SELECT col1, col2 from table1',
            executed_sql='SELECT col1, col2 from table1 LIMIT 100',
            changed_on=now - timedelta(days=3),
        )
        updated_query: Query = self.insert_query(
            example_db.id,
            admin.id,
            client_id,
            sql='SELECT col1, col2 from table1',
            select_sql='SELECT col1, col2 from table1',
            executed_sql='SELECT col1, col2 from table1 LIMIT 100',
            changed_on=now - timedelta(days=1),
        )
        self.login(ADMIN_USERNAME)
        timestamp: float = datetime.timestamp(now - timedelta(days=2)) * 1000
        uri: str = f"api/v1/query/updated_since?q={prison.dumps({'last_updated_ms': timestamp})}"
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        expected_result: dict = updated_query.to_dict()
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert len(data['result']) == 1
        for key, value in data['result'][0].items():
            if key not in ('changed_on', 'end_time', 'start_running_time', 'start_time', 'id'):
                assert value == expected_result[key]
        db.session.delete(old_query)
        db.session.delete(updated_query)
        db.session.commit()

    @mock.patch('superset.sql_lab.cancel_query')
    @mock.patch('superset.views.core.db.session')
    def test_stop_query_not_found(
        self,
        mock_superset_db_session: Any,
        mock_sql_lab_cancel_query: Any,
    ) -> None:
        """
        Handles stop query when the DB engine spec does not
        have a cancel query method (with invalid client_id).
        """
        form_data: dict = {'client_id': 'foo2'}
        query_mock: Any = mock.Mock()
        query_mock.return_value = None
        self.login(ADMIN_USERNAME)
        mock_superset_db_session.query().filter_by().one_or_none = query_mock
        mock_sql_lab_cancel_query.return_value = True
        rv: Any = self.client.post(
            '/api/v1/query/stop',
            data=json.dumps(form_data),
            content_type='application/json',
        )
        assert rv.status_code == 404
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['message'] == 'Query with client_id foo2 not found'

    @mock.patch('superset.sql_lab.cancel_query')
    @mock.patch('superset.views.core.db.session')
    def test_stop_query(
        self,
        mock_superset_db_session: Any,
        mock_sql_lab_cancel_query: Any,
    ) -> None:
        """
        Handles stop query when the DB engine spec does not
        have a cancel query method.
        """
        form_data: dict = {'client_id': 'foo'}
        query_mock: Any = mock.Mock()
        query_mock.client_id = 'foo'
        query_mock.status = QueryStatus.RUNNING
        self.login(ADMIN_USERNAME)
        mock_superset_db_session.query().filter_by().one_or_none().return_value = query_mock
        mock_sql_lab_cancel_query.return_value = True
        rv: Any = self.client.post(
            '/api/v1/query/stop',
            data=json.dumps(form_data),
            content_type='application/json',
        )
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['result'] == 'OK'