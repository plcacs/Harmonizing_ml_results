from datetime import datetime
from io import BytesIO
from typing import Optional, List, Generator, Any
from unittest.mock import patch
from zipfile import is_zipfile, ZipFile

import yaml
import pytest
import prison
from freezegun import freeze_time
from sqlalchemy.sql import func, and_

from superset import db
from superset.models.core import Database, FavStar
from superset.models.sql_lab import SavedQuery
from superset.tags.models import ObjectType, Tag, TaggedObject
from superset.utils.database import get_example_database
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_SQLLAB_USERNAME
from tests.integration_tests.fixtures.importexport import (
    database_config,
    saved_queries_config,
    saved_queries_metadata_config,
)
from tests.integration_tests.fixtures.tags import create_custom_tags, get_filter_params

SAVED_QUERIES_FIXTURE_COUNT: int = 10


class TestSavedQueryApi(SupersetTestCase):

    def insert_saved_query(
        self,
        label: str,
        sql: str,
        db_id: Optional[int] = None,
        created_by: Optional[Any] = None,
        schema: str = '',
        description: str = ''
    ) -> SavedQuery:
        database: Optional[Database] = None
        if db_id:
            database = db.session.query(Database).get(db_id)
        query: SavedQuery = SavedQuery(
            database=database,
            created_by=created_by,
            sql=sql,
            label=label,
            schema=schema,
            description=description
        )
        db.session.add(query)
        db.session.commit()
        return query

    def insert_default_saved_query(
        self,
        label: str = 'saved1',
        schema: str = 'schema1',
        username: str = 'admin'
    ) -> SavedQuery:
        admin: Any = self.get_user(username)
        example_db: Database = get_example_database()
        return self.insert_saved_query(
            label, 'SELECT col1, col2 from table1', db_id=example_db.id, created_by=admin, schema=schema, description='cool description'
        )

    @pytest.fixture
    def create_saved_queries(self) -> Generator[List[SavedQuery], None, None]:
        with self.create_app().app_context():
            saved_queries: List[SavedQuery] = []
            admin: Any = self.get_user('admin')
            for cx in range(SAVED_QUERIES_FIXTURE_COUNT - 1):
                saved_queries.append(self.insert_default_saved_query(label=f'label{cx}', schema=f'schema{cx}'))
            saved_queries.append(self.insert_default_saved_query(label=f'label{SAVED_QUERIES_FIXTURE_COUNT}', schema=f'schema{SAVED_QUERIES_FIXTURE_COUNT}', username='gamma_sqllab'))
            fav_saved_queries: List[FavStar] = []
            for cx in range(round(SAVED_QUERIES_FIXTURE_COUNT / 2)):
                fav_star: FavStar = FavStar(user_id=admin.id, class_name='query', obj_id=saved_queries[cx].id)
                db.session.add(fav_star)
                db.session.commit()
                fav_saved_queries.append(fav_star)
            yield saved_queries
            for saved_query in saved_queries:
                db.session.delete(saved_query)
            for fav_saved_query in fav_saved_queries:
                db.session.delete(fav_saved_query)
            db.session.commit()

    @pytest.fixture
    def create_saved_queries_some_with_tags(self, create_custom_tags: Any) -> Generator[List[SavedQuery], None, None]:
        """
        Fixture that creates 4 saved queries:
            - ``first_query`` is associated with ``first_tag``
            - ``second_query`` is associated with ``second_tag``
            - ``third_query`` is associated with both ``first_tag`` and ``second_tag``
            - ``fourth_query`` is not associated with any tag

        Relies on the ``create_custom_tags`` fixture for the tag creation.
        """
        with self.create_app().app_context():
            tags: dict[str, Optional[Tag]] = {
                'first_tag': db.session.query(Tag).filter(Tag.name == 'first_tag').first(),
                'second_tag': db.session.query(Tag).filter(Tag.name == 'second_tag').first()
            }
            query_labels: List[str] = ['first_query', 'second_query', 'third_query', 'fourth_query']
            queries: List[SavedQuery] = [self.insert_default_saved_query(label=name) for name in query_labels]
            tag_associations: List[TaggedObject] = [
                TaggedObject(object_id=queries[0].id, object_type=ObjectType.chart, tag=tags['first_tag']),
                TaggedObject(object_id=queries[1].id, object_type=ObjectType.chart, tag=tags['second_tag']),
                TaggedObject(object_id=queries[2].id, object_type=ObjectType.chart, tag=tags['first_tag']),
                TaggedObject(object_id=queries[2].id, object_type=ObjectType.chart, tag=tags['second_tag'])
            ]
            for association in tag_associations:
                db.session.add(association)
            db.session.commit()
            yield queries
            for association in tag_associations:
                db.session.delete(association)
            for chart in queries:
                db.session.delete(chart)
            db.session.commit()

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_saved_query(self) -> None:
        """
        Saved Query API: Test get list saved query
        """
        admin: Any = self.get_user('admin')
        saved_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).all()
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/saved_query/'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(saved_queries)
        expected_columns: List[str] = ['changed_on_delta_humanized', 'created_on', 'created_by', 'database', 'db_id', 'description', 'id', 'label', 'schema', 'sql', 'sql_tables']
        for expected_column in expected_columns:
            assert expected_column in data['result'][0]

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_saved_query_gamma(self) -> None:
        """
        Saved Query API: Test get list saved query
        """
        user: Any = self.get_user('gamma_sqllab')
        saved_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == user).all()
        self.login(user.username)
        uri: str = 'api/v1/saved_query/'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(saved_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_sort_saved_query(self) -> None:
        """
        Saved Query API: Test get list and sort saved query
        """
        admin: Any = self.get_user('admin')
        saved_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).order_by(SavedQuery.schema.asc()).all()
        self.login(ADMIN_USERNAME)
        query_string: dict = {'order_column': 'schema', 'order_direction': 'asc'}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(saved_queries)
        for i, query in enumerate(saved_queries):
            assert query.schema == data['result'][i]['schema']
        query_string = {'order_column': 'database.database_name', 'order_direction': 'asc'}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        query_string = {'order_column': 'created_by.first_name', 'order_direction': 'asc'}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_saved_query(self) -> None:
        """
        Saved Query API: Test get list and filter saved query
        """
        all_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.label.ilike('%2%')).all()
        self.login(ADMIN_USERNAME)
        query_string: dict = {'filters': [{'col': 'label', 'opr': 'ct', 'value': '2'}]}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_database_saved_query(self) -> None:
        """
        Saved Query API: Test get list and database saved query
        """
        example_db: Database = get_example_database()
        admin_user: Any = self.get_user('admin')
        all_db_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.db_id == example_db.id).filter(SavedQuery.created_by_fk == admin_user.id).all()
        self.login(ADMIN_USERNAME)
        query_string: dict = {'filters': [{'col': 'database', 'opr': 'rel_o_m', 'value': example_db.id}]}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_db_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_schema_saved_query(self) -> None:
        """
        Saved Query API: Test get list and schema saved query
        """
        schema_name: str = 'schema1'
        admin_user: Any = self.get_user('admin')
        all_db_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.schema == schema_name).filter(SavedQuery.created_by_fk == admin_user.id).all()
        self.login(ADMIN_USERNAME)
        query_string: dict = {'filters': [{'col': 'schema', 'opr': 'eq', 'value': schema_name}]}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_db_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_schema_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (schema) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user('admin')
        all_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).filter(SavedQuery.schema.ilike('%2%')).all()
        query_string: dict = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'schema2'}]}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_label_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (label) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user('admin')
        all_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).filter(SavedQuery.label.ilike('%3%')).all()
        query_string: dict = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'label3'}]}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_sql_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (sql) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user('admin')
        all_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).filter(SavedQuery.sql.ilike('%table%')).all()
        query_string: dict = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'table'}]}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_description_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (description) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user('admin')
        all_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).filter(SavedQuery.description.ilike('%cool%')).all()
        query_string: dict = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'cool'}]}
        uri: str = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv: Any = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries_some_with_tags')
    def test_get_saved_queries_tag_filters(self) -> None:
        """
        Saved Query API: Test get saved queries with tag filters
        """
        tags: dict[str, Optional[Tag]] = {
            'first_tag': db.session.query(Tag).filter(Tag.name == 'first_tag').first(),
            'second_tag': db.session.query(Tag).filter(Tag.name == 'second_tag').first(),
            'third_tag': db.session.query(Tag).filter(Tag.name == 'third_tag').first()
        }
        saved_queries_tag_relationship: dict[str, List[Any]] = {
            tag.name: db.session.query(SavedQuery.id).join(SavedQuery.tags).filter(Tag.id == tag.id).all()
            for tag in tags.values() if tag is not None
        }
        for tag_name, tag in tags.items():
            expected_saved_queries: List[Any] = saved_queries_tag_relationship.get(tag_name, [])
            filter_params: Any = get_filter_params('saved_query_tag_id', tag.id) if tag is not None else {}
            response_by_id: Any = self.get_list('saved_query', filter_params)
            assert response_by_id.status_code == 200
            data_by_id: dict = json.loads(response_by_id.data.decode('utf-8'))
            filter_params = get_filter_params('saved_query_tags', tag.name) if tag is not None else {}
            response_by_name: Any = self.get_list('saved_query', filter_params)
            assert response_by_name.status_code == 200
            data_by_name: dict = json.loads(response_by_name.data.decode('utf-8'))
            assert data_by_id['count'] == data_by_name['count'], len(expected_saved_queries)
            assert set((query['id'] for query in data_by_id['result'])) == set((query['id'] for query in data_by_name['result'])), set((query.id for query in expected_saved_queries))

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_saved_query_favorite_filter(self) -> None:
        """
        SavedQuery API: Test get saved queries favorite filter
        """
        admin: Any = self.get_user('admin')
        users_favorite_query = db.session.query(FavStar.obj_id).filter(and_(FavStar.user_id == admin.id, FavStar.class_name == 'query'))
        expected_models: List[SavedQuery] = db.session.query(SavedQuery).filter(and_(SavedQuery.id.in_(users_favorite_query))).order_by(SavedQuery.label.asc()).all()
        arguments: dict = {
            'filters': [{'col': 'id', 'opr': 'saved_query_is_fav', 'value': True}],
            'order_column': 'label',
            'order_direction': 'asc',
            'keys': ['none'],
            'columns': ['label']
        }
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/?q={prison.dumps(arguments)}'
        rv: Any = self.client.get(uri)
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert len(expected_models) == data['count']
        for i, expected_model in enumerate(expected_models):
            assert expected_model.label == data['result'][i]['label']
        expected_models = db.session.query(SavedQuery).filter(and_(~SavedQuery.id.in_(users_favorite_query), SavedQuery.created_by == admin)).order_by(SavedQuery.label.asc()).all()
        arguments['filters'][0]['value'] = False
        uri = f'api/v1/saved_query/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        data = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert len(expected_models) == data['count']

    def test_info_saved_query(self) -> None:
        """
        SavedQuery API: Test info
        """
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/saved_query/_info'
        rv: Any = self.get_assert_metric(uri, 'info')
        assert rv.status_code == 200

    def test_info_security_saved_query(self) -> None:
        """
        SavedQuery API: Test info security
        """
        self.login(ADMIN_USERNAME)
        params: dict = {'keys': ['permissions']}
        uri: str = f'api/v1/saved_query/_info?q={prison.dumps(params)}'
        rv: Any = self.get_assert_metric(uri, 'info')
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert set(data['permissions']) == {'can_read', 'can_write', 'can_export'}

    def test_related_saved_query(self) -> None:
        """
        SavedQuery API: Test related databases
        """
        self.login(ADMIN_USERNAME)
        databases: List[Database] = db.session.query(Database).all()
        expected_result: dict = {
            'count': len(databases),
            'result': [{'extra': {}, 'text': str(database), 'value': database.id} for database in databases]
        }
        uri: str = 'api/v1/saved_query/related/database'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert data == expected_result

    def test_related_saved_query_not_found(self) -> None:
        """
        SavedQuery API: Test related user not found
        """
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/saved_query/related/user'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 404

    @pytest.mark.usefixtures('create_saved_queries')
    def test_distinct_saved_query(self) -> None:
        """
        SavedQuery API: Test distinct schemas
        """
        admin: Any = self.get_user('admin')
        saved_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).all()
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/saved_query/distinct/schema'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        data: dict = json.loads(rv.data.decode('utf-8'))
        expected_response: dict = {
            'count': len(saved_queries),
            'result': [{'text': f'schema{i}', 'value': f'schema{i}'} for i in range(len(saved_queries))]
        }
        assert data == expected_response

    def test_get_saved_query_not_allowed(self) -> None:
        """
        SavedQuery API: Test related user not allowed
        """
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/saved_query/wrong'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 405

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_saved_query(self) -> None:
        """
        Saved Query API: Test get saved query
        """
        saved_query: SavedQuery = db.session.query(SavedQuery).filter(SavedQuery.label == 'label1').all()[0]
        self.login(ADMIN_USERNAME)
        with freeze_time(datetime.now()):
            uri: str = f'api/v1/saved_query/{saved_query.id}'
            rv: Any = self.get_assert_metric(uri, 'get')
            assert rv.status_code == 200
        expected_result: dict = {
            'id': saved_query.id,
            'catalog': None,
            'database': {'id': saved_query.database.id, 'database_name': 'examples'},
            'description': 'cool description',
            'changed_by': None,
            'changed_on_delta_humanized': 'now',
            'created_by': {
                'first_name': saved_query.created_by.first_name,
                'id': saved_query.created_by.id,
                'last_name': saved_query.created_by.last_name
            },
            'sql': 'SELECT col1, col2 from table1',
            'sql_tables': [{'catalog': None, 'schema': None, 'table': 'table1'}],
            'schema': 'schema1',
            'label': 'label1',
            'template_parameters': None
        }
        data: dict = json.loads(rv.data.decode('utf-8'))
        for key, value in data['result'].items():
            if key != 'changed_on':
                assert value == expected_result[key]

    def test_get_saved_query_not_found(self) -> None:
        """
        Saved Query API: Test get saved query not found
        """
        query: SavedQuery = self.insert_default_saved_query()
        max_id: Optional[int] = db.session.query(func.max(SavedQuery.id)).scalar()
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/{max_id + 1 if max_id is not None else 1}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 404
        db.session.delete(query)
        db.session.commit()

    def test_create_saved_query(self) -> None:
        """
        Saved Query API: Test create
        """
        self.get_user('admin')
        example_db: Database = get_example_database()
        post_data: dict = {
            'schema': 'schema1',
            'label': 'label1',
            'description': 'some description',
            'sql': 'SELECT col1, col2 from table1',
            'db_id': example_db.id
        }
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/saved_query/'
        rv: Any = self.client.post(uri, json=post_data)
        data: dict = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        saved_query_id: Any = data.get('id')
        model: Optional[SavedQuery] = db.session.query(SavedQuery).get(saved_query_id)
        for key in post_data:
            assert getattr(model, key) == data['result'][key]
        if model:
            db.session.delete(model)
            db.session.commit()

    @pytest.mark.usefixtures('create_saved_queries')
    def test_update_saved_query(self) -> None:
        """
        Saved Query API: Test update
        """
        saved_query: SavedQuery = db.session.query(SavedQuery).filter(SavedQuery.label == 'label1').all()[0]
        put_data: dict = {'schema': 'schema_changed', 'label': 'label_changed'}
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/{saved_query.id}'
        rv: Any = self.client.put(uri, json=put_data)
        assert rv.status_code == 200
        model: Optional[SavedQuery] = db.session.query(SavedQuery).get(saved_query.id)
        assert model is not None
        assert model.label == 'label_changed'
        assert model.schema == 'schema_changed'

    @pytest.mark.usefixtures('create_saved_queries')
    def test_update_saved_query_not_found(self) -> None:
        """
        Saved Query API: Test update not found
        """
        max_id: Optional[int] = db.session.query(func.max(SavedQuery.id)).scalar()
        self.login(ADMIN_USERNAME)
        put_data: dict = {'schema': 'schema_changed', 'label': 'label_changed'}
        uri: str = f'api/v1/saved_query/{max_id + 1 if max_id is not None else 1}'
        rv: Any = self.client.put(uri, json=put_data)
        assert rv.status_code == 404

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_saved_query(self) -> None:
        """
        Saved Query API: Test delete
        """
        saved_query: SavedQuery = db.session.query(SavedQuery).filter(SavedQuery.label == 'label1').all()[0]
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/{saved_query.id}'
        rv: Any = self.client.delete(uri)
        assert rv.status_code == 200
        model: Optional[SavedQuery] = db.session.query(SavedQuery).get(saved_query.id)
        assert model is None

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_saved_query_not_found(self) -> None:
        """
        Saved Query API: Test delete not found
        """
        max_id: Optional[int] = db.session.query(func.max(SavedQuery.id)).scalar()
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/{max_id + 1 if max_id is not None else 1}'
        rv: Any = self.client.delete(uri)
        assert rv.status_code == 404

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_bulk_saved_queries(self) -> None:
        """
        Saved Query API: Test delete bulk
        """
        admin: Any = self.get_user('admin')
        saved_queries: List[SavedQuery] = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).all()
        saved_query_ids: List[int] = [saved_query.id for saved_query in saved_queries]
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/?q={prison.dumps(saved_query_ids)}'
        rv: Any = self.delete_assert_metric(uri, 'bulk_delete')
        assert rv.status_code == 200
        response: dict = json.loads(rv.data.decode('utf-8'))
        expected_response: dict = {'message': f'Deleted {len(saved_query_ids)} saved queries'}
        assert response == expected_response
        saved_queries = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).all()
        assert saved_queries == []

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_one_bulk_saved_queries(self) -> None:
        """
        Saved Query API: Test delete one in bulk
        """
        saved_query: SavedQuery = db.session.query(SavedQuery).first()
        saved_query_ids: List[int] = [saved_query.id]
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/?q={prison.dumps(saved_query_ids)}'
        rv: Any = self.delete_assert_metric(uri, 'bulk_delete')
        assert rv.status_code == 200
        response: dict = json.loads(rv.data.decode('utf-8'))
        expected_response: dict = {'message': f'Deleted {len(saved_query_ids)} saved query'}
        assert response == expected_response
        saved_query_ = db.session.query(SavedQuery).get(saved_query_ids[0])
        assert saved_query_ is None

    def test_delete_bulk_saved_query_bad_request(self) -> None:
        """
        Saved Query API: Test delete bulk bad request
        """
        saved_query_ids: List[Any] = [1, 'a']
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/?q={prison.dumps(saved_query_ids)}'
        rv: Any = self.delete_assert_metric(uri, 'bulk_delete')
        assert rv.status_code == 400

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_bulk_saved_query_not_found(self) -> None:
        """
        Saved Query API: Test delete bulk not found
        """
        max_id: Optional[int] = db.session.query(func.max(SavedQuery.id)).scalar()
        saved_query_ids: List[int] = [max_id + 1 if max_id is not None else 1, max_id + 2 if max_id is not None else 2]
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/saved_query/?q={prison.dumps(saved_query_ids)}'
        rv: Any = self.delete_assert_metric(uri, 'bulk_delete')
        assert rv.status_code == 404

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export(self) -> None:
        """
        Saved Query API: Test export
        """
        admin: Any = self.get_user('admin')
        sample_query: SavedQuery = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).first()
        self.login(ADMIN_USERNAME)
        argument: List[int] = [sample_query.id]
        uri: str = f'api/v1/saved_query/export/?q={prison.dumps(argument)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 200
        buf: BytesIO = BytesIO(rv.data)
        assert is_zipfile(buf)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export_not_found(self) -> None:
        """
        Saved Query API: Test export
        """
        max_id: Optional[int] = db.session.query(func.max(SavedQuery.id)).scalar()
        self.login(ADMIN_USERNAME)
        argument: List[int] = [max_id + 1 if max_id is not None else 1, max_id + 2 if max_id is not None else 2]
        uri: str = f'api/v1/saved_query/export/?q={prison.dumps(argument)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 404

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export_not_allowed(self) -> None:
        """
        Saved Query API: Test export
        """
        admin: Any = self.get_user('admin')
        sample_query: SavedQuery = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).first()
        self.login(GAMMA_SQLLAB_USERNAME)
        argument: List[int] = [sample_query.id]
        uri: str = f'api/v1/saved_query/export/?q={prison.dumps(argument)}'
        rv: Any = self.client.get(uri)
        assert rv.status_code == 404

    def create_saved_query_import(self) -> BytesIO:
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('saved_query_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(saved_queries_metadata_config).encode())
            with bundle.open('saved_query_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_config).encode())
            with bundle.open('saved_query_export/queries/imported_database/public/imported_saved_query.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(saved_queries_config).encode())
        buf.seek(0)
        return buf

    @patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_saved_queries(self, mock_add_permissions: Any) -> None:
        """
        Saved Query API: Test import
        """
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/saved_query/import/'
        buf: BytesIO = self.create_saved_query_import()
        form_data: dict = {'formData': (buf, 'saved_query.zip')}
        rv: Any = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response: dict = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}
        database: Database = db.session.query(Database).filter_by(uuid=database_config['uuid']).one()
        assert database.database_name == 'imported_database'
        saved_query: SavedQuery = db.session.query(SavedQuery).filter_by(uuid=saved_queries_config['uuid']).one()
        assert saved_query.database == database
        db.session.delete(saved_query)
        db.session.delete(database)
        db.session.commit()