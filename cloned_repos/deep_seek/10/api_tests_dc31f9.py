"""Unit tests for Superset"""
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, cast
from unittest.mock import patch
from zipfile import is_zipfile, ZipFile
import yaml
import pytest
import prison
from freezegun import freeze_time
from flask import Response
from flask.testing import FlaskClient
from sqlalchemy.sql import func, and_
from superset import db
from superset.models.core import Database
from superset.models.core import FavStar
from superset.models.sql_lab import SavedQuery
from superset.tags.models import ObjectType, Tag, TaggedObject
from superset.utils.database import get_example_database
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_SQLLAB_USERNAME
from tests.integration_tests.fixtures.importexport import database_config, saved_queries_config, saved_queries_metadata_config
from tests.integration_tests.fixtures.tags import create_custom_tags, get_filter_params

SAVED_QUERIES_FIXTURE_COUNT = 10

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
        database = None
        if db_id:
            database = db.session.query(Database).get(db_id)
        query = SavedQuery(
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
        admin = self.get_user(username)
        example_db = get_example_database()
        return self.insert_saved_query(
            label,
            'SELECT col1, col2 from table1',
            db_id=example_db.id,
            created_by=admin,
            schema=schema,
            description='cool description'
        )

    @pytest.fixture
    def create_saved_queries(self) -> Generator[List[SavedQuery], None, None]:
        with self.create_app().app_context():
            saved_queries: List[SavedQuery] = []
            admin = self.get_user('admin')
            for cx in range(SAVED_QUERIES_FIXTURE_COUNT - 1):
                saved_queries.append(
                    self.insert_default_saved_query(label=f'label{cx}', schema=f'schema{cx}')
                )
            saved_queries.append(
                self.insert_default_saved_query(
                    label=f'label{SAVED_QUERIES_FIXTURE_COUNT}',
                    schema=f'schema{SAVED_QUERIES_FIXTURE_COUNT}',
                    username='gamma_sqllab'
                )
            )
            fav_saved_queries: List[FavStar] = []
            for cx in range(round(SAVED_QUERIES_FIXTURE_COUNT / 2)):
                fav_star = FavStar(
                    user_id=admin.id,
                    class_name='query',
                    obj_id=saved_queries[cx].id
                )
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
    def create_saved_queries_some_with_tags(
        self,
        create_custom_tags: Any
    ) -> Generator[List[SavedQuery], None, None]:
        with self.create_app().app_context():
            tags = {
                'first_tag': db.session.query(Tag).filter(Tag.name == 'first_tag').first(),
                'second_tag': db.session.query(Tag).filter(Tag.name == 'second_tag').first()
            }
            query_labels = ['first_query', 'second_query', 'third_query', 'fourth_query']
            queries = [self.insert_default_saved_query(label=name) for name in query_labels]
            tag_associations = [
                TaggedObject(
                    object_id=queries[0].id,
                    object_type=ObjectType.chart,
                    tag=tags['first_tag']
                ),
                TaggedObject(
                    object_id=queries[1].id,
                    object_type=ObjectType.chart,
                    tag=tags['second_tag']
                ),
                TaggedObject(
                    object_id=queries[2].id,
                    object_type=ObjectType.chart,
                    tag=tags['first_tag']
                ),
                TaggedObject(
                    object_id=queries[2].id,
                    object_type=ObjectType.chart,
                    tag=tags['second_tag']
                )
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
        admin = self.get_user('admin')
        saved_queries = db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).all()
        self.login(ADMIN_USERNAME)
        uri = 'api/v1/saved_query/'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(saved_queries)
        expected_columns = [
            'changed_on_delta_humanized', 'created_on', 'created_by', 'database',
            'db_id', 'description', 'id', 'label', 'schema', 'sql', 'sql_tables'
        ]
        for expected_column in expected_columns:
            assert expected_column in data['result'][0]

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_saved_query_gamma(self) -> None:
        user = self.get_user('gamma_sqllab')
        saved_queries = db.session.query(SavedQuery).filter(SavedQuery.created_by == user).all()
        self.login(user.username)
        uri = 'api/v1/saved_query/'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(saved_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_sort_saved_query(self) -> None:
        admin = self.get_user('admin')
        saved_queries = db.session.query(SavedQuery).filter(
            SavedQuery.created_by == admin
        ).order_by(SavedQuery.schema.asc()).all()
        self.login(ADMIN_USERNAME)
        query_string = {'order_column': 'schema', 'order_direction': 'asc'}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
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
        all_queries = db.session.query(SavedQuery).filter(SavedQuery.label.ilike('%2%')).all()
        self.login(ADMIN_USERNAME)
        query_string = {'filters': [{'col': 'label', 'opr': 'ct', 'value': '2'}]}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_database_saved_query(self) -> None:
        example_db = get_example_database()
        admin_user = self.get_user('admin')
        all_db_queries = db.session.query(SavedQuery).filter(
            SavedQuery.db_id == example_db.id
        ).filter(SavedQuery.created_by_fk == admin_user.id).all()
        self.login(ADMIN_USERNAME)
        query_string = {'filters': [{'col': 'database', 'opr': 'rel_o_m', 'value': example_db.id}]}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_db_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_schema_saved_query(self) -> None:
        schema_name = 'schema1'
        admin_user = self.get_user('admin')
        all_db_queries = db.session.query(SavedQuery).filter(
            SavedQuery.schema == schema_name
        ).filter(SavedQuery.created_by_fk == admin_user.id).all()
        self.login(ADMIN_USERNAME)
        query_string = {'filters': [{'col': 'schema', 'opr': 'eq', 'value': schema_name}]}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_db_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_schema_saved_query(self) -> None:
        self.login(ADMIN_USERNAME)
        admin = self.get_user('admin')
        all_queries = db.session.query(SavedQuery).filter(
            SavedQuery.created_by == admin
        ).filter(SavedQuery.schema.ilike('%2%')).all()
        query_string = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'schema2'}]}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_label_saved_query(self) -> None:
        self.login(ADMIN_USERNAME)
        admin = self.get_user('admin')
        all_queries = db.session.query(SavedQuery).filter(
            SavedQuery.created_by == admin
        ).filter(SavedQuery.label.ilike('%3%')).all()
        query_string = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'label3'}]}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_sql_saved_query(self) -> None:
        self.login(ADMIN_USERNAME)
        admin = self.get_user('admin')
        all_queries = db.session.query(SavedQuery).filter(
            SavedQuery.created_by == admin
        ).filter(SavedQuery.sql.ilike('%table%')).all()
        query_string = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'table'}]}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_description_saved_query(self) -> None:
        self.login(ADMIN_USERNAME)
        admin = self.get_user('admin')
        all_queries = db.session.query(SavedQuery).filter(
            SavedQuery.created_by == admin
        ).filter(SavedQuery.description.ilike('%cool%')).all()
        query_string = {'filters': [{'col': 'label', 'opr': 'all_text', 'value': 'cool'}]}
        uri = f'api/v1/saved_query/?q={prison.dumps(query_string)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(all_queries)

    @pytest.mark.usefixtures('create_saved_queries_some_with_tags')
    def test_get_saved_queries_tag_filters(self) -> None:
        tags = {
            'first_tag': db.session.query(Tag).filter(Tag.name == 'first_tag').first(),
            'second_tag': db.session.query(Tag).filter(Tag.name == 'second_tag').first(),
            'third_tag': db.session.query(Tag).filter(Tag.name == 'third_tag').first()
        }
        saved_queries_tag_relationship = {
            tag.name: db.session.query(SavedQuery.id).join(
                SavedQuery.tags
            ).filter(Tag.id == tag.id).all()
            for tag in tags.values()
        }
        for tag_name, tag in tags.items():
            expected_saved_queries = saved_queries_tag_relationship[tag_name]
            filter_params = get_filter_params('saved_query_tag_id', tag.id)
            response_by_id = self.get_list('saved_query', filter_params)
            assert response_by_id.status_code == 200
            data_by_id = json.loads(response_by_id.data.decode('utf-8'))
            filter_params = get_filter_params('saved_query_tags', tag.name)
            response_by_name = self.get_list('saved_query', filter_params)
            assert response_by_name.status_code == 200
            data_by_name = json.loads(response_by_name.data.decode('utf-8'))
            assert data_by_id['count'] == data_by_name['count'], len(expected_saved_queries)
            assert set((query['id'] for query in data_by_id['result'])) == set(
                (query['id'] for query in data_by_name['result'])), set(
                (query.id for query in expected_saved_queries))

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_saved_query_favorite_filter(self) -> None:
        admin = self.get_user('admin')
        users_favorite_query = db.session.query(FavStar.obj_id).filter(
            and_(FavStar.user_id == admin.id, FavStar.class_name == 'query')
        )
        expected_models = db.session.query(SavedQuery).filter(
            and_(SavedQuery.id.in_(users_favorite_query))
        ).order_by(SavedQuery.label.asc()).all()
        arguments = {
            'filters': [{'col': 'id', 'opr': 'saved_query_is_fav', 'value': True}],
            'order_column': 'label',
            'order_direction': 'asc',
            'keys': ['none'],
            'columns': ['label']
        }
        self.login(ADMIN_USERNAME)
        uri = f'api/v1/saved_query/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        data = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert len(expected_models) == data['count']
        for i, expected_model in enumerate(expected_models):
            assert expected_model.label == data['result'][i]['label']
        expected_models = db.session.query(SavedQuery).filter(
            and_(~SavedQuery.id.in_(users_favorite_query), SavedQuery.created_by == admin)
        ).order_by(SavedQuery.label.asc()).all()
        arguments['filters'][0]['value'] = False
        uri = f'api/v1/saved_query/?q={prison.dumps(arguments)}'
       