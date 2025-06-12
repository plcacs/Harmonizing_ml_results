"""Unit tests for Superset"""
import unittest
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple, cast
from unittest.mock import ANY, patch
from zipfile import is_zipfile, ZipFile
import prison
import pytest
import yaml
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
from superset import app
from superset.commands.dataset.exceptions import DatasetCreateFailedError
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.extensions import db, security_manager
from superset.models.core import Database
from superset.models.slice import Slice
from superset.utils import json
from superset.utils.core import backend, get_example_default_schema
from superset.utils.database import get_example_database, get_main_database
from superset.utils.dict_import_export import export_to_dict
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.conftest import CTAS_SCHEMA_NAME, with_feature_flags
from tests.integration_tests.constants import ADMIN_USERNAME, ALPHA_USERNAME, GAMMA_USERNAME
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,
    load_birth_names_data,
)
from tests.integration_tests.fixtures.energy_dashboard import (
    load_energy_table_data,
    load_energy_table_with_slice,
)
from tests.integration_tests.fixtures.importexport import (
    database_config,
    database_metadata_config,
    dataset_config,
    dataset_metadata_config,
    dataset_ui_export,
)

class TestDatasetApi(SupersetTestCase):
    fixture_tables_names: Tuple[str, str, str] = ('ab_permission', 'ab_permission_view', 'ab_view_menu')
    fixture_virtual_table_names: Tuple[str, str] = ('sql_virtual_dataset_1', 'sql_virtual_dataset_2')

    @staticmethod
    def insert_dataset(
        table_name: str,
        owners: List[int],
        database: Database,
        sql: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> SqlaTable:
        obj_owners = []
        for owner in owners:
            user = db.session.query(security_manager.user_model).get(owner)
            obj_owners.append(user)
        table = SqlaTable(
            table_name=table_name,
            schema=schema,
            owners=obj_owners,
            database=database,
            sql=sql,
        )
        db.session.add(table)
        db.session.commit()
        table.fetch_metadata()
        return table

    def insert_default_dataset(self) -> SqlaTable:
        return self.insert_dataset(
            'ab_permission',
            [self.get_user('admin').id],
            get_main_database(),
        )

    def get_fixture_datasets(self) -> List[SqlaTable]:
        return (
            db.session.query(SqlaTable)
            .options(joinedload(SqlaTable.database))
            .filter(SqlaTable.table_name.in_(self.fixture_tables_names))
            .all()
        )

    def get_fixture_virtual_datasets(self) -> List[SqlaTable]:
        return (
            db.session.query(SqlaTable)
            .filter(SqlaTable.table_name.in_(self.fixture_virtual_table_names))
            .all()
        )

    @pytest.fixture
    def create_virtual_datasets(self) -> Any:
        with self.create_app().app_context():
            datasets = []
            admin = self.get_user('admin')
            main_db = get_main_database()
            for table_name in self.fixture_virtual_table_names:
                datasets.append(
                    self.insert_dataset(
                        table_name,
                        [admin.id],
                        main_db,
                        'SELECT * from ab_view_menu;',
                    )
                )
            yield datasets
            for dataset in datasets:
                db.session.delete(dataset)
            db.session.commit()

    @pytest.fixture
    def create_datasets(self) -> Any:
        with self.create_app().app_context():
            datasets = []
            admin = self.get_user('admin')
            main_db = get_main_database()
            for tables_name in self.fixture_tables_names:
                datasets.append(self.insert_dataset(tables_name, [admin.id], main_db))
            yield datasets
            for dataset in datasets:
                state = inspect(dataset)
                if not state.was_deleted:
                    db.session.delete(dataset)
            db.session.commit()

    @staticmethod
    def get_energy_usage_dataset() -> SqlaTable:
        example_db = get_example_database()
        return (
            db.session.query(SqlaTable)
            .filter_by(
                database=example_db,
                table_name='energy_usage',
                schema=get_example_default_schema(),
            )
            .one()
        )

    def create_dataset_import(self) -> BytesIO:
        buf = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('dataset_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_metadata_config).encode())
            with bundle.open('dataset_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_config).encode())
            with bundle.open('dataset_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        return buf

    @pytest.mark.usefixtures('load_energy_table_with_slice')
    def test_user_gets_all_datasets(self) -> None:
        gamma_user = security_manager.find_user(username='gamma')

        def count_datasets() -> int:
            uri = 'api/v1/chart/'
            rv = self.client.get(uri, 'get_list')
            assert rv.status_code == 200
            data = rv.get_json()
            return data['count']

        with self.temporary_user(gamma_user, login=True) as user:
            assert count_datasets() == 0
        all_db_pvm = ('all_database_access', 'all_database_access')
        with self.temporary_user(gamma_user, extra_pvms=[all_db_pvm], login=True) as user:
            self.login(username=user.username)
            assert count_datasets() > 0
        all_db_pvm = ('all_datasource_access', 'all_datasource_access')
        with self.temporary_user(gamma_user, extra_pvms=[all_db_pvm], login=True) as user:
            self.login(username=user.username)
            assert count_datasets() > 0
        with self.temporary_user(gamma_user, login=True):
            assert count_datasets() == 0

    def test_get_dataset_list(self) -> None:
        """
        Dataset API: Test get dataset list
        """
        example_db = get_example_database()
        self.login(ADMIN_USERNAME)
        arguments = {
            'filters': [
                {'col': 'database', 'opr': 'rel_o_m', 'value': f'{example_db.id}'},
                {'col': 'table_name', 'opr': 'eq', 'value': 'birth_names'},
            ]
        }
        uri = f'api/v1/dataset/?q={prison.dumps(arguments)}'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['count'] == 1
        expected_columns = [
            'catalog',
            'changed_by',
            'changed_by_name',
            'changed_on_delta_humanized',
            'changed_on_utc',
            'database',
            'datasource_type',
            'default_endpoint',
            'description',
            'explore_url',
            'extra',
            'id',
            'kind',
            'owners',
            'schema',
            'sql',
            'table_name',
        ]
        assert sorted(list(response['result'][0].keys())) == expected_columns

    def test_get_dataset_list_gamma(self) -> None:
        """
        Dataset API: Test get dataset list gamma
        """
        if backend() == 'postgresql':
            return
        self.login(GAMMA_USERNAME)
        uri = 'api/v1/dataset/'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['result'] == []

    def test_get_dataset_list_gamma_has_database_access(self) -> None:
        """
        Dataset API: Test get dataset list with database access
        """
        if backend() == 'postgresql':
            return
        self.login(GAMMA_USERNAME)
        main_db = get_main_database()
        dataset = self.insert_dataset('ab_user', [], main_db)
        uri = 'api/v1/dataset/'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['count'] == 0
        main_db_pvm = security_manager.find_permission_view_menu(
            'database_access', main_db.perm
        )
        gamma_role = security_manager.find_role('Gamma')
        gamma_role.permissions.append(main_db_pvm)
        db.session.commit()
        uri = 'api/v1/dataset/'
        rv = self.get_assert_metric(uri, 'get_list')
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        tables = {tbl['table_name'] for tbl in response['result']}
        assert tables == {'ab_user'}
        gamma_role.permissions.remove(main_db_pvm)
        db.session.delete(dataset)
        db.session.commit()

    def test_get_dataset_related_database_gamma(self) -> None:
        """
        Dataset API: Test get dataset related databases gamma
        """
        main_db = get_main_database()
        main_db_pvm = security_manager.find_permission_view_menu(
            'database_access', main_db.perm
        )
        gamma_role = security_manager.find_role('Gamma')
        gamma_role.permissions.append(main_db_pvm)
        db.session.commit()
        self.login(GAMMA_USERNAME)
        uri = 'api/v1/dataset/related/database'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['count'] == 1
        main_db = get_main_database()
        assert filter(lambda x: x.text == main_db, response['result']) != []
        gamma_role.permissions.remove(main_db_pvm)
        db.session.commit()

    @pytest.mark.usefixtures('load_energy_table_with_slice')
    def test_get_dataset_item(self) -> None:
        """
        Dataset API: Test get dataset item
        """
        table = self.get_energy_usage_dataset()
        main_db = get_main_database()
        self.login(ADMIN_USERNAME)
        uri = f'api/v1/dataset/{table.id}'
        rv = self.get_assert_metric(uri, 'get')
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        expected_result = {
            'cache_timeout': None,
            'database': {
                'allow_multi_catalog': False,
                'backend': main_db.backend,
                'database_name': 'examples',
                'id': 1,
            },
            'default_endpoint': None,
            'description': 'Energy consumption',
            'extra': None,
            'fetch_values_predicate': None,
            'filter_select_enabled': True,
            'is_sqllab_view': False,
            'kind': 'physical',
            'main_dttm_col': None,
            'offset': 0,
            'owners': [],
            'schema': get_example_default_schema(),
            'sql': None,
            'table_name': 'energy_usage',
            'template_params': None,
            'uid': ANY,
            'datasource_name': 'energy_usage',
            'name': f'{get_example_default_schema()}.energy_usage',
            'column_formats': {},
            'granularity_sqla': [],
            'time_grain_sqla': ANY,
            'order_by_choices': [
                ['["source", true]', 'source [asc]'],
                ['["source", false]', 'source [desc]'],
                ['["target", true]', 'target [asc]'],
                ['["target", false]', 'target [desc]'],
                ['["value", true]', 'value [asc]'],
                ['["value", false]', 'value [desc]'],
            ],
            'verbose_map': {
                '__timestamp': 'Time',
                'count': 'COUNT(*)',
                'source': 'source',
                'sum__value': 'sum__value',
                'target': 'target',
                'value': 'value',
            },
        }
        if response['result']['database']['backend'] not in ('presto', 'hive'):
            assert {
                k: v
                for k, v in response['result'].items()
                if k in expected_result
            } == expected_result
        assert len(response['result']['columns']) == 3
        assert len(response['result']['metrics']) == 2

    def test_get_dataset_render_jinja(self) -> None:
        """
        Dataset API: Test get dataset with the render parameter.
        """
        database = get_example_database()
        dataset = SqlaTable(
            table_name='test_sql_table_with_jinja',
            database=database,
            schema=get_example_default_schema(),
            main_dttm_col='default_dttm',
            columns=[
                TableColumn(
                    column_name='my_user_id',
                    type='INTEGER',
                    is_dttm=False,
                ),
                TableColumn(
                    column_name='calculated_test',
                    type='VARCHAR(255)',
                    is_dttm=False,
                    expression="'{{ current_username() }}'",
                ),
            ],
            metrics=[
                SqlMetric(
                    metric_name='param_test',
                    expression="{{ url_param('multiplier') }} * 1.4",
                )
            ],
            sql='SELECT {{ current_user_id() }} as my_user_id',
        )
        db.session.add(dataset)
        db.session.commit()
        self.login(ADMIN_USERNAME)
        admin = self.get_user(ADMIN_USERNAME)
        uri = f'api/v1/dataset/{dataset.id}?q=(columns:!(id,sql,columns.column_name,columns.expression,metrics.metric_name,metrics.expression))&include_rendered_sql=true&multiplier=4'
        rv = self.get_assert_metric(uri, 'get')
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['result'] == {
            'id': dataset.id,
            'sql': 'SELECT {{ current_user_id() }} as my_user_id',
            'rendered_sql': f'SELECT {admin.id} as my_user_id',
            'columns': [
                {'column_name': 'my_user_id', 'expression': None},
                {
                    'column_name': 'calculated_test',
                    'expression': "'{{ current_username() }}'",
                    'rendered_expression': f"'{admin.username}'",
                },
            ],
            'metrics': [
                {
                    'metric_name': 'param_test',
                    'expression': "{{ url_param('multiplier') }} * 1.4",
                    'rendered_expression': '4 * 1.4',
                }
            ],
        }
        db.session.delete(dataset)
        db.session.commit()

    def test_get_dataset_render_jinja_exceptions(self) -> None:
        """
        Dataset API: Test get dataset with the render parameter
        when rendering raises an exception.
        """
        database = get_example_database()
        dataset = SqlaTable(
            table_name='test_sql_table_with_incorrect_jinja',
            database=database,
            schema=get_example_default_schema(),
            main_dttm_col='default_dttm',
            columns=[
                TableColumn(
                    column_name='my_user_id',
                    type='INTEGER',
                    is_dttm=False,
                ),
                TableColumn(
                    column_name='calculated_test',
                    type='VARCHAR(255)',
                    is_dttm=False,
                    expression="'{{ current_username() }'",
                ),
            ],
            metrics=[
                SqlMetric(
                    metric_name='param_test',
                    expression="{{ url_param('multiplier') } * 1.4",
                )
            ],
            sql='SELECT {{ current_user_id() } as my_user_id',
        )
        db.session.add(dataset)
        db.session.commit()
        self.login(ADMIN_USERNAME)
        uri = f'api/v1/dataset/{dataset.id}?q=(columns:!(id,sql))&include_rendered_sql=true'
        rv = self.get_assert_metric(uri, 'get')
        assert rv.status_code == 400
        response = json.loads(rv.data.decode('utf-8'))
        assert response['message'] == 'Unable to render expression from dataset query.'
        uri = f'api/v1/dataset/{dataset.id}?q=(columns:!(id,metrics.expression))&include_rendered_sql=true&multiplier=4'
        rv = self.get_assert_metric(uri, 'get')
        assert rv.status_code == 400
        response = json.loads(rv.data.decode('utf-8'))
        assert response['message'] == 'Unable to render expression from dataset metric.'
        uri = f'api/v1/dataset/{dataset.id}?q=(columns:!(id,columns.expression))&include_rendered_sql=true'
        rv = self.get_assert_metric(uri, 'get')
        assert rv.status_code == 400
        response = json.loads(rv.data.decode('utf-8'))
        assert (
            response['message']
            == 'Unable to render expression from dataset calculated column.'
        )
        db.session.delete(dataset)
        db.session.commit()

