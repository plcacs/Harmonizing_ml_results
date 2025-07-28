from typing import Any, Iterator
import dataclasses
from collections import defaultdict
from io import BytesIO
from unittest import mock
from unittest.mock import patch, MagicMock, Mock
from zipfile import is_zipfile, ZipFile
import prison
import pytest
import yaml
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql import func
from superset import db, security_manager
from superset.connectors.sqla.models import SqlaTable
from superset.databases.ssh_tunnel.models import SSHTunnel
from superset.databases.utils import make_url_safe
from superset.db_engine_specs.mysql import MySQLEngineSpec
from superset.db_engine_specs.postgres import PostgresEngineSpec
from superset.db_engine_specs.redshift import RedshiftEngineSpec
from superset.db_engine_specs.bigquery import BigQueryEngineSpec
from superset.db_engine_specs.gsheets import GSheetsEngineSpec
from superset.db_engine_specs.hana import HanaEngineSpec
from superset.errors import SupersetError
from superset.models.core import Database, ConfigurationMethod
from superset.reports.models import ReportSchedule, ReportScheduleType
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_USERNAME
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.energy_dashboard import load_energy_table_with_slice, load_energy_table_data
from tests.integration_tests.fixtures.world_bank_dashboard import load_world_bank_dashboard_with_slices, load_world_bank_data
from tests.integration_tests.fixtures.importexport import (
    database_config,
    dataset_config,
    database_metadata_config,
    dataset_metadata_config,
    database_with_ssh_tunnel_config_password,
    database_with_ssh_tunnel_config_private_key,
    database_with_ssh_tunnel_config_mix_credentials,
    database_with_ssh_tunnel_config_no_credentials,
    database_with_ssh_tunnel_config_private_pass_only,
)
from tests.integration_tests.fixtures.unicode_dashboard import load_unicode_dashboard_with_position, load_unicode_data
from tests.integration_tests.test_app import app

SQL_VALIDATORS_BY_ENGINE = {'presto': 'PrestoDBSQLValidator', 'postgresql': 'PostgreSQLValidator'}
PRESTO_SQL_VALIDATORS_BY_ENGINE = {'presto': 'PrestoDBSQLValidator', 'sqlite': 'PrestoDBSQLValidator', 'postgresql': 'PrestoDBSQLValidator', 'mysql': 'PrestoDBSQLValidator'}

class TestDatabaseApi(SupersetTestCase):

    def insert_database(
        self,
        database_name: str,
        sqlalchemy_uri: str,
        extra: str = '',
        encrypted_extra: str = '',
        server_cert: str = '',
        expose_in_sqllab: bool = False,
        allow_file_upload: bool = False,
    ) -> Database:
        database: Database = Database(
            database_name=database_name,
            sqlalchemy_uri=sqlalchemy_uri,
            extra=extra,
            encrypted_extra=encrypted_extra,
            server_cert=server_cert,
            expose_in_sqllab=expose_in_sqllab,
            allow_file_upload=allow_file_upload,
        )
        db.session.add(database)
        db.session.commit()
        return database

    @pytest.fixture
    def create_database_with_report(self) -> Iterator[Database]:
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            database: Database = self.insert_database(
                'database_with_report', example_db.sqlalchemy_uri_decrypted, expose_in_sqllab=True
            )
            report_schedule = ReportSchedule(
                type=ReportScheduleType.ALERT, name='report_with_database', crontab='* * * * *', database=database
            )
            db.session.add(report_schedule)
            db.session.commit()
            yield database
            db.session.delete(report_schedule)
            db.session.delete(database)
            db.session.commit()

    @pytest.fixture
    def create_database_with_dataset(self) -> Iterator[Database]:
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            self._database = self.insert_database(
                'database_with_dataset', example_db.sqlalchemy_uri_decrypted, expose_in_sqllab=True
            )
            table = SqlaTable(schema='main', table_name='ab_permission', database=self._database)
            db.session.add(table)
            db.session.commit()
            yield self._database
            db.session.delete(table)
            db.session.delete(self._database)
            db.session.commit()
            self._database = None

    def create_database_import(self) -> BytesIO:
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        return buf

    def test_get_items(self) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        expected_columns = [
            'allow_ctas',
            'allow_cvas',
            'allow_dml',
            'allow_file_upload',
            'allow_multi_catalog',
            'allow_run_async',
            'allows_cost_estimate',
            'allows_subquery',
            'allows_virtual_table_explore',
            'backend',
            'changed_by',
            'changed_on',
            'changed_on_delta_humanized',
            'created_by',
            'database_name',
            'disable_data_preview',
            'disable_drill_to_detail',
            'engine_information',
            'explore_database_id',
            'expose_in_sqllab',
            'extra',
            'force_ctas_schema',
            'id',
            'uuid',
        ]
        assert response['count'] > 0
        assert list(response['result'][0].keys()) == expected_columns

    def test_get_items_filter(self) -> None:
        example_db: Database = get_example_database()
        test_database: Database = self.insert_database('test-database', example_db.sqlalchemy_uri_decrypted, expose_in_sqllab=True)
        dbs = db.session.query(Database).filter_by(expose_in_sqllab=True).all()
        self.login(ADMIN_USERNAME)
        arguments = {
            'keys': ['none'],
            'filters': [{'col': 'expose_in_sqllab', 'opr': 'eq', 'value': True}],
            'order_columns': 'database_name',
            'order_direction': 'asc',
            'page': 0,
            'page_size': -1,
        }
        uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
        rv = self.client.get(uri)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response['count'] == len(dbs)
        db.session.delete(test_database)
        db.session.commit()

    def test_get_items_not_allowed(self) -> None:
        self.login(GAMMA_USERNAME)
        uri: str = 'api/v1/database/'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['count'] == 0

    def test_create_database(self) -> None:
        extra = {'metadata_params': {}, 'engine_params': {}, 'metadata_cache_timeout': {}, 'schemas_allowed_for_file_upload': []}
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        database_data = {
            'database_name': 'test-create-database',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
            'server_cert': None,
            'extra': json.dumps(extra),
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        model = db.session.query(Database).get(response.get('id'))
        assert model.configuration_method == ConfigurationMethod.SQLALCHEMY_FORM
        db.session.delete(model)
        db.session.commit()

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_create_database_with_ssh_tunnel(
        self,
        mock_get_all_schema_names: Any,
        mock_get_all_catalog_names: Any,
        mock_create_is_feature_enabled: Any,
        mock_test_connection_database_command_run: Any,
    ) -> None:
        mock_create_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'foo', 'password': 'bar'}
        database_data = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'ssh_tunnel': ssh_tunnel_properties,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == response.get('id')).one()
        assert response.get('result')['ssh_tunnel']['password'] == 'XXXXXXXXXX'
        assert model_ssh_tunnel.database_id == response.get('id')
        model = db.session.query(Database).get(response.get('id'))
        db.session.delete(model)
        db.session.commit()

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_create_database_with_missing_port_raises_error(
        self,
        mock_get_all_schema_names: Any,
        mock_get_all_catalog_names: Any,
        mock_create_is_feature_enabled: Any,
        mock_test_connection_database_command_run: Any,
    ) -> None:
        mock_create_is_feature_enabled.return_value = True
        self.login(username='admin')
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        modified_sqlalchemy_uri: str = 'postgresql://foo:bar@localhost/test-db'
        ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'foo', 'password': 'bar'}
        database_data_with_ssh_tunnel = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': modified_sqlalchemy_uri,
            'ssh_tunnel': ssh_tunnel_properties,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data_with_ssh_tunnel)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert response.get('message') == 'A database port is required when connecting via SSH Tunnel.'

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_update_database_with_ssh_tunnel(
        self,
        mock_get_all_schema_names: Any,
        mock_get_all_catalog_names: Any,
        mock_update_is_feature_enabled: Any,
        mock_create_is_feature_enabled: Any,
        mock_test_connection_database_command_run: Any,
    ) -> None:
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'foo', 'password': 'bar'}
        database_data = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
        }
        database_data_with_ssh_tunnel = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'ssh_tunnel': ssh_tunnel_properties,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        uri = f"api/v1/database/{response.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel)
        response_update = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == response_update.get('id')).one()
        assert model_ssh_tunnel.database_id == response_update.get('id')
        model = db.session.query(Database).get(response.get('id'))
        db.session.delete(model)
        db.session.commit()

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_update_database_with_missing_port_raises_error(
        self,
        mock_get_all_schema_names: Any,
        mock_get_all_catalog_names: Any,
        mock_update_is_feature_enabled: Any,
        mock_create_is_feature_enabled: Any,
        mock_test_connection_database_command_run: Any,
    ) -> None:
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(username='admin')
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        modified_sqlalchemy_uri: str = 'postgresql://foo:bar@localhost/test-db'
        ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'foo', 'password': 'bar'}
        database_data_with_ssh_tunnel = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': modified_sqlalchemy_uri,
            'ssh_tunnel': ssh_tunnel_properties,
        }
        database_data = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response_create = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        uri = f"api/v1/database/{response_create.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert response.get('message') == 'A database port is required when connecting via SSH Tunnel.'
        model = db.session.query(Database).get(response_create.get('id'))
        db.session.delete(model)
        db.session.commit()

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.commands.database.ssh_tunnel.delete.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_delete_ssh_tunnel(
        self,
        mock_get_all_schema_names: Any,
        mock_get_all_catalog_names: Any,
        mock_delete_is_feature_enabled: Any,
        mock_update_is_feature_enabled: Any,
        mock_create_is_feature_enabled: Any,
        mock_test_connection_database_command_run: Any,
    ) -> None:
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        mock_delete_is_feature_enabled.return_value = True
        self.login(username='admin')
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'foo', 'password': 'bar'}
        database_data = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
        }
        database_data_with_ssh_tunnel = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'ssh_tunnel': ssh_tunnel_properties,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        uri = f"api/v1/database/{response.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel)
        response_update = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == response_update.get('id')).one()
        assert model_ssh_tunnel.database_id == response_update.get('id')
        database_data_with_ssh_tunnel_null = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'ssh_tunnel': None,
        }
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel_null)
        response_update = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == response_update.get('id')).one_or_none()
        assert model_ssh_tunnel is None
        model = db.session.query(Database).get(response.get('id'))
        db.session.delete(model)
        db.session.commit()

    @mock.patch('superset.commands.database.test_connection.TestConnectionDatabaseCommand.run')
    @mock.patch('superset.commands.database.create.is_feature_enabled')
    @mock.patch('superset.commands.database.update.is_feature_enabled')
    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_update_ssh_tunnel_via_database_api(
        self,
        mock_get_all_schema_names: Any,
        mock_get_all_catalog_names: Any,
        mock_update_is_feature_enabled: Any,
        mock_create_is_feature_enabled: Any,
        mock_test_connection_database_command_run: Any,
    ) -> None:
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        initial_ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'foo', 'password': 'bar'}
        updated_ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'Test', 'password': 'new_bar'}
        database_data_with_ssh_tunnel = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'ssh_tunnel': initial_ssh_tunnel_properties,
        }
        database_data_with_ssh_tunnel_update = {
            'database_name': 'test-db-with-ssh-tunnel',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'ssh_tunnel': updated_ssh_tunnel_properties,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data_with_ssh_tunnel)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == response.get('id')).one()
        assert model_ssh_tunnel.database_id == response.get('id')
        assert model_ssh_tunnel.username == 'foo'
        uri = f"api/v1/database/{response.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel_update)
        response_update = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == response_update.get('id')).one()
        assert model_ssh_tunnel.database_id == response_update.get('id')
        assert response_update.get('result')['ssh_tunnel']['password'] == 'XXXXXXXXXX'
        assert model_ssh_tunnel.username == 'Test'
        assert model_ssh_tunnel.server_address == '123.132.123.1'
        assert model_ssh_tunnel.server_port == 8080
        model = db.session.query(Database).get(response.get('id'))
        db.session.delete(model)
        db.session.commit()

    @mock.patch('superset.models.core.Database.get_all_catalog_names')
    @mock.patch('superset.models.core.Database.get_all_schema_names')
    def test_if_ssh_tunneling_flag_is_not_active_it_raises_new_exception(
        self, mock_get_all_schema_names: Any, mock_get_all_catalog_names: Any
    ) -> None:
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        ssh_tunnel_properties = {'server_address': '123.132.123.1', 'server_port': 8080, 'username': 'foo', 'password': 'bar'}
        database_data = {
            'database_name': 'test-db-with-ssh-tunnel-7',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'ssh_tunnel': ssh_tunnel_properties,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert response == {'message': 'SSH Tunneling is not enabled'}
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == response.get('id')).one_or_none()
        assert model_ssh_tunnel is None
        model = db.session.query(Database).filter(Database.database_name == 'test-db-with-ssh-tunnel-7').one_or_none()
        assert model is None

    def test_get_table_details_with_slash_in_table_name(self) -> None:
        table_name: str = 'table_with/slash'
        database = get_example_database()
        query = f'CREATE TABLE IF NOT EXISTS "{table_name}" (col VARCHAR(256))'
        if database.backend == 'mysql':
            query = query.replace('"', '`')
        with database.get_sqla_engine() as engine:
            engine.execute(query)
        self.login(ADMIN_USERNAME)
        uri: str = f'api/v1/database/{database.id}/table/{table_name}/null/'
        rv = self.client.get(uri)
        assert rv.status_code == 200

    def test_create_database_invalid_configuration_method(self) -> None:
        extra = {'metadata_params': {}, 'engine_params': {}, 'metadata_cache_timeout': {}, 'schemas_allowed_for_file_upload': []}
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        database_data = {
            'database_name': 'test-create-database',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': 'BAD_FORM',
            'server_cert': None,
            'extra': json.dumps(extra),
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert response == {'message': {'configuration_method': ['Must be one of: sqlalchemy_form, dynamic_form.']}}
        assert rv.status_code == 400

    def test_create_database_no_configuration_method(self) -> None:
        extra = {'metadata_params': {}, 'engine_params': {}, 'metadata_cache_timeout': {}, 'schemas_allowed_for_file_upload': []}
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        database_data = {
            'database_name': 'test-create-database',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'server_cert': None,
            'extra': json.dumps(extra),
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        assert 'sqlalchemy_form' in response['result']['configuration_method']

    def test_create_database_server_cert_validate(self) -> None:
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        self.login(ADMIN_USERNAME)
        database_data = {
            'database_name': 'test-create-database-invalid-cert',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
            'server_cert': 'INVALID CERT',
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {'message': {'server_cert': ['Invalid certificate']}}
        assert rv.status_code == 400
        assert response == expected_response

    def test_create_database_json_validate(self) -> None:
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        self.login(ADMIN_USERNAME)
        database_data = {
            'database_name': 'test-create-database-invalid-json',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
            'masked_encrypted_extra': '{"A": "a", "B", "C"}',
            'extra': '["A": "a", "B", "C"]',
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {
            'message': {
                'extra': ["Field cannot be decoded by JSON. Expecting ',' delimiter or ']': line 1 column 5 (char 4)"],
                'masked_encrypted_extra': ["Field cannot be decoded by JSON. Expecting ':' delimiter: line 1 column 15 (char 14)"],
            }
        }
        assert rv.status_code == 400
        assert response == expected_response

    def test_create_database_extra_metadata_validate(self) -> None:
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        extra = {'metadata_params': {'wrong_param': 'some_value'}, 'engine_params': {}, 'metadata_cache_timeout': {}, 'schemas_allowed_for_file_upload': []}
        self.login(ADMIN_USERNAME)
        database_data = {
            'database_name': 'test-create-database-invalid-extra',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
            'extra': json.dumps(extra),
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {'message': {'extra': ['The metadata_params in Extra field is not configured correctly. The key wrong_param is invalid.']}}
        assert rv.status_code == 400
        assert response == expected_response

    def test_create_database_unique_validate(self) -> None:
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        self.login(ADMIN_USERNAME)
        database_data = {
            'database_name': 'examples',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {'message': {'database_name': 'A database with the same name already exists.'}}
        assert rv.status_code == 422
        assert response == expected_response

    def test_create_database_uri_validate(self) -> None:
        self.login(ADMIN_USERNAME)
        database_data = {
            'database_name': 'test-database-invalid-uri',
            'sqlalchemy_uri': 'wrong_uri',
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert 'Invalid connection string' in response['message']['sqlalchemy_uri'][0]

    @mock.patch('superset.views.core.app.config', {**app.config, 'PREVENT_UNSAFE_DB_CONNECTIONS': True})
    def test_create_database_fail_sqlite(self) -> None:
        database_data = {
            'database_name': 'test-create-sqlite-database',
            'sqlalchemy_uri': 'sqlite:////some.db',
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = 'api/v1/database/'
        self.login(ADMIN_USERNAME)
        response = self.client.post(uri, json=database_data)
        response_data = json.loads(response.data.decode('utf-8'))
        expected_response = {'message': {'sqlalchemy_uri': ['SQLiteDialect_pysqlite cannot be used as a data source for security reasons.']}}
        assert response_data == expected_response
        assert response.status_code == 400

    def test_create_database_conn_fail(self) -> None:
        example_db = get_example_database()
        if example_db.backend in ('sqlite', 'hive', 'presto'):
            return
        example_db.password = 'wrong_password'
        database_data = {
            'database_name': 'test-create-database-wrong-password',
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = 'api/v1/database/'
        self.login(ADMIN_USERNAME)
        response = self.client.post(uri, json=database_data)
        response_data = json.loads(response.data.decode('utf-8'))
        superset_error_mysql = SupersetError(
            message='Either the username "superset" or the password is incorrect.',
            error_type='CONNECTION_ACCESS_DENIED_ERROR',
            level='error',
            extra={
                'engine_name': 'MySQL',
                'invalid': ['username', 'password'],
                'issue_codes': [
                    {'code': 1014, 'message': 'Issue 1014 - Either the username or the password is wrong.'},
                    {'code': 1015, 'message': 'Issue 1015 - Issue 1015 - Either the database is spelled incorrectly or does not exist.'},
                ],
            },
        )
        superset_error_postgres = SupersetError(
            message='The password provided for username "superset" is incorrect.',
            error_type='CONNECTION_INVALID_PASSWORD_ERROR',
            level='error',
            extra={
                'engine_name': 'PostgreSQL',
                'invalid': ['username', 'password'],
                'issue_codes': [{'code': 1013, 'message': 'Issue 1013 - The password provided when connecting to a database is not valid.'}],
            },
        )
        expected_response_mysql = {'errors': [dataclasses.asdict(superset_error_mysql)]}
        expected_response_postgres = {'errors': [dataclasses.asdict(superset_error_postgres)]}
        assert response.status_code == 500
        if example_db.backend == 'mysql':
            assert response_data == expected_response_mysql
        else:
            assert response_data == expected_response_postgres

    def test_update_database(self) -> None:
        example_db = get_example_database()
        test_database = self.insert_database('test-database', example_db.sqlalchemy_uri_decrypted)
        self.login(ADMIN_USERNAME)
        database_data = {
            'database_name': 'test-database-updated',
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        assert rv.status_code == 200
        model = db.session.query(Database).get(test_database.id)
        db.session.delete(model)
        db.session.commit()

    def test_update_database_conn_fail(self) -> None:
        example_db = get_example_database()
        if example_db.backend in ('sqlite', 'hive', 'presto'):
            return
        test_database = self.insert_database('test-database1', example_db.sqlalchemy_uri_decrypted)
        example_db.password = 'wrong_password'
        database_data = {'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted}
        uri: str = f"api/v1/database/{test_database.id}"
        self.login(ADMIN_USERNAME)
        rv = self.client.put(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {'message': 'Connection failed, please check your connection settings'}
        assert rv.status_code == 422
        assert response == expected_response
        model = db.session.query(Database).get(test_database.id)
        db.session.delete(model)
        db.session.commit()

    def test_update_database_uniqueness(self) -> None:
        example_db = get_example_database()
        test_database1 = self.insert_database('test-database1', example_db.sqlalchemy_uri_decrypted)
        test_database2 = self.insert_database('test-database2', example_db.sqlalchemy_uri_decrypted)
        self.login(ADMIN_USERNAME)
        database_data = {'database_name': 'test-database2'}
        uri: str = f"api/v1/database/{test_database1.id}"
        rv = self.client.put(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {'message': {'database_name': 'A database with the same name already exists.'}}
        assert rv.status_code == 422
        assert response == expected_response
        db.session.delete(test_database1)
        db.session.delete(test_database2)
        db.session.commit()

    def test_update_database_invalid(self) -> None:
        self.login(ADMIN_USERNAME)
        database_data = {'database_name': 'test-database-updated'}
        uri: str = 'api/v1/database/invalid'
        rv = self.client.put(uri, json=database_data)
        assert rv.status_code == 404

    def test_update_database_uri_validate(self) -> None:
        example_db = get_example_database()
        test_database = self.insert_database('test-database', example_db.sqlalchemy_uri_decrypted)
        self.login(ADMIN_USERNAME)
        database_data = {'database_name': 'test-database-updated', 'sqlalchemy_uri': 'wrong_uri'}
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert 'Invalid connection string' in response['message']['sqlalchemy_uri'][0]
        db.session.delete(test_database)
        db.session.commit()

    def test_update_database_with_invalid_configuration_method(self) -> None:
        example_db = get_example_database()
        test_database = self.insert_database('test-database', example_db.sqlalchemy_uri_decrypted)
        self.login(ADMIN_USERNAME)
        database_data = {'database_name': 'test-database-updated', 'configuration_method': 'BAD_FORM'}
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        response = json.loads(rv.data.decode('utf-8'))
        assert response == {'message': {'configuration_method': ['Must be one of: sqlalchemy_form, dynamic_form.']}}
        assert rv.status_code == 400
        db.session.delete(test_database)
        db.session.commit()

    def test_update_database_with_no_configuration_method(self) -> None:
        example_db = get_example_database()
        test_database = self.insert_database('test-database', example_db.sqlalchemy_uri_decrypted)
        self.login(ADMIN_USERNAME)
        database_data = {'database_name': 'test-database-updated'}
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        assert rv.status_code == 200
        db.session.delete(test_database)
        db.session.commit()

    def test_delete_database(self) -> None:
        database_id: int = self.insert_database('test-database', 'test_uri').id
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{database_id}"
        rv = self.delete_assert_metric(uri, 'delete')
        assert rv.status_code == 200
        model = db.session.query(Database).get(database_id)
        assert model is None

    def test_delete_database_not_found(self) -> None:
        max_id: Any = db.session.query(func.max(Database.id)).scalar()
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{max_id + 1}"
        rv = self.delete_assert_metric(uri, 'delete')
        assert rv.status_code == 404

    @pytest.mark.usefixtures('create_database_with_dataset')
    def test_delete_database_with_datasets(self) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{self._database.id}"
        rv = self.delete_assert_metric(uri, 'delete')
        assert rv.status_code == 422

    @pytest.mark.usefixtures('create_database_with_report')
    def test_delete_database_with_report(self) -> None:
        self.login(ADMIN_USERNAME)
        database = db.session.query(Database).filter(Database.database_name == 'database_with_report').one_or_none()
        uri: str = f"api/v1/database/{database.id}"
        rv = self.client.delete(uri)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        expected_response = {'message': 'There are associated alerts or reports: report_with_database'}
        assert response == expected_response

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_table_metadata(self) -> None:
        example_db = get_example_database()
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{example_db.id}/table/birth_names/null/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['name'] == 'birth_names'
        assert response['comment'] is None
        assert len(response['columns']) > 5
        assert response.get('selectStar').startswith('SELECT')

    def test_info_security_database(self) -> None:
        self.login(ADMIN_USERNAME)
        params = {'keys': ['permissions']}
        uri: str = f"api/v1/database/_info?q={prison.dumps(params)}"
        rv = self.get_assert_metric(uri, 'info')
        data = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert set(data['permissions']) == {'can_read', 'can_write', 'can_export', 'can_upload'}

    def test_get_invalid_database_table_metadata(self) -> None:
        database_id: int = 1000
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{database_id}/table/some_table/some_schema/"
        rv = self.client.get(uri)
        assert rv.status_code == 404
        uri = 'api/v1/database/some_database/table/some_table/some_schema/'
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_invalid_table_table_metadata(self) -> None:
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/table/wrong_table/null/"
        self.login(ADMIN_USERNAME)
        rv = self.client.get(uri)
        data = json.loads(rv.data.decode('utf-8'))
        if example_db.backend == 'sqlite':
            assert rv.status_code == 200
            assert data == {
                'columns': [],
                'comment': None,
                'foreignKeys': [],
                'indexes': [],
                'name': 'wrong_table',
                'primaryKey': {'constrained_columns': None, 'name': None},
                'selectStar': 'SELECT\n  *\nFROM wrong_table\nLIMIT 100\nOFFSET 0',
            }
        elif example_db.backend == 'mysql':
            assert rv.status_code == 422
            assert data == {'message': '`wrong_table`'}
        else:
            assert rv.status_code == 422
            assert data == {'message': 'wrong_table'}

    def test_get_table_metadata_no_db_permission(self) -> None:
        self.login(GAMMA_USERNAME)
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/birth_names/null/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_table_extra_metadata_deprecated(self) -> None:
        example_db = get_example_database()
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{example_db.id}/table_extra/birth_names/null/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response == {}

    def test_get_invalid_database_table_extra_metadata_deprecated(self) -> None:
        database_id: int = 1000
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{database_id}/table_extra/some_table/some_schema/"
        rv = self.client.get(uri)
        assert rv.status_code == 404
        uri = 'api/v1/database/some_database/table_extra/some_table/some_schema/'
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_invalid_table_table_extra_metadata_deprecated(self) -> None:
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/table_extra/wrong_table/null/"
        self.login(ADMIN_USERNAME)
        rv = self.client.get(uri)
        data = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert data == {}

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_select_star(self) -> None:
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/select_star/birth_names/"
        rv = self.client.get(uri)
        assert rv.status_code == 200

    def test_get_select_star_not_allowed(self) -> None:
        self.login(GAMMA_USERNAME)
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/select_star/birth_names/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_select_star_not_found_database(self) -> None:
        self.login(ADMIN_USERNAME)
        max_id: Any = db.session.query(func.max(Database.id)).scalar()
        uri: str = f"api/v1/database/{max_id + 1}/select_star/birth_names/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_select_star_not_found_table(self) -> None:
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        uri: str = f"api/v1/database/{example_db.id}/select_star/table_does_not_exist/"
        rv = self.client.get(uri)
        assert rv.status_code == (404 if example_db.backend != 'presto' else 500)

    def test_get_allow_file_upload_filter(self) -> None:
        with self.create_app().app_context():
            example_db = get_example_database()
            extra = {
                'metadata_params': {},
                'engine_params': {},
                'metadata_cache_timeout': {},
                'schemas_allowed_for_file_upload': ['public'],
            }
            self.login(ADMIN_USERNAME)
            database = self.insert_database('database_with_upload', example_db.sqlalchemy_uri_decrypted, extra=json.dumps(extra), allow_file_upload=True)
            db.session.commit()
            yield database
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 1
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_filter_no_schema(self) -> None:
        with self.create_app().app_context():
            example_db = get_example_database()
            extra = {
                'metadata_params': {},
                'engine_params': {},
                'metadata_cache_timeout': {},
                'schemas_allowed_for_file_upload': [],
            }
            self.login(ADMIN_USERNAME)
            database = self.insert_database('database_with_upload', example_db.sqlalchemy_uri_decrypted, extra=json.dumps(extra), allow_file_upload=True)
            db.session.commit()
            yield database
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_filter_allow_file_false(self) -> None:
        with self.create_app().app_context():
            example_db = get_example_database()
            extra = {
                'metadata_params': {},
                'engine_params': {},
                'metadata_cache_timeout': {},
                'schemas_allowed_for_file_upload': ['public'],
            }
            self.login(ADMIN_USERNAME)
            database = self.insert_database('database_with_upload', example_db.sqlalchemy_uri_decrypted, extra=json.dumps(extra), allow_file_upload=False)
            db.session.commit()
            yield database
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_false(self) -> None:
        with self.create_app().app_context():
            example_db = get_example_database()
            extra = {
                'metadata_params': {},
                'engine_params': {},
                'metadata_cache_timeout': {},
                'schemas_allowed_for_file_upload': [],
            }
            self.login(ADMIN_USERNAME)
            database = self.insert_database('database_with_upload', example_db.sqlalchemy_uri_decrypted, extra=json.dumps(extra), allow_file_upload=False)
            db.session.commit()
            yield database
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_false_no_extra(self) -> None:
        with self.create_app().app_context():
            example_db = get_example_database()
            self.login(ADMIN_USERNAME)
            database = self.insert_database('database_with_upload', example_db.sqlalchemy_uri_decrypted, allow_file_upload=False)
            db.session.commit()
            yield database
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 0
            db.session.delete(database)
            db.session.commit()

    def mock_csv_function(d: Any, user: Any) -> Any:
        return d.get_all_schema_names()

    @mock.patch('superset.views.core.app.config', {**app.config, 'ALLOWED_USER_CSV_SCHEMA_FUNC': mock_csv_function})
    def test_get_allow_file_upload_true_csv(self) -> None:
        with self.create_app().app_context():
            example_db = get_example_database()
            extra = {
                'metadata_params': {},
                'engine_params': {},
                'metadata_cache_timeout': {},
                'schemas_allowed_for_file_upload': [],
            }
            self.login(ADMIN_USERNAME)
            database = self.insert_database('database_with_upload', example_db.sqlalchemy_uri_decrypted, extra=json.dumps(extra), allow_file_upload=True)
            db.session.commit()
            yield database
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 1
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_filter_no_permission(self) -> None:
        with self.create_app().app_context():
            example_db = get_example_database()
            extra = {
                'metadata_params': {},
                'engine_params': {},
                'metadata_cache_timeout': {},
                'schemas_allowed_for_file_upload': ['public'],
            }
            self.login(GAMMA_USERNAME)
            database = self.insert_database('database_with_upload', example_db.sqlalchemy_uri_decrypted, extra=json.dumps(extra), allow_file_upload=True)
            db.session.commit()
            yield database
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_filter_with_permission(self) -> None:
        with self.create_app().app_context():
            main_db = get_main_database()
            main_db.allow_file_upload = True
            table = SqlaTable(schema='public', table_name='ab_permission', database=get_main_database())
            db.session.add(table)
            db.session.commit()
            tmp_table_perm = security_manager.find_permission_view_menu('datasource_access', table.get_perm())
            gamma_role = security_manager.find_role('Gamma')
            security_manager.add_permission_role(gamma_role, tmp_table_perm)
            self.login(GAMMA_USERNAME)
            arguments = {'columns': ['allow_file_upload'], 'filters': [{'col': 'allow_file_upload', 'opr': 'upload_is_enabled', 'value': True}]}
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 1
            security_manager.del_permission_role(gamma_role, tmp_table_perm)
            db.session.delete(table)
            db.session.delete(main_db)
            db.session.commit()

    def test_database_schemas(self) -> None:
        self.login(ADMIN_USERNAME)
        database = db.session.query(Database).filter_by(database_name='examples').one()
        schemas = database.get_all_schema_names()
        rv = self.client.get(f"api/v1/database/{database.id}/schemas/")
        response = json.loads(rv.data.decode('utf-8'))
        assert schemas == set(response['result'])
        rv = self.client.get(f"api/v1/database/{database.id}/schemas/?q={prison.dumps({'force': True})}")
        response = json.loads(rv.data.decode('utf-8'))
        assert schemas == set(response['result'])

    def test_database_schemas_not_found(self) -> None:
        self.login(GAMMA_USERNAME)
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/schemas/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_database_schemas_invalid_query(self) -> None:
        self.login(ADMIN_USERNAME)
        database = db.session.query(Database).first()
        rv = self.client.get(f"api/v1/database/{database.id}/schemas/?q={prison.dumps({'force': 'nop'})}")
        assert rv.status_code == 400

    def test_database_tables(self) -> None:
        self.login(ADMIN_USERNAME)
        database = db.session.query(Database).filter_by(database_name='examples').one()
        schema_name = self.default_schema_backend_map[database.backend]
        rv = self.client.get(f"api/v1/database/{database.id}/tables/?q={prison.dumps({'schema_name': schema_name})}")
        assert rv.status_code == 200
        if database.backend == 'postgresql':
            response = json.loads(rv.data.decode('utf-8'))
            schemas = [s[0] for s in database.get_all_table_names_in_schema(None, schema_name)]
            assert response['count'] == len(schemas)
            for option in response['result']:
                assert option['extra'] is None
                assert option['type'] == 'table'
                assert option['value'] in schemas

    @patch('superset.utils.log.logger')
    def test_database_tables_not_found(self, logger_mock: Any) -> None:
        self.login(GAMMA_USERNAME)
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/tables/?q={prison.dumps({'schema_name': 'non_existent'})}"
        rv = self.client.get(uri)
        assert rv.status_code == 404
        logger_mock.warning.assert_called_once_with('Database not found.', exc_info=True)

    def test_database_tables_invalid_query(self) -> None:
        self.login(ADMIN_USERNAME)
        database = db.session.query(Database).first()
        rv = self.client.get(f"api/v1/database/{database.id}/tables/?q={prison.dumps({'force': 'nop'})}")
        assert rv.status_code == 400

    @patch('superset.utils.log.logger')
    @mock.patch('superset.security.manager.SupersetSecurityManager.can_access_database')
    @mock.patch('superset.models.core.Database.get_all_table_names_in_schema')
    def test_database_tables_unexpected_error(self, mock_get_all_table_names_in_schema: Any, mock_can_access_database: Any, logger_mock: Any) -> None:
        self.login(ADMIN_USERNAME)
        database = db.session.query(Database).filter_by(database_name='examples').one()
        mock_can_access_database.side_effect = Exception('Test Error')
        rv = self.client.get(f"api/v1/database/{database.id}/tables/?q={prison.dumps({'schema_name': 'main'})}")
        assert rv.status_code == 422
        logger_mock.warning.assert_called_once_with('Test Error', exc_info=True)

    def test_test_connection(self) -> None:
        extra = {'metadata_params': {}, 'engine_params': {}, 'metadata_cache_timeout': {}, 'schemas_allowed_for_file_upload': []}
        app.config['PREVENT_UNSAFE_DB_CONNECTIONS'] = False
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        data = {
            'database_name': 'examples',
            'masked_encrypted_extra': '{}',
            'extra': json.dumps(extra),
            'impersonate_user': False,
            'sqlalchemy_uri': example_db.safe_sqlalchemy_uri(),
            'server_cert': None,
        }
        url: str = 'api/v1/database/test_connection/'
        rv = self.post_assert_metric(url, data, 'test_connection')
        assert rv.status_code == 200
        assert rv.headers['Content-Type'] == 'application/json; charset=utf-8'
        data = {
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'database_name': 'examples',
            'impersonate_user': False,
            'extra': json.dumps(extra),
            'server_cert': None,
        }
        rv = self.post_assert_metric(url, data, 'test_connection')
        assert rv.status_code == 200
        assert rv.headers['Content-Type'] == 'application/json; charset=utf-8'

    def test_test_connection_failed(self) -> None:
        self.login(ADMIN_USERNAME)
        data = {
            'sqlalchemy_uri': 'broken://url',
            'database_name': 'examples',
            'impersonate_user': False,
            'server_cert': None,
        }
        url: str = 'api/v1/database/test_connection/'
        rv = self.post_assert_metric(url, data, 'test_connection')
        assert rv.status_code == 422
        assert rv.headers['Content-Type'] == 'application/json; charset=utf-8'
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {
            'errors': [
                dataclasses.asdict(
                    SupersetError(
                        message='Could not load database driver: BaseEngineSpec',
                        error_type='GENERIC_COMMAND_ERROR',
                        level='warning',
                        extra={'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}]},
                    )
                )
            ]
        }
        assert response == expected_response
        data = {
            'sqlalchemy_uri': 'mssql+pymssql://url',
            'database_name': 'examples',
            'impersonate_user': False,
            'server_cert': None,
        }
        rv = self.post_assert_metric(url, data, 'test_connection')
        assert rv.status_code == 422
        assert rv.headers['Content-Type'] == 'application/json; charset=utf-8'
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {
            'errors': [
                dataclasses.asdict(
                    SupersetError(
                        message='Could not load database driver: MssqlEngineSpec',
                        error_type='GENERIC_COMMAND_ERROR',
                        level='warning',
                        extra={'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}]},
                    )
                )
            ]
        }
        assert response == expected_response

    def test_test_connection_unsafe_uri(self) -> None:
        self.login(ADMIN_USERNAME)
        app.config['PREVENT_UNSAFE_DB_CONNECTIONS'] = True
        data = {
            'sqlalchemy_uri': 'sqlite:///home/superset/unsafe.db',
            'database_name': 'unsafe',
            'impersonate_user': False,
            'server_cert': None,
        }
        url: str = 'api/v1/database/test_connection/'
        rv = self.post_assert_metric(url, data, 'test_connection')
        assert rv.status_code == 400
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {
            'message': {'sqlalchemy_uri': ['SQLiteDialect_pysqlite cannot be used as a data source for security reasons.']}
        }
        assert response == expected_response
        app.config['PREVENT_UNSAFE_DB_CONNECTIONS'] = False

    @mock.patch('superset.commands.database.test_connection.DatabaseDAO.build_db_for_connection_test')
    @mock.patch('superset.commands.database.test_connection.event_logger')
    def test_test_connection_failed_invalid_hostname(self, mock_event_logger: Any, mock_build_db: Any) -> None:
        msg = 'psql: error: could not translate host name "localhost_" to address: nodename nor servname provided, or not known'
        mock_build_db.return_value.set_sqlalchemy_uri.side_effect = DBAPIError(msg, None, None)
        mock_build_db.return_value.db_engine_spec.__name__ = 'Some name'
        superset_error = SupersetError(
            message='Unable to resolve hostname "localhost_".',
            error_type='CONNECTION_INVALID_HOSTNAME_ERROR',
            level='error',
            extra={'hostname': 'localhost_', 'issue_codes': [{'code': 1007, 'message': "Issue 1007 - The hostname provided can't be resolved."}]},
        )
        mock_build_db.return_value.db_engine_spec.extract_errors.return_value = [superset_error]
        self.login(ADMIN_USERNAME)
        data = {
            'sqlalchemy_uri': 'postgres://username:password@localhost_:12345/db',
            'database_name': 'examples',
            'impersonate_user': False,
            'server_cert': None,
        }
        url: str = 'api/v1/database/test_connection/'
        rv = self.post_assert_metric(url, data, 'test_connection')
        assert rv.status_code == 500
        assert rv.headers['Content-Type'] == 'application/json; charset=utf-8'
        response = json.loads(rv.data.decode('utf-8'))
        expected_response = {'errors': [dataclasses.asdict(superset_error)]}
        assert response == expected_response

    @pytest.mark.usefixtures('load_unicode_dashboard_with_position', 'load_energy_table_with_slice', 'load_world_bank_dashboard_with_slices', 'load_birth_names_dashboard_with_slices')
    def test_get_database_related_objects(self) -> None:
        self.login(ADMIN_USERNAME)
        database = get_example_database()
        uri: str = f"api/v1/database/{database.id}/related_objects/"
        rv = self.get_assert_metric(uri, 'related_objects')
        assert rv.status_code == 200
        response = json.loads(rv.data.decode('utf-8'))
        assert response['charts']['count'] == 33
        assert response['dashboards']['count'] == 3

    def test_get_database_related_objects_not_found(self) -> None:
        max_id: Any = db.session.query(func.max(Database.id)).scalar()
        invalid_id = max_id + 1
        uri: str = f"api/v1/database/{invalid_id}/related_objects/"
        self.login(ADMIN_USERNAME)
        rv = self.get_assert_metric(uri, 'related_objects')
        assert rv.status_code == 404
        self.logout()
        self.login(GAMMA_USERNAME)
        database = get_example_database()
        uri = f"api/v1/database/{database.id}/related_objects/"
        rv = self.get_assert_metric(uri, 'related_objects')
        assert rv.status_code == 404

    def test_export_database(self) -> None:
        self.login(ADMIN_USERNAME)
        database = get_example_database()
        argument = [database.id]
        uri: str = f"api/v1/database/export/?q={prison.dumps(argument)}"
        rv = self.get_assert_metric(uri, 'export')
        assert rv.status_code == 200
        buf = BytesIO(rv.data)
        assert is_zipfile(buf)

    def test_export_database_not_allowed(self) -> None:
        self.login(GAMMA_USERNAME)
        database = get_example_database()
        argument = [database.id]
        uri: str = f"api/v1/database/export/?q={prison.dumps(argument)}"
        rv = self.client.get(uri)
        assert rv.status_code == 403

    def test_export_database_non_existing(self) -> None:
        max_id: Any = db.session.query(func.max(Database.id)).scalar()
        invalid_id = max_id + 1
        self.login(ADMIN_USERNAME)
        argument = [invalid_id]
        uri: str = f"api/v1/database/export/?q={prison.dumps(argument)}"
        rv = self.get_assert_metric(uri, 'export')
        assert rv.status_code == 404

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database(self, mock_add_permissions: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        buf: BytesIO = self.create_database_import()
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}
        database = db.session.query(Database).filter_by(uuid=database_config['uuid']).one()
        assert database.database_name == 'imported_database'
        assert len(database.tables) == 1
        dataset = database.tables[0]
        assert dataset.table_name == 'imported_dataset'
        assert str(dataset.uuid) == dataset_config['uuid']
        db.session.delete(dataset)
        db.session.commit()
        db.session.delete(database)
        db.session.commit()

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_overwrite(self, mock_add_permissions: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        buf: BytesIO = self.create_database_import()
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}
        buf = self.create_database_import()
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Error importing database',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {
                        'databases/imported_database.yaml': 'Database already exists and `overwrite=true` was not passed',
                        'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}],
                    },
                }
            ]
        }
        buf = self.create_database_import()
        form_data = {'formData': (buf, 'database_export.zip'), 'overwrite': 'true'}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}
        database = db.session.query(Database).filter_by(uuid=database_config['uuid']).one()
        dataset = database.tables[0]
        db.session.delete(dataset)
        db.session.commit()
        db.session.delete(database)
        db.session.commit()

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_invalid(self, mock_add_permissions: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Error importing database',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {
                        'metadata.yaml': {'type': ['Must be equal to Database.']},
                        'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}],
                    },
                }
            ]
        }

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_password(self, mock_add_permissions: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        masked_database_config = database_config.copy()
        masked_database_config['sqlalchemy_uri'] = 'postgresql://username:XXXXXXXXXX@host:12345/db'
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Error importing database',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {
                        'databases/imported_database.yaml': {'_schema': ['Must provide a password for the database']},
                        'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}],
                    },
                }
            ]
        }

    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_password_provided(self, mock_add_permissions: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        masked_database_config = database_config.copy()
        masked_database_config['sqlalchemy_uri'] = 'vertica+vertica_python://hackathon:XXXXXXXXXX@host:5433/dbname?ssl=1'
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
        buf.seek(0)
        form_data = {
            'formData': (buf, 'database_export.zip'),
            'passwords': json.dumps({'databases/imported_database.yaml': 'SECRET'}),
        }
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}
        database = db.session.query(Database).filter_by(uuid=database_config['uuid']).one()
        assert database.database_name == 'imported_database'
        assert database.sqlalchemy_uri == 'vertica+vertica_python://hackathon:XXXXXXXXXX@host:5433/dbname?ssl=1'
        assert database.password == 'SECRET'
        db.session.delete(database)
        db.session.commit()

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_password(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        mock_schema_is_feature_enabled.return_value = True
        masked_database_config = database_with_ssh_tunnel_config_password.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Error importing database',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {
                        'databases/imported_database.yaml': {'_schema': ['Must provide a password for the ssh tunnel']},
                        'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}],
                    },
                }
            ]
        }

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_password_provided(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        mock_schema_is_feature_enabled.return_value = True
        masked_database_config = database_with_ssh_tunnel_config_password.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
        buf.seek(0)
        form_data = {
            'formData': (buf, 'database_export.zip'),
            'ssh_tunnel_passwords': json.dumps({'databases/imported_database.yaml': 'TEST'}),
        }
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}
        database = db.session.query(Database).filter_by(uuid=database_config['uuid']).one()
        assert database.database_name == 'imported_database'
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == database.id).one()
        assert model_ssh_tunnel.password == 'TEST'
        db.session.delete(database)
        db.session.commit()

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_private_key_and_password(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        mock_schema_is_feature_enabled.return_value = True
        masked_database_config = database_with_ssh_tunnel_config_private_key.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Error importing database',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {
                        'databases/imported_database.yaml': {
                            '_schema': ['Must provide a private key for the ssh tunnel', 'Must provide a private key password for the ssh tunnel']
                        },
                        'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}],
                    },
                }
            ]
        }

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_private_key_and_password_provided(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        mock_schema_is_feature_enabled.return_value = True
        masked_database_config = database_with_ssh_tunnel_config_private_key.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
        buf.seek(0)
        form_data = {
            'formData': (buf, 'database_export.zip'),
            'ssh_tunnel_private_keys': json.dumps({'databases/imported_database.yaml': 'TestPrivateKey'}),
            'ssh_tunnel_private_key_passwords': json.dumps({'databases/imported_database.yaml': 'TEST'}),
        }
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}
        database = db.session.query(Database).filter_by(uuid=database_config['uuid']).one()
        assert database.database_name == 'imported_database'
        model_ssh_tunnel = db.session.query(SSHTunnel).filter(SSHTunnel.database_id == database.id).one()
        assert model_ssh_tunnel.private_key == 'TestPrivateKey'
        assert model_ssh_tunnel.private_key_password == 'TEST'
        db.session.delete(database)
        db.session.commit()

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_flag_disabled(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        masked_database_config = database_with_ssh_tunnel_config_private_key.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert response == {
            'errors': [
                {
                    'message': 'SSH Tunneling is not enabled',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}]},
                }
            ]
        }

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_no_credentials(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        mock_schema_is_feature_enabled.return_value = True
        masked_database_config = database_with_ssh_tunnel_config_no_credentials.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Must provide credentials for the SSH Tunnel',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}]},
                }
            ]
        }

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_mix_credentials(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        mock_schema_is_feature_enabled.return_value = True
        masked_database_config = database_with_ssh_tunnel_config_mix_credentials.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Cannot have multiple credentials for the SSH Tunnel',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}]},
                }
            ]
        }

    @mock.patch('superset.databases.schemas.is_feature_enabled')
    @mock.patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_database_masked_ssh_tunnel_feature_only_pk_passwd(self, mock_add_permissions: Any, mock_schema_is_feature_enabled: Any) -> None:
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/import/'
        mock_schema_is_feature_enabled.return_value = True
        masked_database_config = database_with_ssh_tunnel_config_private_pass_only.copy()
        buf: BytesIO = BytesIO()
        with ZipFile(buf, 'w') as bundle:
            with bundle.open('database_export/metadata.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open('database_export/databases/imported_database.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(masked_database_config).encode())
            with bundle.open('database_export/datasets/imported_dataset.yaml', 'w') as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        form_data = {'formData': (buf, 'database_export.zip')}
        rv = self.client.post(uri, data=form_data, content_type='multipart/form-data')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Error importing database',
                    'error_type': 'GENERIC_COMMAND_ERROR',
                    'level': 'warning',
                    'extra': {
                        'databases/imported_database.yaml': {'_schema': ['Must provide a private key for the ssh tunnel', 'Must provide a private key password for the ssh tunnel']},
                        'issue_codes': [{'code': 1010, 'message': 'Issue 1010 - Superset encountered an error while running a command.'}],
                    },
                }
            ]
        }

    def test_function_names(self) -> None:
        example_db = get_example_database()
        if example_db.backend in {'hive', 'presto', 'sqlite'}:
            return
        from unittest.mock import MagicMock
        # pylint: disable=no-member
        mock_get_function_names = MagicMock(return_value=['AVG', 'MAX', 'SUM'])
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/1/function_names/'
        rv = self.client.get(uri)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'function_names': ['AVG', 'MAX', 'SUM']}

    def test_function_names_sqlite(self) -> None:
        example_db = get_example_database()
        if example_db.backend != 'sqlite':
            return
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/1/function_names/'
        rv = self.client.get(uri)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {
            'function_names': [
                'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'avg',
                'ceil', 'ceiling', 'changes', 'char', 'coalesce', 'cos', 'cosh', 'count',
                'cume_dist', 'date', 'datetime', 'degrees', 'dense_rank', 'exp', 'first_value',
                'floor', 'format', 'glob', 'group_concat', 'hex', 'ifnull', 'iif', 'instr', 'json',
                'json_array', 'json_array_length', 'json_each', 'json_error_position', 'json_extract',
                'json_group_array', 'json_group_object', 'json_insert', 'json_object', 'json_patch',
                'json_quote', 'json_remove', 'json_replace', 'json_set', 'json_tree', 'json_type',
                'json_valid', 'julianday', 'lag', 'last_insert_rowid', 'last_value', 'lead', 'length',
                'like', 'likelihood', 'likely', 'ln', 'load_extension', 'log', 'log10', 'log2', 'lower',
                'ltrim', 'max', 'min', 'mod', 'nth_value', 'ntile', 'nullif', 'percent_rank', 'pi',
                'pow', 'power', 'printf', 'quote', 'radians', 'random', 'randomblob', 'rank', 'replace',
                'round', 'row_number', 'rtrim', 'sign', 'sin', 'sinh', 'soundex', 'sqlite_compileoption_get',
                'sqlite_compileoption_used', 'sqlite_offset', 'sqlite_source_id', 'sqlite_version', 'sqrt',
                'strftime', 'substr', 'substring', 'sum', 'tan', 'tanh', 'time', 'total_changes', 'trim',
                'trunc', 'typeof', 'unhex', 'unicode', 'unixepoch', 'unlikely', 'upper', 'zeroblob',
            ]
        }

    @mock.patch('superset.databases.api.get_available_engine_specs')
    @mock.patch('superset.databases.api.app')
    def test_available(self, app: Any, get_available_engine_specs: Any) -> None:
        app.config = {'PREFERRED_DATABASES': ['PostgreSQL', 'Google BigQuery']}
        get_available_engine_specs.return_value = {
            PostgresEngineSpec: {'psycopg2'},
            BigQueryEngineSpec: {'bigquery'},
            MySQLEngineSpec: {'mysqlconnector', 'mysqldb'},
            GSheetsEngineSpec: {'apsw'},
            RedshiftEngineSpec: {'psycopg2'},
            HanaEngineSpec: {''},
        }
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/available/'
        rv = self.client.get(uri)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {
            'databases': [
                {
                    'available_drivers': ['psycopg2'],
                    'default_driver': 'psycopg2',
                    'engine': 'postgresql',
                    'name': 'PostgreSQL',
                    'parameters': {
                        'properties': {
                            'database': {'description': 'Database name', 'type': 'string'},
                            'encryption': {'description': 'Use an encrypted connection to the database', 'type': 'boolean'},
                            'host': {'description': 'Hostname or IP address', 'type': 'string'},
                            'password': {'description': 'Password', 'nullable': True, 'type': 'string'},
                            'port': {'description': 'Database port', 'maximum': 65536, 'minimum': 0, 'type': 'integer'},
                            'query': {'additionalProperties': {}, 'description': 'Additional parameters', 'type': 'object'},
                            'ssh': {'description': 'Use an ssh tunnel connection to the database', 'type': 'boolean'},
                            'username': {'description': 'Username', 'nullable': True, 'type': 'string'},
                        },
                        'required': ['database', 'host', 'port', 'username'],
                        'type': 'object',
                    },
                    'preferred': True,
                    'sqlalchemy_uri_placeholder': 'postgresql://user:password@host:port/dbname[?key=value&key=value...]',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': True,
                        'disable_ssh_tunneling': False,
                        'supports_oauth2': False,
                    },
                    'supports_oauth2': False,
                },
                {
                    'available_drivers': ['bigquery'],
                    'default_driver': 'bigquery',
                    'engine': 'bigquery',
                    'name': 'Google BigQuery',
                    'parameters': {
                        'properties': {
                            'credentials_info': {'description': 'Contents of BigQuery JSON credentials.', 'type': 'string', 'x-encrypted-extra': True},
                            'query': {'type': 'object'},
                        },
                        'type': 'object',
                    },
                    'preferred': True,
                    'sqlalchemy_uri_placeholder': 'bigquery://{project_id}',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': True,
                        'disable_ssh_tunneling': True,
                        'supports_oauth2': False,
                    },
                    'supports_oauth2': False,
                },
                {
                    'available_drivers': ['psycopg2'],
                    'default_driver': 'psycopg2',
                    'engine': 'redshift',
                    'name': 'Amazon Redshift',
                    'parameters': {
                        'properties': {
                            'database': {'description': 'Database name', 'type': 'string'},
                            'encryption': {'description': 'Use an encrypted connection to the database', 'type': 'boolean'},
                            'host': {'description': 'Hostname or IP address', 'type': 'string'},
                            'password': {'description': 'Password', 'nullable': True, 'type': 'string'},
                            'port': {'description': 'Database port', 'maximum': 65536, 'minimum': 0, 'type': 'integer'},
                            'query': {'additionalProperties': {}, 'description': 'Additional parameters', 'type': 'object'},
                            'ssh': {'description': 'Use an ssh tunnel connection to the database', 'type': 'boolean'},
                            'username': {'description': 'Username', 'nullable': True, 'type': 'string'},
                        },
                        'required': ['database', 'host', 'port', 'username'],
                        'type': 'object',
                    },
                    'preferred': False,
                    'sqlalchemy_uri_placeholder': 'redshift+psycopg2://user:password@host:port/dbname[?key=value&key=value...]',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': False,
                        'disable_ssh_tunneling': False,
                        'supports_oauth2': False,
                    },
                    'supports_oauth2': False,
                },
                {
                    'available_drivers': ['apsw'],
                    'default_driver': 'apsw',
                    'engine': 'gsheets',
                    'name': 'Google Sheets',
                    'parameters': {
                        'properties': {
                            'catalog': {'type': 'object'},
                            'service_account_info': {'description': 'Contents of GSheets JSON credentials.', 'type': 'string', 'x-encrypted-extra': True},
                        },
                        'type': 'object',
                    },
                    'preferred': False,
                    'sqlalchemy_uri_placeholder': 'gsheets://',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': False,
                        'disable_ssh_tunneling': True,
                        'supports_oauth2': True,
                    },
                    'supports_oauth2': True,
                },
                {
                    'available_drivers': ['mysqlconnector', 'mysqldb'],
                    'default_driver': 'mysqldb',
                    'engine': 'mysql',
                    'name': 'MySQL',
                    'parameters': {
                        'properties': {
                            'database': {'description': 'Database name', 'type': 'string'},
                            'encryption': {'description': 'Use an encrypted connection to the database', 'type': 'boolean'},
                            'host': {'description': 'Hostname or IP address', 'type': 'string'},
                            'password': {'description': 'Password', 'nullable': True, 'type': 'string'},
                            'port': {'description': 'Database port', 'maximum': 65536, 'minimum': 0, 'type': 'integer'},
                            'query': {'additionalProperties': {}, 'description': 'Additional parameters', 'type': 'object'},
                            'ssh': {'description': 'Use an ssh tunnel connection to the database', 'type': 'boolean'},
                            'username': {'description': 'Username', 'nullable': True, 'type': 'string'},
                        },
                        'required': ['database', 'host', 'port', 'username'],
                        'type': 'object',
                    },
                    'preferred': False,
                    'sqlalchemy_uri_placeholder': 'mysql://user:password@host:port/dbname[?key=value&key=value...]',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': False,
                        'disable_ssh_tunneling': False,
                        'supports_oauth2': False,
                    },
                    'supports_oauth2': False,
                },
                {
                    'available_drivers': [''],
                    'engine': 'hana',
                    'name': 'SAP HANA',
                    'preferred': False,
                    'sqlalchemy_uri_placeholder': 'engine+driver://user:password@host:port/dbname[?key=value&key=value...]',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': False,
                        'disable_ssh_tunneling': False,
                        'supports_oauth2': False,
                    },
                    'supports_oauth2': False,
                },
            ]
        }

    @mock.patch('superset.databases.api.get_available_engine_specs')
    @mock.patch('superset.databases.api.app')
    def test_available_no_default(self, app: Any, get_available_engine_specs: Any) -> None:
        app.config = {'PREFERRED_DATABASES': ['MySQL']}
        get_available_engine_specs.return_value = {MySQLEngineSpec: {'mysqlconnector'}, HanaEngineSpec: {''}}
        self.login(ADMIN_USERNAME)
        uri: str = 'api/v1/database/available/'
        rv = self.client.get(uri)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {
            'databases': [
                {
                    'available_drivers': ['mysqlconnector'],
                    'default_driver': 'mysqldb',
                    'engine': 'mysql',
                    'name': 'MySQL',
                    'preferred': True,
                    'sqlalchemy_uri_placeholder': 'mysql://user:password@host:port/dbname[?key=value&key=value...]',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': False,
                        'disable_ssh_tunneling': False,
                        'supports_oauth2': False,
                    },
                    'supports_oauth2': False,
                },
                {
                    'available_drivers': [''],
                    'engine': 'hana',
                    'name': 'SAP HANA',
                    'preferred': False,
                    'sqlalchemy_uri_placeholder': 'engine+driver://user:password@host:port/dbname[?key=value&key=value...]',
                    'engine_information': {
                        'supports_file_upload': True,
                        'supports_dynamic_catalog': False,
                        'disable_ssh_tunneling': False,
                        'supports_oauth2': False,
                    },
                    'supports_oauth2': False,
                },
            ]
        }

    def test_validate_parameters_invalid_payload_format(self) -> None:
        self.login(ADMIN_USERNAME)
        url: str = 'api/v1/database/validate_parameters/'
        rv = self.client.post(url, data='INVALID', content_type='text/plain')
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert response == {
            'errors': [
                {
                    'message': 'Request is not JSON',
                    'error_type': 'INVALID_PAYLOAD_FORMAT_ERROR',
                    'level': 'error',
                    'extra': {'issue_codes': [{'code': 1019, 'message': 'Issue 1019 - The submitted payload has the incorrect format.'}]},
                }
            ]
        }

    def test_validate_parameters_invalid_payload_schema(self) -> None:
        self.login(ADMIN_USERNAME)
        url: str = 'api/v1/database/validate_parameters/'
        payload = {'foo': 'bar'}
        rv = self.client.post(url, json=payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        response['errors'].sort(key=lambda error: error['extra']['invalid'][0])
        assert response == {
            'errors': [
                {
                    'message': 'Missing data for required field.',
                    'error_type': 'INVALID_PAYLOAD_SCHEMA_ERROR',
                    'level': 'error',
                    'extra': {'invalid': ['configuration_method'], 'issue_codes': [{'code': 1020, 'message': 'Issue 1020 - The submitted payload has the incorrect schema.'}]},
                },
                {
                    'message': 'Missing data for required field.',
                    'error_type': 'INVALID_PAYLOAD_SCHEMA_ERROR',
                    'level': 'error',
                    'extra': {'invalid': ['engine'], 'issue_codes': [{'code': 1020, 'message': 'Issue 1020 - The submitted payload has the incorrect schema.'}]},
                },
            ]
        }

    def test_validate_parameters_missing_fields(self) -> None:
        self.login(ADMIN_USERNAME)
        url: str = 'api/v1/database/validate_parameters/'
        payload = {
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
            'engine': 'postgresql',
            'parameters': defaultdict(dict),
        }
        payload['parameters'].update({'host': '', 'port': 5432, 'username': '', 'password': '', 'database': '', 'query': {}})
        rv = self.client.post(url, json=payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'One or more parameters are missing: database, host, username',
                    'error_type': 'CONNECTION_MISSING_PARAMETERS_ERROR',
                    'level': 'warning',
                    'extra': {'missing': ['database', 'host', 'username'], 'issue_codes': [{'code': 1018, 'message': 'Issue 1018 - One or more parameters needed to configure a database are missing.'}]},
                }
            ]
        }

    @mock.patch('superset.db_engine_specs.base.is_hostname_valid')
    @mock.patch('superset.db_engine_specs.base.is_port_open')
    @mock.patch('superset.databases.api.ValidateDatabaseParametersCommand')
    def test_validate_parameters_valid_payload(
        self, ValidateDatabaseParametersCommand: Any, is_port_open: Any, is_hostname_valid: Any
    ) -> None:
        is_hostname_valid.return_value = True
        is_port_open.return_value = True
        self.login(ADMIN_USERNAME)
        url: str = 'api/v1/database/validate_parameters/'
        payload = {
            'engine': 'postgresql',
            'parameters': defaultdict(dict),
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        payload['parameters'].update({'host': 'localhost', 'port': 6789, 'username': 'superset', 'password': 'XXX', 'database': 'test', 'query': {}})
        rv = self.client.post(url, json=payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response == {'message': 'OK'}

    def test_validate_parameters_invalid_port(self) -> None:
        self.login(ADMIN_USERNAME)
        url: str = 'api/v1/database/validate_parameters/'
        payload = {
            'engine': 'postgresql',
            'parameters': defaultdict(dict),
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        payload['parameters'].update({'host': 'localhost', 'port': 'string', 'username': 'superset', 'password': 'XXX', 'database': 'test', 'query': {}})
        rv = self.client.post(url, json=payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'Port must be a valid integer.',
                    'error_type': 'CONNECTION_INVALID_PORT_ERROR',
                    'level': 'error',
                    'extra': {'invalid': ['port'], 'issue_codes': [{'code': 1034, 'message': 'Issue 1034 - The port number is invalid.'}]},
                },
                {
                    'message': 'The port must be an integer between 0 and 65535 (inclusive).',
                    'error_type': 'CONNECTION_INVALID_PORT_ERROR',
                    'level': 'error',
                    'extra': {'invalid': ['port'], 'issue_codes': [{'code': 1034, 'message': 'Issue 1034 - The port number is invalid.'}]},
                },
            ]
        }

    @mock.patch('superset.db_engine_specs.base.is_hostname_valid')
    def test_validate_parameters_invalid_host(self, is_hostname_valid: Any) -> None:
        is_hostname_valid.return_value = False
        self.login(ADMIN_USERNAME)
        url: str = 'api/v1/database/validate_parameters/'
        payload = {
            'engine': 'postgresql',
            'parameters': defaultdict(dict),
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        payload['parameters'].update({'host': 'localhost', 'port': 5432, 'username': '', 'password': '', 'database': '', 'query': {}})
        rv = self.client.post(url, json=payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'One or more parameters are missing: database, username',
                    'error_type': 'CONNECTION_MISSING_PARAMETERS_ERROR',
                    'level': 'warning',
                    'extra': {'missing': ['database', 'username'], 'issue_codes': [{'code': 1018, 'message': 'Issue 1018 - One or more parameters needed to configure a database are missing.'}]},
                },
                {
                    'message': "The hostname provided can't be resolved.",
                    'error_type': 'CONNECTION_INVALID_HOSTNAME_ERROR',
                    'level': 'error',
                    'extra': {'invalid': ['host'], 'issue_codes': [{'code': 1007, 'message': "Issue 1007 - The hostname provided can't be resolved."}]},
                },
            ]
        }

    @mock.patch('superset.db_engine_specs.base.is_hostname_valid')
    def test_validate_parameters_invalid_port_range(self, is_hostname_valid: Any) -> None:
        is_hostname_valid.return_value = True
        self.login(ADMIN_USERNAME)
        url: str = 'api/v1/database/validate_parameters/'
        payload = {
            'engine': 'postgresql',
            'parameters': defaultdict(dict),
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
        }
        payload['parameters'].update({'host': 'localhost', 'port': 65536, 'username': '', 'password': '', 'database': '', 'query': {}})
        rv = self.client.post(url, json=payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': 'One or more parameters are missing: database, username',
                    'error_type': 'CONNECTION_MISSING_PARAMETERS_ERROR',
                    'level': 'warning',
                    'extra': {'missing': ['database', 'username'], 'issue_codes': [{'code': 1018, 'message': 'Issue 1018 - One or more parameters needed to configure a database are missing.'}]},
                },
                {
                    'message': 'The port must be an integer between 0 and 65535 (inclusive).',
                    'error_type': 'CONNECTION_INVALID_PORT_ERROR',
                    'level': 'error',
                    'extra': {'invalid': ['port'], 'issue_codes': [{'code': 1034, 'message': 'Issue 1034 - The port number is invalid.'}]},
                },
            ]
        }

    def test_get_related_objects(self) -> None:
        example_db = get_example_database()
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{example_db.id}/related_objects/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        assert 'charts' in rv.json
        assert 'dashboards' in rv.json
        assert 'sqllab_tab_states' in rv.json

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql(self) -> None:
        request_payload = {'sql': 'SELECT * from birth_names', 'schema': None, 'template_params': None}
        example_db = get_example_database()
        if example_db.backend not in ('presto', 'postgresql'):
            pytest.skip('Only presto and PG are implemented')
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{example_db.id}/validate_sql/"
        rv = self.client.post(uri, json=request_payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response['result'] == []

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_errors(self) -> None:
        request_payload = {'sql': 'SELECT col1 from_ table1', 'schema': None, 'template_params': None}
        example_db = get_example_database()
        if example_db.backend not in ('presto', 'postgresql'):
            pytest.skip('Only presto and PG are implemented')
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{example_db.id}/validate_sql/"
        rv = self.client.post(uri, json=request_payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 200
        assert response['result'] == [{'end_column': None, 'line_number': 1, 'message': 'ERROR: syntax error at or near "table1"', 'start_column': None}]

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_not_found(self) -> None:
        request_payload = {'sql': 'SELECT * from birth_names', 'schema': None, 'template_params': None}
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{self.get_nonexistent_numeric_id(Database)}/validate_sql/"
        rv = self.client.post(uri, json=request_payload)
        assert rv.status_code == 404

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_validation_fails(self) -> None:
        request_payload = {'sql': None, 'schema': None, 'template_params': None}
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{self.get_nonexistent_numeric_id(Database)}/validate_sql/"
        rv = self.client.post(uri, json=request_payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 400
        assert response == {'message': {'sql': ['Field may not be null.']}}

    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', {}, clear=True)
    def test_validate_sql_endpoint_noconfig(self) -> None:
        request_payload = {'sql': 'SELECT col1 from table1', 'schema': None, 'template_params': None}
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/validate_sql/"
        rv = self.client.post(uri, json=request_payload)
        response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 422
        assert response == {
            'errors': [
                {
                    'message': f'no SQL validator is configured for {example_db.backend}',
                    'error_type': 'GENERIC_DB_ENGINE_ERROR',
                    'level': 'error',
                    'extra': {'issue_codes': [{'code': 1002, 'message': 'Issue 1002 - The database returned an unexpected error.'}]},
                }
            ]
        }

    @mock.patch('superset.commands.database.validate_sql.get_validator_by_name')
    @mock.patch.dict('superset.config.SQL_VALIDATORS_BY_ENGINE', PRESTO_SQL_VALIDATORS_BY_ENGINE, clear=True)
    def test_validate_sql_endpoint_failure(self, get_validator_by_name: Any) -> None:
        request_payload = {'sql': 'SELECT * FROM birth_names', 'schema': None, 'template_params': None}
        self.login(ADMIN_USERNAME)
        validator = MagicMock()
        get_validator_by_name.return_value = validator
        validator.validate.side_effect = Exception('Kaboom!')
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/validate_sql/"
        rv = self.client.post(uri, json=request_payload)
        response = json.loads(rv.data.decode('utf-8'))
        if get_example_database().backend == 'hive':
            return
        assert rv.status_code == 422
        assert 'Kaboom!' in response['errors'][0]['message']

    def test_get_databases_with_extra_filters(self) -> None:
        self.login(ADMIN_USERNAME)
        extra = {'metadata_params': {}, 'engine_params': {}, 'metadata_cache_timeout': {}, 'schemas_allowed_for_file_upload': []}
        example_db = get_example_database()
        if example_db.backend == 'sqlite':
            return
        database_data = {
            'sqlalchemy_uri': example_db.sqlalchemy_uri_decrypted,
            'configuration_method': ConfigurationMethod.SQLALCHEMY_FORM,
            'server_cert': None,
            'extra': json.dumps(extra),
        }
        uri: str = 'api/v1/database/'
        rv = self.client.post(uri, json={**database_data, 'database_name': 'dyntest-create-database-1'})
        first_response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201
        uri = 'api/v1/database/'
        rv = self.client.post(uri, json={**database_data, 'database_name': 'create-database-2'})
        second_response = json.loads(rv.data.decode('utf-8'))
        assert rv.status_code == 201

        def _base_filter(query: Any) -> Any:
            from superset.models.core import Database
            return query.filter(Database.database_name.startswith('dyntest'))
        base_filter_mock = Mock(side_effect=_base_filter)
        dbs = db.session.query(Database).all()
        expected_names = [db.database_name for db in dbs]
        expected_names.sort()
        uri = 'api/v1/database/'
        rv = self.client.get(uri)
        data = json.loads(rv.data.decode('utf-8'))
        assert data['count'] == len(dbs)
        database_names = [item['database_name'] for item in data['result']]
        database_names.sort()
        assert database_names == expected_names
        assert rv.status_code == 200
        base_filter_mock.assert_not_called()
        from unittest.mock import patch as local_patch
        with local_patch.dict('superset.views.filters.current_app.config', {'EXTRA_DYNAMIC_QUERY_FILTERS': {'databases': base_filter_mock}}):
            uri = 'api/v1/database/'
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode('utf-8'))
            assert data['count'] == 1
            database_names = [item['database_name'] for item in data['result']]
            assert database_names == ['dyntest-create-database-1']
            assert rv.status_code == 200
            base_filter_mock.assert_called()
        first_model = db.session.query(Database).get(first_response.get('id'))
        second_model = db.session.query(Database).get(second_response.get('id'))
        db.session.delete(first_model)
        db.session.delete(second_model)
        db.session.commit()