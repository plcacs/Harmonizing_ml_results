from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import Any
from unittest.mock import ANY, Mock
from uuid import UUID
import pytest
from flask import current_app
from freezegun import freeze_time
from pytest_mock import MockerFixture
from sqlalchemy.orm.session import Session
from superset import db
from superset.commands.database.uploaders.base import UploadCommand
from superset.commands.database.uploaders.columnar_reader import ColumnarReader
from superset.commands.database.uploaders.csv_reader import CSVReader
from superset.commands.database.uploaders.excel_reader import ExcelReader
from superset.db_engine_specs.sqlite import SqliteEngineSpec
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import OAuth2RedirectError, SupersetSecurityException
from superset.sql_parse import Table
from superset.utils import json
from tests.unit_tests.fixtures.common import create_columnar_file, create_csv_file, create_excel_file


def func_42teek6f(session, client, full_api_access):
    """
    Test that we can filter databases by UUID.

    Note: this functionality is not used by the Superset UI, but is needed by 3rd
    party tools that use the Superset API. If this tests breaks, please make sure
    that the functionality is properly deprecated between major versions with
    enough warning so that tools can be adapted.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database
    DatabaseRestApi.datamodel.session = session
    Database.metadata.create_all(session.get_bind())
    db.session.add(Database(database_name='my_db', sqlalchemy_uri=
        'sqlite://', uuid=UUID('7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb')))
    db.session.commit()
    response = client.get(
        '/api/v1/database/?q=(filters:!((col:uuid,opr:eq,value:%277c1b7880-a59d-47cd-8bf1-f1eb8d2863cb%27)))'
        )
    assert response.status_code == 200
    payload = response.json
    assert len(payload['result']) == 1
    assert payload['result'][0]['uuid'
        ] == '7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb'


def func_g6p877mv(session, client, full_api_access):
    """
    Test that we can set the database UUID when creating it.
    """
    from superset.models.core import Database
    Database.metadata.create_all(session.get_bind())
    response = client.post('/api/v1/database/', json={'database_name':
        'my_db', 'sqlalchemy_uri': 'sqlite://', 'uuid':
        '7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb'})
    assert response.status_code == 201
    payload = response.json
    assert payload['result']['uuid'] == '7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb'
    database = session.query(Database).one()
    assert database.uuid == UUID('7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb')


def func_snkp5oce(mocker, app, session, client, full_api_access):
    """
    Test that sensitive information is masked.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database
    DatabaseRestApi.datamodel.session = session
    Database.metadata.create_all(session.get_bind())
    database = Database(uuid=UUID('02feae18-2dd6-4bb4-a9c0-49e9d4f29d58'),
        database_name='my_database', sqlalchemy_uri='gsheets://',
        encrypted_extra=json.dumps({'service_account_info': {'type':
        'service_account', 'project_id': 'black-sanctum-314419',
        'private_key_id': '259b0d419a8f840056158763ff54d8b08f7b8173',
        'private_key': 'SECRET', 'client_email':
        'google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com'
        , 'client_id': '114567578578109757129', 'auth_uri':
        'https://accounts.google.com/o/oauth2/auth', 'token_uri':
        'https://oauth2.googleapis.com/token',
        'auth_provider_x509_cert_url':
        'https://www.googleapis.com/oauth2/v1/certs',
        'client_x509_cert_url':
        'https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com'
        }}))
    db.session.add(database)
    db.session.commit()
    mocker.patch('sqlalchemy.engine.URL.get_driver_name', return_value=
        'gsheets')
    mocker.patch('superset.utils.log.DBEventLogger.log')
    response = client.get('/api/v1/database/1/connection')
    assert response.json['result']['parameters']['service_account_info'][
        'private_key'] == 'XXXXXXXXXX'
    assert 'encrypted_extra' not in response.json['result']


def func_ulhr4zsc(mocker, app, session, client, full_api_access):
    """
    Test that connection info is only returned in ``api/v1/database/${id}/connection``.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database
    DatabaseRestApi.datamodel.session = session
    Database.metadata.create_all(session.get_bind())
    database = Database(uuid=UUID('02feae18-2dd6-4bb4-a9c0-49e9d4f29d58'),
        database_name='my_database', sqlalchemy_uri='gsheets://',
        encrypted_extra=json.dumps({'service_account_info': {'type':
        'service_account', 'project_id': 'black-sanctum-314419',
        'private_key_id': '259b0d419a8f840056158763ff54d8b08f7b8173',
        'private_key': 'SECRET', 'client_email':
        'google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com'
        , 'client_id': '114567578578109757129', 'auth_uri':
        'https://accounts.google.com/o/oauth2/auth', 'token_uri':
        'https://oauth2.googleapis.com/token',
        'auth_provider_x509_cert_url':
        'https://www.googleapis.com/oauth2/v1/certs',
        'client_x509_cert_url':
        'https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com'
        }}))
    db.session.add(database)
    db.session.commit()
    mocker.patch('sqlalchemy.engine.URL.get_driver_name', return_value=
        'gsheets')
    mocker.patch('superset.utils.log.DBEventLogger.log')
    response = client.get('/api/v1/database/1/connection')
    assert response.json == {'id': 1, 'result': {'allow_ctas': False,
        'allow_cvas': False, 'allow_dml': False, 'allow_file_upload': False,
        'allow_run_async': False, 'backend': 'gsheets', 'cache_timeout':
        None, 'configuration_method': 'sqlalchemy_form', 'database_name':
        'my_database', 'driver': 'gsheets', 'engine_information': {
        'disable_ssh_tunneling': True, 'supports_dynamic_catalog': False,
        'supports_file_upload': True, 'supports_oauth2': True},
        'expose_in_sqllab': True, 'extra':
        """{
    "metadata_params": {},
    "engine_params": {},
    "metadata_cache_timeout": {},
    "schemas_allowed_for_file_upload": []
}
"""
        , 'force_ctas_schema': None, 'id': 1, 'impersonate_user': False,
        'is_managed_externally': False, 'masked_encrypted_extra': json.
        dumps({'service_account_info': {'type': 'service_account',
        'project_id': 'black-sanctum-314419', 'private_key_id':
        '259b0d419a8f840056158763ff54d8b08f7b8173', 'private_key':
        'XXXXXXXXXX', 'client_email':
        'google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com'
        , 'client_id': '114567578578109757129', 'auth_uri':
        'https://accounts.google.com/o/oauth2/auth', 'token_uri':
        'https://oauth2.googleapis.com/token',
        'auth_provider_x509_cert_url':
        'https://www.googleapis.com/oauth2/v1/certs',
        'client_x509_cert_url':
        'https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com'
        }}), 'parameters': {'service_account_info': {
        'auth_provider_x509_cert_url':
        'https://www.googleapis.com/oauth2/v1/certs', 'auth_uri':
        'https://accounts.google.com/o/oauth2/auth', 'client_email':
        'google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com'
        , 'client_id': '114567578578109757129', 'client_x509_cert_url':
        'https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com'
        , 'private_key': 'XXXXXXXXXX', 'private_key_id':
        '259b0d419a8f840056158763ff54d8b08f7b8173', 'project_id':
        'black-sanctum-314419', 'token_uri':
        'https://oauth2.googleapis.com/token', 'type': 'service_account'}},
        'parameters_schema': {'properties': {'catalog': {'type': 'object'},
        'service_account_info': {'description':
        'Contents of GSheets JSON credentials.', 'type': 'string',
        'x-encrypted-extra': True}}, 'type': 'object'}, 'server_cert': None,
        'sqlalchemy_uri': 'gsheets://', 'uuid':
        '02feae18-2dd6-4bb4-a9c0-49e9d4f29d58'}}
    response = client.get('/api/v1/database/1')
    assert response.json == {'id': 1, 'result': {'allow_ctas': False,
        'allow_cvas': False, 'allow_dml': False, 'allow_file_upload': False,
        'allow_run_async': False, 'backend': 'gsheets', 'cache_timeout':
        None, 'configuration_method': 'sqlalchemy_form', 'database_name':
        'my_database', 'driver': 'gsheets', 'engine_information': {
        'disable_ssh_tunneling': True, 'supports_dynamic_catalog': False,
        'supports_file_upload': True, 'supports_oauth2': True},
        'expose_in_sqllab': True, 'force_ctas_schema': None, 'id': 1,
        'impersonate_user': False, 'is_managed_externally': False, 'uuid':
        '02feae18-2dd6-4bb4-a9c0-49e9d4f29d58'}}


@pytest.mark.skip(reason='Works locally but fails on CI')
def func_znuvwzyn(app, session, client, full_api_access):
    """
    Test that an update with a masked password doesn't overwrite the existing password.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database
    DatabaseRestApi.datamodel.session = session
    Database.metadata.create_all(session.get_bind())
    database = Database(database_name='my_database', sqlalchemy_uri=
        'gsheets://', encrypted_extra=json.dumps({'service_account_info': {
        'project_id': 'black-sanctum-314419', 'private_key': 'SECRET'}}))
    db.session.add(database)
    db.session.commit()
    client.put('/api/v1/database/1', json={'encrypted_extra': json.dumps({
        'service_account_info': {'project_id': 'yellow-unicorn-314419',
        'private_key': 'XXXXXXXXXX'}})})
    database = db.session.query(Database).one()
    assert database.encrypted_extra == '{"service_account_info": {"project_id": "yellow-unicorn-314419", "private_key": "SECRET"}}'


def func_umuz0ibk(client, full_api_access):
    """
    Test that non-ZIP imports are not allowed.
    """
    buf = BytesIO(b'definitely_not_a_zip_file')
    form_data = {'formData': (buf, 'evil.pdf')}
    response = client.post('/api/v1/database/import/', data=form_data,
        content_type='multipart/form-data')
    assert response.status_code == 422
    assert response.json == {'errors': [{'message': 'Not a ZIP file',
        'error_type': 'GENERIC_COMMAND_ERROR', 'level': 'warning', 'extra':
        {'issue_codes': [{'code': 1010, 'message':
        'Issue 1010 - Superset encountered an error while running a command.'
        }]}}]}


def func_yas9lr53(mocker, app, session, client, full_api_access):
    """
    Test that we can delete SSH Tunnel
    """
    with app.app_context():
        from superset.daos.database import DatabaseDAO
        from superset.databases.api import DatabaseRestApi
        from superset.databases.ssh_tunnel.models import SSHTunnel
        from superset.models.core import Database
        DatabaseRestApi.datamodel.session = session
        Database.metadata.create_all(session.get_bind())
        database = Database(database_name='my_database', sqlalchemy_uri=
            'gsheets://', encrypted_extra=json.dumps({
            'service_account_info': {'type': 'service_account',
            'project_id': 'black-sanctum-314419', 'private_key_id':
            '259b0d419a8f840056158763ff54d8b08f7b8173', 'private_key':
            'SECRET', 'client_email':
            'google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com'
            , 'client_id': 'SSH_TUNNEL_CREDENTIALS_CLIENT', 'auth_uri':
            'https://accounts.google.com/o/oauth2/auth', 'token_uri':
            'https://oauth2.googleapis.com/token',
            'auth_provider_x509_cert_url':
            'https://www.googleapis.com/oauth2/v1/certs',
            'client_x509_cert_url':
            'https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com'
            }}))
        db.session.add(database)
        db.session.commit()
        mocker.patch('sqlalchemy.engine.URL.get_driver_name', return_value=
            'gsheets')
        mocker.patch('superset.utils.log.DBEventLogger.log')
        mocker.patch(
            'superset.commands.database.ssh_tunnel.delete.is_feature_enabled',
            return_value=True)
        tunnel = SSHTunnel(database_id=1, database=database)
        db.session.add(tunnel)
        db.session.commit()
        response_tunnel = DatabaseDAO.get_ssh_tunnel(1)
        assert response_tunnel
        assert isinstance(response_tunnel, SSHTunnel)
        assert 1 == response_tunnel.database_id
        response_delete_tunnel = client.delete(
            f'/api/v1/database/{database.id}/ssh_tunnel/')
        assert response_delete_tunnel.json['message'] == 'OK'
        response_tunnel = DatabaseDAO.get_ssh_tunnel(1)
        assert response_tunnel is None


def func_wnkhxin0(mocker, app, session, client, full_api_access):
    """
    Test that we cannot delete a tunnel that does not exist
    """
    with app.app_context():
        from superset.daos.database import DatabaseDAO
        from superset.databases.api import DatabaseRestApi
        from superset.databases.ssh_tunnel.models import SSHTunnel
        from superset.models.core import Database
        DatabaseRestApi.datamodel.session = session
        Database.metadata.create_all(session.get_bind())
        database = Database(database_name='my_database', sqlalchemy_uri=
            'gsheets://', encrypted_extra=json.dumps({
            'service_account_info': {'type': 'service_account',
            'project_id': 'black-sanctum-314419', 'private_key_id':
            '259b0d419a8f840056158763ff54d8b08f7b8173', 'private_key':
            'SECRET', 'client_email':
            'google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com'
            , 'client_id': 'SSH_TUNNEL_CREDENTIALS_CLIENT', 'auth_uri':
            'https://accounts.google.com/o/oauth2/auth', 'token_uri':
            'https://oauth2.googleapis.com/token',
            'auth_provider_x509_cert_url':
            'https://www.googleapis.com/oauth2/v1/certs',
            'client_x509_cert_url':
            'https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com'
            }}))
        db.session.add(database)
        db.session.commit()
        mocker.patch('sqlalchemy.engine.URL.get_driver_name', return_value=
            'gsheets')
        mocker.patch('superset.utils.log.DBEventLogger.log')
        mocker.patch(
            'superset.commands.database.ssh_tunnel.delete.is_feature_enabled',
            return_value=True)
        tunnel = SSHTunnel(database_id=1, database=database)
        db.session.add(tunnel)
        db.session.commit()
        response_delete_tunnel = client.delete('/api/v1/database/2/ssh_tunnel/'
            )
        assert response_delete_tunnel.json['message'] == 'Not found'
        response_tunnel = DatabaseDAO.get_ssh_tunnel(1)
        assert response_tunnel
        assert isinstance(response_tunnel, SSHTunnel)
        assert 1 == response_tunnel.database_id
        response_tunnel = DatabaseDAO.get_ssh_tunnel(2)
        assert response_tunnel is None


def func_q5xlczjt(mocker, app, session, client, full_api_access):
    """
    Test that we can filter the list of databases.
    First test the default behavior without a filter and then
    defining a filter function and patching the config to get
    the filtered results.
    """
    with app.app_context():
        from superset.daos.database import DatabaseDAO
        from superset.databases.api import DatabaseRestApi
        from superset.models.core import Database
        DatabaseRestApi.datamodel.session = session
        Database.metadata.create_all(session.get_bind())
        database = Database(database_name='first-database', sqlalchemy_uri=
            'gsheets://', encrypted_extra=json.dumps({'metadata_params': {},
            'engine_params': {}, 'metadata_cache_timeout': {},
            'schemas_allowed_for_file_upload': []}))
        db.session.add(database)
        db.session.commit()
        database = Database(database_name='second-database', sqlalchemy_uri
            ='gsheets://', encrypted_extra=json.dumps({'metadata_params': {
            }, 'engine_params': {}, 'metadata_cache_timeout': {},
            'schemas_allowed_for_file_upload': []}))
        db.session.add(database)
        db.session.commit()
        mocker.patch('sqlalchemy.engine.URL.get_driver_name', return_value=
            'gsheets')
        mocker.patch('superset.utils.log.DBEventLogger.log')
        mocker.patch(
            'superset.commands.database.ssh_tunnel.delete.is_feature_enabled',
            return_value=False)

        def func_snhevp0z(query):
            from superset.models.core import Database
            return query.filter(Database.database_name.startswith('second'))
        base_filter_mock = Mock(side_effect=_base_filter)
        response_databases = DatabaseDAO.find_all()
        assert response_databases
        expected_db_names = ['first-database', 'second-database']
        actual_db_names = [db.database_name for db in response_databases]
        assert actual_db_names == expected_db_names
        assert base_filter_mock.call_count == 0
        original_config = current_app.config.copy()
        original_config['EXTRA_DYNAMIC_QUERY_FILTERS'] = {'databases':
            base_filter_mock}
        mocker.patch('superset.views.filters.current_app.config', new=
            original_config)
        response_databases = DatabaseDAO.find_all()
        assert response_databases
        expected_db_names = ['second-database']
        actual_db_names = [db.database_name for db in response_databases]
        assert actual_db_names == expected_db_names
        assert base_filter_mock.call_count == 1


def func_r4a70w4t(mocker, session, client, full_api_access):
    """
    Test the OAuth2 endpoint when everything goes well.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database, DatabaseUserOAuth2Tokens
    DatabaseRestApi.datamodel.session = session
    Database.metadata.create_all(session.get_bind())
    db.session.add(Database(database_name='my_db', sqlalchemy_uri=
        'sqlite://', uuid=UUID('7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb')))
    db.session.commit()
    mocker.patch.object(SqliteEngineSpec, 'get_oauth2_config', return_value
        ={'id': 'one', 'secret': 'two'})
    get_oauth2_token = mocker.patch.object(SqliteEngineSpec, 'get_oauth2_token'
        )
    get_oauth2_token.return_value = {'access_token': 'YYY', 'expires_in': 
        3600, 'refresh_token': 'ZZZ'}
    state = {'user_id': 1, 'database_id': 1, 'tab_id': 42}
    decode_oauth2_state = mocker.patch(
        'superset.databases.api.decode_oauth2_state')
    decode_oauth2_state.return_value = state
    mocker.patch('superset.databases.api.render_template', return_value='OK')
    with freeze_time('2024-01-01T00:00:00Z'):
        response = client.get('/api/v1/database/oauth2/', query_string={
            'state': 'some%2Estate', 'code': 'XXX'})
    assert response.status_code == 200
    decode_oauth2_state.assert_called_with('some%2Estate')
    get_oauth2_token.assert_called_with({'id': 'one', 'secret': 'two'}, 'XXX')
    token = db.session.query(DatabaseUserOAuth2Tokens).one()
    assert token.user_id == 1
    assert token.database_id == 1
    assert token.access_token == 'YYY'
    assert token.access_token_expiration == datetime(2024, 1, 1, 1, 0)
    assert token.refresh_token == 'ZZZ'


def func_v2zj4nz0(mocker, session, client, full_api_access):
    """
    Test the OAuth2 endpoint when a second token is added.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database, DatabaseUserOAuth2Tokens
    DatabaseRestApi.datamodel.session = session
    Database.metadata.create_all(session.get_bind())
    db.session.add(Database(database_name='my_db', sqlalchemy_uri=
        'sqlite://', uuid=UUID('7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb')))
    db.session.commit()
    mocker.patch.object(SqliteEngineSpec, 'get_oauth2_config', return_value
        ={'id': 'one', 'secret': 'two'})
    get_oauth2_token = mocker.patch.object(SqliteEngineSpec, 'get_oauth2_token'
        )
    get_oauth2_token.side_effect = [{'access_token': 'YYY', 'expires_in': 
        3600, 'refresh_token': 'ZZZ'}, {'access_token': 'YYY2',
        'expires_in': 3600, 'refresh_token': 'ZZZ2'}]
    state = {'user_id': 1, 'database_id': 1, 'tab_id': 42}
    decode_oauth2_state = mocker.patch(
        'superset.databases.api.decode_oauth2_state')
    decode_oauth2_state.return_value = state
    mocker.patch('superset.databases.api.render_template', return_value='OK')
    with freeze_time('2024-01-01T00:00:00Z'):
        response = client.get('/api/v1/database/oauth2/', query_string={
            'state': 'some%2Estate', 'code': 'XXX'})
        response = client.get('/api/v1/database/oauth2/', query_string={
            'state': 'some%2Estate', 'code': 'XXX'})
    assert response.status_code == 200
    tokens = db.session.query(DatabaseUserOAuth2Tokens).all()
    assert len(tokens) == 1
    token = tokens[0]
    assert token.access_token == 'YYY2'
    assert token.refresh_token == 'ZZZ2'


def func_74psz6zs(mocker, session, client, full_api_access):
    """
    Test the OAuth2 endpoint when OAuth2 errors.
    """
    response = client.get('/api/v1/database/oauth2/', query_string={'error':
        'Something bad hapened'})
    assert response.status_code == 500
    assert response.json == {'errors': [{'message':
        'Something went wrong while doing OAuth2', 'error_type':
        'OAUTH2_REDIRECT_ERROR', 'level': 'error', 'extra': {'error':
        'Something bad hapened'}}]}


@pytest.mark.parametrize('payload,upload_called_with,reader_called_with', [
    ({'type': 'csv', 'file': (create_csv_file(), 'out.csv'), 'table_name':
    'table1', 'delimiter': ','}, (1, 'table1', ANY, None, ANY), ({'type':
    'csv', 'already_exists': 'fail', 'delimiter': ',', 'file': ANY,
    'table_name': 'table1'},)), ({'type': 'csv', 'file': (create_csv_file(),
    'out.csv'), 'table_name': 'table2', 'delimiter': ';', 'already_exists':
    'replace', 'column_dates': 'col1,col2'}, (1, 'table2', ANY, None, ANY),
    ({'type': 'csv', 'already_exists': 'replace', 'column_dates': ['col1',
    'col2'], 'delimiter': ';', 'file': ANY, 'table_name': 'table2'},)), ({
    'type': 'csv', 'file': (create_csv_file(), 'out.csv'), 'table_name':
    'table2', 'delimiter': ';', 'already_exists': 'replace', 'columns_read':
    'col1,col2', 'day_first': True, 'rows_to_read': '1', 'skip_blank_lines':
    True, 'skip_initial_space': True, 'skip_rows': '10', 'null_values':
    "None,N/A,''", 'column_data_types': '{"col1": "str"}'}, (1, 'table2',
    ANY, None, ANY), ({'type': 'csv', 'already_exists': 'replace',
    'columns_read': ['col1', 'col2'], 'null_values': ['None', 'N/A', "''"],
    'day_first': True, 'rows_to_read': 1, 'skip_blank_lines': True,
    'skip_initial_space': True, 'skip_rows': 10, 'delimiter': ';', 'file':
    ANY, 'column_data_types': {'col1': 'str'}, 'table_name': 'table2'},))])
def func_h3qsmvcw(payload, upload_called_with, reader_called_with, mocker,
    client, full_api_access):
    """
    Test CSV Upload success.
    """
    init_mock = mocker.patch.object(UploadCommand, '__init__')
    init_mock.return_value = None
    _ = mocker.patch.object(UploadCommand, 'run')
    reader_mock = mocker.patch.object(CSVReader, '__init__')
    reader_mock.return_value = None
    response = client.post('/api/v1/database/1/upload/', data=payload,
        content_type='multipart/form-data')
    assert response.status_code == 201
    assert response.json == {'message': 'OK'}
    init_mock.assert_called_with(*upload_called_with)
    reader_mock.assert_called_with(*reader_called_with)


@pytest.mark.parametrize('payload,expected_response', [({'type': 'csv',
    'file': (create_csv_file(), 'out.csv'), 'delimiter': ',',
    'already_exists': 'fail'}, {'message': {'table_name': [
    'Missing data for required field.']}}), ({'type': 'csv', 'file': (
    create_csv_file(), 'out.csv'), 'table_name': '', 'delimiter': ',',
    'already_exists': 'fail'}, {'message': {'table_name': [
    'Length must be between 1 and 10000.']}}), ({'type': 'csv',
    'table_name': 'table1', 'delimiter': ',', 'already_exists': 'fail'}, {
    'message': {'file': ['Field may not be null.']}}), ({'type': 'csv',
    'file': 'xpto', 'table_name': 'table1', 'delimiter': ',',
    'already_exists': 'fail'}, {'message': {'file': [
    'Field may not be null.']}}), ({'type': 'csv', 'file': (create_csv_file
    (), 'out.csv'), 'table_name': 'table1', 'delimiter': ',',
    'already_exists': 'xpto'}, {'message': {'already_exists': [
    'Must be one of: fail, replace, append.']}}), ({'type': 'csv', 'file':
    (create_csv_file(), 'out.csv'), 'table_name': 'table1', 'delimiter':
    ',', 'already_exists': 'fail', 'day_first': 'test1'}, {'message': {
    'day_first': ['Not a valid boolean.']}}), ({'type': 'csv', 'file': (
    create_csv_file(), 'out.csv'), 'table_name': 'table1', 'delimiter': ',',
    'already_exists': 'fail', 'header_row': 'test1'}, {'message': {
    'header_row': ['Not a valid integer.']}}), ({'type': 'csv', 'file': (
    create_csv_file(), 'out.csv'), 'table_name': 'table1', 'delimiter': ',',
    'already_exists': 'fail', 'rows_to_read': 0}, {'message': {
    'rows_to_read': ['Must be greater than or equal to 1.']}}), ({'type':
    'csv', 'file': (create_csv_file(), 'out.csv'), 'table_name': 'table1',
    'delimiter': ',', 'already_exists': 'fail', 'skip_blank_lines': 'test1'
    }, {'message': {'skip_blank_lines': ['Not a valid boolean.']}}), ({
    'type': 'csv', 'file': (create_csv_file(), 'out.csv'), 'table_name':
    'table1', 'delimiter': ',', 'already_exists': 'fail',
    'skip_initial_space': 'test1'}, {'message': {'skip_initial_space': [
    'Not a valid boolean.']}}), ({'type': 'csv', 'file': (create_csv_file(),
    'out.csv'), 'table_name': 'table1', 'delimiter': ',', 'already_exists':
    'fail', 'skip_rows': 'test1'}, {'message': {'skip_rows': [
    'Not a valid integer.']}}), ({'type': 'csv', 'file': (create_csv_file(),
    'out.csv'), 'table_name': 'table1', 'delimiter': ',', 'already_exists':
    'fail', 'column_data_types': '{test:1}'}, {'message': {'_schema': [
    'Invalid JSON format for column_data_types']}})])
def func_z2a21j2s(payload, expected_response, mocker, client, full_api_access):
    """
    Test CSV Upload validation fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data=payload,
        content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == expected_response


@pytest.mark.parametrize('filename', ['out.xpto', 'out.exe', 'out',
    'out csv', '', 'out.csv.exe', '.csv', 'out.', '.', 'out csv a.exe'])
def func_tk5lylvv(filename, mocker, client, full_api_access):
    """
    Test CSV Upload validation fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data={'type':
        'csv', 'file': create_csv_file(filename=filename), 'table_name':
        'table1', 'delimiter': ','}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': [
        'File extension is not allowed.']}}


@pytest.mark.parametrize('filename', ['out.csv', 'out.txt', 'out.tsv',
    'spaced name.csv', 'spaced name.txt', 'spaced name.tsv', 'out.exe.csv',
    'out.csv.csv'])
def func_kzrwserm(filename, mocker, client, full_api_access):
    """
    Test CSV Upload validation fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data={'type':
        'csv', 'file': create_csv_file(filename=filename), 'table_name':
        'table1', 'delimiter': ','}, content_type='multipart/form-data')
    assert response.status_code == 201


@pytest.mark.parametrize('payload,upload_called_with,reader_called_with', [
    ({'type': 'excel', 'file': (create_excel_file(), 'out.xls'),
    'table_name': 'table1'}, (1, 'table1', ANY, None, ANY), ({'type':
    'excel', 'already_exists': 'fail', 'file': ANY, 'table_name': 'table1'}
    ,)), ({'type': 'excel', 'file': (create_excel_file(), 'out.xls'),
    'table_name': 'table2', 'sheet_name': 'Sheet1', 'already_exists':
    'replace', 'column_dates': 'col1,col2'}, (1, 'table2', ANY, None, ANY),
    ({'type': 'excel', 'already_exists': 'replace', 'column_dates': ['col1',
    'col2'], 'sheet_name': 'Sheet1', 'file': ANY, 'table_name': 'table2'},)
    ), ({'type': 'excel', 'file': (create_excel_file(), 'out.xls'),
    'table_name': 'table2', 'sheet_name': 'Sheet1', 'already_exists':
    'replace', 'columns_read': 'col1,col2', 'rows_to_read': '1',
    'skip_rows': '10', 'null_values': "None,N/A,''"}, (1, 'table2', ANY,
    None, ANY), ({'type': 'excel', 'already_exists': 'replace',
    'columns_read': ['col1', 'col2'], 'null_values': ['None', 'N/A', "''"],
    'rows_to_read': 1, 'skip_rows': 10, 'sheet_name': 'Sheet1', 'file': ANY,
    'table_name': 'table2'},))])
def func_lxnph2sf(payload, upload_called_with, reader_called_with, mocker,
    client, full_api_access):
    """
    Test Excel Upload success.
    """
    init_mock = mocker.patch.object(UploadCommand, '__init__')
    init_mock.return_value = None
    _ = mocker.patch.object(UploadCommand, 'run')
    reader_mock = mocker.patch.object(ExcelReader, '__init__')
    reader_mock.return_value = None
    response = client.post('/api/v1/database/1/upload/', data=payload,
        content_type='multipart/form-data')
    assert response.status_code == 201
    assert response.json == {'message': 'OK'}
    init_mock.assert_called_with(*upload_called_with)
    reader_mock.assert_called_with(*reader_called_with)


@pytest.mark.parametrize('payload,expected_response', [({'type': 'excel',
    'file': (create_excel_file(), 'out.xls'), 'sheet_name': 'Sheet1',
    'already_exists': 'fail'}, {'message': {'table_name': [
    'Missing data for required field.']}}), ({'type': 'excel', 'file': (
    create_excel_file(), 'out.xls'), 'table_name': '', 'sheet_name':
    'Sheet1', 'already_exists': 'fail'}, {'message': {'table_name': [
    'Length must be between 1 and 10000.']}}), ({'type': 'excel',
    'table_name': 'table1', 'already_exists': 'fail'}, {'message': {'file':
    ['Field may not be null.']}}), ({'type': 'excel', 'file': 'xpto',
    'table_name': 'table1', 'already_exists': 'fail'}, {'message': {'file':
    ['Field may not be null.']}}), ({'type': 'excel', 'file': (
    create_excel_file(), 'out.xls'), 'table_name': 'table1',
    'already_exists': 'xpto'}, {'message': {'already_exists': [
    'Must be one of: fail, replace, append.']}}), ({'type': 'excel', 'file':
    (create_excel_file(), 'out.xls'), 'table_name': 'table1',
    'already_exists': 'fail', 'header_row': 'test1'}, {'message': {
    'header_row': ['Not a valid integer.']}}), ({'type': 'excel', 'file': (
    create_excel_file(), 'out.xls'), 'table_name': 'table1',
    'already_exists': 'fail', 'rows_to_read': 0}, {'message': {
    'rows_to_read': ['Must be greater than or equal to 1.']}}), ({'type':
    'excel', 'file': (create_excel_file(), 'out.xls'), 'table_name':
    'table1', 'already_exists': 'fail', 'skip_rows': 'test1'}, {'message':
    {'skip_rows': ['Not a valid integer.']}})])
def func_ne9aluax(payload, expected_response, mocker, client, full_api_access):
    """
    Test Excel Upload validation fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data=payload,
        content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == expected_response


@pytest.mark.parametrize('filename', ['out.xpto', 'out.exe', 'out',
    'out xls', '', 'out.slx.exe', '.xls', 'out.', '.', 'out xls a.exe'])
def func_o7c8zfjv(filename, mocker, client, full_api_access):
    """
    Test Excel Upload file extension fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data={'type':
        'excel', 'file': create_excel_file(filename=filename), 'table_name':
        'table1'}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': [
        'File extension is not allowed.']}}


@pytest.mark.parametrize('payload,upload_called_with,reader_called_with', [
    ({'type': 'columnar', 'file': (create_columnar_file(), 'out.parquet'),
    'table_name': 'table1'}, (1, 'table1', ANY, None, ANY), ({'type':
    'columnar', 'already_exists': 'fail', 'file': ANY, 'table_name':
    'table1'},)), ({'type': 'columnar', 'file': (create_columnar_file(),
    'out.parquet'), 'table_name': 'table2', 'already_exists': 'replace',
    'columns_read': 'col1,col2', 'dataframe_index': True, 'index_label':
    'label'}, (1, 'table2', ANY, None, ANY), ({'type': 'columnar',
    'already_exists': 'replace', 'columns_read': ['col1', 'col2'], 'file':
    ANY, 'table_name': 'table2', 'dataframe_index': True, 'index_label':
    'label'},))])
def func_mz30q159(payload, upload_called_with, reader_called_with, mocker,
    client, full_api_access):
    """
    Test Excel Upload success.
    """
    init_mock = mocker.patch.object(UploadCommand, '__init__')
    init_mock.return_value = None
    _ = mocker.patch.object(UploadCommand, 'run')
    reader_mock = mocker.patch.object(ColumnarReader, '__init__')
    reader_mock.return_value = None
    response = client.post('/api/v1/database/1/upload/', data=payload,
        content_type='multipart/form-data')
    assert response.status_code == 201
    assert response.json == {'message': 'OK'}
    init_mock.assert_called_with(*upload_called_with)
    reader_mock.assert_called_with(*reader_called_with)


@pytest.mark.parametrize('payload,expected_response', [({'type': 'columnar',
    'file': (create_columnar_file(), 'out.parquet'), 'already_exists':
    'fail'}, {'message': {'table_name': ['Missing data for required field.'
    ]}}), ({'type': 'columnar', 'file': (create_columnar_file(),
    'out.parquet'), 'table_name': '', 'already_exists': 'fail'}, {'message':
    {'table_name': ['Length must be between 1 and 10000.']}}), ({'type':
    'columnar', 'table_name': 'table1', 'already_exists': 'fail'}, {
    'message': {'file': ['Field may not be null.']}}), ({'type': 'columnar',
    'file': 'xpto', 'table_name': 'table1', 'already_exists': 'fail'}, {
    'message': {'file': ['Field may not be null.']}}), ({'type': 'columnar',
    'file': (create_columnar_file(), 'out.parquet'), 'table_name': 'table1',
    'already_exists': 'xpto'}, {'message': {'already_exists': [
    'Must be one of: fail, replace, append.']}})])
def func_9te1vz3y(payload, expected_response, mocker, client, full_api_access):
    """
    Test Excel Upload validation fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data=payload,
        content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == expected_response


@pytest.mark.parametrize('filename', ['out.parquet', 'out.zip',
    'out.parquet.zip', 'out something.parquet', 'out something.zip'])
def func_8yp5vk35(filename, mocker, client, full_api_access):
    """
    Test Excel Upload file extension fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data={'type':
        'columnar', 'file': (create_columnar_file(), filename),
        'table_name': 'table1'}, content_type='multipart/form-data')
    assert response.status_code == 201


@pytest.mark.parametrize('filename', ['out.xpto', 'out.exe', 'out',
    'out zip', '', 'out.parquet.exe', '.parquet', 'out.', '.',
    'out parquet a.exe'])
def func_jymyzhy0(filename, mocker, client, full_api_access):
    """
    Test Excel Upload file extension fails.
    """
    _ = mocker.patch.object(UploadCommand, 'run')
    response = client.post('/api/v1/database/1/upload/', data={'type':
        'columnar', 'file': create_columnar_file(filename=filename),
        'table_name': 'table1'}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': [
        'File extension is not allowed.']}}


def func_abbnk8fc(mocker, client, full_api_access):
    _ = mocker.patch.object(CSVReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'csv', 'file': create_csv_file()}, content_type=
        'multipart/form-data')
    assert response.status_code == 200


def func_o31itm0y(mocker, client, full_api_access):
    _ = mocker.patch.object(CSVReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'csv', 'file': create_csv_file(filename='test.out')},
        content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': [
        'File extension is not allowed.']}}


def func_fgtci24p(mocker, client, full_api_access):
    _ = mocker.patch.object(CSVReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'csv'}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': ['Field may not be null.']}}
    response = client.post('/api/v1/database/upload_metadata/', data={
        'file': create_csv_file(filename='test.csv')}, content_type=
        'multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'type': [
        'Missing data for required field.']}}


def func_v7jzi8ul(mocker, client, full_api_access):
    _ = mocker.patch.object(ExcelReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'excel', 'file': create_excel_file()}, content_type=
        'multipart/form-data')
    assert response.status_code == 200


def func_bmb49e8t(mocker, client, full_api_access):
    _ = mocker.patch.object(ExcelReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'excel', 'file': create_excel_file(filename='test.out')},
        content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': [
        'File extension is not allowed.']}}


def func_78nsw5vj(mocker, client, full_api_access):
    _ = mocker.patch.object(ExcelReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'excel'}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': ['Field may not be null.']}}


def func_zvwbdso4(mocker, client, full_api_access):
    _ = mocker.patch.object(ColumnarReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'columnar', 'file': create_columnar_file()}, content_type=
        'multipart/form-data')
    assert response.status_code == 200


def func_7kkxxr3o(mocker, client, full_api_access):
    _ = mocker.patch.object(ColumnarReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'columnar', 'file': create_columnar_file(filename=
        'test.out')}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': [
        'File extension is not allowed.']}}


def func_ccdvhm4e(mocker, client, full_api_access):
    _ = mocker.patch.object(ColumnarReader, 'file_metadata')
    response = client.post('/api/v1/database/upload_metadata/', data={
        'type': 'columnar'}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json == {'message': {'file': ['Field may not be null.']}}


def func_myl0bgmk(mocker, client, full_api_access):
    """
    Test the `table_metadata` endpoint.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.get_table_metadata.return_value = {'hello': 'world'
        }
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    mocker.patch('superset.databases.api.security_manager.raise_for_access')
    response = client.get('/api/v1/database/1/table_metadata/?name=t')
    assert response.json == {'hello': 'world'}
    database.db_engine_spec.get_table_metadata.assert_called_with(database,
        Table('t'))
    response = client.get('/api/v1/database/1/table_metadata/?name=t&schema=s')
    database.db_engine_spec.get_table_metadata.assert_called_with(database,
        Table('t', 's'))
    response = client.get('/api/v1/database/1/table_metadata/?name=t&catalog=c'
        )
    database.db_engine_spec.get_table_metadata.assert_called_with(database,
        Table('t', None, 'c'))
    response = client.get(
        '/api/v1/database/1/table_metadata/?name=t&schema=s&catalog=c')
    database.db_engine_spec.get_table_metadata.assert_called_with(database,
        Table('t', 's', 'c'))


def func_x8mooax4(mocker, client, full_api_access):
    """
    Test the `table_metadata` endpoint when no table name is passed.
    """
    database = mocker.MagicMock()
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    response = client.get(
        '/api/v1/database/1/table_metadata/?schema=s&catalog=c')
    assert response.status_code == 422
    assert response.json == {'errors': [{'message':
        'An error happened when validating the request', 'error_type':
        'INVALID_PAYLOAD_SCHEMA_ERROR', 'level': 'error', 'extra': {
        'messages': {'name': ['Missing data for required field.']},
        'issue_codes': [{'code': 1020, 'message':
        'Issue 1020 - The submitted payload has the incorrect schema.'}]}}]}


def func_29himklm(mocker, client, full_api_access):
    """
    Test the `table_metadata` endpoint with names that have slashes.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.get_table_metadata.return_value = {'hello': 'world'
        }
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    mocker.patch('superset.databases.api.security_manager.raise_for_access')
    client.get('/api/v1/database/1/table_metadata/?name=foo/bar')
    database.db_engine_spec.get_table_metadata.assert_called_with(database,
        Table('foo/bar'))


def func_s04qfkwx(mocker, client, full_api_access):
    """
    Test the `table_metadata` endpoint when the database is invalid.
    """
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=None)
    response = client.get('/api/v1/database/1/table_metadata/?name=t')
    assert response.status_code == 404
    assert response.json == {'errors': [{'message': 'No such database',
        'error_type': 'DATABASE_NOT_FOUND_ERROR', 'level': 'error', 'extra':
        {'issue_codes': [{'code': 1011, 'message':
        'Issue 1011 - Superset encountered an unexpected error.'}, {'code':
        1036, 'message': 'Issue 1036 - The database was deleted.'}]}}]}


def func_wmtf41ci(mocker, client, full_api_access):
    """
    Test the `table_metadata` endpoint when the user is unauthorized.
    """
    database = mocker.MagicMock()
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    mocker.patch('superset.databases.api.security_manager.raise_for_access',
        side_effect=SupersetSecurityException(SupersetError(error_type=
        SupersetErrorType.TABLE_SECURITY_ACCESS_ERROR, message=
        "You don't have access to the table", level=ErrorLevel.ERROR)))
    response = client.get('/api/v1/database/1/table_metadata/?name=t')
    assert response.status_code == 404
    assert response.json == {'errors': [{'message': 'No such table',
        'error_type': 'TABLE_NOT_FOUND_ERROR', 'level': 'error', 'extra':
        None}]}


def func_ccvrrn5u(mocker, client, full_api_access):
    """
    Test the `table_extra_metadata` endpoint.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.get_extra_table_metadata.return_value = {'hello':
        'world'}
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    mocker.patch('superset.databases.api.security_manager.raise_for_access')
    response = client.get('/api/v1/database/1/table_metadata/extra/?name=t')
    assert response.json == {'hello': 'world'}
    database.db_engine_spec.get_extra_table_metadata.assert_called_with(
        database, Table('t'))
    response = client.get(
        '/api/v1/database/1/table_metadata/extra/?name=t&schema=s')
    database.db_engine_spec.get_extra_table_metadata.assert_called_with(
        database, Table('t', 's'))
    response = client.get(
        '/api/v1/database/1/table_metadata/extra/?name=t&catalog=c')
    database.db_engine_spec.get_extra_table_metadata.assert_called_with(
        database, Table('t', None, 'c'))
    response = client.get(
        '/api/v1/database/1/table_metadata/extra/?name=t&schema=s&catalog=c')
    database.db_engine_spec.get_extra_table_metadata.assert_called_with(
        database, Table('t', 's', 'c'))


def func_q72tk33g(mocker, client, full_api_access):
    """
    Test the `table_extra_metadata` endpoint when no table name is passed.
    """
    database = mocker.MagicMock()
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    response = client.get(
        '/api/v1/database/1/table_metadata/extra/?schema=s&catalog=c')
    assert response.status_code == 422
    assert response.json == {'errors': [{'message':
        'An error happened when validating the request', 'error_type':
        'INVALID_PAYLOAD_SCHEMA_ERROR', 'level': 'error', 'extra': {
        'messages': {'name': ['Missing data for required field.']},
        'issue_codes': [{'code': 1020, 'message':
        'Issue 1020 - The submitted payload has the incorrect schema.'}]}}]}


def func_3lxwmpx3(mocker, client, full_api_access):
    """
    Test the `table_extra_metadata` endpoint with names that have slashes.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.get_extra_table_metadata.return_value = {'hello':
        'world'}
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    mocker.patch('superset.databases.api.security_manager.raise_for_access')
    client.get('/api/v1/database/1/table_metadata/extra/?name=foo/bar')
    database.db_engine_spec.get_extra_table_metadata.assert_called_with(
        database, Table('foo/bar'))


def func_i4vxl6j3(mocker, client, full_api_access):
    """
    Test the `table_extra_metadata` endpoint when the database is invalid.
    """
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=None)
    response = client.get('/api/v1/database/1/table_metadata/extra/?name=t')
    assert response.status_code == 404
    assert response.json == {'errors': [{'message': 'No such database',
        'error_type': 'DATABASE_NOT_FOUND_ERROR', 'level': 'error', 'extra':
        {'issue_codes': [{'code': 1011, 'message':
        'Issue 1011 - Superset encountered an unexpected error.'}, {'code':
        1036, 'message': 'Issue 1036 - The database was deleted.'}]}}]}


def func_ctp94qn3(mocker, client, full_api_access):
    """
    Test the `table_extra_metadata` endpoint when the user is unauthorized.
    """
    database = mocker.MagicMock()
    mocker.patch('superset.databases.api.DatabaseDAO.find_by_id',
        return_value=database)
    mocker.patch('superset.databases.api.security_manager.raise_for_access',
        side_effect=SupersetSecurityException(SupersetError(error_type=
        SupersetErrorType.TABLE_SECURITY_ACCESS_ERROR, message=
        "You don't have access to the table", level=ErrorLevel.ERROR)))
    response = client.get('/api/v1/database/1/table_metadata/extra/?name=t')
    assert response.status_code == 404
    assert response.json == {'errors': [{'message': 'No such table',
        'error_type': 'TABLE_NOT_FOUND_ERROR', 'level': 'error', 'extra':
        None}]}


def func_ego1watg(mocker, client, full_api_access):
    """
    Test the `catalogs` endpoint.
    """
    database = mocker.MagicMock()
    database.get_all_catalog_names.return_value = {'db1', 'db2'}
    DatabaseDAO = mocker.patch('superset.databases.api.DatabaseDAO')
    DatabaseDAO.find_by_id.return_value = database
    security_manager = mocker.patch('superset.databases.api.security_manager',
        new=mocker.MagicMock())
    security_manager.get_catalogs_accessible_by_user.return_value = {'db2'}
    response = client.get('/api/v1/database/1/catalogs/')
    assert response.status_code == 200
    assert response.json == {'result': ['db2']}
    database.get_all_catalog_names.assert_called_with(cache=database.
        catalog_cache_enabled, cache_timeout=database.catalog_cache_timeout,
        force=False)
    security_manager.get_catalogs_accessible_by_user.assert_called_with(
        database, {'db1', 'db2'})
    response = client.get('/api/v1/database/1/catalogs/?q=(force:!t)')
    database.get_all_catalog_names.assert_called_with(cache=database.
        catalog_cache_enabled, cache_timeout=database.catalog_cache_timeout,
        force=True)


def func_bilt9sc4(mocker, client, full_api_access):
    """
    Test the `catalogs` endpoint when OAuth2 is needed.
    """
    database = mocker.MagicMock()
    database.get_all_catalog_names.side_effect = OAuth2RedirectError('url',
        'tab_id', 'redirect_uri')
    DatabaseDAO = mocker.patch('superset.databases.api.DatabaseDAO')
    DatabaseDAO.find_by_id.return_value = database
    security_manager = mocker.patch('superset.databases.api.security_manager',
        new=mocker.MagicMock())
    security_manager.get_catalogs_accessible_by_user.return_value = {'db2'}
    response = client.get('/api/v1/database/1/catalogs/')
    assert response.status_code == 500
    assert response.json == {'errors': [{'message':
        "You don't have permission to access the data.", 'error_type':
        'OAUTH2_REDIRECT', 'level': 'warning', 'extra': {'url': 'url',
        'tab_id': 'tab_id', 'redirect_uri': 'redirect_uri'}}]}


def func_orfnbnm2(mocker, client, full_api_access):
    """
    Test the `schemas` endpoint.
    """
    from superset.databases.api import DatabaseRestApi
    database = mocker.MagicMock()
    database.get_all_schema_names.return_value = {'schema1', 'schema2'}
    datamodel = mocker.patch.object(DatabaseRestApi, 'datamodel')
    datamodel.get.return_value = database
    security_manager = mocker.patch('superset.databases.api.security_manager',
        new=mocker.MagicMock())
    security_manager.get_schemas_accessible_by_user.return_value = {'schema2'}
    response = client.get('/api/v1/database/1/schemas/')
    assert response.status_code == 200
    assert response.json == {'result': ['schema2']}
    database.get_all_schema_names.assert_called_with(catalog=None, cache=
        database.schema_cache_enabled, cache_timeout=database.
        schema_cache_timeout, force=False)
    security_manager.get_schemas_accessible_by_user.assert_called_with(database
        , None, {'schema1', 'schema2'})
    response = client.get('/api/v1/database/1/schemas/?q=(force:!t)')
    database.get_all_schema_names.assert_called_with(catalog=None, cache=
        database.schema_cache_enabled, cache_timeout=database.
        schema_cache_timeout, force=True)
    response = client.get(
        '/api/v1/database/1/schemas/?q=(force:!t,catalog:catalog2)')
    database.get_all_schema_names.assert_called_with(catalog='catalog2',
        cache=database.schema_cache_enabled, cache_timeout=database.
        schema_cache_timeout, force=True)
    security_manager.get_schemas_accessible_by_user.assert_called_with(database
        , 'catalog2', {'schema1', 'schema2'})


def func_l2l3ss9k(mocker, client, full_api_access):
    """
    Test the `schemas` endpoint when OAuth2 is needed.
    """
    from superset.databases.api import DatabaseRestApi
    database = mocker.MagicMock()
    database.get_all_schema_names.side_effect = OAuth2RedirectError('url',
        'tab_id', 'redirect_uri')
    datamodel = mocker.patch.object(DatabaseRestApi, 'datamodel')
    datamodel.get.return_value = database
    security_manager = mocker.patch('superset.databases.api.security_manager',
        new=mocker.MagicMock())
    security_manager.get_schemas_accessible_by_user.return_value = {'schema2'}
    response = client.get('/api/v1/database/1/schemas/')
    assert response.status_code == 500
    assert response.json == {'errors': [{'message':
        "You don't have permission to access the data.", 'error_type':
        'OAUTH2_REDIRECT', 'level': 'warning', 'extra': {'url': 'url',
        'tab_id': 'tab_id', 'redirect_uri': 'redirect_uri'}}]}
