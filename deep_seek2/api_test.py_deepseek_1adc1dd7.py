from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from io import BytesIO
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
from tests.unit_tests.fixtures.common import (
    create_columnar_file,
    create_csv_file,
    create_excel_file,
)

def test_filter_by_uuid(
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database

    DatabaseRestApi.datamodel.session = session

    Database.metadata.create_all(session.get_bind())
    db.session.add(
        Database(
            database_name="my_db",
            sqlalchemy_uri="sqlite://",
            uuid=UUID("7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb"),
        )
    )
    db.session.commit()

    response = client.get(
        "/api/v1/database/?q=(filters:!((col:uuid,opr:eq,value:"
        "%277c1b7880-a59d-47cd-8bf1-f1eb8d2863cb%27)))"
    )
    assert response.status_code == 200

    payload = response.json
    assert len(payload["result"]) == 1
    assert payload["result"][0]["uuid"] == "7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb"

def test_post_with_uuid(
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    from superset.models.core import Database

    Database.metadata.create_all(session.get_bind())

    response = client.post(
        "/api/v1/database/",
        json={
            "database_name": "my_db",
            "sqlalchemy_uri": "sqlite://",
            "uuid": "7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb",
        },
    )
    assert response.status_code == 201

    payload = response.json
    assert payload["result"]["uuid"] == "7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb"

    database = session.query(Database).one()
    assert database.uuid == UUID("7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb")

def test_password_mask(
    mocker: MockerFixture,
    app: Any,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database

    DatabaseRestApi.datamodel.session = session

    Database.metadata.create_all(session.get_bind())

    database = Database(
        uuid=UUID("02feae18-2dd6-4bb4-a9c0-49e9d4f29d58"),
        database_name="my_database",
        sqlalchemy_uri="gsheets://",
        encrypted_extra=json.dumps(
            {
                "service_account_info": {
                    "type": "service_account",
                    "project_id": "black-sanctum-314419",
                    "private_key_id": "259b0d419a8f840056158763ff54d8b08f7b8173",
                    "private_key": "SECRET",
                    "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",
                    "client_id": "114567578578109757129",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com",
                },
            }
        ),
    )
    db.session.add(database)
    db.session.commit()

    mocker.patch("sqlalchemy.engine.URL.get_driver_name", return_value="gsheets")
    mocker.patch("superset.utils.log.DBEventLogger.log")

    response = client.get("/api/v1/database/1/connection")
    assert (
        response.json["result"]["parameters"]["service_account_info"]["private_key"]
        == "XXXXXXXXXX"
    )
    assert "encrypted_extra" not in response.json["result"]

def test_database_connection(
    mocker: MockerFixture,
    app: Any,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database

    DatabaseRestApi.datamodel.session = session

    Database.metadata.create_all(session.get_bind())

    database = Database(
        uuid=UUID("02feae18-2dd6-4bb4-a9c0-49e9d4f29d58"),
        database_name="my_database",
        sqlalchemy_uri="gsheets://",
        encrypted_extra=json.dumps(
            {
                "service_account_info": {
                    "type": "service_account",
                    "project_id": "black-sanctum-314419",
                    "private_key_id": "259b0d419a8f840056158763ff54d8b08f7b8173",
                    "private_key": "SECRET",
                    "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",
                    "client_id": "114567578578109757129",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com",
                },
            }
        ),
    )
    db.session.add(database)
    db.session.commit()

    mocker.patch("sqlalchemy.engine.URL.get_driver_name", return_value="gsheets")
    mocker.patch("superset.utils.log.DBEventLogger.log")

    response = client.get("/api/v1/database/1/connection")
    assert response.json == {
        "id": 1,
        "result": {
            "allow_ctas": False,
            "allow_cvas": False,
            "allow_dml": False,
            "allow_file_upload": False,
            "allow_run_async": False,
            "backend": "gsheets",
            "cache_timeout": None,
            "configuration_method": "sqlalchemy_form",
            "database_name": "my_database",
            "driver": "gsheets",
            "engine_information": {
                "disable_ssh_tunneling": True,
                "supports_dynamic_catalog": False,
                "supports_file_upload": True,
                "supports_oauth2": True,
            },
            "expose_in_sqllab": True,
            "extra": '{\n    "metadata_params": {},\n    "engine_params": {},\n    "metadata_cache_timeout": {},\n    "schemas_allowed_for_file_upload": []\n}\n',
            "force_ctas_schema": None,
            "id": 1,
            "impersonate_user": False,
            "is_managed_externally": False,
            "masked_encrypted_extra": json.dumps(
                {
                    "service_account_info": {
                        "type": "service_account",
                        "project_id": "black-sanctum-314419",
                        "private_key_id": "259b0d419a8f840056158763ff54d8b08f7b8173",
                        "private_key": "XXXXXXXXXX",
                        "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",
                        "client_id": "114567578578109757129",
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com",
                    }
                }
            ),
            "parameters": {
                "service_account_info": {
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",
                    "client_id": "114567578578109757129",
                    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com",
                    "private_key": "XXXXXXXXXX",
                    "private_key_id": "259b0d419a8f840056158763ff54d8b08f7b8173",
                    "project_id": "black-sanctum-314419",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "type": "service_account",
                }
            },
            "parameters_schema": {
                "properties": {
                    "catalog": {"type": "object"},
                    "service_account_info": {
                        "description": "Contents of GSheets JSON credentials.",
                        "type": "string",
                        "x-encrypted-extra": True,
                    },
                },
                "type": "object",
            },
            "server_cert": None,
            "sqlalchemy_uri": "gsheets://",
            "uuid": "02feae18-2dd6-4bb4-a9c0-49e9d4f29d58",
        },
    }

    response = client.get("/api/v1/database/1")
    assert response.json == {
        "id": 1,
        "result": {
            "allow_ctas": False,
            "allow_cvas": False,
            "allow_dml": False,
            "allow_file_upload": False,
            "allow_run_async": False,
            "backend": "gsheets",
            "cache_timeout": None,
            "configuration_method": "sqlalchemy_form",
            "database_name": "my_database",
            "driver": "gsheets",
            "engine_information": {
                "disable_ssh_tunneling": True,
                "supports_dynamic_catalog": False,
                "supports_file_upload": True,
                "supports_oauth2": True,
            },
            "expose_in_sqllab": True,
            "force_ctas_schema": None,
            "id": 1,
            "impersonate_user": False,
            "is_managed_externally": False,
            "uuid": "02feae18-2dd6-4bb4-a9c0-49e9d4f29d58",
        },
    }

@pytest.mark.skip(reason="Works locally but fails on CI")
def test_update_with_password_mask(
    app: Any,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database

    DatabaseRestApi.datamodel.session = session

    Database.metadata.create_all(session.get_bind())

    database = Database(
        database_name="my_database",
        sqlalchemy_uri="gsheets://",
        encrypted_extra=json.dumps(
            {
                "service_account_info": {
                    "project_id": "black-sanctum-314419",
                    "private_key": "SECRET",
                },
            }
        ),
    )
    db.session.add(database)
    db.session.commit()

    client.put(
        "/api/v1/database/1",
        json={
            "encrypted_extra": json.dumps(
                {
                    "service_account_info": {
                        "project_id": "yellow-unicorn-314419",
                        "private_key": "XXXXXXXXXX",
                    },
                }
            ),
        },
    )
    database = db.session.query(Database).one()
    assert (
        database.encrypted_extra
        == '{"service_account_info": {"project_id": "yellow-unicorn-314419", "private_key": "SECRET"}}'
    )

def test_non_zip_import(client: Any, full_api_access: None) -> None:
    buf = BytesIO(b"definitely_not_a_zip_file")
    form_data = {
        "formData": (buf, "evil.pdf"),
    }
    response = client.post(
        "/api/v1/database/import/",
        data=form_data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    assert response.json == {
        "errors": [
            {
                "message": "Not a ZIP file",
                "error_type": "GENERIC_COMMAND_ERROR",
                "level": "warning",
                "extra": {
                    "issue_codes": [
                        {
                            "code": 1010,
                            "message": "Issue 1010 - Superset encountered an error while running a command.",
                        }
                    ]
                },
            }
        ]
    }

def test_delete_ssh_tunnel(
    mocker: MockerFixture,
    app: Any,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    with app.app_context():
        from superset.daos.database import DatabaseDAO
        from superset.databases.api import DatabaseRestApi
        from superset.databases.ssh_tunnel.models import SSHTunnel
        from superset.models.core import Database

        DatabaseRestApi.datamodel.session = session

        Database.metadata.create_all(session.get_bind())

        database = Database(
            database_name="my_database",
            sqlalchemy_uri="gsheets://",
            encrypted_extra=json.dumps(
                {
                    "service_account_info": {
                        "type": "service_account",
                        "project_id": "black-sanctum-314419",
                        "private_key_id": "259b0d419a8f840056158763ff54d8b08f7b8173",
                        "private_key": "SECRET",
                        "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",
                        "client_id": "SSH_TUNNEL_CREDENTIALS_CLIENT",
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com",
                    },
                }
            ),
        )
        db.session.add(database)
        db.session.commit()

        mocker.patch("sqlalchemy.engine.URL.get_driver_name", return_value="gsheets")
        mocker.patch("superset.utils.log.DBEventLogger.log")
        mocker.patch(
            "superset.commands.database.ssh_tunnel.delete.is_feature_enabled",
            return_value=True,
        )

        tunnel = SSHTunnel(
            database_id=1,
            database=database,
        )

        db.session.add(tunnel)
        db.session.commit()

        response_tunnel = DatabaseDAO.get_ssh_tunnel(1)
        assert response_tunnel
        assert isinstance(response_tunnel, SSHTunnel)
        assert 1 == response_tunnel.database_id

        response_delete_tunnel = client.delete(
            f"/api/v1/database/{database.id}/ssh_tunnel/"
        )
        assert response_delete_tunnel.json["message"] == "OK"

        response_tunnel = DatabaseDAO.get_ssh_tunnel(1)
        assert response_tunnel is None

def test_delete_ssh_tunnel_not_found(
    mocker: MockerFixture,
    app: Any,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    with app.app_context():
        from superset.daos.database import DatabaseDAO
        from superset.databases.api import DatabaseRestApi
        from superset.databases.ssh_tunnel.models import SSHTunnel
        from superset.models.core import Database

        DatabaseRestApi.datamodel.session = session

        Database.metadata.create_all(session.get_bind())

        database = Database(
            database_name="my_database",
            sqlalchemy_uri="gsheets://",
            encrypted_extra=json.dumps(
                {
                    "service_account_info": {
                        "type": "service_account",
                        "project_id": "black-sanctum-314419",
                        "private_key_id": "259b0d419a8f840056158763ff54d8b08f7b8173",
                        "private_key": "SECRET",
                        "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",
                        "client_id": "SSH_TUNNEL_CREDENTIALS_CLIENT",
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-spreadsheets-demo-servi%40black-sanctum-314419.iam.gserviceaccount.com",
                    },
                }
            ),
        )
        db.session.add(database)
        db.session.commit()

        mocker.patch("sqlalchemy.engine.URL.get_driver_name", return_value="gsheets")
        mocker.patch("superset.utils.log.DBEventLogger.log")
        mocker.patch(
            "superset.commands.database.ssh_tunnel.delete.is_feature_enabled",
            return_value=True,
        )

        tunnel = SSHTunnel(
            database_id=1,
            database=database,
        )

        db.session.add(tunnel)
        db.session.commit()

        response_delete_tunnel = client.delete("/api/v1/database/2/ssh_tunnel/")
        assert response_delete_tunnel.json["message"] == "Not found"

        response_tunnel = DatabaseDAO.get_ssh_tunnel(1)
        assert response_tunnel
        assert isinstance(response_tunnel, SSHTunnel)
        assert 1 == response_tunnel.database_id

        response_tunnel = DatabaseDAO.get_ssh_tunnel(2)
        assert response_tunnel is None

def test_apply_dynamic_database_filter(
    mocker: MockerFixture,
    app: Any,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    with app.app_context():
        from superset.daos.database import DatabaseDAO
        from superset.databases.api import DatabaseRestApi
        from superset.models.core import Database

        DatabaseRestApi.datamodel.session = session

        Database.metadata.create_all(session.get_bind())

        database = Database(
            database_name="first-database",
            sqlalchemy_uri="gsheets://",
            encrypted_extra=json.dumps(
                {
                    "metadata_params": {},
                    "engine_params": {},
                    "metadata_cache_timeout": {},
                    "schemas_allowed_for_file_upload": [],
                }
            ),
        )
        db.session.add(database)
        db.session.commit()

        database = Database(
            database_name="second-database",
            sqlalchemy_uri="gsheets://",
            encrypted_extra=json.dumps(
                {
                    "metadata_params": {},
                    "engine_params": {},
                    "metadata_cache_timeout": {},
                    "schemas_allowed_for_file_upload": [],
                }
            ),
        )
        db.session.add(database)
        db.session.commit()

        mocker.patch("sqlalchemy.engine.URL.get_driver_name", return_value="gsheets")
        mocker.patch("superset.utils.log.DBEventLogger.log")
        mocker.patch(
            "superset.commands.database.ssh_tunnel.delete.is_feature_enabled",
            return_value=False,
        )

        def _base_filter(query):
            from superset.models.core import Database

            return query.filter(Database.database_name.startswith("second"))

        base_filter_mock = Mock(side_effect=_base_filter)

        response_databases = DatabaseDAO.find_all()
        assert response_databases
        expected_db_names = ["first-database", "second-database"]
        actual_db_names = [db.database_name for db in response_databases]
        assert actual_db_names == expected_db_names

        assert base_filter_mock.call_count == 0

        original_config = current_app.config.copy()
        original_config["EXTRA_DYNAMIC_QUERY_FILTERS"] = {"databases": base_filter_mock}

        mocker.patch("superset.views.filters.current_app.config", new=original_config)
        response_databases = DatabaseDAO.find_all()
        assert response_databases
        expected_db_names = ["second-database"]
        actual_db_names = [db.database_name for db in response_databases]
        assert actual_db_names == expected_db_names

        assert base_filter_mock.call_count == 1

def test_oauth2_happy_path(
    mocker: MockerFixture,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database, DatabaseUserOAuth2Tokens

    DatabaseRestApi.datamodel.session = session

    Database.metadata.create_all(session.get_bind())
    db.session.add(
        Database(
            database_name="my_db",
            sqlalchemy_uri="sqlite://",
            uuid=UUID("7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb"),
        )
    )
    db.session.commit()

    mocker.patch.object(
        SqliteEngineSpec,
        "get_oauth2_config",
        return_value={"id": "one", "secret": "two"},
    )
    get_oauth2_token = mocker.patch.object(SqliteEngineSpec, "get_oauth2_token")
    get_oauth2_token.return_value = {
        "access_token": "YYY",
        "expires_in": 3600,
        "refresh_token": "ZZZ",
    }

    state = {
        "user_id": 1,
        "database_id": 1,
        "tab_id": 42,
    }
    decode_oauth2_state = mocker.patch("superset.databases.api.decode_oauth2_state")
    decode_oauth2_state.return_value = state

    mocker.patch("superset.databases.api.render_template", return_value="OK")

    with freeze_time("2024-01-01T00:00:00Z"):
        response = client.get(
            "/api/v1/database/oauth2/",
            query_string={
                "state": "some%2Estate",
                "code": "XXX",
            },
        )

    assert response.status_code == 200
    decode_oauth2_state.assert_called_with("some%2Estate")
    get_oauth2_token.assert_called_with({"id": "one", "secret": "two"}, "XXX")

    token = db.session.query(DatabaseUserOAuth2Tokens).one()
    assert token.user_id == 1
    assert token.database_id == 1
    assert token.access_token == "YYY"
    assert token.access_token_expiration == datetime(2024, 1, 1, 1, 0)
    assert token.refresh_token == "ZZZ"

def test_oauth2_multiple_tokens(
    mocker: MockerFixture,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database, DatabaseUserOAuth2Tokens

    DatabaseRestApi.datamodel.session = session

    Database.metadata.create_all(session.get_bind())
    db.session.add(
        Database(
            database_name="my_db",
            sqlalchemy_uri="sqlite://",
            uuid=UUID("7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb"),
        )
    )
    db.session.commit()

    mocker.patch.object(
        SqliteEngineSpec,
        "get_oauth2_config",
        return_value={"id": "one", "secret": "two"},
    )
    get_oauth2_token = mocker.patch.object(SqliteEngineSpec, "get_oauth2_token")
    get_oauth2_token.side_effect = [
        {
            "access_token": "YYY",
            "expires_in": 3600,
            "refresh_token": "ZZZ",
        },
        {
            "access_token": "YYY2",
            "expires_in": 3600,
            "refresh_token": "ZZZ2",
        },
    ]

    state = {
        "user_id": 1,
        "database_id": 1,
        "tab_id": 42,
    }
    decode_oauth2_state = mocker.patch("superset.databases.api.decode_oauth2_state")
    decode_oauth2_state.return_value = state

    mocker.patch("superset.databases.api.render_template", return_value="OK")

    with freeze_time("2024-01-01T00:00:00Z"):
        response = client.get(
            "/api/v1/database/oauth2/",
            query_string={
                "state": "some%2Estate",
                "code": "XXX",
            },
        )

        response = client.get(
            "/api/v1/database/oauth2/",
            query_string={
                "state": "some%2Estate",
                "code": "XXX",
            },
        )

    assert response.status_code == 200
    tokens = db.session.query(DatabaseUserOAuth2Tokens).all()
    assert len(tokens) == 1
    token = tokens[0]
    assert token.access_token == "YYY2"
    assert token.refresh_token == "ZZZ2"

def test_oauth2_error(
    mocker: MockerFixture,
    session: Session,
    client: Any,
    full_api_access: None,
) -> None:
    response = client.get(
        "/api/v1/database/oauth2/",
        query_string={
            "error": "Something bad hapened",
        },
    )

    assert response.status_code == 500
    assert response.json == {
        "errors": [
            {
                "message": "Something went wrong while doing OAuth2",
                "error_type": "OAUTH2_REDIRECT_ERROR",
                "level": "error",
                "extra": {"error": "Something bad hapened"},
            }
        ]
    }

@pytest.mark.parametrize(
    "payload,upload_called_with,reader_called_with",
    [
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
            },
            (
                1,
                "table1",
                ANY,
                None,
                ANY,
            ),
            (
                {
                    "type": "csv",
                    "already_exists": "fail",
                    "delimiter": ",",
                    "file": ANY,
                    "table_name": "table1",
                },
            ),
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table2",
                "delimiter": ";",
                "already_exists": "replace",
                "column_dates": "col1,col2",
            },
            (
                1,
                "table2",
                ANY,
                None,
                ANY,
            ),
            (
                {
                    "type": "csv",
                    "already_exists": "replace",
                    "column_dates": ["col1", "col2"],
                    "delimiter": ";",
                    "file": ANY,
                    "table_name": "table2",
                },
            ),
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table2",
                "delimiter": ";",
                "already_exists": "replace",
                "columns_read": "col1,col2",
                "day_first": True,
                "rows_to_read": "1",
                "skip_blank_lines": True,
                "skip_initial_space": True,
                "skip_rows": "10",
                "null_values": "None,N/A,''",
                "column_data_types": '{"col1": "str"}',
            },
            (
                1,
                "table2",
                ANY,
                None,
                ANY,
            ),
            (
                {
                    "type": "csv",
                    "already_exists": "replace",
                    "columns_read": ["col1", "col2"],
                    "null_values": ["None", "N/A", "''"],
                    "day_first": True,
                    "rows_to_read": 1,
                    "skip_blank_lines": True,
                    "skip_initial_space": True,
                    "skip_rows": 10,
                    "delimiter": ";",
                    "file": ANY,
                    "column_data_types": {"col1": "str"},
                    "table_name": "table2",
                },
            ),
        ),
    ],
)
def test_csv_upload(
    payload: Dict[str, Any],
    upload_called_with: Tuple[int, str, Any, Dict[str, Any]],
    reader_called_with: Dict[str, Any],
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
) -> None:
    init_mock = mocker.patch.object(UploadCommand, "__init__")
    init_mock.return_value = None
    _ = mocker.patch.object(UploadCommand, "run")
    reader_mock = mocker.patch.object(CSVReader, "__init__")
    reader_mock.return_value = None
    response = client.post(
        "/api/v1/database/1/upload/",
        data=payload,
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    assert response.json == {"message": "OK"}
    init_mock.assert_called_with(*upload_called_with)
    reader_mock.assert_called_with(*reader_called_with)

@pytest.mark.parametrize(
    "payload,expected_response",
    [
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "delimiter": ",",
                "already_exists": "fail",
            },
            {"message": {"table_name": ["Missing data for required field."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "",
                "delimiter": ",",
                "already_exists": "fail",
            },
            {"message": {"table_name": ["Length must be between 1 and 10000."]}},
        ),
        (
            {
                "type": "csv",
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
            },
            {"message": {"file": ["Field may not be null."]}},
        ),
        (
            {
                "type": "csv",
                "file": "xpto",
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
            },
            {"message": {"file": ["Field may not be null."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "xpto",
            },
            {"message": {"already_exists": ["Must be one of: fail, replace, append."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
                "day_first": "test1",
            },
            {"message": {"day_first": ["Not a valid boolean."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
                "header_row": "test1",
            },
            {"message": {"header_row": ["Not a valid integer."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
                "rows_to_read": 0,
            },
            {"message": {"rows_to_read": ["Must be greater than or equal to 1."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
                "skip_blank_lines": "test1",
            },
            {"message": {"skip_blank_lines": ["Not a valid boolean."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
                "skip_initial_space": "test1",
            },
            {"message": {"skip_initial_space": ["Not a valid boolean."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
                "skip_rows": "test1",
            },
            {"message": {"skip_rows": ["Not a valid integer."]}},
        ),
        (
            {
                "type": "csv",
                "file": (create_csv_file(), "out.csv"),
                "table_name": "table1",
                "delimiter": ",",
                "already_exists": "fail",
                "column_data_types":