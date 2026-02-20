# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=unused-argument, import-outside-toplevel, line-too-long, invalid-name

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, cast
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

    # create table for databases
    Database.metadata.create_all(session.get_bind())  # pylint: disable=no-member
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
    """
    Test that we can set the database UUID when creating it.
    """
    from superset.models.core import Database

    # create table for databases
    Database.metadata.create_all(session.get_bind())  # pylint: disable=no-member

    response = client.post(
        "/api/v1/database/",
        json={
            "database_name": "my_db",
            "sqlalchemy_uri": "sqlite://",
            "uuid": "7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb",
        },
    )
    assert response.status_code == 201

    # check that response includes UUID
    payload = response.json
    assert payload["result"]["uuid"] == "7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb"

    database = session.query(Database).one()
    assert database.uuid == UUID("7c1b7880-a59d-47cd-8bf1-f1eb8d2863cb")


def test_password_mask(
    mocker: MockerFixture,
    app: Any,
    session: Session,
    client: Any,
    full_api_access: Any,
) -> None:
    """
    Test that sensitive information is masked.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database

    DatabaseRestApi.datamodel.session = session

    # create table for databases
    Database.metadata.create_all(session.get_bind())  # pylint: disable=no-member

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
                    "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",  # noqa: E501
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

    # mock the lookup so that we don't need to include the driver
    mocker.patch("sqlalchemy.engine.URL.get_driver_name", return_value="gsheets")
    mocker.patch("superset.utils.log.DBEventLogger.log")

    response = client.get("/api/v1/database/1/connection")

    # check that private key is masked
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
    """
    Test that connection info is only returned in ``api/v1/database/${id}/connection``.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database

    DatabaseRestApi.datamodel.session = session

    # create table for databases
    Database.metadata.create_all(session.get_bind())  # pylint: disable=no-member

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
                    "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",  # noqa: E501
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

    # mock the lookup so that we don't need to include the driver
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
            "extra": '{\n    "metadata_params": {},\n    "engine_params": {},\n    "metadata_cache_timeout": {},\n    "schemas_allowed_for_file_upload": []\n}\n',  # noqa: E501
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
                        "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",  # noqa: E501
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
                    "client_email": "google-spreadsheets-demo-servi@black-sanctum-314419.iam.gserviceaccount.com",  # noqa: E501
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
    """
    Test that an update with a masked password doesn't overwrite the existing password.
    """
    from superset.databases.api import DatabaseRestApi
    from superset.models.core import Database

    DatabaseRestApi.datamodel.session = session

    # create table for databases
    Database.metadata.create_all(session.get_bind())  # pylint: disable=no-member

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
        == '{"service_account_info": {"project_id": "yellow-unicorn-314419", "private_key": "SECRET"}}'  # noqa: E501
    )


def test_non_zip_import(client: Any, full_api_access: None) -> None:
    """
    Test that non-ZIP imports are not allowed.
    """
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
                            "message": "Issue 1010 - Superset encountered an error while running a command.",  # noqa: E501
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
    full_api_access: Any,
) -> None:
    """
    Test that we can delete SSH Tunnel
    """
    with app.app_context():
        from superset.daos.database import DatabaseDAO
        from superset.databases.api import DatabaseRestApi
        from superset.databases.ssh_tunnel.models import SSHTunnel
        from superset.models.core import Database

        DatabaseRestApi.datamodel.session = session

        # create table for databases
        Database.metadata.create_all(session.get_bind())  # pylint: disable=no-member

        # Create our Database
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
                       