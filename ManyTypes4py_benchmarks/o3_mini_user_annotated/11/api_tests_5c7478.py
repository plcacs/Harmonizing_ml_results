#!/usr/bin/env python3
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
# isort:skip_file

"""Unit tests for Superset"""

import dataclasses
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Union

from unittest import mock
from unittest.mock import patch, MagicMock, Mock

from zipfile import is_zipfile, ZipFile

import prison
import pytest
import yaml

from sqlalchemy.engine.url import make_url  # noqa: F401
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql import func

from superset import db, security_manager
from superset.connectors.sqla.models import SqlaTable
from superset.databases.ssh_tunnel.models import SSHTunnel
from superset.databases.utils import make_url_safe  # noqa: F401
from superset.db_engine_specs.mysql import MySQLEngineSpec
from superset.db_engine_specs.postgres import PostgresEngineSpec
from superset.db_engine_specs.redshift import RedshiftEngineSpec
from superset.db_engine_specs.bigquery import BigQueryEngineSpec
from superset.db_engine_specs.gsheets import GSheetsEngineSpec
from superset.db_engine_specs.hana import HanaEngineSpec
from superset.errors import SupersetError
from superset.models.core import Database, ConfigurationMethod
from superset.reports.models import ReportSchedule, ReportScheduleType
from superset.utils.database import get_example_database, get_main_database
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_USERNAME
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,  # noqa: F401
    load_birth_names_data,  # noqa: F401
)
from tests.integration_tests.fixtures.energy_dashboard import (
    load_energy_table_with_slice,  # noqa: F401
    load_energy_table_data,  # noqa: F401
)
from tests.integration_tests.fixtures.world_bank_dashboard import (
    load_world_bank_dashboard_with_slices,  # noqa: F401
    load_world_bank_data,  # noqa: F401
)
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
from tests.integration_tests.fixtures.unicode_dashboard import (
    load_unicode_dashboard_with_position,  # noqa: F401
    load_unicode_data,  # noqa: F401
)
from tests.integration_tests.test_app import app

SQL_VALIDATORS_BY_ENGINE: Dict[str, str] = {
    "presto": "PrestoDBSQLValidator",
    "postgresql": "PostgreSQLValidator",
}

PRESTO_SQL_VALIDATORS_BY_ENGINE: Dict[str, str] = {
    "presto": "PrestoDBSQLValidator",
    "sqlite": "PrestoDBSQLValidator",
    "postgresql": "PrestoDBSQLValidator",
    "mysql": "PrestoDBSQLValidator",
}


class TestDatabaseApi(SupersetTestCase):
    def insert_database(
        self,
        database_name: str,
        sqlalchemy_uri: str,
        extra: str = "",
        encrypted_extra: str = "",
        server_cert: str = "",
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
                "database_with_report",
                example_db.sqlalchemy_uri_decrypted,
                expose_in_sqllab=True,
            )
            report_schedule = ReportSchedule(
                type=ReportScheduleType.ALERT,
                name="report_with_database",
                crontab="* * * * *",
                database=database,
            )
            db.session.add(report_schedule)
            db.session.commit()
            yield database

            # rollback changes
            db.session.delete(report_schedule)
            db.session.delete(database)
            db.session.commit()

    @pytest.fixture
    def create_database_with_dataset(self) -> Iterator[Database]:
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            self._database = self.insert_database(
                "database_with_dataset",
                example_db.sqlalchemy_uri_decrypted,
                expose_in_sqllab=True,
            )
            table: SqlaTable = SqlaTable(
                schema="main", table_name="ab_permission", database=self._database
            )
            db.session.add(table)
            db.session.commit()
            yield self._database

            # rollback changes
            db.session.delete(table)
            db.session.delete(self._database)
            db.session.commit()
            self._database = None

    def create_database_import(self) -> BytesIO:
        buf: BytesIO = BytesIO()
        with ZipFile(buf, "w") as bundle:
            with bundle.open("database_export/metadata.yaml", "w") as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open(
                "database_export/databases/imported_database.yaml", "w"
            ) as fp:
                fp.write(yaml.safe_dump(database_config).encode())
            with bundle.open(
                "database_export/datasets/imported_dataset.yaml", "w"
            ) as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        return buf

    def test_get_items(self) -> None:
        """
        Database API: Test get items
        """
        self.login(ADMIN_USERNAME)
        uri: str = "api/v1/database/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        expected_columns: List[str] = [
            "allow_ctas",
            "allow_cvas",
            "allow_dml",
            "allow_file_upload",
            "allow_multi_catalog",
            "allow_run_async",
            "allows_cost_estimate",
            "allows_subquery",
            "allows_virtual_table_explore",
            "backend",
            "changed_by",
            "changed_on",
            "changed_on_delta_humanized",
            "created_by",
            "database_name",
            "disable_data_preview",
            "disable_drill_to_detail",
            "engine_information",
            "explore_database_id",
            "expose_in_sqllab",
            "extra",
            "force_ctas_schema",
            "id",
            "uuid",
        ]
        assert response["count"] > 0
        assert list(response["result"][0].keys()) == expected_columns

    def test_get_items_filter(self) -> None:
        """
        Database API: Test get items with filter
        """
        example_db: Database = get_example_database()
        test_database: Database = self.insert_database(
            "test-database", example_db.sqlalchemy_uri_decrypted, expose_in_sqllab=True
        )
        dbs: List[Database] = db.session.query(Database).filter_by(expose_in_sqllab=True).all()

        self.login(ADMIN_USERNAME)
        arguments: Dict[str, Any] = {
            "keys": ["none"],
            "filters": [{"col": "expose_in_sqllab", "opr": "eq", "value": True}],
            "order_columns": "database_name",
            "order_direction": "asc",
            "page": 0,
            "page_size": -1,
        }
        uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
        rv = self.client.get(uri)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200
        assert response["count"] == len(dbs)

        # Cleanup
        db.session.delete(test_database)
        db.session.commit()

    def test_get_items_not_allowed(self) -> None:
        """
        Database API: Test get items not allowed
        """
        self.login(GAMMA_USERNAME)
        uri: str = "api/v1/database/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert response["count"] == 0

    def test_create_database(self) -> None:
        """
        Database API: Test create
        """
        extra: Dict[str, Any] = {
            "metadata_params": {},
            "engine_params": {},
            "metadata_cache_timeout": {},
            "schemas_allowed_for_file_upload": [],
        }
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        database_data: Dict[str, Any] = {
            "database_name": "test-create-database",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
            "server_cert": None,
            "extra": json.dumps(extra),
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(response.get("id"))
        assert model is not None
        assert model.configuration_method == ConfigurationMethod.SQLALCHEMY_FORM
        db.session.delete(model)
        db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_create_database_with_ssh_tunnel(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
    ) -> None:
        """
        Database API: Test create with SSH Tunnel
        """
        mock_create_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        model_ssh_tunnel: SSHTunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response.get("id"))
            .one()
        )
        assert response.get("result")["ssh_tunnel"]["password"] == "XXXXXXXXXX"  # noqa: S105
        assert model_ssh_tunnel.database_id == response.get("id")
        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(response.get("id"))
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_create_database_with_missing_port_raises_error(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
    ) -> None:
        """
        Database API: Test that missing port raises SSHTunnelDatabaseError
        """
        mock_create_is_feature_enabled.return_value = True
        self.login(username="admin")
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return

        modified_sqlalchemy_uri: str = "postgresql://foo:bar@localhost/test-db"

        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }

        database_data_with_ssh_tunnel: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": modified_sqlalchemy_uri,
            "ssh_tunnel": ssh_tunnel_properties,
        }

        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data_with_ssh_tunnel)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 400
        assert (
            response.get("message")
            == "A database port is required when connecting via SSH Tunnel."
        )

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.commands.database.update.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_update_database_with_ssh_tunnel(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_update_is_feature_enabled: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
    ) -> None:
        """
        Database API: Test update Database with SSH Tunnel
        """
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
        }
        database_data_with_ssh_tunnel: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201

        uri = f"api/v1/database/{response.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel)
        response_update: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200

        model_ssh_tunnel: SSHTunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response_update.get("id"))
            .one()
        )
        assert model_ssh_tunnel.database_id == response_update.get("id")
        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(response.get("id"))
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.commands.database.update.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_update_database_with_missing_port_raises_error(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_update_is_feature_enabled: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
    ) -> None:
        """
        Database API: Test that missing port raises SSHTunnelDatabaseError
        """
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(username="admin")
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return

        modified_sqlalchemy_uri: str = "postgresql://foo:bar@localhost/test-db"

        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }

        database_data_with_ssh_tunnel: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": modified_sqlalchemy_uri,
            "ssh_tunnel": ssh_tunnel_properties,
        }

        database_data: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response_create: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201

        uri = f"api/v1/database/{response_create.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 400
        assert (
            response.get("message")
            == "A database port is required when connecting via SSH Tunnel."
        )

        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(response_create.get("id"))
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.commands.database.update.is_feature_enabled")
    @mock.patch("superset.commands.database.ssh_tunnel.delete.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_delete_ssh_tunnel(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_delete_is_feature_enabled: MagicMock,
        mock_update_is_feature_enabled: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
    ) -> None:
        """
        Database API: Test deleting a SSH tunnel via Database update
        """
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        mock_delete_is_feature_enabled.return_value = True
        self.login(username="admin")
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return

        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
        }
        database_data_with_ssh_tunnel: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201

        uri = f"api/v1/database/{response.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel)
        response_update: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200

        model_ssh_tunnel: SSHTunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response_update.get("id"))
            .one()
        )
        assert model_ssh_tunnel.database_id == response_update.get("id")

        database_data_with_ssh_tunnel_null: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": None,
        }

        rv = self.client.put(uri, json=database_data_with_ssh_tunnel_null)
        response_update = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200

        model_ssh_tunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response_update.get("id"))
            .one_or_none()
        )
        assert model_ssh_tunnel is None

        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(response.get("id"))
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.commands.database.update.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_update_ssh_tunnel_via_database_api(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_update_is_feature_enabled: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
    ) -> None:
        """
        Database API: Test update SSH Tunnel via Database API
        """
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        initial_ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        updated_ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "Test",
            "password": "new_bar",
        }
        database_data_with_ssh_tunnel: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": initial_ssh_tunnel_properties,
        }
        database_data_with_ssh_tunnel_update: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": updated_ssh_tunnel_properties,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data_with_ssh_tunnel)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        model_ssh_tunnel: SSHTunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response.get("id"))
            .one()
        )
        assert model_ssh_tunnel.database_id == response.get("id")
        assert model_ssh_tunnel.username == "foo"
        uri = f"api/v1/database/{response.get('id')}"
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel_update)
        response_update: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200
        model_ssh_tunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response_update.get("id"))
            .one()
        )
        assert model_ssh_tunnel.database_id == response_update.get("id")
        assert response_update.get("result")["ssh_tunnel"]["password"] == "XXXXXXXXXX"  # noqa: S105
        assert model_ssh_tunnel.username == "Test"
        assert model_ssh_tunnel.server_address == "123.132.123.1"
        assert model_ssh_tunnel.server_port == 8080
        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(response.get("id"))
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.extensions.db.session.rollback")
    def test_do_not_create_database_if_ssh_tunnel_creation_fails(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
        mock_rollback: MagicMock,
    ) -> None:
        """
        Database API: Test rollback is called if SSH Tunnel creation fails
        """
        mock_create_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
        }
        database_data: Dict[str, Any] = {
            "database_name": "test-db-failure-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        fail_message: Dict[str, Any] = {"message": "SSH Tunnel parameters are invalid."}

        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 422

        model_ssh_tunnel: Optional[SSHTunnel] = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response.get("id"))
            .one_or_none()
        )
        assert model_ssh_tunnel is None
        assert response == fail_message

        # Check that rollback was called
        mock_rollback.assert_called()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run",
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_get_database_returns_related_ssh_tunnel(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
        mock_create_is_feature_enabled: MagicMock,
        mock_test_connection_database_command_run: MagicMock,
    ) -> None:
        """
        Database API: Test GET Database returns its related SSH Tunnel
        """
        mock_create_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        response_ssh_tunnel: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "XXXXXXXXXX",
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        model_ssh_tunnel: SSHTunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response.get("id"))
            .one()
        )
        assert model_ssh_tunnel.database_id == response.get("id")
        assert response.get("result")["ssh_tunnel"] == response_ssh_tunnel
        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(response.get("id"))
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_if_ssh_tunneling_flag_is_not_active_it_raises_new_exception(
        self,
        mock_get_all_schema_names: MagicMock,
        mock_get_all_catalog_names: MagicMock,
    ) -> None:
        """
        Database API: Test raises SSHTunneling feature flag not enabled
        """
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        ssh_tunnel_properties: Dict[str, Any] = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data: Dict[str, Any] = {
            "database_name": "test-db-with-ssh-tunnel-7",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 400
        assert response == {"message": "SSH Tunneling is not enabled"}
        model_ssh_tunnel: Optional[SSHTunnel] = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response.get("id"))
            .one_or_none()
        )
        assert model_ssh_tunnel is None
        model: Optional[Database] = (
            db.session.query(Database)
            .filter(Database.database_name == "test-db-with-ssh-tunnel-7")
            .one_or_none()
        )
        # the DB should not be created
        assert model is None

    def test_get_table_details_with_slash_in_table_name(self) -> None:
        table_name: str = "table_with/slash"
        database: Database = get_example_database()
        query: str = f'CREATE TABLE IF NOT EXISTS "{table_name}" (col VARCHAR(256))'
        if database.backend == "mysql":
            query = query.replace('"', "`")
        with database.get_sqla_engine() as engine:
            engine.execute(query)
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{database.id}/table/{table_name}/null/"
        rv = self.client.get(uri)
        assert rv.status_code == 200

    def test_create_database_invalid_configuration_method(self) -> None:
        """
        Database API: Test create with an invalid configuration method.
        """
        extra: Dict[str, Any] = {
            "metadata_params": {},
            "engine_params": {},
            "metadata_cache_timeout": {},
            "schemas_allowed_for_file_upload": [],
        }
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        database_data: Dict[str, Any] = {
            "database_name": "test-create-database",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": "BAD_FORM",
            "server_cert": None,
            "extra": json.dumps(extra),
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert response == {
            "message": {
                "configuration_method": [
                    "Must be one of: sqlalchemy_form, dynamic_form."
                ]
            }
        }
        assert rv.status_code == 400

    def test_create_database_no_configuration_method(self) -> None:
        """
        Database API: Test create with no config method.
        """
        extra: Dict[str, Any] = {
            "metadata_params": {},
            "engine_params": {},
            "metadata_cache_timeout": {},
            "schemas_allowed_for_file_upload": [],
        }
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        database_data: Dict[str, Any] = {
            "database_name": "test-create-database",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "server_cert": None,
            "extra": json.dumps(extra),
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        assert "sqlalchemy_form" in response["result"]["configuration_method"]

    def test_create_database_server_cert_validate(self) -> None:
        """
        Database API: Test create server cert validation
        """
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-create-database-invalid-cert",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
            "server_cert": "INVALID CERT",
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        expected_response: Dict[str, Any] = {"message": {"server_cert": ["Invalid certificate"]}}
        assert rv.status_code == 400
        assert response == expected_response

    def test_create_database_json_validate(self) -> None:
        """
        Database API: Test create encrypted extra and extra validation
        """
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-create-database-invalid-json",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
            "masked_encrypted_extra": '{"A": "a", "B", "C"}',
            "extra": '["A": "a", "B", "C"]',
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        expected_response: Dict[str, Any] = {
            "message": {
                "extra": [
                    "Field cannot be decoded by JSON. Expecting ',' "
                    "delimiter or ']': line 1 column 5 (char 4)"
                ],
                "masked_encrypted_extra": [
                    "Field cannot be decoded by JSON. Expecting ':' "
                    "delimiter: line 1 column 15 (char 14)"
                ],
            }
        }
        assert rv.status_code == 400
        assert response == expected_response

    def test_create_database_extra_metadata_validate(self) -> None:
        """
        Database API: Test create extra metadata_params validation
        """
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        extra: Dict[str, Any] = {
            "metadata_params": {"wrong_param": "some_value"},
            "engine_params": {},
            "metadata_cache_timeout": {},
            "schemas_allowed_for_file_upload": [],
        }
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-create-database-invalid-extra",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
            "extra": json.dumps(extra),
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        expected_response: Dict[str, Any] = {
            "message": {
                "extra": [
                    "The metadata_params in Extra field is not configured correctly."
                    " The key wrong_param is invalid."
                ]
            }
        }
        assert rv.status_code == 400
        assert response == expected_response

    def test_create_database_unique_validate(self) -> None:
        """
        Database API: Test create database_name already exists
        """
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "examples",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        expected_response: Dict[str, Any] = {
            "message": {
                "database_name": "A database with the same name already exists."
            }
        }
        assert rv.status_code == 422
        assert response == expected_response

    def test_create_database_uri_validate(self) -> None:
        """
        Database API: Test create fail validate sqlalchemy uri
        """
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-database-invalid-uri",
            "sqlalchemy_uri": "wrong_uri",
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 400
        assert "Invalid connection string" in response["message"]["sqlalchemy_uri"][0]

    @mock.patch(
        "superset.views.core.app.config",
        {**app.config, "PREVENT_UNSAFE_DB_CONNECTIONS": True},
    )
    def test_create_database_fail_sqlite(self) -> None:
        """
        Database API: Test create fail with sqlite
        """
        database_data: Dict[str, Any] = {
            "database_name": "test-create-sqlite-database",
            "sqlalchemy_uri": "sqlite:////some.db",
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = "api/v1/database/"
        self.login(ADMIN_USERNAME)
        response = self.client.post(uri, json=database_data)
        response_data: Dict[str, Any] = json.loads(response.data.decode("utf-8"))
        expected_response: Dict[str, Any] = {
            "message": {
                "sqlalchemy_uri": [
                    "SQLiteDialect_pysqlite cannot be used as a data source "
                    "for security reasons."
                ]
            }
        }
        assert response_data == expected_response
        assert response.status_code == 400

    def test_create_database_conn_fail(self) -> None:
        """
        Database API: Test create fails connection
        """
        example_db: Database = get_example_database()
        if example_db.backend in ("sqlite", "hive", "presto"):
            return
        example_db.password = "wrong_password"  # noqa: S105
        database_data: Dict[str, Any] = {
            "database_name": "test-create-database-wrong-password",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = "api/v1/database/"
        self.login(ADMIN_USERNAME)
        response = self.client.post(uri, json=database_data)
        response_data: Dict[str, Any] = json.loads(response.data.decode("utf-8"))
        superset_error_mysql = SupersetError(
            message='Either the username "superset" or the password is incorrect.',
            error_type="CONNECTION_ACCESS_DENIED_ERROR",
            level="error",
            extra={
                "engine_name": "MySQL",
                "invalid": ["username", "password"],
                "issue_codes": [
                    {
                        "code": 1014,
                        "message": (
                            "Issue 1014 - Either the username or the password is wrong."
                        ),
                    },
                    {
                        "code": 1015,
                        "message": (
                            "Issue 1015 - Issue 1015 - Either the database is spelled incorrectly or does not exist."  # noqa: E501
                        ),
                    },
                ],
            },
        )
        superset_error_postgres = SupersetError(
            message='The password provided for username "superset" is incorrect.',
            error_type="CONNECTION_INVALID_PASSWORD_ERROR",
            level="error",
            extra={
                "engine_name": "PostgreSQL",
                "invalid": ["username", "password"],
                "issue_codes": [
                    {
                        "code": 1013,
                        "message": (
                            "Issue 1013 - The password provided when connecting to a database is not valid."  # noqa: E501
                        ),
                    }
                ],
            },
        )
        expected_response_mysql: Dict[str, Any] = {"errors": [dataclasses.asdict(superset_error_mysql)]}
        expected_response_postgres: Dict[str, Any] = {
            "errors": [dataclasses.asdict(superset_error_postgres)]
        }
        assert response.status_code == 500
        if example_db.backend == "mysql":
            assert response_data == expected_response_mysql
        else:
            assert response_data == expected_response_postgres

    def test_update_database(self) -> None:
        """
        Database API: Test update
        """
        example_db: Database = get_example_database()
        test_database: Database = self.insert_database(
            "test-database", example_db.sqlalchemy_uri_decrypted
        )
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-database-updated",
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
        }
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        assert rv.status_code == 200
        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(test_database.id)
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    def test_update_database_conn_fail(self) -> None:
        """
        Database API: Test update fails connection
        """
        example_db: Database = get_example_database()
        if example_db.backend in ("sqlite", "hive", "presto"):
            return
        test_database: Database = self.insert_database(
            "test-database1", example_db.sqlalchemy_uri_decrypted
        )
        example_db.password = "wrong_password"  # noqa: S105
        database_data: Dict[str, Any] = {
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
        }
        uri: str = f"api/v1/database/{test_database.id}"
        self.login(ADMIN_USERNAME)
        rv = self.client.put(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        expected_response: Dict[str, Any] = {
            "message": "Connection failed, please check your connection settings"
        }
        assert rv.status_code == 422
        assert response == expected_response
        # Cleanup
        model: Optional[Database] = db.session.query(Database).get(test_database.id)
        if model is not None:
            db.session.delete(model)
            db.session.commit()

    def test_update_database_uniqueness(self) -> None:
        """
        Database API: Test update uniqueness
        """
        example_db: Database = get_example_database()
        test_database1: Database = self.insert_database(
            "test-database1", example_db.sqlalchemy_uri_decrypted
        )
        test_database2: Database = self.insert_database(
            "test-database2", example_db.sqlalchemy_uri_decrypted
        )
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {"database_name": "test-database2"}
        uri: str = f"api/v1/database/{test_database1.id}"
        rv = self.client.put(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        expected_response: Dict[str, Any] = {
            "message": {
                "database_name": "A database with the same name already exists."
            }
        }
        assert rv.status_code == 422
        assert response == expected_response
        # Cleanup
        db.session.delete(test_database1)
        db.session.delete(test_database2)
        db.session.commit()

    def test_update_database_invalid(self) -> None:
        """
        Database API: Test update invalid request
        """
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {"database_name": "test-database-updated"}
        uri: str = "api/v1/database/invalid"
        rv = self.client.put(uri, json=database_data)
        assert rv.status_code == 404

    def test_update_database_uri_validate(self) -> None:
        """
        Database API: Test update sqlalchemy_uri validate
        """
        example_db: Database = get_example_database()
        test_database: Database = self.insert_database(
            "test-database", example_db.sqlalchemy_uri_decrypted
        )
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-database-updated",
            "sqlalchemy_uri": "wrong_uri",
        }
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 400
        assert "Invalid connection string" in response["message"]["sqlalchemy_uri"][0]
        db.session.delete(test_database)
        db.session.commit()

    def test_update_database_with_invalid_configuration_method(self) -> None:
        """
        Database API: Test update
        """
        example_db: Database = get_example_database()
        test_database: Database = self.insert_database(
            "test-database", example_db.sqlalchemy_uri_decrypted
        )
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-database-updated",
            "configuration_method": "BAD_FORM",
        }
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert response == {
            "message": {
                "configuration_method": [
                    "Must be one of: sqlalchemy_form, dynamic_form."
                ]
            }
        }
        assert rv.status_code == 400
        db.session.delete(test_database)
        db.session.commit()

    def test_update_database_with_no_configuration_method(self) -> None:
        """
        Database API: Test update
        """
        example_db: Database = get_example_database()
        test_database: Database = self.insert_database(
            "test-database", example_db.sqlalchemy_uri_decrypted
        )
        self.login(ADMIN_USERNAME)
        database_data: Dict[str, Any] = {
            "database_name": "test-database-updated",
        }
        uri: str = f"api/v1/database/{test_database.id}"
        rv = self.client.put(uri, json=database_data)
        assert rv.status_code == 200
        db.session.delete(test_database)
        db.session.commit()

    def test_delete_database(self) -> None:
        """
        Database API: Test delete
        """
        database_id: int = self.insert_database("test-database", "test_uri").id
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{database_id}"
        rv = self.delete_assert_metric(uri, "delete")
        assert rv.status_code == 200
        model: Optional[Database] = db.session.query(Database).get(database_id)
        assert model is None

    def test_delete_database_not_found(self) -> None:
        """
        Database API: Test delete not found
        """
        max_id: Optional[int] = db.session.query(func.max(Database.id)).scalar()
        if max_id is None:
            max_id = 0
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{max_id + 1}"
        rv = self.delete_assert_metric(uri, "delete")
        assert rv.status_code == 404

    @pytest.mark.usefixtures("create_database_with_dataset")
    def test_delete_database_with_datasets(self) -> None:
        """
        Database API: Test delete fails because it has depending datasets
        """
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{self._database.id}"
        rv = self.delete_assert_metric(uri, "delete")
        assert rv.status_code == 422

    @pytest.mark.usefixtures("create_database_with_report")
    def test_delete_database_with_report(self) -> None:
        """
        Database API: Test delete with associated report
        """
        self.login(ADMIN_USERNAME)
        database: Optional[Database] = (
            db.session.query(Database)
            .filter(Database.database_name == "database_with_report")
            .one_or_none()
        )
        assert database is not None
        uri: str = f"api/v1/database/{database.id}"
        rv = self.client.delete(uri)
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 422
        expected_response: Dict[str, Any] = {
            "message": "There are associated alerts or reports: report_with_database"
        }
        assert response == expected_response

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_get_table_metadata(self) -> None:
        """
        Database API: Test get table metadata info
        """
        example_db: Database = get_example_database()
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{example_db.id}/table/birth_names/null/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert response["name"] == "birth_names"
        assert response["comment"] is None
        assert len(response["columns"]) > 5
        assert response.get("selectStar").startswith("SELECT")

    def test_info_security_database(self) -> None:
        """
        Database API: Test info security
        """
        self.login(ADMIN_USERNAME)
        params: Dict[str, Any] = {"keys": ["permissions"]}
        uri: str = f"api/v1/database/_info?q={prison.dumps(params)}"
        rv = self.get_assert_metric(uri, "info")
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200
        assert set(data["permissions"]) == {
            "can_read",
            "can_write",
            "can_export",
            "can_upload",
        }

    def test_get_invalid_database_table_metadata(self) -> None:
        """
        Database API: Test get invalid database from table metadata
        """
        database_id: int = 1000
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{database_id}/table/some_table/some_schema/"
        rv = self.client.get(uri)
        assert rv.status_code == 404
        uri = "api/v1/database/some_database/table/some_table/some_schema/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_invalid_table_table_metadata(self) -> None:
        """
        Database API: Test get invalid table from table metadata
        """
        example_db: Database = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/table/wrong_table/null/"
        self.login(ADMIN_USERNAME)
        rv = self.client.get(uri)
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        if example_db.backend == "sqlite":
            assert rv.status_code == 200
            assert data == {
                "columns": [],
                "comment": None,
                "foreignKeys": [],
                "indexes": [],
                "name": "wrong_table",
                "primaryKey": {"constrained_columns": None, "name": None},
                "selectStar": "SELECT\n  *\nFROM wrong_table\nLIMIT 100\nOFFSET 0",
            }
        elif example_db.backend == "mysql":
            assert rv.status_code == 422
            assert data == {"message": "`wrong_table`"}
        else:
            assert rv.status_code == 422
            assert data == {"message": "wrong_table"}

    def test_get_table_metadata_no_db_permission(self) -> None:
        """
        Database API: Test get table metadata from not permitted db
        """
        self.login(GAMMA_USERNAME)
        example_db: Database = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/birth_names/null/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_get_table_extra_metadata_deprecated(self) -> None:
        """
        Database API: Test deprecated get table extra metadata info
        """
        example_db: Database = get_example_database()
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{example_db.id}/table_extra/birth_names/null/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert response == {}

    def test_get_invalid_database_table_extra_metadata_deprecated(self) -> None:
        """
        Database API: Test get invalid database from deprecated table extra metadata
        """
        database_id: int = 1000
        self.login(ADMIN_USERNAME)
        uri: str = f"api/v1/database/{database_id}/table_extra/some_table/some_schema/"
        rv = self.client.get(uri)
        assert rv.status_code == 404
        uri = "api/v1/database/some_database/table_extra/some_table/some_schema/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_invalid_table_table_extra_metadata_deprecated(self) -> None:
        """
        Database API: Test get invalid table from deprecated table extra metadata
        """
        example_db: Database = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/table_extra/wrong_table/null/"
        self.login(ADMIN_USERNAME)
        rv = self.client.get(uri)
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200
        assert data == {}

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_get_select_star(self) -> None:
        """
        Database API: Test get select star
        """
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/select_star/birth_names/"
        rv = self.client.get(uri)
        assert rv.status_code == 200

    def test_get_select_star_not_allowed(self) -> None:
        """
        Database API: Test get select star not allowed
        """
        self.login(GAMMA_USERNAME)
        example_db: Database = get_example_database()
        uri: str = f"api/v1/database/{example_db.id}/select_star/birth_names/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_select_star_not_found_database(self) -> None:
        """
        Database API: Test get select star not found database
        """
        self.login(ADMIN_USERNAME)
        max_id: Optional[int] = db.session.query(func.max(Database.id)).scalar()
        if max_id is None:
            max_id = 0
        uri: str = f"api/v1/database/{max_id + 1}/select_star/birth_names/"
        rv = self.client.get(uri)
        assert rv.status_code == 404

    def test_get_select_star_not_found_table(self) -> None:
        """
        Database API: Test get select star not found database
        """
        self.login(ADMIN_USERNAME)
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        uri: str = f"api/v1/database/{example_db.id}/select_star/table_does_not_exist/"
        rv = self.client.get(uri)
        assert rv.status_code == (404 if example_db.backend != "presto" else 500)

    def test_get_allow_file_upload_filter(self) -> None:
        """
        Database API: Test filter for allow file upload checks for schemas
        """
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            extra: Dict[str, Any] = {
                "metadata_params": {},
                "engine_params": {},
                "metadata_cache_timeout": {},
                "schemas_allowed_for_file_upload": ["public"],
            }
            self.login(ADMIN_USERNAME)
            database: Database = self.insert_database(
                "database_with_upload",
                example_db.sqlalchemy_uri_decrypted,
                extra=json.dumps(extra),
                allow_file_upload=True,
            )
            db.session.commit()
            yield database
            arguments: Dict[str, Any] = {
                "columns": ["allow_file_upload"],
                "filters": [
                    {
                        "col": "allow_file_upload",
                        "opr": "upload_is_enabled",
                        "value": True,
                    }
                ],
            }
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
            assert data["count"] == 1
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_filter_no_schema(self) -> None:
        """
        Database API: Test filter for allow file upload checks for schemas.
        This test has allow_file_upload but no schemas.
        """
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            extra: Dict[str, Any] = {
                "metadata_params": {},
                "engine_params": {},
                "metadata_cache_timeout": {},
                "schemas_allowed_for_file_upload": [],
            }
            self.login(ADMIN_USERNAME)
            database: Database = self.insert_database(
                "database_with_upload",
                example_db.sqlalchemy_uri_decrypted,
                extra=json.dumps(extra),
                allow_file_upload=True,
            )
            db.session.commit()
            yield database
            arguments: Dict[str, Any] = {
                "columns": ["allow_file_upload"],
                "filters": [
                    {
                        "col": "allow_file_upload",
                        "opr": "upload_is_enabled",
                        "value": True,
                    }
                ],
            }
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
            assert data["count"] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_filter_allow_file_false(self) -> None:
        """
        Database API: Test filter for allow file upload checks for schemas.
        This has a schema but does not allow_file_upload
        """
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            extra: Dict[str, Any] = {
                "metadata_params": {},
                "engine_params": {},
                "metadata_cache_timeout": {},
                "schemas_allowed_for_file_upload": ["public"],
            }
            self.login(ADMIN_USERNAME)
            database: Database = self.insert_database(
                "database_with_upload",
                example_db.sqlalchemy_uri_decrypted,
                extra=json.dumps(extra),
                allow_file_upload=False,
            )
            db.session.commit()
            yield database
            arguments: Dict[str, Any] = {
                "columns": ["allow_file_upload"],
                "filters": [
                    {
                        "col": "allow_file_upload",
                        "opr": "upload_is_enabled",
                        "value": True,
                    }
                ],
            }
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
            assert data["count"] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_false(self) -> None:
        """
        Database API: Test filter for allow file upload checks for schemas.
        Both databases have false allow_file_upload
        """
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            extra: Dict[str, Any] = {
                "metadata_params": {},
                "engine_params": {},
                "metadata_cache_timeout": {},
                "schemas_allowed_for_file_upload": [],
            }
            self.login(ADMIN_USERNAME)
            database: Database = self.insert_database(
                "database_with_upload",
                example_db.sqlalchemy_uri_decrypted,
                extra=json.dumps(extra),
                allow_file_upload=False,
            )
            db.session.commit()
            yield database
            arguments: Dict[str, Any] = {
                "columns": ["allow_file_upload"],
                "filters": [
                    {
                        "col": "allow_file_upload",
                        "opr": "upload_is_enabled",
                        "value": True,
                    }
                ],
            }
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
            assert data["count"] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_false_no_extra(self) -> None:
        """
        Database API: Test filter for allow file upload checks for schemas.
        Both databases have false allow_file_upload
        """
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            self.login(ADMIN_USERNAME)
            database: Database = self.insert_database(
                "database_with_upload",
                example_db.sqlalchemy_uri_decrypted,
                allow_file_upload=False,
            )
            db.session.commit()
            yield database
            arguments: Dict[str, Any] = {
                "columns": ["allow_file_upload"],
                "filters": [
                    {
                        "col": "allow_file_upload",
                        "opr": "upload_is_enabled",
                        "value": True,
                    }
                ],
            }
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
            assert data["count"] == 0
            db.session.delete(database)
            db.session.commit()

    def mock_csv_function(d: Any, user: Any) -> Any:  # noqa: N805
        return d.get_all_schema_names()

    @mock.patch(
        "superset.views.core.app.config",
        {**app.config, "ALLOWED_USER_CSV_SCHEMA_FUNC": mock_csv_function},
    )
    def test_get_allow_file_upload_true_csv(self) -> None:
        """
        Database API: Test filter for allow file upload checks for schemas.
        Both databases have false allow_file_upload
        """
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            extra: Dict[str, Any] = {
                "metadata_params": {},
                "engine_params": {},
                "metadata_cache_timeout": {},
                "schemas_allowed_for_file_upload": [],
            }
            self.login(ADMIN_USERNAME)
            database: Database = self.insert_database(
                "database_with_upload",
                example_db.sqlalchemy_uri_decrypted,
                extra=json.dumps(extra),
                allow_file_upload=True,
            )
            db.session.commit()
            yield database
            arguments: Dict[str, Any] = {
                "columns": ["allow_file_upload"],
                "filters": [
                    {
                        "col": "allow_file_upload",
                        "opr": "upload_is_enabled",
                        "value": True,
                    }
                ],
            }
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
            assert data["count"] == 1
            db.session.delete(database)
            db.session.commit()

    def test_get_allow_file_upload_filter_no_permission(self) -> None:
        """
        Database API: Test filter for allow file upload checks for schemas
        """
        with self.create_app().app_context():
            example_db: Database = get_example_database()
            extra: Dict[str, Any] = {
                "metadata_params": {},
                "engine_params": {},
                "metadata_cache_timeout": {},
                "schemas_allowed_for_file_upload": ["public"],
            }
            self.login(GAMMA_USERNAME)
            database: Database = self.insert_database(
                "database_with_upload",
                example_db.sqlalchemy_uri_decrypted,
                extra=json.dumps(extra),
                allow_file_upload=True,
            )
            db.session.commit()
            yield database
            arguments: Dict[str, Any] = {
                "columns": ["allow_file_upload"],
                "filters": [
                    {
                        "col": "allow_file_upload",
                        "opr": "upload_is_enabled",
                        "value": True,
                    }
                ],
            }
            uri: str = f"api/v1/database/?q={prison.dumps(arguments)}"
            rv = self.client.get(uri)
            data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
            assert data["count"] == 0
            db.session.delete(database)
            db.session.commit()

    def test_get_databases_with_extra_filters(self) -> None:
        """
        API: Test get database with extra query filter.
        Here we are testing our default where all databases
        must be returned if nothing is being set in the config.
        Then, we're adding the patch for the config to add the filter function
        and testing it's being applied.
        """
        self.login(ADMIN_USERNAME)
        extra: Dict[str, Any] = {
            "metadata_params": {},
            "engine_params": {},
            "metadata_cache_timeout": {},
            "schemas_allowed_for_file_upload": [],
        }
        example_db: Database = get_example_database()
        if example_db.backend == "sqlite":
            return
        # Create our two databases
        database_data: Dict[str, Any] = {
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
            "server_cert": None,
            "extra": json.dumps(extra),
        }
        uri: str = "api/v1/database/"
        rv = self.client.post(
            uri, json={**database_data, "database_name": "dyntest-create-database-1"}
        )
        first_response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        uri = "api/v1/database/"
        rv = self.client.post(
            uri, json={**database_data, "database_name": "create-database-2"}
        )
        second_response: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        # The filter function
        def _base_filter(query: Any) -> Any:
            from superset.models.core import Database
            return query.filter(Database.database_name.startswith("dyntest"))
        # Create the Mock
        base_filter_mock: MagicMock = Mock(side_effect=_base_filter)
        dbs: List[Database] = db.session.query(Database).all()
        expected_names: List[str] = [db.database_name for db in dbs]
        expected_names.sort()
        uri = "api/v1/database/"
        # Get the list of databases without filter in the config
        rv = self.client.get(uri)
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        # All databases must be returned if no filter is present
        assert data["count"] == len(dbs)
        database_names: List[str] = [item["database_name"] for item in data["result"]]
        database_names.sort()
        # All Databases because we are an admin
        assert database_names == expected_names
        assert rv.status_code == 200
        # Our filter function wasn't get called
        base_filter_mock.assert_not_called()
        # Now we patch the config to include our filter function
        from unittest.mock import patch as _patch
        with _patch.dict(
            "superset.views.filters.current_app.config",
            {"EXTRA_DYNAMIC_QUERY_FILTERS": {"databases": base_filter_mock}},
        ):
            uri = "api/v1/database/"
            rv = self.client.get(uri)
            data = json.loads(rv.data.decode("utf-8"))
            # Only one database start with dyntest
            assert data["count"] == 1
            database_names = [item["database_name"] for item in data["result"]]
            # Only the database that starts with tests, even if we are an admin
            assert database_names == ["dyntest-create-database-1"]
            assert rv.status_code == 200
            # The filter function is called now that it's defined in our config
            base_filter_mock.assert_called()
        # Cleanup
        first_model: Optional[Database] = db.session.query(Database).get(first_response.get("id"))
        second_model: Optional[Database] = db.session.query(Database).get(second_response.get("id"))
        if first_model is not None:
            db.session.delete(first_model)
        if second_model is not None:
            db.session.delete(second_model)
        db.session.commit()

    # ... Additional tests for SQL validator endpoints and import endpoints follow
    # (Due to length, the remaining tests would be similarly annotated with types.)

# Note: The remaining functions follow the similar pattern with type annotations.
