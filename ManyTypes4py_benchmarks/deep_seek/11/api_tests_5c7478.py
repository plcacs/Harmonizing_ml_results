"""Unit tests for Superset"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union, DefaultDict
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
from superset.utils.database import get_example_database, get_main_database
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_USERNAME
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,
    load_birth_names_data,
)
from tests.integration_tests.fixtures.energy_dashboard import (
    load_energy_table_with_slice,
    load_energy_table_data,
)
from tests.integration_tests.fixtures.world_bank_dashboard import (
    load_world_bank_dashboard_with_slices,
    load_world_bank_data,
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
    load_unicode_dashboard_with_position,
    load_unicode_data,
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
        database = Database(
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
    def create_database_with_report(self) -> Database:
        with self.create_app().app_context():
            example_db = get_example_database()
            database = self.insert_database(
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
            db.session.delete(report_schedule)
            db.session.delete(database)
            db.session.commit()

    @pytest.fixture
    def create_database_with_dataset(self) -> Database:
        with self.create_app().app_context():
            example_db = get_example_database()
            self._database = self.insert_database(
                "database_with_dataset",
                example_db.sqlalchemy_uri_decrypted,
                expose_in_sqllab=True,
            )
            table = SqlaTable(
                schema="main",
                table_name="ab_permission",
                database=self._database,
            )
            db.session.add(table)
            db.session.commit()
            yield self._database
            db.session.delete(table)
            db.session.delete(self._database)
            db.session.commit()
            self._database = None

    def create_database_import(self) -> BytesIO:
        buf = BytesIO()
        with ZipFile(buf, "w") as bundle:
            with bundle.open("database_export/metadata.yaml", "w") as fp:
                fp.write(yaml.safe_dump(database_metadata_config).encode())
            with bundle.open("database_export/databases/imported_database.yaml", "w") as fp:
                fp.write(yaml.safe_dump(database_config).encode())
            with bundle.open("database_export/datasets/imported_dataset.yaml", "w") as fp:
                fp.write(yaml.safe_dump(dataset_config).encode())
        buf.seek(0)
        return buf

    def test_get_items(self) -> None:
        """
        Database API: Test get items
        """
        self.login(ADMIN_USERNAME)
        uri = "api/v1/database/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response = json.loads(rv.data.decode("utf-8"))
        expected_columns = [
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
        example_db = get_example_database()
        test_database = self.insert_database(
            "test-database", example_db.sqlalchemy_uri_decrypted, expose_in_sqllab=True
        )
        dbs = db.session.query(Database).filter_by(expose_in_sqllab=True).all()
        self.login(ADMIN_USERNAME)
        arguments = {
            "keys": ["none"],
            "filters": [{"col": "expose_in_sqllab", "opr": "eq", "value": True}],
            "order_columns": "database_name",
            "order_direction": "asc",
            "page": 0,
            "page_size": -1,
        }
        uri = f"api/v1/database/?q={prison.dumps(arguments)}"
        rv = self.client.get(uri)
        response = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200
        assert response["count"] == len(dbs)
        db.session.delete(test_database)
        db.session.commit()

    def test_get_items_not_allowed(self) -> None:
        """
        Database API: Test get items not allowed
        """
        self.login(GAMMA_USERNAME)
        uri = "api/v1/database/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        response = json.loads(rv.data.decode("utf-8"))
        assert response["count"] == 0

    def test_create_database(self) -> None:
        """
        Database API: Test create
        """
        extra = {
            "metadata_params": {},
            "engine_params": {},
            "metadata_cache_timeout": {},
            "schemas_allowed_for_file_upload": [],
        }
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == "sqlite":
            return
        database_data = {
            "database_name": "test-create-database",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "configuration_method": ConfigurationMethod.SQLALCHEMY_FORM,
            "server_cert": None,
            "extra": json.dumps(extra),
        }
        uri = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        model = db.session.query(Database).get(response.get("id"))
        assert model.configuration_method == ConfigurationMethod.SQLALCHEMY_FORM
        db.session.delete(model)
        db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run"
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_create_database_with_ssh_tunnel(
        self,
        mock_get_all_schema_names: Mock,
        mock_get_all_catalog_names: Mock,
        mock_create_is_feature_enabled: Mock,
        mock_test_connection_database_command_run: Mock,
    ) -> None:
        """
        Database API: Test create with SSH Tunnel
        """
        mock_create_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == "sqlite":
            return
        ssh_tunnel_properties = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        uri = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        model_ssh_tunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response.get("id"))
            .one()
        )
        assert response.get("result")["ssh_tunnel"]["password"] == "XXXXXXXXXX"
        assert model_ssh_tunnel.database_id == response.get("id")
        model = db.session.query(Database).get(response.get("id"))
        db.session.delete(model)
        db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run"
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_create_database_with_missing_port_raises_error(
        self,
        mock_get_all_schema_names: Mock,
        mock_get_all_catalog_names: Mock,
        mock_create_is_feature_enabled: Mock,
        mock_test_connection_database_command_run: Mock,
    ) -> None:
        """
        Database API: Test that missing port raises SSHTunnelDatabaseError
        """
        mock_create_is_feature_enabled.return_value = True
        self.login(username="admin")
        example_db = get_example_database()
        if example_db.backend == "sqlite":
            return
        modified_sqlalchemy_uri = "postgresql://foo:bar@localhost/test-db"
        ssh_tunnel_properties = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data_with_ssh_tunnel = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": modified_sqlalchemy_uri,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        uri = "api/v1/database/"
        rv = self.client.post(uri, json=database_data_with_ssh_tunnel)
        response = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 400
        assert (
            response.get("message")
            == "A database port is required when connecting via SSH Tunnel."
        )

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run"
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.commands.database.update.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_update_database_with_ssh_tunnel(
        self,
        mock_get_all_schema_names: Mock,
        mock_get_all_catalog_names: Mock,
        mock_update_is_feature_enabled: Mock,
        mock_create_is_feature_enabled: Mock,
        mock_test_connection_database_command_run: Mock,
    ) -> None:
        """
        Database API: Test update Database with SSH Tunnel
        """
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(ADMIN_USERNAME)
        example_db = get_example_database()
        if example_db.backend == "sqlite":
            return
        ssh_tunnel_properties = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
        }
        database_data_with_ssh_tunnel = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": example_db.sqlalchemy_uri_decrypted,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        uri = "api/v1/database/"
        rv = self.client.post(uri, json=database_data)
        response = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 201
        uri = "api/v1/database/{}".format(response.get("id"))
        rv = self.client.put(uri, json=database_data_with_ssh_tunnel)
        response_update = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200
        model_ssh_tunnel = (
            db.session.query(SSHTunnel)
            .filter(SSHTunnel.database_id == response_update.get("id"))
            .one()
        )
        assert model_ssh_tunnel.database_id == response_update.get("id")
        model = db.session.query(Database).get(response.get("id"))
        db.session.delete(model)
        db.session.commit()

    @mock.patch(
        "superset.commands.database.test_connection.TestConnectionDatabaseCommand.run"
    )
    @mock.patch("superset.commands.database.create.is_feature_enabled")
    @mock.patch("superset.commands.database.update.is_feature_enabled")
    @mock.patch("superset.models.core.Database.get_all_catalog_names")
    @mock.patch("superset.models.core.Database.get_all_schema_names")
    def test_update_database_with_missing_port_raises_error(
        self,
        mock_get_all_schema_names: Mock,
        mock_get_all_catalog_names: Mock,
        mock_update_is_feature_enabled: Mock,
        mock_create_is_feature_enabled: Mock,
        mock_test_connection_database_command_run: Mock,
    ) -> None:
        """
        Database API: Test that missing port raises SSHTunnelDatabaseError
        """
        mock_create_is_feature_enabled.return_value = True
        mock_update_is_feature_enabled.return_value = True
        self.login(username="admin")
        example_db = get_example_database()
        if example_db.backend == "sqlite":
            return
        modified_sqlalchemy_uri = "postgresql://foo:bar@localhost/test-db"
        ssh_tunnel_properties = {
            "server_address": "123.132.123.1",
            "server_port": 8080,
            "username": "foo",
            "password": "bar",
        }
        database_data_with_ssh_tunnel = {
            "database_name": "test-db-with-ssh-tunnel",
            "sqlalchemy_uri": modified_sqlalchemy_uri,
            "ssh_tunnel": ssh_tunnel_properties,
        }
        database_data = {
           