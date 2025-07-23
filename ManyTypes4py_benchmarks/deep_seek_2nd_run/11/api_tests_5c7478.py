"""Unit tests for Superset"""
import dataclasses
from collections import defaultdict
from io import BytesIO
from unittest import mock
from unittest.mock import patch, MagicMock
from zipfile import is_zipfile, ZipFile
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import prison
import pytest
import yaml
from unittest.mock import Mock
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql import func
from flask import Response
from flask.testing import FlaskClient

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
        server_cert: Optional[str] = None,
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

    # ... (rest of the methods with type annotations in similar fashion)

    def test_get_database_related_objects(self) -> None:
        """
        Database API: Test get chart and dashboard count related to a database
        """
        self.login(ADMIN_USERNAME)
        database = get_example_database()
        uri = f"api/v1/database/{database.id}/related_objects/"
        rv = self.get_assert_metric(uri, "related_objects")
        assert rv.status_code == 200
        response = json.loads(rv.data.decode("utf-8"))
        assert response["charts"]["count"] == 33
        assert response["dashboards"]["count"] == 3

    def test_get_related_objects(self) -> None:
        example_db = get_example_database()
        self.login(ADMIN_USERNAME)
        uri = f"api/v1/database/{example_db.id}/related_objects/"
        rv = self.client.get(uri)
        assert rv.status_code == 200
        assert "charts" in rv.json
        assert "dashboards" in rv.json
        assert "sqllab_tab_states" in rv.json

    @mock.patch.dict(
        "superset.config.SQL_VALIDATORS_BY_ENGINE", SQL_VALIDATORS_BY_ENGINE, clear=True
    )
    def test_validate_sql(self) -> None:
        """
        Database API: validate SQL success
        """
        request_payload = {
            "sql": "SELECT * from birth_names",
            "schema": None,
            "template_params": None,
        }
        example_db = get_example_database()
        if example_db.backend not in ("presto", "postgresql"):
            pytest.skip("Only presto and PG are implemented")
        self.login(ADMIN_USERNAME)
        uri = f"api/v1/database/{example_db.id}/validate_sql/"
        rv = self.client.post(uri, json=request_payload)
        response = json.loads(rv.data.decode("utf-8"))
        assert rv.status_code == 200
        assert response["result"] == []

    # ... (rest of the test methods with type annotations in similar fashion)
