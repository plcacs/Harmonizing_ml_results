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

from datetime import datetime
from io import BytesIO
from typing import Optional, List, Dict, Any, Union, Generator, cast
from unittest.mock import patch, Mock
from zipfile import is_zipfile, ZipFile

import yaml
import pytest
import prison
from freezegun import freeze_time
from sqlalchemy.sql import func, and_
from flask import Response
from flask.testing import FlaskClient

from superset import db
from superset.models.core import Database
from superset.models.core import FavStar
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
from tests.integration_tests.fixtures.tags import (
    create_custom_tags,  # noqa: F401
    get_filter_params,
)


SAVED_QUERIES_FIXTURE_COUNT: int = 10


class TestSavedQueryApi(SupersetTestCase):
    def insert_saved_query(
        self,
        label: str,
        sql: str,
        db_id: Optional[int] = None,
        created_by: Optional[Any] = None,
        schema: Optional[str] = "",
        description: Optional[str] = "",
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
            description=description,
        )
        db.session.add(query)
        db.session.commit()
        return query

    def insert_default_saved_query(
        self, label: str = "saved1", schema: str = "schema1", username: str = "admin"
    ) -> SavedQuery:
        admin: Any = self.get_user(username)
        example_db: Database = get_example_database()
        return self.insert_saved_query(
            label,
            "SELECT col1, col2 from table1",
            db_id=example_db.id,
            created_by=admin,
            schema=schema,
            description="cool description",
        )

    @pytest.fixture
    def create_saved_queries(self) -> Generator[List[SavedQuery], None, None]:
        with self.create_app().app_context():
            saved_queries: List[SavedQuery] = []
            admin: Any = self.get_user("admin")
            for cx in range(SAVED_QUERIES_FIXTURE_COUNT - 1):
                saved_queries.append(
                    self.insert_default_saved_query(
                        label=f"label{cx}", schema=f"schema{cx}"
                    )
                )
            saved_queries.append(
                self.insert_default_saved_query(
                    label=f"label{SAVED_QUERIES_FIXTURE_COUNT}",
                    schema=f"schema{SAVED_QUERIES_FIXTURE_COUNT}",
                    username="gamma_sqllab",
                )
            )

            fav_saved_queries: List[FavStar] = []
            for cx in range(round(SAVED_QUERIES_FIXTURE_COUNT / 2)):
                fav_star: FavStar = FavStar(
                    user_id=admin.id, class_name="query", obj_id=saved_queries[cx].id
                )
                db.session.add(fav_star)
                db.session.commit()
                fav_saved_queries.append(fav_star)

            yield saved_queries

            # rollback changes
            for saved_query in saved_queries:
                db.session.delete(saved_query)
            for fav_saved_query in fav_saved_queries:
                db.session.delete(fav_saved_query)
            db.session.commit()

    @pytest.fixture
    def create_saved_queries_some_with_tags(
        self, create_custom_tags: Any  # noqa: F811
    ) -> Generator[List[SavedQuery], None, None]:
        """
        Fixture that creates 4 saved queries:
            - ``first_query`` is associated with ``first_tag``
            - ``second_query`` is associated with ``second_tag``
            - ``third_query`` is associated with both ``first_tag`` and ``second_tag``
            - ``fourth_query`` is not associated with any tag

        Relies on the ``create_custom_tags`` fixture for the tag creation.
        """
        with self.create_app().app_context():
            tags: Dict[str, Tag] = {
                "first_tag": db.session.query(Tag)
                .filter(Tag.name == "first_tag")
                .first(),
                "second_tag": db.session.query(Tag)
                .filter(Tag.name == "second_tag")
                .first(),
            }

            query_labels: List[str] = [
                "first_query",
                "second_query",
                "third_query",
                "fourth_query",
            ]
            queries: List[SavedQuery] = [
                self.insert_default_saved_query(label=name) for name in query_labels
            ]

            tag_associations: List[TaggedObject] = [
                TaggedObject(
                    object_id=queries[0].id,
                    object_type=ObjectType.chart,
                    tag=tags["first_tag"],
                ),
                TaggedObject(
                    object_id=queries[1].id,
                    object_type=ObjectType.chart,
                    tag=tags["second_tag"],
                ),
                TaggedObject(
                    object_id=queries[2].id,
                    object_type=ObjectType.chart,
                    tag=tags["first_tag"],
                ),
                TaggedObject(
                    object_id=queries[2].id,
                    object_type=ObjectType.chart,
                    tag=tags["second_tag"],
                ),
            ]

            for association in tag_associations:
                db.session.add(association)
            db.session.commit()

            yield queries

            # rollback changes
            for association in tag_associations:
                db.session.delete(association)
            for chart in queries:
                db.session.delete(chart)
            db.session.commit()

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_saved_query(self) -> None:
        """
        Saved Query API: Test get list saved query
        """
        admin: Any = self.get_user("admin")
        saved_queries: List[SavedQuery] = (
            db.session.query(SavedQuery).filter(SavedQuery.created_by == admin).all()
        )

        self.login(ADMIN_USERNAME)
        uri: str = "api/v1/saved_query/"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(saved_queries)
        expected_columns: List[str] = [
            "changed_on_delta_humanized",
            "created_on",
            "created_by",
            "database",
            "db_id",
            "description",
            "id",
            "label",
            "schema",
            "sql",
            "sql_tables",
        ]
        for expected_column in expected_columns:
            assert expected_column in data["result"][0]

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_saved_query_gamma(self) -> None:
        """
        Saved Query API: Test get list saved query
        """
        user: Any = self.get_user("gamma_sqllab")
        saved_queries: List[SavedQuery] = (
            db.session.query(SavedQuery).filter(SavedQuery.created_by == user).all()
        )

        self.login(user.username)
        uri: str = "api/v1/saved_query/"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(saved_queries)

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_sort_saved_query(self) -> None:
        """
        Saved Query API: Test get list and sort saved query
        """
        admin: Any = self.get_user("admin")
        saved_queries: List[SavedQuery] = (
            db.session.query(SavedQuery)
            .filter(SavedQuery.created_by == admin)
            .order_by(SavedQuery.schema.asc())
        ).all()
        self.login(ADMIN_USERNAME)
        query_string: Dict[str, str] = {"order_column": "schema", "order_direction": "asc"}
        uri: str = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(saved_queries)
        for i, query in enumerate(saved_queries):
            assert query.schema == data["result"][i]["schema"]

        query_string = {
            "order_column": "database.database_name",
            "order_direction": "asc",
        }
        uri = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200

        query_string = {
            "order_column": "created_by.first_name",
            "order_direction": "asc",
        }
        uri = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_filter_saved_query(self) -> None:
        """
        Saved Query API: Test get list and filter saved query
        """
        all_queries: List[SavedQuery] = (
            db.session.query(SavedQuery).filter(SavedQuery.label.ilike("%2%")).all()
        )
        self.login(ADMIN_USERNAME)
        query_string: Dict[str, List[Dict[str, str]]] = {
            "filters": [{"col": "label", "opr": "ct", "value": "2"}],
        }
        uri: str = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(all_queries)

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_filter_database_saved_query(self) -> None:
        """
        Saved Query API: Test get list and database saved query
        """
        example_db: Database = get_example_database()
        admin_user: Any = self.get_user("admin")

        all_db_queries: List[SavedQuery] = (
            db.session.query(SavedQuery)
            .filter(SavedQuery.db_id == example_db.id)
            .filter(SavedQuery.created_by_fk == admin_user.id)
            .all()
        )

        self.login(ADMIN_USERNAME)
        query_string: Dict[str, List[Dict[str, Union[str, int]]]] = {
            "filters": [{"col": "database", "opr": "rel_o_m", "value": example_db.id}],
        }
        uri: str = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(all_db_queries)

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_filter_schema_saved_query(self) -> None:
        """
        Saved Query API: Test get list and schema saved query
        """
        schema_name: str = "schema1"
        admin_user: Any = self.get_user("admin")

        all_db_queries: List[SavedQuery] = (
            db.session.query(SavedQuery)
            .filter(SavedQuery.schema == schema_name)
            .filter(SavedQuery.created_by_fk == admin_user.id)
            .all()
        )

        self.login(ADMIN_USERNAME)
        query_string: Dict[str, List[Dict[str, str]]] = {
            "filters": [{"col": "schema", "opr": "eq", "value": schema_name}],
        }
        uri: str = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(all_db_queries)

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_custom_filter_schema_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (schema) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user("admin")

        all_queries: List[SavedQuery] = (
            db.session.query(SavedQuery)
            .filter(SavedQuery.created_by == admin)
            .filter(SavedQuery.schema.ilike("%2%"))
            .all()
        )
        query_string: Dict[str, List[Dict[str, str]]] = {
            "filters": [{"col": "label", "opr": "all_text", "value": "schema2"}],
        }
        uri: str = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(all_queries)

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_custom_filter_label_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (label) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user("admin")
        all_queries: List[SavedQuery] = (
            db.session.query(SavedQuery)
            .filter(SavedQuery.created_by == admin)
            .filter(SavedQuery.label.ilike("%3%"))
            .all()
        )
        query_string: Dict[str, List[Dict[str, str]]] = {
            "filters": [{"col": "label", "opr": "all_text", "value": "label3"}],
        }
        uri: str = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(all_queries)

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_custom_filter_sql_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (sql) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user("admin")
        all_queries: List[SavedQuery] = (
            db.session.query(SavedQuery)
            .filter(SavedQuery.created_by == admin)
            .filter(SavedQuery.sql.ilike("%table%"))
            .all()
        )
        query_string: Dict[str, List[Dict[str, str]]] = {
            "filters": [{"col": "label", "opr": "all_text", "value": "table"}],
        }
        uri: str = f"api/v1/saved_query/?q={prison.dumps(query_string)}"
        rv: Response = self.get_assert_metric(uri, "get_list")
        assert rv.status_code == 200
        data: Dict[str, Any] = json.loads(rv.data.decode("utf-8"))
        assert data["count"] == len(all_queries)

    @pytest.mark.usefixtures("create_saved_queries")
    def test_get_list_custom_filter_description_saved_query(self) -> None:
        """
        Saved Query API: Test get list and custom filter (description) saved query
        """
        self.login(ADMIN_USERNAME)
        admin: Any = self.get_user("admin")
        all_queries: List[SavedQuery] = (
            db.session.query(SavedQuery)
            .