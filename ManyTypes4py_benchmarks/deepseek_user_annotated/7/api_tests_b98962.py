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

import unittest
import copy
from datetime import datetime
from io import BytesIO
import time
from typing import Any, Optional, Dict, List, Set, Union, cast
from unittest import mock
from zipfile import ZipFile

from flask import Response
from flask.ctx import AppContext
from tests.integration_tests.conftest import with_feature_flags
from superset.charts.data.api import ChartDataRestApi
from superset.models.sql_lab import Query
from tests.integration_tests.base_tests import SupersetTestCase, test_client
from tests.integration_tests.annotation_layers.fixtures import create_annotation_layers  # noqa: F401
from tests.integration_tests.constants import (
    ADMIN_USERNAME,
    GAMMA_NO_CSV_USERNAME,
    GAMMA_USERNAME,
)
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,  # noqa: F401
    load_birth_names_data,  # noqa: F401
)
from tests.integration_tests.test_app import app
from tests.integration_tests.fixtures.energy_dashboard import (
    load_energy_table_with_slice,  # noqa: F401
    load_energy_table_data,  # noqa: F401
)
import pytest
from superset.models.slice import Slice

from superset.commands.chart.data.get_data_command import ChartDataCommand
from superset.connectors.sqla.models import TableColumn, SqlaTable
from superset.errors import SupersetErrorType
from superset.extensions import async_query_manager_factory, db
from superset.models.annotations import AnnotationLayer
from superset.superset_typing import AdhocColumn
from superset.utils.core import (
    AnnotationType,
    backend,
    get_example_default_schema,
    AdhocMetricExpressionType,
    ExtraFiltersReasonType,
)
from superset.utils import json
from superset.utils.database import get_example_database, get_main_database
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType

from tests.common.query_context_generator import ANNOTATION_LAYERS
from tests.integration_tests.fixtures.query_context import get_query_context

from tests.integration_tests.test_app import app  # noqa: F811


CHART_DATA_URI: str = "api/v1/chart/data"
CHARTS_FIXTURE_COUNT: int = 10
ADHOC_COLUMN_FIXTURE: AdhocColumn = {
    "hasCustomLabel": True,
    "label": "male_or_female",
    "sqlExpression": "case when gender = 'boy' then 'male' "
    "when gender = 'girl' then 'female' else 'other' end",
}

INCOMPATIBLE_ADHOC_COLUMN_FIXTURE: AdhocColumn = {
    "hasCustomLabel": True,
    "label": "exciting_or_boring",
    "sqlExpression": "case when genre = 'Action' then 'Exciting' else 'Boring' end",
}


@pytest.fixture(autouse=True)
def skip_by_backend(app_context: AppContext) -> None:
    if backend() == "hive":
        pytest.skip("Skipping tests for Hive backend")


class BaseTestChartDataApi(SupersetTestCase):
    query_context_payload_template: Optional[Dict[str, Any]] = None

    def setUp(self) -> None:
        self.login(ADMIN_USERNAME)
        if self.query_context_payload_template is None:
            BaseTestChartDataApi.query_context_payload_template = get_query_context(
                "birth_names"
            )
        self.query_context_payload: Dict[str, Any] = (
            copy.deepcopy(self.query_context_payload_template) or {}
        )

    def get_expected_row_count(self, client_id: str) -> int:
        start_date: datetime = datetime.now()
        start_date = start_date.replace(
            year=start_date.year - 100, hour=0, minute=0, second=0
        )

        quoted_table_name: str = self.quote_name("birth_names")
        sql: str = f"""
                            SELECT COUNT(*) AS rows_count FROM (
                                SELECT name AS name, SUM(num) AS sum__num
                                FROM {quoted_table_name}
                                WHERE ds >= '{start_date.strftime("%Y-%m-%d %H:%M:%S")}'
                                AND gender = 'boy'
                                GROUP BY name
                                ORDER BY sum__num DESC
                                LIMIT 100) AS inner__query
                        """  # noqa: S608
        resp: Dict[str, Any] = self.run_sql(sql, client_id, raise_on_error=True)
        db.session.query(Query).delete()
        db.session.commit()
        return resp["data"][0]["rows_count"]

    def quote_name(self, name: str) -> str:
        if get_main_database().backend in {"presto", "hive"}:
            with get_example_database().get_inspector() as inspector:
                return inspector.engine.dialect.identifier_preparer.quote_identifier(
                    name
                )
        return name


@pytest.mark.chart_data_flow
class TestPostChartDataApi(BaseTestChartDataApi):
    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test__map_form_data_datasource_to_dataset_id(self) -> None:
        # arrange
        self.query_context_payload["datasource"] = {"id": 1, "type": "table"}
        # act
        response: Dict[str, Optional[int]] = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload
        )
        # assert
        assert response == {"dashboard_id": None, "dataset_id": 1, "slice_id": None}

        # takes malformed content without raising an error
        self.query_context_payload["datasource"] = "1__table"
        # act
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload
        )
        # assert
        assert response == {"dashboard_id": None, "dataset_id": None, "slice_id": None}

        # takes a slice id
        self.query_context_payload["datasource"] = None
        self.query_context_payload["form_data"] = {"slice_id": 1}
        # act
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload
        )
        # assert
        assert response == {"dashboard_id": None, "dataset_id": None, "slice_id": 1}

        # takes missing slice id
        self.query_context_payload["datasource"] = None
        self.query_context_payload["form_data"] = {"foo": 1}
        # act
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload
        )
        # assert
        assert response == {"dashboard_id": None, "dataset_id": None, "slice_id": None}

        # takes a dashboard id
        self.query_context_payload["form_data"] = {"dashboardId": 1}
        # act
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload
        )
        # assert
        assert response == {"dashboard_id": 1, "dataset_id": None, "slice_id": None}

        # takes a dashboard id and a slice id
        self.query_context_payload["form_data"] = {"dashboardId": 1, "slice_id": 2}
        # act
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload
        )
        # assert
        assert response == {"dashboard_id": 1, "dataset_id": None, "slice_id": 2}

        # takes a dashboard id, slice id and a dataset id
        self.query_context_payload["datasource"] = {"id": 3, "type": "table"}
        self.query_context_payload["form_data"] = {"dashboardId": 1, "slice_id": 2}
        # act
        response = ChartDataRestApi._map_form_data_datasource_to_dataset_id(
            ChartDataRestApi, self.query_context_payload
        )
        # assert
        assert response == {"dashboard_id": 1, "dataset_id": 3, "slice_id": 2}

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    @mock.patch("superset.utils.decorators.g")
    def test_with_valid_qc__data_is_returned(self, mock_g: mock.Mock) -> None:
        mock_g.logs_context = {}
        # arrange
        expected_row_count: int = self.get_expected_row_count("client_id_1")
        # act
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        # assert
        assert rv.status_code == 200
        self.assert_row_count(rv, expected_row_count)

        # check that global logs decorator is capturing from form_data
        assert isinstance(mock_g.logs_context.get("dataset_id"), int)

    @staticmethod
    def assert_row_count(rv: Response, expected_row_count: int) -> None:
        assert rv.json["result"][0]["rowcount"] == expected_row_count

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    @mock.patch(
        "superset.common.query_context_factory.config",
        {**app.config, "ROW_LIMIT": 7},
    )
    def test_without_row_limit__row_count_as_default_row_limit(self) -> None:
        # arrange
        expected_row_count: int = 7
        del self.query_context_payload["queries"][0]["row_limit"]
        # act
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        # assert
        self.assert_row_count(rv, expected_row_count)

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    @mock.patch(
        "superset.common.query_context_factory.config",
        {**app.config, "SAMPLES_ROW_LIMIT": 5},
    )
    def test_as_samples_without_row_limit__row_count_as_default_samples_row_limit(self) -> None:
        # arrange
        expected_row_count: int = 5
        app.config["SAMPLES_ROW_LIMIT"] = expected_row_count
        self.query_context_payload["result_type"] = ChartDataResultType.SAMPLES
        del self.query_context_payload["queries"][0]["row_limit"]

        # act
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")

        # assert
        self.assert_row_count(rv, expected_row_count)
        assert "GROUP BY" not in rv.json["result"][0]["query"]

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    @mock.patch(
        "superset.utils.core.current_app.config",
        {**app.config, "SQL_MAX_ROW": 10},
    )
    def test_with_row_limit_bigger_then_sql_max_row__rowcount_as_sql_max_row(self) -> None:
        # arrange
        expected_row_count: int = 10
        self.query_context_payload["queries"][0]["row_limit"] = 10000000

        # act
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")

        # assert
        self.assert_row_count(rv, expected_row_count)

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    @mock.patch(
        "superset.utils.core.current_app.config",
        {**app.config, "SQL_MAX_ROW": 5},
    )
    def test_as_samples_with_row_limit_bigger_then_sql_max_row_rowcount_as_sql_max_row(
        self,
    ) -> None:
        expected_row_count: int = app.config["SQL_MAX_ROW"]
        self.query_context_payload["result_type"] = ChartDataResultType.SAMPLES
        self.query_context_payload["queries"][0]["row_limit"] = 10000000
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")

        # assert
        self.assert_row_count(rv, expected_row_count)
        assert "GROUP BY" not in rv.json["result"][0]["query"]

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    @mock.patch(
        "superset.common.query_actions.config",
        {**app.config, "SAMPLES_ROW_LIMIT": 5, "SQL_MAX_ROW": 15},
    )
    def test_with_row_limit_as_samples__rowcount_as_row_limit(self) -> None:
        expected_row_count: int = 10
        self.query_context_payload["result_type"] = ChartDataResultType.SAMPLES
        self.query_context_payload["queries"][0]["row_limit"] = expected_row_count

        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")

        # assert
        self.assert_row_count(rv, expected_row_count)
        assert "GROUP BY" not in rv.json["result"][0]["query"]

    def test_with_incorrect_result_type__400(self) -> None:
        self.query_context_payload["result_type"] = "qwerty"
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")

        assert rv.status_code == 400

    def test_with_incorrect_result_format__400(self) -> None:
        self.query_context_payload["result_format"] = "qwerty"
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        assert rv.status_code == 400

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_with_invalid_payload__400(self) -> None:
        invalid_query_context: Dict[str, str] = {"form_data": "NOT VALID JSON"}

        rv: Response = self.client.post(
            CHART_DATA_URI,
            data=invalid_query_context,
            content_type="multipart/form-data",
        )

        assert rv.status_code == 400
        assert rv.json["message"] == "Request is not JSON"

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_with_query_result_type__200(self) -> None:
        self.query_context_payload["result_type"] = ChartDataResultType.QUERY
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        assert rv.status_code == 200

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_empty_request_with_csv_result_format(self) -> None:
        """
        Chart data API: Test empty chart data with CSV result format
        """
        self.query_context_payload["result_format"] = "csv"
        self.query_context_payload["queries"] = []
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        assert rv.status_code == 400

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_empty_request_with_excel_result_format(self) -> None:
        """
        Chart data API: Test empty chart data with Excel result format
        """
        self.query_context_payload["result_format"] = "xlsx"
        self.query_context_payload["queries"] = []
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        assert rv.status_code == 400

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_with_csv_result_format(self) -> None:
        """
        Chart data API: Test chart data with CSV result format
        """
        self.query_context_payload["result_format"] = "csv"
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        assert rv.status_code == 200
        assert rv.mimetype == "text/csv"

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_with_excel_result_format(self) -> None:
        """
        Chart data API: Test chart data with Excel result format
        """
        self.query_context_payload["result_format"] = "xlsx"
        mimetype: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        rv: Response = self.post_assert_metric(CHART_DATA_URI, self.query_context_payload, "data")
        assert rv.status_code == 200
        assert rv.mimetype == mimetype

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def