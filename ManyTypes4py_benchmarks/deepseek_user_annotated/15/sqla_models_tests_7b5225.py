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
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Pattern, Set, Tuple, Union
from re import Pattern
from unittest.mock import Mock, patch
import pytest

import numpy as np
import pandas as pd
from flask.ctx import AppContext
from pytest_mock import MockerFixture
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import TextClause

from superset import db
from superset.connectors.sqla.models import SqlaTable, TableColumn, SqlMetric
from superset.constants import EMPTY_STRING, NULL_STRING
from superset.db_engine_specs.bigquery import BigQueryEngineSpec
from superset.db_engine_specs.druid import DruidEngineSpec
from superset.exceptions import (
    QueryObjectValidationError,
)  # noqa: F401
from superset.models.core import Database
from superset.utils.core import (
    AdhocMetricExpressionType,
    FilterOperator,
    GenericDataType,
)
from superset.utils.database import get_example_database
from tests.integration_tests.fixtures.birth_names_dashboard import (
    load_birth_names_dashboard_with_slices,  # noqa: F401
    load_birth_names_data,  # noqa: F401
)

from .base_tests import SupersetTestCase
from .conftest import only_postgresql

VIRTUAL_TABLE_INT_TYPES: Dict[str, Pattern[str]] = {
    "hive": re.compile(r"^INT_TYPE$"),
    "mysql": re.compile("^LONGLONG$"),
    "postgresql": re.compile(r"^INTEGER$"),
    "presto": re.compile(r"^INTEGER$"),
    "sqlite": re.compile(r"^INT$"),
}

VIRTUAL_TABLE_STRING_TYPES: Dict[str, Pattern[str]] = {
    "hive": re.compile(r"^STRING_TYPE$"),
    "mysql": re.compile(r"^VAR_STRING$"),
    "postgresql": re.compile(r"^STRING$"),
    "presto": re.compile(r"^VARCHAR*"),
    "sqlite": re.compile(r"^STRING$"),
}


class FilterTestCase(NamedTuple):
    column: str
    operator: str
    value: Union[float, int, List[Any], str]
    expected: Union[str, List[str]]


class TestDatabaseModel(SupersetTestCase):
    def test_is_time_druid_time_col(self) -> None:
        """Druid has a special __time column"""

        database = Database(database_name="druid_db", sqlalchemy_uri="druid://db")
        tbl = SqlaTable(table_name="druid_tbl", database=database)
        col = TableColumn(column_name="__time", type="INTEGER", table=tbl)
        assert col.is_dttm is None
        DruidEngineSpec.alter_new_orm_column(col)
        assert col.is_dttm is True

        col = TableColumn(column_name="__not_time", type="INTEGER", table=tbl)
        assert col.is_temporal is False

    def test_temporal_varchar(self) -> None:
        """Ensure a column with is_dttm set to true evaluates to is_temporal == True"""

        database = get_example_database()
        tbl = SqlaTable(table_name="test_tbl", database=database)
        col = TableColumn(column_name="ds", type="VARCHAR", table=tbl)
        # by default, VARCHAR should not be assumed to be temporal
        assert col.is_temporal is False
        # changing to `is_dttm = True`, calling `is_temporal` should return True
        col.is_dttm = True
        assert col.is_temporal is True

    def test_db_column_types(self) -> None:
        test_cases: Dict[str, GenericDataType] = {
            # string
            "CHAR": GenericDataType.STRING,
            "VARCHAR": GenericDataType.STRING,
            "NVARCHAR": GenericDataType.STRING,
            "STRING": GenericDataType.STRING,
            "TEXT": GenericDataType.STRING,
            "NTEXT": GenericDataType.STRING,
            # numeric
            "INTEGER": GenericDataType.NUMERIC,
            "BIGINT": GenericDataType.NUMERIC,
            "DECIMAL": GenericDataType.NUMERIC,
            # temporal
            "DATE": GenericDataType.TEMPORAL,
            "DATETIME": GenericDataType.TEMPORAL,
            "TIME": GenericDataType.TEMPORAL,
            "TIMESTAMP": GenericDataType.TEMPORAL,
        }

        tbl = SqlaTable(table_name="col_type_test_tbl", database=get_example_database())
        for str_type, db_col_type in test_cases.items():
            col = TableColumn(column_name="foo", type=str_type, table=tbl)
            assert col.is_temporal == (db_col_type == GenericDataType.TEMPORAL)
            assert col.is_numeric == (db_col_type == GenericDataType.NUMERIC)
            assert col.is_string == (db_col_type == GenericDataType.STRING)

        for str_type, db_col_type in test_cases.items():  # noqa: B007
            col = TableColumn(column_name="foo", type=str_type, table=tbl, is_dttm=True)
            assert col.is_temporal

    @patch("superset.jinja_context.get_username", return_value="abc")
    def test_jinja_metrics_and_calc_columns(self, mock_username: Mock) -> None:
        base_query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
            "columns": [
                "user",
                "expr",
                {
                    "hasCustomLabel": True,
                    "label": "adhoc_column",
                    "sqlExpression": "'{{ 'foo_' + time_grain }}'",
                },
            ],
            "metrics": [
                {
                    "hasCustomLabel": True,
                    "label": "adhoc_metric",
                    "expressionType": AdhocMetricExpressionType.SQL,
                    "sqlExpression": "SUM(case when user = '{{ 'user_' + "
                    "current_username() }}' then 1 else 0 end)",
                },
                "count_timegrain",
            ],
            "is_timeseries": False,
            "filter": [],
            "extras": {"time_grain_sqla": "P1D"},
        }

        table = SqlaTable(
            table_name="test_has_jinja_metric_and_expr",
            sql="SELECT '{{ 'user_' + current_username() }}' as user, "
            "'{{ 'xyz_' + time_grain }}' as time_grain",
            database=get_example_database(),
        )
        TableColumn(
            column_name="expr",
            expression="case when '{{ current_username() }}' = 'abc' "
            "then 'yes' else 'no' end",
            type="VARCHAR(100)",
            table=table,
        )
        SqlMetric(
            metric_name="count_timegrain",
            expression="count('{{ 'bar_' + time_grain }}')",
            table=table,
        )
        db.session.commit()

        sqla_query = table.get_sqla_query(**base_query_obj)
        query = table.database.compile_sqla_query(sqla_query.sqla_query)

        # assert virtual dataset
        assert "SELECT 'user_abc' as user, 'xyz_P1D' as time_grain" in query
        # assert dataset calculated column
        assert "case when 'abc' = 'abc' then 'yes' else 'no' end" in query
        # assert adhoc column
        assert "'foo_P1D'" in query
        # assert dataset saved metric
        assert "count('bar_P1D')" in query
        # assert adhoc metric
        assert "SUM(case when user = 'user_abc' then 1 else 0 end)" in query
        # Cleanup
        db.session.delete(table)
        db.session.commit()

    @patch("superset.jinja_context.get_dataset_id_from_context")
    def test_jinja_metric_macro(self, mock_dataset_id_from_context: Mock) -> None:
        self.login(username="admin")
        table = self.get_table(name="birth_names")
        metric = SqlMetric(
            metric_name="count_jinja_metric", expression="count(*)", table=table
        )
        db.session.commit()

        base_query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
            "columns": [],
            "metrics": [
                {
                    "hasCustomLabel": True,
                    "label": "Metric using Jinja macro",
                    "expressionType": AdhocMetricExpressionType.SQL,
                    "sqlExpression": "{{ metric('count_jinja_metric') }}",
                },
                {
                    "hasCustomLabel": True,
                    "label": "Same but different",
                    "expressionType": AdhocMetricExpressionType.SQL,
                    "sqlExpression": "{{ metric('count_jinja_metric', "
                    + str(table.id)
                    + ") }}",
                },
            ],
            "is_timeseries": False,
            "filter": [],
            "extras": {"time_grain_sqla": "P1D"},
        }
        mock_dataset_id_from_context.return_value = table.id

        sqla_query = table.get_sqla_query(**base_query_obj)
        query = table.database.compile_sqla_query(sqla_query.sqla_query)

        database = table.database
        with database.get_sqla_engine() as engine:
            quote = engine.dialect.identifier_preparer.quote_identifier

        for metric_label in {"metric using jinja macro", "same but different"}:
            assert f"count(*) as {quote(metric_label)}" in query.lower()

        db.session.delete(metric)
        db.session.commit()

    def test_adhoc_metrics_and_calc_columns(self) -> None:
        base_query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
            "groupby": ["user", "expr"],
            "metrics": [
                {
                    "expressionType": AdhocMetricExpressionType.SQL,
                    "sqlExpression": "(SELECT (SELECT * from birth_names) "
                    "from test_validate_adhoc_sql)",
                    "label": "adhoc_metrics",
                }
            ],
            "is_timeseries": False,
            "filter": [],
        }

        table = SqlaTable(
            table_name="test_validate_adhoc_sql", database=get_example_database()
        )
        db.session.commit()

        with pytest.raises(QueryObjectValidationError):
            table.get_sqla_query(**base_query_obj)
        # Cleanup
        db.session.delete(table)
        db.session.commit()

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_where_operators(self) -> None:
        filters: Tuple[FilterTestCase, ...] = (
            FilterTestCase("num", FilterOperator.IS_NULL, "", "IS NULL"),
            FilterTestCase("num", FilterOperator.IS_NOT_NULL, "", "IS NOT NULL"),
            # Some db backends translate true/false to 1/0
            FilterTestCase("num", FilterOperator.IS_TRUE, "", ["IS 1", "IS true"]),
            FilterTestCase("num", FilterOperator.IS_FALSE, "", ["IS 0", "IS false"]),
            FilterTestCase("num", FilterOperator.GREATER_THAN, 0, "> 0"),
            FilterTestCase("num", FilterOperator.GREATER_THAN_OR_EQUALS, 0, ">= 0"),
            FilterTestCase("num", FilterOperator.LESS_THAN, 0, "< 0"),
            FilterTestCase("num", FilterOperator.LESS_THAN_OR_EQUALS, 0, "<= 0"),
            FilterTestCase("num", FilterOperator.EQUALS, 0, "= 0"),
            FilterTestCase("num", FilterOperator.NOT_EQUALS, 0, "!= 0"),
            FilterTestCase("num", FilterOperator.IN, ["1", "2"], "IN (1, 2)"),
            FilterTestCase("num", FilterOperator.NOT_IN, ["1", "2"], "NOT IN (1, 2)"),
            FilterTestCase(
                "ds", FilterOperator.TEMPORAL_RANGE, "2020 : 2021", "2020-01-01"
            ),
        )
        table = self.get_table(name="birth_names")
        for filter_ in filters:
            query_obj = {
                "granularity": None,
                "from_dttm": None,
                "to_dttm": None,
                "groupby": ["gender"],
                "metrics": ["count"],
                "is_timeseries": False,
                "filter": [
                    {
                        "col": filter_.column,
                        "op": filter_.operator,
                        "val": filter_.value,
                    }
                ],
                "extras": {},
            }
            sqla_query = table.get_sqla_query(**query_obj)
            sql = table.database.compile_sqla_query(sqla_query.sqla_query)
            if isinstance(filter_.expected, list):
                assert any([candidate in sql for candidate in filter_.expected])  # noqa: C419
            else:
                assert filter_.expected in sql

    @pytest.mark.usefixtures("load_birth_names_dashboard_with_slices")
    def test_boolean_type_where_operators(self) -> None:
        table = self.get_table(name="birth_names")
        db.session.add(
            TableColumn(
                column_name="boolean_gender",
                expression="case when gender = 'boy' then True else False end",
                type="BOOLEAN",
                table=table,
            )
        )
        query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
            "groupby": ["boolean_gender"],
            "metrics": ["count"],
            "is_timeseries": False,
            "filter": [
                {
                    "col": "boolean_gender",
                    "op": FilterOperator.IN,
                    "val": ["true", "false"],
                }
            ],
            "extras": {},
        }
        sqla_query = table.get_sqla_query(**query_obj)
        sql = table.database.compile_sqla_query(sqla_query.sqla_query)
        dialect = table.database.get_dialect()
        operand = "(true, false)"
        # override native_boolean=False behavior in MySQLCompiler
        # https://github.com/sqlalchemy/sqlalchemy/blob/master/lib/sqlalchemy/dialects/mysql/base.py
        if not dialect.supports_native_boolean and dialect.name != "mysql":
            operand = "(1, 0)"
        assert f"IN {operand}" in sql

    def test_incorrect_jinja_syntax_raises_correct_exception(self) -> None:
        query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
            "groupby": ["user"],
            "metrics": [],
            "is_timeseries": False,
            "filter": [],
            "extras": {},
        }

        # Table with Jinja callable.
        table = SqlaTable(
            table_name="test_table",
            sql="SELECT '{{ abcd xyz + 1 ASDF }}' as user",
            database=get_example_database(),
        )
        # TODO(villebro): make it work with presto
        if get_example_database().backend != "presto":
            with pytest.raises(QueryObjectValidationError):
                table.get_sqla_query(**query_obj)

    def test_query_format_strip_trailing_semicolon(self) -> None:
        query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
            "groupby": ["user"],
            "metrics": [],
            "is_timeseries": False,
            "filter": [],
            "extras": {},
        }

        table = SqlaTable(
            table_name="another_test_table",
            sql="SELECT * from test_table;",
            database=get_example_database(),
        )
        sqlaq = table.get_sqla_query(**query_obj)
        sql = table.database.compile_sqla_query(sqlaq.sqla_query)
        assert sql[-1] != ";"

    def test_multiple_sql_statements_raises_exception(self) -> None:
        base_query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
            "groupby": ["grp"],
            "metrics": [],
            "is_timeseries": False,
            "filter": [],
        }

        table = SqlaTable(
            table_name="test_multiple_sql_statements",
            sql="SELECT 'foo' as grp, 1 as num; SELECT 'bar' as grp, 2 as num",
            database=get_example_database(),
        )

        query_obj = dict(**base_query_obj, extras={})
        with pytest.raises(QueryObjectValidationError):
            table.get_sqla_query(**query_obj)

    def test_dml_statement_raises_exception(self) -> None:
        base_query_obj = {
            "granularity": None,
            "from_dttm": None,
            "to_dttm": None,
           