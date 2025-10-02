from __future__ import annotations
from typing import Any, List, Dict, Optional
import pytest
from freezegun import freeze_time
from pytest_mock import MockerFixture
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects.postgresql import dialect
from superset import app
from superset.commands.dataset.exceptions import DatasetNotFoundError
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.exceptions import SupersetTemplateException
from superset.jinja_context import dataset_macro, ExtraCache, metric_macro, safe_proxy, TimeFilter, WhereInMacro
from superset.models.core import Database
from superset.models.slice import Slice
from superset.utils import json

def test_filter_values_adhoc_filters() -> None:
    ...

def test_filter_values_extra_filters() -> None:
    ...

def test_filter_values_default() -> None:
    ...

def test_filter_values_remove_not_present() -> None:
    ...

def test_filter_values_no_default() -> None:
    ...

def test_get_filters_adhoc_filters() -> None:
    ...

def test_get_filters_remove_not_present() -> None:
    ...

def test_url_param_query() -> None:
    ...

def test_url_param_default() -> None:
    ...

def test_url_param_no_default() -> None:
    ...

def test_url_param_form_data() -> None:
    ...

def test_url_param_escaped_form_data() -> None:
    ...

def test_url_param_escaped_default_form_data() -> None:
    ...

def test_url_param_unescaped_form_data() -> None:
    ...

def test_url_param_unescaped_default_form_data() -> None:
    ...

def test_safe_proxy_primitive() -> None:
    ...

def test_safe_proxy_dict() -> None:
    ...

def test_safe_proxy_lambda() -> None:
    ...

def test_safe_proxy_nested_lambda() -> None:
    ...

def test_user_macros(mocker: MockerFixture) -> None:
    ...

def test_user_macros_without_cache_key_inclusion(mocker: MockerFixture) -> None:
    ...

def test_user_macros_without_user_info(mocker: MockerFixture) -> None:
    ...

def test_where_in() -> None:
    ...

def test_dataset_macro(mocker: MockerFixture) -> None:
    ...

def test_dataset_macro_mutator_with_comments(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_with_dataset_id(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_recursive(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_recursive_compound(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_recursive_cyclic(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_recursive_infinite(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_with_dataset_id_invalid_key(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_invalid_dataset_id(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_no_context(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_with_context_missing_info(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_with_context_datasource_id(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_with_context_datasource_id_none(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_with_context_chart_id(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_with_context_slice_id_none(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_with_context_deleted_chart(mocker: MockerFixture) -> None:
    ...

def test_metric_macro_no_dataset_id_available_in_request_form_data(mocker: MockerFixture) -> None:
    ...

def test_get_time_filter(description: str, args: List[str], kwargs: Dict[str, Any], sqlalchemy_uri: str, queries: List[Dict[str, Any]], time_filter: TimeFilter, removed_filters: List[str], applied_filters: List[str]) -> None:
    ...
