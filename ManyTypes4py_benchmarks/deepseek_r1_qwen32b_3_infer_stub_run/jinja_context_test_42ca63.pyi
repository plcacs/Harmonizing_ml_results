from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from datetime import datetime
from pytest_mock import MockerFixture
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.daos.dataset import DatasetDAO
from superset.exceptions import SupersetTemplateException, DatasetNotFoundError
from superset.jinja_context import (
    ExtraCache,
    TimeFilter,
    WhereInMacro,
    dataset_macro,
    metric_macro,
    safe_proxy,
)
from superset.models.core import Database
from superset.models.slice import Slice
from sqlalchemy.dialects.postgresql import dialect

class ExtraCache:
    def __init__(self, database: Optional[Database] = ..., table: Optional[SqlaTable] = ..., dialect: Any = ...):
        ...
    
    def filter_values(self, column_name: str, default: Optional[Union[str, List[str]]] = ..., remove_filter: bool = ...) -> Union[List[str], List[bytes]]:
        ...
    
    def get_filters(self, column_name: str, remove_filter: bool = ...) -> List[Dict[str, Any]]:
        ...
    
    def url_param(self, param_name: str, default: Optional[str] = ..., escape_result: bool = ...) -> Optional[Union[str, bytes]]:
        ...
    
    def current_user_id(self, add_to_cache_keys: bool = ...) -> Optional[int]:
        ...
    
    def current_username(self, add_to_cache_keys: bool = ...) -> Optional[str]:
        ...
    
    def current_user_email(self, add_to_cache_keys: bool = ...) -> Optional[str]:
        ...
    
    def get_time_filter(
        self,
        column_name: Optional[str] = ...,
        default: Optional[str] = ...,
        target_type: Optional[str] = ...,
        remove_filter: bool = ...,
        strftime: Optional[str] = ...,
    ) -> TimeFilter:
        ...

class TimeFilter:
    from_expr: str
    to_expr: str
    time_range: str

class WhereInMacro:
    def __call__(self, value: List[Union[int, str]], mark: Optional[str] = ...) -> str:
        ...

def safe_proxy(func: Any, value: Any) -> Any:
    ...

def dataset_macro(dataset_id: int, include_metrics: bool = ..., columns: Optional[List[str]] = ...) -> str:
    ...

def metric_macro(
    metric_key: str,
    dataset_id: Optional[int] = ...,
    recursive: bool = ...,
) -> str:
    ...

def test_filter_values_adhoc_filters(form_data: Dict[str, List[Dict[str, Any]]]) -> None:
    ...

def test_filter_values_extra_filters(form_data: Dict[str, List[Dict[str, Any]]]) -> None:
    ...

def test_filter_values_default() -> None:
    ...

def test_filter_values_remove_not_present() -> None:
    ...

def test_filter_values_no_default() -> None:
    ...

def test_get_filters_adhoc_filters(form_data: Dict[str, List[Dict[str, Any]]]) -> None:
    ...

def test_get_filters_remove_not_present() -> None:
    ...

def test_url_param_query(query_string: Dict[str, str]) -> None:
    ...

def test_url_param_default() -> None:
    ...

def test_url_param_no_default() -> None:
    ...

def test_url_param_form_data(form_data: Dict[str, str]) -> None:
    ...

def test_url_param_escaped_form_data(form_data: Dict[str, str], dialect: dialect) -> None:
    ...

def test_url_param_escaped_default_form_data(form_data: Dict[str, str], dialect: dialect) -> None:
    ...

def test_url_param_unescaped_form_data(form_data: Dict[str, str], dialect: dialect) -> None:
    ...

def test_url_param_unescaped_default_form_data(form_data: Dict[str, str], dialect: dialect) -> None:
    ...

def test_safe_proxy_primitive(func: Any) -> None:
    ...

def test_safe_proxy_dict(func: Any) -> None:
    ...

def test_safe_proxy_lambda(func: Any) -> None:
    ...

def test_safe_proxy_nested_lambda(func: Any) -> None:
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

def test_get_time_filter(
    description: str,
    args: List[Any],
    kwargs: Dict[str, Any],
    sqlalchemy_uri: str,
    queries: List[Dict[str, Any]],
    time_filter: TimeFilter,
    removed_filters: List[str],
    applied_filters: List[str],
) -> None:
    ...