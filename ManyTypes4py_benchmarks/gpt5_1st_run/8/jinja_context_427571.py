from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    TYPE_CHECKING,
    TypedDict,
)

import dateutil
from flask import current_app, g, has_request_context, request
from flask_babel import gettext as _
from jinja2 import DebugUndefined, Environment, nodes
from jinja2.nodes import Call, Node
from jinja2.sandbox import SandboxedEnvironment
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql.expression import bindparam
from sqlalchemy.types import String

from superset.commands.dataset.exceptions import DatasetNotFoundError
from superset.common.utils.time_range_utils import get_since_until_from_time_range
from superset.constants import LRU_CACHE_MAX_SIZE, NO_TIME_RANGE
from superset.exceptions import SupersetTemplateException
from superset.extensions import feature_flag_manager
from superset.sql_parse import Table
from superset.utils import json
from superset.utils.core import (
    AdhocFilterClause,
    FilterOperator,
    convert_legacy_filters_into_adhoc,
    get_user_email,
    get_user_id,
    get_username,
    merge_extra_filters,
)

if TYPE_CHECKING:
    from superset.connectors.sqla.models import SqlaTable
    from superset.models.core import Database
    from superset.models.sql_lab import Query

NONE_TYPE: str = type(None).__name__
ALLOWED_TYPES: Tuple[str, ...] = (
    NONE_TYPE,
    "bool",
    "str",
    "unicode",
    "int",
    "long",
    "float",
    "list",
    "dict",
    "tuple",
    "set",
    "TimeFilter",
)
COLLECTION_TYPES: Tuple[str, ...] = ("list", "dict", "tuple", "set")


@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def context_addons() -> Dict[str, Any]:
    return current_app.config.get("JINJA_CONTEXT_ADDONS", {})


class Filter(TypedDict):
    op: Optional[str]
    col: str
    val: Any


@dataclass
class TimeFilter:
    """
    Container for temporal filter.
    """

    from_expr: Optional[str]
    to_expr: Optional[str]
    time_range: str


class ExtraCache:
    """
    Dummy class that exposes a method used to store additional values used in
    calculation of query object cache keys.
    """

    regex: ClassVar[Pattern[str]] = re.compile(
        "(\{\{|\{%)[^{}]*?(current_user_id\([^()]*\)|current_username\([^()]*\)|current_user_email\([^()]*\)|cache_key_wrapper\([^()]*\)|url_param\([^()]*\))[^{}]*?(\}\}|\%})"
    )

    def __init__(
        self,
        extra_cache_keys: Optional[List[Any]] = None,
        applied_filters: Optional[List[str]] = None,
        removed_filters: Optional[List[str]] = None,
        database: Optional[Database] = None,
        dialect: Optional[Dialect] = None,
        table: Optional[SqlaTable] = None,
    ) -> None:
        self.extra_cache_keys: Optional[List[Any]] = extra_cache_keys
        self.applied_filters: List[str] = applied_filters if applied_filters is not None else []
        self.removed_filters: List[str] = removed_filters if removed_filters is not None else []
        self.database: Optional[Database] = database
        self.dialect: Optional[Dialect] = dialect
        self.table: Optional[SqlaTable] = table

    def current_user_id(self, add_to_cache_keys: bool = True) -> Optional[int]:
        """
        Return the user ID of the user who is currently logged in.

        :param add_to_cache_keys: Whether the value should be included in the cache key
        :returns: The user ID
        """
        if (user_id := get_user_id()):
            if add_to_cache_keys:
                self.cache_key_wrapper(user_id)
            return user_id
        return None

    def current_username(self, add_to_cache_keys: bool = True) -> Optional[str]:
        """
        Return the username of the user who is currently logged in.

        :param add_to_cache_keys: Whether the value should be included in the cache key
        :returns: The username
        """
        if (username := get_username()):
            if add_to_cache_keys:
                self.cache_key_wrapper(username)
            return username
        return None

    def current_user_email(self, add_to_cache_keys: bool = True) -> Optional[str]:
        """
        Return the email address of the user who is currently logged in.

        :param add_to_cache_keys: Whether the value should be included in the cache key
        :returns: The user email address
        """
        if (email_address := get_user_email()):
            if add_to_cache_keys:
                self.cache_key_wrapper(email_address)
            return email_address
        return None

    def cache_key_wrapper(self, key: Any) -> Any:
        """
        Adds values to a list that is added to the query object used for calculating a
        cache key.

        This is needed if the following applies:
            - Caching is enabled
            - The query is dynamically generated using a jinja template
            - A `JINJA_CONTEXT_ADDONS` or similar is used as a filter in the query

        :param key: Any value that should be considered when calculating the cache key
        :return: the original value ``key`` passed to the function
        """
        if self.extra_cache_keys is not None:
            self.extra_cache_keys.append(key)
        return key

    def url_param(
        self,
        param: str,
        default: Any = None,
        add_to_cache_keys: bool = True,
        escape_result: bool = True,
    ) -> Any:
        """
        Read a url or post parameter and use it in your SQL Lab query.
        """
        from superset.views.utils import get_form_data

        if has_request_context() and request.args.get(param):
            return request.args.get(param, default)

        form_data, _ = get_form_data()
        url_params = form_data.get("url_params") or {}
        result: Any = url_params.get(param, default)
        if result and escape_result and self.dialect:
            result = String().literal_processor(dialect=self.dialect)(value=result)[1:-1]
        if add_to_cache_keys:
            self.cache_key_wrapper(result)
        return result

    def filter_values(
        self, column: str, default: Any = None, remove_filter: bool = False
    ) -> List[Any]:
        """Gets a values for a particular filter as a list"""
        return_val: List[Any] = []
        filters = self.get_filters(column, remove_filter)
        for flt in filters:
            val = flt.get("val")
            if isinstance(val, list):
                return_val.extend(val)
            elif val:
                return_val.append(val)
        if not return_val and default:
            return_val = [default]
        return return_val

    def get_filters(self, column: str, remove_filter: bool = False) -> List[Filter]:
        """Get the filters applied to the given column."""
        from superset.views.utils import get_form_data

        form_data, _ = get_form_data()
        convert_legacy_filters_into_adhoc(form_data)
        merge_extra_filters(form_data)
        filters: List[Filter] = []
        for flt in form_data.get("adhoc_filters", []):
            val: Any = flt.get("comparator")
            op: Optional[str] = flt["operator"].upper() if flt.get("operator") else None
            if (
                flt.get("expressionType") == "SIMPLE"
                and flt.get("clause") == "WHERE"
                and (flt.get("subject") == column)
                and val
            ):
                if remove_filter:
                    if column not in self.removed_filters:
                        self.removed_filters.append(column)
                if column not in self.applied_filters:
                    self.applied_filters.append(column)
                if op in (FilterOperator.IN.value, FilterOperator.NOT_IN.value) and (
                    not isinstance(val, list)
                ):
                    val = [val]
                filters.append({"op": op, "col": column, "val": val})
        return filters

    def get_time_filter(
        self,
        column: Optional[str] = None,
        default: Optional[str] = None,
        target_type: Optional[str] = None,
        strftime: Optional[str] = None,
        remove_filter: bool = False,
    ) -> TimeFilter:
        """Get the time filter with appropriate formatting."""
        from superset.views.utils import get_form_data

        form_data, _ = get_form_data()
        convert_legacy_filters_into_adhoc(form_data)
        merge_extra_filters(form_data)
        time_range: Optional[str] = form_data.get("time_range")
        if column:
            flt = next(
                (
                    flt
                    for flt in form_data.get("adhoc_filters", [])
                    if flt["operator"] == FilterOperator.TEMPORAL_RANGE
                    and flt["subject"] == column
                ),
                None,
            )
            if flt:
                if remove_filter:
                    if column not in self.removed_filters:
                        self.removed_filters.append(column)
                if column not in self.applied_filters:
                    self.applied_filters.append(column)
                time_range = cast(str, flt["comparator"])
                if not target_type and self.table:
                    target_type = self.table.columns_types.get(column)
        time_range = time_range or NO_TIME_RANGE
        if time_range == NO_TIME_RANGE and default:
            time_range = default
        from_expr_dt, to_expr_dt = get_since_until_from_time_range(time_range)

        def _format_dttm(dttm: Optional[datetime]) -> Optional[str]:
            if strftime and dttm:
                return dttm.strftime(strftime)
            return (
                self.database.db_engine_spec.convert_dttm(target_type or "", dttm)
                if self.database and dttm
                else None
            )

        return TimeFilter(
            from_expr=_format_dttm(from_expr_dt),
            to_expr=_format_dttm(to_expr_dt),
            time_range=time_range,
        )


def safe_proxy(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return_value = func(*args, **kwargs)
    value_type = type(return_value).__name__
    if value_type not in ALLOWED_TYPES:
        raise SupersetTemplateException(
            _(
                "Unsafe return type for function %(func)s: %(value_type)s",
                func=func.__name__,
                value_type=value_type,
            )
        )
    if value_type in COLLECTION_TYPES:
        try:
            return_value = json.loads(json.dumps(return_value))
        except TypeError as ex:
            raise SupersetTemplateException(
                _("Unsupported return value for method %(name)s", name=func.__name__)
            ) from ex
    return return_value


def validate_context_types(context: Dict[str, Any]) -> Dict[str, Any]:
    for key in context:
        arg_type = type(context[key]).__name__
        if arg_type not in ALLOWED_TYPES and key not in context_addons():
            if arg_type == "partial" and context[key].func.__name__ == "safe_proxy":
                continue
            raise SupersetTemplateException(
                _(
                    "Unsafe template value for key %(key)s: %(value_type)s",
                    key=key,
                    value_type=arg_type,
                )
            )
        if arg_type in COLLECTION_TYPES:
            try:
                context[key] = json.loads(json.dumps(context[key]))
            except TypeError as ex:
                raise SupersetTemplateException(
                    _("Unsupported template value for key %(key)s", key=key)
                ) from ex
    return context


def validate_template_context(engine: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
    if engine and engine in context:
        engine_context = validate_context_types(context.pop(engine))
        valid_context = validate_context_types(context)
        valid_context[engine] = engine_context
        return valid_context
    return validate_context_types(context)


class WhereInMacro:
    def __init__(self, dialect: Dialect) -> None:
        self.dialect: Dialect = dialect

    def __call__(self, values: Sequence[Any], mark: Optional[str] = None) -> str:
        """
        Given a list of values, build a parenthesis list suitable for an IN expression.
        """
        binds = [bindparam(f"value_{i}", value) for i, value in enumerate(values)]
        string_representations = [
            str(bind.compile(dialect=self.dialect, compile_kwargs={"literal_binds": True}))
            for bind in binds
        ]
        joined_values = ", ".join(string_representations)
        result = f"({joined_values})"
        if mark:
            result += (
                "\n-- WARNING: the `mark` parameter was removed from the `where_in` macro for security reasons\n"
            )
        return result


class BaseTemplateProcessor:
    """
    Base class for database-specific jinja context
    """

    engine: ClassVar[Optional[str]] = None

    def __init__(
        self,
        database: Database,
        query: Optional[Query] = None,
        table: Optional[SqlaTable] = None,
        extra_cache_keys: Optional[List[Any]] = None,
        removed_filters: Optional[List[str]] = None,
        applied_filters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._database: Database = database
        self._query: Optional[Query] = query
        self._schema: Optional[str] = None
        if query and query.schema:
            self._schema = query.schema
        elif table:
            self._schema = table.schema
        self._table: Optional[SqlaTable] = table
        self._extra_cache_keys: Optional[List[Any]] = extra_cache_keys
        self._applied_filters: Optional[List[str]] = applied_filters
        self._removed_filters: Optional[List[str]] = removed_filters
        self._context: Dict[str, Any] = {}
        self.env: SandboxedEnvironment = SandboxedEnvironment(undefined=DebugUndefined)
        self.set_context(**kwargs)
        self.env.filters["where_in"] = WhereInMacro(database.get_dialect())

    def set_context(self, **kwargs: Any) -> None:
        self._context.update(kwargs)
        self._context.update(context_addons())

    def process_template(self, sql: str, **kwargs: Any) -> str:
        """Processes a sql template"""
        template = self.env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        return template.render(context)


class JinjaTemplateProcessor(BaseTemplateProcessor):
    def _parse_datetime(self, dttm: str) -> Optional[datetime]:
        """
        Try to parse a datetime and default to None in the worst case.
        """
        try:
            return dateutil.parser.parse(dttm)
        except dateutil.parser.ParserError:
            return None

    def set_context(self, **kwargs: Any) -> None:
        super().set_context(**kwargs)
        extra_cache = ExtraCache(
            extra_cache_keys=self._extra_cache_keys,
            applied_filters=self._applied_filters,
            removed_filters=self._removed_filters,
            database=self._database,
            dialect=self._database.get_dialect(),
            table=self._table,
        )
        from_dttm = self._parse_datetime(dttm) if (dttm := self._context.get("from_dttm")) else None
        to_dttm = self._parse_datetime(dttm) if (dttm := self._context.get("to_dttm")) else None
        dataset_macro_with_context = partial(dataset_macro, from_dttm=from_dttm, to_dttm=to_dttm)
        self._context.update(
            {
                "url_param": partial(safe_proxy, extra_cache.url_param),
                "current_user_id": partial(safe_proxy, extra_cache.current_user_id),
                "current_username": partial(safe_proxy, extra_cache.current_username),
                "current_user_email": partial(safe_proxy, extra_cache.current_user_email),
                "cache_key_wrapper": partial(safe_proxy, extra_cache.cache_key_wrapper),
                "filter_values": partial(safe_proxy, extra_cache.filter_values),
                "get_filters": partial(safe_proxy, extra_cache.get_filters),
                "dataset": partial(safe_proxy, dataset_macro_with_context),
                "metric": partial(safe_proxy, metric_macro),
                "get_time_filter": partial(safe_proxy, extra_cache.get_time_filter),
            }
        )


class NoOpTemplateProcessor(BaseTemplateProcessor):
    def process_template(self, sql: Any, **kwargs: Any) -> str:
        """
        Makes processing a template a noop
        """
        return str(sql)


class PrestoTemplateProcessor(JinjaTemplateProcessor):
    """Presto Jinja context

    The methods described here are namespaced under ``presto`` in the
    jinja context as in ``SELECT '{{ presto.some_macro_call() }}'``
    """

    engine: ClassVar[str] = "presto"

    def set_context(self, **kwargs: Any) -> None:
        super().set_context(**kwargs)
        self._context[self.engine] = {
            "first_latest_partition": partial(safe_proxy, self.first_latest_partition),
            "latest_partitions": partial(safe_proxy, self.latest_partitions),
            "latest_sub_partition": partial(safe_proxy, self.latest_sub_partition),
            "latest_partition": partial(safe_proxy, self.latest_partition),
        }

    @staticmethod
    def _schema_table(table_name: str, schema: Optional[str]) -> Tuple[str, Optional[str]]:
        if "." in table_name:
            schema, table_name = table_name.split(".")
        return (table_name, schema)

    def first_latest_partition(self, table_name: str) -> Optional[str]:
        """
        Gets the first value in the array of all latest partitions
        """
        latest_partitions = self.latest_partitions(table_name)
        return latest_partitions[0] if latest_partitions else None

    def latest_partitions(self, table_name: str) -> List[str]:
        """
        Gets the array of all latest partitions
        """
        from superset.db_engine_specs.presto import PrestoEngineSpec

        table_name, schema = self._schema_table(table_name, self._schema)
        return cast(
            PrestoEngineSpec, self._database.db_engine_spec
        ).latest_partition(database=self._database, table=Table(table_name, schema))[1]

    def latest_sub_partition(self, table_name: str, **kwargs: Any) -> Any:
        table_name, schema = self._schema_table(table_name, self._schema)
        from superset.db_engine_specs.presto import PrestoEngineSpec

        return cast(
            PrestoEngineSpec, self._database.db_engine_spec
        ).latest_sub_partition(database=self._database, table=Table(table_name, schema), **kwargs)

    latest_partition = first_latest_partition


class HiveTemplateProcessor(PrestoTemplateProcessor):
    engine: ClassVar[str] = "hive"


class SparkTemplateProcessor(HiveTemplateProcessor):
    engine: ClassVar[str] = "spark"

    def process_template(self, sql: str, **kwargs: Any) -> str:
        template = self.env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        context["hive"] = context["spark"]
        return template.render(context)


class TrinoTemplateProcessor(PrestoTemplateProcessor):
    engine: ClassVar[str] = "trino"

    def process_template(self, sql: str, **kwargs: Any) -> str:
        template = self.env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        context["presto"] = context["trino"]
        return template.render(context)


DEFAULT_PROCESSORS: Dict[str, Type[BaseTemplateProcessor]] = {
    "presto": PrestoTemplateProcessor,
    "hive": HiveTemplateProcessor,
    "spark": SparkTemplateProcessor,
    "trino": TrinoTemplateProcessor,
}


@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def get_template_processors() -> Dict[str, Type[BaseTemplateProcessor]]:
    processors: Dict[str, Type[BaseTemplateProcessor]] = current_app.config.get(
        "CUSTOM_TEMPLATE_PROCESSORS", {}
    )
    for engine, processor in DEFAULT_PROCESSORS.items():
        if engine not in processors:
            processors[engine] = processor
    return processors


def get_template_processor(
    database: Database,
    table: Optional[SqlaTable] = None,
    query: Optional[Query] = None,
    **kwargs: Any,
) -> BaseTemplateProcessor:
    if feature_flag_manager.is_feature_enabled("ENABLE_TEMPLATE_PROCESSING"):
        template_processor: Type[BaseTemplateProcessor] = get_template_processors().get(
            database.backend, JinjaTemplateProcessor
        )
    else:
        template_processor = NoOpTemplateProcessor
    return template_processor(database=database, table=table, query=query, **kwargs)


def dataset_macro(
    dataset_id: Union[int, str],
    include_metrics: bool = False,
    columns: Optional[List[str]] = None,
    from_dttm: Optional[datetime] = None,
    to_dttm: Optional[datetime] = None,
) -> str:
    """
    Given a dataset ID, return the SQL that represents it.
    """
    from superset.daos.dataset import DatasetDAO

    dataset = DatasetDAO.find_by_id(dataset_id)
    if not dataset:
        raise DatasetNotFoundError(f"Dataset {dataset_id} not found!")
    columns = columns or [column.column_name for column in dataset.columns]
    metrics = [metric.metric_name for metric in dataset.metrics]
    query_obj: Dict[str, Any] = {
        "is_timeseries": False,
        "filter": [],
        "metrics": metrics if include_metrics else None,
        "columns": columns,
        "from_dttm": from_dttm,
        "to_dttm": to_dttm,
    }
    sqla_query = dataset.get_query_str_extended(query_obj, mutate=False)
    sql: str = sqla_query.sql
    return f"(\n{sql}\n) AS dataset_{dataset_id}"


def get_dataset_id_from_context(metric_key: str) -> Union[int, str]:
    """
    Retrieves the Dataset ID from the request context.

    :param metric_key: the metric key.
    :returns: the dataset ID.
    """
    from superset.daos.chart import ChartDAO
    from superset.views.utils import loads_request_json

    form_data: Dict[str, Any] = {}
    exc_message = _(
        "Please specify the Dataset ID for the ``%(name)s`` metric in the Jinja macro.",
        name=metric_key,
    )
    if has_request_context():
        payload: Optional[Dict[str, Any]] = request.get_json(cache=True) if request.is_json else None
        if payload:
            datasource = payload.get("datasource", {})
            dataset_id = datasource.get("id")
            if dataset_id:
                return dataset_id
            form_data.update(payload.get("form_data", {}))
        request_form = loads_request_json(request.form.get("form_data"))
        form_data.update(request_form)
        request_args = loads_request_json(request.args.get("form_data"))
        form_data.update(request_args)
    incoming_form_data: Dict[str, Any] = form_data or getattr(g, "form_data", {})
    if incoming_form_data:
        if (datasource_info := incoming_form_data.get("datasource")):
            if isinstance(datasource_info, dict):
                return datasource_info["id"]
            return datasource_info.split("__")[0]
        url_params = incoming_form_data.get("queries", [{}])[0].get("url_params", {})
        if (dataset_id := url_params.get("datasource_id")):
            return dataset_id
        chart_id = incoming_form_data.get("slice_id") or url_params.get("slice_id")
        if chart_id:
            chart_data = ChartDAO.find_by_id(chart_id)
            if not chart_data:
                raise SupersetTemplateException(exc_message)
            return chart_data.datasource_id
    raise SupersetTemplateException(exc_message)


def has_metric_macro(template_string: str, env: Environment) -> bool:
    """
    Checks if a template string contains a metric macro.
    """
    ast = env.parse(template_string)

    def visit_node(node: Node) -> bool:
        return (
            isinstance(node, Call)
            and isinstance(node.node, nodes.Name)
            and (node.node.name == "metric")
        ) or any((visit_node(child) for child in node.iter_child_nodes()))

    return visit_node(ast)


def metric_macro(metric_key: str, dataset_id: Optional[Union[int, str]] = None) -> str:
    """
    Given a metric key, returns its syntax.
    """
    from superset.daos.dataset import DatasetDAO

    if not dataset_id:
        dataset_id = get_dataset_id_from_context(metric_key)
    dataset = DatasetDAO.find_by_id(dataset_id)
    if not dataset:
        raise DatasetNotFoundError(f"Dataset ID {dataset_id} not found.")
    metrics: Dict[str, str] = {metric.metric_name: metric.expression for metric in dataset.metrics}
    if metric_key not in metrics:
        raise SupersetTemplateException(
            _(
                "Metric ``%(metric_name)s`` not found in %(dataset_name)s.",
                metric_name=metric_key,
                dataset_name=dataset.table_name,
            )
        )
    definition: str = metrics[metric_key]
    env = SandboxedEnvironment(undefined=DebugUndefined)
    context: Dict[str, Any] = {"metric": partial(safe_proxy, metric_macro)}
    while has_metric_macro(definition, env):
        old_definition = definition
        template = env.from_string(definition)
        try:
            definition = template.render(context)
        except RecursionError as ex:
            raise SupersetTemplateException("Cyclic metric macro detected") from ex
        if definition == old_definition:
            break
    return definition