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
"""Defines the templating context for SQL Lab"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
from typing import Any, Callable, cast, Optional, TYPE_CHECKING, TypedDict, Union, Dict, List, Tuple

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
    convert_legacy_filters_into_adhoc,
    FilterOperator,
    get_user_email,
    get_user_id,
    get_username,
    merge_extra_filters,
)

if TYPE_CHECKING:
    from superset.connectors.sqla.models import SqlaTable
    from superset.models.core import Database
    from superset.models.sql_lab import Query

NONE_TYPE = type(None).__name__
ALLOWED_TYPES = (
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
COLLECTION_TYPES = ("list", "dict", "tuple", "set")


@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def context_addons() -> Dict[str, Any]:
    return current_app.config.get("JINJA_CONTEXT_ADDONS", {})


class Filter(TypedDict):
    op: str  # pylint: disable=C0103
    col: str
    val: Union[None, Any, List[Any]]


@dataclass
class TimeFilter:
    """
    Container for temporal filter.
    """

    from_expr: Optional[str]
    to_expr: Optional[str]
    time_range: Optional[str]


class ExtraCache:
    """
    Dummy class that exposes a method used to store additional values used in
    calculation of query object cache keys.
    """

    # Regular expression for detecting the presence of templated methods which could
    # be added to the cache key.
    regex = re.compile(
        r"(\{\{|\{%)[^{}]*?("
        r"current_user_id\([^()]*\)|"
        r"current_username\([^()]*\)|"
        r"current_user_email\([^()]*\)|"
        r"cache_key_wrapper\([^()]*\)|"
        r"url_param\([^()]*\)"
        r")"
        r"[^{}]*?(\}\}|\%\})"
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        extra_cache_keys: Optional[List[Any]] = None,
        applied_filters: Optional[List[str]] = None,
        removed_filters: Optional[List[str]] = None,
        database: Optional["Database"] = None,
        dialect: Optional[Dialect] = None,
        table: Optional["SqlaTable"] = None,
    ):
        self.extra_cache_keys = extra_cache_keys
        self.applied_filters = applied_filters if applied_filters is not None else []
        self.removed_filters = removed_filters if removed_filters is not None else []
        self.database = database
        self.dialect = dialect
        self.table = table

    def current_user_id(self, add_to_cache_keys: bool = True) -> Optional[int]:
        """
        Return the user ID of the user who is currently logged in.

        :param add_to_cache_keys: Whether the value should be included in the cache key
        :returns: The user ID
        """
        if user_id := get_user_id():
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
        if username := get_username():
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
        if email_address := get_user_email():
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
        default: Optional[str] = None,
        add_to_cache_keys: bool = True,
        escape_result: bool = True,
    ) -> Optional[str]:
        """
        Read a url or post parameter and use it in your SQL Lab query.

        When in SQL Lab, it's possible to add arbitrary URL "query string" parameters,
        and use those in your SQL code. For instance you can alter your url and add
        `?foo=bar`, as in `{domain}/sqllab?foo=bar`. Then if your query is
        something like SELECT * FROM foo = '{{ url_param('foo') }}', it will be parsed
        at runtime and replaced by the value in the URL.

        As you create a visualization form this SQL Lab query, you can pass parameters
        in the explore view as well as from the dashboard, and it should carry through
        to your queries.

        Default values for URL parameters can be defined in chart metadata by adding the
        key-value pair `url_params: {'foo': 'bar'}`

        :param param: the parameter to lookup
        :param default: the value to return in the absence of the parameter
        :param add_to_cache_keys: Whether the value should be included in the cache key
        :param escape_result: Should special characters in the result be escaped
        :returns: The URL parameters
        """
        # pylint: disable=import-outside-toplevel
        from superset.views.utils import get_form_data

        if has_request_context() and request.args.get(param):
            return request.args.get(param, default)

        form_data, _ = get_form_data()
        url_params = form_data.get("url_params") or {}
        result = url_params.get(param, default)
        if result and escape_result and self.dialect:
            # use the dialect specific quoting logic to escape string
            result = String().literal_processor(dialect=self.dialect)(value=result)[
                1:-1
            ]
        if add_to_cache_keys:
            self.cache_key_wrapper(result)
        return result

    def filter_values(
        self, column: str, default: Optional[str] = None, remove_filter: bool = False
    ) -> List[Any]:
        """Gets a values for a particular filter as a list

        This is useful if:
            - you want to use a filter component to filter a query where the name of
             filter component column doesn't match the one in the select statement
            - you want to have the ability for filter inside the main query for speed
            purposes

        Usage example::

            SELECT action, count(*) as times
            FROM logs
            WHERE
                action in ({{ "'" + "','".join(filter_values('action_type')) + "'" }})
            GROUP BY action

        :param column: column/filter name to lookup
        :param default: default value to return if there's no matching columns
        :param remove_filter: When set to true, mark the filter as processed,
            removing it from the outer query. Useful when a filter should
            only apply to the inner query
        :return: returns a list of filter values
        """
        return_val: List[Any] = []
        filters = self.get_filters(column, remove_filter)
        for flt in filters:
            val = flt.get("val")
            if isinstance(val, list):
                return_val.extend(val)
            elif val:
                return_val.append(val)

        if (not return_val) and default:
            # If no values are found, return the default provided.
            return_val = [default]

        return return_val

    def get_filters(self, column: str, remove_filter: bool = False) -> List[Filter]:
        """Get the filters applied to the given column. In addition
           to returning values like the filter_values function
           the get_filters function returns the operator specified in the explorer UI.

        This is useful if:
            - you want to handle more than the IN operator in your SQL clause
            - you want to handle generating custom SQL conditions for a filter
            - you want to have the ability for filter inside the main query for speed
            purposes

        Usage example::


            WITH RECURSIVE
                superiors(employee_id, manager_id, full_name, level, lineage) AS (
                SELECT
                    employee_id,
                    manager_id,
                    full_name,
                1 as level,
                employee_id as lineage
                FROM
                    employees
                WHERE
                1=1
                {# Render a blank line #}
                {%- for filter in get_filters('full_name', remove_filter=True) -%}
                {%- if filter.get('op') == 'IN' -%}
                    AND
                    full_name IN ( {{ "'" + "', '".join(filter.get('val')) + "'" }} )
                {%- endif -%}
                {%- if filter.get('op') == 'LIKE' -%}
                    AND
                    full_name LIKE {{ "'" + filter.get('val') + "'" }}
                {%- endif -%}
                {%- endfor -%}
                UNION ALL
                    SELECT
                        e.employee_id,
                        e.manager_id,
                        e.full_name,
                s.level + 1 as level,
                s.lineage
                    FROM
                        employees e,
                    superiors s
                    WHERE s.manager_id = e.employee_id
            )


            SELECT
                employee_id, manager_id, full_name, level, lineage
            FROM
                superiors
            order by lineage, level

        :param column: column/filter name to lookup
        :param remove_filter: When set to true, mark the filter as processed,
            removing it from the outer query. Useful when a filter should
            only apply to the inner query
        :return: returns a list of filters
        """
        # pylint: disable=import-outside-toplevel
        from superset.views.utils import get_form_data

        form_data, _ = get_form_data()
        convert_legacy_filters_into_adhoc(form_data)
        merge_extra_filters(form_data)

        filters: List[Filter] = []

        for flt in form_data.get("adhoc_filters", []):
            val: Union[Any, List[Any]] = flt.get("comparator")
            op: str = flt["operator"].upper() if flt.get("operator") else None  # type: ignore
            if (
                flt.get("expressionType") == "SIMPLE"
                and flt.get("clause") == "WHERE"
                and flt.get("subject") == column
                and val
            ):
                if remove_filter:
                    if column not in self.removed_filters:
                        self.removed_filters.append(column)
                if column not in self.applied_filters:
                    self.applied_filters.append(column)

                if op in (
                    FilterOperator.IN.value,
                    FilterOperator.NOT_IN.value,
                ) and not isinstance(val, list):
                    val = [val]

                filters.append({"op": op, "col": column, "val": val})

        return filters

    # pylint: disable=too-many-arguments
    def get_time_filter(
        self,
        column: Optional[str] = None,
        default: Optional[str] = None,
        target_type: Optional[str] = None,
        strftime: Optional[str] = None,
        remove_filter: bool = False,
    ) -> TimeFilter:
        """Get the time filter with appropriate formatting,
        either for a specific column, or whichever time range is being emitted
        from a dashboard.

        :param column: Name of the temporal column. Leave undefined to reference the
            time range from a Dashboard Native Time Range filter (when present).
        :param default: The default value to fall back to if the time filter is
            not present, or has the value `No filter`
        :param target_type: The target temporal type as recognized by the target
            database (e.g. `TIMESTAMP`, `DATE` or `DATETIME`). If `column` is defined,
            the format will default to the type of the column. This is used to produce
            the format of the `from_expr` and `to_expr` properties of the returned
            `TimeFilter` object.
        :param strftime: format using the `strftime` method of `datetime`. When defined
            `target_type` will be ignored.
        :param remove_filter: When set to true, mark the filter as processed,
            removing it from the outer query. Useful when a filter should
            only apply to the inner query.
        :return: The corresponding time filter.
        """
        # pylint: disable=import-outside-toplevel
        from superset.views.utils import get_form_data

        form_data, _ = get_form_data()
        convert_legacy_filters_into_adhoc(form_data)
        merge_extra_filters(form_data)
        time_range = form_data.get("time_range")
        if column:
            flt: Optional[AdhocFilterClause] = next(
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
        from_expr, to_expr = get_since_until_from_time_range(time_range)

        def _format_dttm(dttm: Optional[datetime]) -> Optional[str]:
            if strftime and dttm:
                return dttm.strftime(strftime)
            return (
                self.database.db_engine_spec.convert_dttm(target_type or "", dttm)
                if self.database and dttm
                else None
            )

        return TimeFilter(
            from_expr=_format_dttm(from_expr),
            to_expr=_format_dttm(to_expr),
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
                _(
                    "Unsupported return value for method %(name)s",
                    name=func.__name__,
                )
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
                    "Unsafe template value for key %(key)s: %(value