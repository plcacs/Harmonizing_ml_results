from typing import Any, Dict, List
import pytest
from freezegun import freeze_time
from superset import app
from superset.connectors.sqla.models import SqlaTable, TableColumn
from superset.jinja_context import ExtraCache, TimeFilter, WhereInMacro
from superset.utils import json

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
    """
    Test the ``get_time_filter`` macro.
    """
    columns = [TableColumn(column_name='dt', is_dttm=1, type='DATE'), TableColumn(column_name='dttm', is_dttm=1, type='TIMESTAMP')]
    database = Database(database_name='my_database', sqlalchemy_uri=sqlalchemy_uri)
    table = SqlaTable(table_name='my_dataset', columns=columns, main_dttm_col='dt', database=database)
    with freeze_time('2024-09-03'), app.test_request_context(json={'queries': queries}):
        cache = ExtraCache(database=database, table=table)
        assert cache.get_time_filter(*args, **kwargs) == time_filter, description
        assert cache.removed_filters == removed_filters
        assert cache.applied_filters == applied_filters
