from typing import Optional
import flask
from superset import db
from superset.result_set import SupersetResultSet
from superset.db_engine_specs.base import BaseEngineSpec
from superset.errors import ErrorLevel, SupersetErrorType
from superset.models.sql_lab import Query
from superset.sql_parse import ParsedQuery, CtasMethod
from superset.utils.core import backend
from superset.utils.database import get_example_database

def get_query_by_id(id: int) -> Optional[Query]:
    ...

def run_sql(test_client, sql: str, cta: bool = False, ctas_method: CtasMethod = CtasMethod.TABLE, tmp_table: str = 'tmp', async_: bool = False):
    ...

def drop_table_if_exists(table_name: str, table_type: str):
    ...

def quote_f(value: str) -> str:
    ...

def cta_result(ctas_method: CtasMethod) -> tuple:
    ...

def get_select_star(table: str, limit: int, schema: Optional[str] = None) -> str:
    ...

def delete_tmp_view_or_table(name: str, db_object_type: str):
    ...

def wait_for_success(result: dict) -> Query:
    ...

def my_task(self) -> bool:
    ...
