from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import string
import pytest
import prison
from sqlalchemy.sql import func
from superset import db, security_manager
from superset.common.db_query_status import QueryStatus
from superset.models.core import Database
from superset.models.sql_lab import Query
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_SQLLAB_USERNAME

QUERIES_FIXTURE_COUNT: int = 10

class TestQueryApi(SupersetTestCase):

    def insert_query(self, database_id: int, user_id: int, client_id: str, sql: str = '', select_sql: str = '', executed_sql: str = '', limit: int = 100, progress: int = 100, rows: int = 100, tab_name: str = '', status: str = 'success', changed_on: datetime = datetime(2020, 1, 1)) -> Query:
    
    def create_queries(self) -> List[Query]:
    
    def get_random_string(length: int = 10) -> str:
    
    def test_get_query(self) -> None:
    
    def test_get_query_not_found(self) -> None:
    
    def test_get_query_no_data_access(self) -> None:
    
    def test_get_list_query(self) -> None:
    
    def test_get_list_query_filter_sql(self) -> None:
    
    def test_get_list_query_filter_database(self) -> None:
    
    def test_get_list_query_filter_user(self) -> None:
    
    def test_get_list_query_filter_changed_on(self) -> None:
    
    def test_get_list_query_order(self) -> None:
    
    def test_get_list_query_no_data_access(self) -> None:
    
    def test_get_updated_since(self) -> None:
    
    def test_stop_query_not_found(self) -> None:
    
    def test_stop_query(self) -> None:
