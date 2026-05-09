class TestQueryApi(SupersetTestCase):
    def insert_query(self, database_id: int, user_id: int, client_id: str, sql: str = '', select_sql: str = '', executed_sql: str = '', limit: int = 100, progress: int = 100, rows: int = 100, tab_name: str = '', status: str = 'success', changed_on: datetime = datetime(2020, 1, 1)) -> Query:
        # ... rest of the method ...

    @pytest.fixture
    def create_queries(self) -> List[Query]:
        # ... rest of the method ...

    @staticmethod
    def get_random_string(length: int = 10) -> str:
        # ... rest of the method ...

    def test_get_query(self) -> None:
        # ... rest of the method ...

    def test_get_query_not_found(self) -> None:
        # ... rest of the method ...

    def test_get_query_no_data_access(self) -> None:
        # ... rest of the method ...

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query(self) -> None:
        # ... rest of the method ...

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_sql(self) -> None:
        # ... rest of the method ...

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_database(self) -> None:
        # ... rest of the method ...

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_user(self) -> None:
        # ... rest of the method ...

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_filter_changed_on(self) -> None:
        # ... rest of the method ...

    @pytest.mark.usefixtures('create_queries')
    def test_get_list_query_order(self) -> None:
        # ... rest of the method ...

    def test_get_list_query_no_data_access(self) -> None:
        # ... rest of the method ...

    def test_get_updated_since(self) -> None:
        # ... rest of the method ...

    @mock.patch('superset.sql_lab.cancel_query')
    @mock.patch('superset.views.core.db.session')
    def test_stop_query_not_found(self, mock_superset_db_session: Any, mock_sql_lab_cancel_query: Any) -> None:
        # ... rest of the method ...

    @mock.patch('superset.sql_lab.cancel_query')
    @mock.patch('superset.views.core.db.session')
    def test_stop_query(self, mock_superset_db_session: Any, mock_sql_lab_cancel_query: Any) -> None:
        # ... rest of the method ...
