from typing import Optional
from unittest.mock import patch
from freezegun import freeze_time
from superset.models.core import Database, FavStar, SavedQuery
from superset.tags.models import ObjectType, Tag, TaggedObject
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_SQLLAB_USERNAME
from tests.integration_tests.fixtures.importexport import database_config, saved_queries_config, saved_queries_metadata_config
from tests.integration_tests.fixtures.tags import create_custom_tags, get_filter_params

class TestSavedQueryApi(SupersetTestCase):
    def insert_saved_query(self, label: str, sql: str, db_id: Optional[int] = None, created_by: Optional[str] = None, schema: str = '', description: str = '') -> SavedQuery:
        # ...

    @pytest.fixture
    def create_saved_queries(self) -> list[SavedQuery]:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_saved_query_gamma(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_sort_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_schema_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_label_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_sql_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_description_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_distinct_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_create_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_update_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_update_saved_query_not_found(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_saved_query(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_saved_query_not_found(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_bulk_saved_queries(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_one_bulk_saved_queries(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_bulk_saved_query_bad_request(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export_not_found(self) -> None:
        # ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export_not_allowed(self) -> None:
        # ...

    def create_saved_query_import(self) -> bytes:
        # ...

    @patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_saved_queries(self, mock_add_permissions: patch) -> None:
        # ...
