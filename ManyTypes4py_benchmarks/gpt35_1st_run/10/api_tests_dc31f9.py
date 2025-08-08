from datetime import datetime
from io import BytesIO
from typing import Optional
from unittest.mock import patch
from zipfile import is_zipfile, ZipFile

import yaml
import pytest
import prison
from freezegun import freeze_time
from sqlalchemy.sql import func, and_

from superset import db
from superset.models.core import Database, FavStar
from superset.models.sql_lab import SavedQuery
from superset.tags.models import ObjectType, Tag, TaggedObject
from superset.utils.database import get_example_database
from superset.utils import json
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.constants import ADMIN_USERNAME, GAMMA_SQLLAB_USERNAME
from tests.integration_tests.fixtures.importexport import database_config, saved_queries_config, saved_queries_metadata_config

SAVED_QUERIES_FIXTURE_COUNT: int = 10

class TestSavedQueryApi(SupersetTestCase):

    def insert_saved_query(self, label: str, sql: str, db_id: Optional[int] = None, created_by: Optional[str] = None, schema: str = '', description: str = '') -> SavedQuery:
        ...

    def insert_default_saved_query(self, label: str = 'saved1', schema: str = 'schema1', username: str = 'admin') -> SavedQuery:
        ...

    @pytest.fixture
    def create_saved_queries(self) -> list[SavedQuery]:
        ...

    @pytest.fixture
    def create_saved_queries_some_with_tags(self, create_custom_tags) -> list[SavedQuery]:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_saved_query_gamma(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_sort_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_database_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_filter_schema_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_schema_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_label_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_sql_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_list_custom_filter_description_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries_some_with_tags')
    def test_get_saved_queries_tag_filters(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_saved_query_favorite_filter(self) -> None:
        ...

    def test_info_saved_query(self) -> None:
        ...

    def test_info_security_saved_query(self) -> None:
        ...

    def test_related_saved_query(self) -> None:
        ...

    def test_related_saved_query_not_found(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_distinct_saved_query(self) -> None:
        ...

    def test_get_saved_query_not_allowed(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_get_saved_query(self) -> None:
        ...

    def test_get_saved_query_not_found(self) -> None:
        ...

    def test_create_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_update_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_update_saved_query_not_found(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_saved_query(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_saved_query_not_found(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_bulk_saved_queries(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_one_bulk_saved_queries(self) -> None:
        ...

    def test_delete_bulk_saved_query_bad_request(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_delete_bulk_saved_query_not_found(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export_not_found(self) -> None:
        ...

    @pytest.mark.usefixtures('create_saved_queries')
    def test_export_not_allowed(self) -> None:
        ...

    def create_saved_query_import(self) -> BytesIO:
        ...

    @patch('superset.commands.database.importers.v1.utils.add_permissions')
    def test_import_saved_queries(self, mock_add_permissions) -> None:
        ...

