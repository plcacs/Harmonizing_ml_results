from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import ANY, Mock
from uuid import UUID
import pytest
from flask import Flask
from pytest_mock import MockerFixture
from sqlalchemy.orm.session import Session
from superset import db
from superset.models.core import Database, SSHTunnel
from superset.utils.log import DBEventLogger
from superset.commands.database.ssh_tunnel.models import SSHTunnel
from superset.databases.api import DatabaseRestApi
from superset.databases.ssh_tunnel.models import SSHTunnel
from superset.utils import json
from superset.sql_parse import Table
from superset.errors import SupersetError
from superset.exceptions import OAuth2RedirectError, SupersetSecurityException
from superset.commands.database.uploaders.base import UploadCommand
from superset.commands.database.uploaders.columnar_reader import ColumnarReader
from superset.commands.database.uploaders.csv_reader import CSVReader
from superset.commands.database.uploaders.excel_reader import ExcelReader
from superset.db_engine_specs.sqlite import SqliteEngineSpec
from superset.models.core import DatabaseUserOAuth2Tokens

def test_filter_by_uuid(session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_post_with_uuid(session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_password_mask(mocker: MockerFixture, app: Flask, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_database_connection(mocker: MockerFixture, app: Flask, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_update_with_password_mask(app: Flask, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_non_zip_import(client: Flask, full_api_access: Any) -> None:
    ...

def test_delete_ssh_tunnel(mocker: MockerFixture, app: Flask, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_delete_ssh_tunnel_not_found(mocker: MockerFixture, app: Flask, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_apply_dynamic_database_filter(mocker: MockerFixture, app: Flask, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_oauth2_happy_path(mocker: MockerFixture, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_oauth2_multiple_tokens(mocker: MockerFixture, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_oauth2_error(mocker: MockerFixture, session: Session, client: Flask, full_api_access: Any) -> None:
    ...

def test_csv_upload(payload: Dict[str, Any], upload_called_with: Tuple[int, str, Any, Optional[Session], Any], reader_called_with: Dict[str, Any], mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_csv_upload_validation(payload: Dict[str, Any], expected_response: Dict[str, Any], mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_csv_upload_file_extension_invalid(filename: str, mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_csv_upload_file_extension_valid(filename: str, mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_excel_upload(payload: Dict[str, Any], upload_called_with: Tuple[int, str, Any, Optional[Session], Any], reader_called_with: Dict[str, Any], mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_excel_upload_validation(payload: Dict[str, Any], expected_response: Dict[str, Any], mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_excel_upload_file_extension_invalid(filename: str, mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_columnar_upload(payload: Dict[str, Any], upload_called_with: Tuple[int, str, Any, Optional[Session], Any], reader_called_with: Dict[str, Any], mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_columnar_upload_validation(payload: Dict[str, Any], expected_response: Dict[str, Any], mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_columnar_upload_file_extension_valid(filename: str, mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_columnar_upload_file_extension_invalid(filename: str, mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_csv_metadata(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_csv_metadata_bad_extension(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_csv_metadata_validation(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_excel_metadata(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_excel_metadata_bad_extension(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_excel_metadata_validation(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_columnar_metadata(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_columnar_metadata_bad_extension(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_columnar_metadata_validation(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_metadata_happy_path(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_metadata_no_table(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_metadata_slashes(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_metadata_invalid_database(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_metadata_unauthorized(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_extra_metadata_happy_path(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_extra_metadata_no_table(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_extra_metadata_slashes(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_extra_metadata_invalid_database(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_table_extra_metadata_unauthorized(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_catalogs(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_catalogs_with_oauth2(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_schemas(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...

def test_schemas_with_oauth2(mocker: MockerFixture, client: Flask, full_api_access: Any) -> None:
    ...