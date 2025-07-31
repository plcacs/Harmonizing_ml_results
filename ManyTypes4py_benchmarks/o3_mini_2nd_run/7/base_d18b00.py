import logging
from abc import abstractmethod
from functools import partial
from typing import Any, Dict, Optional
import pandas as pd
from flask_babel import lazy_gettext as _
from werkzeug.datastructures import FileStorage
from superset import db
from superset.commands.base import BaseCommand
from superset.commands.database.exceptions import (
    DatabaseNotFoundError,
    DatabaseSchemaUploadNotAllowed,
    DatabaseUploadFailed,
    DatabaseUploadNotSupported,
    DatabaseUploadSaveMetadataFailed,
)
from superset.connectors.sqla.models import SqlaTable
from superset.daos.database import DatabaseDAO
from superset.models.core import Database
from superset.sql_parse import Table
from superset.utils.backports import StrEnum
from superset.utils.core import get_user
from superset.utils.decorators import on_error, transaction
from superset.views.database.validators import schema_allows_file_upload

logger = logging.getLogger(__name__)
READ_CHUNK_SIZE: int = 1000

class UploadFileType(StrEnum):
    CSV = 'csv'
    EXCEL = 'excel'
    COLUMNAR = 'columnar'

class ReaderOptions(Dict[str, Any]):
    ...

class FileMetadataItem(Dict[str, Any]):
    ...

class FileMetadata(Dict[str, Any]):
    ...

class BaseDataReader:
    """
    Base class for reading data from a file and uploading it to a database.
    These child objects are used by the UploadCommand as a dependency injection
    to read data from multiple file types (e.g. CSV, Excel, etc.)
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self._options: Dict[str, Any] = options or {}

    @abstractmethod
    def file_to_dataframe(self, file: FileStorage) -> pd.DataFrame:
        ...

    @abstractmethod
    def file_metadata(self, file: FileStorage) -> FileMetadata:
        ...

    def read(
        self, file: FileStorage, database: Database, table_name: str, schema_name: Optional[str]
    ) -> None:
        dataframe: pd.DataFrame = self.file_to_dataframe(file)
        self._dataframe_to_database(dataframe, database, table_name, schema_name)

    def _dataframe_to_database(
        self, df: pd.DataFrame, database: Database, table_name: str, schema_name: Optional[str]
    ) -> None:
        """
        Upload DataFrame to database

        :param df: DataFrame to upload
        :throws DatabaseUploadFailed: if there is an error uploading the DataFrame
        """
        try:
            data_table = Table(table=table_name, schema=schema_name)
            to_sql_kwargs: Dict[str, Any] = {
                'chunksize': READ_CHUNK_SIZE,
                'if_exists': self._options.get('already_exists', 'fail'),
                'index': self._options.get('dataframe_index', False),
            }
            if self._options.get('index_label') and self._options.get('dataframe_index'):
                to_sql_kwargs['index_label'] = self._options.get('index_label')
            database.db_engine_spec.df_to_sql(database, data_table, df, to_sql_kwargs=to_sql_kwargs)
        except ValueError as ex:
            raise DatabaseUploadFailed(
                message=_(
                    "Table already exists. You can change your 'if table already exists' strategy to append or replace or provide a different Table Name to use."
                )
            ) from ex
        except Exception as ex:
            raise DatabaseUploadFailed(exception=ex) from ex

class UploadCommand(BaseCommand):
    def __init__(
        self,
        model_id: int,
        table_name: str,
        file: FileStorage,
        schema: Optional[str],
        reader: BaseDataReader,
    ) -> None:
        self._model_id: int = model_id
        self._model: Optional[Database] = None
        self._table_name: str = table_name
        self._schema: Optional[str] = schema
        self._file: FileStorage = file
        self._reader: BaseDataReader = reader

    @transaction(on_error=partial(on_error, reraise=DatabaseUploadSaveMetadataFailed))
    def run(self) -> None:
        self.validate()
        if not self._model:
            return
        self._reader.read(self._file, self._model, self._table_name, self._schema)
        sqla_table: Optional[SqlaTable] = (
            db.session.query(SqlaTable)
            .filter_by(table_name=self._table_name, schema=self._schema, database_id=self._model_id)
            .one_or_none()
        )
        if not sqla_table:
            sqla_table = SqlaTable(
                table_name=self._table_name,
                database=self._model,
                database_id=self._model_id,
                owners=[get_user()],
                schema=self._schema,
            )
            db.session.add(sqla_table)
        sqla_table.fetch_metadata()

    def validate(self) -> None:
        self._model = DatabaseDAO.find_by_id(self._model_id)
        if not self._model:
            raise DatabaseNotFoundError()
        if not schema_allows_file_upload(self._model, self._schema):
            raise DatabaseSchemaUploadNotAllowed()
        if not self._model.db_engine_spec.supports_file_upload:
            raise DatabaseUploadNotSupported()