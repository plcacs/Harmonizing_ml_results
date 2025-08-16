from typing import Any, Optional, TypedDict
from werkzeug.datastructures import FileStorage
from superset.commands.base import BaseCommand
from superset.commands.database.exceptions import DatabaseNotFoundError, DatabaseSchemaUploadNotAllowed, DatabaseUploadFailed, DatabaseUploadNotSupported, DatabaseUploadSaveMetadataFailed
from superset.connectors.sqla.models import SqlaTable
from superset.daos.database import DatabaseDAO
from superset.models.core import Database
from superset.sql_parse import Table
from superset.utils.backports import StrEnum
from superset.utils.core import get_user
from superset.utils.decorators import on_error, transaction
from superset.views.database.validators import schema_allows_file_upload

class UploadFileType(StrEnum):
    CSV: str = 'csv'
    EXCEL: str = 'excel'
    COLUMNAR: str = 'columnar'

class ReaderOptions(TypedDict, total=False):
    pass

class FileMetadataItem(TypedDict):
    pass

class FileMetadata(TypedDict, total=False):
    pass

class BaseDataReader:
    def __init__(self, options: Optional[ReaderOptions] = None) -> None:
        self._options = options or {}

    def file_to_dataframe(self, file: FileStorage) -> Any:
        ...

    def file_metadata(self, file: FileStorage) -> Any:
        ...

    def read(self, file: FileStorage, database: Database, table_name: str, schema_name: str) -> None:
        self._dataframe_to_database(self.file_to_dataframe(file), database, table_name, schema_name)

    def _dataframe_to_database(self, df: Any, database: Database, table_name: str, schema_name: str) -> None:
        try:
            data_table = Table(table=table_name, schema=schema_name)
            to_sql_kwargs = {'chunksize': READ_CHUNK_SIZE, 'if_exists': self._options.get('already_exists', 'fail'), 'index': self._options.get('dataframe_index', False)}
            if self._options.get('index_label') and self._options.get('dataframe_index'):
                to_sql_kwargs['index_label'] = self._options.get('index_label')
            database.db_engine_spec.df_to_sql(database, data_table, df, to_sql_kwargs=to_sql_kwargs)
        except ValueError as ex:
            raise DatabaseUploadFailed(message=_("Table already exists. You can change your 'if table already exists' strategy to append or replace or provide a different Table Name to use.")) from ex
        except Exception as ex:
            raise DatabaseUploadFailed(exception=ex) from ex

class UploadCommand(BaseCommand):
    def __init__(self, model_id: int, table_name: str, file: FileStorage, schema: str, reader: BaseDataReader) -> None:
        self._model_id = model_id
        self._model: Optional[Database] = None
        self._table_name = table_name
        self._schema = schema
        self._file = file
        self._reader = reader

    @transaction(on_error=partial(on_error, reraise=DatabaseUploadSaveMetadataFailed))
    def run(self) -> None:
        self.validate()
        if not self._model:
            return
        self._reader.read(self._file, self._model, self._table_name, self._schema)
        sqla_table = db.session.query(SqlaTable).filter_by(table_name=self._table_name, schema=self._schema, database_id=self._model_id).one_or_none()
        if not sqla_table:
            sqla_table = SqlaTable(table_name=self._table_name, database=self._model, database_id=self._model_id, owners=[get_user()], schema=self._schema)
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
