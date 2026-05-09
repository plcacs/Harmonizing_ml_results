import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from flask import Flask
from flask_babel import lazy_gettext as _
from sqlalchemy import text, TypeDecorator
from sqlalchemy.engine import Connection, Dialect, Row
from sqlalchemy_utils import EncryptedType as SqlaEncryptedType

class EncryptedType(SqlaEncryptedType):
    cache_ok: bool = True

ENC_ADAPTER_TAG_ATTR_NAME: str = '__created_by_enc_field_adapter__'
logger = logging.getLogger(__name__)

class AbstractEncryptedFieldAdapter(ABC):
    @abstractmethod
    def create(self, app_config: Any, *args: Any, **kwargs: Any) -> Any:
        pass

class SQLAlchemyUtilsAdapter(AbstractEncryptedFieldAdapter):
    def create(self, app_config: Any, *args: Any, **kwargs: Any) -> EncryptedType:
        if app_config:
            return EncryptedType(*args, app_config['SECRET_KEY'], **kwargs)
        raise Exception('Missing app_config kwarg')

class EncryptedFieldFactory:

    def __init__(self) -> None:
        self._concrete_type_adapter: Optional[AbstractEncryptedFieldAdapter] = None
        self._config: Optional[Any] = None

    def init_app(self, app: Flask) -> None:
        self._config = app.config
        self._concrete_type_adapter = self._config['SQLALCHEMY_ENCRYPTED_FIELD_TYPE_ADAPTER']()

    def create(self, *args: Any, **kwargs: Any) -> EncryptedType:
        if self._concrete_type_adapter:
            adapter = self._concrete_type_adapter.create(self._config, *args, **kwargs)
            setattr(adapter, ENC_ADAPTER_TAG_ATTR_NAME, True)
            return adapter
        raise Exception('App not initialized yet. Please call init_app first')

    @staticmethod
    def created_by_enc_field_factory(field: Any) -> bool:
        return getattr(field, ENC_ADAPTER_TAG_ATTR_NAME, False)

class SecretsMigrator:

    def __init__(self, previous_secret_key: Any) -> None:
        from superset import db
        self._db: Any = db
        self._previous_secret_key: Any = previous_secret_key
        self._dialect: Dialect = self._db.engine.url.get_dialect()

    def discover_encrypted_fields(self) -> Dict[str, Dict[str, EncryptedType]]:
        """
        Iterates over SqlAlchemy's metadata, looking for EncryptedType
        columns along the way. Builds up a dict of
        table_name -> dict of col_name: enc type instance
        :return:
        """
        meta_info: Dict[str, Dict[str, EncryptedType]] = {}
        for table_name, table in self._db.metadata.tables.items():
            for col_name, col in table.columns.items():
                if isinstance(col.type, EncryptedType):
                    cols = meta_info.get(table_name, {})
                    cols[col_name] = col.type
                    meta_info[table_name] = cols
        return meta_info

    @staticmethod
    def _read_bytes(col_name: str, value: Any) -> bytes:
        if value is None or isinstance(value, bytes):
            return value
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, str):
            return bytes(value.encode('utf8'))
        raise ValueError(_('DB column %(col_name)s has unknown type: %(value_type)s', col_name=col_name, value_type=type(value)))

    @staticmethod
    def _select_columns_from_table(conn: Connection, column_names: List[str], table_name: str) -> Row:
        return conn.execute(f'SELECT id, {','.join(column_names)} FROM {table_name}')

    def _re_encrypt_row(self, conn: Connection, row: Row, table_name: str, columns: Dict[str, EncryptedType]) -> None:
        """
        Re encrypts all columns in a Row
        :param row: Current row to reencrypt
        :param columns: Meta info from columns
        """
        re_encrypted_columns: Dict[str, Any] = {}
        for column_name, encrypted_type in columns.items():
            previous_encrypted_type = EncryptedType(type_in=encrypted_type.underlying_type, key=self._previous_secret_key)
            try:
                unencrypted_value = previous_encrypted_type.process_result_value(self._read_bytes(column_name, row[column_name]), self._dialect)
            except ValueError as ex:
                try:
                    encrypted_type.process_result_value(self._read_bytes(column_name, row[column_name]), self._dialect)
                    logger.info('Current secret is able to decrypt value on column [%s.%s], nothing to do', table_name, column_name)
                    return
                except Exception:
                    raise Exception from ex
            re_encrypted_columns[column_name] = encrypted_type.process_bind_param(unencrypted_value, self._dialect)
        set_cols = ','.join([f'{name} = :{name}' for name in list(re_encrypted_columns.keys())])
        logger.info('Processing table: %s', table_name)
        conn.execute(text(f'UPDATE {table_name} SET {set_cols} WHERE id = :id'), id=row['id'], **re_encrypted_columns)

    def run(self) -> None:
        encrypted_meta_info: Dict[str, Dict[str, EncryptedType]] = self.discover_encrypted_fields()
        with self._db.engine.begin() as conn:
            logger.info('Collecting info for re encryption')
            for table_name, columns in encrypted_meta_info.items():
                column_names: List[str] = list(columns.keys())
                rows: Row = self._select_columns_from_table(conn, column_names, table_name)
                for row in rows:
                    self._re_encrypt_row(conn, row, table_name, columns)
        logger.info('All tables processed')
