import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from uuid import UUID, uuid3
from flask import current_app, Flask, has_app_context
from flask_caching import BaseCache
from sqlalchemy.exc import SQLAlchemyError
from superset import db
from superset.key_value.exceptions import KeyValueCreateFailedError
from superset.key_value.types import KeyValueCodec, KeyValueResource, PickleKeyValueCodec
from superset.key_value.utils import get_uuid_namespace
from superset.utils.decorators import transaction

RESOURCE = KeyValueResource.METASTORE_CACHE
logger = logging.getLogger(__name__)


class SupersetMetastoreCache(BaseCache):

    def __init__(self, namespace: UUID, codec: KeyValueCodec, default_timeout: int = 300) -> None:
        super().__init__(default_timeout)
        self.namespace: UUID = namespace
        self.codec: KeyValueCodec = codec

    @classmethod
    def factory(cls, app: Flask, config: Dict[str, Any], *args: Any, **kwargs: Any) -> 'SupersetMetastoreCache':
        seed: str = config.get('CACHE_KEY_PREFIX', '')
        kwargs['namespace'] = get_uuid_namespace(seed)
        codec: KeyValueCodec = config.get('CODEC') or PickleKeyValueCodec()
        if has_app_context() and (not current_app.debug) and isinstance(codec, PickleKeyValueCodec):
            logger.warning('Using PickleKeyValueCodec with SupersetMetastoreCache may be unsafe, use at your own risk.')
        kwargs['codec'] = codec
        return cls(*args, **kwargs)

    def get_key(self, key: str) -> UUID:
        return uuid3(self.namespace, key)

    def _get_expiry(self, timeout: Optional[int]) -> Optional[datetime]:
        timeout = self._normalize_timeout(timeout)
        if timeout is not None and timeout > 0:
            return datetime.now() + timedelta(seconds=timeout)
        return None

    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        from superset.daos.key_value import KeyValueDAO
        KeyValueDAO.upsert_entry(
            resource=RESOURCE,
            key=self.get_key(key),
            value=value,
            codec=self.codec,
            expires_on=self._get_expiry(timeout)
        )
        db.session.commit()
        return True

    def add(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        from superset.daos.key_value import KeyValueDAO
        try:
            KeyValueDAO.delete_expired_entries(RESOURCE)
            KeyValueDAO.create_entry(
                resource=RESOURCE,
                value=value,
                codec=self.codec,
                key=self.get_key(key),
                expires_on=self._get_expiry(timeout)
            )
            db.session.commit()
            return True
        except (SQLAlchemyError, KeyValueCreateFailedError):
            db.session.rollback()
            return False

    def get(self, key: str) -> Optional[Any]:
        from superset.daos.key_value import KeyValueDAO
        return KeyValueDAO.get_value(RESOURCE, self.get_key(key), self.codec)

    def has(self, key: str) -> bool:
        entry: Optional[Any] = self.get(key)
        return entry is not None

    @transaction()
    def delete(self, key: str) -> bool:
        from superset.daos.key_value import KeyValueDAO
        return KeyValueDAO.delete_entry(RESOURCE, self.get_key(key))
