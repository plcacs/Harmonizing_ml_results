```python
import datetime
from collections.abc import Iterable
from typing import Any, List, Tuple, Optional, Union
from django.db.models import Model
from django.utils.timezone import datetime as datetime_with_tz

MESSAGE_BATCH_SIZE: int = ...
STREAM_MESSAGE_BATCH_SIZE: int = ...
TRANSACTION_DELETION_BATCH_SIZE: int = ...
models_with_message_key: List[dict] = ...
EXCLUDE_FIELDS: Any = ...

def move_rows(
    base_model: Any,
    raw_query: Any,
    *,
    src_db_table: Optional[str] = None,
    returning_id: bool = False,
    **kwargs: Any
) -> List[Any]: ...

def run_archiving(
    query: Any,
    type: Any,
    realm: Optional[Any] = None,
    chunk_size: Optional[int] = MESSAGE_BATCH_SIZE,
    **kwargs: Any
) -> int: ...

def move_expired_messages_to_archive_by_recipient(
    recipient: Any,
    message_retention_days: int,
    realm: Any,
    chunk_size: int = MESSAGE_BATCH_SIZE
) -> int: ...

def move_expired_direct_messages_to_archive(
    realm: Any,
    chunk_size: int = MESSAGE_BATCH_SIZE
) -> int: ...

def move_models_with_message_key_to_archive(msg_ids: List[Any]) -> None: ...

def move_attachments_to_archive(msg_ids: List[Any]) -> None: ...

def move_attachment_messages_to_archive(msg_ids: List[Any]) -> None: ...

def delete_messages(msg_ids: List[Any]) -> None: ...

def delete_expired_attachments(realm: Any) -> None: ...

def move_related_objects_to_archive(msg_ids: List[Any]) -> None: ...

def archive_messages_by_recipient(
    recipient: Any,
    message_retention_days: int,
    realm: Any,
    chunk_size: int = MESSAGE_BATCH_SIZE
) -> int: ...

def archive_direct_messages(
    realm: Any,
    chunk_size: int = MESSAGE_BATCH_SIZE
) -> None: ...

def archive_stream_messages(
    realm: Any,
    streams: List[Any],
    chunk_size: int = STREAM_MESSAGE_BATCH_SIZE
) -> None: ...

def archive_messages(chunk_size: int = MESSAGE_BATCH_SIZE) -> None: ...

def get_realms_and_streams_for_archiving() -> List[Tuple[Any, List[Any]]]: ...

def move_messages_to_archive(
    message_ids: List[Any],
    realm: Optional[Any] = None,
    chunk_size: int = MESSAGE_BATCH_SIZE
) -> None: ...

def restore_messages_from_archive(archive_transaction_id: Any) -> List[Any]: ...

def restore_models_with_message_key_from_archive(archive_transaction_id: Any) -> None: ...

def restore_attachments_from_archive(archive_transaction_id: Any) -> None: ...

def restore_attachment_messages_from_archive(archive_transaction_id: Any) -> None: ...

def restore_data_from_archive(archive_transaction: Any) -> int: ...

def restore_data_from_archive_by_transactions(archive_transactions: List[Any]) -> int: ...

def restore_data_from_archive_by_realm(realm: Any) -> None: ...

def restore_all_data_from_archive(restore_manual_transactions: bool = True) -> None: ...

def restore_retention_policy_deletions_for_stream(stream: Any) -> None: ...

def clean_archived_data() -> None: ...

def parse_message_retention_days(
    value: Union[str, int],
    special_values_map: dict
) -> int: ...
```