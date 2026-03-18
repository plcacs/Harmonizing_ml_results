```python
import datetime
from collections.abc import Iterable
from typing import Any, List, Tuple

from django.db.models import Model
from django.utils.timezone import datetime as DateTime

from zerver.models import (
    ArchivedAttachment,
    ArchivedReaction,
    ArchivedSubMessage,
    ArchivedUserMessage,
    ArchiveTransaction,
    Attachment,
    Message,
    Realm,
    Recipient,
    Stream,
    SubMessage,
    UserMessage,
)

MESSAGE_BATCH_SIZE: int = ...
STREAM_MESSAGE_BATCH_SIZE: int = ...
TRANSACTION_DELETION_BATCH_SIZE: int = ...
models_with_message_key: List[dict] = ...
EXCLUDE_FIELDS: Any = ...

def move_rows(
    base_model: type[Model],
    raw_query: Any,
    *,
    src_db_table: str | None = ...,
    returning_id: bool = ...,
    **kwargs: Any,
) -> List[Any]: ...

def run_archiving(
    query: Any,
    type: Any,
    realm: Realm | None = ...,
    chunk_size: int | None = ...,
    **kwargs: Any,
) -> int: ...

def move_expired_messages_to_archive_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = ...,
) -> int: ...

def move_expired_direct_messages_to_archive(
    realm: Realm,
    chunk_size: int = ...,
) -> int: ...

def move_models_with_message_key_to_archive(msg_ids: Iterable[int]) -> None: ...

def move_attachments_to_archive(msg_ids: Iterable[int]) -> None: ...

def move_attachment_messages_to_archive(msg_ids: Iterable[int]) -> None: ...

def delete_messages(msg_ids: Iterable[int]) -> None: ...

def delete_expired_attachments(realm: Realm) -> None: ...

def move_related_objects_to_archive(msg_ids: Iterable[int]) -> None: ...

def archive_messages_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = ...,
) -> int: ...

def archive_direct_messages(realm: Realm, chunk_size: int = ...) -> None: ...

def archive_stream_messages(
    realm: Realm,
    streams: Iterable[Stream],
    chunk_size: int = ...,
) -> None: ...

def archive_messages(chunk_size: int = ...) -> None: ...

def get_realms_and_streams_for_archiving() -> List[Tuple[Realm, List[Stream]]]: ...

def move_messages_to_archive(
    message_ids: List[int],
    realm: Realm | None = ...,
    chunk_size: int = ...,
) -> None: ...

def restore_messages_from_archive(archive_transaction_id: int) -> List[int]: ...

def restore_models_with_message_key_from_archive(archive_transaction_id: int) -> None: ...

def restore_attachments_from_archive(archive_transaction_id: int) -> None: ...

def restore_attachment_messages_from_archive(archive_transaction_id: int) -> None: ...

def restore_data_from_archive(archive_transaction: ArchiveTransaction) -> int: ...

def restore_data_from_archive_by_transactions(
    archive_transactions: Iterable[ArchiveTransaction],
) -> int: ...

def restore_data_from_archive_by_realm(realm: Realm) -> None: ...

def restore_all_data_from_archive(restore_manual_transactions: bool = ...) -> None: ...

def restore_retention_policy_deletions_for_stream(stream: Stream) -> None: ...

def clean_archived_data() -> None: ...

def parse_message_retention_days(
    value: str | int,
    special_values_map: dict[str, int],
) -> int: ...
```