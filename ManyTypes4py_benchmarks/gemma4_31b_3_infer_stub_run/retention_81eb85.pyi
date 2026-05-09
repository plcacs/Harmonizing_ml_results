import logging
from collections.abc import Iterable
from typing import Any, Optional, Sequence, Union, overload
from psycopg2.sql import SQL, Identifier
from zerver.models import (
    ArchivedAttachment,
    ArchivedReaction,
    ArchivedSubMessage,
    ArchivedUserMessage,
    ArchiveTransaction,
    Attachment,
    Message,
    Reaction,
    Realm,
    Recipient,
    Stream,
    SubMessage,
    UserMessage,
)

logger: logging.Logger ...
MESSAGE_BATCH_SIZE: int ...
STREAM_MESSAGE_BATCH_SIZE: int ...
TRANSACTION_DELETION_BATCH_SIZE: int ...
models_with_message_key: list[dict[str, Any]] ...
EXCLUDE_FIELDS: set[Any] ...

def move_rows(
    base_model: Any,
    raw_query: SQL,
    *,
    src_db_table: Optional[str] = None,
    returning_id: bool = False,
    **kwargs: Any,
) -> list[int]: ...

def run_archiving(
    query: SQL,
    type: str,
    realm: Optional[Realm] = None,
    chunk_size: Optional[int] = MESSAGE_BATCH_SIZE,
    **kwargs: Any,
) -> int: ...

def move_expired_messages_to_archive_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = MESSAGE_BATCH_SIZE,
) -> int: ...

def move_expired_direct_messages_to_archive(
    realm: Realm,
    chunk_size: int = MESSAGE_BATCH_SIZE,
) -> int: ...

def move_models_with_message_key_to_archive(msg_ids: Sequence[int]) -> None: ...

def move_attachments_to_archive(msg_ids: Sequence[int]) -> None: ...

def move_attachment_messages_to_archive(msg_ids: Sequence[int]) -> None: ...

def delete_messages(msg_ids: Sequence[int]) -> None: ...

def delete_expired_attachments(realm: Realm) -> None: ...

def move_related_objects_to_archive(msg_ids: Sequence[int]) -> None: ...

def archive_messages_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = MESSAGE_BATCH_SIZE,
) -> int: ...

def archive_direct_messages(realm: Realm, chunk_size: int = MESSAGE_BATCH_SIZE) -> None: ...

def archive_stream_messages(
    realm: Realm,
    streams: Sequence[Stream],
    chunk_size: int = STREAM_MESSAGE_BATCH_SIZE,
) -> None: ...

def archive_messages(chunk_size: int = MESSAGE_BATCH_SIZE) -> None: ...

def get_realms_and_streams_for_archiving() -> list[tuple[Realm, list[Stream]]]: ...

def move_messages_to_archive(
    message_ids: Sequence[int],
    realm: Optional[Realm] = None,
    chunk_size: int = MESSAGE_BATCH_SIZE,
) -> None: ...

def restore_messages_from_archive(archive_transaction_id: int) -> list[int]: ...

def restore_models_with_message_key_from_archive(archive_transaction_id: int) -> None: ...

def restore_attachments_from_archive(archive_transaction_id: int) -> None: ...

def restore_attachment_messages_from_archive(archive_transaction_id: int) -> None: ...

def restore_data_from_archive(archive_transaction: ArchiveTransaction) -> int: ...

def restore_data_from_archive_by_transactions(
    archive_transactions: Iterable[ArchiveTransaction],
) -> int: ...

def restore_data_from_archive_by_realm(realm: Realm) -> None: ...

def restore_all_data_from_archive(restore_manual_transactions: bool = True) -> None: ...

def restore_retention_policy_deletions_for_stream(stream: Stream) -> None: ...

def clean_archived_data() -> None: ...

def parse_message_retention_days(
    value: Union[str, int],
    special_values_map: dict[str, int],
) -> int: ...