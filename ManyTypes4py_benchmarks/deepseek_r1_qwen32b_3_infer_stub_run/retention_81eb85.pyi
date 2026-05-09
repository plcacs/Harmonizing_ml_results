import logging
import time
from collections.abc import Iterable, Mapping
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from django.db import transaction
from django.db.models import Model
from django.utils.timezone import datetime
from psycopg2.sql import Composable, Identifier, Literal
from zerver.lib.request import RequestVariableConversionError
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

logger: logging.Logger = ...

MESSAGE_BATCH_SIZE: int = ...
STREAM_MESSAGE_BATCH_SIZE: int = ...
TRANSACTION_DELETION_BATCH_SIZE: int = ...

models_with_message_key: List[Dict[str, Any]] = ...

EXCLUDE_FIELDS: Iterable[Any] = ...


@transaction.atomic(savepoint=False)
def move_rows(
    base_model: type[Model],
    raw_query: Composable,
    src_db_table: Optional[str] = None,
    returning_id: bool = False,
    **kwargs: Any,
) -> List[int]:
    ...


def run_archiving(
    query: Composable,
    type: Literal["manual", "retention_policy_based"],
    realm: Optional[Realm] = None,
    chunk_size: Optional[int] = None,
    **kwargs: Any,
) -> int:
    ...


def move_expired_messages_to_archive_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = ...,
) -> int:
    ...


def move_expired_direct_messages_to_archive(
    realm: Realm,
    chunk_size: int = ...,
) -> int:
    ...


def move_models_with_message_key_to_archive(msg_ids: List[int]) -> None:
    ...


def move_attachments_to_archive(msg_ids: List[int]) -> None:
    ...


def move_attachment_messages_to_archive(msg_ids: List[int]) -> None:
    ...


def delete_messages(msg_ids: List[int]) -> None:
    ...


def delete_expired_attachments(realm: Realm) -> Tuple[int, int]:
    ...


def move_related_objects_to_archive(msg_ids: List[int]) -> None:
    ...


def archive_messages_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = ...,
) -> int:
    ...


def archive_direct_messages(realm: Realm, chunk_size: int = ...) -> int:
    ...


def archive_stream_messages(
    realm: Realm,
    streams: Iterable[Stream],
    chunk_size: int = ...,
) -> int:
    ...


@transaction.atomic(durable=True)
def archive_messages(chunk_size: int = ...) -> None:
    ...


def get_realms_and_streams_for_archiving() -> List[Tuple[Realm, List[Stream]]]:
    ...


def move_messages_to_archive(
    message_ids: List[int],
    realm: Optional[Realm] = None,
    chunk_size: int = ...,
) -> int:
    ...


def restore_messages_from_archive(archive_transaction_id: int) -> List[int]:
    ...


def restore_models_with_message_key_from_archive(archive_transaction_id: int) -> None:
    ...


def restore_attachments_from_archive(archive_transaction_id: int) -> None:
    ...


def restore_attachment_messages_from_archive(archive_transaction_id: int) -> None:
    ...


def restore_data_from_archive(archive_transaction: ArchiveTransaction) -> int:
    ...


def restore_data_from_archive_by_transactions(
    archive_transactions: Iterable[ArchiveTransaction],
) -> int:
    ...


def restore_data_from_archive_by_realm(realm: Realm) -> int:
    ...


def restore_all_data_from_archive(restore_manual_transactions: bool = True) -> None:
    ...


def restore_retention_policy_deletions_for_stream(stream: Stream) -> None:
    ...


def clean_archived_data() -> None:
    ...


def parse_message_retention_days(
    value: Any,
    special_values_map: Dict[str, int],
) -> int:
    ...