import logging
import time
from collections.abc import Iterable, Mapping
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from django.conf import settings
from django.db import connection, transaction
from django.db.models import Model, QuerySet
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Composable, Identifier, Literal
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


class DoesNotExist(Exception):
    ...


class RequestVariableConversionError(Exception):
    ...


class QuerySet(QuerySet):
    ...


class Model(Model):
    ...


class ArchiveTransaction(Model):
    class Type:
        MANUAL: str
        RETENTION_POLICY_BASED: str

    id: int
    type: str
    realm: Optional[Realm]
    restored: bool
    restored_timestamp: Optional[datetime]
    protect_from_deletion: bool

    @classmethod
    def objects(cls) -> QuerySet:
        ...


class Message(Model):
    id: int
    realm_id: int
    recipient_id: int
    date_sent: datetime
    _meta: Any

    @classmethod
    def objects(cls) -> QuerySet:
        ...


class Recipient(Model):
    id: int
    type: str
    type_id: int

    @classmethod
    def objects(cls) -> QuerySet:
        ...


class Stream(Model):
    id: int
    realm: Realm
    recipient: Recipient
    message_retention_days: Optional[int]

    @classmethod
    def objects(cls) -> QuerySet:
        ...


class Realm(Model):
    id: int
    message_retention_days: int
    string_id: str

    @classmethod
    def objects(cls) -> QuerySet:
        ...


def move_rows(
    base_model: type[Model],
    raw_query: Composable,
    src_db_table: Optional[str] = None,
    returning_id: bool = False,
    **kwargs: Any
) -> list[int]:
    ...


def run_archiving(
    query: SQL,
    type: str,
    realm: Optional[Realm] = None,
    chunk_size: int = 1000,
    **kwargs: Any
) -> int:
    ...


def move_expired_messages_to_archive_by_recipient(
    recipient: Model,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = 1000,
) -> int:
    ...


def move_expired_direct_messages_to_archive(
    realm: Realm, chunk_size: int = 1000
) -> int:
    ...


def move_models_with_message_key_to_archive(msg_ids: list[int]) -> None:
    ...


def move_attachments_to_archive(msg_ids: list[int]) -> None:
    ...


def move_attachment_messages_to_archive(msg_ids: list[int]) -> None:
    ...


def delete_messages(msg_ids: list[int]) -> None:
    ...


def delete_expired_attachments(realm: Realm) -> tuple[int, int]:
    ...


def move_related_objects_to_archive(msg_ids: list[int]) -> None:
    ...


def archive_messages_by_recipient(
    recipient: Model,
    message_retention_days: int,
    realm: Realm,
    chunk_size: int = 1000,
) -> int:
    ...


def archive_direct_messages(realm: Realm, chunk_size: int = 1000) -> int:
    ...


def archive_stream_messages(
    realm: Realm, streams: list[Stream], chunk_size: int = 100
) -> int:
    ...


@transaction.atomic(durable=True)
def archive_messages(chunk_size: int = 1000) -> None:
    ...


def get_realms_and_streams_for_archiving() -> list[tuple[Realm, list[Stream]]]:
    ...


def move_messages_to_archive(
    message_ids: list[int], realm: Optional[Realm] = None, chunk_size: int = 1000
) -> int:
    ...


def restore_messages_from_archive(archive_transaction_id: int) -> list[int]:
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
    archive_transactions: list[ArchiveTransaction],
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
    value: Union[str, int], special_values_map: dict[str, int]
) -> int:
    ...