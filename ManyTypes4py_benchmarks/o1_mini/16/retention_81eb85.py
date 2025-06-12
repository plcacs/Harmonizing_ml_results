import logging
import time
from collections.abc import Iterable, Mapping
from datetime import timedelta, datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from django.conf import settings
from django.db import connection, transaction
from django.db.models import Model
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Composable, Identifier, Literal
from zerver.lib.logging_util import log_to_file
from zerver.lib.request import RequestVariableConversionError
from zerver.models import (
    ArchivedAttachment, ArchivedReaction, ArchivedSubMessage,
    ArchivedUserMessage, ArchiveTransaction, Attachment, Message,
    Reaction, Realm, Recipient, Stream, SubMessage, UserMessage
)

logger: logging.Logger = logging.getLogger('zulip.retention')
log_to_file(logger, settings.RETENTION_LOG_PATH)

MESSAGE_BATCH_SIZE: int = 1000
STREAM_MESSAGE_BATCH_SIZE: int = 100
TRANSACTION_DELETION_BATCH_SIZE: int = 100

models_with_message_key: List[Dict[str, Any]] = [
    {
        'class': Reaction,
        'archive_class': ArchivedReaction,
        'table_name': 'zerver_reaction',
        'archive_table_name': 'zerver_archivedreaction'
    },
    {
        'class': SubMessage,
        'archive_class': ArchivedSubMessage,
        'table_name': 'zerver_submessage',
        'archive_table_name': 'zerver_archivedsubmessage'
    },
    {
        'class': UserMessage,
        'archive_class': ArchivedUserMessage,
        'table_name': 'zerver_usermessage',
        'archive_table_name': 'zerver_archivedusermessage'
    }
]

EXCLUDE_FIELDS: set = {Message._meta.get_field('search_tsvector')}

@transaction.atomic(savepoint=False)
def move_rows(
    base_model: Type[Model],
    raw_query: SQL,
    *,
    src_db_table: Optional[str] = None,
    returning_id: bool = False,
    **kwargs: Any
) -> List[int]:
    """Core helper for bulk moving rows between a table and its archive table"""
    if src_db_table is None:
        src_db_table = base_model._meta.db_table
    fields = [field for field in base_model._meta.fields if field not in EXCLUDE_FIELDS]
    src_fields = [Identifier(src_db_table, field.column) for field in fields]
    dst_fields = [Identifier(field.column) for field in fields]
    with connection.cursor() as cursor:
        cursor.execute(
            raw_query.format(
                src_fields=SQL(',').join(src_fields),
                dst_fields=SQL(',').join(dst_fields),
                **kwargs
            )
        )
        if returning_id:
            return [id for id, in cursor.fetchall()]
        else:
            return []

def run_archiving(
    query: SQL,
    type: int,
    realm: Optional[Realm] = None,
    chunk_size: Optional[int] = MESSAGE_BATCH_SIZE,
    **kwargs: Any
) -> int:
    assert type in (ArchiveTransaction.MANUAL, ArchiveTransaction.RETENTION_POLICY_BASED)
    if chunk_size is not None:
        kwargs['chunk_size'] = Literal(chunk_size)
    message_count: int = 0
    while True:
        start_time: float = time.time()
        with transaction.atomic(savepoint=False):
            archive_transaction: ArchiveTransaction = ArchiveTransaction.objects.create(
                type=type, realm=realm
            )
            new_chunk: List[int] = move_rows(
                Message,
                query,
                src_db_table=None,
                returning_id=True,
                archive_transaction_id=Literal(archive_transaction.id),
                **kwargs
            )
            if new_chunk:
                move_related_objects_to_archive(new_chunk)
                delete_messages(new_chunk)
                message_count += len(new_chunk)
            else:
                archive_transaction.delete()
        total_time: float = time.time() - start_time
        if len(new_chunk) > 0:
            logger.info(
                'Archived %s messages in %.2fs in transaction %s.',
                len(new_chunk), total_time, archive_transaction.id
            )
        if chunk_size is None or len(new_chunk) < chunk_size:
            break
    return message_count

def move_expired_messages_to_archive_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: Optional[int] = MESSAGE_BATCH_SIZE
) -> int:
    assert message_retention_days != -1
    query: SQL = SQL(
        '\n    INSERT INTO zerver_archivedmessage ({dst_fields}, archive_transaction_id)\n'
        '        SELECT {src_fields}, {archive_transaction_id}\n'
        '        FROM zerver_message\n'
        '        WHERE zerver_message.realm_id = {realm_id}\n'
        '            AND zerver_message.recipient_id = {recipient_id}\n'
        '            AND zerver_message.date_sent < {check_date}\n'
        '        LIMIT {chunk_size}\n'
        '    ON CONFLICT (id) DO UPDATE SET archive_transaction_id = {archive_transaction_id}\n'
        '    RETURNING id\n    '
    )
    check_date: datetime = timezone_now() - timedelta(days=message_retention_days)
    return run_archiving(
        query,
        type=ArchiveTransaction.RETENTION_POLICY_BASED,
        realm=realm,
        realm_id=Literal(realm.id),
        recipient_id=Literal(recipient.id),
        check_date=Literal(check_date.isoformat()),
        chunk_size=chunk_size
    )

def move_expired_direct_messages_to_archive(
    realm: Realm,
    chunk_size: Optional[int] = MESSAGE_BATCH_SIZE
) -> int:
    message_retention_days: int = realm.message_retention_days
    assert message_retention_days != -1
    check_date: datetime = timezone_now() - timedelta(days=message_retention_days)
    recipient_types: Tuple[int, ...] = (
        Recipient.PERSONAL, Recipient.DIRECT_MESSAGE_GROUP
    )
    query: SQL = SQL(
        '\n    INSERT INTO zerver_archivedmessage ({dst_fields}, archive_transaction_id)\n'
        '        SELECT {src_fields}, {archive_transaction_id}\n'
        '        FROM zerver_message\n'
        '        INNER JOIN zerver_recipient ON zerver_recipient.id = zerver_message.recipient_id\n'
        '        WHERE zerver_message.realm_id = {realm_id}\n'
        '            AND zerver_recipient.type in {recipient_types}\n'
        '            AND zerver_message.date_sent < {check_date}\n'
        '        LIMIT {chunk_size}\n'
        '    ON CONFLICT (id) DO UPDATE SET archive_transaction_id = {archive_transaction_id}\n'
        '    RETURNING id\n    '
    )
    message_count: int = run_archiving(
        query,
        type=ArchiveTransaction.RETENTION_POLICY_BASED,
        realm=realm,
        realm_id=Literal(realm.id),
        recipient_types=Literal(recipient_types),
        check_date=Literal(check_date.isoformat()),
        chunk_size=chunk_size
    )
    return message_count

def move_models_with_message_key_to_archive(msg_ids: Iterable[int]) -> None:
    assert len(msg_ids) > 0
    for model in models_with_message_key:
        query: SQL = SQL(
            '\n        INSERT INTO {archive_table_name} ({dst_fields})\n'
            '            SELECT {src_fields}\n'
            '            FROM {table_name}\n'
            '            WHERE {table_name}.message_id IN {message_ids}\n'
            '        ON CONFLICT (id) DO NOTHING\n        '
        )
        move_rows(
            model['class'],
            query,
            table_name=Identifier(model['table_name']),
            archive_table_name=Identifier(model['archive_table_name']),
            message_ids=Literal(tuple(msg_ids))
        )

def move_attachments_to_archive(msg_ids: Iterable[int]) -> None:
    assert len(msg_ids) > 0
    query: SQL = SQL(
        '\n    INSERT INTO zerver_archivedattachment ({dst_fields})\n'
        '        SELECT {src_fields}\n'
        '        FROM zerver_attachment\n'
        '        INNER JOIN zerver_attachment_messages\n'
        '            ON zerver_attachment_messages.attachment_id = zerver_attachment.id\n'
        '        WHERE zerver_attachment_messages.message_id IN {message_ids}\n'
        '        GROUP BY zerver_attachment.id\n'
        '    ON CONFLICT (id) DO NOTHING\n    '
    )
    move_rows(
        Attachment,
        query,
        message_ids=Literal(tuple(msg_ids))
    )

def move_attachment_messages_to_archive(msg_ids: Iterable[int]) -> None:
    assert len(msg_ids) > 0
    query: SQL = SQL(
        '\n    INSERT INTO zerver_archivedattachment_messages (id, archivedattachment_id, archivedmessage_id)\n'
        '        SELECT zerver_attachment_messages.id, zerver_attachment_messages.attachment_id,\n'
        '            zerver_attachment_messages.message_id\n'
        '        FROM zerver_attachment_messages\n'
        '        WHERE  zerver_attachment_messages.message_id IN %(message_ids)s\n'
        '    ON CONFLICT (id) DO NOTHING\n    '
    )
    with connection.cursor() as cursor:
        cursor.execute(
            query, dict(message_ids=tuple(msg_ids))
        )

def delete_messages(msg_ids: Iterable[int]) -> None:
    Message.objects.filter(id__in=msg_ids).delete()

def delete_expired_attachments(realm: Realm) -> None:
    num_deleted: int
    ignored: Any
    num_deleted, ignored = Attachment.objects.filter(
        messages__isnull=True,
        scheduled_messages__isnull=True,
        realm_id=realm.id,
        id__in=ArchivedAttachment.objects.filter(realm_id=realm.id)
    ).delete()
    if num_deleted > 0:
        logger.info(
            'Cleaned up %s attachments for realm %s', num_deleted, realm.string_id
        )

def move_related_objects_to_archive(msg_ids: Iterable[int]) -> None:
    move_models_with_message_key_to_archive(msg_ids)
    move_attachments_to_archive(msg_ids)
    move_attachment_messages_to_archive(msg_ids)

def archive_messages_by_recipient(
    recipient: Recipient,
    message_retention_days: int,
    realm: Realm,
    chunk_size: Optional[int] = MESSAGE_BATCH_SIZE
) -> int:
    return move_expired_messages_to_archive_by_recipient(
        recipient, message_retention_days, realm, chunk_size
    )

def archive_direct_messages(
    realm: Realm,
    chunk_size: Optional[int] = MESSAGE_BATCH_SIZE
) -> None:
    logger.info(
        'Archiving personal and group direct messages for realm %s',
        realm.string_id
    )
    message_count: int = move_expired_direct_messages_to_archive(realm, chunk_size)
    logger.info('Done. Archived %s messages', message_count)

def archive_stream_messages(
    realm: Realm,
    streams: Iterable[Stream],
    chunk_size: int = STREAM_MESSAGE_BATCH_SIZE
) -> None:
    if not streams:
        return
    logger.info('Archiving stream messages for realm %s', realm.string_id)
    retention_policy_dict: Dict[int, int] = {}
    for stream in streams:
        if stream.message_retention_days:
            retention_policy_dict[stream.id] = stream.message_retention_days
        else:
            assert realm.message_retention_days != -1
            retention_policy_dict[stream.id] = realm.message_retention_days
    recipients: List[Recipient] = [stream.recipient for stream in streams]
    message_count: int = 0
    for recipient in recipients:
        assert recipient is not None
        message_count += archive_messages_by_recipient(
            recipient, retention_policy_dict[recipient.type_id], realm, chunk_size
        )
    logger.info('Done. Archived %s messages.', message_count)

@transaction.atomic(durable=True)
def archive_messages(chunk_size: int = MESSAGE_BATCH_SIZE) -> None:
    logger.info(
        'Starting the archiving process with chunk_size %s', chunk_size
    )
    for realm, streams in get_realms_and_streams_for_archiving():
        archive_stream_messages(realm, streams, chunk_size=STREAM_MESSAGE_BATCH_SIZE)
        if realm.message_retention_days != -1:
            archive_direct_messages(realm, chunk_size)
        delete_expired_attachments(realm)

def get_realms_and_streams_for_archiving() -> List[Tuple[Realm, List[Stream]]]:
    """
    This function constructs a list of (realm, streams_of_the_realm) tuples
    where each realm is a Realm that requires calling the archiving functions on it,
    and streams_of_the_realm is a list of streams of the realm to call archive_stream_messages with.

    The purpose of this is performance - for servers with thousands of realms, it is important
    to fetch all this data in bulk.
    """
    realm_id_to_realm: Dict[int, Realm] = {}
    realm_id_to_streams_list: Dict[int, List[Stream]] = {}
    for realm in Realm.objects.exclude(message_retention_days=-1):
        realm_id_to_realm[realm.id] = realm
        realm_id_to_streams_list[realm.id] = []
    query_one = Stream.objects.exclude(message_retention_days=-1).exclude(
        realm__message_retention_days=-1
    ).select_related('realm', 'recipient')
    query_two = Stream.objects.filter(
        realm__message_retention_days=-1
    ).exclude(
        message_retention_days__isnull=True
    ).exclude(
        message_retention_days=-1
    ).select_related('realm', 'recipient')
    query = query_one.union(query_two)
    for stream in query:
        realm = stream.realm
        realm_id_to_realm[realm.id] = realm
        if realm.id not in realm_id_to_streams_list:
            realm_id_to_streams_list[realm.id] = []
        realm_id_to_streams_list[realm.id].append(stream)
    return [
        (realm_id_to_realm[realm_id], realm_id_to_streams_list[realm_id])
        for realm_id in realm_id_to_realm
    ]

def move_messages_to_archive(
    message_ids: Iterable[int],
    realm: Optional[Realm] = None,
    chunk_size: int = MESSAGE_BATCH_SIZE
) -> None:
    """
    Callers using this to archive a large amount of messages should ideally make sure the message_ids are
    ordered, as that'll allow better performance here by keeping the batches that'll be sent to the database
    ordered rather than randomly scattered.
    """
    count: int = 0
    message_ids_head = list(message_ids)
    while message_ids_head:
        message_ids_chunk: List[int] = message_ids_head[0:chunk_size]
        message_ids_head = message_ids_head[chunk_size:]
        query: SQL = SQL(
            '\n        INSERT INTO zerver_archivedmessage ({dst_fields}, archive_transaction_id)\n'
            '            SELECT {src_fields}, {archive_transaction_id}\n'
            '            FROM zerver_message\n'
            '            WHERE zerver_message.id IN {message_ids}\n'
            '        ON CONFLICT (id) DO UPDATE SET archive_transaction_id = {archive_transaction_id}\n'
            '        RETURNING id\n        '
        )
        count += run_archiving(
            query,
            type=ArchiveTransaction.MANUAL,
            message_ids=Literal(tuple(message_ids_chunk)),
            realm=realm,
            chunk_size=None
        )
        archived_attachments = ArchivedAttachment.objects.filter(
            messages__id__in=message_ids_chunk
        ).distinct()
        Attachment.objects.filter(
            messages__isnull=True,
            scheduled_messages__isnull=True,
            id__in=archived_attachments
        ).delete()
    if count == 0:
        raise Message.DoesNotExist

def restore_messages_from_archive(archive_transaction_id: int) -> List[int]:
    query: SQL = SQL(
        '\n        INSERT INTO zerver_message ({dst_fields})\n'
        '            SELECT {src_fields}\n'
        '            FROM zerver_archivedmessage\n'
        '            WHERE zerver_archivedmessage.archive_transaction_id = {archive_transaction_id}\n'
        '        ON CONFLICT (id) DO NOTHING\n'
        '        RETURNING id\n        '
    )
    return move_rows(
        Message,
        query,
        src_db_table='zerver_archivedmessage',
        returning_id=True,
        archive_transaction_id=Literal(archive_transaction_id)
    )

def restore_models_with_message_key_from_archive(archive_transaction_id: int) -> None:
    for model in models_with_message_key:
        query: SQL = SQL(
            '\n        INSERT INTO {table_name} ({dst_fields})\n'
            '            SELECT {src_fields}\n'
            '            FROM {archive_table_name}\n'
            '            INNER JOIN zerver_archivedmessage ON {archive_table_name}.message_id = zerver_archivedmessage.id\n'
            '            WHERE zerver_archivedmessage.archive_transaction_id = {archive_transaction_id}\n'
            '        ON CONFLICT (id) DO NOTHING\n        '
        )
        move_rows(
            model['class'],
            query,
            src_db_table=model['archive_table_name'],
            table_name=Identifier(model['table_name']),
            archive_transaction_id=Literal(archive_transaction_id),
            archive_table_name=Identifier(model['archive_table_name'])
        )

def restore_attachments_from_archive(archive_transaction_id: int) -> None:
    query: SQL = SQL(
        '\n    INSERT INTO zerver_attachment ({dst_fields})\n'
        '        SELECT {src_fields}\n'
        '        FROM zerver_archivedattachment\n'
        '        INNER JOIN zerver_archivedattachment_messages\n'
        '            ON zerver_archivedattachment_messages.archivedattachment_id = zerver_archivedattachment.id\n'
        '        INNER JOIN zerver_archivedmessage\n'
        '            ON  zerver_archivedattachment_messages.archivedmessage_id = zerver_archivedmessage.id\n'
        '        WHERE zerver_archivedmessage.archive_transaction_id = {archive_transaction_id}\n'
        '        GROUP BY zerver_archivedattachment.id\n'
        '    ON CONFLICT (id) DO NOTHING\n    '
    )
    move_rows(
        Attachment,
        query,
        src_db_table='zerver_archivedattachment',
        archive_transaction_id=Literal(archive_transaction_id)
    )

def restore_attachment_messages_from_archive(archive_transaction_id: int) -> None:
    query: SQL = SQL(
        '\n    INSERT INTO zerver_attachment_messages (id, attachment_id, message_id)\n'
        '        SELECT zerver_archivedattachment_messages.id,\n'
        '            zerver_archivedattachment_messages.archivedattachment_id,\n'
        '            zerver_archivedattachment_messages.archivedmessage_id\n'
        '        FROM zerver_archivedattachment_messages\n'
        '        INNER JOIN zerver_archivedmessage\n'
        '            ON  zerver_archivedattachment_messages.archivedmessage_id = zerver_archivedmessage.id\n'
        '        WHERE zerver_archivedmessage.archive_transaction_id = %(archive_transaction_id)s\n'
        '    ON CONFLICT (id) DO NOTHING\n    '
    )
    with connection.cursor() as cursor:
        cursor.execute(
            query,
            dict(archive_transaction_id=archive_transaction_id)
        )

def restore_data_from_archive(archive_transaction: ArchiveTransaction) -> int:
    logger.info('Restoring %s', archive_transaction)
    with transaction.atomic(durable=True):
        msg_ids: List[int] = restore_messages_from_archive(archive_transaction.id)
        restore_models_with_message_key_from_archive(archive_transaction.id)
        restore_attachments_from_archive(archive_transaction.id)
        restore_attachment_messages_from_archive(archive_transaction.id)
        archive_transaction.restored = True
        archive_transaction.restored_timestamp = timezone_now()
        archive_transaction.save()
    logger.info('Finished. Restored %s messages', len(msg_ids))
    return len(msg_ids)

def restore_data_from_archive_by_transactions(
    archive_transactions: Iterable[ArchiveTransaction]
) -> int:
    message_count: int = 0
    for archive_transaction in archive_transactions:
        message_count += restore_data_from_archive(archive_transaction)
    return message_count

def restore_data_from_archive_by_realm(realm: Realm) -> int:
    transactions: List[ArchiveTransaction] = list(
        ArchiveTransaction.objects.exclude(restored=True).filter(
            realm=realm,
            type=ArchiveTransaction.RETENTION_POLICY_BASED
        )
    )
    logger.info(
        'Restoring %s transactions from realm %s',
        len(transactions),
        realm.string_id
    )
    message_count: int = restore_data_from_archive_by_transactions(transactions)
    logger.info(
        'Finished. Restored %s messages from realm %s',
        message_count,
        realm.string_id
    )
    return message_count

def restore_all_data_from_archive(
    restore_manual_transactions: bool = True
) -> None:
    for realm in Realm.objects.all():
        restore_data_from_archive_by_realm(realm)
    if restore_manual_transactions:
        restore_data_from_archive_by_transactions(
            ArchiveTransaction.objects.exclude(restored=True).filter(
                type=ArchiveTransaction.MANUAL
            )
        )

def restore_retention_policy_deletions_for_stream(stream: Stream) -> None:
    """
    Utility function for calling in the Django shell if a stream's policy was
    set to something too aggressive and the administrator wants to restore
    the messages deleted as a result.
    """
    relevant_transactions: List[ArchiveTransaction] = list(
        ArchiveTransaction.objects.filter(
            archivedmessage__recipient=stream.recipient,
            type=ArchiveTransaction.RETENTION_POLICY_BASED
        ).distinct('id')
    )
    restore_data_from_archive_by_transactions(relevant_transactions)

def clean_archived_data() -> None:
    """This function deletes archived data that was archived at least
    settings.ARCHIVED_DATA_VACUUMING_DELAY_DAYS days ago.

    It works by deleting ArchiveTransaction objects that are
    sufficiently old. We've configured most archive tables, like
    ArchiveMessage, with on_delete=CASCADE, so that deleting an
    ArchiveTransaction entails deleting associated objects, including
    ArchivedMessage, ArchivedUserMessage, ArchivedReaction.

    The exception to this rule is ArchivedAttachment. Archive
    attachment objects that were only referenced by ArchivedMessage
    objects that have now been deleted will be left with an empty
    `.messages` relation. A separate step,
    delete_old_unclaimed_attachments, will delete those
    ArchivedAttachment objects (and delete the files themselves from
    the storage).
    """
    logger.info('Cleaning old archive data.')
    check_date: datetime = timezone_now() - timedelta(
        days=settings.ARCHIVED_DATA_VACUUMING_DELAY_DAYS
    )
    count: int = 0
    transaction_ids: List[int] = list(
        ArchiveTransaction.objects.filter(
            timestamp__lt=check_date,
            protect_from_deletion=False
        ).values_list('id', flat=True)
    )
    while len(transaction_ids) > 0:
        transaction_block: List[int] = transaction_ids[0:TRANSACTION_DELETION_BATCH_SIZE]
        transaction_ids = transaction_ids[TRANSACTION_DELETION_BATCH_SIZE:]
        ArchiveTransaction.objects.filter(
            id__in=transaction_block,
            protect_from_deletion=False
        ).delete()
        count += len(transaction_block)
    logger.info('Deleted %s old ArchiveTransactions.', count)

def parse_message_retention_days(
    value: Union[str, int],
    special_values_map: Mapping[str, Any]
) -> int:
    if isinstance(value, str) and value in special_values_map:
        return special_values_map[value]
    if isinstance(value, str) or value <= 0:
        raise RequestVariableConversionError('message_retention_days', value)
    assert isinstance(value, int)
    return value
