#!/usr/bin/env python3
from collections import defaultdict
import collections
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from difflib import unified_diff
from typing import Any, Dict, List, Tuple, Callable

import bmemcached
import orjson
import pyvips
from bs4 import BeautifulSoup
from django.conf import settings
from django.core.cache import cache
from django.core.management.base import CommandError
from django.core.validators import validate_email
from django.db import connection, transaction
from django.db.backends.utils import CursorWrapper
from django.utils.timezone import now as timezone_now
from psycopg2.extras import execute_values
from psycopg2.sql import SQL, Identifier

from analytics.models import RealmCount, StreamCount, UserCount
from version import ZULIP_VERSION
from zerver.actions.create_realm import set_default_for_realm_permission_group_settings
from zerver.actions.realm_settings import do_change_realm_plan_type
from zerver.actions.user_settings import do_change_avatar_fields
from zerver.lib.avatar_hash import user_avatar_base_path_from_ids
from zerver.lib.bulk_create import bulk_set_users_or_streams_recipient_fields
from zerver.lib.export import DATE_FIELDS, Field, Path, Record, TableData, TableName
from zerver.lib.markdown import markdown_convert
from zerver.lib.markdown import version as markdown_version
from zerver.lib.message import get_last_message_id
from zerver.lib.migration_status import (
    MigrationStatusJson,
    get_migration_status,
    parse_migration_status,
)
from zerver.lib.mime_types import guess_type
from zerver.lib.partial import partial
from zerver.lib.push_notifications import sends_notifications_directly
from zerver.lib.remote_server import maybe_enqueue_audit_log_upload
from zerver.lib.server_initialization import create_internal_realm, server_initialized
from zerver.lib.streams import (
    get_stream_permission_default_group,
    render_stream_description,
    update_stream_active_status_for_realm,
)
from zerver.lib.thumbnail import (
    THUMBNAIL_ACCEPT_IMAGE_TYPES,
    BadImageError,
    get_user_upload_previews,
    maybe_thumbnail,
)
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.upload import ensure_avatar_image, sanitize_name, upload_backend, upload_emoji_image
from zerver.lib.upload.s3 import get_bucket
from zerver.lib.user_counts import realm_user_count_by_role
from zerver.lib.user_groups import create_system_user_groups_for_realm
from zerver.lib.user_message import UserMessageLite, bulk_insert_ums
from zerver.lib.utils import generate_api_key, process_list_in_batches
from zerver.lib.zulip_update_announcements import send_zulip_update_announcements_to_realm
from zerver.models import (
    AlertWord,
    Attachment,
    BotConfigData,
    BotStorageData,
    Client,
    CustomProfileField,
    CustomProfileFieldValue,
    DefaultStream,
    DirectMessageGroup,
    GroupGroupMembership,
    Message,
    MutedUser,
    NamedUserGroup,
    OnboardingStep,
    OnboardingUserMessage,
    Reaction,
    Realm,
    RealmAuditLog,
    RealmAuthenticationMethod,
    RealmDomain,
    RealmEmoji,
    RealmFilter,
    RealmPlayground,
    RealmUserDefault,
    Recipient,
    SavedSnippet,
    ScheduledMessage,
    Service,
    Stream,
    Subscription,
    UserActivity,
    UserActivityInterval,
    UserGroup,
    UserGroupMembership,
    UserMessage,
    UserPresence,
    UserProfile,
    UserStatus,
    UserTopic,
)
from zerver.models.groups import SystemGroups
from zerver.models.presence import PresenceSequence
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_realm
from zerver.models.recipients import get_direct_message_group_hash
from zerver.models.users import get_system_bot, get_user_profile_by_id
from zproject.backends import AUTH_BACKEND_NAME_MAP

realm_tables: List[Tuple[str, Any, str]] = [
    ("zerver_realmauthenticationmethod", RealmAuthenticationMethod, "realmauthenticationmethod"),
    ("zerver_defaultstream", DefaultStream, "defaultstream"),
    ("zerver_realmemoji", RealmEmoji, "realmemoji"),
    ("zerver_realmdomain", RealmDomain, "realmdomain"),
    ("zerver_realmfilter", RealmFilter, "realmfilter"),
    ("zerver_realmplayground", RealmPlayground, "realmplayground"),
]

# ID_MAP maps table names (str) to mappings from old id to new id.
ID_MAP: Dict[str, Dict[int, int]] = {
    "alertword": {},
    "client": {},
    "user_profile": {},
    "huddle": {},
    "realm": {},
    "stream": {},
    "recipient": {},
    "subscription": {},
    "defaultstream": {},
    "onboardingstep": {},
    "presencesequence": {},
    "reaction": {},
    "realmauthenticationmethod": {},
    "realmemoji": {},
    "realmdomain": {},
    "realmfilter": {},
    "realmplayground": {},
    "message": {},
    "user_presence": {},
    "userstatus": {},
    "useractivity": {},
    "useractivityinterval": {},
    "usermessage": {},
    "customprofilefield": {},
    "customprofilefieldvalue": {},
    "attachment": {},
    "realmauditlog": {},
    "recipient_to_huddle_map": {},
    "usertopic": {},
    "muteduser": {},
    "service": {},
    "usergroup": {},
    "usergroupmembership": {},
    "groupgroupmembership": {},
    "botstoragedata": {},
    "botconfigdata": {},
    "analytics_realmcount": {},
    "analytics_streamcount": {},
    "analytics_usercount": {},
    "realmuserdefault": {},
    "scheduledmessage": {},
    "onboardingusermessage": {},
}

id_map_to_list: Dict[str, Dict[int, List[int]]] = {
    "huddle_to_user_list": {},
}

path_maps: Dict[str, Dict[str, str]] = {
    "old_attachment_path_to_new_path": {},
    "new_attachment_path_to_old_path": {},
    "new_attachment_path_to_local_data_path": {},
}

message_id_to_attachments: Dict[str, Dict[int, List[str]]] = {
    "zerver_message": defaultdict(list),
    "zerver_scheduledmessage": defaultdict(list),
}


def map_messages_to_attachments(data: TableData) -> None:
    for attachment in data["zerver_attachment"]:
        for message_id in attachment["messages"]:
            message_id_to_attachments["zerver_message"][message_id].append(attachment["path_id"])
        for scheduled_message_id in attachment["scheduled_messages"]:
            message_id_to_attachments["zerver_scheduledmessage"][scheduled_message_id].append(
                attachment["path_id"]
            )


def update_id_map(table: str, old_id: int, new_id: int) -> None:
    if table not in ID_MAP:
        raise Exception(
            f"""
            Table {table} is not initialized in ID_MAP, which could
            mean that we have not thought through circular
            dependencies.
            """
        )
    ID_MAP[table][old_id] = new_id


def fix_datetime_fields(data: TableData, table: str) -> None:
    for item in data[table]:
        for field_name in DATE_FIELDS[table]:
            if item[field_name] is not None:
                item[field_name] = datetime.fromtimestamp(item[field_name], tz=timezone.utc)


def fix_upload_links(data: TableData, message_table: str) -> None:
    for message in data[message_table]:
        if message.get("has_attachment") is True:
            for attachment_path in message_id_to_attachments[message_table][message["id"]]:
                old_path = path_maps["new_attachment_path_to_old_path"][attachment_path]
                message["content"] = message["content"].replace(old_path, attachment_path)
                if message.get("rendered_content"):
                    message["rendered_content"] = message["rendered_content"].replace(old_path, attachment_path)


def fix_stream_permission_group_settings(
    data: TableData, system_groups_name_dict: Dict[str, NamedUserGroup]
) -> None:
    table: str = get_db_table(Stream)
    for stream in data[table]:
        for setting_name in Stream.stream_permission_group_settings:
            if setting_name == "can_send_message_group" and "stream_post_policy" in stream:
                if stream["stream_post_policy"] == Stream.STREAM_POST_POLICY_MODERATORS:
                    stream[setting_name] = system_groups_name_dict[SystemGroups.MODERATORS]
                else:
                    stream[setting_name] = get_stream_permission_default_group(setting_name, system_groups_name_dict)
                del stream["stream_post_policy"]
                continue
            stream[setting_name] = get_stream_permission_default_group(setting_name, system_groups_name_dict)


def create_subscription_events(data: TableData, realm_id: int) -> None:
    all_subscription_logs: List[RealmAuditLog] = []
    event_last_message_id: int = get_last_message_id()
    event_time: datetime = timezone_now()
    recipient_id_to_stream_id: Dict[int, int] = {
        d["id"]: d["type_id"] for d in data["zerver_recipient"] if d["type"] == Recipient.STREAM
    }
    for sub in data["zerver_subscription"]:
        recipient_id: int = sub["recipient_id"]
        stream_id: int = recipient_id_to_stream_id.get(recipient_id, None)
        if stream_id is None:
            continue
        user_id: int = sub["user_profile_id"]
        all_subscription_logs.append(
            RealmAuditLog(
                realm_id=realm_id,
                acting_user_id=user_id,
                modified_user_id=user_id,
                modified_stream_id=stream_id,
                event_last_message_id=event_last_message_id,
                event_time=event_time,
                event_type=AuditLogEventType.SUBSCRIPTION_CREATED,
            )
        )
    RealmAuditLog.objects.bulk_create(all_subscription_logs)


def fix_service_tokens(data: TableData, table: str) -> None:
    for item in data[table]:
        item["token"] = generate_api_key()


def process_direct_message_group_hash(data: TableData, table: str) -> None:
    for direct_message_group in data[table]:
        user_id_list: List[int] = id_map_to_list["huddle_to_user_list"][direct_message_group["id"]]
        direct_message_group["huddle_hash"] = get_direct_message_group_hash(user_id_list)


def get_direct_message_groups_from_subscription(data: TableData, table: str) -> None:
    id_map_to_list["huddle_to_user_list"] = {value: [] for value in ID_MAP["recipient_to_huddle_map"].values()}
    for subscription in data[table]:
        if subscription["recipient"] in ID_MAP["recipient_to_huddle_map"]:
            direct_message_group_id: int = ID_MAP["recipient_to_huddle_map"][subscription["recipient"]]
            id_map_to_list["huddle_to_user_list"][direct_message_group_id].append(subscription["user_profile_id"])


def fix_customprofilefield(data: TableData) -> None:
    field_type_USER_ids = {
        item["id"]
        for item in data["zerver_customprofilefield"]
        if item["field_type"] == CustomProfileField.USER
    }
    for item in data["zerver_customprofilefieldvalue"]:
        if item["field_id"] in field_type_USER_ids:
            old_user_id_list: List[int] = orjson.loads(item["value"])
            new_id_list: List[int] = re_map_foreign_keys_many_to_many_internal(
                table="zerver_customprofilefieldvalue",
                field_name="value",
                related_table="user_profile",
                old_id_list=old_user_id_list,
            )
            item["value"] = orjson.dumps(new_id_list).decode()


def fix_message_rendered_content(
    realm: Realm,
    sender_map: Dict[int, Record],
    messages: List[Record],
    content_key: str = "content",
    rendered_content_key: str = "rendered_content",
) -> None:
    for message in messages:
        if content_key not in message:
            continue
        if message.get(rendered_content_key) is not None:
            soup: BeautifulSoup = BeautifulSoup(message[rendered_content_key], "html.parser")
            user_mentions = soup.findAll("span", {"class": "user-mention"})
            if user_mentions:
                user_id_map: Dict[int, int] = ID_MAP["user_profile"]
                for mention in user_mentions:
                    if not mention.has_attr("data-user-id"):
                        continue
                    if mention["data-user-id"] == "*":
                        continue
                    old_user_id: int = int(mention["data-user-id"])
                    if old_user_id in user_id_map:
                        mention["data-user-id"] = str(user_id_map[old_user_id])
                message[rendered_content_key] = str(soup)
            stream_mentions = soup.findAll("a", {"class": "stream"})
            if stream_mentions:
                stream_id_map: Dict[int, int] = ID_MAP["stream"]
                for mention in stream_mentions:
                    old_stream_id: int = int(mention["data-stream-id"])
                    if old_stream_id in stream_id_map:
                        mention["data-stream-id"] = str(stream_id_map[old_stream_id])
                message[rendered_content_key] = str(soup)
            user_group_mentions = soup.findAll("span", {"class": "user-group-mention"})
            if user_group_mentions:
                user_group_id_map: Dict[int, int] = ID_MAP["usergroup"]
                for mention in user_group_mentions:
                    old_user_group_id: int = int(mention["data-user-group-id"])
                    if old_user_group_id in user_group_id_map:
                        mention["data-user-group-id"] = str(user_group_id_map[old_user_group_id])
                message[rendered_content_key] = str(soup)
            get_user_upload_previews(realm.id, message[content_key], lock=True)
            continue
        try:
            content: str = message[content_key]
            sender_id: int = message["sender_id"]
            sender: Record = sender_map[sender_id]
            sent_by_bot: bool = sender["is_bot"]
            translate_emoticons: bool = sender["translate_emoticons"]
            realm_alert_words_automaton = None
            rendered_content: str = markdown_convert(
                content=content,
                realm_alert_words_automaton=realm_alert_words_automaton,
                message_realm=realm,
                sent_by_bot=sent_by_bot,
                translate_emoticons=translate_emoticons,
            ).rendered_content
            message[rendered_content_key] = rendered_content
            if "scheduled_timestamp" not in message:
                message["rendered_content_version"] = markdown_version
        except Exception:
            logging.warning("Error in Markdown rendering for message ID %s; continuing", message["id"])


def fix_message_edit_history(
    realm: Realm, sender_map: Dict[int, Record], messages: List[Record]
) -> None:
    user_id_map: Dict[int, int] = ID_MAP["user_profile"]
    for message in messages:
        edit_history_json = message.get("edit_history")
        if not edit_history_json:
            continue
        edit_history: List[Record] = orjson.loads(edit_history_json)
        for edit_history_message_dict in edit_history:
            edit_history_message_dict["user_id"] = user_id_map[edit_history_message_dict["user_id"]]
        fix_message_rendered_content(
            realm,
            sender_map,
            messages=edit_history,
            content_key="prev_content",
            rendered_content_key="prev_rendered_content",
        )
        message["edit_history"] = orjson.dumps(edit_history).decode()


def current_table_ids(data: TableData, table: str) -> List[int]:
    return [item["id"] for item in data[table]]


def idseq(model_class: Any, cursor: CursorWrapper) -> str:
    sequences = connection.introspection.get_sequences(cursor, model_class._meta.db_table)
    for sequence in sequences:
        if sequence["column"] == "id":
            return sequence["name"]
    raise Exception(f"No sequence found for 'id' of {model_class}")


def allocate_ids(model_class: Any, count: int) -> List[int]:
    with connection.cursor() as cursor:
        sequence: str = idseq(model_class, cursor)
        cursor.execute("select nextval(%s) from generate_series(1, %s)", [sequence, count])
        query: List[Tuple[int]] = cursor.fetchall()
    return [item[0] for item in query]


def convert_to_id_fields(data: TableData, table: str, field_name: Field) -> None:
    for item in data[table]:
        item[field_name + "_id"] = item[field_name]
        del item[field_name]


def re_map_foreign_keys(
    data: TableData,
    table: str,
    field_name: Field,
    related_table: str,
    verbose: bool = False,
    id_field: bool = False,
    recipient_field: bool = False,
) -> None:
    assert related_table != "usermessage"
    re_map_foreign_keys_internal(
        data[table],
        table,
        field_name,
        related_table,
        verbose,
        id_field,
        recipient_field,
    )


def re_map_foreign_keys_internal(
    data_table: List[Record],
    table: str,
    field_name: Field,
    related_table: str,
    verbose: bool = False,
    id_field: bool = False,
    recipient_field: bool = False,
) -> None:
    lookup_table: Dict[int, int] = ID_MAP[related_table]
    for item in data_table:
        old_id: int = item[field_name]
        if recipient_field:
            if related_table == "stream" and item["type"] == 2:
                pass
            elif related_table == "user_profile" and item["type"] == 1:
                pass
            elif related_table == "huddle" and item["type"] == 3:
                ID_MAP["recipient_to_huddle_map"][item["id"]] = lookup_table[old_id]
            else:
                continue
        old_id = item[field_name]
        if old_id in lookup_table:
            new_id: int = lookup_table[old_id]
            if verbose:
                logging.info("Remapping %s %s from %s to %s", table, field_name + "_id", old_id, new_id)
        else:
            new_id = old_id
        if not id_field:
            item[field_name + "_id"] = new_id
            del item[field_name]
        else:
            item[field_name] = new_id


def re_map_realm_emoji_codes(data: TableData, *, table_name: str) -> None:
    realm_emoji_dct: Dict[int, Record] = {}
    for row in data["zerver_realmemoji"]:
        realm_emoji_dct[row["id"]] = row
    for row in data[table_name]:
        if row["reaction_type"] == Reaction.REALM_EMOJI:
            old_realm_emoji_id: int = int(row["emoji_code"])
            new_realm_emoji_id: int = ID_MAP["realmemoji"][old_realm_emoji_id]
            realm_emoji_row: Record = realm_emoji_dct[new_realm_emoji_id]
            assert realm_emoji_row["name"] == row["emoji_name"]
            row["emoji_code"] = str(new_realm_emoji_id)


def re_map_foreign_keys_many_to_many(
    data: TableData,
    table: str,
    field_name: Field,
    related_table: str,
    verbose: bool = False,
) -> None:
    for item in data[table]:
        old_id_list: List[int] = item[field_name]
        new_id_list: List[int] = re_map_foreign_keys_many_to_many_internal(
            table, field_name, related_table, old_id_list, verbose
        )
        item[field_name] = new_id_list
        del item[field_name]


def re_map_foreign_keys_many_to_many_internal(
    table: str,
    field_name: Field,
    related_table: str,
    old_id_list: List[int],
    verbose: bool = False,
) -> List[int]:
    lookup_table: Dict[int, int] = ID_MAP[related_table]
    new_id_list: List[int] = []
    for old_id in old_id_list:
        if old_id in lookup_table:
            new_id: int = lookup_table[old_id]
            if verbose:
                logging.info("Remapping %s %s from %s to %s", table, field_name + "_id", old_id, new_id)
        else:
            new_id = old_id
        new_id_list.append(new_id)
    return new_id_list


def fix_bitfield_keys(data: TableData, table: str, field_name: Field) -> None:
    for item in data[table]:
        item[field_name] = item[field_name + "_mask"]
        del item[field_name + "_mask"]


def remove_denormalized_recipient_column_from_data(data: TableData) -> None:
    for stream_dict in data["zerver_stream"]:
        if "recipient" in stream_dict:
            del stream_dict["recipient"]
    for user_profile_dict in data["zerver_userprofile"]:
        if "recipient" in user_profile_dict:
            del user_profile_dict["recipient"]
    for direct_message_group_dict in data["zerver_huddle"]:
        if "recipient" in direct_message_group_dict:
            del direct_message_group_dict["recipient"]


def get_db_table(model_class: Any) -> str:
    return model_class._meta.db_table


def update_model_ids(model: Any, data: TableData, related_table: str) -> None:
    table: str = get_db_table(model)
    assert table != "usermessage"
    old_id_list: List[int] = current_table_ids(data, table)
    allocated_id_list: List[int] = allocate_ids(model, len(data[table]))
    for item in range(len(data[table])):
        update_id_map(related_table, old_id_list[item], allocated_id_list[item])
    re_map_foreign_keys(data, table, "id", related_table=related_table, id_field=True)


def bulk_import_user_message_data(data: TableData, dump_file_id: int) -> None:
    model = UserMessage
    table: str = "zerver_usermessage"
    lst: List[Dict[str, Any]] = data[table]

    def process_batch(items: List[Dict[str, Any]]) -> None:
        ums = [
            UserMessageLite(
                user_profile_id=item["user_profile_id"],
                message_id=item["message_id"],
                flags=item["flags"],
            )
            for item in items
        ]
        bulk_insert_ums(ums)

    chunk_size: int = 10000
    process_list_in_batches(lst=lst, chunk_size=chunk_size, process_batch=process_batch)
    logging.info("Successfully imported %s from %s[%s].", model, table, dump_file_id)


def bulk_import_model(data: TableData, model: Any, dump_file_id: str | None = None) -> None:
    table: str = get_db_table(model)
    model.objects.bulk_create(model(**item) for item in data[table])
    if dump_file_id is None:
        logging.info("Successfully imported %s from %s.", model, table)
    else:
        logging.info("Successfully imported %s from %s[%s].", model, table, dump_file_id)


def bulk_import_named_user_groups(data: TableData) -> None:
    vals: List[Tuple[Any, ...]] = [
        (
            group["usergroup_ptr_id"],
            group["realm_for_sharding_id"],
            group["name"],
            group["description"],
            group["is_system_group"],
            group["can_add_members_group_id"],
            group["can_join_group_id"],
            group["can_leave_group_id"],
            group["can_manage_group_id"],
            group["can_mention_group_id"],
            group["can_remove_members_group_id"],
            group["deactivated"],
            group["date_created"],
        )
        for group in data["zerver_namedusergroup"]
    ]
    query = SQL(
        """
        INSERT INTO zerver_namedusergroup (usergroup_ptr_id, realm_id, name, description, is_system_group, can_add_members_group_id, can_join_group_id,  can_leave_group_id, can_manage_group_id, can_mention_group_id, can_remove_members_group_id, deactivated, date_created)
        VALUES %s
        """
    )
    with connection.cursor() as cursor:
        execute_values(cursor.cursor, query, vals)


def bulk_import_client(data: TableData, model: Any, table: str) -> None:
    for item in data[table]:
        try:
            client = Client.objects.get(name=item["name"])
        except Client.DoesNotExist:
            client = Client.objects.create(name=item["name"])
        update_id_map(table="client", old_id=item["id"], new_id=client.id)


def fix_subscriptions_is_user_active_column(
    data: TableData, user_profiles: List[UserProfile], crossrealm_user_ids: set[int]
) -> None:
    table: str = get_db_table(Subscription)
    user_id_to_active_status: Dict[int, bool] = {user.id: user.is_active for user in user_profiles}
    for sub in data[table]:
        if sub["user_profile_id"] in crossrealm_user_ids:
            sub["is_user_active"] = True
        else:
            sub["is_user_active"] = user_id_to_active_status[sub["user_profile_id"]]


def process_avatars(record: Dict[str, Any]) -> None:
    if not record["s3_path"].endswith(".original"):
        return
    user_profile: UserProfile = get_user_profile_by_id(record["user_profile_id"])
    if settings.LOCAL_AVATARS_DIR is not None:
        avatar_path: str = user_avatar_base_path_from_ids(
            user_profile.id, user_profile.avatar_version, record["realm_id"]
        )
        medium_file_path: str = os.path.join(settings.LOCAL_AVATARS_DIR, avatar_path) + "-medium.png"
        if os.path.exists(medium_file_path):
            os.remove(medium_file_path)
    try:
        ensure_avatar_image(user_profile=user_profile, medium=True)
        if record.get("importer_should_thumbnail"):
            ensure_avatar_image(user_profile=user_profile)
    except BadImageError:
        logging.warning("Could not thumbnail avatar image for user %s; ignoring", user_profile.id)
        do_change_avatar_fields(user_profile, UserProfile.AVATAR_FROM_GRAVATAR, acting_user=None)


def process_emojis(
    import_dir: Path,
    default_user_profile_id: int | None,
    filename_to_has_original: Dict[str, bool],
    record: Dict[str, Any],
) -> None:
    should_use_as_original: bool = not filename_to_has_original[record["file_name"]]
    if not (record["s3_path"].endswith(".original") or should_use_as_original):
        return
    if "author_id" in record and record["author_id"] is not None:
        user_profile: UserProfile = get_user_profile_by_id(record["author_id"])
    else:
        assert default_user_profile_id is not None
        user_profile = get_user_profile_by_id(default_user_profile_id)
    content_type: str = guess_type(record["file_name"])[0] or "application/octet-stream"
    emoji_import_data_file_dath: str = os.path.join(import_dir, record["path"])
    with open(emoji_import_data_file_dath, "rb") as f:
        try:
            is_animated: bool = upload_emoji_image(f, record["file_name"], user_profile, content_type)
        except BadImageError:
            logging.warning("Could not thumbnail emoji image %s; ignoring", record["s3_path"])
            return
    if is_animated and not record.get("deactivated", False):
        RealmEmoji.objects.filter(
            file_name=record["file_name"], realm_id=user_profile.realm_id, deactivated=False
        ).update(is_animated=True)


def import_uploads(
    realm: Realm,
    import_dir: Path,
    processes: int,
    default_user_profile_id: int | None = None,
    processing_avatars: bool = False,
    processing_emojis: bool = False,
    processing_realm_icons: bool = False,
) -> None:
    if processing_avatars and processing_emojis:
        raise AssertionError("Cannot import avatars and emojis at the same time!")
    if processing_avatars:
        logging.info("Importing avatars")
    elif processing_emojis:
        logging.info("Importing emojis")
    elif processing_realm_icons:
        logging.info("Importing realm icons and logos")
    else:
        logging.info("Importing uploaded files")
    records_filename: str = os.path.join(import_dir, "records.json")
    with open(records_filename, "rb") as records_file:
        records: List[Dict[str, Any]] = orjson.loads(records_file.read())
    timestamp: int = datetime_to_timestamp(timezone_now())
    re_map_foreign_keys_internal(
        records, "records", "realm_id", related_table="realm", id_field=True
    )
    if not processing_emojis and not processing_realm_icons:
        re_map_foreign_keys_internal(
            records, "records", "user_profile_id", related_table="user_profile", id_field=True
        )
    if processing_emojis:
        filename_to_has_original: Dict[str, bool] = {record["file_name"]: False for record in records}
        for record in records:
            if record["s3_path"].endswith(".original"):
                filename_to_has_original[record["file_name"]] = True
        if records and "author" in records[0]:
            re_map_foreign_keys_internal(
                records, "records", "author", related_table="user_profile", id_field=False
            )
    s3_uploads: bool = settings.LOCAL_UPLOADS_DIR is None
    if s3_uploads:
        if processing_avatars or processing_emojis or processing_realm_icons:
            bucket_name: str = settings.S3_AVATAR_BUCKET
        else:
            bucket_name: str = settings.S3_AUTH_UPLOADS_BUCKET
        bucket = get_bucket(bucket_name)
    for count, record in enumerate(records, 1):
        if processing_avatars:
            relative_path: str = user_avatar_base_path_from_ids(
                record["user_profile_id"], record["avatar_version"], record["realm_id"]
            )
            if record["s3_path"].endswith(".original"):
                relative_path += ".original"
            else:
                relative_path = upload_backend.get_avatar_path(relative_path, medium=False)
        elif processing_emojis:
            relative_path = RealmEmoji.PATH_ID_TEMPLATE.format(
                realm_id=record["realm_id"], emoji_file_name=record["file_name"]
            )
            if record["s3_path"].endswith(".original"):
                relative_path += ".original"
            record["last_modified"] = timestamp
        elif processing_realm_icons:
            icon_name: str = os.path.basename(record["path"])
            relative_path = os.path.join(str(record["realm_id"]), "realm", icon_name)
            record["last_modified"] = timestamp
        else:
            relative_path = upload_backend.generate_message_upload_path(
                str(record["realm_id"]), sanitize_name(os.path.basename(record["path"]))
            )
            path_maps["old_attachment_path_to_new_path"][record["s3_path"]] = relative_path
            path_maps["new_attachment_path_to_old_path"][relative_path] = record["s3_path"]
            path_maps["new_attachment_path_to_local_data_path"][relative_path] = os.path.join(import_dir, record["path"])
        if s3_uploads:
            key = bucket.Object(relative_path)
            metadata: Dict[str, str] = {}
            if "user_profile_id" not in record:
                assert default_user_profile_id is not None
                metadata["user_profile_id"] = str(default_user_profile_id)
            else:
                user_profile_id: int = int(record["user_profile_id"])
                if user_profile_id in ID_MAP["user_profile"]:
                    logging.info("Uploaded by ID mapped user: %s!", user_profile_id)
                    user_profile_id = ID_MAP["user_profile"][user_profile_id]
                user_profile = get_user_profile_by_id(user_profile_id)
                metadata["user_profile_id"] = str(user_profile.id)
            if "last_modified" in record:
                metadata["orig_last_modified"] = str(record["last_modified"])
            metadata["realm_id"] = str(record["realm_id"])
            content_type = record.get("content_type")
            if content_type is None:
                content_type = guess_type(record["s3_path"])[0]
                if content_type is None:
                    content_type = "application/octet-stream"
            key.upload_file(
                Filename=os.path.join(import_dir, record["path"]),
                ExtraArgs={"ContentType": content_type, "Metadata": metadata},
            )
        else:
            assert settings.LOCAL_UPLOADS_DIR is not None
            assert settings.LOCAL_AVATARS_DIR is not None
            assert settings.LOCAL_FILES_DIR is not None
            if processing_avatars or processing_emojis or processing_realm_icons:
                file_path: str = os.path.join(settings.LOCAL_AVATARS_DIR, relative_path)
            else:
                file_path = os.path.join(settings.LOCAL_FILES_DIR, relative_path)
            orig_file_path: str = os.path.join(import_dir, record["path"])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy(orig_file_path, file_path)
        if count % 1000 == 0:
            logging.info("Processed %s/%s uploads", count, len(records))
    if processing_avatars or processing_emojis:
        if processing_avatars:
            process_func: Callable[[Dict[str, Any]], None] = process_avatars
        else:
            process_func = partial(process_emojis, import_dir, default_user_profile_id, filename_to_has_original)
        if processes == 1:
            for record in records:
                process_func(record)
        else:
            connection.close()
            _cache = cache._cache  # type: ignore[attr-defined]
            assert isinstance(_cache, bmemcached.Client)
            _cache.disconnect_all()
            with ProcessPoolExecutor(max_workers=processes) as executor:
                futures = [executor.submit(process_func, record) for record in records]
                for future in as_completed(futures):
                    future.result()


def disable_restricted_authentication_methods(data: TableData) -> None:
    realm_authentication_methods: List[Dict[str, Any]] = data["zerver_realmauthenticationmethod"]
    non_restricted_methods: List[Dict[str, Any]] = []
    for auth_method in realm_authentication_methods:
        if AUTH_BACKEND_NAME_MAP[auth_method["name"]].available_for_cloud_plans is None:
            non_restricted_methods.append(auth_method)
        else:
            logging.warning("Dropped restricted authentication method: %s", auth_method["name"])
    data["zerver_realmauthenticationmethod"] = non_restricted_methods


def do_import_realm(import_dir: Path, subdomain: str, processes: int = 1) -> Realm:
    logging.info("Importing realm dump %s", import_dir)
    if not os.path.exists(import_dir):
        raise Exception("Missing import directory!")
    migration_status_filename: str = os.path.join(import_dir, "migration_status.json")
    if not os.path.exists(migration_status_filename):
        raise Exception(
            "Missing migration_status.json file! Make sure you're using the same Zulip version as the exported realm."
        )
    logging.info("Checking migration status of exported realm")
    with open(migration_status_filename) as f:
        migration_status: MigrationStatusJson = orjson.loads(f.read())
    check_migration_status(migration_status)
    realm_data_filename: str = os.path.join(import_dir, "realm.json")
    if not os.path.exists(realm_data_filename):
        raise Exception("Missing realm.json file!")
    if not server_initialized():
        create_internal_realm()
    logging.info("Importing realm data from %s", realm_data_filename)
    with open(realm_data_filename, "rb") as f:
        data: TableData = orjson.loads(f.read())
    data["zerver_userprofile"] += data["zerver_userprofile_mirrordummy"]
    del data["zerver_userprofile_mirrordummy"]
    data["zerver_userprofile"].sort(key=lambda r: r["id"])
    remove_denormalized_recipient_column_from_data(data)
    sort_by_date: bool = data.get("sort_by_date", False)
    bulk_import_client(data, Client, "zerver_client")
    internal_realm: Realm = get_realm(settings.SYSTEM_BOT_REALM)
    crossrealm_user_ids: set[int] = set()
    for item in data["zerver_userprofile_crossrealm"]:
        logging.info("Adding to ID map: %s %s", item["id"], get_system_bot(item["email"], internal_realm.id).id)
        new_user_id: int = get_system_bot(item["email"], internal_realm.id).id
        update_id_map(table="user_profile", old_id=item["id"], new_id=new_user_id)
        crossrealm_user_ids.add(new_user_id)
        new_recipient_id: int = Recipient.objects.get(type=Recipient.PERSONAL, type_id=new_user_id).id
        update_id_map(table="recipient", old_id=item["recipient_id"], new_id=new_recipient_id)
    update_model_ids(Realm, data, "realm")
    update_model_ids(Stream, data, "stream")
    update_model_ids(UserProfile, data, "user_profile")
    if "zerver_usergroup" in data:
        update_model_ids(UserGroup, data, "usergroup")
    if "zerver_presencesequence" in data:
        update_model_ids(PresenceSequence, data, "presencesequence")
    re_map_foreign_keys(data, "zerver_realm", "moderation_request_channel", related_table="stream")
    re_map_foreign_keys(data, "zerver_realm", "new_stream_announcements_stream", related_table="stream")
    re_map_foreign_keys(data, "zerver_realm", "signup_announcements_stream", related_table="stream")
    re_map_foreign_keys(data, "zerver_realm", "zulip_update_announcements_stream", related_table="stream")
    if "zerver_usergroup" in data:
        for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
            re_map_foreign_keys(data, "zerver_realm", setting_name, related_table="usergroup")
    fix_datetime_fields(data, "zerver_realm")
    data["zerver_realm"][0]["string_id"] = subdomain
    data["zerver_realm"][0]["name"] = subdomain
    realm_properties: Dict[str, Any] = dict(**data["zerver_realm"][0])
    realm_properties["deactivated"] = True
    realm_properties["push_notifications_enabled"] = sends_notifications_directly()
    with transaction.atomic(durable=True):
        realm: Realm = Realm(**realm_properties)
        if "zerver_usergroup" not in data:
            for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
                setattr(realm, setting_name + "_id", -1)
        realm.save()
        if "zerver_presencesequence" in data:
            re_map_foreign_keys(data, "zerver_presencesequence", "realm", related_table="realm")
            bulk_import_model(data, PresenceSequence)
        else:
            PresenceSequence.objects.create(realm=realm, last_update_id=0)
        named_user_group_id_to_creator_id: Dict[int, Any] = {}
        if "zerver_usergroup" in data:
            re_map_foreign_keys(data, "zerver_usergroup", "realm", related_table="realm")
            bulk_import_model(data, UserGroup)
            if "zerver_namedusergroup" in data:
                re_map_foreign_keys(data, "zerver_namedusergroup", "creator", related_table="user_profile")
                fix_datetime_fields(data, "zerver_namedusergroup")
                re_map_foreign_keys(data, "zerver_namedusergroup", "usergroup_ptr", related_table="usergroup")
                re_map_foreign_keys(data, "zerver_namedusergroup", "realm_for_sharding", related_table="realm")
                for group in data["zerver_namedusergroup"]:
                    creator_id = group.pop("creator_id", None)
                    named_user_group_id_to_creator_id[group["id"]] = creator_id
                for setting_name in NamedUserGroup.GROUP_PERMISSION_SETTINGS:
                    re_map_foreign_keys(data, "zerver_namedusergroup", setting_name, related_table="usergroup")
                bulk_import_named_user_groups(data)
        system_groups_name_dict: Dict[str, NamedUserGroup] | None = None
        if "zerver_usergroup" not in data:
            system_groups_name_dict = create_system_user_groups_for_realm(realm)
        fix_datetime_fields(data, "zerver_stream")
        re_map_foreign_keys(data, "zerver_stream", "realm", related_table="realm")
        re_map_foreign_keys(data, "zerver_stream", "creator", related_table="user_profile")
        stream_id_to_creator_id: Dict[int, Any] = {}
        for stream in data["zerver_stream"]:
            creator_id = stream.pop("creator_id", None)
            stream_id_to_creator_id[stream["id"]] = creator_id
        if system_groups_name_dict is not None:
            fix_stream_permission_group_settings(data, system_groups_name_dict)
        else:
            for setting_name in Stream.stream_permission_group_settings:
                re_map_foreign_keys(data, "zerver_stream", setting_name, related_table="usergroup")
        for stream in data["zerver_stream"]:
            stream["rendered_description"] = render_stream_description(stream["description"], realm)
        bulk_import_model(data, Stream)
        if "zerver_usergroup" not in data:
            set_default_for_realm_permission_group_settings(realm)
    update_message_foreign_keys(import_dir=import_dir, sort_by_date=sort_by_date)
    fix_datetime_fields(data, "zerver_userprofile")
    re_map_foreign_keys(data, "zerver_userprofile", "realm", related_table="realm")
    re_map_foreign_keys(data, "zerver_userprofile", "bot_owner", related_table="user_profile")
    re_map_foreign_keys(data, "zerver_userprofile", "default_sending_stream", related_table="stream")
    re_map_foreign_keys(data, "zerver_userprofile", "default_events_register_stream", related_table="stream")
    re_map_foreign_keys(data, "zerver_userprofile", "last_active_message_id", related_table="message", id_field=True)
    for user_profile_dict in data["zerver_userprofile"]:
        user_profile_dict["password"] = None
        user_profile_dict["api_key"] = generate_api_key()
        del user_profile_dict["user_permissions"]
        del user_profile_dict["groups"]
        if "short_name" in user_profile_dict:
            del user_profile_dict["short_name"]
    user_profiles: List[UserProfile] = [UserProfile(**item) for item in data["zerver_userprofile"]]
    for user_profile in user_profiles:
        validate_email(user_profile.delivery_email)
        validate_email(user_profile.email)
        user_profile.set_unusable_password()
        user_profile.tos_version = UserProfile.TOS_VERSION_BEFORE_FIRST_LOGIN
    UserProfile.objects.bulk_create(user_profiles)
    streams = Stream.objects.filter(id__in=list(stream_id_to_creator_id.keys()))
    for stream in streams:
        stream.creator_id = stream_id_to_creator_id[stream.id]
    Stream.objects.bulk_update(streams, ["creator_id"])
    if "zerver_namedusergroup" in data:
        named_user_groups = NamedUserGroup.objects.filter(id__in=list(named_user_group_id_to_creator_id.keys()))
        for group in named_user_groups:
            group.creator_id = named_user_group_id_to_creator_id[group.id]
        NamedUserGroup.objects.bulk_update(named_user_groups, ["creator_id"])
    re_map_foreign_keys(data, "zerver_defaultstream", "stream", related_table="stream")
    re_map_foreign_keys(data, "zerver_realmemoji", "author", related_table="user_profile")
    if settings.BILLING_ENABLED:
        disable_restricted_authentication_methods(data)
    for table_name, model, related_table in realm_tables:
        re_map_foreign_keys(data, table_name, "realm", related_table="realm")
        update_model_ids(model, data, related_table)
        bulk_import_model(data, model)
    first_user_profile: UserProfile | None = (
        UserProfile.objects.filter(realm=realm, is_active=True, role=UserProfile.ROLE_REALM_OWNER)
        .order_by("id")
        .first()
    )
    for realm_emoji in RealmEmoji.objects.filter(realm=realm):
        if realm_emoji.author_id is None:
            assert first_user_profile is not None
            realm_emoji.author_id = first_user_profile.id
            realm_emoji.save(update_fields=["author_id"])
    if "zerver_huddle" in data:
        update_model_ids(DirectMessageGroup, data, "huddle")
    re_map_foreign_keys(data, "zerver_recipient", "type_id", related_table="stream", recipient_field=True, id_field=True)
    re_map_foreign_keys(data, "zerver_recipient", "type_id", related_table="user_profile", recipient_field=True, id_field=True)
    re_map_foreign_keys(data, "zerver_recipient", "type_id", related_table="huddle", recipient_field=True, id_field=True)
    update_model_ids(Recipient, data, "recipient")
    bulk_import_model(data, Recipient)
    bulk_set_users_or_streams_recipient_fields(Stream, Stream.objects.filter(realm=realm))
    bulk_set_users_or_streams_recipient_fields(UserProfile, UserProfile.objects.filter(realm=realm))
    re_map_foreign_keys(data, "zerver_subscription", "user_profile", related_table="user_profile")
    get_direct_message_groups_from_subscription(data, "zerver_subscription")
    re_map_foreign_keys(data, "zerver_subscription", "recipient", related_table="recipient")
    update_model_ids(Subscription, data, "subscription")
    fix_subscriptions_is_user_active_column(data, user_profiles, crossrealm_user_ids)
    bulk_import_model(data, Subscription)
    if "zerver_realmauditlog" in data:
        fix_datetime_fields(data, "zerver_realmauditlog")
        re_map_foreign_keys(data, "zerver_realmauditlog", "realm", related_table="realm")
        re_map_foreign_keys(data, "zerver_realmauditlog", "modified_user", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_realmauditlog", "acting_user", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_realmauditlog", "modified_stream", related_table="stream")
        re_map_foreign_keys(data, "zerver_realmauditlog", "modified_user_group", related_table="usergroup")
        update_model_ids(RealmAuditLog, data, "realmauditlog")
        bulk_import_model(data, RealmAuditLog)
    else:
        logging.info("about to call create_subscription_events")
        create_subscription_events(data=data, realm_id=realm.id)
        logging.info("done with create_subscription_events")
    if not RealmAuditLog.objects.filter(realm=realm, event_type=AuditLogEventType.REALM_CREATED).exists():
        RealmAuditLog.objects.create(
            realm=realm,
            event_type=AuditLogEventType.REALM_CREATED,
            event_time=realm.date_created,
            backfilled=True,
        )
    if "zerver_huddle" in data:
        process_direct_message_group_hash(data, "zerver_huddle")
        bulk_import_model(data, DirectMessageGroup)
        for direct_message_group in DirectMessageGroup.objects.filter(recipient=None):
            recipient = Recipient.objects.get(
                type=Recipient.DIRECT_MESSAGE_GROUP, type_id=direct_message_group.id
            )
            direct_message_group.recipient = recipient
            direct_message_group.save(update_fields=["recipient"])
    if "zerver_alertword" in data:
        re_map_foreign_keys(data, "zerver_alertword", "user_profile", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_alertword", "realm", related_table="realm")
        update_model_ids(AlertWord, data, "alertword")
        bulk_import_model(data, AlertWord)
    if "zerver_savedsnippet" in data:
        re_map_foreign_keys(data, "zerver_savedsnippet", "user_profile", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_savedsnippet", "realm", related_table="realm")
        update_model_ids(SavedSnippet, data, "savedsnippet")
        bulk_import_model(data, SavedSnippet)
    if "zerver_onboardingstep" in data:
        fix_datetime_fields(data, "zerver_onboardingstep")
        re_map_foreign_keys(data, "zerver_onboardingstep", "user", related_table="user_profile")
        update_model_ids(OnboardingStep, data, "onboardingstep")
        bulk_import_model(data, OnboardingStep)
    if "zerver_usertopic" in data:
        fix_datetime_fields(data, "zerver_usertopic")
        re_map_foreign_keys(data, "zerver_usertopic", "user_profile", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_usertopic", "stream", related_table="stream")
        re_map_foreign_keys(data, "zerver_usertopic", "recipient", related_table="recipient")
        update_model_ids(UserTopic, data, "usertopic")
        bulk_import_model(data, UserTopic)
    if "zerver_muteduser" in data:
        fix_datetime_fields(data, "zerver_muteduser")
        re_map_foreign_keys(data, "zerver_muteduser", "user_profile", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_muteduser", "muted_user", related_table="user_profile")
        update_model_ids(MutedUser, data, "muteduser")
        bulk_import_model(data, MutedUser)
    if "zerver_service" in data:
        re_map_foreign_keys(data, "zerver_service", "user_profile", related_table="user_profile")
        fix_service_tokens(data, "zerver_service")
        update_model_ids(Service, data, "service")
        bulk_import_model(data, Service)
    if "zerver_usergroup" in data:
        re_map_foreign_keys(data, "zerver_usergroupmembership", "user_group", related_table="usergroup")
        re_map_foreign_keys(data, "zerver_usergroupmembership", "user_profile", related_table="user_profile")
        update_model_ids(UserGroupMembership, data, "usergroupmembership")
        bulk_import_model(data, UserGroupMembership)
        re_map_foreign_keys(data, "zerver_groupgroupmembership", "supergroup", related_table="usergroup")
        re_map_foreign_keys(data, "zerver_groupgroupmembership", "subgroup", related_table="usergroup")
        update_model_ids(GroupGroupMembership, data, "groupgroupmembership")
        bulk_import_model(data, GroupGroupMembership)
    if system_groups_name_dict is not None:
        add_users_to_system_user_groups(realm, user_profiles, system_groups_name_dict)
    if "zerver_botstoragedata" in data:
        re_map_foreign_keys(data, "zerver_botstoragedata", "bot_profile", related_table="user_profile")
        update_model_ids(BotStorageData, data, "botstoragedata")
        bulk_import_model(data, BotStorageData)
    if "zerver_botconfigdata" in data:
        re_map_foreign_keys(data, "zerver_botconfigdata", "bot_profile", related_table="user_profile")
        update_model_ids(BotConfigData, data, "botconfigdata")
        bulk_import_model(data, BotConfigData)
    if "zerver_realmuserdefault" in data:
        re_map_foreign_keys(data, "zerver_realmuserdefault", "realm", related_table="realm")
        update_model_ids(RealmUserDefault, data, "realmuserdefault")
        bulk_import_model(data, RealmUserDefault)
    if not RealmUserDefault.objects.filter(realm=realm).exists():
        RealmUserDefault.objects.create(realm=realm)
    fix_datetime_fields(data, "zerver_userpresence")
    re_map_foreign_keys(data, "zerver_userpresence", "user_profile", related_table="user_profile")
    re_map_foreign_keys(data, "zerver_userpresence", "realm", related_table="realm")
    update_model_ids(UserPresence, data, "user_presence")
    bulk_import_model(data, UserPresence)
    fix_datetime_fields(data, "zerver_useractivity")
    re_map_foreign_keys(data, "zerver_useractivity", "user_profile", related_table="user_profile")
    re_map_foreign_keys(data, "zerver_useractivity", "client", related_table="client")
    update_model_ids(UserActivity, data, "useractivity")
    bulk_import_model(data, UserActivity)
    fix_datetime_fields(data, "zerver_useractivityinterval")
    re_map_foreign_keys(data, "zerver_useractivityinterval", "user_profile", related_table="user_profile")
    update_model_ids(UserActivityInterval, data, "useractivityinterval")
    bulk_import_model(data, UserActivityInterval)
    re_map_foreign_keys(data, "zerver_customprofilefield", "realm", related_table="realm")
    update_model_ids(CustomProfileField, data, "customprofilefield")
    bulk_import_model(data, CustomProfileField)
    re_map_foreign_keys(data, "zerver_customprofilefieldvalue", "user_profile", related_table="user_profile")
    re_map_foreign_keys(data, "zerver_customprofilefieldvalue", "field", related_table="customprofilefield")
    fix_customprofilefield(data)
    update_model_ids(CustomProfileFieldValue, data, "customprofilefieldvalue")
    bulk_import_model(data, CustomProfileFieldValue)
    import_uploads(
        realm,
        os.path.join(import_dir, "avatars"),
        processes,
        default_user_profile_id=None,
        processing_avatars=True,
    )
    import_uploads(
        realm,
        os.path.join(import_dir, "uploads"),
        processes,
        default_user_profile_id=None,
    )
    if os.path.exists(os.path.join(import_dir, "emoji")):
        import_uploads(
            realm,
            os.path.join(import_dir, "emoji"),
            processes,
            default_user_profile_id=first_user_profile.id if first_user_profile else None,
            processing_emojis=True,
        )
    if os.path.exists(os.path.join(import_dir, "realm_icons")):
        import_uploads(
            realm,
            os.path.join(import_dir, "realm_icons"),
            processes,
            default_user_profile_id=first_user_profile.id if first_user_profile else None,
            processing_realm_icons=True,
        )
    sender_map: Dict[int, Record] = {user["id"]: user for user in data["zerver_userprofile"]}
    attachments_file: str = os.path.join(import_dir, "attachment.json")
    if not os.path.exists(attachments_file):
        raise Exception("Missing attachment.json file!")
    fix_attachments_data(attachment_data := orjson.loads(open(attachments_file, "rb").read()))
    create_image_attachments(realm, attachment_data)
    map_messages_to_attachments(attachment_data)
    import_message_data(realm=realm, sender_map=sender_map, import_dir=import_dir)
    if "zerver_onboardingusermessage" in data:
        fix_bitfield_keys(data, "zerver_onboardingusermessage", "flags")
        re_map_foreign_keys(data, "zerver_onboardingusermessage", "realm", related_table="realm")
        re_map_foreign_keys(data, "zerver_onboardingusermessage", "message", related_table="message")
        update_model_ids(OnboardingUserMessage, data, "onboardingusermessage")
        bulk_import_model(data, OnboardingUserMessage)
    if "zerver_scheduledmessage" in data:
        fix_datetime_fields(data, "zerver_scheduledmessage")
        re_map_foreign_keys(data, "zerver_scheduledmessage", "sender", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_scheduledmessage", "recipient", related_table="recipient")
        re_map_foreign_keys(data, "zerver_scheduledmessage", "sending_client", related_table="client")
        re_map_foreign_keys(data, "zerver_scheduledmessage", "stream", related_table="stream")
        re_map_foreign_keys(data, "zerver_scheduledmessage", "realm", related_table="realm")
        re_map_foreign_keys(data, "zerver_scheduledmessage", "delivered_message", related_table="message")
        fix_upload_links(data, "zerver_scheduledmessage")
        fix_message_rendered_content(
            realm=realm,
            sender_map=sender_map,
            messages=data["zerver_scheduledmessage"],
        )
        update_model_ids(ScheduledMessage, data, "scheduledmessage")
        bulk_import_model(data, ScheduledMessage)
    re_map_foreign_keys(data, "zerver_reaction", "message", related_table="message")
    re_map_foreign_keys(data, "zerver_reaction", "user_profile", related_table="user_profile")
    re_map_realm_emoji_codes(data, table_name="zerver_reaction")
    update_model_ids(Reaction, data, "reaction")
    bulk_import_model(data, Reaction)
    update_first_message_id_query = SQL(
        """
    UPDATE zerver_stream
    SET first_message_id = subquery.first_message_id
    FROM (
        SELECT r.type_id id, min(m.id) first_message_id
        FROM zerver_message m
        JOIN zerver_recipient r ON
        r.id = m.recipient_id
        WHERE r.type = 2 AND m.realm_id = %(realm_id)s
        GROUP BY r.type_id
        ) AS subquery
    WHERE zerver_stream.id = subquery.id
    """
    )
    with connection.cursor() as cursor:
        cursor.execute(update_first_message_id_query, {"realm_id": realm.id})
    if "zerver_userstatus" in data:
        fix_datetime_fields(data, "zerver_userstatus")
        re_map_foreign_keys(data, "zerver_userstatus", "user_profile", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_userstatus", "client", related_table="client")
        update_model_ids(UserStatus, data, "userstatus")
        re_map_realm_emoji_codes(data, table_name="zerver_userstatus")
        bulk_import_model(data, UserStatus)
    logging.info("Importing attachment data from %s", attachments_file)
    import_attachments(attachment_data)
    import_analytics_data(
        realm=realm, import_dir=import_dir, crossrealm_user_ids=crossrealm_user_ids
    )
    if settings.BILLING_ENABLED:
        do_change_realm_plan_type(realm, Realm.PLAN_TYPE_LIMITED, acting_user=None)
    else:
        do_change_realm_plan_type(realm, Realm.PLAN_TYPE_SELF_HOSTED, acting_user=None)
    realm.deactivated = data["zerver_realm"][0]["deactivated"]
    realm.save()
    if not realm.deactivated:
        number_of_days: int = Stream.LAST_ACTIVITY_DAYS_BEFORE_FOR_ACTIVE
        date_days_ago: datetime = timezone_now() - timedelta(days=number_of_days)
        update_stream_active_status_for_realm(realm, date_days_ago)
    RealmAuditLog.objects.create(
        realm=realm,
        event_type=AuditLogEventType.REALM_IMPORTED,
        event_time=timezone_now(),
        extra_data={RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(realm)},
    )
    maybe_enqueue_audit_log_upload(realm)
    is_realm_imported_from_other_zulip_server: bool = RealmAuditLog.objects.filter(
        realm=realm, event_type=AuditLogEventType.REALM_EXPORTED
    ).exists()
    if not is_realm_imported_from_other_zulip_server:
        send_zulip_update_announcements_to_realm(
            realm, skip_delay=False, realm_imported_from_other_product=True
        )
    return realm


def update_message_foreign_keys(import_dir: Path, sort_by_date: bool) -> None:
    old_id_list: List[int] = get_incoming_message_ids(import_dir=import_dir, sort_by_date=sort_by_date)
    count: int = len(old_id_list)
    new_id_list: List[int] = allocate_ids(model_class=Message, count=count)
    for old_id, new_id in zip(old_id_list, new_id_list, strict=False):
        update_id_map(table="message", old_id=old_id, new_id=new_id)


def get_incoming_message_ids(import_dir: Path, sort_by_date: bool) -> List[int]:
    if sort_by_date:
        tups: List[Tuple[int, int]] = []
    else:
        message_ids: List[int] = []
    dump_file_id: int = 1
    while True:
        message_filename: str = os.path.join(import_dir, f"messages-{dump_file_id:06}.json")
        if not os.path.exists(message_filename):
            break
        with open(message_filename, "rb") as f:
            data = orjson.loads(f.read())
        del data["zerver_usermessage"]
        for row in data["zerver_message"]:
            message_id: int = row["id"]
            if sort_by_date:
                date_sent: int = int(row["date_sent"])
                tup = (date_sent, message_id)
                tups.append(tup)
            else:
                message_ids.append(message_id)
        dump_file_id += 1
    if sort_by_date:
        tups.sort()
        message_ids = [tup[1] for tup in tups]
    return message_ids


def import_message_data(realm: Realm, sender_map: Dict[int, Record], import_dir: Path) -> None:
    dump_file_id: int = 1
    while True:
        message_filename: str = os.path.join(import_dir, f"messages-{dump_file_id:06}.json")
        if not os.path.exists(message_filename):
            break
        with open(message_filename, "rb") as f:
            data: TableData = orjson.loads(f.read())
        logging.info("Importing message dump %s", message_filename)
        re_map_foreign_keys(data, "zerver_message", "sender", related_table="user_profile")
        re_map_foreign_keys(data, "zerver_message", "recipient", related_table="recipient")
        re_map_foreign_keys(data, "zerver_message", "sending_client", related_table="client")
        fix_datetime_fields(data, "zerver_message")
        fix_upload_links(data, "zerver_message")
        message_id_map: Dict[int, int] = ID_MAP["message"]
        for row in data["zerver_message"]:
            del row["realm"]
            row["realm_id"] = realm.id
            row["id"] = message_id_map[row["id"]]
        for row in data["zerver_usermessage"]:
            assert row["message"] in message_id_map
        fix_message_rendered_content(
            realm=realm,
            sender_map=sender_map,
            messages=data["zerver_message"],
        )
        logging.info("Successfully rendered Markdown for message batch")
        fix_message_edit_history(
            realm=realm, sender_map=sender_map, messages=data["zerver_message"]
        )
        bulk_import_model(data, Message)
        re_map_foreign_keys(data, "zerver_usermessage", "message", related_table="message")
        re_map_foreign_keys(data, "zerver_usermessage", "user_profile", related_table="user_profile")
        fix_bitfield_keys(data, "zerver_usermessage", "flags")
        bulk_import_user_message_data(data, dump_file_id)
        dump_file_id += 1


def import_attachments(data: TableData) -> None:
    fix_datetime_fields(data, "zerver_attachment")
    re_map_foreign_keys(data, "zerver_attachment", "owner", related_table="user_profile")
    re_map_foreign_keys(data, "zerver_attachment", "realm", related_table="realm")
    parent_model = Attachment
    parent_db_table_name: str = "zerver_attachment"
    parent_singular: str = "attachment"
    parent_id: str = "attachment_id"
    update_model_ids(parent_model, data, "attachment")
    def format_m2m_data(
        child_singular: str, child_plural: str, m2m_table_name: str, child_id: str
    ) -> Tuple[str, List[Record], str]:
        m2m_rows: List[Record] = [
            {
                parent_singular: parent_row["id"],
                child_singular: ID_MAP[child_singular][fk_id],
            }
            for parent_row in data[parent_db_table_name]
            for fk_id in parent_row[child_plural]
        ]
        m2m_data: TableData = {m2m_table_name: m2m_rows}
        convert_to_id_fields(m2m_data, m2m_table_name, parent_singular)
        convert_to_id_fields(m2m_data, m2m_table_name, child_singular)
        m2m_rows = m2m_data[m2m_table_name]
        for parent_row in data[parent_db_table_name]:
            del parent_row[child_plural]
        return m2m_table_name, m2m_rows, child_id
    messages_m2m_tuple: Tuple[str, List[Record], str] = format_m2m_data(
        "message", "messages", "zerver_attachment_messages", "message_id"
    )
    scheduled_messages_m2m_tuple: Tuple[str, List[Record], str] = format_m2m_data(
        "scheduledmessage",
        "scheduled_messages",
        "zerver_attachment_scheduled_messages",
        "scheduledmessage_id",
    )
    bulk_import_model(data, parent_model)
    with connection.cursor() as cursor:
        for m2m_table_name, m2m_rows, child_id in [messages_m2m_tuple, scheduled_messages_m2m_tuple]:
            sql_template = SQL(
                """
                INSERT INTO {m2m_table_name} ({parent_id}, {child_id}) VALUES %s
            """
            ).format(
                m2m_table_name=Identifier(m2m_table_name),
                parent_id=Identifier(parent_id),
                child_id=Identifier(child_id),
            )
            tups = [(row[parent_id], row[child_id]) for row in m2m_rows]
            execute_values(cursor.cursor, sql_template, tups)
            logging.info("Successfully imported M2M table %s", m2m_table_name)


def fix_attachments_data(attachment_data: TableData) -> None:
    for attachment in attachment_data["zerver_attachment"]:
        attachment["path_id"] = path_maps["old_attachment_path_to_new_path"][attachment["path_id"]]
        if attachment.get("content_type") is None:
            guessed_content_type = guess_type(attachment["path_id"])[0]
            if guessed_content_type in THUMBNAIL_ACCEPT_IMAGE_TYPES:
                attachment["content_type"] = guessed_content_type


def create_image_attachments(realm: Realm, attachment_data: TableData) -> None:
    for attachment in attachment_data["zerver_attachment"]:
        if attachment["content_type"] not in THUMBNAIL_ACCEPT_IMAGE_TYPES:
            continue
        path_id: str = attachment["path_id"]
        content_type: str = attachment["content_type"]
        local_filename: str = path_maps["new_attachment_path_to_local_data_path"][path_id]
        pyvips_source = pyvips.Source.new_from_file(local_filename)
        maybe_thumbnail(pyvips_source, content_type, path_id, realm.id, skip_events=True)


def import_analytics_data(realm: Realm, import_dir: Path, crossrealm_user_ids: set[int]) -> None:
    analytics_filename: str = os.path.join(import_dir, "analytics.json")
    if not os.path.exists(analytics_filename):
        return
    logging.info("Importing analytics data from %s", analytics_filename)
    with open(analytics_filename, "rb") as f:
        data: TableData = orjson.loads(f.read())
    fix_datetime_fields(data, "analytics_realmcount")
    re_map_foreign_keys(data, "analytics_realmcount", "realm", related_table="realm")
    update_model_ids(RealmCount, data, "analytics_realmcount")
    bulk_import_model(data, RealmCount)
    fix_datetime_fields(data, "analytics_usercount")
    re_map_foreign_keys(data, "analytics_usercount", "realm", related_table="realm")
    re_map_foreign_keys(data, "analytics_usercount", "user", related_table="user_profile")
    data["analytics_usercount"] = [row for row in data["analytics_usercount"] if row["user_id"] not in crossrealm_user_ids]
    update_model_ids(UserCount, data, "analytics_usercount")
    bulk_import_model(data, UserCount)
    fix_datetime_fields(data, "analytics_streamcount")
    re_map_foreign_keys(data, "analytics_streamcount", "realm", related_table="realm")
    re_map_foreign_keys(data, "analytics_streamcount", "stream", related_table="stream")
    update_model_ids(StreamCount, data, "analytics_streamcount")
    bulk_import_model(data, StreamCount)


def add_users_to_system_user_groups(
    realm: Realm,
    user_profiles: List[UserProfile],
    system_groups_name_dict: Dict[str, NamedUserGroup],
) -> None:
    full_members_system_group: NamedUserGroup = NamedUserGroup.objects.get(
        name=SystemGroups.FULL_MEMBERS,
        realm=realm,
        is_system_group=True,
    )
    role_system_groups_dict: Dict[int, NamedUserGroup] = dict()
    for role in NamedUserGroup.SYSTEM_USER_GROUP_ROLE_MAP:
        group_name = NamedUserGroup.SYSTEM_USER_GROUP_ROLE_MAP[role]["name"]
        role_system_groups_dict[role] = system_groups_name_dict[group_name]
    usergroup_memberships: List[UserGroupMembership] = []
    for user_profile in user_profiles:
        user_group: NamedUserGroup = role_system_groups_dict[user_profile.role]
        usergroup_memberships.append(UserGroupMembership(user_profile=user_profile, user_group=user_group))
        if user_profile.role == UserProfile.ROLE_MEMBER and not user_profile.is_provisional_member:
            usergroup_memberships.append(
                UserGroupMembership(user_profile=user_profile, user_group=full_members_system_group)
            )
    UserGroupMembership.objects.bulk_create(usergroup_memberships)
    now: datetime = timezone_now()
    RealmAuditLog.objects.bulk_create(
        RealmAuditLog(
            realm=realm,
            modified_user=membership.user_profile,
            modified_user_group=membership.user_group.named_user_group,
            event_type=AuditLogEventType.USER_GROUP_DIRECT_USER_MEMBERSHIP_ADDED,
            event_time=now,
            acting_user=None,
        )
        for membership in usergroup_memberships
    )


ZULIP_CLOUD_ONLY_APP_NAMES: List[str] = ["zilencer", "corporate"]


def check_migration_status(exported_migration_status: MigrationStatusJson) -> None:
    mismatched_migrations_log: Dict[str, str] = {}
    local_showmigrations = get_migration_status(close_connection_when_done=False)
    local_migration_status: MigrationStatusJson = MigrationStatusJson(
        migrations_by_app=parse_migration_status(local_showmigrations), zulip_version=ZULIP_VERSION
    )
    exported_primary_version: str = exported_migration_status["zulip_version"].split(".")[0]
    local_primary_version: str = local_migration_status["zulip_version"].split(".")[0]
    if exported_primary_version != local_primary_version:
        raise CommandError(
            "Error: Export was generated on a different Zulip major version.\n"
            f"Export version: {exported_migration_status['zulip_version']}\n"
            f"Server version: {local_migration_status['zulip_version']}"
        )
    exported_migrations_by_app: Dict[str, Any] = exported_migration_status["migrations_by_app"]
    local_migrations_by_app: Dict[str, Any] = local_migration_status["migrations_by_app"]
    all_apps = set(exported_migrations_by_app.keys()).union(set(local_migrations_by_app.keys()))
    for app in all_apps:
        exported_app_migrations = exported_migrations_by_app.get(app)
        local_app_migrations = local_migrations_by_app.get(app)
        if app in ZULIP_CLOUD_ONLY_APP_NAMES and (local_app_migrations is None or exported_app_migrations is None):
            continue
        if not exported_app_migrations:
            logging.warning("This server has '%s' app installed, but exported realm does not.", app)
        elif not local_app_migrations:
            logging.warning("Exported realm has '%s' app installed, but this server does not.", app)
        elif set(local_app_migrations) != set(exported_app_migrations):
            diff = list(
                unified_diff(
                    sorted(exported_app_migrations), sorted(local_app_migrations), lineterm="", n=1
                )
            )
            mismatched_migrations_log[f"\n'{app}' app:\n"] = "\n".join(diff[3:])
    if mismatched_migrations_log:
        sorted_error_log: List[str] = [f"{key}{value}" for key, value in sorted(mismatched_migrations_log.items())]
        error_message = (
            "Error: Export was generated on a different Zulip version.\n"
            f"Export version: {exported_migration_status['zulip_version']}\n"
            f"Server version: {local_migration_status['zulip_version']}\n"
            "\n"
            "Database formats differ between the exported realm and this server.\n"
            "Printing migrations that differ between the versions:\n"
            "--- exported realm\n"
            "+++ this server"
        ) + "\n".join(sorted_error_log)
        raise CommandError(error_message)
