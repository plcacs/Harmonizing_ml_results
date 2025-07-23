import collections
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from difflib import unified_diff
from typing import Any, Dict, List, Set, Tuple, Optional, Union, DefaultDict, Callable
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
from zerver.lib.migration_status import MigrationStatusJson, get_migration_status, parse_migration_status
from zerver.lib.mime_types import guess_type
from zerver.lib.partial import partial
from zerver.lib.push_notifications import sends_notifications_directly
from zerver.lib.remote_server import maybe_enqueue_audit_log_upload
from zerver.lib.server_initialization import create_internal_realm, server_initialized
from zerver.lib.streams import get_stream_permission_default_group, render_stream_description, update_stream_active_status_for_realm
from zerver.lib.thumbnail import THUMBNAIL_ACCEPT_IMAGE_TYPES, BadImageError, get_user_upload_previews, maybe_thumbnail
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.upload import ensure_avatar_image, sanitize_name, upload_backend, upload_emoji_image
from zerver.lib.upload.s3 import get_bucket
from zerver.lib.user_counts import realm_user_count_by_role
from zerver.lib.user_groups import create_system_user_groups_for_realm
from zerver.lib.user_message import UserMessageLite, bulk_insert_ums
from zerver.lib.utils import generate_api_key, process_list_in_batches
from zerver.lib.zulip_update_announcements import send_zulip_update_announcements_to_realm
from zerver.models import AlertWord, Attachment, BotConfigData, BotStorageData, Client, CustomProfileField, CustomProfileFieldValue, DefaultStream, DirectMessageGroup, GroupGroupMembership, Message, MutedUser, NamedUserGroup, OnboardingStep, OnboardingUserMessage, Reaction, Realm, RealmAuditLog, RealmAuthenticationMethod, RealmDomain, RealmEmoji, RealmFilter, RealmPlayground, RealmUserDefault, Recipient, SavedSnippet, ScheduledMessage, Service, Stream, Subscription, UserActivity, UserActivityInterval, UserProfile, UserPresence, UserStatus, UserTopic
from zerver.models.groups import SystemGroups
from zerver.models.presence import PresenceSequence
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_realm
from zerver.models.recipients import get_direct_message_group_hash
from zerver.models.users import get_system_bot, get_user_profile_by_id
from zproject.backends import AUTH_BACKEND_NAME_MAP

realm_tables: List[Tuple[str, Any, str]] = [
    ('zerver_realmauthenticationmethod', RealmAuthenticationMethod, 'realmauthenticationmethod'),
    ('zerver_defaultstream', DefaultStream, 'defaultstream'),
    ('zerver_realmemoji', RealmEmoji, 'realmemoji'),
    ('zerver_realmdomain', RealmDomain, 'realmdomain'),
    ('zerver_realmfilter', RealmFilter, 'realmfilter'),
    ('zerver_realmplayground', RealmPlayground, 'realmplayground')
]

ID_MAP: Dict[str, Dict[int, int]] = {
    'alertword': {}, 'client': {}, 'user_profile': {}, 'huddle': {}, 'realm': {},
    'stream': {}, 'recipient': {}, 'subscription': {}, 'defaultstream': {},
    'onboardingstep': {}, 'presencesequence': {}, 'reaction': {},
    'realmauthenticationmethod': {}, 'realmemoji': {}, 'realmdomain': {},
    'realmfilter': {}, 'realmplayground': {}, 'message': {}, 'user_presence': {},
    'userstatus': {}, 'useractivity': {}, 'useractivityinterval': {},
    'usermessage': {}, 'customprofilefield': {}, 'customprofilefieldvalue': {},
    'attachment': {}, 'realmauditlog': {}, 'recipient_to_huddle_map': {},
    'usertopic': {}, 'muteduser': {}, 'service': {}, 'usergroup': {},
    'usergroupmembership': {}, 'groupgroupmembership': {}, 'botstoragedata': {},
    'botconfigdata': {}, 'analytics_realmcount': {}, 'analytics_streamcount': {},
    'analytics_usercount': {}, 'realmuserdefault': {}, 'scheduledmessage': {},
    'onboardingusermessage': {}
}

id_map_to_list: Dict[str, Dict[int, List[int]]] = {'huddle_to_user_list': {}}

path_maps: Dict[str, Dict[str, str]] = {
    'old_attachment_path_to_new_path': {},
    'new_attachment_path_to_old_path': {},
    'new_attachment_path_to_local_data_path': {}
}

message_id_to_attachments: Dict[str, DefaultDict[int, List[str]]] = {
    'zerver_message': collections.defaultdict(list),
    'zerver_scheduledmessage': collections.defaultdict(list)
}

def map_messages_to_attachments(data: TableData) -> None:
    for attachment in data['zerver_attachment']:
        for message_id in attachment['messages']:
            message_id_to_attachments['zerver_message'][message_id].append(attachment['path_id'])
        for scheduled_message_id in attachment['scheduled_messages']:
            message_id_to_attachments['zerver_scheduledmessage'][scheduled_message_id].append(attachment['path_id'])

def update_id_map(table: str, old_id: int, new_id: int) -> None:
    if table not in ID_MAP:
        raise Exception(f'\n            Table {table} is not initialized in ID_MAP, which could\n            mean that we have not thought through circular\n            dependencies.\n            ')
    ID_MAP[table][old_id] = new_id

def fix_datetime_fields(data: TableData, table: str) -> None:
    for item in data[table]:
        for field_name in DATE_FIELDS[table]:
            if item[field_name] is not None:
                item[field_name] = datetime.fromtimestamp(item[field_name], tz=timezone.utc)

def fix_upload_links(data: TableData, message_table: str) -> None:
    """
    Because the URLs for uploaded files encode the realm ID of the
    organization being imported (which is only determined at import
    time), we need to rewrite the URLs of links to uploaded files
    during the import process.

    Applied to attachments path_id found in messages of zerver_message and zerver_scheduledmessage tables.
    """
    for message in data[message_table]:
        if message['has_attachment'] is True:
            for attachment_path in message_id_to_attachments[message_table][message['id']]:
                old_path = path_maps['new_attachment_path_to_old_path'][attachment_path]
                message['content'] = message['content'].replace(old_path, attachment_path)
                if message['rendered_content']:
                    message['rendered_content'] = message['rendered_content'].replace(old_path, attachment_path)

def fix_stream_permission_group_settings(data: TableData, system_groups_name_dict: Dict[str, int]) -> None:
    table = get_db_table(Stream)
    for stream in data[table]:
        for setting_name in Stream.stream_permission_group_settings:
            if setting_name == 'can_send_message_group' and 'stream_post_policy' in stream:
                if stream['stream_post_policy'] == Stream.STREAM_POST_POLICY_MODERATORS:
                    stream[setting_name] = system_groups_name_dict[SystemGroups.MODERATORS]
                else:
                    stream[setting_name] = get_stream_permission_default_group(setting_name, system_groups_name_dict)
                del stream['stream_post_policy']
                continue
            stream[setting_name] = get_stream_permission_default_group(setting_name, system_groups_name_dict)

def create_subscription_events(data: TableData, realm_id: int) -> None:
    """
    When the export data doesn't contain the table `zerver_realmauditlog`,
    this function creates RealmAuditLog objects for `subscription_created`
    type event for all the existing Stream subscriptions.

    This is needed for all the export tools which do not include the
    table `zerver_realmauditlog` (e.g. Slack) because the appropriate
    data about when a user was subscribed is not exported by the third-party
    service.
    """
    all_subscription_logs = []
    event_last_message_id = get_last_message_id()
    event_time = timezone_now()
    recipient_id_to_stream_id = {d['id']: d['type_id'] for d in data['zerver_recipient'] if d['type'] == Recipient.STREAM}
    for sub in data['zerver_subscription']:
        recipient_id = sub['recipient_id']
        stream_id = recipient_id_to_stream_id.get(recipient_id)
        if stream_id is None:
            continue
        user_id = sub['user_profile_id']
        all_subscription_logs.append(RealmAuditLog(realm_id=realm_id, acting_user_id=user_id, modified_user_id=user_id, modified_stream_id=stream_id, event_last_message_id=event_last_message_id, event_time=event_time, event_type=AuditLogEventType.SUBSCRIPTION_CREATED))
    RealmAuditLog.objects.bulk_create(all_subscription_logs)

def fix_service_tokens(data: TableData, table: str) -> None:
    """
    The tokens in the services are created by 'generate_api_key'.
    As the tokens are unique, they should be re-created for the imports.
    """
    for item in data[table]:
        item['token'] = generate_api_key()

def process_direct_message_group_hash(data: TableData, table: str) -> None:
    """
    Build new direct message group hashes with the updated ids of the users
    """
    for direct_message_group in data[table]:
        user_id_list = id_map_to_list['huddle_to_user_list'][direct_message_group['id']]
        direct_message_group['huddle_hash'] = get_direct_message_group_hash(user_id_list)

def get_direct_message_groups_from_subscription(data: TableData, table: str) -> None:
    """
    Extract the IDs of the user_profiles involved in a direct message group from
    the subscription object
    This helps to generate a unique direct message group hash from the updated
    user_profile ids
    """
    id_map_to_list['huddle_to_user_list'] = {value: [] for value in ID_MAP['recipient_to_huddle_map'].values()}
    for subscription in data[table]:
        if subscription['recipient'] in ID_MAP['recipient_to_huddle_map']:
            direct_message_group_id = ID_MAP['recipient_to_huddle_map'][subscription['recipient']]
            id_map_to_list['huddle_to_user_list'][direct_message_group_id].append(subscription['user_profile_id'])

def fix_customprofilefield(data: TableData) -> None:
    """
    In CustomProfileField with 'field_type' like 'USER', the IDs need to be
    re-mapped.
    """
    field_type_USER_ids = {item['id'] for item in data['zerver_customprofilefield'] if item['field_type'] == CustomProfileField.USER}
    for item in data['zerver_customprofilefieldvalue']:
        if item['field_id'] in field_type_USER_ids:
            old_user_id_list = orjson.loads(item['value'])
            new_id_list = re_map_foreign_keys_many_to_many_internal(table='zerver_customprofilefieldvalue', field_name='value', related_table='user_profile', old_id_list=old_user_id_list)
            item['value'] = orjson.dumps(new_id_list).decode()

def fix_message_rendered_content(realm: Realm, sender_map: Dict[int, Dict[str, Any]], messages: List[Dict[str, Any]], content_key: str = 'content', rendered_content_key: str = 'rendered_content') -> None:
    """
    This function sets the rendered_content of the messages we're importing.
    """
    for message in messages:
        if content_key not in message:
            continue
        if message[rendered_content_key] is not None:
            soup = BeautifulSoup(message[rendered_content_key], 'html.parser')
            user_mentions = soup.findAll('span', {'class': 'user-mention'})
            if len(user_mentions) != 0:
                user_id_map = ID_MAP['user_profile']
                for mention in user_mentions:
                    if not mention.has_attr('data-user-id'):
                        continue
                    if mention['data-user-id'] == '*':
                        continue
                    old_user_id = int(mention['data-user-id'])
                    if old_user_id in user_id_map:
                        mention['data-user-id'] = str(user_id_map[old_user_id])
                message[rendered_content_key] = str(soup)
            stream_mentions = soup.findAll('a', {'class': 'stream'})
            if len(stream_mentions) != 0:
                stream_id_map = ID_MAP['stream']
                for mention in stream_mentions:
                    old_stream_id = int(mention['data-stream-id'])
                    if old_stream_id in stream_id_map:
                        mention['data-stream-id'] = str(stream_id_map[old_stream_id])
                message[rendered_content_key] = str(soup)
            user_group_mentions = soup.findAll('span', {'class': 'user-group-mention'})
            if len(user_group_mentions) != 0:
                user_group_id_map = ID_MAP['usergroup']
                for mention in user_group_mentions:
                    old_user_group_id = int(mention['data-user-group-id'])
                    if old_user_group_id in user_group_id_map:
                        mention['data-user-group-id'] = str(user_group_id_map[old_user_group_id])
                message[rendered_content_key] = str(soup)
            get_user_upload_previews(realm.id, message[content_key], lock=True)
            continue
        try:
            content = message[content_key]
            sender_id = message['sender_id']
            sender = sender_map[sender_id]
            sent_by_bot = sender['is_bot']
            translate_emoticons = sender['translate_emoticons']
            realm_alert_words_automaton = None
            rendered_content = markdown_convert(content=content, realm_alert_words_automaton=realm_alert_words_automaton, message_realm=realm, sent_by_bot=sent_by_bot, translate_emoticons=translate_emoticons).rendered_content
            message[rendered_content_key] = rendered_content
            if 'scheduled_timestamp' not in message:
                message['rendered_content_version'] = markdown_version
        except Exception:
            logging.warning('Error in Markdown rendering for message ID %s; continuing', message['id'])

def fix_message_edit_history(realm: Realm, sender_map: Dict[int, Dict[str, Any]], messages: List[Dict[str, Any]]) -> None:
    user_id_map = ID_MAP['user_profile']
    for message in messages:
        edit_history_json = message.get('edit_history')
        if not edit_history_json:
            continue
        edit_history = orjson.loads(edit_history_json)
        for edit_history_message_dict in edit_history:
            edit_history_message_dict['user_id'] = user_id_map[edit_history_message_dict['user_id']]
        fix_message_rendered_content(realm, sender_map, messages=edit_history, content_key='prev_content', rendered_content_key='prev_rendered_content')
        message['edit_history'] = orjson.dumps(edit_history).decode()

def current_table_ids(data: TableData, table: str) -> List[int]:
    """
    Returns the ids present in the current table
    """
    return [item['id'] for item in data[table]]

def idseq(model_class: Any, cursor: CursorWrapper) -> str:
    sequences = connection.introspection.get_sequences(cursor, model_class._meta.db_table)
    for sequence in sequences:
        if sequence['column'] == 'id':
            return sequence['name']
    raise Exception(f"No sequence found for 'id' of {model_class}")

def allocate_ids(model_class: Any, count: int) -> List[int]:
    """
    Increases the sequence number for a given table by the amount of objects being
    imported into that table. Hence, this gives a reserved range of IDs to import the
    converted Slack objects into the tables.
    """
    with connection.cursor() as cursor:
        sequence = idseq(model_class, cursor)
        cursor.execute('select nextval(%s) from generate_series(1, %s)', [sequence, count])
        query = cursor.fetchall()
    return [item[0] for item in query]

def convert_to_id_fields(data: TableData, table: str, field_name: str) -> None:
    """
    When Django gives us dict objects via model_to_dict, the foreign
    key fields are `foo`, but we want `foo_id` for the bulk insert.
    This function handles the simple case where we simply rename
    the fields.  For cases where we