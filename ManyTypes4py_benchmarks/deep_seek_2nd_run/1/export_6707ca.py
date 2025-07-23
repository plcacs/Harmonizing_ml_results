import glob
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from datetime import datetime
from functools import cache
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, TypedDict, Dict, List, Set, Tuple, Union
from urllib.parse import urlsplit
import orjson
from django.apps import apps
from django.conf import settings
from django.db.models import Exists, OuterRef, Q, QuerySet
from django.forms.models import model_to_dict
from django.utils.timezone import is_naive as timezone_is_naive
from django.utils.timezone import now as timezone_now
import zerver.lib.upload
from analytics.models import RealmCount, StreamCount, UserCount
from scripts.lib.zulip_tools import overwrite_symlink
from version import ZULIP_VERSION
from zerver.lib.avatar_hash import user_avatar_base_path_from_ids
from zerver.lib.migration_status import MigrationStatusJson, get_migration_status, parse_migration_status
from zerver.lib.pysa import mark_sanitized
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.upload.s3 import get_bucket
from zerver.models import AlertWord, Attachment, BotConfigData, BotStorageData, Client, CustomProfileField, CustomProfileFieldValue, DefaultStream, DirectMessageGroup, GroupGroupMembership, Message, MutedUser, NamedUserGroup, OnboardingStep, OnboardingUserMessage, Reaction, Realm, RealmAuditLog, RealmAuthenticationMethod, RealmDomain, RealmEmoji, RealmExport, RealmFilter, RealmPlayground, RealmUserDefault, Recipient, ScheduledMessage, Service, Stream, Subscription, UserActivity, UserActivityInterval, UserGroup, UserGroupMembership, UserMessage, UserPresence, UserProfile, UserStatus, UserTopic
from zerver.models.presence import PresenceSequence
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_realm
from zerver.models.users import get_system_bot, get_user_profile_by_id

if TYPE_CHECKING:
    from mypy_boto3_s3.service_resource import Object

Record = Dict[str, Any]
TableName = str
TableData = Dict[TableName, List[Record]]
Field = str
Path = str
Context = Dict[str, Any]
FilterArgs = Dict[str, Any]
IdSource = Tuple[TableName, Field]
SourceFilter = Callable[[Record], bool]
CustomFetch = Callable[[TableData, Context], None]

class MessagePartial(TypedDict):
    pass

MESSAGE_BATCH_CHUNK_SIZE: int = 1000
ALL_ZULIP_TABLES: Set[str] = {'analytics_fillstate', 'analytics_installationcount', 'analytics_realmcount', 'analytics_streamcount', 'analytics_usercount', 'otp_static_staticdevice', 'otp_static_statictoken', 'otp_totp_totpdevice', 'social_auth_association', 'social_auth_code', 'social_auth_nonce', 'social_auth_partial', 'social_auth_usersocialauth', 'two_factor_phonedevice', 'zerver_alertword', 'zerver_archivedattachment', 'zerver_archivedattachment_messages', 'zerver_archivedmessage', 'zerver_archivedusermessage', 'zerver_attachment', 'zerver_attachment_messages', 'zerver_attachment_scheduled_messages', 'zerver_archivedreaction', 'zerver_archivedsubmessage', 'zerver_archivetransaction', 'zerver_botconfigdata', 'zerver_botstoragedata', 'zerver_channelemailaddress', 'zerver_client', 'zerver_customprofilefield', 'zerver_customprofilefieldvalue', 'zerver_defaultstream', 'zerver_defaultstreamgroup', 'zerver_defaultstreamgroup_streams', 'zerver_draft', 'zerver_emailchangestatus', 'zerver_groupgroupmembership', 'zerver_huddle', 'zerver_imageattachment', 'zerver_message', 'zerver_missedmessageemailaddress', 'zerver_multiuseinvite', 'zerver_multiuseinvite_streams', 'zerver_multiuseinvite_groups', 'zerver_namedusergroup', 'zerver_onboardingstep', 'zerver_onboardingusermessage', 'zerver_preregistrationrealm', 'zerver_preregistrationuser', 'zerver_preregistrationuser_streams', 'zerver_preregistrationuser_groups', 'zerver_presencesequence', 'zerver_pushdevicetoken', 'zerver_reaction', 'zerver_realm', 'zerver_realmauditlog', 'zerver_realmauthenticationmethod', 'zerver_realmdomain', 'zerver_realmemoji', 'zerver_realmexport', 'zerver_realmfilter', 'zerver_realmplayground', 'zerver_realmreactivationstatus', 'zerver_realmuserdefault', 'zerver_recipient', 'zerver_savedsnippet', 'zerver_scheduledemail', 'zerver_scheduledemail_users', 'zerver_scheduledmessage', 'zerver_scheduledmessagenotificationemail', 'zerver_service', 'zerver_stream', 'zerver_submessage', 'zerver_subscription', 'zerver_useractivity', 'zerver_useractivityinterval', 'zerver_usergroup', 'zerver_usergroupmembership', 'zerver_usermessage', 'zerver_userpresence', 'zerver_userprofile', 'zerver_userprofile_groups', 'zerver_userprofile_user_permissions', 'zerver_userstatus', 'zerver_usertopic', 'zerver_muteduser'}
NON_EXPORTED_TABLES: Set[str] = {'zerver_emailchangestatus', 'zerver_multiuseinvite', 'zerver_multiuseinvite_streams', 'zerver_multiuseinvite_groups', 'zerver_preregistrationrealm', 'zerver_preregistrationuser', 'zerver_preregistrationuser_streams', 'zerver_preregistrationuser_groups', 'zerver_realmreactivationstatus', 'zerver_missedmessageemailaddress', 'zerver_scheduledmessagenotificationemail', 'zerver_pushdevicetoken', 'zerver_userprofile_groups', 'zerver_userprofile_user_permissions', 'zerver_scheduledemail', 'zerver_scheduledemail_users', 'two_factor_phonedevice', 'otp_static_staticdevice', 'otp_static_statictoken', 'otp_totp_totpdevice', 'zerver_archivedmessage', 'zerver_archivedusermessage', 'zerver_archivedattachment', 'zerver_archivedattachment_messages', 'zerver_archivedreaction', 'zerver_archivedsubmessage', 'zerver_archivetransaction', 'social_auth_association', 'social_auth_code', 'social_auth_nonce', 'social_auth_partial', 'social_auth_usersocialauth', 'analytics_installationcount', 'analytics_fillstate', 'zerver_defaultstreamgroup', 'zerver_defaultstreamgroup_streams', 'zerver_submessage', 'zerver_draft', 'zerver_imageattachment', 'zerver_channelemailaddress'}
IMPLICIT_TABLES: Set[str] = {'zerver_attachment_messages', 'zerver_attachment_scheduled_messages'}
ATTACHMENT_TABLES: Set[str] = {'zerver_attachment'}
MESSAGE_TABLES: Set[str] = {'zerver_message', 'zerver_usermessage', 'zerver_reaction'}
ANALYTICS_TABLES: Set[str] = {'analytics_realmcount', 'analytics_streamcount', 'analytics_usercount'}
DATE_FIELDS: Dict[str, List[str]] = {'analytics_installationcount': ['end_time'], 'analytics_realmcount': ['end_time'], 'analytics_streamcount': ['end_time'], 'analytics_usercount': ['end_time'], 'zerver_attachment': ['create_time'], 'zerver_message': ['last_edit_time', 'date_sent'], 'zerver_muteduser': ['date_muted'], 'zerver_realmauditlog': ['event_time'], 'zerver_realm': ['date_created'], 'zerver_realmexport': ['date_requested', 'date_started', 'date_succeeded', 'date_failed', 'date_deleted'], 'zerver_scheduledmessage': ['scheduled_timestamp'], 'zerver_stream': ['date_created'], 'zerver_namedusergroup': ['date_created'], 'zerver_useractivityinterval': ['start', 'end'], 'zerver_useractivity': ['last_visit'], 'zerver_onboardingstep': ['timestamp'], 'zerver_userpresence': ['last_active_time', 'last_connected_time'], 'zerver_userprofile': ['date_joined', 'last_login', 'last_reminder'], 'zerver_userprofile_mirrordummy': ['date_joined', 'last_login', 'last_reminder'], 'zerver_userstatus': ['timestamp'], 'zerver_usertopic': ['last_updated']}

def sanity_check_output(data: TableData) -> None:
    target_models = [*apps.get_app_config('analytics').get_models(include_auto_created=True), *apps.get_app_config('django_otp').get_models(include_auto_created=True), *apps.get_app_config('otp_static').get_models(include_auto_created=True), *apps.get_app_config('otp_totp').get_models(include_auto_created=True), *apps.get_app_config('phonenumber').get_models(include_auto_created=True), *apps.get_app_config('social_django').get_models(include_auto_created=True), *apps.get_app_config('two_factor').get_models(include_auto_created=True), *apps.get_app_config('zerver').get_models(include_auto_created=True)]
    all_tables_db = {model._meta.db_table for model in target_models}
    error_message = f"\n    It appears you've added a new database table, but haven't yet\n    registered it in ALL_ZULIP_TABLES and the related declarations\n    in {__file__} for what to include in data exports.\n    "
    assert all_tables_db == ALL_ZULIP_TABLES, error_message
    assert NON_EXPORTED_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    assert IMPLICIT_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    assert ATTACHMENT_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    assert ANALYTICS_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    tables = set(ALL_ZULIP_TABLES)
    tables -= NON_EXPORTED_TABLES
    tables -= IMPLICIT_TABLES
    tables -= MESSAGE_TABLES
    tables -= ATTACHMENT_TABLES
    tables -= ANALYTICS_TABLES
    for table in tables:
        if table not in data:
            logging.warning('??? NO DATA EXPORTED FOR TABLE %s!!!', table)

def write_data_to_file(output_file: str, data: Any) -> None:
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_PASSTHROUGH_DATETIME))
    logging.info('Finished writing %s', output_file)

def write_table_data(output_file: str, data: TableData) -> None:
    for table in data.values():
        table.sort(key=lambda row: row['id'])
    assert output_file.endswith('.json')
    write_data_to_file(output_file, data)

def write_records_json_file(output_dir: str, records: List[Dict[str, Any]]) -> None:
    records.sort(key=lambda record: record['path'])
    output_file = os.path.join(output_dir, 'records.json')
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(records, option=orjson.OPT_INDENT_2))
    logging.info('Finished writing %s', output_file)

def make_raw(query: QuerySet, exclude: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows = []
    for instance in query:
        data = model_to_dict(instance, exclude=exclude)
        for field in instance._meta.many_to_many:
            if exclude is not None and field.name in exclude:
                continue
            value = data[field.name]
            data[field.name] = [row.id for row in value]
        rows.append(data)
    return rows

def floatify_datetime_fields(data: TableData, table: str) -> None:
    for item in data[table]:
        for field in DATE_FIELDS[table]:
            dt = item[field]
            if dt is None:
                continue
            assert isinstance(dt, datetime)
            assert not timezone_is_naive(dt)
            item[field] = dt.timestamp()

class Config:
    def __init__(
        self,
        table: Optional[str] = None,
        model: Optional[Any] = None,
        normal_parent: Optional['Config'] = None,
        virtual_parent: Optional['Config'] = None,
        filter_args: Optional[FilterArgs] = None,
        custom_fetch: Optional[CustomFetch] = None,
        custom_tables: Optional[List[str]] = None,
        concat_and_destroy: Optional[List[str]] = None,
        id_source: Optional[IdSource] = None,
        source_filter: Optional[SourceFilter] = None,
        include_rows: Optional[str] = None,
        use_all: bool = False,
        is_seeded: bool = False,
        exclude: Optional[List[str]] = None
    ) -> None:
        assert table or custom_tables
        self.table = table
        self.model = model
        self.normal_parent = normal_parent
        self.virtual_parent = virtual_parent
        self.filter_args = filter_args
        self.include_rows = include_rows
        self.use_all = use_all
        self.is_seeded = is_seeded
        self.exclude = exclude
        self.custom_fetch = custom_fetch
        self.custom_tables = custom_tables
        self.concat_and_destroy = concat_and_destroy
        self.id_source = id_source
        self.source_filter = source_filter
        self.children: List['Config'] = []
        if self.include_rows:
            assert self.include_rows.endswith('_id__in')
        if self.custom_fetch:
            assert self.custom_fetch.__name__.startswith('custom_fetch_')
            if self.normal_parent is not None:
                raise AssertionError('\n                    If you have a custom fetcher, then specify\n                    your parent as a virtual_parent.\n                    ')
        if normal_parent is not None:
            self.parent = normal_parent
        else:
            self.parent = None
        if virtual_parent is not None and normal_parent is not None:
            raise AssertionError('\n                If you specify a normal_parent, please\n                do not create a virtual_parent.\n                ')
        if normal_parent is not None:
            normal_parent.children.append(self)
        elif virtual_parent is not None:
            virtual_parent.children.append(self)
        elif is_seeded is None:
            raise AssertionError('\n                You must specify a parent if you are\n                not using is_seeded.\n                ')
        if self.id_source is not None:
            if self.virtual_parent is None:
                raise AssertionError('\n                    You must specify a virtual_parent if you are\n                    using id_source.')
            if self.id_source[0] != self.virtual_parent.table:
                raise AssertionError(f'\n                    Configuration error.  To populate {self.table}, you\n                    want data from {self.id_source[0]}, but that differs from\n                    the table name of your virtual parent ({self.virtual_parent.table}),\n                    which suggests you many not have set up\n                    the ordering correctly.  You may simply\n                    need to assign a virtual_parent, or there\n                    may be deeper issues going on.')

def export_from_config(
    response: TableData,
    config: Config,
    seed_object: Optional[Any] = None,
    context: Optional[Context] = None
) -> None:
    table = config.table
    parent = config.parent
    model = config.model
    if context is None:
        context = {}
    if config.custom_tables:
        exported_tables = config.custom_tables
    else:
        assert table is not None, '\n            You must specify config.custom_tables if you\n            are not specifying config.table'
        exported_tables = [table]
    for t in exported_tables:
        logging.info('Exporting via export_from_config:  %s', t)
    rows = None
    if config.is_seeded:
        rows = [seed_object]
    elif config.custom_fetch:
        config.custom_fetch(response, context)
        if config.custom_tables:
            for t in config.custom_tables:
                if t not in response:
                    raise AssertionError(f'Custom fetch failed to populate {t}')
    elif config.concat_and_destroy:
        data = []
        for t in config.concat_and_destroy:
            data += response[t]
            del response[t]
            logging.info('Deleted temporary %s', t)
        assert table is not None
        response[table] = data
    elif config.use_all:
        assert model is not None
        query = model.objects.all()
        rows = list(query)
    elif config.normal_parent:
        model = config.model
        assert parent is not None
        assert parent.table is not None
        assert config.include_rows is not None
        parent_ids = [r['id'] for r in response[parent.table]]
        filter_params = {config.include_rows: parent_ids}
        if config.filter_args is not None:
            filter_params.update(config.filter_args)
        assert model is not None
        try:
            query = model.objects.filter(**filter_params)
        except Exception:
            print(f'\n                Something about your Config seems to make it difficult\n                to construct a query.\n\n                table: {table}\n                parent: {parent.table}\n\n               