#!/usr/bin/env python3
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
from typing import Any, Optional, TypeAlias, TypedDict, List, Dict, Set
from urllib.parse import urlsplit

import orjson
from django.apps import apps
from django.conf import settings
from django.db.models import Exists, OuterRef, Q
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
from zerver.models import (AlertWord, Attachment, BotConfigData, BotStorageData, Client, CustomProfileField,
                           CustomProfileFieldValue, DefaultStream, DirectMessageGroup, GroupGroupMembership, Message,
                           MutedUser, NamedUserGroup, OnboardingStep, OnboardingUserMessage, Reaction, Realm,
                           RealmAuditLog, RealmAuthenticationMethod, RealmDomain, RealmEmoji, RealmExport, RealmFilter,
                           RealmPlayground, RealmUserDefault, Recipient, ScheduledMessage, Service, Stream, Subscription,
                           UserActivity, UserActivityInterval, UserGroup, UserGroupMembership, UserMessage, UserPresence,
                           UserProfile, UserStatus, UserTopic)
from zerver.models.presence import PresenceSequence
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_realm
from zerver.models.users import get_system_bot, get_user_profile_by_id

Record: TypeAlias = Dict[str, Any]
TableName: TypeAlias = str
TableData: TypeAlias = Dict[TableName, List[Record]]
Field: TypeAlias = str
Path: TypeAlias = str
Context: TypeAlias = Dict[str, Any]
FilterArgs: TypeAlias = Dict[str, Any]
IdSource: TypeAlias = tuple[TableName, Field]
SourceFilter: TypeAlias = Callable[[Record], bool]
CustomFetch: TypeAlias = Callable[[TableData, Context], None]

class MessagePartial(TypedDict):
    pass

MESSAGE_BATCH_CHUNK_SIZE: int = 1000
ALL_ZULIP_TABLES: Set[str] = {'analytics_fillstate', 'analytics_installationcount', 'analytics_realmcount', 'analytics_streamcount', 'analytics_usercount', 'otp_static_staticdevice', 'otp_static_statictoken', 'otp_totp_totpdevice', 'social_auth_association', 'social_auth_code', 'social_auth_nonce', 'social_auth_partial', 'social_auth_usersocialauth', 'two_factor_phonedevice', 'zerver_alertword', 'zerver_archivedattachment', 'zerver_archivedattachment_messages', 'zerver_archivedmessage', 'zerver_archivedusermessage', 'zerver_attachment', 'zerver_attachment_messages', 'zerver_attachment_scheduled_messages', 'zerver_archivedreaction', 'zerver_archivedsubmessage', 'zerver_archivetransaction', 'zerver_botconfigdata', 'zerver_botstoragedata', 'zerver_channelemailaddress', 'zerver_client', 'zerver_customprofilefield', 'zerver_customprofilefieldvalue', 'zerver_defaultstream', 'zerver_defaultstreamgroup', 'zerver_defaultstreamgroup_streams', 'zerver_draft', 'zerver_emailchangestatus', 'zerver_groupgroupmembership', 'zerver_huddle', 'zerver_imageattachment', 'zerver_message', 'zerver_missedmessageemailaddress', 'zerver_multiuseinvite', 'zerver_multiuseinvite_streams', 'zerver_multiuseinvite_groups', 'zerver_namedusergroup', 'zerver_onboardingstep', 'zerver_onboardingusermessage', 'zerver_preregistrationrealm', 'zerver_preregistrationuser', 'zerver_preregistrationuser_streams', 'zerver_preregistrationuser_groups', 'zerver_presencesequence', 'zerver_pushdevicetoken', 'zerver_reaction', 'zerver_realm', 'zerver_realmauditlog', 'zerver_realmauthenticationmethod', 'zerver_realmdomain', 'zerver_realmemoji', 'zerver_realmexport', 'zerver_realmfilter', 'zerver_realmplayground', 'zerver_realmreactivationstatus', 'zerver_realmuserdefault', 'zerver_recipient', 'zerver_savedsnippet', 'zerver_scheduledemail', 'zerver_scheduledemail_users', 'zerver_scheduledmessage', 'zerver_scheduledmessagenotificationemail', 'zerver_service', 'zerver_stream', 'zerver_submessage', 'zerver_subscription', 'zerver_useractivity', 'zerver_useractivityinterval', 'zerver_usergroup', 'zerver_usergroupmembership', 'zerver_usermessage', 'zerver_userpresence', 'zerver_userprofile', 'zerver_userprofile_groups', 'zerver_userprofile_user_permissions', 'zerver_userstatus', 'zerver_usertopic', 'zerver_muteduser'}
NON_EXPORTED_TABLES: Set[str] = {'zerver_emailchangestatus', 'zerver_multiuseinvite', 'zerver_multiuseinvite_streams', 'zerver_multiuseinvite_groups', 'zerver_preregistrationrealm', 'zerver_preregistrationuser', 'zerver_preregistrationuser_streams', 'zerver_preregistrationuser_groups', 'zerver_realmreactivationstatus', 'zerver_missedmessageemailaddress', 'zerver_scheduledmessagenotificationemail', 'zerver_pushdevicetoken', 'zerver_userprofile_groups', 'zerver_userprofile_user_permissions', 'zerver_scheduledemail', 'zerver_scheduledemail_users', 'two_factor_phonedevice', 'otp_static_staticdevice', 'otp_static_statictoken', 'otp_totp_totpdevice', 'zerver_archivedmessage', 'zerver_archivedusermessage', 'zerver_archivedattachment', 'zerver_archivedattachment_messages', 'zerver_archivedreaction', 'zerver_archivedsubmessage', 'zerver_archivetransaction', 'social_auth_association', 'social_auth_code', 'social_auth_nonce', 'social_auth_partial', 'social_auth_usersocialauth', 'analytics_installationcount', 'analytics_fillstate', 'zerver_defaultstreamgroup', 'zerver_defaultstreamgroup_streams', 'zerver_submessage', 'zerver_draft', 'zerver_imageattachment', 'zerver_channelemailaddress'}
IMPLICIT_TABLES: Set[str] = {'zerver_attachment_messages', 'zerver_attachment_scheduled_messages'}
ATTACHMENT_TABLES: Set[str] = {'zerver_attachment'}
MESSAGE_TABLES: Set[str] = {'zerver_message', 'zerver_usermessage', 'zerver_reaction'}
ANALYTICS_TABLES: Set[str] = {'analytics_realmcount', 'analytics_streamcount', 'analytics_usercount'}
DATE_FIELDS: Dict[str, List[str]] = {
    'analytics_installationcount': ['end_time'],
    'analytics_realmcount': ['end_time'],
    'analytics_streamcount': ['end_time'],
    'analytics_usercount': ['end_time'],
    'zerver_attachment': ['create_time'],
    'zerver_message': ['last_edit_time', 'date_sent'],
    'zerver_muteduser': ['date_muted'],
    'zerver_realmauditlog': ['event_time'],
    'zerver_realm': ['date_created'],
    'zerver_realmexport': ['date_requested', 'date_started', 'date_succeeded', 'date_failed', 'date_deleted'],
    'zerver_scheduledmessage': ['scheduled_timestamp'],
    'zerver_stream': ['date_created'],
    'zerver_namedusergroup': ['date_created'],
    'zerver_useractivityinterval': ['start', 'end'],
    'zerver_useractivity': ['last_visit'],
    'zerver_onboardingstep': ['timestamp'],
    'zerver_userpresence': ['last_active_time', 'last_connected_time'],
    'zerver_userprofile': ['date_joined', 'last_login', 'last_reminder'],
    'zerver_userprofile_mirrordummy': ['date_joined', 'last_login', 'last_reminder'],
    'zerver_userstatus': ['timestamp'],
    'zerver_usertopic': ['last_updated']
}

def sanity_check_output(data: TableData) -> None:
    target_models = [*apps.get_app_config('analytics').get_models(include_auto_created=True),
                     *apps.get_app_config('django_otp').get_models(include_auto_created=True),
                     *apps.get_app_config('otp_static').get_models(include_auto_created=True),
                     *apps.get_app_config('otp_totp').get_models(include_auto_created=True),
                     *apps.get_app_config('phonenumber').get_models(include_auto_created=True),
                     *apps.get_app_config('social_django').get_models(include_auto_created=True),
                     *apps.get_app_config('two_factor').get_models(include_auto_created=True),
                     *apps.get_app_config('zerver').get_models(include_auto_created=True)]
    all_tables_db: Set[str] = {model._meta.db_table for model in target_models}
    error_message: str = f"\n    It appears you've added a new database table, but haven't yet\n    registered it in ALL_ZULIP_TABLES and the related declarations\n    in {__file__} for what to include in data exports.\n    "
    assert all_tables_db == ALL_ZULIP_TABLES, error_message
    assert NON_EXPORTED_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    assert IMPLICIT_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    assert ATTACHMENT_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    assert ANALYTICS_TABLES.issubset(ALL_ZULIP_TABLES), error_message
    tables: Set[str] = set(ALL_ZULIP_TABLES)
    tables -= NON_EXPORTED_TABLES
    tables -= IMPLICIT_TABLES
    tables -= MESSAGE_TABLES
    tables -= ATTACHMENT_TABLES
    tables -= ANALYTICS_TABLES
    for table in tables:
        if table not in data:
            logging.warning('??? NO DATA EXPORTED FOR TABLE %s!!!', table)

def write_data_to_file(output_file: str, data: Any) -> None:
    """
    IMPORTANT: You generally don't want to call this directly.

    Instead use one of the higher level helpers:

        write_table_data
        write_records_json_file

    The one place we call this directly is for message partials.
    """
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_PASSTHROUGH_DATETIME))
    logging.info('Finished writing %s', output_file)

def write_table_data(output_file: str, data: TableData) -> None:
    for table in data.values():
        table.sort(key=lambda row: row['id'])
    assert output_file.endswith('.json')
    write_data_to_file(output_file, data)

def write_records_json_file(output_dir: str, records: List[Record]) -> None:
    records.sort(key=lambda record: record['path'])
    output_file: str = os.path.join(output_dir, 'records.json')
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(records, option=orjson.OPT_INDENT_2))
    logging.info('Finished writing %s', output_file)

def make_raw(query: Iterable[Any], exclude: Optional[List[str]] = None) -> List[Record]:
    """
    Takes a Django query and returns a JSONable list
    of dictionaries corresponding to the database rows.
    """
    rows: List[Record] = []
    for instance in query:
        data: Record = model_to_dict(instance, exclude=exclude)
        '''
        In Django 1.11.5, model_to_dict evaluates the QuerySet of
        many-to-many field to give us a list of instances. We require
        a list of primary keys, so we get the primary keys from the
        instances below.
        '''
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
    """A Config object configures a single table for exporting (and, maybe
    some day importing as well).  This configuration defines what
    process needs to be followed to correctly extract the set of
    objects to export.
    """
    def __init__(self,
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
                 exclude: Optional[List[str]] = None) -> None:
        assert table or custom_tables
        self.table: Optional[str] = table
        self.model: Optional[Any] = model
        self.normal_parent: Optional[Config] = normal_parent
        self.virtual_parent: Optional[Config] = virtual_parent
        self.filter_args: Optional[FilterArgs] = filter_args
        self.include_rows: Optional[str] = include_rows
        self.use_all: bool = use_all
        self.is_seeded: bool = is_seeded
        self.exclude: Optional[List[str]] = exclude
        self.custom_fetch: Optional[CustomFetch] = custom_fetch
        self.custom_tables: Optional[List[str]] = custom_tables
        self.concat_and_destroy: Optional[List[str]] = concat_and_destroy
        self.id_source: Optional[IdSource] = id_source
        self.source_filter: Optional[SourceFilter] = source_filter
        self.children: List[Config] = []
        if self.include_rows:
            assert self.include_rows.endswith('_id__in')
        if self.custom_fetch:
            assert self.custom_fetch.__name__.startswith('custom_fetch_')
            if self.normal_parent is not None:
                raise AssertionError('\n                    If you have a custom fetcher, then specify\n                    your parent as a virtual_parent.\n                    ')
        if normal_parent is not None:
            self.parent: Optional[Config] = normal_parent
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

def export_from_config(response: TableData, config: Config, seed_object: Optional[Any] = None, context: Optional[Context] = None) -> None:
    table: Optional[str] = config.table
    parent: Optional[Config] = config.parent
    model: Optional[Any] = config.model
    if context is None:
        context = {}
    if config.custom_tables:
        exported_tables: List[str] = config.custom_tables
    else:
        assert table is not None, '\n            You must specify config.custom_tables if you\n            are not specifying config.table'
        exported_tables = [table]
    for t in exported_tables:
        logging.info('Exporting via export_from_config:  %s', t)
    rows: Optional[List[Any]] = None
    if config.is_seeded:
        rows = [seed_object]
    elif config.custom_fetch:
        config.custom_fetch(response, context)
        if config.custom_tables:
            for t in config.custom_tables:
                if t not in response:
                    raise AssertionError(f'Custom fetch failed to populate {t}')
    elif config.concat_and_destroy:
        data: List[Any] = []
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
        assert config.include_rows is not None
        parent_ids: List[Any] = [r['id'] for r in response[parent.table]]  # type: ignore
        filter_params: FilterArgs = {config.include_rows: parent_ids}
        if config.filter_args is not None:
            filter_params.update(config.filter_args)
        assert model is not None
        try:
            query = model.objects.filter(**filter_params)
        except Exception:
            print(f'\n                Something about your Config seems to make it difficult\n                to construct a query.\n\n                table: {table}\n                parent: {parent.table}\n\n                filter_params: {filter_params}\n                ')
            raise
        rows = list(query)
    elif config.id_source:
        assert model is not None
        child_table, field = config.id_source
        child_rows: List[Record] = response[child_table]
        if config.source_filter:
            child_rows = [r for r in child_rows if config.source_filter(r)]
        lookup_ids: List[Any] = [r[field] for r in child_rows]
        filter_params = dict(id__in=lookup_ids)
        if config.filter_args:
            filter_params.update(config.filter_args)
        query = model.objects.filter(**filter_params)
        rows = list(query)
    if rows is not None:
        assert table is not None
        response[table] = make_raw(rows, exclude=config.exclude)
    for t in exported_tables:
        if t in DATE_FIELDS:
            floatify_datetime_fields(response, t)
    for child_config in config.children:
        export_from_config(response=response, config=child_config, context=context)

def get_realm_config() -> Config:
    realm_config: Config = Config(table='zerver_realm', is_seeded=True)
    Config(table='zerver_realmauthenticationmethod', model=RealmAuthenticationMethod, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_presencesequence', model=PresenceSequence, normal_parent=realm_config, include_rows='realm_id__in')
    Config(custom_tables=['zerver_scheduledmessage'], virtual_parent=realm_config, custom_fetch=custom_fetch_scheduled_messages)
    Config(table='zerver_defaultstream', model=DefaultStream, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_customprofilefield', model=CustomProfileField, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_realmauditlog', virtual_parent=realm_config, custom_fetch=custom_fetch_realm_audit_logs_for_realm)
    Config(table='zerver_realmemoji', model=RealmEmoji, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_realmdomain', model=RealmDomain, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_realmexport', model=RealmExport, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_realmfilter', model=RealmFilter, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_realmplayground', model=RealmPlayground, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_client', model=Client, virtual_parent=realm_config, use_all=True)
    Config(table='zerver_realmuserdefault', model=RealmUserDefault, normal_parent=realm_config, include_rows='realm_id__in')
    Config(table='zerver_onboardingusermessage', model=OnboardingUserMessage, virtual_parent=realm_config, custom_fetch=custom_fetch_onboarding_usermessage)
    user_profile_config: Config = Config(custom_tables=['zerver_userprofile', 'zerver_userprofile_mirrordummy'],
                                          table='zerver_userprofile', virtual_parent=realm_config, custom_fetch=custom_fetch_user_profile)
    user_groups_config: Config = Config(table='zerver_usergroup', model=UserGroup, normal_parent=realm_config, include_rows='realm_id__in', exclude=['direct_members', 'direct_subgroups'])
    Config(table='zerver_namedusergroup', model=NamedUserGroup, normal_parent=realm_config, include_rows='realm_for_sharding_id__in', exclude=['realm', 'direct_members', 'direct_subgroups'])
    Config(custom_tables=['zerver_userprofile_crossrealm'], virtual_parent=user_profile_config, custom_fetch=custom_fetch_user_profile_cross_realm)
    Config(table='zerver_service', model=Service, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_botstoragedata', model=BotStorageData, normal_parent=user_profile_config, include_rows='bot_profile_id__in')
    Config(table='zerver_botconfigdata', model=BotConfigData, normal_parent=user_profile_config, include_rows='bot_profile_id__in')
    user_subscription_config: Config = Config(table='_user_subscription', model=Subscription, normal_parent=user_profile_config, filter_args={'recipient__type': Recipient.PERSONAL}, include_rows='user_profile_id__in')
    Config(table='_user_recipient', model=Recipient, virtual_parent=user_subscription_config, id_source=('_user_subscription', 'recipient'))
    stream_config: Config = Config(table='zerver_stream', model=Stream, normal_parent=realm_config, include_rows='realm_id__in')
    stream_recipient_config: Config = Config(table='_stream_recipient', model=Recipient, normal_parent=stream_config, include_rows='type_id__in', filter_args={'type': Recipient.STREAM})
    Config(table='_stream_subscription', model=Subscription, normal_parent=stream_recipient_config, include_rows='recipient_id__in')
    Config(custom_tables=['_huddle_recipient', '_huddle_subscription', 'zerver_huddle'], virtual_parent=user_profile_config, custom_fetch=custom_fetch_direct_message_groups)
    Config(table='zerver_recipient', virtual_parent=realm_config, concat_and_destroy=['_user_recipient', '_stream_recipient', '_huddle_recipient'])
    Config(table='zerver_subscription', virtual_parent=realm_config, concat_and_destroy=['_user_subscription', '_stream_subscription', '_huddle_subscription'])
    add_user_profile_child_configs(user_profile_config)
    return realm_config

def add_user_profile_child_configs(user_profile_config: Config) -> None:
    """
    We add tables here that are keyed by user, and for which
    we fetch rows using the same scheme whether we are
    exporting a realm or a single user.

    For any table where there is nuance between how you
    fetch for realms vs. single users, it's best to just
    keep things simple and have each caller maintain its
    own slightly different 4/5 line Config (while still
    possibly calling common code deeper in the stack).

    As of now, we do NOT include bot tables like Service.
    """
    Config(table='zerver_alertword', model=AlertWord, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_customprofilefieldvalue', model=CustomProfileFieldValue, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_muteduser', model=MutedUser, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_onboardingstep', model=OnboardingStep, normal_parent=user_profile_config, include_rows='user_id__in')
    Config(table='zerver_useractivity', model=UserActivity, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_useractivityinterval', model=UserActivityInterval, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_userpresence', model=UserPresence, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_userstatus', model=UserStatus, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    Config(table='zerver_usertopic', model=UserTopic, normal_parent=user_profile_config, include_rows='user_profile_id__in')

EXCLUDED_USER_PROFILE_FIELDS: List[str] = ['api_key', 'password', 'uuid']

def custom_fetch_user_profile(response: TableData, context: Context) -> None:
    realm = context['realm']
    exportable_user_ids: Optional[Set[int]] = context['exportable_user_ids']
    query = UserProfile.objects.filter(realm_id=realm.id).exclude(email__in=settings.CROSS_REALM_BOT_EMAILS)
    exclude = EXCLUDED_USER_PROFILE_FIELDS
    rows: List[Record] = make_raw(list(query), exclude=exclude)
    normal_rows: List[Record] = []
    dummy_rows: List[Record] = []
    for row in rows:
        if exportable_user_ids is not None:
            if row['id'] in exportable_user_ids:
                assert not row['is_mirror_dummy']
            else:
                row['is_mirror_dummy'] = True
                row['is_active'] = False
        if row.get('is_mirror_dummy'):
            dummy_rows.append(row)
        else:
            normal_rows.append(row)
    response['zerver_userprofile'] = normal_rows
    response['zerver_userprofile_mirrordummy'] = dummy_rows

def custom_fetch_user_profile_cross_realm(response: TableData, context: Context) -> None:
    realm = context['realm']
    response['zerver_userprofile_crossrealm'] = []
    bot_name_to_default_email: Dict[str, str] = {'NOTIFICATION_BOT': 'notification-bot@zulip.com', 'EMAIL_GATEWAY_BOT': 'emailgateway@zulip.com', 'WELCOME_BOT': 'welcome-bot@zulip.com'}
    if realm.string_id == settings.SYSTEM_BOT_REALM:
        return
    internal_realm = get_realm(settings.SYSTEM_BOT_REALM)
    for bot in settings.INTERNAL_BOTS:
        bot_name: str = bot['var_name']
        if bot_name not in bot_name_to_default_email:
            continue
        bot_email: str = bot['email_template'] % (settings.INTERNAL_BOT_DOMAIN,)
        bot_default_email: str = bot_name_to_default_email[bot_name]
        bot_user_id: int = get_system_bot(bot_email, internal_realm.id).id
        recipient_id: int = Recipient.objects.get(type_id=bot_user_id, type=Recipient.PERSONAL).id
        response['zerver_userprofile_crossrealm'].append(dict(email=bot_default_email, id=bot_user_id, recipient_id=recipient_id))

def fetch_attachment_data(response: TableData, realm_id: int, message_ids: Set[int], scheduled_message_ids: Set[int]) -> List[Attachment]:
    attachments: List[Attachment] = list(Attachment.objects.filter(Q(messages__in=message_ids) | Q(scheduled_messages__in=scheduled_message_ids), realm_id=realm_id).distinct())
    response['zerver_attachment'] = make_raw(attachments)
    floatify_datetime_fields(response, 'zerver_attachment')
    """
    We usually export most messages for the realm, but not
    quite ALL messages for the realm.  So, we need to
    clean up our attachment data to have correct
    values for response['zerver_attachment'][<n>]['messages'].

    Same reasoning applies to scheduled_messages.
    """
    for row in response['zerver_attachment']:
        filtered_message_ids: Set[int] = set(row['messages']).intersection(message_ids)
        row['messages'] = sorted(filtered_message_ids)
        filtered_scheduled_message_ids: Set[int] = set(row['scheduled_messages']).intersection(scheduled_message_ids)
        row['scheduled_messages'] = sorted(filtered_scheduled_message_ids)
    return attachments

def custom_fetch_realm_audit_logs_for_user(response: TableData, context: Context) -> None:
    """To be expansive, we include audit log entries for events that
    either modified the target user or where the target user modified
    something (E.g. if they changed the settings for a stream).
    """
    user = context['user']
    query = RealmAuditLog.objects.filter(Q(modified_user_id=user.id) | Q(acting_user_id=user.id))
    rows: List[Record] = make_raw(list(query))
    response['zerver_realmauditlog'] = rows

def fetch_reaction_data(response: TableData, message_ids: Set[int]) -> None:
    query = Reaction.objects.filter(message_id__in=list(message_ids))
    response['zerver_reaction'] = make_raw(list(query))

def custom_fetch_direct_message_groups(response: TableData, context: Context) -> None:
    realm = context['realm']
    user_profile_ids: Set[int] = {r['id'] for r in response['zerver_userprofile'] + response['zerver_userprofile_mirrordummy']}
    realm_direct_message_group_subs = Subscription.objects.select_related('recipient').filter(recipient__type=Recipient.DIRECT_MESSAGE_GROUP, user_profile__in=user_profile_ids)
    realm_direct_message_group_recipient_ids: Set[int] = {sub.recipient_id for sub in realm_direct_message_group_subs}
    unsafe_direct_message_group_recipient_ids: Set[int] = set()
    for sub in Subscription.objects.select_related('user_profile').filter(recipient__in=realm_direct_message_group_recipient_ids):
        if sub.user_profile.realm_id != realm.id:
            unsafe_direct_message_group_recipient_ids.add(sub.recipient_id)
    direct_message_group_subs = [sub for sub in realm_direct_message_group_subs if sub.recipient_id not in unsafe_direct_message_group_recipient_ids]
    direct_message_group_recipient_ids: Set[int] = {sub.recipient_id for sub in direct_message_group_subs}
    direct_message_group_ids: Set[int] = {sub.recipient.type_id for sub in direct_message_group_subs}
    direct_message_group_subscription_dicts: List[Record] = make_raw(direct_message_group_subs)
    direct_message_group_recipients: List[Record] = make_raw(Recipient.objects.filter(id__in=direct_message_group_recipient_ids))
    response['_huddle_recipient'] = direct_message_group_recipients
    response['_huddle_subscription'] = direct_message_group_subscription_dicts
    response['zerver_huddle'] = make_raw(DirectMessageGroup.objects.filter(id__in=direct_message_group_ids))

def custom_fetch_scheduled_messages(response: TableData, context: Context) -> None:
    """
    Simple custom fetch function to fetch only the ScheduledMessage objects that we're allowed to.
    """
    realm = context['realm']
    exportable_scheduled_message_ids: Set[int] = context['exportable_scheduled_message_ids']
    query = ScheduledMessage.objects.filter(realm=realm, id__in=exportable_scheduled_message_ids)
    rows: List[Record] = make_raw(list(query))
    response['zerver_scheduledmessage'] = rows

def custom_fetch_realm_audit_logs_for_realm(response: TableData, context: Context) -> None:
    """
    Simple custom fetch function to fix up .acting_user for some RealmAuditLog objects.

    Certain RealmAuditLog objects have an acting_user that is in a different .realm, due to
    the possibility of server administrators (typically with the .is_staff permission) taking
    certain actions to modify UserProfiles or Realms, which will set the .acting_user to
    the administrator's UserProfile, which can be in a different realm. Such an acting_user
    cannot be imported during organization import on another server, so we need to just set it
    to None.
    """
    realm = context['realm']
    query = RealmAuditLog.objects.filter(realm=realm).select_related('acting_user')
    realmauditlog_objects = list(query)
    for realmauditlog in realmauditlog_objects:
        if realmauditlog.acting_user is not None and realmauditlog.acting_user.realm_id != realm.id:
            realmauditlog.acting_user = None
    rows: List[Record] = make_raw(realmauditlog_objects)
    response['zerver_realmauditlog'] = rows

def custom_fetch_onboarding_usermessage(response: TableData, context: Context) -> None:
    realm = context['realm']
    response['zerver_onboardingusermessage'] = []
    onboarding_usermessage_query = OnboardingUserMessage.objects.filter(realm=realm)
    for onboarding_usermessage in onboarding_usermessage_query:
        onboarding_usermessage_obj: Record = model_to_dict(onboarding_usermessage)
        onboarding_usermessage_obj['flags_mask'] = onboarding_usermessage.flags.mask
        del onboarding_usermessage_obj['flags']
        response['zerver_onboardingusermessage'].append(onboarding_usermessage_obj)

def fetch_usermessages(realm: Realm, message_ids: Set[int], user_profile_ids: Set[int],
                       message_filename: str, export_full_with_consent: bool) -> List[Dict[str, Any]]:
    user_message_query = UserMessage.objects.filter(user_profile__realm=realm, message_id__in=message_ids)
    if export_full_with_consent:
        consented_user_ids: Set[int] = get_consented_user_ids(realm)
        user_profile_ids = consented_user_ids & user_profile_ids
    user_message_chunk: List[Dict[str, Any]] = []
    for user_message in user_message_query:
        if user_message.user_profile_id not in user_profile_ids:
            continue
        user_message_obj: Dict[str, Any] = model_to_dict(user_message)
        user_message_obj['flags_mask'] = user_message.flags.mask
        del user_message_obj['flags']
        user_message_chunk.append(user_message_obj)
    logging.info('Fetched UserMessages for %s', message_filename)
    return user_message_chunk

def export_usermessages_batch(input_path: str, output_path: str, export_full_with_consent: bool) -> None:
    """As part of the system for doing parallel exports, this runs on one
    batch of Message objects and adds the corresponding UserMessage
    objects. (This is called by the export_usermessage_batch
    management command).

    See write_message_partial_for_query for more context."""
    assert input_path.endswith(('.partial', '.locked'))
    assert output_path.endswith('.json')
    with open(input_path, 'rb') as input_file:
        input_data: Dict[str, Any] = orjson.loads(input_file.read())
    message_ids: Set[int] = {item['id'] for item in input_data['zerver_message']}
    user_profile_ids: Set[int] = set(input_data['zerver_userprofile_ids'])
    realm: Realm = Realm.objects.get(id=input_data['realm_id'])
    zerver_usermessage_data: List[Dict[str, Any]] = fetch_usermessages(realm, message_ids, user_profile_ids, output_path, export_full_with_consent)
    output_data: Dict[str, Any] = dict(zerver_message=input_data['zerver_message'], zerver_usermessage=zerver_usermessage_data)
    write_table_data(output_path, output_data)
    os.unlink(input_path)

def export_partial_message_files(realm: Realm, response: TableData, export_type: Any, chunk_size: int = MESSAGE_BATCH_CHUNK_SIZE, output_dir: Optional[str] = None) -> Set[int]:
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='zulip-export')

    def get_ids(records: List[Record]) -> Set[int]:
        return {x['id'] for x in records}
    user_ids_for_us: Set[int] = get_ids(response['zerver_userprofile'])
    ids_of_our_possible_senders: Set[int] = get_ids(response['zerver_userprofile'] + response['zerver_userprofile_mirrordummy'] + response['zerver_userprofile_crossrealm'])
    consented_user_ids: Set[int] = set()
    if export_type == RealmExport.EXPORT_FULL_WITH_CONSENT:
        consented_user_ids = get_consented_user_ids(realm)
    if export_type == RealmExport.EXPORT_PUBLIC:
        recipient_streams = Stream.objects.filter(realm=realm, invite_only=False)
        recipient_ids = Recipient.objects.filter(type=Recipient.STREAM, type_id__in=recipient_streams).values_list('id', flat=True)
        recipient_ids_for_us: Set[int] = get_ids(response['zerver_recipient']) & set(recipient_ids)
    elif export_type == RealmExport.EXPORT_FULL_WITH_CONSENT:
        public_streams = Stream.objects.filter(realm=realm, invite_only=False)
        public_stream_recipient_ids = Recipient.objects.filter(type=Recipient.STREAM, type_id__in=public_streams).values_list('id', flat=True)
        streams_with_protected_history_recipient_ids = Stream.objects.filter(realm=realm, history_public_to_subscribers=False).values_list('recipient_id', flat=True)
        consented_recipient_ids = Subscription.objects.filter(user_profile_id__in=consented_user_ids).values_list('recipient_id', flat=True)
        recipient_ids_set = set(public_stream_recipient_ids) | set(consented_recipient_ids) - set(streams_with_protected_history_recipient_ids)
        recipient_ids_for_us = get_ids(response['zerver_recipient']) & recipient_ids_set
    else:
        recipient_ids_for_us = get_ids(response['zerver_recipient'])
        consented_user_ids = user_ids_for_us
    if export_type == RealmExport.EXPORT_PUBLIC:
        messages_we_received = Message.objects.filter(realm_id=realm.id, sender__in=ids_of_our_possible_senders, recipient__in=recipient_ids_for_us)
        message_queries: List[Any] = [messages_we_received]
    else:
        message_queries = []
        messages_we_received = Message.objects.filter(realm_id=realm.id, sender__in=ids_of_our_possible_senders, recipient__in=recipient_ids_for_us)
        message_queries.append(messages_we_received)
        if export_type == RealmExport.EXPORT_FULL_WITH_CONSENT:
            has_usermessage_expression = Exists(UserMessage.objects.filter(user_profile_id__in=consented_user_ids, message_id=OuterRef('id')))
            messages_we_received_in_protected_history_streams = Message.objects.alias(has_usermessage=has_usermessage_expression).filter(
                realm_id=realm.id,
                sender__in=ids_of_our_possible_senders,
                recipient_id__in=set(consented_recipient_ids) & set(streams_with_protected_history_recipient_ids),
                has_usermessage=True)
            message_queries.append(messages_we_received_in_protected_history_streams)
        ids_of_non_exported_possible_recipients = ids_of_our_possible_senders - consented_user_ids
        recipients_for_them = Recipient.objects.filter(type=Recipient.PERSONAL, type_id__in=ids_of_non_exported_possible_recipients).values('id')
        recipient_ids_for_them: Set[int] = get_ids(list(recipients_for_them))
        messages_we_sent_to_them = Message.objects.filter(realm_id=realm.id, sender__in=consented_user_ids, recipient__in=recipient_ids_for_them)
        message_queries.append(messages_we_sent_to_them)
    all_message_ids: Set[int] = set()
    for message_query in message_queries:
        message_ids: Set[int] = set(get_id_list_gently_from_database(base_query=message_query, id_field='id'))
        assert len(message_ids.intersection(all_message_ids)) == 0
        all_message_ids |= message_ids
    message_id_chunks: List[List[int]] = chunkify(sorted(all_message_ids), chunk_size=MESSAGE_BATCH_CHUNK_SIZE)
    write_message_partials(realm=realm, message_id_chunks=message_id_chunks, output_dir=output_dir, user_profile_ids=user_ids_for_us)
    return all_message_ids

def write_message_partials(*, realm: Realm, message_id_chunks: List[List[int]], output_dir: str, user_profile_ids: Set[int]) -> None:
    dump_file_id: int = 1
    for message_id_chunk in message_id_chunks:
        actual_query = Message.objects.filter(id__in=message_id_chunk).order_by('id')
        message_chunk: List[Record] = make_raw(actual_query)
        message_filename: str = os.path.join(output_dir, f'messages-{dump_file_id:06}.json')
        message_filename += '.partial'
        logging.info('Fetched messages for %s', message_filename)
        table_data: Dict[str, Any] = {}
        table_data['zerver_message'] = message_chunk
        floatify_datetime_fields(table_data, 'zerver_message')
        output: Dict[str, Any] = dict(zerver_message=table_data['zerver_message'], zerver_userprofile_ids=list(user_profile_ids), realm_id=realm.id)
        write_data_to_file(message_filename, output)
        dump_file_id += 1

def export_uploads_and_avatars(realm: Realm, *, attachments: Optional[List[Attachment]] = None, user: Optional[UserProfile] = None, output_dir: str) -> None:
    uploads_output_dir: str = os.path.join(output_dir, 'uploads')
    avatars_output_dir: str = os.path.join(output_dir, 'avatars')
    realm_icons_output_dir: str = os.path.join(output_dir, 'realm_icons')
    emoji_output_dir: str = os.path.join(output_dir, 'emoji')
    for dir_path in (uploads_output_dir, avatars_output_dir, emoji_output_dir):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    if user is None and (not os.path.exists(realm_icons_output_dir)):
        os.makedirs(realm_icons_output_dir)
    if user is None:
        handle_system_bots: bool = True
        users: List[UserProfile] = list(UserProfile.objects.filter(realm=realm))
        assert attachments is not None
        realm_emojis: List[RealmEmoji] = list(RealmEmoji.objects.filter(realm_id=realm.id))
    else:
        handle_system_bots = False
        users = [user]
        attachments = list(Attachment.objects.filter(owner_id=user.id))
        realm_emojis = list(RealmEmoji.objects.filter(author_id=user.id))
    if settings.LOCAL_UPLOADS_DIR:
        assert settings.LOCAL_FILES_DIR
        assert settings.LOCAL_AVATARS_DIR
        export_uploads_from_local(realm, local_dir=settings.LOCAL_FILES_DIR, output_dir=uploads_output_dir, attachments=attachments)
        export_avatars_from_local(realm, local_dir=settings.LOCAL_AVATARS_DIR, output_dir=avatars_output_dir, users=users, handle_system_bots=handle_system_bots)
        export_emoji_from_local(realm, local_dir=settings.LOCAL_AVATARS_DIR, output_dir=emoji_output_dir, realm_emojis=realm_emojis)
        if user is None:
            export_realm_icons(realm, local_dir=settings.LOCAL_AVATARS_DIR, output_dir=realm_icons_output_dir)
    else:
        user_ids: Set[int] = {user.id for user in users}
        path_ids: Set[str] = {attachment.path_id for attachment in attachments}  # type: ignore
        export_files_from_s3(realm, handle_system_bots=handle_system_bots, flavor='upload', bucket_name=settings.S3_AUTH_UPLOADS_BUCKET, object_prefix=f'{realm.id}/', output_dir=uploads_output_dir, user_ids=user_ids, valid_hashes=path_ids)
        avatar_hash_values: Set[str] = set()
        for avatar_user in users:
            avatar_path: str = user_avatar_base_path_from_ids(avatar_user.id, avatar_user.avatar_version, realm.id)
            avatar_hash_values.add(avatar_path)
            avatar_hash_values.add(avatar_path + '.original')
        export_files_from_s3(realm, handle_system_bots=handle_system_bots, flavor='avatar', bucket_name=settings.S3_AVATAR_BUCKET, object_prefix=f'{realm.id}/', output_dir=avatars_output_dir, user_ids=user_ids, valid_hashes=avatar_hash_values)
        emoji_paths: Set[str] = set()
        for realm_emoji in realm_emojis:
            emoji_path: str = get_emoji_path(realm_emoji)
            emoji_paths.add(emoji_path)
            emoji_paths.add(emoji_path + '.original')
        export_files_from_s3(realm, handle_system_bots=handle_system_bots, flavor='emoji', bucket_name=settings.S3_AVATAR_BUCKET, object_prefix=f'{realm.id}/emoji/images/', output_dir=emoji_output_dir, user_ids=user_ids, valid_hashes=emoji_paths)
        if user is None:
            export_files_from_s3(realm, handle_system_bots=handle_system_bots, flavor='realm_icon_or_logo', bucket_name=settings.S3_AVATAR_BUCKET, object_prefix=f'{realm.id}/realm/', output_dir=realm_icons_output_dir, user_ids=user_ids, valid_hashes=None)

def _get_exported_s3_record(bucket_name: str, key: Any, processing_emoji: bool) -> Dict[str, Any]:
    record: Dict[str, Any] = dict(s3_path=key.key, bucket=bucket_name, size=key.content_length, last_modified=key.last_modified, content_type=key.content_type, md5=key.e_tag)
    record.update(key.metadata)
    if processing_emoji:
        file_name = os.path.basename(key.key)
        file_name = file_name.removesuffix('.original')
        record['file_name'] = file_name
    if 'user_profile_id' in record:
        user_profile = get_user_profile_by_id(int(record['user_profile_id']))
        record['user_profile_email'] = user_profile.email
        record['user_profile_id'] = int(record['user_profile_id'])
        if 'realm_id' not in record:
            record['realm_id'] = user_profile.realm_id
    else:
        pass
    if 'realm_id' in record:
        record['realm_id'] = int(record['realm_id'])
    else:
        raise Exception('Missing realm_id')
    if 'avatar_version' in record:
        record['avatar_version'] = int(record['avatar_version'])
    return record

def _save_s3_object_to_file(key: Any, output_dir: str, processing_uploads: bool) -> None:
    if not processing_uploads:
        filename: str = os.path.join(output_dir, key.key)
    else:
        fields = key.key.split('/')
        if len(fields) != 3:
            raise AssertionError(f'Suspicious key with invalid format {key.key}')
        filename = os.path.join(output_dir, key.key)
    if '../' in filename:
        raise AssertionError(f'Suspicious file with invalid format {filename}')
    dirname: str = mark_sanitized(os.path.dirname(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    key.download_file(Filename=filename)

def export_files_from_s3(realm: Realm, handle_system_bots: bool, flavor: str, bucket_name: str, object_prefix: str, output_dir: str, user_ids: Set[int], valid_hashes: Optional[Set[str]]) -> None:
    processing_uploads: bool = (flavor == 'upload')
    processing_emoji: bool = (flavor == 'emoji')
    bucket = get_bucket(bucket_name)
    records: List[Dict[str, Any]] = []
    logging.info('Downloading %s files from %s', flavor, bucket_name)
    email_gateway_bot: Optional[Any] = None
    if handle_system_bots and settings.EMAIL_GATEWAY_BOT is not None:
        internal_realm = get_realm(settings.SYSTEM_BOT_REALM)
        email_gateway_bot = get_system_bot(settings.EMAIL_GATEWAY_BOT, internal_realm.id)
        user_ids.add(email_gateway_bot.id)
    count: int = 0
    for bkey in bucket.objects.filter(Prefix=object_prefix):
        if valid_hashes is not None and bkey.Object().key not in valid_hashes:
            continue
        key = bucket.Object(bkey.key)
        '''
        For very old realms we may not have proper metadata. If you really need
        an export to bypass these checks, flip the following flag.
        '''
        checking_metadata: bool = True
        if checking_metadata:
            if 'realm_id' not in key.metadata:
                raise AssertionError(f'Missing realm_id in key metadata: {key.metadata}')
            if 'user_profile_id' not in key.metadata:
                raise AssertionError(f'Missing user_profile_id in key metadata: {key.metadata}')
            if int(key.metadata['user_profile_id']) not in user_ids:
                continue
            if key.metadata['realm_id'] != str(realm.id):
                if email_gateway_bot is None or key.metadata['user_profile_id'] != str(email_gateway_bot.id):
                    raise AssertionError(f'Key metadata problem: {key.key} / {key.metadata} / {realm.id}')
                print(f'File uploaded by email gateway bot: {key.key} / {key.metadata}')
        record: Dict[str, Any] = _get_exported_s3_record(bucket_name, key, processing_emoji)
        record['path'] = key.key
        _save_s3_object_to_file(key, output_dir, processing_uploads)
        records.append(record)
        count += 1
        if count % 100 == 0:
            logging.info('Finished %s', count)
    write_records_json_file(output_dir, records)

def export_uploads_from_local(realm: Realm, local_dir: str, output_dir: str, attachments: List[Attachment]) -> None:
    records: List[Dict[str, Any]] = []
    for count, attachment in enumerate(attachments, 1):
        path_id: str = mark_sanitized(attachment.path_id)
        local_path: str = os.path.join(local_dir, path_id)
        output_path: str = os.path.join(output_dir, path_id)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(local_path, output_path)
        stat = os.stat(local_path)
        record: Dict[str, Any] = dict(realm_id=attachment.realm_id, user_profile_id=attachment.owner.id, user_profile_email=attachment.owner.email, s3_path=path_id, path=path_id, size=stat.st_size, last_modified=stat.st_mtime, content_type=None)
        records.append(record)
        if count % 100 == 0:
            logging.info('Finished %s', count)
    write_records_json_file(output_dir, records)

def export_avatars_from_local(realm: Realm, local_dir: str, output_dir: str, users: List[UserProfile], handle_system_bots: bool) -> None:
    count: int = 0
    records: List[Dict[str, Any]] = []
    if handle_system_bots:
        internal_realm = get_realm(settings.SYSTEM_BOT_REALM)
        users += [get_system_bot(settings.NOTIFICATION_BOT, internal_realm.id), get_system_bot(settings.EMAIL_GATEWAY_BOT, internal_realm.id), get_system_bot(settings.WELCOME_BOT, internal_realm.id)]
    for user in users:
        if user.avatar_source == UserProfile.AVATAR_FROM_GRAVATAR:
            continue
        avatar_path: str = user_avatar_base_path_from_ids(user.id, user.avatar_version, realm.id)
        wildcard: str = os.path.join(local_dir, avatar_path + '.*')
        for local_path in glob.glob(wildcard):
            logging.info('Copying avatar file for user %s from %s', user.email, local_path)
            fn: str = os.path.relpath(local_path, local_dir)
            output_path: str = os.path.join(output_dir, fn)
            os.makedirs(str(os.path.dirname(output_path)), exist_ok=True)
            shutil.copy2(str(local_path), str(output_path))
            stat = os.stat(local_path)
            record: Dict[str, Any] = dict(realm_id=realm.id, user_profile_id=user.id, user_profile_email=user.email, avatar_version=user.avatar_version, s3_path=fn, path=fn, size=stat.st_size, last_modified=stat.st_mtime, content_type=None)
            records.append(record)
            count += 1
            if count % 100 == 0:
                logging.info('Finished %s', count)
    write_records_json_file(output_dir, records)

def export_realm_icons(realm: Realm, local_dir: str, output_dir: str) -> None:
    records: List[Dict[str, Any]] = []
    dir_relative_path: str = zerver.lib.upload.upload_backend.realm_avatar_and_logo_path(realm)
    icons_wildcard: str = os.path.join(local_dir, dir_relative_path, '*')
    for icon_absolute_path in glob.glob(icons_wildcard):
        icon_file_name: str = os.path.basename(icon_absolute_path)
        icon_relative_path: str = os.path.join(str(realm.id), icon_file_name)
        output_path: str = os.path.join(output_dir, icon_relative_path)
        os.makedirs(str(os.path.dirname(output_path)), exist_ok=True)
        shutil.copy2(str(icon_absolute_path), str(output_path))
        record: Dict[str, Any] = dict(realm_id=realm.id, path=icon_relative_path, s3_path=icon_relative_path)
        records.append(record)
    write_records_json_file(output_dir, records)

def get_emoji_path(realm_emoji: RealmEmoji) -> str:
    return RealmEmoji.PATH_ID_TEMPLATE.format(realm_id=realm_emoji.realm_id, emoji_file_name=realm_emoji.file_name)

def export_emoji_from_local(realm: Realm, local_dir: str, output_dir: str, realm_emojis: List[RealmEmoji]) -> None:
    records: List[Dict[str, Any]] = []
    realm_emoji_helper_tuples: List[tuple[RealmEmoji, str]] = []
    for realm_emoji in realm_emojis:
        realm_emoji_path: str = get_emoji_path(realm_emoji)
        realm_emoji_path = mark_sanitized(realm_emoji_path)
        realm_emoji_path_original: str = realm_emoji_path + '.original'
        realm_emoji_helper_tuples.append((realm_emoji, realm_emoji_path))
        realm_emoji_helper_tuples.append((realm_emoji, realm_emoji_path_original))
    for count, realm_emoji_helper_tuple in enumerate(realm_emoji_helper_tuples, 1):
        realm_emoji_object, emoji_path = realm_emoji_helper_tuple
        local_path: str = os.path.join(local_dir, emoji_path)
        output_path: str = os.path.join(output_dir, emoji_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(local_path, output_path)
        author = realm_emoji_object.author
        author_id: Optional[int] = None
        if author:
            author_id = author.id
        record: Dict[str, Any] = dict(realm_id=realm.id, author=author_id, path=emoji_path, s3_path=emoji_path, file_name=realm_emoji_object.file_name, name=realm_emoji_object.name, deactivated=realm_emoji_object.deactivated)
        records.append(record)
        if count % 100 == 0:
            logging.info('Finished %s', count)
    write_records_json_file(output_dir, records)

def do_write_stats_file_for_realm_export(output_dir: str) -> Dict[str, Any]:
    stats_file: str = os.path.join(output_dir, 'stats.json')
    realm_file: str = os.path.join(output_dir, 'realm.json')
    attachment_file: str = os.path.join(output_dir, 'attachment.json')
    analytics_file: str = os.path.join(output_dir, 'analytics.json')
    message_files: List[str] = glob.glob(os.path.join(output_dir, 'messages-*.json'))
    filenames: List[str] = sorted([analytics_file, attachment_file, *message_files, realm_file])
    logging.info('Writing stats file: %s\n', stats_file)
    stats: Dict[str, Any] = {}
    for filename in filenames:
        name = os.path.splitext(os.path.basename(filename))[0]
        with open(filename, 'rb') as json_file:
            data = orjson.loads(json_file.read())
        stats[name] = {k: len(data[k]) for k in sorted(data)}
    for category in ['avatars', 'uploads', 'emoji', 'realm_icons']:
        filename = os.path.join(output_dir, category, 'records.json')
        with open(filename, 'rb') as json_file:
            data = orjson.loads(json_file.read())
        stats[f'{category}_records'] = len(data)
    with open(stats_file, 'wb') as f:
        f.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    return stats

def get_exportable_scheduled_message_ids(realm: Realm, export_type: Any) -> Set[int]:
    """
    Scheduled messages are private to the sender, so which ones we export depends on the
    public/consent/full export mode.
    """
    if export_type == RealmExport.EXPORT_PUBLIC:
        return set()
    if export_type == RealmExport.EXPORT_FULL_WITH_CONSENT:
        sender_ids: Set[int] = get_consented_user_ids(realm)
        return set(ScheduledMessage.objects.filter(sender_id__in=sender_ids, realm=realm).values_list('id', flat=True))
    return set(ScheduledMessage.objects.filter(realm=realm).values_list('id', flat=True))

def do_export_realm(realm: Realm, output_dir: str, threads: int, export_type: Any, exportable_user_ids: Optional[Set[int]] = None, export_as_active: Optional[bool] = None) -> tuple[str, Dict[str, Any]]:
    response: TableData = {}
    if not settings.TEST_SUITE:
        assert threads >= 1
    realm_config: Config = get_realm_config()
    create_soft_link(source=output_dir, in_progress=True)
    exportable_scheduled_message_ids: Set[int] = get_exportable_scheduled_message_ids(realm, export_type)
    logging.info('Exporting data from get_realm_config()...')
    export_from_config(response=response, config=realm_config, seed_object=realm, context=dict(realm=realm, exportable_user_ids=exportable_user_ids, exportable_scheduled_message_ids=exportable_scheduled_message_ids))
    logging.info('...DONE with get_realm_config() data')
    sanity_check_output(response)
    logging.info('Exporting .partial files messages')
    message_ids: Set[int] = export_partial_message_files(realm, response, export_type=export_type, output_dir=output_dir)
    logging.info('%d messages were exported', len(message_ids))
    zerver_reaction: TableData = {}
    fetch_reaction_data(response=zerver_reaction, message_ids=message_ids)
    response.update(zerver_reaction)
    if export_as_active is not None:
        response['zerver_realm'][0]['deactivated'] = not export_as_active
    export_file: str = os.path.join(output_dir, 'realm.json')
    write_table_data(output_file=export_file, data=response)
    export_analytics_tables(realm=realm, output_dir=output_dir)
    attachments: List[Attachment] = export_attachment_table(realm=realm, output_dir=output_dir, message_ids=message_ids, scheduled_message_ids=exportable_scheduled_message_ids)
    logging.info('Exporting uploaded files and avatars')
    export_uploads_and_avatars(realm, attachments=attachments, user=None, output_dir=output_dir)
    launch_user_message_subprocesses(threads=threads, output_dir=output_dir, export_full_with_consent=export_type == RealmExport.EXPORT_FULL_WITH_CONSENT)
    do_common_export_processes(output_dir)
    logging.info('Finished exporting %s', realm.string_id)
    create_soft_link(source=output_dir, in_progress=False)
    stats: Dict[str, Any] = do_write_stats_file_for_realm_export(output_dir)
    logging.info('Compressing tarball...')
    tarball_path: str = output_dir.rstrip('/') + '.tar.gz'
    subprocess.check_call(['tar', f'-czf{tarball_path}', f'-C{os.path.dirname(output_dir)}', os.path.basename(output_dir)])
    return (tarball_path, stats)

def export_attachment_table(realm: Realm, output_dir: str, message_ids: Set[int], scheduled_message_ids: Set[int]) -> List[Attachment]:
    response: TableData = {}
    attachments: List[Attachment] = fetch_attachment_data(response=response, realm_id=realm.id, message_ids=message_ids, scheduled_message_ids=scheduled_message_ids)
    output_file: str = os.path.join(output_dir, 'attachment.json')
    write_table_data(output_file=output_file, data=response)
    return attachments

def create_soft_link(source: str, in_progress: bool = True) -> None:
    is_done: bool = not in_progress
    if settings.DEVELOPMENT:
        in_progress_link: str = os.path.join(settings.DEPLOY_ROOT, 'var', 'export-in-progress')
        done_link: str = os.path.join(settings.DEPLOY_ROOT, 'var', 'export-most-recent')
    else:
        in_progress_link = '/home/zulip/export-in-progress'
        done_link = '/home/zulip/export-most-recent'
    if in_progress:
        new_target: str = in_progress_link
    else:
        with suppress(FileNotFoundError):
            os.remove(in_progress_link)
        new_target = done_link
    overwrite_symlink(source, new_target)
    if is_done:
        logging.info('See %s for output files', new_target)

def launch_user_message_subprocesses(threads: int, output_dir: str, export_full_with_consent: bool) -> None:
    logging.info('Launching %d PARALLEL subprocesses to export UserMessage rows', threads)
    pids: Dict[int, int] = {}
    for shard_id in range(threads):
        arguments: List[str] = [os.path.join(settings.DEPLOY_ROOT, 'manage.py'), 'export_usermessage_batch', f'--path={output_dir}', f'--thread={shard_id}']
        if export_full_with_consent:
            arguments.append('--export-full-with-consent')
        process = subprocess.Popen(arguments)
        pids[process.pid] = shard_id
    while pids:
        pid, status = os.wait()
        shard: int = pids.pop(pid)
        print(f'Shard {shard} finished, status {status}')

def do_export_user(user_profile: UserProfile, output_dir: str) -> None:
    response: TableData = {}
    export_single_user(user_profile, response)
    export_file: str = os.path.join(output_dir, 'user.json')
    write_table_data(output_file=export_file, data=response)
    reaction_message_ids: Set[int] = {row['message'] for row in response['zerver_reaction']}
    logging.info('Exporting messages')
    export_messages_single_user(user_profile, output_dir=output_dir, reaction_message_ids=reaction_message_ids)
    logging.info('Exporting images')
    export_uploads_and_avatars(user_profile.realm, user=user_profile, output_dir=output_dir)

def export_single_user(user_profile: UserProfile, response: TableData) -> None:
    config: Config = get_single_user_config()
    export_from_config(response=response, config=config, seed_object=user_profile, context=dict(user=user_profile))

def get_single_user_config() -> Config:
    user_profile_config: Config = Config(table='zerver_userprofile', is_seeded=True, exclude=EXCLUDED_USER_PROFILE_FIELDS)
    subscription_config: Config = Config(table='zerver_subscription', model=Subscription, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    recipient_config: Config = Config(table='zerver_recipient', model=Recipient, virtual_parent=subscription_config, id_source=('_user_subscription', 'recipient'))
    Config(table='zerver_stream', model=Stream, virtual_parent=recipient_config, id_source=('zerver_recipient', 'type_id'), source_filter=lambda r: r['type'] == Recipient.STREAM)
    Config(table='analytics_usercount', model=UserCount, normal_parent=user_profile_config, include_rows='user_id__in')
    Config(table='zerver_realmauditlog', model=RealmAuditLog, virtual_parent=user_profile_config, custom_fetch=custom_fetch_realm_audit_logs_for_user)
    Config(table='zerver_reaction', model=Reaction, normal_parent=user_profile_config, include_rows='user_profile_id__in')
    add_user_profile_child_configs(user_profile_config)
    return user_profile_config

def get_id_list_gently_from_database(*, base_query: Any, id_field: str) -> List[int]:
    """
    Use this function if you need a HUGE number of ids from
    the database, and you don't mind a few extra trips.  Particularly
    for exports, we don't really care about a little extra time
    to finish the export--the much bigger concern is that we don't
    want to overload our database all at once, nor do we want to
    keep a whole bunch of Django objects around in memory.
    """
    min_id: int = -1
    all_ids: List[int] = []
    batch_size: int = 10000
    assert id_field == 'id' or id_field.endswith('_id')
    while True:
        filter_args = {f'{id_field}__gt': min_id}
        new_ids: List[int] = list(base_query.values_list(id_field, flat=True).filter(**filter_args).order_by(id_field)[:batch_size])
        if len(new_ids) == 0:
            break
        all_ids += new_ids
        min_id = new_ids[-1]
    return all_ids

def chunkify(lst: List[int], chunk_size: int) -> List[List[int]]:
    result: List[List[int]] = []
    i: int = 0
    while True:
        chunk: List[int] = lst[i:i + chunk_size]
        if len(chunk) == 0:
            break
        else:
            result.append(chunk)
            i += chunk_size
    return result

def export_messages_single_user(user_profile: UserProfile, *, output_dir: str, reaction_message_ids: Set[int]) -> None:
    @cache
    def get_recipient(recipient_id: int) -> str:
        recipient = Recipient.objects.get(id=recipient_id)
        if recipient.type == Recipient.STREAM:
            stream = Stream.objects.values('name').get(id=recipient.type_id)
            return stream['name']
        user_names = UserProfile.objects.filter(subscription__recipient_id=recipient.id).order_by('full_name').values_list('full_name', flat=True)
        return ', '.join(user_names)
    messages_from_me = Message.objects.filter(realm_id=user_profile.realm_id, sender=user_profile)
    my_subscriptions = Subscription.objects.filter(user_profile=user_profile, recipient__type__in=[Recipient.PERSONAL, Recipient.DIRECT_MESSAGE_GROUP])
    my_recipient_ids = [sub.recipient_id for sub in my_subscriptions]
    all_message_ids: Set[int] = set()
    for query in [messages_from_me, Message.objects.filter(realm_id=user_profile.realm_id, recipient_id__in=my_recipient_ids)]:
        all_message_ids |= set(get_id_list_gently_from_database(base_query=query, id_field='id'))
    all_message_ids |= reaction_message_ids
    dump_file_id: int = 1
    for message_id_chunk in chunkify(sorted(all_message_ids), MESSAGE_BATCH_CHUNK_SIZE):
        fat_query = UserMessage.objects.select_related('message', 'message__sending_client').filter(user_profile=user_profile, message_id__in=message_id_chunk).order_by('message_id')
        user_message_chunk: List[Any] = list(fat_query)
        message_chunk: List[Dict[str, Any]] = []
        for user_message in user_message_chunk:
            item: Dict[str, Any] = model_to_dict(user_message.message)
            item['flags'] = user_message.flags_list()
            item['flags_mask'] = user_message.flags.mask
            item['sending_client_name'] = user_message.message.sending_client.name
            item['recipient_name'] = get_recipient(user_message.message.recipient_id)
            message_chunk.append(item)
        message_filename: str = os.path.join(output_dir, f'messages-{dump_file_id:06}.json')
        logging.info('Fetched messages for %s', message_filename)
        output: Dict[str, Any] = {'zerver_message': message_chunk}
        floatify_datetime_fields(output, 'zerver_message')
        write_table_data(message_filename, output)
        dump_file_id += 1

def export_analytics_tables(realm: Realm, output_dir: str) -> None:
    response: TableData = {}
    logging.info('Fetching analytics table data')
    config: Config = get_analytics_config()
    export_from_config(response=response, config=config, seed_object=realm)
    del response['zerver_realm']
    export_file: str = os.path.join(output_dir, 'analytics.json')
    write_table_data(output_file=export_file, data=response)

def get_analytics_config() -> Config:
    analytics_config: Config = Config(table='zerver_realm', is_seeded=True)
    Config(table='analytics_realmcount', model=RealmCount, normal_parent=analytics_config, include_rows='realm_id__in')
    Config(table='analytics_usercount', model=UserCount, normal_parent=analytics_config, include_rows='realm_id__in')
    Config(table='analytics_streamcount', model=StreamCount, normal_parent=analytics_config, include_rows='realm_id__in')
    return analytics_config

def get_consented_user_ids(realm: Realm) -> Set[int]:
    return set(UserProfile.objects.filter(realm=realm, is_active=True, is_bot=False, allow_private_data_export=True).values_list('id', flat=True))

def export_realm_wrapper(export_row: Any, output_dir: str, threads: int, upload: bool, percent_callback: Optional[Callable[[int], None]] = None, export_as_active: Optional[bool] = None) -> Optional[str]:
    try:
        export_row.status = RealmExport.STARTED
        export_row.date_started = timezone_now()
        export_row.save(update_fields=['status', 'date_started'])
        tarball_path, stats = do_export_realm(realm=export_row.realm, output_dir=output_dir, threads=threads, export_type=export_row.type, export_as_active=export_as_active)
        RealmAuditLog.objects.create(acting_user=export_row.acting_user, realm=export_row.realm, event_type=AuditLogEventType.REALM_EXPORTED, event_time=timezone_now(), extra_data={'realm_export_id': export_row.id})
        shutil.rmtree(output_dir)
        print(f'Tarball written to {tarball_path}')
        print('Calculating SHA-256 checksum of tarball...')
        sha256_hash = hashlib.sha256()
        with open(tarball_path, 'rb') as f:
            buf = bytearray(2 ** 18)
            view = memoryview(buf)
            while True:
                size = f.readinto(buf)
                if size == 0:
                    break
                sha256_hash.update(view[:size])
        export_row.sha256sum_hex = sha256_hash.hexdigest()
        export_row.tarball_size_bytes = os.path.getsize(tarball_path)
        export_row.status = RealmExport.SUCCEEDED
        export_row.date_succeeded = timezone_now()
        export_row.stats = stats
        print(f'SHA-256 checksum is {export_row.sha256sum_hex}')
        if not upload:
            export_row.save(update_fields=['sha256sum_hex', 'tarball_size_bytes', 'status', 'date_succeeded', 'stats'])
            return None
        print('Uploading export tarball...')
        public_url = zerver.lib.upload.upload_backend.upload_export_tarball(export_row.realm, tarball_path, percent_callback=percent_callback)
        print(f'\nUploaded to {public_url}')
        export_row.export_path = urlsplit(public_url).path
        export_row.save(update_fields=['sha256sum_hex', 'tarball_size_bytes', 'status', 'date_succeeded', 'stats', 'export_path'])
        os.remove(tarball_path)
        print(f'Successfully deleted the tarball at {tarball_path}')
        return public_url
    except Exception:
        export_row.status = RealmExport.FAILED
        export_row.date_failed = timezone_now()
        export_row.save(update_fields=['status', 'date_failed'])
        raise

def get_realm_exports_serialized(realm: Realm) -> List[Dict[str, Any]]:
    all_exports = RealmExport.objects.filter(realm=realm).exclude(acting_user=None)
    exports_dict: Dict[int, Dict[str, Any]] = {}
    for export in all_exports:
        export_url: Optional[str] = None
        export_path = export.export_path
        pending: bool = export.status in [RealmExport.REQUESTED, RealmExport.STARTED]
        if export.status == RealmExport.SUCCEEDED:
            assert export_path is not None
            export_url = zerver.lib.upload.upload_backend.get_export_tarball_url(realm, export_path)
        deleted_timestamp = datetime_to_timestamp(export.date_deleted) if export.date_deleted else None
        failed_timestamp = datetime_to_timestamp(export.date_failed) if export.date_failed else None
        acting_user = export.acting_user
        assert acting_user is not None
        exports_dict[export.id] = dict(id=export.id, export_time=datetime_to_timestamp(export.date_requested), acting_user_id=acting_user.id, export_url=export_url, deleted_timestamp=deleted_timestamp, failed_timestamp=failed_timestamp, pending=pending, export_type=export.type)
    return sorted(exports_dict.values(), key=lambda export_dict: export_dict['id'])

def export_migration_status(output_dir: str) -> None:
    export_showmigration = get_migration_status(close_connection_when_done=False)
    migration_status_json = MigrationStatusJson(migrations_by_app=parse_migration_status(export_showmigration), zulip_version=ZULIP_VERSION)
    output_file: str = os.path.join(output_dir, 'migration_status.json')
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(migration_status_json, option=orjson.OPT_INDENT_2))

def do_common_export_processes(output_dir: str) -> None:
    logging.info('Exporting migration status')
    export_migration_status(output_dir)