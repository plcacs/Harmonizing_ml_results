#!/usr/bin/env python3
import itertools
import logging
import os
import posixpath
import random
import re
import secrets
import shutil
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from email.headerregistry import Address
from typing import Any, Dict, Iterator as IteratorType, List, Tuple, Optional, Set, TypeVar
from urllib.parse import SplitResult, urlsplit

import orjson
import requests
from django.conf import settings
from django.forms.models import model_to_dict
from django.utils.timezone import now as timezone_now
from zerver.data_import.import_util import (
    ZerverFieldsT, build_attachment, build_avatar, build_defaultstream, build_direct_message_group,
    build_message, build_realm, build_recipient, build_stream, build_subscription, build_usermessages,
    build_zerver_realm, create_converted_data_files, long_term_idle_helper, make_subscriber_map,
    process_avatars, process_emojis, process_uploads, validate_user_emails_for_import)
from zerver.data_import.sequencer import NEXT_ID
from zerver.data_import.slack_message_conversion import convert_to_zulip_markdown, get_user_full_name
from zerver.lib.emoji import codepoint_to_name, get_emoji_file_name
from zerver.lib.export import MESSAGE_BATCH_CHUNK_SIZE, do_common_export_processes
from zerver.lib.mime_types import guess_type
from zerver.lib.storage import static_path
from zerver.lib.thumbnail import THUMBNAIL_ACCEPT_IMAGE_TYPES, resize_realm_icon
from zerver.lib.upload import sanitize_name
from zerver.models import CustomProfileField, CustomProfileFieldValue, Reaction, Realm, RealmEmoji, Recipient, UserProfile

SlackBotEmailT = TypeVar('SlackBotEmailT', bound='SlackBotEmail')
SlackToZulipUserIDT = Dict[str, int]
AddedChannelsT = Dict[str, Tuple[str, int]]
AddedMPIMsT = Dict[str, Tuple[str, int]]
DMMembersT = Dict[str, Tuple[str, str]]
SlackToZulipRecipientT = Dict[str, int]

emoji_data_file_path: str = static_path('generated/emoji/emoji-datasource-google-emoji.json')
with open(emoji_data_file_path, 'rb') as emoji_data_file:
    emoji_data: List[Dict[str, Any]] = orjson.loads(emoji_data_file.read())

def get_emoji_code(emoji_dict: Dict[str, Any]) -> str:
    emoji_code: str = emoji_dict.get('non_qualified') or emoji_dict['unified']
    return emoji_code.lower()

slack_emoji_name_to_codepoint: Dict[str, str] = {}
for emoji_dict in emoji_data:
    short_name: str = emoji_dict['short_name']
    emoji_code: str = get_emoji_code(emoji_dict)
    slack_emoji_name_to_codepoint[short_name] = emoji_code
    for sn in emoji_dict['short_names']:
        if sn != short_name:
            slack_emoji_name_to_codepoint[sn] = emoji_code

class SlackBotEmail:
    duplicate_email_count: Dict[str, int] = {}
    assigned_email: Dict[str, str] = {}

    @classmethod
    def get_email(cls, user_profile: Dict[str, Any], domain_name: str) -> str:
        slack_bot_id: str = user_profile['bot_id']
        if slack_bot_id in cls.assigned_email:
            return cls.assigned_email[slack_bot_id]
        if 'real_name_normalized' in user_profile:
            slack_bot_name: str = user_profile['real_name_normalized']
        elif 'first_name' in user_profile:
            slack_bot_name = user_profile['first_name']
        else:
            raise AssertionError('Could not identify bot type')
        email = Address(username=slack_bot_name.replace('Bot', '').replace(' ', '').lower() + '-bot',
                        domain=domain_name).addr_spec
        if email in cls.duplicate_email_count:
            cls.duplicate_email_count[email] += 1
            address = Address(addr_spec=email)
            email_username = address.username + '-' + str(cls.duplicate_email_count[email])
            email = Address(username=email_username, domain=address.domain).addr_spec
        else:
            cls.duplicate_email_count[email] = 1
        cls.assigned_email[slack_bot_id] = email
        return email

def rm_tree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)

def slack_workspace_to_realm(domain_name: str, realm_id: int, user_list: List[Dict[str, Any]], realm_subdomain: str,
                             slack_data_dir: str, custom_emoji_list: Dict[str, str]
                             ) -> Tuple[Dict[str, Any],
                                        SlackToZulipUserIDT,
                                        SlackToZulipRecipientT,
                                        AddedChannelsT,
                                        AddedMPIMsT,
                                        DMMembersT,
                                        List[Any],
                                        Dict[str, str]]:
    """
    Returns:
    1. realm, converted realm data
    2. slack_user_id_to_zulip_user_id, which is a dictionary mapping Slack user id to Zulip user id
    3. slack_recipient_name_to_zulip_recipient_id, mapping Slack recipient name to Zulip recipient id
    4. added_channels, mapping channel name to (channel id, Zulip stream_id)
    5. added_mpims, mapping MPIM name to (mpim id, Zulip direct_message_group_id)
    6. dm_members, mapping DM id to tuple of DM participants.
    7. avatars, list of avatars for Zulip avatar records.json
    8. emoji_url_map, mapping emoji name to its Slack URL
    """
    NOW: float = float(timezone_now().timestamp())
    zerver_realm: List[Any] = build_zerver_realm(realm_id, realm_subdomain, NOW, 'Slack')
    realm: Dict[str, Any] = build_realm(zerver_realm, realm_id, domain_name)
    (zerver_userprofile, avatars, slack_user_id_to_zulip_user_id,
     zerver_customprofilefield, zerver_customprofilefield_value) = users_to_zerver_userprofile(slack_data_dir, user_list, realm_id, int(NOW), domain_name)
    (realm, added_channels, added_mpims, dm_members,
     slack_recipient_name_to_zulip_recipient_id) = channels_to_zerver_stream(slack_data_dir, realm_id, realm, slack_user_id_to_zulip_user_id, zerver_userprofile)
    zerver_realmemoji, emoji_url_map = build_realmemoji(custom_emoji_list, realm_id)
    realm['zerver_realmemoji'] = zerver_realmemoji
    realm['zerver_userprofile'] = zerver_userprofile
    realm['zerver_customprofilefield'] = zerver_customprofilefield
    realm['zerver_customprofilefieldvalue'] = zerver_customprofilefield_value
    return (realm, slack_user_id_to_zulip_user_id, slack_recipient_name_to_zulip_recipient_id,
            added_channels, added_mpims, dm_members, avatars, emoji_url_map)

def build_realmemoji(custom_emoji_list: Dict[str, str], realm_id: int) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    zerver_realmemoji: List[Dict[str, Any]] = []
    emoji_url_map: Dict[str, str] = {}
    emoji_id: int = 0
    for emoji_name, url in custom_emoji_list.items():
        split_url = urlsplit(url)
        if split_url.hostname == 'emoji.slack-edge.com':
            content_type: Optional[str] = guess_type(posixpath.basename(split_url.path))[0]
            assert content_type is not None
            realmemoji = RealmEmoji(name=emoji_name, id=emoji_id, file_name=get_emoji_file_name(content_type, emoji_id), deactivated=False)
            realmemoji_dict: Dict[str, Any] = model_to_dict(realmemoji, exclude=['realm', 'author'])
            realmemoji_dict['author'] = None
            realmemoji_dict['realm'] = realm_id
            emoji_url_map[emoji_name] = url
            zerver_realmemoji.append(realmemoji_dict)
            emoji_id += 1
    return (zerver_realmemoji, emoji_url_map)

def users_to_zerver_userprofile(slack_data_dir: str, users: List[Dict[str, Any]], realm_id: int, timestamp: int, domain_name: str
                                ) -> Tuple[List[Dict[str, Any]], List[Any], SlackToZulipUserIDT, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
    1. zerver_userprofile list of user profiles
    2. avatar_list for Zulip avatar records.json
    3. slack_user_id_to_zulip_user_id mapping from Slack user ID to Zulip user id
    4. zerver_customprofilefield list of custom profile fields
    5. zerver_customprofilefield_values list of user profile fields
    """
    logging.info('######### IMPORTING USERS STARTED #########\n')
    zerver_userprofile: List[Dict[str, Any]] = []
    zerver_customprofilefield: List[Dict[str, Any]] = []
    zerver_customprofilefield_values: List[Dict[str, Any]] = []
    avatar_list: List[Any] = []
    slack_user_id_to_zulip_user_id: SlackToZulipUserIDT = {}
    slack_data_file_user_list: List[Any] = get_data_file(slack_data_dir + '/users.json')
    slack_user_id_to_custom_profile_fields: Dict[str, Dict[str, Any]] = {}
    slack_custom_field_name_to_zulip_custom_field_id: Dict[str, int] = {}
    for user in slack_data_file_user_list:
        process_slack_custom_fields(user, slack_user_id_to_custom_profile_fields)
    user_id_count: int = 0
    custom_profile_field_value_id_count: int = 0
    custom_profile_field_id_count: int = 0
    primary_owner_id: int = user_id_count
    user_id_count += 1
    found_emails: Dict[str, int] = {}
    for user in users:
        slack_user_id: str = user['id']
        if user.get('is_primary_owner', False):
            user_id: int = primary_owner_id
        else:
            user_id = user_id_count
        email: str = get_user_email(user, domain_name)
        if email.lower() in found_emails:
            slack_user_id_to_zulip_user_id[slack_user_id] = found_emails[email.lower()]
            logging.info('%s: %s MERGED', slack_user_id, email)
            continue
        found_emails[email.lower()] = user_id
        avatar_source, avatar_url = build_avatar_url(slack_user_id, user)
        if avatar_source == UserProfile.AVATAR_FROM_USER:
            build_avatar(user_id, realm_id, email, avatar_url, timestamp, avatar_list)
        role: int = UserProfile.ROLE_MEMBER
        if get_owner(user):
            role = UserProfile.ROLE_REALM_OWNER
        elif get_admin(user):
            role = UserProfile.ROLE_REALM_ADMINISTRATOR
        if get_guest(user):
            role = UserProfile.ROLE_GUEST
        tz: str = get_user_timezone(user)
        if slack_user_id in slack_user_id_to_custom_profile_fields:
            slack_custom_field_name_to_zulip_custom_field_id, custom_profile_field_id_count = build_customprofile_field(
                zerver_customprofilefield, slack_user_id_to_custom_profile_fields[slack_user_id],
                custom_profile_field_id_count, realm_id, slack_custom_field_name_to_zulip_custom_field_id)
            custom_profile_field_value_id_count = build_customprofilefields_values(
                slack_custom_field_name_to_zulip_custom_field_id, slack_user_id_to_custom_profile_fields[slack_user_id],
                user_id, custom_profile_field_value_id_count, zerver_customprofilefield_values)
        userprofile = UserProfile(full_name=get_user_full_name(user),
                                  is_active=not user.get('deleted', False) and (not user['is_mirror_dummy']),
                                  is_mirror_dummy=user['is_mirror_dummy'], id=user_id,
                                  email=email, delivery_email=email, avatar_source=avatar_source,
                                  is_bot=user.get('is_bot', False),
                                  role=role, bot_type=1 if user.get('is_bot', False) else None,
                                  date_joined=timestamp, timezone=tz, last_login=timestamp)
        userprofile_dict: Dict[str, Any] = model_to_dict(userprofile)
        userprofile_dict['realm'] = realm_id
        zerver_userprofile.append(userprofile_dict)
        slack_user_id_to_zulip_user_id[slack_user_id] = user_id
        if not user.get('is_primary_owner', False):
            user_id_count += 1
        logging.info('%s: %s -> %s', slack_user_id, user['name'], userprofile_dict['email'])
    validate_user_emails_for_import(list(found_emails))
    process_customprofilefields(zerver_customprofilefield, zerver_customprofilefield_values)
    logging.info('######### IMPORTING USERS FINISHED #########\n')
    return (zerver_userprofile, avatar_list, slack_user_id_to_zulip_user_id, zerver_customprofilefield, zerver_customprofilefield_values)

def build_customprofile_field(customprofile_field: List[Dict[str, Any]], fields: Dict[str, Any],
                              custom_profile_field_id: int, realm_id: int,
                              slack_custom_field_name_to_zulip_custom_field_id: Dict[str, int]
                              ) -> Tuple[Dict[str, int], int]:
    for field in fields:
        if field not in slack_custom_field_name_to_zulip_custom_field_id:
            slack_custom_fields: List[str] = ['phone', 'skype']
            if field in slack_custom_fields:
                field_name: str = field
            else:
                field_name = f'Slack custom field {custom_profile_field_id + 1}'
            customprofilefield = CustomProfileField(id=custom_profile_field_id, name=field_name, field_type=1)
            customprofilefield_dict: Dict[str, Any] = model_to_dict(customprofilefield, exclude=['realm'])
            customprofilefield_dict['realm'] = realm_id
            slack_custom_field_name_to_zulip_custom_field_id[field] = custom_profile_field_id
            custom_profile_field_id += 1
            customprofile_field.append(customprofilefield_dict)
    return (slack_custom_field_name_to_zulip_custom_field_id, custom_profile_field_id)

def process_slack_custom_fields(user: Dict[str, Any], slack_user_id_to_custom_profile_fields: Dict[str, Dict[str, Any]]) -> None:
    slack_user_id_to_custom_profile_fields[user['id']] = {}
    if user['profile'].get('fields'):
        slack_user_id_to_custom_profile_fields[user['id']] = user['profile']['fields']
    slack_custom_fields: List[str] = ['phone', 'skype']
    for field in slack_custom_fields:
        if field in user['profile']:
            slack_user_id_to_custom_profile_fields[user['id']][field] = {'value': user['profile'][field]}

def build_customprofilefields_values(slack_custom_field_name_to_zulip_custom_field_id: Dict[str, int],
                                     fields: Dict[str, Any], user_id: int, custom_field_id: int,
                                     custom_field_values: List[Dict[str, Any]]) -> int:
    for field, value in fields.items():
        if value['value'] == '':
            continue
        custom_field_value = CustomProfileFieldValue(id=custom_field_id, value=value['value'])
        custom_field_value_dict: Dict[str, Any] = model_to_dict(custom_field_value, exclude=['user_profile', 'field'])
        custom_field_value_dict['user_profile'] = user_id
        custom_field_value_dict['field'] = slack_custom_field_name_to_zulip_custom_field_id[field]
        custom_field_values.append(custom_field_value_dict)
        custom_field_id += 1
    return custom_field_id

def process_customprofilefields(customprofilefield: List[Dict[str, Any]], customprofilefield_value: List[Dict[str, Any]]) -> None:
    for field in customprofilefield:
        for field_value in customprofilefield_value:
            if field_value['field'] == field['id'] and len(field_value['value']) > 50:
                field['field_type'] = 2
                break

def get_user_email(user: Dict[str, Any], domain_name: str) -> str:
    if 'email' in user['profile']:
        return user['profile']['email']
    if user['is_mirror_dummy']:
        return Address(username=user['name'], domain=f"{user['team_domain']}.slack.com").addr_spec
    if 'bot_id' in user['profile']:
        return SlackBotEmail.get_email(user['profile'], domain_name)
    if get_user_full_name(user).lower() == 'slackbot':
        return Address(username='imported-slackbot-bot', domain=domain_name).addr_spec
    raise AssertionError(f'Could not find email address for Slack user {user}')

def build_avatar_url(slack_user_id: str, user: Dict[str, Any]) -> Tuple[int, str]:
    avatar_url: str = ''
    avatar_source: int = UserProfile.AVATAR_FROM_GRAVATAR
    if user['profile'].get('avatar_hash'):
        team_id: str = user['team_id']
        avatar_hash: str = user['profile']['avatar_hash']
        avatar_url = f'https://ca.slack-edge.com/{team_id}-{slack_user_id}-{avatar_hash}'
        avatar_source = UserProfile.AVATAR_FROM_USER
    elif user.get('is_integration_bot'):
        avatar_url = user['profile']['image_72']
        content_type: Optional[str] = guess_type(avatar_url)[0]
        if content_type not in THUMBNAIL_ACCEPT_IMAGE_TYPES:
            logging.info('Unsupported avatar type (%s) for user -> %s\n', content_type, user.get('name'))
            avatar_source = UserProfile.AVATAR_FROM_GRAVATAR
        else:
            avatar_source = UserProfile.AVATAR_FROM_USER
    else:
        logging.info('Failed to process avatar for user -> %s\n', user.get('name'))
    return (avatar_source, avatar_url)

def get_owner(user: Dict[str, Any]) -> bool:
    owner: bool = user.get('is_owner', False)
    primary_owner: bool = user.get('is_primary_owner', False)
    return primary_owner or owner

def get_admin(user: Dict[str, Any]) -> bool:
    return user.get('is_admin', False)

def get_guest(user: Dict[str, Any]) -> bool:
    restricted_user: bool = user.get('is_restricted', False)
    ultra_restricted_user: bool = user.get('is_ultra_restricted', False)
    return restricted_user or ultra_restricted_user

def get_user_timezone(user: Dict[str, Any]) -> str:
    _default_timezone: str = 'America/New_York'
    tz: Optional[str] = user.get('tz', _default_timezone)
    if tz is None or '/' not in tz:
        tz = _default_timezone
    return tz

def channels_to_zerver_stream(slack_data_dir: str, realm_id: int, realm: Dict[str, Any],
                              slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
                              zerver_userprofile: List[Dict[str, Any]]
                              ) -> Tuple[Dict[str, Any],
                                         AddedChannelsT,
                                         AddedMPIMsT,
                                         DMMembersT,
                                         SlackToZulipRecipientT]:
    """
    Returns:
    1. realm, converted realm data
    2. added_channels mapping from channel name to (channel id, Zulip stream_id)
    3. added_mpims mapping from MPIM name to (mpim id, Zulip direct_message_group_id)
    4. dm_members mapping from DM id to tuple of DM participants.
    5. slack_recipient_name_to_zulip_recipient_id mapping from Slack recipient name to Zulip recipient_id
    """
    logging.info('######### IMPORTING CHANNELS STARTED #########\n')
    added_channels: AddedChannelsT = {}
    added_mpims: AddedMPIMsT = {}
    dm_members: DMMembersT = {}
    slack_recipient_name_to_zulip_recipient_id: SlackToZulipRecipientT = {}
    realm['zerver_stream'] = []
    realm['zerver_huddle'] = []
    realm['zerver_subscription'] = []
    realm['zerver_recipient'] = []
    realm['zerver_defaultstream'] = []
    subscription_id_count: int = 0
    recipient_id_count: int = 0
    stream_id_count: int = 0
    defaultstream_id: int = 0
    direct_message_group_id_count: int = 0

    def process_channels(channels: List[Dict[str, Any]], invite_only: bool = False) -> None:
        nonlocal stream_id_count, recipient_id_count, defaultstream_id, subscription_id_count
        for channel in channels:
            description: str = channel['purpose']['value']
            stream_id: int = stream_id_count
            recipient_id: int = recipient_id_count
            stream: Dict[str, Any] = build_stream(float(channel['created']), realm_id, channel['name'], description, stream_id, channel['is_archived'], invite_only)
            realm['zerver_stream'].append(stream)
            slack_default_channels: List[str] = ['general', 'random']
            if channel['name'] in slack_default_channels and (not stream['deactivated']):
                defaultstream = build_defaultstream(realm_id, stream_id, defaultstream_id)
                realm['zerver_defaultstream'].append(defaultstream)
                defaultstream_id += 1
            added_channels[stream['name']] = (channel['id'], stream_id)
            recipient = build_recipient(stream_id, recipient_id, Recipient.STREAM)
            realm['zerver_recipient'].append(recipient)
            slack_recipient_name_to_zulip_recipient_id[stream['name']] = recipient_id
            subscription_id_count = get_subscription(channel['members'], realm['zerver_subscription'], recipient_id, slack_user_id_to_zulip_user_id, subscription_id_count)
            stream_id_count += 1
            recipient_id_count += 1
            logging.info('%s -> created', channel['name'])
    public_channels: List[Any] = get_data_file(slack_data_dir + '/channels.json')
    process_channels(public_channels)
    try:
        private_channels: List[Any] = get_data_file(slack_data_dir + '/groups.json')
    except FileNotFoundError:
        private_channels = []
    process_channels(private_channels, True)

    def process_mpims(mpims: List[Dict[str, Any]]) -> None:
        nonlocal direct_message_group_id_count, recipient_id_count, subscription_id_count
        for mpim in mpims:
            direct_message_group = build_direct_message_group(direct_message_group_id_count, len(mpim['members']))
            realm['zerver_huddle'].append(direct_message_group)
            added_mpims[mpim['name']] = (mpim['id'], direct_message_group_id_count)
            recipient = build_recipient(direct_message_group_id_count, recipient_id_count, Recipient.DIRECT_MESSAGE_GROUP)
            realm['zerver_recipient'].append(recipient)
            slack_recipient_name_to_zulip_recipient_id[mpim['name']] = recipient_id_count
            subscription_id_count = get_subscription(mpim['members'], realm['zerver_subscription'], recipient_id_count, slack_user_id_to_zulip_user_id, subscription_id_count)
            direct_message_group_id_count += 1
            recipient_id_count += 1
            logging.info('%s -> created', mpim['name'])
    try:
        mpims: List[Any] = get_data_file(slack_data_dir + '/mpims.json')
    except FileNotFoundError:
        mpims = []
    process_mpims(mpims)
    zulip_user_to_recipient: Dict[int, int] = {}
    for slack_user_id, zulip_user_id in slack_user_id_to_zulip_user_id.items():
        if zulip_user_id in zulip_user_to_recipient:
            slack_recipient_name_to_zulip_recipient_id[slack_user_id] = zulip_user_to_recipient[zulip_user_id]
            continue
        recipient = build_recipient(zulip_user_id, recipient_id_count, Recipient.PERSONAL)
        slack_recipient_name_to_zulip_recipient_id[slack_user_id] = recipient_id_count
        zulip_user_to_recipient[zulip_user_id] = recipient_id_count
        sub = build_subscription(recipient_id_count, zulip_user_id, subscription_id_count)
        realm['zerver_recipient'].append(recipient)
        realm['zerver_subscription'].append(sub)
        recipient_id_count += 1
        subscription_id_count += 1

    def process_dms(dms: List[Dict[str, Any]]) -> None:
        for dm in dms:
            user_a: str = dm['members'][0]
            user_b: str = dm['members'][1]
            dm_members[dm['id']] = (user_a, user_b)
    try:
        dms: List[Any] = get_data_file(slack_data_dir + '/dms.json')
    except FileNotFoundError:
        dms = []
    process_dms(dms)
    logging.info('######### IMPORTING STREAMS FINISHED #########\n')
    return (realm, added_channels, added_mpims, dm_members, slack_recipient_name_to_zulip_recipient_id)

def get_subscription(channel_members: List[str], zerver_subscription: List[Any], recipient_id: int,
                     slack_user_id_to_zulip_user_id: SlackToZulipUserIDT, subscription_id: int) -> int:
    for slack_user_id in channel_members:
        sub = build_subscription(recipient_id, slack_user_id_to_zulip_user_id[slack_user_id], subscription_id)
        zerver_subscription.append(sub)
        subscription_id += 1
    return subscription_id

def process_long_term_idle_users(slack_data_dir: str, users: List[Dict[str, Any]],
                                 slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
                                 added_channels: AddedChannelsT, added_mpims: AddedMPIMsT,
                                 dm_members: DMMembersT, zerver_userprofile: List[Dict[str, Any]]) -> Any:
    return long_term_idle_helper(get_messages_iterator(slack_data_dir, added_channels, added_mpims, dm_members),
                                 get_message_sending_user, get_timestamp_from_message,
                                 lambda id: slack_user_id_to_zulip_user_id[id],
                                 iter((user['id'] for user in users)), zerver_userprofile)

def convert_slack_workspace_messages(slack_data_dir: str, users: List[Dict[str, Any]], realm_id: int,
                                     slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
                                     slack_recipient_name_to_zulip_recipient_id: SlackToZulipRecipientT,
                                     added_channels: AddedChannelsT, added_mpims: AddedMPIMsT,
                                     dm_members: DMMembersT, realm: Dict[str, Any],
                                     zerver_userprofile: List[Dict[str, Any]],
                                     zerver_realmemoji: List[Dict[str, Any]], domain_name: str,
                                     output_dir: str, convert_slack_threads: bool,
                                     chunk_size: int = MESSAGE_BATCH_CHUNK_SIZE
                                     ) -> Tuple[List[Dict[str, Any]], List[Any], List[Any]]:
    """
    Returns:
    1. reactions: list of reactions
    2. uploads: list of uploads for uploads records.json
    3. attachment: list of attachments
    """
    long_term_idle = process_long_term_idle_users(slack_data_dir, users, slack_user_id_to_zulip_user_id,
                                                  added_channels, added_mpims, dm_members, zerver_userprofile)
    all_messages: IteratorType[Dict[str, Any]] = get_messages_iterator(slack_data_dir, added_channels, added_mpims, dm_members)
    logging.info('######### IMPORTING MESSAGES STARTED #########\n')
    total_reactions: List[Dict[str, Any]] = []
    total_attachments: List[Dict[str, Any]] = []
    total_uploads: List[Any] = []
    dump_file_id: int = 1
    subscriber_map: Dict[Any, Any] = make_subscriber_map(zerver_subscription=realm['zerver_subscription'])
    while (message_data := list(itertools.islice(all_messages, chunk_size))):
        (zerver_message, zerver_usermessage, attachment, uploads, reactions) = channel_message_to_zerver_message(
            realm_id, users, slack_user_id_to_zulip_user_id, slack_recipient_name_to_zulip_recipient_id,
            message_data, zerver_realmemoji, subscriber_map, added_channels, dm_members, domain_name,
            long_term_idle, convert_slack_threads)
        message_json: Dict[str, Any] = dict(zerver_message=zerver_message, zerver_usermessage=zerver_usermessage)
        message_file: str = f'/messages-{dump_file_id:06}.json'
        logging.info('Writing messages to %s\n', output_dir + message_file)
        create_converted_data_files(message_json, output_dir, message_file)
        total_reactions += reactions
        total_attachments += attachment
        total_uploads += uploads
        dump_file_id += 1
    logging.info('######### IMPORTING MESSAGES FINISHED #########\n')
    return (total_reactions, total_uploads, total_attachments)

def get_messages_iterator(slack_data_dir: str, added_channels: AddedChannelsT, added_mpims: AddedMPIMsT,
                          dm_members: DMMembersT) -> IteratorType[Dict[str, Any]]:
    """Iterator returning all messages across Slack channels, ordered by timestamp."""
    dir_names: List[str] = [*added_channels, *added_mpims, *dm_members]
    all_json_names: Dict[str, List[str]] = defaultdict(list)
    for dir_name in dir_names:
        dir_path: str = os.path.join(slack_data_dir, dir_name)
        json_names: List[str] = os.listdir(dir_path)
        for json_name in json_names:
            if json_name.endswith('.json'):
                all_json_names[json_name].append(dir_path)
    for json_name in sorted(all_json_names.keys()):
        messages_for_one_day: List[Dict[str, Any]] = []
        for dir_path in all_json_names[json_name]:
            message_dir: str = os.path.join(dir_path, json_name)
            dir_basename: str = os.path.basename(dir_path)
            messages: List[Dict[str, Any]] = []
            for message in get_data_file(message_dir):
                if message.get('user') == 'U00':
                    continue
                if message.get('mimetype') == 'application/vnd.slack-docs':
                    continue
                if dir_basename in added_channels:
                    message['channel_name'] = dir_basename
                elif dir_basename in added_mpims:
                    message['mpim_name'] = dir_basename
                elif dir_basename in dm_members:
                    message['pm_name'] = dir_basename
                messages.append(message)
            messages_for_one_day += messages
        yield from sorted(messages_for_one_day, key=get_timestamp_from_message)

def channel_message_to_zerver_message(realm_id: int, users: List[Dict[str, Any]],
                                      slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
                                      slack_recipient_name_to_zulip_recipient_id: SlackToZulipRecipientT,
                                      all_messages: List[Dict[str, Any]], zerver_realmemoji: List[Dict[str, Any]],
                                      subscriber_map: Dict[Any, Any], added_channels: AddedChannelsT,
                                      dm_members: DMMembersT, domain_name: str, long_term_idle: Any,
                                      convert_slack_threads: bool
                                      ) -> Tuple[List[Dict[str, Any]],
                                                 List[Dict[str, Any]],
                                                 List[Dict[str, Any]],
                                                 List[Any],
                                                 List[Dict[str, Any]]]:
    zerver_message: List[Dict[str, Any]] = []
    zerver_usermessage: List[Dict[str, Any]] = []
    uploads_list: List[Any] = []
    zerver_attachment: List[Dict[str, Any]] = []
    reaction_list: List[Dict[str, Any]] = []
    total_user_messages: int = 0
    total_skipped_user_messages: int = 0
    thread_counter: Dict[str, int] = defaultdict(int)
    thread_map: Dict[str, str] = {}
    sender_recipient_id: Optional[int] = None
    for message in all_messages:
        slack_user_id: Optional[str] = get_message_sending_user(message)
        if not slack_user_id:
            continue
        subtype: Optional[str] = message.get('subtype', False)
        if subtype in ['pinned_item', 'unpinned_item', 'channel_join', 'channel_leave', 'channel_name']:
            continue
        try:
            content, mentioned_user_ids, has_link = convert_to_zulip_markdown(message['text'], users, added_channels, slack_user_id_to_zulip_user_id)
        except Exception:
            print('Slack message unexpectedly missing text representation:')
            print(orjson.dumps(message, option=orjson.OPT_INDENT_2).decode())
            continue
        rendered_content: Optional[str] = None
        if 'channel_name' in message:
            is_private: bool = False
            recipient_id: int = slack_recipient_name_to_zulip_recipient_id[message['channel_name']]
        elif 'mpim_name' in message:
            is_private = True
            recipient_id = slack_recipient_name_to_zulip_recipient_id[message['mpim_name']]
        elif 'pm_name' in message:
            is_private = True
            sender = get_message_sending_user(message)
            members = dm_members[message['pm_name']]
            if sender == members[0]:
                recipient_id = slack_recipient_name_to_zulip_recipient_id[members[1]]
                sender_recipient_id = slack_recipient_name_to_zulip_recipient_id[members[0]]
            else:
                recipient_id = slack_recipient_name_to_zulip_recipient_id[members[0]]
                sender_recipient_id = slack_recipient_name_to_zulip_recipient_id[members[1]]
        message_id: int = NEXT_ID('message')
        if 'reactions' in message:
            build_reactions(reaction_list, message['reactions'], slack_user_id_to_zulip_user_id, message_id, zerver_realmemoji)
        if subtype in ['bot_add', 'sh_room_created', 'me_message']:
            content = f'/me {content}'
        if subtype == 'file_comment':
            message['user'] = message['comment']['user']
        file_info: Dict[str, Any] = process_message_files(message=message, domain_name=domain_name, realm_id=realm_id,
                                                          message_id=message_id, slack_user_id=slack_user_id,
                                                          users=users, slack_user_id_to_zulip_user_id=slack_user_id_to_zulip_user_id,
                                                          zerver_attachment=zerver_attachment, uploads_list=uploads_list)
        content = '\n'.join([part for part in [content, file_info['content']] if part != ''])
        has_link = has_link or file_info['has_link']
        has_attachment: bool = file_info['has_attachment']
        has_image: bool = file_info['has_image']
        topic_name: str = 'imported from Slack'
        if convert_slack_threads and 'thread_ts' in message:
            thread_ts = datetime.fromtimestamp(float(message['thread_ts']), tz=timezone.utc)
            thread_ts_str: str = thread_ts.strftime('%Y/%m/%d %H:%M:%S')
            if thread_ts_str in thread_map:
                topic_name = thread_map[thread_ts_str]
            else:
                thread_date = thread_ts.strftime('%Y-%m-%d')
                thread_counter[thread_date] += 1
                count = thread_counter[thread_date]
                topic_name = f'{thread_date} Slack thread {count}'
                thread_map[thread_ts_str] = topic_name
        zulip_message: Dict[str, Any] = build_message(topic_name=topic_name,
                                                      date_sent=get_timestamp_from_message(message),
                                                      message_id=message_id,
                                                      content=content,
                                                      rendered_content=rendered_content,
                                                      user_id=slack_user_id_to_zulip_user_id[slack_user_id],
                                                      recipient_id=recipient_id,
                                                      realm_id=realm_id,
                                                      has_image=has_image,
                                                      has_link=has_link,
                                                      has_attachment=has_attachment)
        zerver_message.append(zulip_message)
        num_created, num_skipped = build_usermessages(zerver_usermessage=zerver_usermessage,
                                                       subscriber_map=subscriber_map,
                                                       recipient_id=recipient_id,
                                                       mentioned_user_ids=mentioned_user_ids,
                                                       message_id=message_id,
                                                       is_private=is_private,
                                                       long_term_idle=long_term_idle)
        total_user_messages += num_created
        total_skipped_user_messages += num_skipped
        if 'pm_name' in message and sender_recipient_id is not None and recipient_id != sender_recipient_id:
            num_created, num_skipped = build_usermessages(zerver_usermessage=zerver_usermessage,
                                                           subscriber_map=subscriber_map,
                                                           recipient_id=sender_recipient_id,
                                                           mentioned_user_ids=mentioned_user_ids,
                                                           message_id=message_id,
                                                           is_private=is_private,
                                                           long_term_idle=long_term_idle)
            total_user_messages += num_created
            total_skipped_user_messages += num_skipped
    logging.debug('Created %s UserMessages; deferred %s due to long-term idle', total_user_messages, total_skipped_user_messages)
    return (zerver_message, zerver_usermessage, zerver_attachment, uploads_list, reaction_list)

def process_message_files(message: Dict[str, Any], domain_name: str, realm_id: int, message_id: int,
                          slack_user_id: str, users: List[Dict[str, Any]],
                          slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
                          zerver_attachment: List[Dict[str, Any]], uploads_list: List[Any]
                          ) -> Dict[str, Any]:
    has_attachment: bool = False
    has_image: bool = False
    has_link: bool = False
    files: List[Dict[str, Any]] = message.get('files', [])
    subtype: Optional[str] = message.get('subtype')
    if subtype == 'file_share':
        files = [message['file']]
    markdown_links: List[str] = []
    for fileinfo in files:
        if fileinfo.get('mode', '') in ['tombstone', 'hidden_by_limit']:
            continue
        if fileinfo.get('file_access', '') in ['access_denied', 'file_not_found']:
            continue
        url: str = fileinfo['url_private']
        split_url = urlsplit(url)
        if split_url.hostname == 'files.slack.com':
            has_attachment = True
            has_link = True
            has_image = 'image' in fileinfo['mimetype']
            file_user: List[Dict[str, Any]] = [iterate_user for iterate_user in users if message['user'] == iterate_user['id']]
            file_user_email: str = get_user_email(file_user[0], domain_name)
            s3_path, content_for_link = get_attachment_path_and_content(fileinfo, realm_id)
            markdown_links.append(content_for_link)
            build_uploads(slack_user_id_to_zulip_user_id[slack_user_id], realm_id, file_user_email, fileinfo, s3_path, uploads_list)
            build_attachment(realm_id, {message_id}, slack_user_id_to_zulip_user_id[slack_user_id], fileinfo, s3_path, zerver_attachment)
        else:
            has_link = True
            if 'title' in fileinfo:
                file_name: str = fileinfo['title']
            else:
                file_name = fileinfo['name']
            markdown_links.append('[{}]({})'.format(file_name, fileinfo['url_private']))
    content: str = '\n'.join(markdown_links)
    return dict(content=content, has_attachment=has_attachment, has_image=has_image, has_link=has_link)

def get_attachment_path_and_content(fileinfo: Dict[str, Any], realm_id: int) -> Tuple[str, str]:
    s3_path: str = '/'.join([str(realm_id), format(random.randint(0, 255), 'x'), secrets.token_urlsafe(18), sanitize_name(fileinfo['name'])])
    attachment_path: str = f'/user_uploads/{s3_path}'
    content: str = '[{}]({})'.format(fileinfo['title'], attachment_path)
    return (s3_path, content)

def build_reactions(reaction_list: List[Dict[str, Any]], reactions: List[Dict[str, Any]],
                    slack_user_id_to_zulip_user_id: SlackToZulipUserIDT, message_id: int,
                    zerver_realmemoji: List[Dict[str, Any]]) -> None:
    realmemoji: Dict[str, int] = {}
    for realm_emoji in zerver_realmemoji:
        realmemoji[realm_emoji['name']] = realm_emoji['id']
    merged_reactions: Dict[str, Set[str]] = defaultdict(set)
    for slack_reaction in reactions:
        emoji_name: str = slack_reaction['name'].split('::', maxsplit=1)[0]
        merged_reactions[emoji_name].update(slack_reaction['users'])
    reactions_list: List[Dict[str, Any]] = [{'name': k, 'users': v, 'count': len(v)} for k, v in merged_reactions.items()]
    for slack_reaction in reactions_list:
        emoji_name: str = slack_reaction['name']
        if emoji_name in slack_emoji_name_to_codepoint:
            emoji_code: str = slack_emoji_name_to_codepoint[emoji_name]
            try:
                zulip_emoji_name: str = codepoint_to_name[emoji_code]
            except KeyError:
                print(f'WARN: Emoji found in iamcal but not Zulip: {emoji_name}')
                continue
            emoji_name = zulip_emoji_name
            reaction_type = Reaction.UNICODE_EMOJI
        elif emoji_name in realmemoji:
            emoji_code = realmemoji[emoji_name]
            reaction_type = Reaction.REALM_EMOJI
        else:
            print(f'WARN: Emoji not found in iamcal: {emoji_name}')
            continue
        for slack_user_id in slack_reaction['users']:
            if slack_user_id not in slack_user_id_to_zulip_user_id:
                continue
            reaction_id: int = NEXT_ID('reaction')
            reaction = Reaction(id=reaction_id, emoji_code=emoji_code, emoji_name=emoji_name, reaction_type=reaction_type)
            reaction_dict: Dict[str, Any] = model_to_dict(reaction, exclude=['message', 'user_profile'])
            reaction_dict['message'] = message_id
            reaction_dict['user_profile'] = slack_user_id_to_zulip_user_id[slack_user_id]
            reaction_list.append(reaction_dict)

def build_uploads(user_id: int, realm_id: int, email: str, fileinfo: Dict[str, Any],
                  s3_path: str, uploads_list: List[Any]) -> None:
    upload: Dict[str, Any] = dict(path=fileinfo['url_private'], realm_id=realm_id, content_type=None,
                                  user_profile_id=user_id, last_modified=fileinfo['timestamp'],
                                  user_profile_email=email, s3_path=s3_path, size=fileinfo['size'])
    uploads_list.append(upload)

def get_message_sending_user(message: Dict[str, Any]) -> Optional[str]:
    if 'user' in message:
        return message['user']
    if message.get('file'):
        return message['file'].get('user')
    return None

def get_timestamp_from_message(message: Dict[str, Any]) -> float:
    return float(message['ts'])

def is_integration_bot_message(message: Dict[str, Any]) -> bool:
    return message.get('subtype') == 'bot_message' and 'user' not in message and ('bot_id' in message)

def convert_bot_info_to_slack_user(bot_info: Dict[str, Any]) -> Dict[str, Any]:
    bot_user: Dict[str, Any] = {'id': bot_info['id'], 'name': bot_info['name'], 'deleted': bot_info['deleted'],
                                'is_mirror_dummy': False, 'real_name': bot_info['name'], 'is_integration_bot': True,
                                'profile': {'image_72': bot_info['icons']['image_72'], 'bot_id': bot_info['id'],
                                            'first_name': bot_info['name']}}
    return bot_user

def fetch_shared_channel_users(user_list: List[Dict[str, Any]], slack_data_dir: str, token: str) -> None:
    normal_user_ids: Set[str] = set()
    mirror_dummy_user_ids: Set[str] = set()
    added_channels: Dict[str, bool] = {}
    integration_bot_users: List[str] = []
    team_id_to_domain: Dict[str, str] = {}
    for user in user_list:
        user['is_mirror_dummy'] = False
        normal_user_ids.add(user['id'])
    public_channels: List[Any] = get_data_file(slack_data_dir + '/channels.json')
    try:
        private_channels: List[Any] = get_data_file(slack_data_dir + '/groups.json')
    except FileNotFoundError:
        private_channels = []
    try:
        direct_message_groups: List[Any] = get_data_file(slack_data_dir + '/mpims.json')
    except FileNotFoundError:
        direct_message_groups = []
    for channel in public_channels + private_channels + direct_message_groups:
        added_channels[channel['name']] = True
        for user_id in channel['members']:
            if user_id not in normal_user_ids:
                mirror_dummy_user_ids.add(user_id)
    if os.path.exists(slack_data_dir + '/dms.json'):
        dms: List[Any] = get_data_file(slack_data_dir + '/dms.json')
        for dm_data in dms:
            for user_id in dm_data['members']:
                if user_id not in normal_user_ids:
                    mirror_dummy_user_ids.add(user_id)
    all_messages: IteratorType[Dict[str, Any]] = get_messages_iterator(slack_data_dir, added_channels, {}, {})
    for message in all_messages:
        if is_integration_bot_message(message):
            bot_id: str = message['bot_id']
            if bot_id in integration_bot_users:
                continue
            bot_info: Dict[str, Any] = get_slack_api_data('https://slack.com/api/bots.info', 'bot', token=token, bot=bot_id)
            bot_user: Dict[str, Any] = convert_bot_info_to_slack_user(bot_info)
            user_list.append(bot_user)
            integration_bot_users.append(bot_id)
        else:
            user_id: Optional[str] = get_message_sending_user(message)
            if user_id is None or user_id in normal_user_ids:
                continue
            mirror_dummy_user_ids.add(user_id)
    for user_id in mirror_dummy_user_ids:
        user: Dict[str, Any] = get_slack_api_data('https://slack.com/api/users.info', 'user', token=token, user=user_id)
        team_id: str = user['team_id']
        if team_id not in team_id_to_domain:
            team: Dict[str, Any] = get_slack_api_data('https://slack.com/api/team.info', 'team', token=token, team=team_id)
            team_id_to_domain[team_id] = team['domain']
        user['team_domain'] = team_id_to_domain[team_id]
        user['is_mirror_dummy'] = True
        user_list.append(user)

def fetch_team_icons(zerver_realm: List[Any], team_info_dict: Dict[str, Any], output_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    team_icons_dict: Dict[str, Any] = team_info_dict['icon']
    if team_icons_dict.get('image_default', False):
        return []
    icon_url: Optional[str] = team_icons_dict.get('image_original', None) or team_icons_dict.get('image_230', None) or team_icons_dict.get('image_132', None) or team_icons_dict.get('image_102', None)
    if icon_url is None:
        return []
    response = requests.get(icon_url, stream=True)
    response_raw = response.raw
    realm_id: int = zerver_realm['id']
    os.makedirs(os.path.join(output_dir, str(realm_id)), exist_ok=True)
    original_icon_output_path: str = os.path.join(output_dir, str(realm_id), 'icon.original')
    with open(original_icon_output_path, 'wb') as output_file:
        shutil.copyfileobj(response_raw, output_file)
    records.append({'realm_id': realm_id, 'path': os.path.join(str(realm_id), 'icon.original'),
                    's3_path': os.path.join(str(realm_id), 'icon.original'), 'content_type': response.headers['Content-Type']})
    resized_icon_output_path: str = os.path.join(output_dir, str(realm_id), 'icon.png')
    with open(resized_icon_output_path, 'wb') as output_file, open(original_icon_output_path, 'rb') as original_file:
        resized_data = resize_realm_icon(original_file.read())
        output_file.write(resized_data)
    records.append({'realm_id': realm_id, 'path': os.path.join(str(realm_id), 'icon.png'),
                    's3_path': os.path.join(str(realm_id), 'icon.png'), 'content_type': 'image/png'})
    zerver_realm['icon_source'] = Realm.ICON_UPLOADED
    return records

def do_convert_zipfile(original_path: str, output_dir: str, token: str, threads: int = 6, convert_slack_threads: bool = False) -> None:
    assert original_path.endswith('.zip')
    slack_data_dir: str = original_path.removesuffix('.zip')
    try:
        os.makedirs(slack_data_dir, exist_ok=True)
        with zipfile.ZipFile(original_path) as zipObj:
            total_size: int = 0
            for fileinfo in zipObj.infolist():
                fileinfo.flag_bits |= 2048
                fileinfo.filename = fileinfo.filename.encode('cp437').decode('utf-8')
                zipObj.NameToInfo[fileinfo.filename] = fileinfo
                if not re.match('[^/]+(\\.json|/([^/]+\\.json)?)$', fileinfo.filename):
                    raise Exception('This zip file does not look like a Slack archive')
                total_size += fileinfo.file_size
            if total_size > 1024 * 1024 * 1024 and total_size > 10 * os.path.getsize(original_path):
                raise Exception('This zip file is possibly malicious')
            zipObj.extractall(slack_data_dir)
        do_convert_directory(slack_data_dir, output_dir, token, threads, convert_slack_threads)
    finally:
        rm_tree(slack_data_dir)

SLACK_IMPORT_TOKEN_SCOPES: Set[str] = {'emoji:read', 'users:read', 'users:read.email', 'team:read'}

def do_convert_directory(slack_data_dir: str, output_dir: str, token: str, threads: int = 6, convert_slack_threads: bool = False) -> None:
    check_token_access(token, SLACK_IMPORT_TOKEN_SCOPES)
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        raise Exception('Output directory should be empty!')
    if not os.path.isfile(os.path.join(slack_data_dir, 'channels.json')):
        raise ValueError('Import does not have the layout we expect from a Slack export!')
    user_list: List[Dict[str, Any]] = get_slack_api_data('https://slack.com/api/users.list', 'members', token=token, pagination_limit=200)
    fetch_shared_channel_users(user_list, slack_data_dir, token)
    custom_emoji_list: Dict[str, str] = get_slack_api_data('https://slack.com/api/emoji.list', 'emoji', token=token)
    realm_subdomain: str = ''
    realm_id: int = 0
    domain_name: Optional[str] = SplitResult('', settings.EXTERNAL_HOST, '', '', '').hostname
    assert isinstance(domain_name, str)
    (realm, slack_user_id_to_zulip_user_id, slack_recipient_name_to_zulip_recipient_id, added_channels,
     added_mpims, dm_members, avatar_list, emoji_url_map) = slack_workspace_to_realm(domain_name, realm_id, user_list, realm_subdomain, slack_data_dir, custom_emoji_list)
    reactions, uploads_list, zerver_attachment = convert_slack_workspace_messages(slack_data_dir, user_list, realm_id,
                                                                                 slack_user_id_to_zulip_user_id,
                                                                                 slack_recipient_name_to_zulip_recipient_id,
                                                                                 added_channels, added_mpims, dm_members,
                                                                                 realm, realm['zerver_userprofile'],
                                                                                 realm['zerver_realmemoji'], domain_name,
                                                                                 output_dir, convert_slack_threads)
    realm['zerver_reaction'] = reactions
    emoji_folder: str = os.path.join(output_dir, 'emoji')
    os.makedirs(emoji_folder, exist_ok=True)
    emoji_records = process_emojis(realm['zerver_realmemoji'], emoji_folder, emoji_url_map, threads)
    avatar_folder: str = os.path.join(output_dir, 'avatars')
    avatar_realm_folder: str = os.path.join(avatar_folder, str(realm_id))
    os.makedirs(avatar_realm_folder, exist_ok=True)
    avatar_records = process_avatars(avatar_list, avatar_folder, realm_id, threads, size_url_suffix='-512')
    uploads_folder: str = os.path.join(output_dir, 'uploads')
    os.makedirs(os.path.join(uploads_folder, str(realm_id)), exist_ok=True)
    uploads_records = process_uploads(uploads_list, uploads_folder, threads)
    attachment = {'zerver_attachment': zerver_attachment}
    team_info_dict: Dict[str, Any] = get_slack_api_data('https://slack.com/api/team.info', 'team', token=token)
    realm_icons_folder: str = os.path.join(output_dir, 'realm_icons')
    realm_icon_records = fetch_team_icons(realm['zerver_realm'][0], team_info_dict, realm_icons_folder)
    create_converted_data_files(realm, output_dir, '/realm.json')
    create_converted_data_files(emoji_records, output_dir, '/emoji/records.json')
    create_converted_data_files(avatar_records, output_dir, '/avatars/records.json')
    create_converted_data_files(uploads_records, output_dir, '/uploads/records.json')
    create_converted_data_files(attachment, output_dir, '/attachment.json')
    create_converted_data_files(realm_icon_records, output_dir, '/realm_icons/records.json')
    do_common_export_processes(output_dir)
    logging.info('######### DATA CONVERSION FINISHED #########\n')
    logging.info('Zulip data dump created at %s', output_dir)

def get_data_file(path: str) -> Any:
    with open(path, 'rb') as fp:
        data = orjson.loads(fp.read())
        return data

def check_token_access(token: str, required_scopes: Set[str]) -> None:
    if token.startswith('xoxp-'):
        logging.info('This is a Slack user token, which grants all rights the user has!')
    elif token.startswith('xoxb-'):
        data = requests.get('https://slack.com/api/api.test', headers={'Authorization': f'Bearer {token}'})
        if data.status_code != 200:
            raise ValueError(f'Failed to fetch data (HTTP status {data.status_code}) for Slack token: {token}')
        if not data.json()['ok']:
            error = data.json()['error']
            if error != 'missing_scope':
                raise ValueError(f'Invalid Slack token: {token}, {error}')
        has_scopes: Set[str] = set(data.headers.get('x-oauth-scopes', '').split(','))
        missing_scopes: Set[str] = required_scopes - has_scopes
        if missing_scopes:
            raise ValueError(f'Slack token is missing the following required scopes: {sorted(missing_scopes)}')
    else:
        raise Exception('Unknown token type -- must start with xoxb- or xoxp-')

def get_slack_api_data(slack_api_url: str, get_param: str, *, pagination_limit: Optional[int] = None,
                       raise_if_rate_limited: bool = False, **kwargs: Any) -> Any:
    if not kwargs.get('token'):
        raise AssertionError('Slack token missing in kwargs')
    token: str = kwargs.pop('token')
    accumulated_result: List[Any] = []
    cursor: Optional[str] = None
    while True:
        if pagination_limit is not None:
            logging.info('Fetching page from %s with cursor: %s and limit: %s', slack_api_url, cursor, pagination_limit)
        params: Dict[str, Any] = {'limit': pagination_limit} if pagination_limit else {}
        if cursor:
            params['cursor'] = cursor
        params.update(kwargs)
        response = requests.get(slack_api_url, headers={'Authorization': f'Bearer {token}'}, params=params)
        if response.status_code == 429:
            if raise_if_rate_limited:
                raise Exception('Exceeded Slack rate limits.')
            retry_after: int = int(response.headers.get('retry-after', 1))
            logging.info('Rate limit exceeded. Retrying in %s seconds...', retry_after)
            time.sleep(retry_after)
            continue
        if response.status_code != requests.codes.ok:
            logging.info('HTTP error: %s, Response: %s', response.status_code, response.text)
            raise Exception('HTTP error accessing the Slack API.')
        result: Dict[str, Any] = response.json()
        if not result['ok']:
            raise Exception('Error accessing Slack API: {}'.format(result['error']))
        result_data: Any = result[get_param]
        if pagination_limit is None:
            return result_data
        accumulated_result.extend(result_data)
        if not result.get('response_metadata', {}).get('next_cursor'):
            break
        cursor = result['response_metadata']['next_cursor']
    return accumulated_result
