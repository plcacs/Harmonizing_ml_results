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
from typing import Any, TypeAlias, TypeVar, Dict, List, Tuple, Set, Optional, Union, DefaultDict, cast
from urllib.parse import SplitResult, urlsplit
import orjson
import requests
from django.conf import settings
from django.forms.models import model_to_dict
from django.utils.timezone import now as timezone_now
from zerver.data_import.import_util import ZerverFieldsT, build_attachment, build_avatar, build_defaultstream, build_direct_message_group, build_message, build_realm, build_recipient, build_stream, build_subscription, build_usermessages, build_zerver_realm, create_converted_data_files, long_term_idle_helper, make_subscriber_map, process_avatars, process_emojis, process_uploads, validate_user_emails_for_import
from zerver.data_import.sequencer import NEXT_ID
from zerver.data_import.slack_message_conversion import convert_to_zulip_markdown, get_user_full_name
from zerver.lib.emoji import codepoint_to_name, get_emoji_file_name
from zerver.lib.export import MESSAGE_BATCH_CHUNK_SIZE, do_common_export_processes
from zerver.lib.mime_types import guess_type
from zerver.lib.storage import static_path
from zerver.lib.thumbnail import THUMBNAIL_ACCEPT_IMAGE_TYPES, resize_realm_icon
from zerver.lib.upload import sanitize_name
from zerver.models import CustomProfileField, CustomProfileFieldValue, Reaction, Realm, RealmEmoji, Recipient, UserProfile

SlackToZulipUserIDT = Dict[str, int]
AddedChannelsT = Dict[str, Tuple[str, int]]
AddedMPIMsT = Dict[str, Tuple[str, int]]
DMMembersT = Dict[str, Tuple[str, str]]
SlackToZulipRecipientT = Dict[str, int]
SlackBotEmailT = TypeVar('SlackBotEmailT', bound='SlackBotEmail')

emoji_data_file_path: str = static_path('generated/emoji/emoji-datasource-google-emoji.json')
with open(emoji_data_file_path, 'rb') as emoji_data_file:
    emoji_data: List[Dict[str, Any]] = orjson.loads(emoji_data_file.read())

def get_emoji_code(emoji_dict: Dict[str, Any]) -> str:
    emoji_code = emoji_dict.get('non_qualified') or emoji_dict['unified']
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
        email: str = Address(username=slack_bot_name.replace('Bot', '').replace(' ', '').lower() + '-bot', domain=domain_name).addr_spec
        if email in cls.duplicate_email_count:
            cls.duplicate_email_count[email] += 1
            address: Address = Address(addr_spec=email)
            email_username: str = address.username + '-' + str(cls.duplicate_email_count[email])
            email = Address(username=email_username, domain=address.domain).addr_spec
        else:
            cls.duplicate_email_count[email] = 1
        cls.assigned_email[slack_bot_id] = email
        return email

def rm_tree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)

def slack_workspace_to_realm(domain_name: str, realm_id: int, user_list: List[Dict[str, Any]], realm_subdomain: str, slack_data_dir: str, custom_emoji_list: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, int], Dict[str, Tuple[str, int]], Dict[str, Tuple[str, int]], Dict[str, Tuple[str, str]], List[Dict[str, Any]], Dict[str, str]]:
    NOW: float = float(timezone_now().timestamp())
    zerver_realm: List[Dict[str, Any]] = build_zerver_realm(realm_id, realm_subdomain, NOW, 'Slack')
    realm: Dict[str, Any] = build_realm(zerver_realm, realm_id, domain_name)
    zerver_userprofile: List[Dict[str, Any]]
    avatars: List[Dict[str, Any]]
    slack_user_id_to_zulip_user_id: Dict[str, int]
    zerver_customprofilefield: List[Dict[str, Any]]
    zerver_customprofilefield_value: List[Dict[str, Any]]
    zerver_userprofile, avatars, slack_user_id_to_zulip_user_id, zerver_customprofilefield, zerver_customprofilefield_value = users_to_zerver_userprofile(slack_data_dir, user_list, realm_id, int(NOW), domain_name)
    added_channels: Dict[str, Tuple[str, int]]
    added_mpims: Dict[str, Tuple[str, int]]
    dm_members: Dict[str, Tuple[str, str]]
    slack_recipient_name_to_zulip_recipient_id: Dict[str, int]
    realm, added_channels, added_mpims, dm_members, slack_recipient_name_to_zulip_recipient_id = channels_to_zerver_stream(slack_data_dir, realm_id, realm, slack_user_id_to_zulip_user_id, zerver_userprofile)
    zerver_realmemoji: List[Dict[str, Any]]
    emoji_url_map: Dict[str, str]
    zerver_realmemoji, emoji_url_map = build_realmemoji(custom_emoji_list, realm_id)
    realm['zerver_realmemoji'] = zerver_realmemoji
    realm['zerver_userprofile'] = zerver_userprofile
    realm['zerver_customprofilefield'] = zerver_customprofilefield
    realm['zerver_customprofilefieldvalue'] = zerver_customprofilefield_value
    return (realm, slack_user_id_to_zulip_user_id, slack_recipient_name_to_zulip_recipient_id, added_channels, added_mpims, dm_members, avatars, emoji_url_map)

[Rest of the code continues with similar type annotations...]
