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
from typing import Any, TypeAlias, TypeVar, Optional, Union, Dict, List, Tuple, Set, DefaultDict
from urllib.parse import SplitResult, urlsplit

import orjson
import requests
from django.conf import settings
from django.forms.models import model_to_dict
from django.utils.timezone import now as timezone_now

from zerver.data_import.import_util import (
    ZerverFieldsT,
    build_attachment,
    build_avatar,
    build_defaultstream,
    build_direct_message_group,
    build_message,
    build_realm,
    build_recipient,
    build_stream,
    build_subscription,
    build_usermessages,
    build_zerver_realm,
    create_converted_data_files,
    long_term_idle_helper,
    make_subscriber_map,
    process_avatars,
    process_emojis,
    process_uploads,
    validate_user_emails_for_import,
)
from zerver.data_import.sequencer import NEXT_ID
from zerver.data_import.slack_message_conversion import (
    convert_to_zulip_markdown,
    get_user_full_name,
)
from zerver.lib.emoji import codepoint_to_name, get_emoji_file_name
from zerver.lib.export import MESSAGE_BATCH_CHUNK_SIZE, do_common_export_processes
from zerver.lib.mime_types import guess_type
from zerver.lib.storage import static_path
from zerver.lib.thumbnail import THUMBNAIL_ACCEPT_IMAGE_TYPES, resize_realm_icon
from zerver.lib.upload import sanitize_name
from zerver.models import (
    CustomProfileField,
    CustomProfileFieldValue,
    Reaction,
    Realm,
    RealmEmoji,
    Recipient,
    UserProfile,
)

SlackToZulipUserIDT: TypeAlias = Dict[str, int]
AddedChannelsT: TypeAlias = Dict[str, Tuple[str, int]]
AddedMPIMsT: TypeAlias = Dict[str, Tuple[str, int]]
DMMembersT: TypeAlias = Dict[str, Tuple[str, str]]
SlackToZulipRecipientT: TypeAlias = Dict[str, int]
SlackBotEmailT = TypeVar("SlackBotEmailT", bound="SlackBotEmail")

emoji_data_file_path = static_path("generated/emoji/emoji-datasource-google-emoji.json")
with open(emoji_data_file_path, "rb") as emoji_data_file:
    emoji_data = orjson.loads(emoji_data_file.read())

def get_emoji_code(emoji_dict: Dict[str, Any]) -> str:
    emoji_code = emoji_dict.get("non_qualified") or emoji_dict["unified"]
    return emoji_code.lower()

slack_emoji_name_to_codepoint: Dict[str, str] = {}
for emoji_dict in emoji_data:
    short_name = emoji_dict["short_name"]
    emoji_code = get_emoji_code(emoji_dict)
    slack_emoji_name_to_codepoint[short_name] = emoji_code
    for sn in emoji_dict["short_names"]:
        if sn != short_name:
            slack_emoji_name_to_codepoint[sn] = emoji_code

class SlackBotEmail:
    duplicate_email_count: Dict[str, int] = {}
    assigned_email: Dict[str, str] = {}

    @classmethod
    def get_email(cls: Type[SlackBotEmailT], user_profile: ZerverFieldsT, domain_name: str) -> str:
        slack_bot_id = user_profile["bot_id"]
        if slack_bot_id in cls.assigned_email:
            return cls.assigned_email[slack_bot_id]

        if "real_name_normalized" in user_profile:
            slack_bot_name = user_profile["real_name_normalized"]
        elif "first_name" in user_profile:
            slack_bot_name = user_profile["first_name"]
        else:
            raise AssertionError("Could not identify bot type")

        email = Address(
            username=slack_bot_name.replace("Bot", "").replace(" ", "").lower() + "-bot",
            domain=domain_name,
        ).addr_spec

        if email in cls.duplicate_email_count:
            cls.duplicate_email_count[email] += 1
            address = Address(addr_spec=email)
            email_username = address.username + "-" + str(cls.duplicate_email_count[email])
            email = Address(username=email_username, domain=address.domain).addr_spec
        else:
            cls.duplicate_email_count[email] = 1

        cls.assigned_email[slack_bot_id] = email
        return email

def rm_tree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)

def slack_workspace_to_realm(
    domain_name: str,
    realm_id: int,
    user_list: List[ZerverFieldsT],
    realm_subdomain: str,
    slack_data_dir: str,
    custom_emoji_list: ZerverFieldsT,
) -> Tuple[
    ZerverFieldsT,
    SlackToZulipUserIDT,
    SlackToZulipRecipientT,
    AddedChannelsT,
    AddedMPIMsT,
    DMMembersT,
    List[ZerverFieldsT],
    ZerverFieldsT,
]:
    NOW = float(timezone_now().timestamp())

    zerver_realm: List[ZerverFieldsT] = build_zerver_realm(realm_id, realm_subdomain, NOW, "Slack")
    realm = build_realm(zerver_realm, realm_id, domain_name)

    (
        zerver_userprofile,
        avatars,
        slack_user_id_to_zulip_user_id,
        zerver_customprofilefield,
        zerver_customprofilefield_value,
    ) = users_to_zerver_userprofile(slack_data_dir, user_list, realm_id, int(NOW), domain_name)
    (
        realm,
        added_channels,
        added_mpims,
        dm_members,
        slack_recipient_name_to_zulip_recipient_id,
    ) = channels_to_zerver_stream(
        slack_data_dir, realm_id, realm, slack_user_id_to_zulip_user_id, zerver_userprofile
    )

    zerver_realmemoji, emoji_url_map = build_realmemoji(custom_emoji_list, realm_id)
    realm["zerver_realmemoji"] = zerver_realmemoji

    realm["zerver_userprofile"] = zerver_userprofile
    realm["zerver_customprofilefield"] = zerver_customprofilefield
    realm["zerver_customprofilefieldvalue"] = zerver_customprofilefield_value

    return (
        realm,
        slack_user_id_to_zulip_user_id,
        slack_recipient_name_to_zulip_recipient_id,
        added_channels,
        added_mpims,
        dm_members,
        avatars,
        emoji_url_map,
    )

def build_realmemoji(
    custom_emoji_list: ZerverFieldsT, realm_id: int
) -> Tuple[List[ZerverFieldsT], ZerverFieldsT]:
    zerver_realmemoji: List[ZerverFieldsT] = []
    emoji_url_map: Dict[str, str] = {}
    emoji_id = 0
    for emoji_name, url in custom_emoji_list.items():
        split_url = urlsplit(url)
        if split_url.hostname == "emoji.slack-edge.com":
            content_type = guess_type(posixpath.basename(split_url.path))[0]
            assert content_type is not None
            realmemoji = RealmEmoji(
                name=emoji_name,
                id=emoji_id,
                file_name=get_emoji_file_name(content_type, emoji_id),
                deactivated=False,
            )

            realmemoji_dict = model_to_dict(realmemoji, exclude=["realm", "author"])
            realmemoji_dict["author"] = None
            realmemoji_dict["realm"] = realm_id

            emoji_url_map[emoji_name] = url
            zerver_realmemoji.append(realmemoji_dict)
            emoji_id += 1
    return zerver_realmemoji, emoji_url_map

def users_to_zerver_userprofile(
    slack_data_dir: str, users: List[ZerverFieldsT], realm_id: int, timestamp: Any, domain_name: str
) -> Tuple[
    List[ZerverFieldsT],
    List[ZerverFieldsT],
    SlackToZulipUserIDT,
    List[ZerverFieldsT],
    List[ZerverFieldsT],
]:
    logging.info("######### IMPORTING USERS STARTED #########\n")
    zerver_userprofile: List[ZerverFieldsT] = []
    zerver_customprofilefield: List[ZerverFieldsT] = []
    zerver_customprofilefield_values: List[ZerverFieldsT] = []
    avatar_list: List[ZerverFieldsT] = []
    slack_user_id_to_zulip_user_id: Dict[str, int] = {}

    slack_data_file_user_list = get_data_file(slack_data_dir + "/users.json")

    slack_user_id_to_custom_profile_fields: Dict[str, Dict[str, Any]] = {}
    slack_custom_field_name_to_zulip_custom_field_id: Dict[str, int] = {}

    for user in slack_data_file_user_list:
        process_slack_custom_fields(user, slack_user_id_to_custom_profile_fields)

    user_id_count = custom_profile_field_value_id_count = custom_profile_field_id_count = 0
    primary_owner_id = user_id_count
    user_id_count += 1

    found_emails: Dict[str, int] = {}
    for user in users:
        slack_user_id = user["id"]

        if user.get("is_primary_owner", False):
            user_id = primary_owner_id
        else:
            user_id = user_id_count

        email = get_user_email(user, domain_name)
        if email.lower() in found_emails:
            slack_user_id_to_zulip_user_id[slack_user_id] = found_emails[email.lower()]
            logging.info("%s: %s MERGED", slack_user_id, email)
            continue
        found_emails[email.lower()] = user_id

        avatar_source, avatar_url = build_avatar_url(slack_user_id, user)
        if avatar_source == UserProfile.AVATAR_FROM_USER:
            build_avatar(user_id, realm_id, email, avatar_url, timestamp, avatar_list)
        role = UserProfile.ROLE_MEMBER
        if get_owner(user):
            role = UserProfile.ROLE_REALM_OWNER
        elif get_admin(user):
            role = UserProfile.ROLE_REALM_ADMINISTRATOR
        if get_guest(user):
            role = UserProfile.ROLE_GUEST
        timezone = get_user_timezone(user)

        if slack_user_id in slack_user_id_to_custom_profile_fields:
            (
                slack_custom_field_name_to_zulip_custom_field_id,
                custom_profile_field_id_count,
            ) = build_customprofile_field(
                zerver_customprofilefield,
                slack_user_id_to_custom_profile_fields[slack_user_id],
                custom_profile_field_id_count,
                realm_id,
                slack_custom_field_name_to_zulip_custom_field_id,
            )
            custom_profile_field_value_id_count = build_customprofilefields_values(
                slack_custom_field_name_to_zulip_custom_field_id,
                slack_user_id_to_custom_profile_fields[slack_user_id],
                user_id,
                custom_profile_field_value_id_count,
                zerver_customprofilefield_values,
            )

        userprofile = UserProfile(
            full_name=get_user_full_name(user),
            is_active=not user.get("deleted", False) and not user["is_mirror_dummy"],
            is_mirror_dummy=user["is_mirror_dummy"],
            id=user_id,
            email=email,
            delivery_email=email,
            avatar_source=avatar_source,
            is_bot=user.get("is_bot", False),
            role=role,
            bot_type=1 if user.get("is_bot", False) else None,
            date_joined=timestamp,
            timezone=timezone,
            last_login=timestamp,
        )
        userprofile_dict = model_to_dict(userprofile)
        userprofile_dict["realm"] = realm_id

        zerver_userprofile.append(userprofile_dict)
        slack_user_id_to_zulip_user_id[slack_user_id] = user_id
        if not user.get("is_primary_owner", False):
            user_id_count += 1

        logging.info("%s: %s -> %s", slack_user_id, user["name"], userprofile_dict["email"])

    validate_user_emails_for_import(list(found_emails))
    process_customprofilefields(zerver_customprofilefield, zerver_customprofilefield_values)
    logging.info("######### IMPORTING USERS FINISHED #########\n")
    return (
        zerver_userprofile,
        avatar_list,
        slack_user_id_to_zulip_user_id,
        zerver_customprofilefield,
        zerver_customprofilefield_values,
    )

def build_customprofile_field(
    customprofile_field: List[ZerverFieldsT],
    fields: Dict[str, Any],
    custom_profile_field_id: int,
    realm_id: int,
    slack_custom_field_name_to_zulip_custom_field_id: Dict[str, int],
) -> Tuple[Dict[str, int], int]:
    for field in fields:
        if field not in slack_custom_field_name_to_zulip_custom_field_id:
            slack_custom_fields = ["phone", "skype"]
            if field in slack_custom_fields:
                field_name = field
            else:
                field_name = f"Slack custom field {custom_profile_field_id + 1}"
            customprofilefield = CustomProfileField(
                id=custom_profile_field_id,
                name=field_name,
                field_type=1,
            )

            customprofilefield_dict = model_to_dict(customprofilefield, exclude=["realm"])
            customprofilefield_dict["realm"] = realm_id

            slack_custom_field_name_to_zulip_custom_field_id[field] = custom_profile_field_id
            custom_profile_field_id += 1
            customprofile_field.append(customprofilefield_dict)
    return slack_custom_field_name_to_zulip_custom_field_id, custom_profile_field_id

def process_slack_custom_fields(
    user: ZerverFieldsT, slack_user_id_to_custom_profile_fields: Dict[str, Dict[str, Any]]
) -> None:
    slack_user_id_to_custom_profile_fields[user["id"]] = {}
    if user["profile"].get("fields"):
        slack_user_id_to_custom_profile_fields[user["id"]] = user["profile"]["fields"]

    slack_custom_fields = ["phone", "skype"]
    for field in slack_custom_fields:
        if field in user["profile"]:
            slack_user_id_to_custom_profile_fields[user["id"]][field] = {
                "value": user["profile"][field]
            }

def build_customprofilefields_values(
    slack_custom_field_name_to_zulip_custom_field_id: Dict[str, int],
    fields: Dict[str, Any],
    user_id: int,
    custom_field_id: int,
    custom_field_values: List[ZerverFieldsT],
) -> int:
    for field, value in fields.items():
        if value["value"] == "":
            continue
        custom_field_value = CustomProfileFieldValue(id=custom_field_id, value=value["value"])

        custom_field_value_dict = model_to_dict(
            custom_field_value, exclude=["user_profile", "field"]
        )
        custom_field_value_dict["user_profile"] = user_id
        custom_field_value_dict["field"] = slack_custom_field_name_to_zulip_custom_field_id[field]

        custom_field_values.append(custom_field_value_dict)
        custom_field_id += 1
    return custom_field_id

def process_customprofilefields(
    customprofilefield: List[ZerverFieldsT], customprofilefield_value: List[ZerverFieldsT]
) -> None:
    for field in customprofilefield:
        for field_value in customprofilefield_value:
            if field_value["field"] == field["id"] and len(field_value["value"]) > 50:
                field["field_type"] = 2
                break

def get_user_email(user: ZerverFieldsT, domain_name: str) -> str:
    if "email" in user["profile"]:
        return user["profile"]["email"]
    if user["is_mirror_dummy"]:
        return Address(username=user["name"], domain=f"{user['team_domain']}.slack.com").addr_spec
    if "bot_id" in user["profile"]:
        return SlackBotEmail.get_email(user["profile"], domain_name)
    if get_user_full_name(user).lower() == "slackbot":
        return Address(username="imported-slackbot-bot", domain=domain_name).addr_spec
    raise AssertionError(f"Could not find email address for Slack user {user}")

def build_avatar_url(slack_user_id: str, user: ZerverFieldsT) -> Tuple[str, str]:
    avatar_url: str = ""
    avatar_source = UserProfile.AVATAR_FROM_GRAVATAR
    if user["profile"].get("avatar_hash"):
        team_id = user["team_id"]
        avatar_hash = user["profile"]["avatar_hash"]
        avatar_url = f"https://ca.slack-edge.com/{team_id}-{slack_user_id}-{avatar_hash}"
        avatar_source = UserProfile.AVATAR_FROM_USER
    elif user.get("is_integration_bot"):
        avatar_url = user["profile"]["image_72"]
        content_type = guess_type(avatar_url)[0]
        if content_type not in THUMBNAIL_ACCEPT_IMAGE_TYPES:
            logging.info(
                "Unsupported avatar type (%s) for user -> %s\n", content_type, user.get("name")
            )
            avatar_source = UserProfile.AVATAR_FROM_GRAVATAR