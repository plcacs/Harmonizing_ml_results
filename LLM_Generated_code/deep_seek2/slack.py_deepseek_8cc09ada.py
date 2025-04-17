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
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
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

emoji_data_file_path: str = static_path("generated/emoji/emoji-datasource-google-emoji.json")
with open(emoji_data_file_path, "rb") as emoji_data_file:
    emoji_data: Dict[str, Any] = orjson.loads(emoji_data_file.read())

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
    zerver_realmemoji = []
    emoji_url_map = {}
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
    zerver_userprofile = []
    zerver_customprofilefield: List[ZerverFieldsT] = []
    zerver_customprofilefield_values: List[ZerverFieldsT] = []
    avatar_list: List[ZerverFieldsT] = []
    slack_user_id_to_zulip_user_id = {}

    slack_data_file_user_list = get_data_file(slack_data_dir + "/users.json")

    slack_user_id_to_custom_profile_fields: ZerverFieldsT = {}
    slack_custom_field_name_to_zulip_custom_field_id: ZerverFieldsT = {}

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
    fields: ZerverFieldsT,
    custom_profile_field_id: int,
    realm_id: int,
    slack_custom_field_name_to_zulip_custom_field_id: ZerverFieldsT,
) -> Tuple[ZerverFieldsT, int]:
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
    user: ZerverFieldsT, slack_user_id_to_custom_profile_fields: ZerverFieldsT
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
    slack_custom_field_name_to_zulip_custom_field_id: ZerverFieldsT,
    fields: ZerverFieldsT,
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
        else:
            avatar_source = UserProfile.AVATAR_FROM_USER
    else:
        logging.info("Failed to process avatar for user -> %s\n", user.get("name"))
    return avatar_source, avatar_url

def get_owner(user: ZerverFieldsT) -> bool:
    owner = user.get("is_owner", False)
    primary_owner = user.get("is_primary_owner", False)

    return primary_owner or owner

def get_admin(user: ZerverFieldsT) -> bool:
    admin = user.get("is_admin", False)
    return admin

def get_guest(user: ZerverFieldsT) -> bool:
    restricted_user = user.get("is_restricted", False)
    ultra_restricted_user = user.get("is_ultra_restricted", False)

    return restricted_user or ultra_restricted_user

def get_user_timezone(user: ZerverFieldsT) -> str:
    _default_timezone = "America/New_York"
    timezone = user.get("tz", _default_timezone)
    if timezone is None or "/" not in timezone:
        timezone = _default_timezone
    return timezone

def channels_to_zerver_stream(
    slack_data_dir: str,
    realm_id: int,
    realm: Dict[str, Any],
    slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
    zerver_userprofile: List[ZerverFieldsT],
) -> Tuple[
    Dict[str, List[ZerverFieldsT]], AddedChannelsT, AddedMPIMsT, DMMembersT, SlackToZulipRecipientT
]:
    logging.info("######### IMPORTING CHANNELS STARTED #########\n")

    added_channels = {}
    added_mpims = {}
    dm_members = {}
    slack_recipient_name_to_zulip_recipient_id = {}

    realm["zerver_stream"] = []
    realm["zerver_huddle"] = []
    realm["zerver_subscription"] = []
    realm["zerver_recipient"] = []
    realm["zerver_defaultstream"] = []

    subscription_id_count = recipient_id_count = 0
    stream_id_count = defaultstream_id = 0
    direct_message_group_id_count = 0

    def process_channels(channels: List[Dict[str, Any]], invite_only: bool = False) -> None:
        nonlocal stream_id_count, recipient_id_count, defaultstream_id, subscription_id_count

        for channel in channels:
            description = channel["purpose"]["value"]
            stream_id = stream_id_count
            recipient_id = recipient_id_count

            stream = build_stream(
                float(channel["created"]),
                realm_id,
                channel["name"],
                description,
                stream_id,
                channel["is_archived"],
                invite_only,
            )
            realm["zerver_stream"].append(stream)

            slack_default_channels = ["general", "random"]
            if channel["name"] in slack_default_channels and not stream["deactivated"]:
                defaultstream = build_defaultstream(realm_id, stream_id, defaultstream_id)
                realm["zerver_defaultstream"].append(defaultstream)
                defaultstream_id += 1

            added_channels[stream["name"]] = (channel["id"], stream_id)

            recipient = build_recipient(stream_id, recipient_id, Recipient.STREAM)
            realm["zerver_recipient"].append(recipient)
            slack_recipient_name_to_zulip_recipient_id[stream["name"]] = recipient_id

            subscription_id_count = get_subscription(
                channel["members"],
                realm["zerver_subscription"],
                recipient_id,
                slack_user_id_to_zulip_user_id,
                subscription_id_count,
            )

            stream_id_count += 1
            recipient_id_count += 1
            logging.info("%s -> created", channel["name"])

    public_channels = get_data_file(slack_data_dir + "/channels.json")
    process_channels(public_channels)

    try:
        private_channels = get_data_file(slack_data_dir + "/groups.json")
    except FileNotFoundError:
        private_channels = []
    process_channels(private_channels, True)

    def process_mpims(mpims: List[Dict[str, Any]]) -> None:
        nonlocal direct_message_group_id_count, recipient_id_count, subscription_id_count

        for mpim in mpims:
            direct_message_group = build_direct_message_group(
                direct_message_group_id_count, len(mpim["members"])
            )
            realm["zerver_huddle"].append(direct_message_group)

            added_mpims[mpim["name"]] = (mpim["id"], direct_message_group_id_count)

            recipient = build_recipient(
                direct_message_group_id_count, recipient_id_count, Recipient.DIRECT_MESSAGE_GROUP
            )
            realm["zerver_recipient"].append(recipient)
            slack_recipient_name_to_zulip_recipient_id[mpim["name"]] = recipient_id_count

            subscription_id_count = get_subscription(
                mpim["members"],
                realm["zerver_subscription"],
                recipient_id_count,
                slack_user_id_to_zulip_user_id,
                subscription_id_count,
            )

            direct_message_group_id_count += 1
            recipient_id_count += 1
            logging.info("%s -> created", mpim["name"])

    try:
        mpims = get_data_file(slack_data_dir + "/mpims.json")
    except FileNotFoundError:
        mpims = []
    process_mpims(mpims)

    zulip_user_to_recipient: Dict[int, int] = {}
    for slack_user_id, zulip_user_id in slack_user_id_to_zulip_user_id.items():
        if zulip_user_id in zulip_user_to_recipient:
            slack_recipient_name_to_zulip_recipient_id[slack_user_id] = zulip_user_to_recipient[
                zulip_user_id
            ]
            continue
        recipient = build_recipient(zulip_user_id, recipient_id_count, Recipient.PERSONAL)
        slack_recipient_name_to_zulip_recipient_id[slack_user_id] = recipient_id_count
        zulip_user_to_recipient[zulip_user_id] = recipient_id_count
        sub = build_subscription(recipient_id_count, zulip_user_id, subscription_id_count)
        realm["zerver_recipient"].append(recipient)
        realm["zerver_subscription"].append(sub)
        recipient_id_count += 1
        subscription_id_count += 1

    def process_dms(dms: List[Dict[str, Any]]) -> None:
        for dm in dms:
            user_a = dm["members"][0]
            user_b = dm["members"][1]
            dm_members[dm["id"]] = (user_a, user_b)

    try:
        dms = get_data_file(slack_data_dir + "/dms.json")
    except FileNotFoundError:
        dms = []
    process_dms(dms)

    logging.info("######### IMPORTING STREAMS FINISHED #########\n")
    return (
        realm,
        added_channels,
        added_mpims,
        dm_members,
        slack_recipient_name_to_zulip_recipient_id,
    )

def get_subscription(
    channel_members: List[str],
    zerver_subscription: List[ZerverFieldsT],
    recipient_id: int,
    slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
    subscription_id: int,
) -> int:
    for slack_user_id in channel_members:
        sub = build_subscription(
            recipient_id, slack_user_id_to_zulip_user_id[slack_user_id], subscription_id
        )
        zerver_subscription.append(sub)
        subscription_id += 1
    return subscription_id

def process_long_term_idle_users(
    slack_data_dir: str,
    users: List[ZerverFieldsT],
    slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
    added_channels: AddedChannelsT,
    added_mpims: AddedMPIMsT,
    dm_members: DMMembersT,
    zerver_userprofile: List[ZerverFieldsT],
) -> Set[int]:
    return long_term_idle_helper(
        get_messages_iterator(slack_data_dir, added_channels, added_mpims, dm_members),
        get_message_sending_user,
        get_timestamp_from_message,
        lambda id: slack_user_id_to_zulip_user_id[id],
        iter(user["id"] for user in users),
        zerver_userprofile,
    )

def convert_slack_workspace_messages(
    slack_data_dir: str,
    users: List[ZerverFieldsT],
    realm_id: int,
    slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
    slack_recipient_name_to_zulip_recipient_id: SlackToZulipRecipientT,
    added_channels: AddedChannelsT,
    added_mpims: AddedMPIMsT,
    dm_members: DMMembersT,
    realm: ZerverFieldsT,
    zerver_userprofile: List[ZerverFieldsT],
    zerver_realmemoji: List[ZerverFieldsT],
    domain_name: str,
    output_dir: str,
    convert_slack_threads: bool,
    chunk_size: int = MESSAGE_BATCH_CHUNK_SIZE,
) -> Tuple[List[ZerverFieldsT], List[ZerverFieldsT], List[ZerverFieldsT]]:
    long_term_idle = process_long_term_idle_users(
        slack_data_dir,
        users,
        slack_user_id_to_zulip_user_id,
        added_channels,
        added_mpims,
        dm_members,
        zerver_userprofile,
    )

    all_messages = get_messages_iterator(slack_data_dir, added_channels, added_mpims, dm_members)
    logging.info("######### IMPORTING MESSAGES STARTED #########\n")

    total_reactions: List[ZerverFieldsT] = []
    total_attachments: List[ZerverFieldsT] = []
    total_uploads: List[ZerverFieldsT] = []

    dump_file_id = 1

    subscriber_map = make_subscriber_map(
        zerver_subscription=realm["zerver_subscription"],
    )

    while message_data := list(itertools.islice(all_messages, chunk_size)):
        (
            zerver_message,
            zerver_usermessage,
            attachment,
            uploads,
            reactions,
        ) = channel_message_to_zerver_message(
            realm_id,
            users,
            slack_user_id_to_zulip_user_id,
            slack_recipient_name_to_zulip_recipient_id,
            message_data,
            zerver_realmemoji,
            subscriber_map,
            added_channels,
            dm_members,
            domain_name,
            long_term_idle,
            convert_slack_threads,
        )

        message_json = dict(zerver_message=zerver_message, zerver_usermessage=zerver_usermessage)

        message_file = f"/messages-{dump_file_id:06}.json"
        logging.info("Writing messages to %s\n", output_dir + message_file)
        create_converted_data_files(message_json, output_dir, message_file)

        total_reactions += reactions
        total_attachments += attachment
        total_uploads += uploads

        dump_file_id += 1

    logging.info("######### IMPORTING MESSAGES FINISHED #########\n")
    return total_reactions, total_uploads, total_attachments

def get_messages_iterator(
    slack_data_dir: str,
    added_channels: Dict[str, Any],
    added_mpims: AddedMPIMsT,
    dm_members: DMMembersT,
) -> Iterator[ZerverFieldsT]:
    dir_names = [*added_channels, *added_mpims, *dm_members]
    all_json_names: Dict[str, List[str]] = defaultdict(list)
    for dir_name in dir_names:
        dir_path = os.path.join(slack_data_dir, dir_name)
        json_names = os.listdir(dir_path)
        for json_name in json_names:
            if json_name.endswith(".json"):
                all_json_names[json_name].append(dir_path)

    for json_name in sorted(all_json_names.keys()):
        messages_for_one_day: List[ZerverFieldsT] = []
        for dir_path in all_json_names[json_name]:
            message_dir = os.path.join(dir_path, json_name)
            dir_name = os.path.basename(dir_path)
            messages = []
            for message in get_data_file(message_dir):
                if message.get("user") == "U00":
                    continue
                if message.get("mimetype") == "application/vnd.slack-docs":
                    continue
                if dir_name in added_channels:
                    message["channel_name"] = dir_name
                elif dir_name in added_mpims:
                    message["mpim_name"] = dir_name
                elif dir_name in dm_members:
                    message["pm_name"] = dir_name
                messages.append(message)
            messages_for_one_day += messages

        yield from sorted(messages_for_one_day, key=get_timestamp_from_message)

def channel_message_to_zerver_message(
    realm_id: int,
    users: List[ZerverFieldsT],
    slack_user_id_to_zulip_user_id: SlackToZulipUserIDT,
    slack_recipient_name_to_zulip_recipient_id: SlackToZulipRecipientT,
    all_messages: List[ZerverFieldsT],
    zerver_realmemoji: List[ZerverFieldsT],
    subscriber_map: Dict[int, Set[int]],
    added_channels: AddedChannelsT,
    dm_members: DMMembersT,
    domain_name: str,
    long_term_idle: Set[int],
    convert_slack_threads: bool,
) -> Tuple[
    List[ZerverFieldsT],
    List[ZerverFieldsT],
    List[ZerverFieldsT],
    List[ZerverFieldsT],
    List[ZerverFieldsT],
]:
    zerver_message = []
    zerver_usermessage: List[ZerverFieldsT] = []
    uploads_list: List[ZerverFieldsT] = []
    zerver_attachment: List[ZerverFieldsT] = []
    reaction_list: List[ZerverFieldsT] = []

    total_user_messages = 0
    total_skipped_user_messages = 0
    thread_counter: Dict[str, int] = defaultdict(int)
    thread_map: Dict[str, str] = {}
    for message in all_messages:
        slack_user_id = get_message_sending_user(message)
        if not slack_user_id:
            continue

        subtype = message.get("subtype", False)
        if subtype in [
            "pinned_item",
            "unpinned_item",
            "channel_join",
            "channel_leave",
            "channel_name",
        ]:
            continue

        try:
            content, mentioned_user_ids, has_link = convert_to_zulip_markdown(
                message["text"], users, added_channels, slack_user_id_to_zulip_user_id
            )
        except Exception:
            print("Slack message unexpectedly missing text representation:")
            print(orjson.dumps(message, option=orjson.OPT_INDENT_2).decode())
            continue
        rendered_content = None

        if "channel_name" in message:
            is_private = False
            recipient_id = slack_recipient_name_to_zulip_recipient_id[message["channel_name"]]
        elif "mpim_name" in message:
            is_private = True
            recipient_id = slack_recipient_name_to_zulip_recipient_id[message["mpim_name"]]
        elif "pm_name" in message:
            is_private = True
            sender = get_message_sending_user(message)
            members = dm_members[message["pm_name"]]
            if sender == members[0]:
                recipient_id = slack_recipient_name_to_zulip_recipient_id[members[1]]
                sender_recipient_id = slack_recipient_name_to_zulip_recipient_id[members[0]]
            else:
                recipient_id = slack_recipient_name_to_zulip_recipient_id[members[0]]
                sender_recipient_id = slack_recipient_name_to_zulip_recipient_id[members[1]]

        message_id = NEXT_ID("message")

        if "reactions" in message:
            build_reactions(
                reaction_list,
                message["reactions"],
                slack_user_id_to_zulip_user_id,
                message_id,
                zerver_realmemoji,
            )

        if subtype in ["bot_add", "sh_room_created", "me_message"]:
            content = f"/me {content}"
        if subtype == "file_comment":
            message["user"] = message["comment"]["user"]

        file_info = process_message_files(
            message=message,
            domain_name=domain_name,
            realm_id=realm_id,
            message_id=message_id,
            slack_user_id=slack_user_id,
            users=users,
            slack_user_id_to_zulip_user_id=slack_user_id_to_zulip_user_id,
            zerver_attachment=zerver_attachment,
            uploads_list=uploads_list,
        )

        content = "\n".join([part for part in [content, file_info["content"]] if part != ""])
        has_link = has_link or file_info["has_link"]

        has_attachment = file_info["has_attachment"]
        has_image = file_info["has_image"]

        topic_name = "imported from Slack"
        if convert_slack_threads and "thread_ts" in message:
            thread_ts = datetime.fromtimestamp(float(message["thread_ts"]), tz=timezone.utc)
            thread_ts_str = thread_ts.strftime(r"%Y/%m/%d %H:%M:%S")
            if thread_ts_str in thread_map:
                topic_name = thread_map[thread_