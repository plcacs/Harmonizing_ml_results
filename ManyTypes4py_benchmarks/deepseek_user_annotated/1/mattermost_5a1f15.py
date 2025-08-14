import logging
import os
import random
import re
import secrets
import shutil
import subprocess
from collections.abc import Callable
from typing import Any, Dict, List, Set, Tuple, Union, Optional, FrozenSet

import orjson
from django.conf import settings
from django.forms.models import model_to_dict
from django.utils.timezone import now as timezone_now

from zerver.data_import.import_util import (
    SubscriberHandler,
    ZerverFieldsT,
    build_attachment,
    build_direct_message_group,
    build_direct_message_group_subscriptions,
    build_message,
    build_personal_subscriptions,
    build_realm,
    build_realm_emoji,
    build_recipients,
    build_stream,
    build_stream_subscriptions,
    build_user_profile,
    build_zerver_realm,
    create_converted_data_files,
    make_subscriber_map,
    make_user_messages,
)
from zerver.data_import.sequencer import NEXT_ID, IdMapper
from zerver.data_import.user_handler import UserHandler
from zerver.lib.emoji import name_to_codepoint
from zerver.lib.export import do_common_export_processes
from zerver.lib.markdown import IMAGE_EXTENSIONS
from zerver.lib.upload import sanitize_name
from zerver.lib.utils import process_list_in_batches
from zerver.models import Reaction, RealmEmoji, Recipient, UserProfile


def make_realm(realm_id: int, team: Dict[str, Any]) -> ZerverFieldsT:
    NOW = float(timezone_now().timestamp())
    domain_name = settings.EXTERNAL_HOST
    realm_subdomain = team["name"]

    zerver_realm = build_zerver_realm(realm_id, realm_subdomain, NOW, "Mattermost")
    realm = build_realm(zerver_realm, realm_id, domain_name)

    realm["zerver_defaultstream"] = []

    return realm


def process_user(
    user_dict: Dict[str, Any], realm_id: int, team_name: str, user_id_mapper: IdMapper[str]
) -> ZerverFieldsT:
    def is_team_admin(user_dict: Dict[str, Any]) -> bool:
        if user_dict["teams"] is None:
            return False
        return any(
            team["name"] == team_name and "team_admin" in team["roles"]
            for team in user_dict["teams"]
        )

    def is_team_guest(user_dict: Dict[str, Any]) -> bool:
        if user_dict["teams"] is None:
            return False
        for team in user_dict["teams"]:
            if team["name"] == team_name and "team_guest" in team["roles"]:
                return True
        return False

    def get_full_name(user_dict: Dict[str, Any]) -> str:
        full_name = "{} {}".format(user_dict["first_name"], user_dict["last_name"])
        if full_name.strip():
            return full_name
        return user_dict["username"]

    avatar_source = "G"
    full_name = get_full_name(user_dict)
    id = user_id_mapper.get(user_dict["username"])
    delivery_email = user_dict["email"]
    email = user_dict["email"]
    short_name = user_dict["username"]
    date_joined = int(timezone_now().timestamp())
    timezone = "UTC"

    if is_team_admin(user_dict):
        role = UserProfile.ROLE_REALM_OWNER
    elif is_team_guest(user_dict):
        role = UserProfile.ROLE_GUEST
    else:
        role = UserProfile.ROLE_MEMBER

    if user_dict["is_mirror_dummy"]:
        is_active = False
        is_mirror_dummy = True
    else:
        is_active = True
        is_mirror_dummy = False

    return build_user_profile(
        avatar_source=avatar_source,
        date_joined=date_joined,
        delivery_email=delivery_email,
        email=email,
        full_name=full_name,
        id=id,
        is_active=is_active,
        role=role,
        is_mirror_dummy=is_mirror_dummy,
        realm_id=realm_id,
        short_name=short_name,
        timezone=timezone,
    )


def convert_user_data(
    user_handler: UserHandler,
    user_id_mapper: IdMapper[str],
    user_data_map: Dict[str, Dict[str, Any]],
    realm_id: int,
    team_name: str,
) -> None:
    for user_data in user_data_map.values():
        if check_user_in_team(user_data, team_name) or user_data["is_mirror_dummy"]:
            user = process_user(user_data, realm_id, team_name, user_id_mapper)
            user_handler.add_user(user)
    user_handler.validate_user_emails()


def convert_channel_data(
    channel_data: List[ZerverFieldsT],
    user_data_map: Dict[str, Dict[str, Any]],
    subscriber_handler: SubscriberHandler,
    stream_id_mapper: IdMapper[str],
    user_id_mapper: IdMapper[str],
    realm_id: int,
    team_name: str,
) -> List[ZerverFieldsT]:
    channel_data_list = [d for d in channel_data if d["team"] == team_name]

    channel_members_map: Dict[str, List[str]] = {}
    channel_admins_map: Dict[str, List[str]] = {}

    def initialize_stream_membership_dicts() -> None:
        for channel in channel_data:
            channel_name = channel["name"]
            channel_members_map[channel_name] = []
            channel_admins_map[channel_name] = []

        for username, user_dict in user_data_map.items():
            teams = user_dict["teams"]
            if user_dict["teams"] is None:
                continue

            for team in teams:
                if team["name"] != team_name:
                    continue
                for channel in team["channels"]:
                    channel_roles = channel["roles"]
                    channel_name = channel["name"]
                    if "channel_admin" in channel_roles:
                        channel_admins_map[channel_name].append(username)
                    elif "channel_user" in channel_roles:
                        channel_members_map[channel_name].append(username)

    def get_invite_only_value_from_channel_type(channel_type: str) -> bool:
        if channel_type == "O":
            return False
        elif channel_type == "P":
            return True
        else:
            raise Exception("unexpected value")

    streams = []
    initialize_stream_membership_dicts()

    for channel_dict in channel_data_list:
        now = int(timezone_now().timestamp())
        stream_id = stream_id_mapper.get(channel_dict["name"])
        stream_name = channel_dict["name"]
        invite_only = get_invite_only_value_from_channel_type(channel_dict["type"])

        stream = build_stream(
            date_created=now,
            realm_id=realm_id,
            name=channel_dict["display_name"],
            description=channel_dict["purpose"] or channel_dict["header"],
            stream_id=stream_id,
            deactivated=False,
            invite_only=invite_only,
        )

        channel_users = {
            *(user_id_mapper.get(username) for username in channel_admins_map[stream_name]),
            *(user_id_mapper.get(username) for username in channel_members_map[stream_name]),
        }

        subscriber_handler.set_info(
            users=channel_users,
            stream_id=stream_id,
        )
        streams.append(stream)
    return streams


def convert_direct_message_group_data(
    direct_message_group_data: List[ZerverFieldsT],
    user_data_map: Dict[str, Dict[str, Any]],
    subscriber_handler: SubscriberHandler,
    direct_message_group_id_mapper: IdMapper[FrozenSet[str]],
    user_id_mapper: IdMapper[str],
    realm_id: int,
    team_name: str,
) -> List[ZerverFieldsT]:
    zerver_direct_message_group = []
    for direct_message_group in direct_message_group_data:
        if len(direct_message_group["members"]) > 2:
            direct_message_group_members = frozenset(direct_message_group["members"])
            if direct_message_group_id_mapper.has(direct_message_group_members):
                logging.info("Duplicate direct message group found in the export data. Skipping.")
                continue
            direct_message_group_id = direct_message_group_id_mapper.get(
                direct_message_group_members
            )
            direct_message_group_dict = build_direct_message_group(
                direct_message_group_id, len(direct_message_group_members)
            )
            direct_message_group_user_ids = {
                user_id_mapper.get(username) for username in direct_message_group["members"]
            }
            subscriber_handler.set_info(
                users=direct_message_group_user_ids,
                direct_message_group_id=direct_message_group_id,
            )
            zerver_direct_message_group.append(direct_message_group_dict)
    return zerver_direct_message_group


def build_reactions(
    realm_id: int,
    total_reactions: List[ZerverFieldsT],
    reactions: List[ZerverFieldsT],
    message_id: int,
    user_id_mapper: IdMapper[str],
    zerver_realmemoji: List[ZerverFieldsT],
) -> None:
    realmemoji = {}
    for realm_emoji in zerver_realmemoji:
        realmemoji[realm_emoji["name"]] = realm_emoji["id"]

    for mattermost_reaction in reactions:
        emoji_name = mattermost_reaction["emoji_name"]
        username = mattermost_reaction["user"]
        if emoji_name in name_to_codepoint:
            emoji_code = name_to_codepoint[emoji_name]
            reaction_type = Reaction.UNICODE_EMOJI
        elif emoji_name in realmemoji:
            emoji_code = realmemoji[emoji_name]
            reaction_type = Reaction.REALM_EMOJI
        else:
            continue

        if not user_id_mapper.has(username):
            continue

        reaction_id = NEXT_ID("reaction")
        reaction = Reaction(
            id=reaction_id,
            emoji_code=emoji_code,
            emoji_name=emoji_name,
            reaction_type=reaction_type,
        )

        reaction_dict = model_to_dict(reaction, exclude=["message", "user_profile"])
        reaction_dict["message"] = message_id
        reaction_dict["user_profile"] = user_id_mapper.get(username)
        total_reactions.append(reaction_dict)


def get_mentioned_user_ids(raw_message: Dict[str, Any], user_id_mapper: IdMapper[str]) -> Set[int]:
    user_ids = set()
    content = raw_message["content"]

    matches = re.findall(r"(?<=^|(?<=[^a-zA-Z0-9-_.]))@(([A-Za-z0-9]+[_.]?)+)", content)

    for match in matches:
        possible_username = match[0]
        if user_id_mapper.has(possible_username):
            user_ids.add(user_id_mapper.get(possible_username))
    return user_ids


def process_message_attachments(
    attachments: List[Dict[str, Any]],
    realm_id: int,
    message_id: int,
    user_id: int,
    user_handler: UserHandler,
    zerver_attachment: List[ZerverFieldsT],
    uploads_list: List[ZerverFieldsT],
    mattermost_data_dir: str,
    output_dir: str,
) -> Tuple[str, bool]:
    has_image = False
    markdown_links = []

    for attachment in attachments:
        attachment_path = attachment["path"]
        attachment_full_path = os.path.join(mattermost_data_dir, "data", attachment_path)

        file_name = attachment_path.split("/")[-1]
        file_ext = f".{file_name.split('.')[-1]}"

        if file_ext.lower() in IMAGE_EXTENSIONS:
            has_image = True

        s3_path = "/".join(
            [
                str(realm_id),
                format(random.randint(0, 255), "x"),
                secrets.token_urlsafe(18),
                sanitize_name(file_name),
            ]
        )
        content_for_link = f"[{file_name}](/user_uploads/{s3_path})"

        markdown_links.append(content_for_link)

        fileinfo = {
            "name": file_name,
            "size": os.path.getsize(attachment_full_path),
            "created": os.path.getmtime(attachment_full_path),
        }

        upload = dict(
            path=s3_path,
            realm_id=realm_id,
            content_type=None,
            user_profile_id=user_id,
            last_modified=fileinfo["created"],
            user_profile_email=user_handler.get_user(user_id=user_id)["email"],
            s3_path=s3_path,
            size=fileinfo["size"],
        )
        uploads_list.append(upload)

        build_attachment(
            realm_id=realm_id,
            message_ids={message_id},
            user_id=user_id,
            fileinfo=fileinfo,
            s3_path=s3_path,
            zerver_attachment=zerver_attachment,
        )

        attachment_out_path = os.path.join(output_dir, "uploads", s3_path)
        os.makedirs(os.path.dirname(attachment_out_path), exist_ok=True)
        shutil.copyfile(attachment_full_path, attachment_out_path)

    content = "\n".join(markdown_links)
    return content, has_image


def process_raw_message_batch(
    realm_id: int,
    raw_messages: List[Dict[str, Any]],
    subscriber_map: Dict[int, Set[int]],
    user_id_mapper: IdMapper[str],
    user_handler: UserHandler,
    get_recipient_id_from_channel_name: Callable[[str], int],
    get_recipient_id_from_direct_message_group_members: Callable[[FrozenSet[str]], int],
    get_recipient_id_from_username: Callable[[str], int],
    is_pm_data: bool,
    output_dir: str,
    zerver_realmemoji: List[Dict[str, Any]],
    total_reactions: List[Dict[str, Any]],
    uploads_list: List[ZerverFieldsT],
    zerver_attachment: List[ZerverFieldsT],
    mattermost_data_dir: str,
) -> None:
    def fix_mentions(content: str, mention_user_ids: Set[int]) -> str:
        for user_id in mention_user_ids:
            user = user_handler.get_user(user_id=user_id)
            mattermost_mention = "@{short_name}".format(**user)
            zulip_mention = "@**{full_name}**".format(**user)
            content = content.replace(mattermost_mention, zulip_mention)

        content = content.replace("@channel", "@**all**")
        content = content.replace("@all", "@**all**")
        content = content.replace("@here", "@**all**")
        return content

    mention_map: Dict[int, Set[int]] = {}
    zerver_message = []
    pm_members = {}

    for raw_message in raw_messages:
        message_id = NEXT_ID("message")
        mention_user_ids = get_mentioned_user_ids(raw_message, user_id_mapper)
        mention_map[message_id] = mention_user_ids

        content = fix_mentions(
            content=raw_message["content"],
            mention_user_ids=mention_user_ids,
        )

        content = subprocess.check_output(["html2text", "--unicode-snob"], input=content, text=True)

        date_sent = raw_message["date_sent"]
        sender_user_id = raw_message["sender_id"]
        if "channel_name" in raw_message:
            recipient_id = get_recipient_id_from_channel_name(raw_message["channel_name"])
        elif "direct_message_group_members" in raw_message:
            recipient_id = get_recipient_id_from_direct_message_group_members(
                raw_message["direct_message_group_members"]
            )
        elif "pm_members" in raw_message:
            members = raw_message["pm_members"]
            member_ids = {user_id_mapper.get(member) for member in members}
            pm_members[message_id] = member_ids
            if sender_user_id == user_id_mapper.get(members[0]):
                recipient_id = get_recipient_id_from_username(members[1])
            else:
                recipient_id = get_recipient_id_from_username(members[0])
        else:
            raise AssertionError(
                "raw_message without channel_name, direct_message_group_members or pm_members key"
            )

        rendered_content = None

        has_attachment = False
        has_image = False
        has_link = False
        if "attachments" in raw_message:
            has_attachment = True
            has_link = True

            attachment_markdown, has_image = process_message_attachments(
                attachments=raw_message["attachments"],
                realm_id=realm_id,
                message_id=message_id,
                user_id=sender_user_id,
                user_handler=user_handler,
                zerver_attachment=zerver_attachment,
                uploads_list=uploads_list,
                mattermost_data_dir=mattermost_data_dir,
                output_dir=output_dir,
            )

            content += attachment_markdown

        topic_name = "imported from mattermost"

        message = build_message(
            content=content,
            message_id=message_id,
            date_sent=date_sent,
            recipient_id=recipient_id,
            realm_id=realm_id,
            rendered_content=rendered_content,
            topic_name=topic_name,
            user_id=sender_user_id,
            has_image=has_image,
            has_link=has_link,
            has_attachment=has_attachment,
        )
        zerver_message.append(message)
        build_reactions(
            realm_id,
            total_reactions,
            raw_message["reactions"],
            message_id,
            user_id_mapper,
            zerver_realmemoji,
        )

    zerver_usermessage = make_user_messages(
        zerver_message=zerver_message,
        subscriber_map=subscriber_map,
        is_pm_data=is_pm_data,
        mention_map=mention_map,
    )

    message_json = dict(
        zerver_message=zerver_message,
        zerver_usermessage=zerver_usermessage,
    )

    dump_file_id = NEXT_ID("dump_file_id" + str(realm_id))
    message_file = f"/messages-{dump_file_id:06}.json"
    create_converted_data_files(message_json, output_dir, message_file)


def process_posts(
    num_teams: int,
    team_name: str,
    realm_id: int,
    post_data: List[Dict[str, Any]],
    get