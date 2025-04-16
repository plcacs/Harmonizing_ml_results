```python
import logging
import os
import random
import secrets
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import bson
from django.conf import settings
from django.forms.models import model_to_dict

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

bson_codec_options = bson.DEFAULT_CODEC_OPTIONS.with_options(tz_aware=True)


def make_realm(
    realm_id: int, realm_subdomain: str, domain_name: str, rc_instance: Dict[str, Any]
) -> ZerverFieldsT:
    created_at = float(rc_instance["_createdAt"].timestamp())

    zerver_realm = build_zerver_realm(realm_id, realm_subdomain, created_at, "Rocket.Chat")
    realm = build_realm(zerver_realm, realm_id, domain_name)

    # We may override these later.
    realm["zerver_defaultstream"] = []

    return realm


def process_users(
    user_id_to_user_map: Dict[str, Dict[str, Any]],
    realm_id: int,
    domain_name: str,
    user_handler: UserHandler,
    user_id_mapper: IdMapper[str],
) -> None:
    realm_owners: List[int] = []
    bots: List[int] = []

    for rc_user_id, user_dict in user_id_to_user_map.items():
        is_mirror_dummy = False
        is_bot = False
        is_active = True

        # Rocket.Chat has three user types:
        # "user": This is a regular user of the system.
        # "bot": A special user types for bots.
        # "unknown": This usually represents a livechat guest.
        if user_dict["type"] != "user":
            is_active = False

            if user_dict["type"] == "bot":
                is_bot = True
            else:
                is_mirror_dummy = True

        if user_dict.get("emails") is None:
            user_dict["emails"] = [
                {
                    "address": "{}-{}@{}".format(
                        user_dict["username"], user_dict["type"], domain_name
                    )
                }
            ]

        # TODO: Change this to use actual exported avatar
        avatar_source = "G"
        full_name = user_dict["name"]
        id = user_id_mapper.get(rc_user_id)
        delivery_email = user_dict["emails"][0]["address"]
        email = user_dict["emails"][0]["address"]
        short_name = user_dict["username"]
        date_joined = float(user_dict["createdAt"].timestamp())
        timezone = "UTC"

        role = UserProfile.ROLE_MEMBER
        if "admin" in user_dict["roles"]:
            role = UserProfile.ROLE_REALM_OWNER
            realm_owners.append(id)
        elif "guest" in user_dict["roles"]:
            role = UserProfile.ROLE_GUEST
        elif "bot" in user_dict["roles"]:
            is_bot = True

        if is_bot:
            bots.append(id)

        user = build_user_profile(
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
            is_bot=is_bot,
            bot_type=1 if is_bot else None,
        )
        user_handler.add_user(user)

    user_handler.validate_user_emails()
    # Set the first realm_owner as the owner of
    # all the bots.
    if realm_owners:
        for bot_id in bots:
            bot_user = user_handler.get_user(user_id=bot_id)
            bot_user["bot_owner"] = realm_owners[0]


def truncate_name(name: str, name_id: int, max_length: int = 60) -> str:
    if len(name) > max_length:
        name_id_suffix = f" [{name_id}]"
        name = name[0 : max_length - len(name_id_suffix)] + name_id_suffix
    return name


def get_stream_name(rc_channel: Dict[str, Any]) -> str:
    if rc_channel.get("teamMain"):
        stream_name = f"[TEAM] {rc_channel['name']}"
    else:
        stream_name = rc_channel["name"]

    stream_name = truncate_name(stream_name, rc_channel["_id"])

    return stream_name


def convert_channel_data(
    room_id_to_room_map: Dict[str, Dict[str, Any]],
    team_id_to_team_map: Dict[str, Dict[str, Any]],
    stream_id_mapper: IdMapper[str],
    realm_id: int,
) -> List[ZerverFieldsT]:
    streams = []

    for rc_room_id, channel_dict in room_id_to_room_map.items():
        date_created = float(channel_dict["ts"].timestamp())
        stream_id = stream_id_mapper.get(rc_room_id)
        invite_only = channel_dict["t"] == "p"

        stream_name = get_stream_name(channel_dict)

        stream_desc = channel_dict.get("description", "")
        if channel_dict.get("teamId") and not channel_dict.get("teamMain"):
            stream_desc = "[Team {} channel]. {}".format(
                team_id_to_team_map[channel_dict["teamId"]]["name"], stream_desc
            )

        # If the channel is read-only, then only admins and moderators
        # should be allowed to post in the converted Zulip stream.
        # For more details: https://zulip.com/help/channel-posting-policy
        #
        # See `Stream` model in `zerver/models/streams.py` to know about what each
        # number represent.
        stream_post_policy = 4 if channel_dict.get("ro", False) else 1

        stream = build_stream(
            date_created=date_created,
            realm_id=realm_id,
            name=stream_name,
            description=stream_desc,
            stream_id=stream_id,
            deactivated=False,
            invite_only=invite_only,
            stream_post_policy=stream_post_policy,
        )
        streams.append(stream)

    return streams


def convert_stream_subscription_data(
    user_id_to_user_map: Dict[str, Dict[str, Any]],
    dsc_id_to_dsc_map: Dict[str, Dict[str, Any]],
    zerver_stream: List[ZerverFieldsT],
    stream_id_mapper: IdMapper[str],
    user_id_mapper: IdMapper[str],
    subscriber_handler: SubscriberHandler,
) -> None:
    stream_members_map: Dict[int, Set[int]] = {}

    for rc_user_id, user_dict in user_id_to_user_map.items():
        if not user_dict.get("__rooms"):
            continue

        for channel in user_dict["__rooms"]:
            if channel in dsc_id_to_dsc_map:
                # Ignore discussion rooms as these are not
                # imported as streams, but topics.
                continue
            stream_id = stream_id_mapper.get(channel)
            if stream_id not in stream_members_map:
                stream_members_map[stream_id] = set()
            stream_members_map[stream_id].add(user_id_mapper.get(rc_user_id))

    for stream in zerver_stream:
        if stream["id"] in stream_members_map:
            users = stream_members_map[stream["id"]]
        else:
            users = set()
            # Set the stream without any subscribers
            # as deactivated.
            stream["deactivated"] = True
        subscriber_handler.set_info(users=users, stream_id=stream["id"])


def convert_direct_message_group_data(
    direct_message_group_id_to_direct_message_group_map: Dict[str, Dict[str, Any]],
    direct_message_group_id_mapper: IdMapper[str],
    user_id_mapper: IdMapper[str],
    subscriber_handler: SubscriberHandler,
) -> List[ZerverFieldsT]:
    zerver_direct_message_group: List[ZerverFieldsT] = []

    for (
        rc_direct_message_group_id,
        direct_message_group_dict,
    ) in direct_message_group_id_to_direct_message_group_map.items():
        direct_message_group_id = direct_message_group_id_mapper.get(rc_direct_message_group_id)
        direct_message_group = build_direct_message_group(
            direct_message_group_id, len(direct_message_group_dict["uids"])
        )
        zerver_direct_message_group.append(direct_message_group)

        direct_message_group_user_ids = {
            user_id_mapper.get(rc_user_id) for rc_user_id in direct_message_group_dict["uids"]
        }
        subscriber_handler.set_info(
            users=direct_message_group_user_ids,
            direct_message_group_id=direct_message_group_id,
        )

    return zerver_direct_message_group


def build_custom_emoji(
    realm_id: int, custom_emoji_data: Dict[str, List[Dict[str, Any]]], output_dir: str
) -> List[ZerverFieldsT]:
    logging.info("Starting to process custom emoji")

    emoji_folder = os.path.join(output_dir, "emoji")
    os.makedirs(emoji_folder, exist_ok=True)

    zerver_realmemoji: List[ZerverFieldsT] = []
    emoji_records: List[ZerverFieldsT] = []

    # Map emoji file_id to emoji file data
    emoji_file_data: Dict[str, Dict[str, Any]] = {}
    for emoji_file in custom_emoji_data["file"]:
        emoji_file_data[str(emoji_file["_id"])] = {**emoji_file, "chunks": []}
    for emoji_chunk in custom_emoji_data["chunk"]:
        emoji_file_data[emoji_chunk["files_id"]]["chunks"].append(emoji_chunk["data"])

    # Build custom emoji
    for rc_emoji in custom_emoji_data["emoji"]:
        # Subject to change with changes in database
        emoji_file_id = f"{rc_emoji['name']}.{rc_emoji['extension']}"

        emoji_file_info = emoji_file_data[emoji_file_id]

        emoji_filename = emoji_file_info["filename"]
        emoji_data = b"".join(emoji_file_info["chunks"])

        target_sub_path = RealmEmoji.PATH_ID_TEMPLATE.format(
            realm_id=realm_id,
            emoji_file_name=emoji_filename,
        )
        target_path = os.path.join(emoji_folder, target_sub_path)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as e_file:
            e_file.write(emoji_data)

        emoji_aliases = [rc_emoji["name"]]
        emoji_aliases.extend(rc_emoji["aliases"])

        for alias in emoji_aliases:
            emoji_record = dict(
                path=target_path,
                s3_path=target_path,
                file_name=emoji_filename,
                realm_id=realm_id,
                name=alias,
            )
            emoji_records.append(emoji_record)

            realmemoji = build_realm_emoji(
                realm_id=realm_id,
                name=alias,
                id=NEXT_ID("realmemoji"),
                file_name=emoji_filename,
            )
            zerver_realmemoji.append(realmemoji)

    create_converted_data_files(emoji_records, output_dir, "/emoji/records.json")
    logging.info("Done processing emoji")

    return zerver_realmemoji


def build_reactions(
    total_reactions: List[ZerverFieldsT],
    reactions: List[Dict[str, Any]],
    message_id: int,
    zerver_realmemoji: List[ZerverFieldsT],
) -> None:
    realmemoji: Dict[str, int] = {}
    for emoji in zerver_realmemoji:
        realmemoji[emoji["name"]] = emoji["id"]

    # For the Unicode emoji codes, we use equivalent of
    # function 'emoji_name_to_emoji_code' in 'zerver/lib/emoji' here
    for reaction_dict in reactions:
        emoji_name = reaction_dict["name"]
        user_id = reaction_dict["user_id"]
        # Check in realm emoji
        if emoji_name in realmemoji:
            emoji_code = realmemoji[emoji_name]
            reaction_type = Reaction.REALM_EMOJI
        # Check in Unicode emoji
        elif emoji_name in name_to_codepoint:
            emoji_code = name_to_codepoint[emoji_name]
            reaction_type = Reaction.UNICODE_EMOJI
        else:  # nocoverage
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
        reaction_dict["user_profile"] = user_id
        total_reactions.append(reaction_dict)


def process_message_attachment(
    upload: Dict[str, Any],
    realm_id: int,
    message_id: int,
    user_id: int,
    user_handler: UserHandler,
    zerver_attachment: List[ZerverFieldsT],
    uploads_list: List[ZerverFieldsT],
    upload_id_to_upload_data_map: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> Tuple[str, bool]:
    if upload["_id"] not in upload_id_to_upload_data_map:  # nocoverage
        logging.info("Skipping unknown attachment of message_id: %s", message_id)
        return "", False

    if "type" not in upload:  # nocoverage
        logging.info("Skipping attachment without type of message_id: %s", message_id)
        return "", False

    upload_file_data = upload_id_to_upload_data_map[upload["_id"]]
    file_name = upload["name"]
    file_ext = f".{upload['type'].split('/')[-1]}"

    has_image = False
    if file_ext.lower() in IMAGE_EXTENSIONS:
        has_image = True

    try:
        sanitized_name = sanitize_name(file_name)
    except AssertionError:  # nocoverage
        logging.info("Replacing invalid attachment name with random uuid: %s", file_name)
        sanitized_name = uuid.uuid4().hex

    if len(sanitized_name.encode("utf-8")) >= 255:  # nocoverage
        logging.info("Replacing too long attachment name with random uuid: %s", file_name)
        sanitized_name = uuid.uuid4().hex

    s3_path = "/".join(
        [
            str(realm_id),
            format(random.randint(0, 255), "x"),
            secrets.token_urlsafe(18),
            sanitized_name,
        ]
    )

    # Build the attachment from chunks and save it to s3_path.
    file_out_path = os.path.join(output_dir, "uploads", s3_path)
    os.makedirs(os.path.dirname(file_out_path), exist_ok=True)
    with open(file_out_path, "wb") as upload_file:
        upload_file.write(b"".join(upload_file_data["chunk"]))

    attachment_content = (
        f"{upload_file_data.get('description', '')}\n\n[{file_name}](/user_uploads/{s3_path})"
    )

    fileinfo = {
        "name": file_name,
        "size": upload_file_data["size"],
        "created": float(upload_file_data["_updatedAt"].timestamp()),
    }

    upload = dict(
        path=s3_path,
        realm_id=realm_id,
        content_type=upload["type"],
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

    return attachment_content, has_image


def process_raw_message_batch(
    realm_id: int,
    raw_messages: List[Dict[str, Any]],
    subscriber_map: Dict[int, Set[int]],
    user_handler: UserHandler,
    is_pm_data: bool,
    output_dir: str,
    zerver_realmemoji: List[ZerverFieldsT],
    total_reactions: List[ZerverFieldsT],
    uploads_list: List[ZerverFieldsT],
    zerver_attachment: List[ZerverFieldsT],
    upload_id_to_upload_data_map: Dict[str, Dict[str, Any]],
) -> None:
    def fix_mentions(
        content: str, mention_user_ids: Set[int], rc_channel_mention_data: List[Dict[str, str]]
    ) -> str:
        # Fix user mentions
        for user_id in mention_user_ids:
            user = user_handler.get_user(user_id=user_id)
            rc_mention =