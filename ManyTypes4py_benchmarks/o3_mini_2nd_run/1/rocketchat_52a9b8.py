#!/usr/bin/env python3
import logging
import os
import random
import secrets
import uuid
from typing import Any, Dict, List, Set, Tuple, Optional
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
) -> Dict[str, Any]:
    created_at: float = float(rc_instance['_createdAt'].timestamp())
    zerver_realm: Dict[str, Any] = build_zerver_realm(realm_id, realm_subdomain, created_at, 'Rocket.Chat')
    realm: Dict[str, Any] = build_realm(zerver_realm, realm_id, domain_name)
    realm['zerver_defaultstream'] = []
    return realm


def process_users(
    user_id_to_user_map: Dict[Any, Dict[str, Any]],
    realm_id: int,
    domain_name: str,
    user_handler: UserHandler,
    user_id_mapper: IdMapper[str],
) -> None:
    realm_owners: List[int] = []
    bots: List[int] = []
    for rc_user_id, user_dict in user_id_to_user_map.items():
        is_mirror_dummy: bool = False
        is_bot: bool = False
        is_active: bool = True
        if user_dict['type'] != 'user':
            is_active = False
            if user_dict['type'] == 'bot':
                is_bot = True
            else:
                is_mirror_dummy = True
        if user_dict.get('emails') is None:
            user_dict['emails'] = [{'address': '{}-{}@{}'.format(user_dict['username'], user_dict['type'], domain_name)}]
        avatar_source: str = 'G'
        full_name: str = user_dict['name']
        id = user_id_mapper.get(rc_user_id)
        delivery_email: str = user_dict['emails'][0]['address']
        email: str = user_dict['emails'][0]['address']
        short_name: str = user_dict['username']
        date_joined: float = float(user_dict['createdAt'].timestamp())
        timezone: str = 'UTC'
        role: int = UserProfile.ROLE_MEMBER
        if 'admin' in user_dict['roles']:
            role = UserProfile.ROLE_REALM_OWNER
            realm_owners.append(id)
        elif 'guest' in user_dict['roles']:
            role = UserProfile.ROLE_GUEST
        elif 'bot' in user_dict['roles']:
            is_bot = True
        if is_bot:
            bots.append(id)
        user: Dict[str, Any] = build_user_profile(
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
    if realm_owners:
        for bot_id in bots:
            bot_user: Dict[str, Any] = user_handler.get_user(user_id=bot_id)
            bot_user['bot_owner'] = realm_owners[0]


def truncate_name(name: str, name_id: Any, max_length: int = 60) -> str:
    if len(name) > max_length:
        name_id_suffix: str = f' [{name_id}]'
        name = name[0 : max_length - len(name_id_suffix)] + name_id_suffix
    return name


def get_stream_name(rc_channel: Dict[str, Any]) -> str:
    if rc_channel.get('teamMain'):
        stream_name: str = f"[TEAM] {rc_channel['name']}"
    else:
        stream_name = rc_channel['name']
    stream_name = truncate_name(stream_name, rc_channel['_id'])
    return stream_name


def convert_channel_data(
    room_id_to_room_map: Dict[Any, Dict[str, Any]],
    team_id_to_team_map: Dict[Any, Dict[str, Any]],
    stream_id_mapper: IdMapper[str],
    realm_id: int,
) -> List[Dict[str, Any]]:
    streams: List[Dict[str, Any]] = []
    for rc_room_id, channel_dict in room_id_to_room_map.items():
        date_created: float = float(channel_dict['ts'].timestamp())
        stream_id = stream_id_mapper.get(rc_room_id)
        invite_only: bool = channel_dict['t'] == 'p'
        stream_name: str = get_stream_name(channel_dict)
        stream_desc: str = channel_dict.get('description', '')
        if channel_dict.get('teamId') and (not channel_dict.get('teamMain')):
            stream_desc = '[Team {} channel]. {}'.format(team_id_to_team_map[channel_dict['teamId']]['name'], stream_desc)
        stream_post_policy: int = 4 if channel_dict.get('ro', False) else 1
        stream: Dict[str, Any] = build_stream(
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
    user_id_to_user_map: Dict[Any, Dict[str, Any]],
    dsc_id_to_dsc_map: Dict[Any, Dict[str, Any]],
    zerver_stream: List[Dict[str, Any]],
    stream_id_mapper: IdMapper[str],
    user_id_mapper: IdMapper[str],
    subscriber_handler: SubscriberHandler,
) -> None:
    stream_members_map: Dict[Any, Set[int]] = {}
    for rc_user_id, user_dict in user_id_to_user_map.items():
        if not user_dict.get('__rooms'):
            continue
        for channel in user_dict['__rooms']:
            if channel in dsc_id_to_dsc_map:
                continue
            stream_id = stream_id_mapper.get(channel)
            if stream_id not in stream_members_map:
                stream_members_map[stream_id] = set()
            stream_members_map[stream_id].add(user_id_mapper.get(rc_user_id))
    for stream in zerver_stream:
        if stream['id'] in stream_members_map:
            users: Set[int] = stream_members_map[stream['id']]
        else:
            users = set()
            stream['deactivated'] = True
        subscriber_handler.set_info(users=users, stream_id=stream['id'])


def convert_direct_message_group_data(
    direct_message_group_id_to_direct_message_group_map: Dict[str, Dict[str, Any]],
    direct_message_group_id_mapper: IdMapper[str],
    user_id_mapper: IdMapper[str],
    subscriber_handler: SubscriberHandler,
) -> List[Dict[str, Any]]:
    zerver_direct_message_group: List[Dict[str, Any]] = []
    for rc_direct_message_group_id, direct_message_group_dict in direct_message_group_id_to_direct_message_group_map.items():
        direct_message_group_id = direct_message_group_id_mapper.get(rc_direct_message_group_id)
        direct_message_group: Dict[str, Any] = build_direct_message_group(direct_message_group_id, len(direct_message_group_dict['uids']))
        zerver_direct_message_group.append(direct_message_group)
        direct_message_group_user_ids: Set[int] = {user_id_mapper.get(rc_user_id) for rc_user_id in direct_message_group_dict['uids']}
        subscriber_handler.set_info(users=direct_message_group_user_ids, direct_message_group_id=direct_message_group_id)
    return zerver_direct_message_group


def build_custom_emoji(
    realm_id: int, custom_emoji_data: Dict[str, Any], output_dir: str
) -> List[Dict[str, Any]]:
    logging.info('Starting to process custom emoji')
    emoji_folder: str = os.path.join(output_dir, 'emoji')
    os.makedirs(emoji_folder, exist_ok=True)
    zerver_realmemoji: List[Dict[str, Any]] = []
    emoji_records: List[Dict[str, Any]] = []
    emoji_file_data: Dict[str, Dict[str, Any]] = {}
    for emoji_file in custom_emoji_data['file']:
        emoji_file_data[str(emoji_file['_id'])] = {'filename': emoji_file['filename'], 'chunks': []}
    for emoji_chunk in custom_emoji_data['chunk']:
        emoji_file_data[emoji_chunk['files_id']]['chunks'].append(emoji_chunk['data'])
    for rc_emoji in custom_emoji_data['emoji']:
        emoji_file_id: str = f"{rc_emoji['name']}.{rc_emoji['extension']}"
        emoji_file_info: Dict[str, Any] = emoji_file_data[emoji_file_id]
        emoji_filename: str = emoji_file_info['filename']
        emoji_data: bytes = b''.join(emoji_file_info['chunks'])
        target_sub_path: str = RealmEmoji.PATH_ID_TEMPLATE.format(realm_id=realm_id, emoji_file_name=emoji_filename)
        target_path: str = os.path.join(emoji_folder, target_sub_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'wb') as e_file:
            e_file.write(emoji_data)
        emoji_aliases: List[str] = [rc_emoji['name']]
        emoji_aliases.extend(rc_emoji['aliases'])
        for alias in emoji_aliases:
            emoji_record: Dict[str, Any] = dict(
                path=target_path,
                s3_path=target_path,
                file_name=emoji_filename,
                realm_id=realm_id,
                name=alias,
            )
            emoji_records.append(emoji_record)
            realmemoji_item: Dict[str, Any] = build_realm_emoji(
                realm_id=realm_id, name=alias, id=NEXT_ID('realmemoji'), file_name=emoji_filename
            )
            zerver_realmemoji.append(realmemoji_item)
    create_converted_data_files(emoji_records, output_dir, '/emoji/records.json')
    logging.info('Done processing emoji')
    return zerver_realmemoji


def build_reactions(
    total_reactions: List[Dict[str, Any]],
    reactions: List[Dict[str, Any]],
    message_id: int,
    zerver_realmemoji: List[Dict[str, Any]],
) -> None:
    realmemoji: Dict[str, Any] = {}
    for emoji in zerver_realmemoji:
        realmemoji[emoji['name']] = emoji['id']
    for reaction_dict in reactions:
        emoji_name: str = reaction_dict['name']
        user_id = reaction_dict['user_id']
        if emoji_name in realmemoji:
            emoji_code = realmemoji[emoji_name]
            reaction_type = Reaction.REALM_EMOJI
        elif emoji_name in name_to_codepoint:
            emoji_code = name_to_codepoint[emoji_name]
            reaction_type = Reaction.UNICODE_EMOJI
        else:
            continue
        reaction_id = NEXT_ID('reaction')
        reaction = Reaction(id=reaction_id, emoji_code=emoji_code, emoji_name=emoji_name, reaction_type=reaction_type)
        reaction_dict_model: Dict[str, Any] = model_to_dict(reaction, exclude=['message', 'user_profile'])
        reaction_dict_model['message'] = message_id
        reaction_dict_model['user_profile'] = user_id
        total_reactions.append(reaction_dict_model)


def process_message_attachment(
    upload: Dict[str, Any],
    realm_id: int,
    message_id: int,
    user_id: int,
    user_handler: UserHandler,
    zerver_attachment: List[Dict[str, Any]],
    uploads_list: List[Dict[str, Any]],
    upload_id_to_upload_data_map: Dict[Any, Dict[str, Any]],
    output_dir: str,
) -> Tuple[str, bool]:
    if upload['_id'] not in upload_id_to_upload_data_map:
        logging.info('Skipping unknown attachment of message_id: %s', message_id)
        return ("", False)
    if 'type' not in upload:
        logging.info('Skipping attachment without type of message_id: %s', message_id)
        return ("", False)
    upload_file_data: Dict[str, Any] = upload_id_to_upload_data_map[upload['_id']]
    file_name: str = upload['name']
    file_ext: str = f".{upload['type'].split('/')[-1]}"
    has_image: bool = False
    if file_ext.lower() in IMAGE_EXTENSIONS:
        has_image = True
    try:
        sanitized_name: str = sanitize_name(file_name)
    except AssertionError:
        logging.info('Replacing invalid attachment name with random uuid: %s', file_name)
        sanitized_name = uuid.uuid4().hex
    if len(sanitized_name.encode('utf-8')) >= 255:
        logging.info('Replacing too long attachment name with random uuid: %s', file_name)
        sanitized_name = uuid.uuid4().hex
    s3_path: str = '/'.join([str(realm_id), format(random.randint(0, 255), 'x'), secrets.token_urlsafe(18), sanitized_name])
    file_out_path: str = os.path.join(output_dir, 'uploads', s3_path)
    os.makedirs(os.path.dirname(file_out_path), exist_ok=True)
    with open(file_out_path, 'wb') as upload_file:
        upload_file.write(b''.join(upload_file_data['chunk']))
    attachment_content: str = f"{upload_file_data.get('description', '')}\n\n[{file_name}](/user_uploads/{s3_path})"
    fileinfo: Dict[str, Any] = {
        'name': file_name,
        'size': upload_file_data['size'],
        'created': float(upload_file_data['_updatedAt'].timestamp()),
    }
    upload_updated: Dict[str, Any] = dict(
        path=s3_path,
        realm_id=realm_id,
        content_type=upload['type'],
        user_profile_id=user_id,
        last_modified=fileinfo['created'],
        user_profile_email=user_handler.get_user(user_id=user_id)['email'],
        s3_path=s3_path,
        size=fileinfo['size'],
    )
    uploads_list.append(upload_updated)
    build_attachment(
        realm_id=realm_id,
        message_ids={message_id},
        user_id=user_id,
        fileinfo=fileinfo,
        s3_path=s3_path,
        zerver_attachment=zerver_attachment,
    )
    return (attachment_content, has_image)


def process_raw_message_batch(
    realm_id: int,
    raw_messages: List[Dict[str, Any]],
    subscriber_map: Dict[Any, Any],
    user_handler: UserHandler,
    is_pm_data: bool,
    output_dir: str,
    zerver_realmemoji: List[Dict[str, Any]],
    total_reactions: List[Dict[str, Any]],
    uploads_list: List[Dict[str, Any]],
    zerver_attachment: List[Dict[str, Any]],
    upload_id_to_upload_data_map: Dict[Any, Dict[str, Any]],
) -> None:
    def fix_mentions(
        content: str, mention_user_ids: Set[int], rc_channel_mention_data: List[Dict[str, Any]]
    ) -> str:
        for user_id in mention_user_ids:
            user: Dict[str, Any] = user_handler.get_user(user_id=user_id)
            rc_mention: str = '@{short_name}'.format(**user)
            zulip_mention: str = '@**{full_name}**'.format(**user)
            content = content.replace(rc_mention, zulip_mention)
        content = content.replace('@all', '@**all**')
        content = content.replace('@here', '@**all**')
        for mention_data in rc_channel_mention_data:
            rc_mention = mention_data['rc_mention']
            zulip_mention = mention_data['zulip_mention']
            content = content.replace(rc_mention, zulip_mention)
        return content

    user_mention_map: Dict[int, Set[int]] = {}
    wildcard_mention_map: Dict[int, bool] = {}
    zerver_message: List[Dict[str, Any]] = []
    for raw_message in raw_messages:
        message_id: int = NEXT_ID('message')
        mention_user_ids: Set[int] = raw_message['mention_user_ids']
        user_mention_map[message_id] = mention_user_ids
        wildcard_mention_map[message_id] = raw_message['wildcard_mention']
        content: str = fix_mentions(
            content=raw_message['content'],
            mention_user_ids=mention_user_ids,
            rc_channel_mention_data=raw_message['rc_channel_mention_data'],
        )
        date_sent = raw_message['date_sent']
        sender_user_id = raw_message['sender_id']
        recipient_id = raw_message['recipient_id']
        rendered_content = None
        has_attachment: bool = False
        has_image: bool = False
        has_link: bool = raw_message['has_link']
        if 'file' in raw_message:
            has_attachment = True
            has_link = True
            attachment_content, has_image = process_message_attachment(
                upload=raw_message['file'],
                realm_id=realm_id,
                message_id=message_id,
                user_id=sender_user_id,
                user_handler=user_handler,
                uploads_list=uploads_list,
                zerver_attachment=zerver_attachment,
                upload_id_to_upload_data_map=upload_id_to_upload_data_map,
                output_dir=output_dir,
            )
            content += attachment_content
        topic_name: str = raw_message['topic_name']
        message: Dict[str, Any] = build_message(
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
            total_reactions=total_reactions,
            reactions=raw_message['reactions'],
            message_id=message_id,
            zerver_realmemoji=zerver_realmemoji,
        )
    zerver_usermessage: List[Dict[str, Any]] = make_user_messages(
        zerver_message=zerver_message,
        subscriber_map=subscriber_map,
        is_pm_data=is_pm_data,
        mention_map=user_mention_map,
        wildcard_mention_map=wildcard_mention_map,
    )
    message_json: Dict[str, Any] = dict(zerver_message=zerver_message, zerver_usermessage=zerver_usermessage)
    dump_file_id: int = NEXT_ID('dump_file_id' + str(realm_id))
    message_file: str = f'/messages-{dump_file_id:06}.json'
    create_converted_data_files(message_json, output_dir, message_file)


def get_topic_name(
    message: Dict[str, Any], dsc_id_to_dsc_map: Dict[Any, Dict[str, Any]], thread_id_mapper: IdMapper[str], is_pm_data: bool = False
) -> str:
    if is_pm_data:
        return ''
    elif message['rid'] in dsc_id_to_dsc_map:
        dsc_channel_name: str = dsc_id_to_dsc_map[message['rid']]['fname']
        return truncate_name(f'{dsc_channel_name} (Imported from Rocket.Chat)', message['rid'])
    elif message.get('replies'):
        thread_id = thread_id_mapper.get(message['_id'])
        return truncate_name(f'Thread {thread_id} (Imported from Rocket.Chat)', message['_id'])
    elif message.get('tmid'):
        thread_id = thread_id_mapper.get(message['tmid'])
        return truncate_name(f'Thread {thread_id} (Imported from Rocket.Chat)', message['tmid'])
    else:
        return 'Imported from Rocket.Chat'


def process_messages(
    realm_id: int,
    messages: List[Dict[str, Any]],
    subscriber_map: Dict[Any, Any],
    is_pm_data: bool,
    username_to_user_id_map: Dict[str, Any],
    user_id_mapper: IdMapper[str],
    user_handler: UserHandler,
    user_id_to_recipient_id: Dict[Any, int],
    stream_id_mapper: IdMapper[str],
    stream_id_to_recipient_id: Dict[Any, int],
    direct_message_group_id_mapper: IdMapper[str],
    direct_message_group_id_to_recipient_id: Dict[Any, int],
    thread_id_mapper: IdMapper[str],
    room_id_to_room_map: Dict[Any, Dict[str, Any]],
    dsc_id_to_dsc_map: Dict[Any, Dict[str, Any]],
    direct_id_to_direct_map: Dict[Any, Dict[str, Any]],
    direct_message_group_id_to_direct_message_group_map: Dict[Any, Dict[str, Any]],
    zerver_realmemoji: List[Dict[str, Any]],
    total_reactions: List[Dict[str, Any]],
    uploads_list: List[Dict[str, Any]],
    zerver_attachment: List[Dict[str, Any]],
    upload_id_to_upload_data_map: Dict[Any, Dict[str, Any]],
    output_dir: str,
) -> None:
    def list_reactions(reactions: Dict[str, Any]) -> List[Dict[str, Any]]:
        reactions_list: List[Dict[str, Any]] = []
        for react_code in reactions:
            name: str = react_code.split(':')[1]
            usernames: List[str] = reactions[react_code]['usernames']
            for username in usernames:
                if username not in username_to_user_id_map:
                    continue
                rc_user_id = username_to_user_id_map[username]
                user_id_val = user_id_mapper.get(rc_user_id)
                reactions_list.append({'name': name, 'user_id': user_id_val})
        return reactions_list

    def message_to_dict(message: Dict[str, Any]) -> Dict[str, Any]:
        rc_sender_id = message['u']['_id']
        sender_id = user_id_mapper.get(rc_sender_id)
        if 'msg' in message:
            content = message['msg']
        else:
            content = 'This message imported from Rocket.Chat had no body in the data export.'
            logging.info('Message %s contains no message content: %s', message['_id'], message)
        if message.get('reactions'):
            reactions = list_reactions(message['reactions'])
        else:
            reactions = []
        message_dict: Dict[str, Any] = dict(
            sender_id=sender_id,
            content=content,
            date_sent=int(message['ts'].timestamp()),
            reactions=reactions,
            has_link=bool(message.get('urls')),
        )
        if is_pm_data:
            rc_channel_id = message['rid']
            if rc_channel_id in direct_message_group_id_to_direct_message_group_map:
                direct_message_group_id = direct_message_group_id_mapper.get(rc_channel_id)
                message_dict['recipient_id'] = direct_message_group_id_to_recipient_id[direct_message_group_id]
            else:
                rc_member_ids = direct_id_to_direct_map[rc_channel_id]['uids']
                if len(rc_member_ids) == 1:
                    rc_member_ids.append(rc_member_ids[0])
                if rc_sender_id == rc_member_ids[0]:
                    zulip_member_id = user_id_mapper.get(rc_member_ids[1])
                    message_dict['recipient_id'] = user_id_to_recipient_id[zulip_member_id]
                else:
                    zulip_member_id = user_id_mapper.get(rc_member_ids[0])
                    message_dict['recipient_id'] = user_id_to_recipient_id[zulip_member_id]
        elif message['rid'] in dsc_id_to_dsc_map:
            dsc_channel = dsc_id_to_dsc_map[message['rid']]
            parent_channel_id = dsc_channel['prid']
            stream_id = stream_id_mapper.get(parent_channel_id)
            message_dict['recipient_id'] = stream_id_to_recipient_id[stream_id]
        else:
            stream_id = stream_id_mapper.get(message['rid'])
            message_dict['recipient_id'] = stream_id_to_recipient_id[stream_id]
        message_dict['topic_name'] = get_topic_name(message, dsc_id_to_dsc_map, thread_id_mapper, is_pm_data)
        mention_user_ids: Set[int] = set()
        wildcard_mention: bool = False
        for mention in message.get('mentions', []):
            mention_id = mention['_id']
            if mention_id in ['all', 'here']:
                wildcard_mention = True
                continue
            if user_id_mapper.has(mention_id):
                user_id_val = user_id_mapper.get(mention_id)
                mention_user_ids.add(user_id_val)
            else:
                logging.info('Message %s contains mention of unknown user %s: %s', message['_id'], mention_id, mention)
        message_dict['mention_user_ids'] = mention_user_ids
        message_dict['wildcard_mention'] = wildcard_mention
        rc_channel_mention_data: List[Dict[str, Any]] = []
        for mention in message.get('channels', []):
            mention_rc_channel_id = mention['_id']
            mention_rc_channel_name = mention['name']
            rc_mention = f'#{mention_rc_channel_name}'
            if mention_rc_channel_id in room_id_to_room_map:
                rc_channel = room_id_to_room_map[mention_rc_channel_id]
                converted_stream_name = get_stream_name(rc_channel)
                zulip_mention = f'#**{converted_stream_name}**'
            elif mention_rc_channel_id in dsc_id_to_dsc_map:
                dsc_channel = dsc_id_to_dsc_map[mention_rc_channel_id]
                parent_channel_id = dsc_channel['prid']
                if parent_channel_id in direct_id_to_direct_map or parent_channel_id in direct_message_group_id_to_direct_message_group_map:
                    logging.info('skipping direct messages discussion mention: %s', dsc_channel['fname'])
                    continue
                converted_topic_name = get_topic_name(message={'rid': mention_rc_channel_id}, dsc_id_to_dsc_map=dsc_id_to_dsc_map, thread_id_mapper=thread_id_mapper)
                parent_rc_channel = room_id_to_room_map[parent_channel_id]
                parent_stream_name = get_stream_name(parent_rc_channel)
                zulip_mention = f'#**{parent_stream_name}>{converted_topic_name}**'
            else:
                logging.info("Failed to map mention '%s' to zulip syntax.", mention)
                continue
            mention_data: Dict[str, Any] = {'rc_mention': rc_mention, 'zulip_mention': zulip_mention}
            rc_channel_mention_data.append(mention_data)
        message_dict['rc_channel_mention_data'] = rc_channel_mention_data
        if message.get('file'):
            message_dict['file'] = message['file']
        return message_dict

    raw_messages: List[Dict[str, Any]] = []
    for message in messages:
        if message.get('t') is not None:
            continue
        raw_messages.append(message_to_dict(message))

    def process_batch(lst: List[Dict[str, Any]]) -> None:
        process_raw_message_batch(
            realm_id=realm_id,
            raw_messages=lst,
            subscriber_map=subscriber_map,
            user_handler=user_handler,
            is_pm_data=is_pm_data,
            output_dir=output_dir,
            zerver_realmemoji=zerver_realmemoji,
            total_reactions=total_reactions,
            uploads_list=uploads_list,
            zerver_attachment=zerver_attachment,
            upload_id_to_upload_data_map=upload_id_to_upload_data_map,
        )

    chunk_size: int = 1000
    process_list_in_batches(lst=raw_messages, chunk_size=chunk_size, process_batch=process_batch)


def map_upload_id_to_upload_data(upload_data: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    upload_id_to_upload_data_map: Dict[Any, Dict[str, Any]] = {}
    for upload in upload_data['upload']:
        upload_id_to_upload_data_map[upload['_id']] = {**upload, 'chunk': []}
    for chunk in upload_data['chunk']:
        if chunk['files_id'] not in upload_id_to_upload_data_map:
            logging.info('Skipping chunk %s without metadata', chunk['files_id'])
            continue
        upload_id_to_upload_data_map[chunk['files_id']]['chunk'].append(chunk['data'])
    return upload_id_to_upload_data_map


def separate_channel_private_and_livechat_messages(
    messages: List[Dict[str, Any]],
    dsc_id_to_dsc_map: Dict[Any, Dict[str, Any]],
    direct_id_to_direct_map: Dict[Any, Dict[str, Any]],
    direct_message_group_id_to_direct_message_group_map: Dict[Any, Dict[str, Any]],
    livechat_id_to_livechat_map: Dict[Any, Dict[str, Any]],
    channel_messages: List[Dict[str, Any]],
    private_messages: List[Dict[str, Any]],
    livechat_messages: List[Dict[str, Any]],
) -> None:
    private_channels_list = [*direct_id_to_direct_map, *direct_message_group_id_to_direct_message_group_map]
    for message in messages:
        if not message.get('rid'):
            continue
        if message['rid'] in dsc_id_to_dsc_map:
            parent_channel_id = dsc_id_to_dsc_map[message['rid']]['prid']
            if parent_channel_id in private_channels_list:
                message['rid'] = parent_channel_id
        if message['rid'] in private_channels_list:
            private_messages.append(message)
        elif message['rid'] in livechat_id_to_livechat_map:
            livechat_messages.append(message)
        else:
            channel_messages.append(message)


def map_receiver_id_to_recipient_id(
    zerver_recipient: List[Dict[str, Any]],
    stream_id_to_recipient_id: Dict[Any, int],
    direct_message_group_id_to_recipient_id: Dict[Any, int],
    user_id_to_recipient_id: Dict[Any, int],
) -> None:
    for recipient in zerver_recipient:
        if recipient['type'] == Recipient.STREAM:
            stream_id_to_recipient_id[recipient['type_id']] = recipient['id']
        elif recipient['type'] == Recipient.DIRECT_MESSAGE_GROUP:
            direct_message_group_id_to_recipient_id[recipient['type_id']] = recipient['id']
        elif recipient['type'] == Recipient.PERSONAL:
            user_id_to_recipient_id[recipient['type_id']] = recipient['id']


def categorize_channels_and_map_with_id(
    channel_data: List[Dict[str, Any]],
    room_id_to_room_map: Dict[Any, Dict[str, Any]],
    team_id_to_team_map: Dict[Any, Dict[str, Any]],
    dsc_id_to_dsc_map: Dict[Any, Dict[str, Any]],
    direct_id_to_direct_map: Dict[Any, Dict[str, Any]],
    direct_message_group_id_to_direct_message_group_map: Dict[Any, Dict[str, Any]],
    livechat_id_to_livechat_map: Dict[Any, Dict[str, Any]],
) -> None:
    direct_message_group_hashed_channels: Dict[frozenset, Dict[str, Any]] = {}
    for channel in channel_data:
        if channel.get('prid'):
            dsc_id_to_dsc_map[channel['_id']] = channel
        elif channel['t'] == 'd':
            if len(channel['uids']) > 2:
                direct_message_group_members = frozenset(channel['uids'])
                logging.info('Direct message group channel found. UIDs: %r', channel['uids'])
                if channel['msgs'] == 0:
                    logging.debug('Skipping direct message group with 0 messages: %s', channel)
                elif direct_message_group_members in direct_message_group_hashed_channels:
                    logging.info('Mapping direct message group %r to existing channel: %s', direct_message_group_members, direct_message_group_hashed_channels[direct_message_group_members])
                    direct_message_group_id_to_direct_message_group_map[channel['_id']] = direct_message_group_hashed_channels[direct_message_group_members]
                    raise NotImplementedError('Mapping multiple direct message groups with messages to one is not fully implemented yet')
                else:
                    direct_message_group_id_to_direct_message_group_map[channel['_id']] = channel
                    direct_message_group_hashed_channels[direct_message_group_members] = channel
            else:
                direct_id_to_direct_map[channel['_id']] = channel
        elif channel['t'] == 'l':
            livechat_id_to_livechat_map[channel['_id']] = channel
        else:
            room_id_to_room_map[channel['_id']] = channel
            if channel.get('teamMain'):
                team_id_to_team_map[channel['teamId']] = channel


def map_username_to_user_id(user_id_to_user_map: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    username_to_user_id_map: Dict[str, Any] = {}
    for user_id, user_dict in user_id_to_user_map.items():
        username_to_user_id_map[user_dict['username']] = user_id
    return username_to_user_id_map


def map_user_id_to_user(user_data_list: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    user_id_to_user_map: Dict[Any, Dict[str, Any]] = {}
    for user in user_data_list:
        user_id_to_user_map[user['_id']] = user
    return user_id_to_user_map


def rocketchat_data_to_dict(rocketchat_data_dir: str, sections: Optional[List[str]] = None) -> Dict[str, Any]:
    """Reads Rocket.Chat data from its BSON files for the requested sections.
    Defaults to fetching everything.
    """
    rocketchat_data: Dict[str, Any] = {}
    if sections is None or 'instance' in sections:
        rocketchat_data['instance'] = []
        with open(os.path.join(rocketchat_data_dir, 'instances.bson'), 'rb') as fcache:
            rocketchat_data['instance'] = bson.decode_all(fcache.read(), bson_codec_options)
    if sections is None or 'user' in sections:
        rocketchat_data['user'] = []
        with open(os.path.join(rocketchat_data_dir, 'users.bson'), 'rb') as fcache:
            rocketchat_data['user'] = bson.decode_all(fcache.read(), bson_codec_options)
    if sections is None or 'avatar' in sections:
        rocketchat_data['avatar'] = {'avatar': [], 'file': [], 'chunk': []}
        with open(os.path.join(rocketchat_data_dir, 'rocketchat_avatars.bson'), 'rb') as fcache:
            rocketchat_data['avatar']['avatar'] = bson.decode_all(fcache.read(), bson_codec_options)
        if rocketchat_data['avatar']['avatar']:
            with open(os.path.join(rocketchat_data_dir, 'rocketchat_avatars.files.bson'), 'rb') as fcache:
                rocketchat_data['avatar']['file'] = bson.decode_all(fcache.read(), bson_codec_options)
            with open(os.path.join(rocketchat_data_dir, 'rocketchat_avatars.chunks.bson'), 'rb') as fcache:
                rocketchat_data['avatar']['chunk'] = bson.decode_all(fcache.read(), bson_codec_options)
    if sections is None or 'room' in sections:
        rocketchat_data['room'] = []
        with open(os.path.join(rocketchat_data_dir, 'rocketchat_room.bson'), 'rb') as fcache:
            rocketchat_data['room'] = bson.decode_all(fcache.read(), bson_codec_options)
    if sections is None or 'message' in sections:
        rocketchat_data['message'] = []
        with open(os.path.join(rocketchat_data_dir, 'rocketchat_message.bson'), 'rb') as fcache:
            rocketchat_data['message'] = bson.decode_all(fcache.read(), bson_codec_options)
    if sections is None or 'custom_emoji' in sections:
        rocketchat_data['custom_emoji'] = {'emoji': [], 'file': [], 'chunk': []}
        with open(os.path.join(rocketchat_data_dir, 'rocketchat_custom_emoji.bson'), 'rb') as fcache:
            rocketchat_data['custom_emoji']['emoji'] = bson.decode_all(fcache.read(), bson_codec_options)
        if rocketchat_data['custom_emoji']['emoji']:
            with open(os.path.join(rocketchat_data_dir, 'custom_emoji.files.bson'), 'rb') as fcache:
                rocketchat_data['custom_emoji']['file'] = bson.decode_all(fcache.read(), bson_codec_options)
            with open(os.path.join(rocketchat_data_dir, 'custom_emoji.chunks.bson'), 'rb') as fcache:
                rocketchat_data['custom_emoji']['chunk'] = bson.decode_all(fcache.read(), bson_codec_options)
    if sections is None or 'upload' in sections:
        rocketchat_data['upload'] = {'upload': [], 'file': [], 'chunk': []}
        with open(os.path.join(rocketchat_data_dir, 'rocketchat_uploads.bson'), 'rb') as fcache:
            rocketchat_data['upload']['upload'] = bson.decode_all(fcache.read(), bson_codec_options)
        if rocketchat_data['upload']['upload']:
            with open(os.path.join(rocketchat_data_dir, 'rocketchat_uploads.files.bson'), 'rb') as fcache:
                rocketchat_data['upload']['file'] = bson.decode_all(fcache.read(), bson_codec_options)
            with open(os.path.join(rocketchat_data_dir, 'rocketchat_uploads.chunks.bson'), 'rb') as fcache:
                rocketchat_data['upload']['chunk'] = bson.decode_all(fcache.read(), bson_codec_options)
    return rocketchat_data


def do_convert_data(rocketchat_data_dir: str, output_dir: str) -> None:
    realm_subdomain: str = ''
    realm_id: int = 0
    domain_name: str = settings.EXTERNAL_HOST
    rocketchat_instance_data: Dict[str, Any] = rocketchat_data_to_dict(rocketchat_data_dir, ['instance'])['instance'][0]
    realm: Dict[str, Any] = make_realm(realm_id, realm_subdomain, domain_name, rocketchat_instance_data)
    rocketchat_user_data: List[Dict[str, Any]] = rocketchat_data_to_dict(rocketchat_data_dir, ['user'])['user']
    user_id_to_user_map: Dict[Any, Dict[str, Any]] = map_user_id_to_user(rocketchat_user_data)
    username_to_user_id_map: Dict[str, Any] = map_username_to_user_id(user_id_to_user_map)
    user_handler: UserHandler = UserHandler()
    subscriber_handler: SubscriberHandler = SubscriberHandler()
    user_id_mapper: IdMapper[str] = IdMapper[str]()
    stream_id_mapper: IdMapper[str] = IdMapper[str]()
    direct_message_group_id_mapper: IdMapper[str] = IdMapper[str]()
    thread_id_mapper: IdMapper[str] = IdMapper[str]()
    process_users(user_id_to_user_map=user_id_to_user_map, realm_id=realm_id, domain_name=domain_name, user_handler=user_handler, user_id_mapper=user_id_mapper)
    rocketchat_emoji_data: Dict[str, Any] = rocketchat_data_to_dict(rocketchat_data_dir, ['custom_emoji'])['custom_emoji']
    zerver_realmemoji: List[Dict[str, Any]] = build_custom_emoji(realm_id=realm_id, custom_emoji_data=rocketchat_emoji_data, output_dir=output_dir)
    realm['zerver_realmemoji'] = zerver_realmemoji
    room_id_to_room_map: Dict[Any, Dict[str, Any]] = {}
    team_id_to_team_map: Dict[Any, Dict[str, Any]] = {}
    dsc_id_to_dsc_map: Dict[Any, Dict[str, Any]] = {}
    direct_id_to_direct_map: Dict[Any, Dict[str, Any]] = {}
    direct_message_group_id_to_direct_message_group_map: Dict[Any, Dict[str, Any]] = {}
    livechat_id_to_livechat_map: Dict[Any, Dict[str, Any]] = {}
    rocketchat_room_data: List[Dict[str, Any]] = rocketchat_data_to_dict(rocketchat_data_dir, ['room'])['room']
    categorize_channels_and_map_with_id(
        channel_data=rocketchat_room_data,
        room_id_to_room_map=room_id_to_room_map,
        team_id_to_team_map=team_id_to_team_map,
        dsc_id_to_dsc_map=dsc_id_to_dsc_map,
        direct_id_to_direct_map=direct_id_to_direct_map,
        direct_message_group_id_to_direct_message_group_map=direct_message_group_id_to_direct_message_group_map,
        livechat_id_to_livechat_map=livechat_id_to_livechat_map,
    )
    zerver_stream: List[Dict[str, Any]] = convert_channel_data(
        room_id_to_room_map=room_id_to_room_map, team_id_to_team_map=team_id_to_team_map, stream_id_mapper=stream_id_mapper, realm_id=realm_id
    )
    realm['zerver_stream'] = zerver_stream
    convert_stream_subscription_data(
        user_id_to_user_map=user_id_to_user_map,
        dsc_id_to_dsc_map=dsc_id_to_dsc_map,
        zerver_stream=zerver_stream,
        stream_id_mapper=stream_id_mapper,
        user_id_mapper=user_id_mapper,
        subscriber_handler=subscriber_handler,
    )
    zerver_direct_message_group: List[Dict[str, Any]] = convert_direct_message_group_data(
        direct_message_group_id_to_direct_message_group_map=direct_message_group_id_to_direct_message_group_map,
        direct_message_group_id_mapper=direct_message_group_id_mapper,
        user_id_mapper=user_id_mapper,
        subscriber_handler=subscriber_handler,
    )
    realm['zerver_huddle'] = zerver_direct_message_group
    all_users: List[Dict[str, Any]] = user_handler.get_all_users()
    zerver_recipient: List[Dict[str, Any]] = build_recipients(
        zerver_userprofile=all_users,
        zerver_stream=zerver_stream,
        zerver_direct_message_group=zerver_direct_message_group,
    )
    realm['zerver_recipient'] = zerver_recipient
    stream_subscriptions: List[Dict[str, Any]] = build_stream_subscriptions(
        get_users=subscriber_handler.get_users,
        zerver_recipient=zerver_recipient,
        zerver_stream=zerver_stream,
    )
    direct_message_group_subscriptions: List[Dict[str, Any]] = build_direct_message_group_subscriptions(
        get_users=subscriber_handler.get_users,
        zerver_recipient=zerver_recipient,
        zerver_direct_message_group=zerver_direct_message_group,
    )
    personal_subscriptions: List[Dict[str, Any]] = build_personal_subscriptions(zerver_recipient=zerver_recipient)
    zerver_subscription: List[Dict[str, Any]] = personal_subscriptions + stream_subscriptions + direct_message_group_subscriptions
    realm['zerver_subscription'] = zerver_subscription
    subscriber_map = make_subscriber_map(zerver_subscription=zerver_subscription)
    stream_id_to_recipient_id: Dict[Any, int] = {}
    direct_message_group_id_to_recipient_id: Dict[Any, int] = {}
    user_id_to_recipient_id: Dict[Any, int] = {}
    map_receiver_id_to_recipient_id(
        zerver_recipient=zerver_recipient,
        stream_id_to_recipient_id=stream_id_to_recipient_id,
        direct_message_group_id_to_recipient_id=direct_message_group_id_to_recipient_id,
        user_id_to_recipient_id=user_id_to_recipient_id,
    )
    channel_messages: List[Dict[str, Any]] = []
    private_messages: List[Dict[str, Any]] = []
    livechat_messages: List[Dict[str, Any]] = []
    rocketchat_message_data: List[Dict[str, Any]] = rocketchat_data_to_dict(rocketchat_data_dir, ['message'])['message']
    separate_channel_private_and_livechat_messages(
        messages=rocketchat_message_data,
        dsc_id_to_dsc_map=dsc_id_to_dsc_map,
        direct_id_to_direct_map=direct_id_to_direct_map,
        direct_message_group_id_to_direct_message_group_map=direct_message_group_id_to_direct_message_group_map,
        livechat_id_to_livechat_map=livechat_id_to_livechat_map,
        channel_messages=channel_messages,
        private_messages=private_messages,
        livechat_messages=livechat_messages,
    )
    rocketchat_message_data = []
    total_reactions: List[Dict[str, Any]] = []
    uploads_list: List[Dict[str, Any]] = []
    zerver_attachment: List[Dict[str, Any]] = []
    rocketchat_upload_data: Dict[str, Any] = rocketchat_data_to_dict(rocketchat_data_dir, ['upload'])['upload']
    upload_id_to_upload_data_map: Dict[Any, Dict[str, Any]] = map_upload_id_to_upload_data(rocketchat_upload_data)
    process_messages(
        realm_id=realm_id,
        messages=channel_messages,
        subscriber_map=subscriber_map,
        is_pm_data=False,
        username_to_user_id_map=username_to_user_id_map,
        user_id_mapper=user_id_mapper,
        user_handler=user_handler,
        user_id_to_recipient_id=user_id_to_recipient_id,
        stream_id_mapper=stream_id_mapper,
        stream_id_to_recipient_id=stream_id_to_recipient_id,
        direct_message_group_id_mapper=direct_message_group_id_mapper,
        direct_message_group_id_to_recipient_id=direct_message_group_id_to_recipient_id,
        thread_id_mapper=thread_id_mapper,
        room_id_to_room_map=room_id_to_room_map,
        dsc_id_to_dsc_map=dsc_id_to_dsc_map,
        direct_id_to_direct_map=direct_id_to_direct_map,
        direct_message_group_id_to_direct_message_group_map=direct_message_group_id_to_direct_message_group_map,
        zerver_realmemoji=zerver_realmemoji,
        total_reactions=total_reactions,
        uploads_list=uploads_list,
        zerver_attachment=zerver_attachment,
        upload_id_to_upload_data_map=upload_id_to_upload_data_map,
        output_dir=output_dir,
    )
    process_messages(
        realm_id=realm_id,
        messages=private_messages,
        subscriber_map=subscriber_map,
        is_pm_data=True,
        username_to_user_id_map=username_to_user_id_map,
        user_id_mapper=user_id_mapper,
        user_handler=user_handler,
        user_id_to_recipient_id=user_id_to_recipient_id,
        stream_id_mapper=stream_id_mapper,
        stream_id_to_recipient_id=stream_id_to_recipient_id,
        direct_message_group_id_mapper=direct_message_group_id_mapper,
        direct_message_group_id_to_recipient_id=direct_message_group_id_to_recipient_id,
        thread_id_mapper=thread_id_mapper,
        room_id_to_room_map=room_id_to_room_map,
        dsc_id_to_dsc_map=dsc_id_to_dsc_map,
        direct_id_to_direct_map=direct_id_to_direct_map,
        direct_message_group_id_to_direct_message_group_map=direct_message_group_id_to_direct_message_group_map,
        zerver_realmemoji=zerver_realmemoji,
        total_reactions=total_reactions,
        uploads_list=uploads_list,
        zerver_attachment=zerver_attachment,
        upload_id_to_upload_data_map=upload_id_to_upload_data_map,
        output_dir=output_dir,
    )
    realm['zerver_reaction'] = total_reactions
    realm['zerver_userprofile'] = user_handler.get_all_users()
    realm['sort_by_date'] = True
    create_converted_data_files(realm, output_dir, '/realm.json')
    create_converted_data_files([], output_dir, '/avatars/records.json')
    attachment = {'zerver_attachment': zerver_attachment}
    create_converted_data_files(attachment, output_dir, '/attachment.json')
    create_converted_data_files(uploads_list, output_dir, '/uploads/records.json')
    do_common_export_processes(output_dir)