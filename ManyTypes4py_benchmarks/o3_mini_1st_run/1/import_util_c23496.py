#!/usr/bin/env python3
from __future__ import annotations
import logging
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional, Protocol, TypeAlias, TypeVar, Union
import orjson
import requests
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.forms.models import model_to_dict
from django.utils.timezone import now as timezone_now
from zerver.data_import.sequencer import NEXT_ID
from zerver.lib.avatar_hash import user_avatar_base_path_from_ids
from zerver.lib.message import normalize_body_for_import
from zerver.lib.mime_types import INLINE_MIME_TYPES, guess_extension
from zerver.lib.partial import partial
from zerver.lib.stream_color import STREAM_ASSIGNMENT_COLORS as STREAM_COLORS
from zerver.lib.thumbnail import THUMBNAIL_ACCEPT_IMAGE_TYPES, BadImageError
from zerver.models import Attachment, DirectMessageGroup, Message, Realm, RealmEmoji, Recipient, Stream, Subscription, UserProfile
from zproject.backends import all_default_backend_names

ZerverFieldsT: TypeAlias = dict[str, Any]

class SubscriberHandler:
    def __init__(self) -> None:
        self.stream_info: dict[int, Iterable[int]] = {}
        self.direct_message_group_info: dict[int, Iterable[int]] = {}

    def set_info(
        self, 
        users: Iterable[int], 
        stream_id: Optional[int] = None, 
        direct_message_group_id: Optional[int] = None
    ) -> None:
        if stream_id is not None:
            self.stream_info[stream_id] = users
        elif direct_message_group_id is not None:
            self.direct_message_group_info[direct_message_group_id] = users
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

    def get_users(
        self, 
        stream_id: Optional[int] = None, 
        direct_message_group_id: Optional[int] = None
    ) -> Iterable[int]:
        if stream_id is not None:
            return self.stream_info[stream_id]
        elif direct_message_group_id is not None:
            return self.direct_message_group_info[direct_message_group_id]
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

def build_zerver_realm(realm_id: int, realm_subdomain: str, time: Any, other_product: str) -> list[ZerverFieldsT]:
    realm = Realm(id=realm_id, name=realm_subdomain, string_id=realm_subdomain, description=f'Organization imported from {other_product}!')
    realm_dict: ZerverFieldsT = model_to_dict(realm)
    realm_dict['date_created'] = time
    del realm_dict['uuid']
    del realm_dict['uuid_owner_secret']
    return [realm_dict]

def build_user_profile(
    avatar_source: str, 
    date_joined: Any, 
    delivery_email: str, 
    email: str, 
    full_name: str, 
    id: int, 
    is_active: bool, 
    role: int, 
    is_mirror_dummy: bool, 
    realm_id: int, 
    short_name: str, 
    timezone: str, 
    is_bot: bool = False, 
    bot_type: Optional[Any] = None
) -> ZerverFieldsT:
    obj = UserProfile(
        avatar_source=avatar_source, 
        date_joined=date_joined, 
        delivery_email=delivery_email, 
        email=email, 
        full_name=full_name, 
        id=id, 
        is_mirror_dummy=is_mirror_dummy, 
        is_active=is_active, 
        role=role, 
        realm_id=realm_id, 
        timezone=timezone, 
        is_bot=is_bot, 
        bot_type=bot_type
    )
    dct: ZerverFieldsT = model_to_dict(obj)
    "\n    Even though short_name is no longer in the Zulip\n    UserProfile, it's helpful to have it in our import\n    dictionaries for legacy reasons.\n    "
    dct['short_name'] = short_name
    return dct

def build_avatar(
    zulip_user_id: int, 
    realm_id: int, 
    email: str, 
    avatar_url: str, 
    timestamp: Any, 
    avatar_list: list[dict[str, Any]]
) -> None:
    avatar = dict(
        path=avatar_url, 
        realm_id=realm_id, 
        content_type=None, 
        avatar_version=1, 
        user_profile_id=zulip_user_id, 
        last_modified=timestamp, 
        user_profile_email=email, 
        s3_path='', 
        size=''
    )
    avatar_list.append(avatar)

def make_subscriber_map(zerver_subscription: Iterable[Mapping[str, Any]]) -> dict[int, set[int]]:
    subscriber_map: dict[int, set[int]] = {}
    for sub in zerver_subscription:
        user_id: int = sub['user_profile']
        recipient_id: int = sub['recipient']
        if recipient_id not in subscriber_map:
            subscriber_map[recipient_id] = set()
        subscriber_map[recipient_id].add(user_id)
    return subscriber_map

def make_user_messages(
    zerver_message: Iterable[Mapping[str, Any]], 
    subscriber_map: Mapping[int, set[int]], 
    is_pm_data: bool, 
    mention_map: Mapping[int, set[int]], 
    wildcard_mention_map: Optional[Mapping[int, bool]] = None
) -> list[dict[str, Any]]:
    if wildcard_mention_map is None:
        wildcard_mention_map = {}
    zerver_usermessage: list[dict[str, Any]] = []
    for message in zerver_message:
        message_id: int = message['id']
        recipient_id: int = message['recipient']
        sender_id: int = message['sender']
        mention_user_ids: set[int] = mention_map[message_id]
        wildcard_mention: bool = wildcard_mention_map.get(message_id, False)
        subscriber_ids: set[int] = subscriber_map.get(recipient_id, set())
        user_ids: set[int] = subscriber_ids | {sender_id}
        for user_id in user_ids:
            is_mentioned: bool = user_id in mention_user_ids
            user_message = build_user_message(
                user_id=user_id, 
                message_id=message_id, 
                is_private=is_pm_data, 
                is_mentioned=is_mentioned, 
                wildcard_mention=wildcard_mention
            )
            zerver_usermessage.append(user_message)
    return zerver_usermessage

def build_subscription(recipient_id: int, user_id: int, subscription_id: int) -> dict[str, Any]:
    subscription = Subscription(color=random.choice(STREAM_COLORS), id=subscription_id)
    subscription_dict: dict[str, Any] = model_to_dict(subscription, exclude=['user_profile', 'recipient_id'])
    subscription_dict['user_profile'] = user_id
    subscription_dict['recipient'] = recipient_id
    return subscription_dict

class GetUsers(Protocol):
    def __call__(self, *, stream_id: Optional[int] = None, direct_message_group_id: Optional[int] = None) -> Iterable[int]:
        ...

def build_stream_subscriptions(
    get_users: GetUsers, 
    zerver_recipient: Iterable[Mapping[str, Any]], 
    zerver_stream: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    subscriptions: list[dict[str, Any]] = []
    stream_ids = {stream['id'] for stream in zerver_stream}
    recipient_map = {
        recipient['id']: recipient['type_id'] 
        for recipient in zerver_recipient 
        if recipient['type'] == Recipient.STREAM and recipient['type_id'] in stream_ids
    }
    for recipient_id, stream_id in recipient_map.items():
        user_ids = get_users(stream_id=stream_id)
        for user_id in user_ids:
            subscription = build_subscription(
                recipient_id=recipient_id, 
                user_id=user_id, 
                subscription_id=NEXT_ID('subscription')
            )
            subscriptions.append(subscription)
    return subscriptions

def build_direct_message_group_subscriptions(
    get_users: GetUsers, 
    zerver_recipient: Iterable[Mapping[str, Any]], 
    zerver_direct_message_group: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    subscriptions: list[dict[str, Any]] = []
    direct_message_group_ids = {direct_message_group['id'] for direct_message_group in zerver_direct_message_group}
    recipient_map = {
        recipient['id']: recipient['type_id'] 
        for recipient in zerver_recipient 
        if recipient['type'] == Recipient.DIRECT_MESSAGE_GROUP and recipient['type_id'] in direct_message_group_ids
    }
    for recipient_id, direct_message_group_id in recipient_map.items():
        user_ids = get_users(direct_message_group_id=direct_message_group_id)
        for user_id in user_ids:
            subscription = build_subscription(
                recipient_id=recipient_id, 
                user_id=user_id, 
                subscription_id=NEXT_ID('subscription')
            )
            subscriptions.append(subscription)
    return subscriptions

def build_personal_subscriptions(zerver_recipient: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    subscriptions: list[dict[str, Any]] = []
    personal_recipients = [recipient for recipient in zerver_recipient if recipient['type'] == Recipient.PERSONAL]
    for recipient in personal_recipients:
        recipient_id: int = recipient['id']
        user_id: int = recipient['type_id']
        subscription = build_subscription(
            recipient_id=recipient_id, 
            user_id=user_id, 
            subscription_id=NEXT_ID('subscription')
        )
        subscriptions.append(subscription)
    return subscriptions

def build_recipient(type_id: int, recipient_id: int, type: int) -> dict[str, Any]:
    recipient = Recipient(type_id=type_id, id=recipient_id, type=type)
    recipient_dict: dict[str, Any] = model_to_dict(recipient)
    return recipient_dict

def build_recipients(
    zerver_userprofile: Iterable[Mapping[str, Any]], 
    zerver_stream: Iterable[Mapping[str, Any]], 
    zerver_direct_message_group: Optional[Iterable[Mapping[str, Any]]] = None
) -> list[dict[str, Any]]:
    recipients: list[dict[str, Any]] = []
    for user in zerver_userprofile:
        type_id: int = user['id']
        typ: int = Recipient.PERSONAL
        recipient = Recipient(type_id=type_id, id=NEXT_ID('recipient'), type=typ)
        recipient_dict: dict[str, Any] = model_to_dict(recipient)
        recipients.append(recipient_dict)
    for stream in zerver_stream:
        type_id: int = stream['id']
        typ: int = Recipient.STREAM
        recipient = Recipient(type_id=type_id, id=NEXT_ID('recipient'), type=typ)
        recipient_dict = model_to_dict(recipient)
        recipients.append(recipient_dict)
    if zerver_direct_message_group is not None:
        for direct_message_group in zerver_direct_message_group:
            type_id: int = direct_message_group['id']
            typ: int = Recipient.DIRECT_MESSAGE_GROUP
            recipient = Recipient(type_id=type_id, id=NEXT_ID('recipient'), type=typ)
            recipient_dict = model_to_dict(recipient)
            recipients.append(recipient_dict)
    return recipients

def build_realm(zerver_realm: dict[str, Any], realm_id: int, domain_name: str) -> dict[str, Any]:
    realm = dict(
        zerver_client=[{'name': 'populate_db', 'id': 1}, {'name': 'website', 'id': 2}, {'name': 'API', 'id': 3}],
        zerver_customprofilefield=[],
        zerver_customprofilefieldvalue=[],
        zerver_userpresence=[],
        zerver_userprofile_mirrordummy=[],
        zerver_realmdomain=[{'realm': realm_id, 'allow_subdomains': False, 'domain': domain_name, 'id': realm_id}],
        zerver_useractivity=[],
        zerver_realm=zerver_realm,
        zerver_huddle=[],
        zerver_userprofile_crossrealm=[],
        zerver_useractivityinterval=[],
        zerver_reaction=[],
        zerver_realmemoji=[],
        zerver_realmfilter=[],
        zerver_realmplayground=[],
        zerver_realmauthenticationmethod=[
            {'realm': realm_id, 'name': name, 'id': i}
            for i, name in enumerate(all_default_backend_names(), start=1)
        ]
    )
    return realm

def build_usermessages(
    zerver_usermessage: list[dict[str, Any]], 
    subscriber_map: Mapping[int, set[int]], 
    recipient_id: int, 
    mentioned_user_ids: set[int], 
    message_id: int, 
    is_private: bool, 
    long_term_idle: set[int] = set()
) -> tuple[int, int]:
    user_ids: set[int] = subscriber_map.get(recipient_id, set())
    user_messages_created: int = 0
    user_messages_skipped: int = 0
    if user_ids:
        for user_id in sorted(user_ids):
            is_mentioned: bool = user_id in mentioned_user_ids
            if not is_mentioned and (not is_private) and (user_id in long_term_idle):
                user_messages_skipped += 1
                continue
            user_messages_created += 1
            usermessage = build_user_message(
                user_id=user_id, 
                message_id=message_id, 
                is_private=is_private, 
                is_mentioned=is_mentioned
            )
            zerver_usermessage.append(usermessage)
    return (user_messages_created, user_messages_skipped)

def build_user_message(
    user_id: int, 
    message_id: int, 
    is_private: bool, 
    is_mentioned: bool, 
    wildcard_mention: bool = False
) -> dict[str, Any]:
    flags_mask: int = 1
    if is_mentioned:
        flags_mask += 8
    if wildcard_mention:
        flags_mask += 16
    if is_private:
        flags_mask += 2048
    uid: int = NEXT_ID('user_message')
    usermessage: dict[str, Any] = dict(id=uid, user_profile=user_id, message=message_id, flags_mask=flags_mask)
    return usermessage

def build_defaultstream(realm_id: int, stream_id: int, defaultstream_id: int) -> dict[str, Any]:
    defaultstream = dict(stream=stream_id, realm=realm_id, id=defaultstream_id)
    return defaultstream

def build_stream(
    date_created: Any, 
    realm_id: int, 
    name: str, 
    description: str, 
    stream_id: int, 
    deactivated: bool = False, 
    invite_only: bool = False, 
    stream_post_policy: int = 1
) -> dict[str, Any]:
    history_public_to_subscribers: bool = not invite_only
    stream = Stream(
        name=name, 
        deactivated=deactivated, 
        description=description.replace('\n', ' '), 
        date_created=date_created, 
        invite_only=invite_only, 
        id=stream_id, 
        history_public_to_subscribers=history_public_to_subscribers
    )
    stream_dict: dict[str, Any] = model_to_dict(stream, exclude=['realm'])
    stream_dict['stream_post_policy'] = stream_post_policy
    stream_dict['realm'] = realm_id
    return stream_dict

def build_direct_message_group(direct_message_group_id: int, group_size: int) -> dict[str, Any]:
    direct_message_group = DirectMessageGroup(id=direct_message_group_id, group_size=group_size)
    return model_to_dict(direct_message_group)

def build_message(
    *, 
    topic_name: str, 
    date_sent: Any, 
    message_id: int, 
    content: str, 
    rendered_content: str, 
    user_id: int, 
    recipient_id: int, 
    realm_id: int, 
    has_image: bool = False, 
    has_link: bool = False, 
    has_attachment: bool = True
) -> dict[str, Any]:
    content = normalize_body_for_import(content)
    zulip_message = Message(
        rendered_content_version=1, 
        id=message_id, 
        content=content, 
        rendered_content=rendered_content, 
        has_image=has_image, 
        has_attachment=has_attachment, 
        has_link=has_link
    )
    zulip_message.set_topic_name(topic_name)
    zulip_message_dict: dict[str, Any] = model_to_dict(zulip_message, exclude=['recipient', 'sender', 'sending_client'])
    zulip_message_dict['sender'] = user_id
    zulip_message_dict['sending_client'] = 1
    zulip_message_dict['recipient'] = recipient_id
    zulip_message_dict['date_sent'] = date_sent
    return zulip_message_dict

def build_attachment(
    realm_id: int, 
    message_ids: Iterable[int], 
    user_id: int, 
    fileinfo: Mapping[str, Any], 
    s3_path: str, 
    zerver_attachment: list[dict[str, Any]]
) -> None:
    attachment_id: int = NEXT_ID('attachment')
    attachment = Attachment(
        id=attachment_id, 
        size=fileinfo['size'], 
        create_time=fileinfo['created'], 
        is_realm_public=True, 
        path_id=s3_path, 
        file_name=fileinfo['name'], 
        content_type=fileinfo.get('mimetype')
    )
    attachment_dict: dict[str, Any] = model_to_dict(attachment, exclude=['owner', 'messages', 'realm'])
    attachment_dict['owner'] = user_id
    attachment_dict['messages'] = list(message_ids)
    attachment_dict['realm'] = realm_id
    zerver_attachment.append(attachment_dict)

def get_avatar(avatar_dir: str, size_url_suffix: str, avatar_upload_item: list[str]) -> None:
    avatar_url: str = avatar_upload_item[0]
    image_path: str = os.path.join(avatar_dir, avatar_upload_item[1])
    original_image_path: str = os.path.join(avatar_dir, avatar_upload_item[2])
    if avatar_url.startswith('https://ca.slack-edge.com/'):
        avatar_url += size_url_suffix
    response = requests.get(avatar_url, stream=True)
    with open(image_path, 'wb') as image_file:
        shutil.copyfileobj(response.raw, image_file)
    shutil.copy(image_path, original_image_path)

def process_avatars(
    avatar_list: list[dict[str, Any]], 
    avatar_dir: str, 
    realm_id: int, 
    threads: int, 
    size_url_suffix: str = ''
) -> list[dict[str, Any]]:
    logging.info('######### GETTING AVATARS #########\n')
    logging.info('DOWNLOADING AVATARS .......\n')
    avatar_original_list: list[dict[str, Any]] = []
    avatar_upload_list: list[list[str]] = []
    for avatar in avatar_list:
        avatar_hash: str = user_avatar_base_path_from_ids(avatar['user_profile_id'], avatar['avatar_version'], realm_id)
        avatar_url: str = avatar['path']
        avatar_original: dict[str, Any] = dict(avatar)
        image_path: str = f'{avatar_hash}.png'
        original_image_path: str = f'{avatar_hash}.original'
        avatar_upload_list.append([avatar_url, image_path, original_image_path])
        avatar['path'] = image_path
        avatar['s3_path'] = image_path
        avatar['content_type'] = 'image/png'
        avatar_original['path'] = original_image_path
        avatar_original['s3_path'] = original_image_path
        avatar_original['content_type'] = 'image/png'
        avatar_original_list.append(avatar_original)
    run_parallel_wrapper(partial(get_avatar, avatar_dir, size_url_suffix), avatar_upload_list, threads=threads)
    logging.info('######### GETTING AVATARS FINISHED #########\n')
    return avatar_list + avatar_original_list

ListJobData = TypeVar('ListJobData')

def wrapping_function(f: Callable[[Any], Any], item: Any) -> None:
    try:
        f(item)
    except Exception:
        logging.exception('Error processing item: %s', item, stack_info=True)

def run_parallel_wrapper(f: Callable[[Any], Any], full_items: Iterable[Any], threads: int = 6) -> None:
    logging.info('Distributing %s items across %s threads', len(list(full_items)), threads)
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = (executor.submit(wrapping_function, f, item) for item in full_items)
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 1000 == 0:
                logging.info('Finished %s items', count)

def get_uploads(upload_dir: str, upload: list[str]) -> None:
    upload_url: str = upload[0]
    upload_path: str = upload[1]
    upload_path = os.path.join(upload_dir, upload_path)
    response = requests.get(upload_url, stream=True)
    os.makedirs(os.path.dirname(upload_path), exist_ok=True)
    with open(upload_path, 'wb') as upload_file:
        shutil.copyfileobj(response.raw, upload_file)

def process_uploads(
    upload_list: list[dict[str, Any]], 
    upload_dir: str, 
    threads: int
) -> list[dict[str, Any]]:
    logging.info('######### GETTING ATTACHMENTS #########\n')
    logging.info('DOWNLOADING ATTACHMENTS .......\n')
    upload_url_list: list[list[str]] = []
    for upload in upload_list:
        upload_url: str = upload['path']
        upload_s3_path: str = upload['s3_path']
        upload_url_list.append([upload_url, upload_s3_path])
        upload['path'] = upload_s3_path
    run_parallel_wrapper(partial(get_uploads, upload_dir), upload_url_list, threads=threads)
    logging.info('######### GETTING ATTACHMENTS FINISHED #########\n')
    return upload_list

def build_realm_emoji(realm_id: int, name: str, id: int, file_name: str) -> dict[str, Any]:
    return model_to_dict(RealmEmoji(realm_id=realm_id, name=name, id=id, file_name=file_name))

def get_emojis(emoji_dir: str, emoji_url: str, emoji_path: str) -> Optional[str]:
    upload_emoji_path: str = os.path.join(emoji_dir, emoji_path)
    response = requests.get(emoji_url, stream=True)
    os.makedirs(os.path.dirname(upload_emoji_path), exist_ok=True)
    with open(upload_emoji_path, 'wb') as emoji_file:
        shutil.copyfileobj(response.raw, emoji_file)
    return response.headers.get('Content-Type')

def process_emojis(
    zerver_realmemoji: list[dict[str, Any]], 
    emoji_dir: str, 
    emoji_url_map: Mapping[str, str], 
    threads: int
) -> list[dict[str, Any]]:
    emoji_records: list[dict[str, Any]] = []
    logging.info('######### GETTING EMOJIS #########\n')
    logging.info('DOWNLOADING EMOJIS .......\n')
    for emoji in zerver_realmemoji:
        emoji_url: str = emoji_url_map[emoji['name']]
        emoji_path: str = RealmEmoji.PATH_ID_TEMPLATE.format(realm_id=emoji['realm'], emoji_file_name=emoji['name'])
        emoji_record: dict[str, Any] = dict(emoji)
        emoji_record['path'] = emoji_path
        emoji_record['s3_path'] = emoji_path
        emoji_record['realm_id'] = emoji_record['realm']
        emoji_record.pop('realm')
        emoji_records.append(emoji_record)
        content_type: Optional[str] = get_emojis(emoji_dir, emoji_url, emoji_path)
        if content_type is None:
            logging.warning('Emoji %s has an unspecified content type. Using the original file extension.', emoji['name'])
            continue
        if content_type not in THUMBNAIL_ACCEPT_IMAGE_TYPES or content_type not in INLINE_MIME_TYPES:
            raise BadImageError(f'Emoji {emoji["name"]} is not an image file. Content type: {content_type}')
        file_extension: Optional[str] = guess_extension(content_type, strict=False)
        assert file_extension is not None
        old_file_name: str = emoji_record['file_name']
        new_file_name: str = f'{old_file_name.rsplit(".", 1)[0]}{file_extension}'
        emoji_record['file_name'] = new_file_name
        emoji['file_name'] = new_file_name
    logging.info('######### GETTING EMOJIS FINISHED #########\n')
    return emoji_records

def create_converted_data_files(data: Any, output_dir: str, file_path: str) -> None:
    output_file: str = output_dir + file_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as fp:
        fp.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

ExternalId = TypeVar('ExternalId')

def long_term_idle_helper(
    message_iterator: Iterable[Any], 
    user_from_message: Callable[[Any], Optional[int]], 
    timestamp_from_message: Callable[[Any], float], 
    zulip_user_id_from_user: Callable[[int], int], 
    all_user_ids_iterator: Iterable[int], 
    zerver_userprofile: list[dict[str, Any]]
) -> set[int]:
    sender_counts: dict[int, int] = defaultdict(int)
    recent_senders: set[int] = set()
    NOW: float = float(timezone_now().timestamp())
    for message in message_iterator:
        timestamp: float = timestamp_from_message(message)
        user: Optional[int] = user_from_message(message)
        if user is None:
            continue
        if user in recent_senders:
            continue
        if NOW - timestamp < 60 * 24 * 60 * 60:
            recent_senders.add(user)
        sender_counts[user] += 1
    for user, count in sender_counts.items():
        if count > 10:
            recent_senders.add(user)
    long_term_idle: set[int] = set()
    for user_id in all_user_ids_iterator:
        if user_id in recent_senders:
            continue
        zulip_user_id: int = zulip_user_id_from_user(user_id)
        long_term_idle.add(zulip_user_id)
    for user_profile_row in zerver_userprofile:
        if user_profile_row['id'] in long_term_idle:
            user_profile_row['long_term_idle'] = True
            user_profile_row['last_active_message_id'] = 1
    return long_term_idle

def validate_user_emails_for_import(user_emails: Iterable[str]) -> None:
    invalid_emails: list[str] = []
    for email in user_emails:
        try:
            validate_email(email)
        except ValidationError:
            invalid_emails.append(email)
    if invalid_emails:
        details: str = ', '.join(invalid_emails)
        error_log: str = f'Invalid email format, please fix the following email(s) and try again: {details}'
        raise ValidationError(error_log)