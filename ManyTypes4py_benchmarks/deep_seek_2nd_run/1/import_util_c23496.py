import logging
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Protocol, TypeVar, Optional, List, Dict, Set, Tuple, Union
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
from typing_extensions import TypedDict

ZerverFieldsT = Dict[str, Any]

class SubscriberHandler:
    def __init__(self) -> None:
        self.stream_info: Dict[int, Set[int]] = {}
        self.direct_message_group_info: Dict[int, Set[int]] = {}

    def set_info(self, users: Set[int], stream_id: Optional[int] = None, direct_message_group_id: Optional[int] = None) -> None:
        if stream_id is not None:
            self.stream_info[stream_id] = users
        elif direct_message_group_id is not None:
            self.direct_message_group_info[direct_message_group_id] = users
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

    def get_users(self, stream_id: Optional[int] = None, direct_message_group_id: Optional[int] = None) -> Set[int]:
        if stream_id is not None:
            return self.stream_info[stream_id]
        elif direct_message_group_id is not None:
            return self.direct_message_group_info[direct_message_group_id]
        else:
            raise AssertionError('stream_id or direct_message_group_id is required')

def build_zerver_realm(realm_id: int, realm_subdomain: str, time: float, other_product: str) -> List[Dict[str, Any]]:
    realm = Realm(id=realm_id, name=realm_subdomain, string_id=realm_subdomain, description=f'Organization imported from {other_product}!')
    realm_dict = model_to_dict(realm)
    realm_dict['date_created'] = time
    del realm_dict['uuid']
    del realm_dict['uuid_owner_secret']
    return [realm_dict]

def build_user_profile(avatar_source: str, date_joined: float, delivery_email: str, email: str, full_name: str, id: int, is_active: bool, role: int, is_mirror_dummy: bool, realm_id: int, short_name: str, timezone: str, is_bot: bool = False, bot_type: Optional[int] = None) -> Dict[str, Any]:
    obj = UserProfile(avatar_source=avatar_source, date_joined=date_joined, delivery_email=delivery_email, email=email, full_name=full_name, id=id, is_mirror_dummy=is_mirror_dummy, is_active=is_active, role=role, realm_id=realm_id, timezone=timezone, is_bot=is_bot, bot_type=bot_type)
    dct = model_to_dict(obj)
    dct['short_name'] = short_name
    return dct

def build_avatar(zulip_user_id: int, realm_id: int, email: str, avatar_url: str, timestamp: float, avatar_list: List[Dict[str, Any]]) -> None:
    avatar = dict(path=avatar_url, realm_id=realm_id, content_type=None, avatar_version=1, user_profile_id=zulip_user_id, last_modified=timestamp, user_profile_email=email, s3_path='', size='')
    avatar_list.append(avatar)

def make_subscriber_map(zerver_subscription: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
    subscriber_map: Dict[int, Set[int]] = {}
    for sub in zerver_subscription:
        user_id = sub['user_profile']
        recipient_id = sub['recipient']
        if recipient_id not in subscriber_map:
            subscriber_map[recipient_id] = set()
        subscriber_map[recipient_id].add(user_id)
    return subscriber_map

def make_user_messages(zerver_message: List[Dict[str, Any]], subscriber_map: Dict[int, Set[int]], is_pm_data: bool, mention_map: Dict[int, Set[int]], wildcard_mention_map: Dict[int, bool] = {}) -> List[Dict[str, Any]]:
    zerver_usermessage: List[Dict[str, Any]] = []
    for message in zerver_message:
        message_id = message['id']
        recipient_id = message['recipient']
        sender_id = message['sender']
        mention_user_ids = mention_map[message_id]
        wildcard_mention = wildcard_mention_map.get(message_id, False)
        subscriber_ids = subscriber_map.get(recipient_id, set())
        user_ids = subscriber_ids | {sender_id}
        for user_id in user_ids:
            is_mentioned = user_id in mention_user_ids
            user_message = build_user_message(user_id=user_id, message_id=message_id, is_private=is_pm_data, is_mentioned=is_mentioned, wildcard_mention=wildcard_mention)
            zerver_usermessage.append(user_message)
    return zerver_usermessage

def build_subscription(recipient_id: int, user_id: int, subscription_id: int) -> Dict[str, Any]:
    subscription = Subscription(color=random.choice(STREAM_COLORS), id=subscription_id)
    subscription_dict = model_to_dict(subscription, exclude=['user_profile', 'recipient_id'])
    subscription_dict['user_profile'] = user_id
    subscription_dict['recipient'] = recipient_id
    return subscription_dict

class GetUsers(Protocol):
    def __call__(self, stream_id: Optional[int] = ..., direct_message_group_id: Optional[int] = ...) -> Set[int]: ...

def build_stream_subscriptions(get_users: GetUsers, zerver_recipient: List[Dict[str, Any]], zerver_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    subscriptions: List[Dict[str, Any]] = []
    stream_ids = {stream['id'] for stream in zerver_stream}
    recipient_map = {recipient['id']: recipient['type_id'] for recipient in zerver_recipient if recipient['type'] == Recipient.STREAM and recipient['type_id'] in stream_ids}
    for recipient_id, stream_id in recipient_map.items():
        user_ids = get_users(stream_id=stream_id)
        for user_id in user_ids:
            subscription = build_subscription(recipient_id=recipient_id, user_id=user_id, subscription_id=NEXT_ID('subscription'))
            subscriptions.append(subscription)
    return subscriptions

def build_direct_message_group_subscriptions(get_users: GetUsers, zerver_recipient: List[Dict[str, Any]], zerver_direct_message_group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    subscriptions: List[Dict[str, Any]] = []
    direct_message_group_ids = {direct_message_group['id'] for direct_message_group in zerver_direct_message_group}
    recipient_map = {recipient['id']: recipient['type_id'] for recipient in zerver_recipient if recipient['type'] == Recipient.DIRECT_MESSAGE_GROUP and recipient['type_id'] in direct_message_group_ids}
    for recipient_id, direct_message_group_id in recipient_map.items():
        user_ids = get_users(direct_message_group_id=direct_message_group_id)
        for user_id in user_ids:
            subscription = build_subscription(recipient_id=recipient_id, user_id=user_id, subscription_id=NEXT_ID('subscription'))
            subscriptions.append(subscription)
    return subscriptions

def build_personal_subscriptions(zerver_recipient: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    subscriptions: List[Dict[str, Any]] = []
    personal_recipients = [recipient for recipient in zerver_recipient if recipient['type'] == Recipient.PERSONAL]
    for recipient in personal_recipients:
        recipient_id = recipient['id']
        user_id = recipient['type_id']
        subscription = build_subscription(recipient_id=recipient_id, user_id=user_id, subscription_id=NEXT_ID('subscription'))
        subscriptions.append(subscription)
    return subscriptions

def build_recipient(type_id: int, recipient_id: int, type: int) -> Dict[str, Any]:
    recipient = Recipient(type_id=type_id, id=recipient_id, type=type)
    recipient_dict = model_to_dict(recipient)
    return recipient_dict

def build_recipients(zerver_userprofile: List[Dict[str, Any]], zerver_stream: List[Dict[str, Any]], zerver_direct_message_group: List[Dict[str, Any]] = []) -> List[Dict[str, Any]]:
    recipients: List[Dict[str, Any]] = []
    for user in zerver_userprofile:
        type_id = user['id']
        type = Recipient.PERSONAL
        recipient = Recipient(type_id=type_id, id=NEXT_ID('recipient'), type=type)
        recipient_dict = model_to_dict(recipient)
        recipients.append(recipient_dict)
    for stream in zerver_stream:
        type_id = stream['id']
        type = Recipient.STREAM
        recipient = Recipient(type_id=type_id, id=NEXT_ID('recipient'), type=type)
        recipient_dict = model_to_dict(recipient)
        recipients.append(recipient_dict)
    for direct_message_group in zerver_direct_message_group:
        type_id = direct_message_group['id']
        type = Recipient.DIRECT_MESSAGE_GROUP
        recipient = Recipient(type_id=type_id, id=NEXT_ID('recipient'), type=type)
        recipient_dict = model_to_dict(recipient)
        recipients.append(recipient_dict)
    return recipients

def build_realm(zerver_realm: List[Dict[str, Any]], realm_id: int, domain_name: str) -> Dict[str, Any]:
    realm = dict(zerver_client=[{'name': 'populate_db', 'id': 1}, {'name': 'website', 'id': 2}, {'name': 'API', 'id': 3}], zerver_customprofilefield=[], zerver_customprofilefieldvalue=[], zerver_userpresence=[], zerver_userprofile_mirrordummy=[], zerver_realmdomain=[{'realm': realm_id, 'allow_subdomains': False, 'domain': domain_name, 'id': realm_id}], zerver_useractivity=[], zerver_realm=zerver_realm, zerver_huddle=[], zerver_userprofile_crossrealm=[], zerver_useractivityinterval=[], zerver_reaction=[], zerver_realmemoji=[], zerver_realmfilter=[], zerver_realmplayground=[], zerver_realmauthenticationmethod=[{'realm': realm_id, 'name': name, 'id': i} for i, name in enumerate(all_default_backend_names(), start=1)])
    return realm

def build_usermessages(zerver_usermessage: List[Dict[str, Any]], subscriber_map: Dict[int, Set[int]], recipient_id: int, mentioned_user_ids: Set[int], message_id: int, is_private: bool, long_term_idle: Set[int] = set()) -> Tuple[int, int]:
    user_ids = subscriber_map.get(recipient_id, set())
    user_messages_created = 0
    user_messages_skipped = 0
    if user_ids:
        for user_id in sorted(user_ids):
            is_mentioned = user_id in mentioned_user_ids
            if not is_mentioned and (not is_private) and (user_id in long_term_idle):
                user_messages_skipped += 1
                continue
            user_messages_created += 1
            usermessage = build_user_message(user_id=user_id, message_id=message_id, is_private=is_private, is_mentioned=is_mentioned)
            zerver_usermessage.append(usermessage)
    return (user_messages_created, user_messages_skipped)

def build_user_message(user_id: int, message_id: int, is_private: bool, is_mentioned: bool, wildcard_mention: bool = False) -> Dict[str, Any]:
    flags_mask = 1
    if is_mentioned:
        flags_mask += 8
    if wildcard_mention:
        flags_mask += 16
    if is_private:
        flags_mask += 2048
    id = NEXT_ID('user_message')
    usermessage = dict(id=id, user_profile=user_id, message=message_id, flags_mask=flags_mask)
    return usermessage

def build_defaultstream(realm_id: int, stream_id: int, defaultstream_id: int) -> Dict[str, Any]:
    defaultstream = dict(stream=stream_id, realm=realm_id, id=defaultstream_id)
    return defaultstream

def build_stream(date_created: float, realm_id: int, name: str, description: str, stream_id: int, deactivated: bool = False, invite_only: bool = False, stream_post_policy: int = 1) -> Dict[str, Any]:
    history_public_to_subscribers = not invite_only
    stream = Stream(name=name, deactivated=deactivated, description=description.replace('\n', ' '), date_created=date_created, invite_only=invite_only, id=stream_id, history_public_to_subscribers=history_public_to_subscribers)
    stream_dict = model_to_dict(stream, exclude=['realm'])
    stream_dict['stream_post_policy'] = stream_post_policy
    stream_dict['realm'] = realm_id
    return stream_dict

def build_direct_message_group(direct_message_group_id: int, group_size: int) -> Dict[str, Any]:
    direct_message_group = DirectMessageGroup(id=direct_message_group_id, group_size=group_size)
    return model_to_dict(direct_message_group)

def build_message(*, topic_name: str, date_sent: float, message_id: int, content: str, rendered_content: str, user_id: int, recipient_id: int, realm_id: int, has_image: bool = False, has_link: bool = False, has_attachment: bool = True) -> Dict[str, Any]:
    content = normalize_body_for_import(content)
    zulip_message = Message(rendered_content_version=1, id=message_id, content=content, rendered_content=rendered_content, has_image=has_image, has_attachment=has_attachment, has_link=has_link)
    zulip_message.set_topic_name(topic_name)
    zulip_message_dict = model_to_dict(zulip_message, exclude=['recipient', 'sender', 'sending_client'])
    zulip_message_dict['sender'] = user_id
    zulip_message_dict['sending_client'] = 1
    zulip_message_dict['recipient'] = recipient_id
    zulip_message_dict['date_sent'] = date_sent
    return zulip_message_dict

def build_attachment(realm_id: int, message_ids: List[int], user_id: int, fileinfo: Dict[str, Any], s3_path: str, zerver_attachment: List[Dict[str, Any]]) -> None:
    attachment_id = NEXT_ID('attachment')
    attachment = Attachment(id=attachment_id, size=fileinfo['size'], create_time=fileinfo['created'], is_realm_public=True, path_id=s3_path, file_name=fileinfo['name'], content_type=fileinfo.get('mimetype'))
    attachment_dict = model_to_dict(attachment, exclude=['owner', 'messages', 'realm'])
    attachment_dict['owner'] = user_id
    attachment_dict['messages'] = list(message_ids)
    attachment_dict['realm'] = realm_id
    zerver_attachment.append(attachment_dict)

def get_avatar(avatar_dir: str, size_url_suffix: str, avatar_upload_item: List[str]) -> None:
    avatar_url = avatar_upload_item[0]
    image_path = os.path.join(avatar_dir, avatar_upload_item[1])
    original_image_path = os.path.join(avatar_dir, avatar_upload_item[2])
    if avatar_url.startswith('https://ca.slack-edge.com/'):
        avatar_url += size_url_suffix
    response = requests.get(avatar_url, stream=True)
    with open(image_path, 'wb') as image_file:
        shutil.copyfileobj(response.raw, image_file)
    shutil.copy(image_path, original_image_path)

def process_avatars(avatar_list: List[Dict[str, Any]], avatar_dir: str, realm_id: int, threads: int, size_url_suffix: str = '') -> List[Dict[str, Any]]:
    logging.info('######### GETTING AVATARS #########\n')
    logging.info('DOWNLOADING AVATARS .......\n')
    avatar_original_list: List[Dict[str, Any]] = []
    avatar_upload_list: List[List[str]] = []
    for avatar in avatar_list:
        avatar_hash = user_avatar_base_path_from_ids(avatar['user_profile_id'], avatar['avatar_version'], realm_id)
        avatar_url = avatar['path']
        avatar_original = dict(avatar)
        image_path = f'{avatar_hash}.png'
        original_image_path = f'{avatar_hash}.original'
        avatar_upload_list.append([avatar_url, image_path, original_image_path])
        avatar['path'] = image_path
        avatar['s3_path'] = image_path
        avatar['content_type'] = 'image/png'
        avatar_original['path'] = original_image_path
        avatar_original['s3_path'] = original_image_path
        avatar_original['content_type'] = 'image/png'
