import copy
import zlib
from collections.abc import Iterable
from datetime import datetime
from email.headerregistry import Address
from typing import Any, TypedDict, Optional, Dict, List, Tuple, Mapping
import orjson
from zerver.lib.avatar import get_avatar_field, get_avatar_for_inaccessible_user
from zerver.lib.cache import cache_set_many, cache_with_key, to_dict_cache_key, to_dict_cache_key_id
from zerver.lib.display_recipient import bulk_fetch_display_recipients
from zerver.lib.markdown import render_message_markdown, topic_links
from zerver.lib.markdown import version as markdown_version
from zerver.lib.query_helpers import query_for_ids
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.topic import DB_TOPIC_NAME, TOPIC_LINKS, TOPIC_NAME
from zerver.lib.types import DisplayRecipientT, EditHistoryEvent, UserDisplayRecipient
from zerver.models import Message, Reaction, Realm, Recipient, Stream, SubMessage, UserProfile
from zerver.models.realms import get_fake_email_domain

class RawReactionRow(TypedDict):
    pass

def sew_messages_and_reactions(
    messages: List[Dict[str, Any]],
    reactions: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Given a iterable of messages and reactions stitch reactions
    into messages.
    """
    for message in messages:
        message['reactions'] = []
    converted_messages: Dict[int, Dict[str, Any]] = {message['id']: message for message in messages}
    for reaction in reactions:
        converted_messages[reaction['message_id']]['reactions'].append(reaction)
    return list(converted_messages.values())

def sew_messages_and_submessages(
    messages: List[Dict[str, Any]],
    submessages: Iterable[Dict[str, Any]],
) -> None:
    for message in messages:
        message['submessages'] = []
    message_dict: Dict[int, Dict[str, Any]] = {message['id']: message for message in messages}
    for submessage in submessages:
        message_id = submessage['message_id']
        if message_id in message_dict:
            message = message_dict[message_id]
            message['submessages'].append(submessage)

def extract_message_dict(message_bytes: bytes) -> Dict[str, Any]:
    return orjson.loads(zlib.decompress(message_bytes))

def stringify_message_dict(message_dict: Mapping[str, Any]) -> bytes:
    return zlib.compress(orjson.dumps(message_dict))

@cache_with_key(to_dict_cache_key, timeout=3600 * 24)
def message_to_encoded_cache(message: Message, realm_id: Optional[int] = None) -> bytes:
    return MessageDict.messages_to_encoded_cache([message], realm_id)[message.id]

def update_message_cache(
    changed_messages: Iterable[Message],
    realm_id: Optional[int] = None,
) -> List[int]:
    """Updates the message as stored in the to_dict cache (for serving
    messages)."""
    items_for_remote_cache: Dict[str, Tuple[bytes]] = {}
    message_ids: List[int] = []
    changed_messages_to_dict = MessageDict.messages_to_encoded_cache(changed_messages, realm_id)
    for msg_id, msg in changed_messages_to_dict.items():
        message_ids.append(msg_id)
        key = to_dict_cache_key_id(msg_id)
        items_for_remote_cache[key] = (msg,)
    cache_set_many(items_for_remote_cache)
    return message_ids

def save_message_rendered_content(message: Message, content: str) -> Optional[str]:
    rendering_result = render_message_markdown(message, content, realm=message.get_realm())
    rendered_content: Optional[str] = None
    if rendering_result is not None:
        rendered_content = rendering_result.rendered_content
    message.rendered_content = rendered_content
    message.rendered_content_version = markdown_version
    message.save_rendered_content()
    return rendered_content

class ReactionDict:

    @staticmethod
    def build_dict_from_raw_db_row(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {'emoji_name': row['emoji_name'], 'emoji_code': row['emoji_code'], 'reaction_type': row['reaction_type'], 'user_id': row['user_profile_id']}

class MessageDict:
    """MessageDict is the core class responsible for marshalling Message
    objects obtained from the database into a format that can be sent
    to clients via the Zulip API, whether via `GET /messages`,
    outgoing webhooks, or other code paths.  There are two core flows through
    which this class is used:

    * For just-sent messages, we construct a single `wide_dict` object
      containing all the data for the message and the related
      UserProfile models (sender_info and recipient_info); this object
      can be stored in queues, caches, etc., and then later turned
      into an API-format JSONable dictionary via finalize_payload.

    * When fetching messages from the database, we fetch their data in
      bulk using messages_for_ids, which makes use of caching, bulk
      fetches that skip the Django ORM, etc., to provide an optimized
      interface for fetching hundreds of thousands of messages from
      the database and then turning them into API-format JSON
      dictionaries.

    """

    @staticmethod
    def wide_dict(message: Message, realm_id: Optional[int] = None) -> Dict[str, Any]:
        """
        The next two lines get the cacheable field related
        to our message object, with the side effect of
        populating the cache.
        """
        encoded_object_bytes = message_to_encoded_cache(message, realm_id)
        obj = extract_message_dict(encoded_object_bytes)
        "\n        The steps below are similar to what we do in\n        post_process_dicts(), except we don't call finalize_payload(),\n        since that step happens later in the queue\n        processor.\n        "
        MessageDict.bulk_hydrate_sender_info([obj])
        MessageDict.bulk_hydrate_recipient_info([obj])
        return obj

    @staticmethod
    def post_process_dicts(
        objs: List[Dict[str, Any]],
        *,
        apply_markdown: bool,
        client_gravatar: bool,
        allow_empty_topic_name: bool,
        realm: Realm,
        user_recipient_id: int,
    ) -> None:
        """
        NOTE: This function mutates the objects in
              the `objs` list, rather than making
              shallow copies.  It might be safer to
              make shallow copies here, but performance
              is somewhat important here, as we are
              often fetching hundreds of messages.
        """
        MessageDict.bulk_hydrate_sender_info(objs)
        MessageDict.bulk_hydrate_recipient_info(objs)
        for obj in objs:
            can_access_sender = obj.get('can_access_sender', True)
            MessageDict.finalize_payload(obj, apply_markdown=apply_markdown, client_gravatar=client_gravatar, allow_empty_topic_name=allow_empty_topic_name, skip_copy=True, can_access_sender=can_access_sender, realm_host=realm.host, is_incoming_1_to_1=obj['recipient_id'] == user_recipient_id)

    @staticmethod
    def finalize_payload(
        obj: Dict[str, Any],
        *,
        apply_markdown: bool,
        client_gravatar: bool,
        allow_empty_topic_name: bool,
        keep_rendered_content: bool = False,
        skip_copy: bool = False,
        can_access_sender: bool,
        realm_host: str,
        is_incoming_1_to_1: bool,
    ) -> Dict[str, Any]:
        """
        By default, we make a shallow copy of the incoming dict to avoid
        mutation-related bugs.  Code paths that are passing a unique object
        can pass skip_copy=True to avoid this extra work.
        """
        if not skip_copy:
            obj = copy.copy(obj)
        if obj['recipient_type'] == Recipient.STREAM and obj['subject'] == '' and (not allow_empty_topic_name):
            obj['subject'] = Message.EMPTY_TOPIC_FALLBACK_NAME
        if obj['sender_email_address_visibility'] != UserProfile.EMAIL_ADDRESS_VISIBILITY_EVERYONE:
            client_gravatar = False
        if not can_access_sender:
            obj['sender_full_name'] = str(UserProfile.INACCESSIBLE_USER_NAME)
            sender_id = obj['sender_id']
            obj['sender_email'] = Address(username=f'user{sender_id}', domain=get_fake_email_domain(realm_host)).addr_spec
        MessageDict.set_sender_avatar(obj, client_gravatar, can_access_sender)
        if apply_markdown:
            obj['content_type'] = 'text/html'
            obj['content'] = obj['rendered_content']
        else:
            obj['content_type'] = 'text/x-markdown'
        if is_incoming_1_to_1 and 'sender_recipient_id' in obj:
            obj['recipient_id'] = obj['sender_recipient_id']
        for item in obj.get('edit_history', []):
            if 'prev_rendered_content_version' in item:
                del item['prev_rendered_content_version']
            if not allow_empty_topic_name:
                if 'prev_topic' in item and item['prev_topic'] == '':
                    item['prev_topic'] = Message.EMPTY_TOPIC_FALLBACK_NAME
                if 'topic' in item and item['topic'] == '':
                    item['topic'] = Message.EMPTY_TOPIC_FALLBACK_NAME
        if not keep_rendered_content:
            del obj['rendered_content']
        obj.pop('sender_recipient_id')
        del obj['sender_realm_id']
        del obj['sender_avatar_source']
        del obj['sender_delivery_email']
        del obj['sender_avatar_version']
        del obj['recipient_type']
        del obj['recipient_type_id']
        del obj['sender_is_mirror_dummy']
        del obj['sender_email_address_visibility']
        if 'can_access_sender' in obj:
            del obj['can_access_sender']
        return obj

    @staticmethod
    def sew_submessages_and_reactions_to_msgs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        msg_ids = [msg['id'] for msg in messages]
        submessages = SubMessage.get_raw_db_rows(msg_ids)
        sew_messages_and_submessages(messages, submessages)
        reactions = Reaction.get_raw_db_rows(msg_ids)
        return sew_messages_and_reactions(messages, reactions)

    @staticmethod
    def messages_to_encoded_cache(
        messages: Iterable[Message],
        realm_id: Optional[int] = None,
    ) -> Dict[int, bytes]:
        messages_dict = MessageDict.messages_to_encoded_cache_helper(messages, realm_id)
        encoded_messages: Dict[int, bytes] = {msg['id']: stringify_message_dict(msg) for msg in messages_dict}
        return encoded_messages

    @staticmethod
    def messages_to_encoded_cache_helper(
        messages: Iterable[Message],
        realm_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        def get_rendering_realm_id(message: Message) -> int:
            if realm_id is not None:
                return realm_id
            if message.recipient.type == Recipient.STREAM:
                return Stream.objects.get(id=message.recipient.type_id).realm_id
            return message.realm_id
        message_rows: List[Dict[str, Any]] = [{'id': message.id, DB_TOPIC_NAME: message.topic_name(), 'date_sent': message.date_sent, 'last_edit_time': message.last_edit_time, 'edit_history': message.edit_history, 'content': message.content, 'rendered_content': message.rendered_content, 'rendered_content_version': message.rendered_content_version, 'recipient_id': message.recipient.id, 'recipient__type': message.recipient.type, 'recipient__type_id': message.recipient.type_id, 'rendering_realm_id': get_rendering_realm_id(message), 'sender_id': message.sender.id, 'sending_client__name': message.sending_client.name, 'sender__realm_id': message.sender.realm_id} for message in messages]
        MessageDict.sew_submessages_and_reactions_to_msgs(message_rows)
        return [MessageDict.build_dict_from_raw_db_row(row) for row in message_rows]

    @staticmethod
    def ids_to_dict(needed_ids: Iterable[int]) -> List[Dict[str, Any]]:
        fields = ['id', DB_TOPIC_NAME, 'date_sent', 'last_edit_time', 'edit_history', 'content', 'rendered_content', 'rendered_content_version', 'recipient_id', 'recipient__type', 'recipient__type_id', 'sender_id', 'sending_client__name', 'sender__realm_id']
        messages = Message.objects.filter(id__in=needed_ids).values(*fields)
        MessageDict.sew_submessages_and_reactions_to_msgs(messages)
        return [MessageDict.build_dict_from_raw_db_row(row) for row in messages]

    @staticmethod
    def build_dict_from_raw_db_row(row: Mapping[str, Any]) -> Dict[str, Any]:
        """
        row is a row from a .values() call, and it needs to have
        all the relevant fields populated
        """
        return MessageDict.build_message_dict(message_id=row['id'], last_edit_time=row['last_edit_time'], edit_history_json=row['edit_history'], content=row['content'], topic_name=row[DB_TOPIC_NAME], date_sent=row['date_sent'], rendered_content=row['rendered_content'], rendered_content_version=row['rendered_content_version'], sender_id=row['sender_id'], sender_realm_id=row['sender__realm_id'], sending_client_name=row['sending_client__name'], rendering_realm_id=row.get('rendering_realm_id', row['sender__realm_id']), recipient_id=row['recipient_id'], recipient_type=row['recipient__type'], recipient_type_id=row['recipient__type_id'], reactions=row['reactions'], submessages=row['submessages'])

    @staticmethod
    def build_message_dict(
        message_id: int,
        last_edit_time: Optional[datetime],
        edit_history_json: Optional[str],
        content: str,
        topic_name: str,
        date_sent: datetime,
        rendered_content: Optional[str],
        rendered_content_version: Optional[int],
        sender_id: int,
        sender_realm_id: int,
        sending_client_name: str,
        rendering_realm_id: int,
        recipient_id: int,
        recipient_type: int,
        recipient_type_id: int,
        reactions: Iterable[Mapping[str, Any]],
        submessages: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        obj: Dict[str, Any] = dict(id=message_id, sender_id=sender_id, content=content, recipient_type_id=recipient_type_id, recipient_type=recipient_type, recipient_id=recipient_id, timestamp=datetime_to_timestamp(date_sent), client=sending_client_name)
        obj[TOPIC_NAME] = topic_name
        obj['sender_realm_id'] = sender_realm_id
        obj[TOPIC_LINKS] = topic_links(rendering_realm_id, topic_name)
        if last_edit_time is not None:
            obj['last_edit_timestamp'] = datetime_to_timestamp(last_edit_time)
            assert edit_history_json is not None
            edit_history = orjson.loads(edit_history_json)
            obj['edit_history'] = edit_history
        if Message.need_to_render_content(rendered_content, rendered_content_version, markdown_version):
            message = Message.objects.select_related('sender').get(id=message_id)
            assert message is not None
            rendered_content = save_message_rendered_content(message, content)
        if rendered_content is not None:
            obj['rendered_content'] = rendered_content
        else:
            obj['rendered_content'] = '<p>[Zulip note: Sorry, we could not understand the formatting of your message]</p>'
        if rendered_content is not None:
            obj['is_me_message'] = Message.is_status_message(content, rendered_content)
        else:
            obj['is_me_message'] = False
        obj['reactions'] = [ReactionDict.build_dict_from_raw_db_row(reaction) for reaction in reactions]
        obj['submessages'] = submessages
        return obj

    @staticmethod
    def bulk_hydrate_sender_info(objs: List[Dict[str, Any]]) -> None:
        sender_ids = list({obj['sender_id'] for obj in objs})
        if not sender_ids:
            return
        query = UserProfile.objects.values('id', 'full_name', 'delivery_email', 'email', 'recipient_id', 'realm__string_id', 'avatar_source', 'avatar_version', 'is_mirror_dummy', 'email_address_visibility')
        rows = query_for_ids(query, sender_ids, 'zerver_userprofile.id')
        sender_dict: Dict[int, Mapping[str, Any]] = {row['id']: row for row in rows}
        for obj in objs:
            sender_id = obj['sender_id']
            user_row = sender_dict[sender_id]
            obj['sender_recipient_id'] = user_row['recipient_id']
            obj['sender_full_name'] = user_row['full_name']
            obj['sender_email'] = user_row['email']
            obj['sender_delivery_email'] = user_row['delivery_email']
            obj['sender_realm_str'] = user_row['realm__string_id']
            obj['sender_avatar_source'] = user_row['avatar_source']
            obj['sender_avatar_version'] = user_row['avatar_version']
            obj['sender_is_mirror_dummy'] = user_row['is_mirror_dummy']
            obj['sender_email_address_visibility'] = user_row['email_address_visibility']

    @staticmethod
    def hydrate_recipient_info(obj: Dict[str, Any], display_recipient: DisplayRecipientT) -> None:
        """
        This method hyrdrates recipient info with things
        like full names and emails of senders.  Eventually
        our clients should be able to hyrdrate these fields
        themselves with info they already have on users.
        """
        recipient_type = obj['recipient_type']
        recipient_type_id = obj['recipient_type_id']
        sender_is_mirror_dummy = obj['sender_is_mirror_dummy']
        sender_email = obj['sender_email']
        sender_full_name = obj['sender_full_name']
        sender_id = obj['sender_id']
        if recipient_type == Recipient.STREAM:
            display_type = 'stream'
        elif recipient_type in (Recipient.DIRECT_MESSAGE_GROUP, Recipient.PERSONAL):
            assert not isinstance(display_recipient, str)
            display_type = 'private'
            if len(display_recipient) == 1:
                recip = {'email': sender_email, 'full_name': sender_full_name, 'id': sender_id, 'is_mirror_dummy': sender_is_mirror_dummy}
                if recip['email'] < display_recipient[0]['email']:
                    display_recipient = [recip, display_recipient[0]]
                elif recip['email'] > display_recipient[0]['email']:
                    display_recipient = [display_recipient[0], recip]
        else:
            raise AssertionError(f'Invalid recipient type {recipient_type}')
        obj['display_recipient'] = display_recipient
        obj['type'] = display_type
        if obj['type'] == 'stream':
            obj['stream_id'] = recipient_type_id

    @staticmethod
    def bulk_hydrate_recipient_info(objs: List[Dict[str, Any]]) -> None:
        recipient_tuples = {(obj['recipient_id'], obj['recipient_type'], obj['recipient_type_id']) for obj in objs}
        display_recipients = bulk_fetch_display_recipients(recipient_tuples)
        for obj in objs:
            MessageDict.hydrate_recipient_info(obj, display_recipients[obj['recipient_id']])

    @staticmethod
    def set_sender_avatar(obj: Dict[str, Any], client_gravatar: bool, can_access_sender: bool = True) -> None:
        if not can_access_sender:
            obj['avatar_url'] = get_avatar_for_inaccessible_user()
            return
        sender_id = obj['sender_id']
        sender_realm_id = obj['sender_realm_id']
        sender_delivery_email = obj['sender_delivery_email']
        sender_avatar_source = obj['sender_avatar_source']
        sender_avatar_version = obj['sender_avatar_version']
        obj['avatar_url'] = get_avatar_field(user_id=sender_id, realm_id=sender_realm_id, email=sender_delivery_email, avatar_source=sender_avatar_source, avatar_version=sender_avatar_version, medium=False, client_gravatar=client_gravatar)