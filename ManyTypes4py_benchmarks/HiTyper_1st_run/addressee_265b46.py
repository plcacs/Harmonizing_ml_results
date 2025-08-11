from collections.abc import Iterable, Sequence
from typing import cast
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.string_validation import check_stream_topic
from zerver.lib.topic import maybe_rename_general_chat_to_empty_topic
from zerver.models import Realm, Stream, UserProfile
from zerver.models.users import get_user_by_id_in_realm_including_cross_realm, get_user_including_cross_realm

def get_user_profiles(emails: zerver.models.Realm, realm: Union[zerver.models.Realm, typing.Sequence[str], str]) -> list:
    user_profiles = []
    for email in emails:
        try:
            user_profile = get_user_including_cross_realm(email, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_("Invalid email '{email}'").format(email=email))
        user_profiles.append(user_profile)
    return user_profiles

def get_user_profiles_by_ids(user_ids: Union[list[int], zerver.models.Recipient], realm: Union[int, Realm, zerver.models.UserProfile]) -> list[typing.Union[models.user.UserModel,list,typing.Type]]:
    user_profiles = []
    for user_id in user_ids:
        try:
            user_profile = get_user_by_id_in_realm_including_cross_realm(user_id, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_('Invalid user ID {user_id}').format(user_id=user_id))
        user_profiles.append(user_profile)
    return user_profiles

class Addressee:

    def __init__(self, msg_type: Union[int, None], user_profiles: Union[None, str, bool, zerver.models.UserProfile]=None, stream: Union[None, typing.IO, int, bytes]=None, stream_name: Union[None, str, int]=None, stream_id: Union[None, int, str]=None, topic_name: Union[None, str, django.http.QueryDict]=None) -> None:
        assert msg_type in ['stream', 'private']
        if msg_type == 'stream' and topic_name is None:
            raise JsonableError(_('Missing topic'))
        self._msg_type = msg_type
        self._user_profiles = user_profiles
        self._stream = stream
        self._stream_name = stream_name
        self._stream_id = stream_id
        self._topic_name = topic_name

    def is_stream(self) -> bool:
        return self._msg_type == 'stream'

    def is_private(self) -> bool:
        return self._msg_type == 'private'

    def user_profiles(self):
        assert self.is_private()
        assert self._user_profiles is not None
        return self._user_profiles

    def stream(self):
        assert self.is_stream()
        return self._stream

    def stream_name(self):
        assert self.is_stream()
        return self._stream_name

    def stream_id(self):
        assert self.is_stream()
        return self._stream_id

    def topic_name(self):
        assert self.is_stream()
        assert self._topic_name is not None
        return self._topic_name

    @staticmethod
    def legacy_build(sender: Union[str, zerver.models.UserProfile], recipient_type_name: Union[str, dict[str, typing.Any], None], message_to: str, topic_name: Union[str, None, bool], realm: Union[None, str, zerver.models.UserProfile, zerver.models.Realm]=None) -> Union[list[str], str]:
        if realm is None:
            realm = sender.realm
        if recipient_type_name == 'stream':
            if len(message_to) > 1:
                raise JsonableError(_('Cannot send to multiple channels'))
            if message_to:
                stream_name_or_id = message_to[0]
            elif sender.default_sending_stream_id:
                stream_name_or_id = sender.default_sending_stream_id
            else:
                raise JsonableError(_('Missing channel'))
            if topic_name is None:
                raise JsonableError(_('Missing topic'))
            if isinstance(stream_name_or_id, int):
                return Addressee.for_stream_id(stream_name_or_id, topic_name)
            return Addressee.for_stream_name(stream_name_or_id, topic_name)
        elif recipient_type_name == 'private':
            if not message_to:
                raise JsonableError(_('Message must have recipients'))
            if isinstance(message_to[0], str):
                emails = cast(Sequence[str], message_to)
                return Addressee.for_private(emails, realm)
            elif isinstance(message_to[0], int):
                user_ids = cast(Sequence[int], message_to)
                return Addressee.for_user_ids(user_ids=user_ids, realm=realm)
        else:
            raise JsonableError(_('Invalid message type'))

    @staticmethod
    def for_stream(stream: Union[str, medtagger.database.models.User], topic_name: str) -> Addressee:
        topic_name = topic_name.strip()
        topic_name = maybe_rename_general_chat_to_empty_topic(topic_name)
        check_stream_topic(topic_name)
        return Addressee(msg_type='stream', stream=stream, topic_name=topic_name)

    @staticmethod
    def for_stream_name(stream_name: Union[str, bool, None], topic_name: str) -> Addressee:
        topic_name = topic_name.strip()
        topic_name = maybe_rename_general_chat_to_empty_topic(topic_name)
        check_stream_topic(topic_name)
        return Addressee(msg_type='stream', stream_name=stream_name, topic_name=topic_name)

    @staticmethod
    def for_stream_id(stream_id: Union[int, str, list[int]], topic_name: str) -> Addressee:
        topic_name = topic_name.strip()
        topic_name = maybe_rename_general_chat_to_empty_topic(topic_name)
        check_stream_topic(topic_name)
        return Addressee(msg_type='stream', stream_id=stream_id, topic_name=topic_name)

    @staticmethod
    def for_private(emails: Union[zerver.models.UserProfile, str, Realm], realm: Union[zerver.models.Realm, typing.Sequence[str], str]) -> Addressee:
        assert len(emails) > 0
        user_profiles = get_user_profiles(emails, realm)
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def for_user_ids(user_ids: Union[list[int], int, set[int]], realm: Union[int, zerver.models.UserProfile, Realm]) -> Addressee:
        assert len(user_ids) > 0
        user_profiles = get_user_profiles_by_ids(user_ids, realm)
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def for_user_profile(user_profile: zerver.models.UserProfile) -> Addressee:
        user_profiles = [user_profile]
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def for_user_profiles(user_profiles: Union[zerver.models.UserProfile, list[zerver.models.UserProfile], zerver.models.UserGroup]) -> Addressee:
        assert len(user_profiles) > 0
        return Addressee(msg_type='private', user_profiles=user_profiles)