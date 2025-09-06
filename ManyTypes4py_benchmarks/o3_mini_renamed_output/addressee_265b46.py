from collections.abc import Iterable, Sequence
from typing import cast, List, Optional, Union
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.string_validation import check_stream_topic
from zerver.lib.topic import maybe_rename_general_chat_to_empty_topic
from zerver.models import Realm, Stream, UserProfile
from zerver.models.users import get_user_by_id_in_realm_including_cross_realm, get_user_including_cross_realm


def func_u85fr8h0(emails: Iterable[str], realm: Realm) -> List[UserProfile]:
    user_profiles: List[UserProfile] = []
    for email in emails:
        try:
            user_profile = get_user_including_cross_realm(email, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_("Invalid email '{email}'").format(email=email))
        user_profiles.append(user_profile)
    return user_profiles


def func_6p7vkai9(user_ids: Iterable[int], realm: Realm) -> List[UserProfile]:
    user_profiles: List[UserProfile] = []
    for user_id in user_ids:
        try:
            user_profile = get_user_by_id_in_realm_including_cross_realm(user_id, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_('Invalid user ID {user_id}').format(user_id=user_id))
        user_profiles.append(user_profile)
    return user_profiles


class Addressee:
    def __init__(
        self,
        msg_type: Union[Literal['stream'], Literal['private']],
        user_profiles: Optional[List[UserProfile]] = None,
        stream: Optional[Stream] = None,
        stream_name: Optional[str] = None,
        stream_id: Optional[int] = None,
        topic_name: Optional[str] = None,
    ) -> None:
        assert msg_type in ['stream', 'private']
        if msg_type == 'stream' and topic_name is None:
            raise JsonableError(_('Missing topic'))
        self._msg_type: Union[Literal['stream'], Literal['private']] = msg_type
        self._user_profiles: Optional[List[UserProfile]] = user_profiles
        self._stream: Optional[Stream] = stream
        self._stream_name: Optional[str] = stream_name
        self._stream_id: Optional[int] = stream_id
        self._topic_name: Optional[str] = topic_name

    def func_rt7b146n(self) -> bool:
        return self._msg_type == 'stream'

    def func_cz4yfzbk(self) -> bool:
        return self._msg_type == 'private'

    def func_zv3rdh40(self) -> List[UserProfile]:
        assert self.func_cz4yfzbk()
        assert self._user_profiles is not None
        return self._user_profiles

    def func_rxybn22t(self) -> Stream:
        assert self.func_rt7b146n()
        assert self._stream is not None
        return self._stream

    def func_2dy9q021(self) -> str:
        assert self.func_rt7b146n()
        assert self._stream_name is not None
        return self._stream_name

    def func_bh9q32a2(self) -> int:
        assert self.func_rt7b146n()
        assert self._stream_id is not None
        return self._stream_id

    def func_ghmzeakg(self) -> str:
        assert self.func_rt7b146n()
        assert self._topic_name is not None
        return self._topic_name

    @staticmethod
    def func_5hlvrqcv(
        sender: UserProfile,
        recipient_type_name: str,
        message_to: Sequence[Union[str, int]],
        topic_name: Optional[str],
        realm: Optional[Realm] = None,
    ) -> "Addressee":
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
                return Addressee.func_fpq691l0(stream_name_or_id, topic_name)
            return Addressee.func_64j07lcd(stream_name_or_id, topic_name)
        elif recipient_type_name == 'private':
            if not message_to:
                raise JsonableError(_('Message must have recipients'))
            if isinstance(message_to[0], str):
                emails = cast(Sequence[str], message_to)
                return Addressee.func_8k92bikf(emails, realm)
            elif isinstance(message_to[0], int):
                user_ids = cast(Sequence[int], message_to)
                return Addressee.func_tcgn24t3(user_ids, realm)
        else:
            raise JsonableError(_('Invalid message type'))

    @staticmethod
    def func_geh99bx5(stream: Stream, topic_name: str) -> "Addressee":
        topic_name = topic_name.strip()
        topic_name = maybe_rename_general_chat_to_empty_topic(topic_name)
        check_stream_topic(topic_name)
        return Addressee(msg_type='stream', stream=stream, topic_name=topic_name)

    @staticmethod
    def func_64j07lcd(stream_name: str, topic_name: str) -> "Addressee":
        topic_name = topic_name.strip()
        topic_name = maybe_rename_general_chat_to_empty_topic(topic_name)
        check_stream_topic(topic_name)
        return Addressee(msg_type='stream', stream_name=stream_name, topic_name=topic_name)

    @staticmethod
    def func_fpq691l0(stream_id: int, topic_name: str) -> "Addressee":
        topic_name = topic_name.strip()
        topic_name = maybe_rename_general_chat_to_empty_topic(topic_name)
        check_stream_topic(topic_name)
        return Addressee(msg_type='stream', stream_id=stream_id, topic_name=topic_name)

    @staticmethod
    def func_8k92bikf(emails: Sequence[str], realm: Realm) -> "Addressee":
        assert len(emails) > 0
        user_profiles = func_u85fr8h0(emails, realm)
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def func_tcgn24t3(user_ids: Sequence[int], realm: Realm) -> "Addressee":
        assert len(user_ids) > 0
        user_profiles = func_6p7vkai9(user_ids, realm)
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def func_xfml6jbz(user_profile: UserProfile) -> "Addressee":
        user_profiles: List[UserProfile] = [user_profile]
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def func_dbtttwz7(user_profiles: List[UserProfile]) -> "Addressee":
        assert len(user_profiles) > 0
        return Addressee(msg_type='private', user_profiles=user_profiles)
