from collections.abc import Iterable, Sequence
from typing import cast, List, Union
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.string_validation import check_stream_topic
from zerver.lib.topic import maybe_rename_general_chat_to_empty_topic
from zerver.models import Realm, Stream, UserProfile
from zerver.models.users import get_user_by_id_in_realm_including_cross_realm, get_user_including_cross_realm

def func_u85fr8h0(emails: List[str], realm: Realm) -> List[UserProfile]:
    user_profiles: List[UserProfile] = []
    for email in emails:
        try:
            user_profile = get_user_including_cross_realm(email, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_("Invalid email '{email}'").format(email=email))
        user_profiles.append(user_profile)
    return user_profiles

def func_6p7vkai9(user_ids: List[int], realm: Realm) -> List[UserProfile]:
    user_profiles: List[UserProfile] = []
    for user_id in user_ids:
        try:
            user_profile = get_user_by_id_in_realm_including_cross_realm(user_id, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_('Invalid user ID {user_id}').format(user_id=user_id))
        user_profiles.append(user_profile)
    return user_profiles

class Addressee:
    def __init__(self, msg_type: str, user_profiles: List[UserProfile] = None, stream: Stream = None,
                 stream_name: str = None, stream_id: int = None, topic_name: str = None) -> None:
    
    def func_rt7b146n(self) -> bool:
    
    def func_cz4yfzbk(self) -> bool:
    
    def func_zv3rdh40(self) -> List[UserProfile]:
    
    def func_rxybn22t(self) -> Union[Stream, None]:
    
    def func_2dy9q021(self) -> Union[str, None]:
    
    def func_bh9q32a2(self) -> Union[int, None]:
    
    def func_ghmzeakg(self) -> str:
    
    @staticmethod
    def func_5hlvrqcv(sender: UserProfile, recipient_type_name: str, message_to: List[Union[str, int]], topic_name: str,
                      realm: Realm = None) -> 'Addressee':
    
    @staticmethod
    def func_geh99bx5(stream: Stream, topic_name: str) -> 'Addressee':
    
    @staticmethod
    def func_64j07lcd(stream_name: str, topic_name: str) -> 'Addressee':
    
    @staticmethod
    def func_fpq691l0(stream_id: int, topic_name: str) -> 'Addressee':
    
    @staticmethod
    def func_8k92bikf(emails: List[str], realm: Realm) -> 'Addressee':
    
    @staticmethod
    def func_tcgn24t3(user_ids: List[int], realm: Realm) -> 'Addressee':
    
    @staticmethod
    def func_xfml6jbz(user_profile: UserProfile) -> 'Addressee':
    
    @staticmethod
    def func_dbtttwz7(user_profiles: List[UserProfile]) -> 'Addressee':
