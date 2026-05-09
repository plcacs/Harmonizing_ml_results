from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from django.db.models import Q, QuerySet
from django.utils.translation import gettext as _
from zerver.models import (
    CustomProfileField,
    CustomProfileFieldValue,
    Realm,
    UserProfile,
    Service,
    Subscription,
    Recipient,
    UserMessage,
    Message,
)
from zerver.models.users import RawUserDict
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)
from zerver.lib.types import ProfileDataElementUpdateDict, ProfileDataElementValue

class Account(TypedDict):
    pass

class APIUserDict(TypedDict):
    email: str
    user_id: int
    avatar_version: int
    is_admin: bool
    is_owner: bool
    is_guest: bool
    is_billing_admin: bool
    role: int
    is_bot: bool
    full_name: str
    timezone: str
    is_active: bool
    date_joined: str
    delivery_email: Optional[str]
    avatar_url: Optional[str]
    bot_type: Optional[int]
    bot_owner_id: Optional[int]
    is_system_bot: Optional[bool]
    profile_data: Optional[Dict[str, Dict[str, Union[str, Optional[str]]]]]

def check_full_name(full_name_raw: str, user_profile: UserProfile, realm: Realm) -> str: ...

def check_bot_name_available(realm_id: int, full_name: str, is_activation: bool) -> None: ...

def check_short_name(short_name_raw: str) -> str: ...

def check_valid_bot_config(bot_type: int, service_name: str, config_data: Dict[str, Any]) -> None: ...

def add_service(name: str, user_profile: UserProfile, base_url: str, interface: str, token: str) -> None: ...

def check_can_create_bot(user_profile: UserProfile, bot_type: int) -> None: ...

def check_valid_bot_type(user_profile: UserProfile, bot_type: int) -> None: ...

def is_administrator_role(role: int) -> bool: ...

def bulk_get_cross_realm_bots() -> Dict[str, UserProfile]: ...

def user_ids_to_users(user_ids: List[int], realm: Realm, allow_deactivated: bool) -> List[UserProfile]: ...

def access_bot_by_id(user_profile: UserProfile, user_id: int) -> UserProfile: ...

def access_user_common(target: UserProfile, user_profile: UserProfile, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> UserProfile: ...

def access_user_by_id(user_profile: UserProfile, target_user_id: int, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> UserProfile: ...

def access_user_by_id_including_cross_realm(user_profile: UserProfile, target_user_id: int, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> UserProfile: ...

def access_user_by_email(user_profile: UserProfile, email: str, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> UserProfile: ...

def bulk_access_users_by_email(emails: List[str], acting_user: UserProfile, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> Set[UserProfile]: ...

def bulk_access_users_by_id(user_ids: List[int], acting_user: UserProfile, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> Set[UserProfile]: ...

def get_accounts_for_email(email: str) -> List[Dict[str, Union[str, int]]]: ...

def validate_user_custom_profile_field(realm_id: int, field: CustomProfileField, value: str) -> None: ...

def validate_user_custom_profile_data(realm_id: int, profile_data: List[Dict[str, Union[str, int]]], acting_user: UserProfile) -> None: ...

def user_access_restricted_in_realm(target_user: UserProfile) -> bool: ...

def check_user_can_access_all_users(acting_user: Optional[UserProfile]) -> bool: ...

def check_can_access_user(target_user: UserProfile, user_profile: Optional[UserProfile]) -> bool: ...

def get_inaccessible_user_ids(target_user_ids: List[int], acting_user: UserProfile) -> Set[int]: ...

def get_user_ids_who_can_access_user(target_user: UserProfile) -> List[int]: ...

def get_subscribers_of_target_user_subscriptions(target_users: List[UserProfile], include_deactivated_users_for_dm_groups: bool) -> Dict[int, Set[int]]: ...

def get_users_involved_in_dms_with_target_users(target_users: List[UserProfile], realm: Realm, include_deactivated_users: bool) -> Dict[int, Set[int]]: ...

def user_profile_to_user_row(user_profile: UserProfile) -> RawUserDict: ...

def get_cross_realm_dicts() -> List[Dict[str, Union[str, int]]]: ...

def get_data_for_inaccessible_user(realm: Realm, user_id: int) -> APIUserDict: ...

def get_accessible_user_ids(realm: Realm, user_profile: UserProfile, include_deactivated_users: bool) -> List[int]: ...

def get_user_dicts_in_realm(realm: Realm, user_profile: UserProfile) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ...

def get_custom_profile_field_values(custom_profile_field_values: Iterable[CustomProfileFieldValue]) -> Dict[int, Dict[str, Dict[str, Union[str, Optional[str]]]]]: ...

def get_users_for_api(realm: Realm, acting_user: UserProfile, target_user: Optional[UserProfile], client_gravatar: bool, user_avatar_url_field_optional: bool, include_custom_profile_fields: bool, user_list_incomplete: bool) -> Dict[int, Dict[str, Any]]: ...

def get_active_bots_owned_by_user(user_profile: UserProfile) -> QuerySet[UserProfile]: ...

def is_2fa_verified(user: UserProfile) -> bool: ...

def get_users_with_access_to_real_email(user_profile: UserProfile) -> List[int]: ...

def max_message_id_for_user(user_profile: Optional[UserProfile]) -> int: ...