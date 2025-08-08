from typing import Any, TypedDict, List, Dict, Set
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import Q, QuerySet
from zerver.models import UserProfile, CustomProfileField, CustomProfileFieldValue, Message, Subscription, UserMessage, Realm

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
    delivery_email: str

def get_accounts_for_email(email: str) -> List[Dict[str, Any]]:
    pass

def validate_user_custom_profile_field(realm_id: int, field: CustomProfileField, value: Any) -> ValidationError:
    pass

def validate_user_custom_profile_data(realm_id: int, profile_data: List[Dict[str, Any]], acting_user: UserProfile) -> None:
    pass

def get_data_for_inaccessible_user(realm: Realm, user_id: int) -> APIUserDict:
    pass

def get_users_for_api(realm: Realm, acting_user: UserProfile, target_user: UserProfile = None, client_gravatar: bool, user_avatar_url_field_optional: bool, include_custom_profile_fields: bool = True, user_list_incomplete: bool = False) -> Dict[int, APIUserDict]:
    pass

def get_active_bots_owned_by_user(user_profile: UserProfile) -> QuerySet:
    pass

def is_2fa_verified(user: UserProfile) -> bool:
    pass

def get_users_with_access_to_real_email(user_profile: UserProfile) -> List[int]:
    pass

def max_message_id_for_user(user_profile: UserProfile) -> int:
    pass
