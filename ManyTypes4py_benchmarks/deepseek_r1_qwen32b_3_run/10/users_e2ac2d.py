import itertools
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from email.headerregistry import Address
from operator import itemgetter
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import Q, QuerySet
from django.db.models.functions import Upper
from django.utils.translation import gettext as _
from django_otp.middleware import is_verified
from typing_extensions import NotRequired
from zulip_bots.custom_exceptions import ConfigValidationError
from zerver.lib.avatar import avatar_url, get_avatar_field, get_avatar_for_inaccessible_user
from zerver.lib.cache import cache_with_key, get_cross_realm_dicts_key
from zerver.lib.create_user import get_dummy_email_address_for_display_regex
from zerver.lib.exceptions import JsonableError, OrganizationOwnerRequiredError
from zerver.lib.string_validation import check_string_is_printable
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.lib.timezone import canonicalize_timezone
from zerver.lib.types import ProfileDataElementUpdateDict, ProfileDataElementValue, RawUserDict
from zerver.lib.user_groups import user_has_permission_for_group_setting
from zerver.models import CustomProfileField, CustomProfileFieldValue, Message, Realm, Recipient, Service, Subscription, UserMessage, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realms import get_fake_email_domain, require_unique_names
from zerver.models.users import active_non_guest_user_ids, active_user_ids, base_bulk_get_user_queryset, base_get_user_queryset, get_realm_user_dicts, get_user_by_id_in_realm_including_cross_realm, get_user_profile_by_id_in_realm, is_cross_realm_bot_email

def check_full_name(full_name_raw: str, *, user_profile: UserProfile, realm: Realm) -> str:
    ...

def check_bot_name_available(realm_id: int, full_name: str, *, is_activation: bool) -> None:
    ...

def check_short_name(short_name_raw: str) -> str:
    ...

def check_valid_bot_config(bot_type: str, service_name: str, config_data: Dict[str, Any]) -> None:
    ...

def add_service(name: str, user_profile: UserProfile, base_url: str, interface: str, token: str) -> None:
    ...

def check_can_create_bot(user_profile: UserProfile, bot_type: str) -> None:
    ...

def check_valid_bot_type(user_profile: UserProfile, bot_type: str) -> None:
    ...

def check_valid_interface_type(interface_type: str) -> None:
    ...

def is_administrator_role(role: str) -> bool:
    ...

def bulk_get_cross_realm_bots() -> Dict[str, UserProfile]:
    ...

def user_ids_to_users(user_ids: List[int], realm: Realm, *, allow_deactivated: bool) -> List[UserProfile]:
    ...

def access_bot_by_id(user_profile: UserProfile, user_id: int) -> UserProfile:
    ...

def access_user_common(target: UserProfile, user_profile: UserProfile, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> UserProfile:
    ...

def access_user_by_id(user_profile: UserProfile, target_user_id: int, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool = False) -> UserProfile:
    ...

def access_user_by_id_including_cross_realm(user_profile: UserProfile, target_user_id: int, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool = False) -> UserProfile:
    ...

def access_user_by_email(user_profile: UserProfile, email: str, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool = False) -> UserProfile:
    ...

def bulk_access_users_by_email(emails: List[str], *, acting_user: UserProfile, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool = False) -> Set[UserProfile]:
    ...

def bulk_access_users_by_id(user_ids: List[int], *, acting_user: UserProfile, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool = False) -> Set[UserProfile]:
    ...

class Account(TypedDict):
    ...

def get_accounts_for_email(email: str) -> List[Dict[str, Union[str, int]]]:
    ...

def validate_user_custom_profile_field(realm_id: int, field: CustomProfileField, value: Any) -> None:
    ...

def validate_user_custom_profile_data(realm_id: int, profile_data: List[Dict[str, Any]], acting_user: UserProfile) -> None:
    ...

def can_access_delivery_email(user_profile: UserProfile, target_user_id: int, email_address_visibility: int) -> bool:
    ...

class APIUserDict(TypedDict):
    ...

def format_user_row(realm_id: int, acting_user: Optional[UserProfile], row: Dict[str, Any], client_gravatar: bool, user_avatar_url_field_optional: bool, custom_profile_field_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ...

def user_access_restricted_in_realm(target_user: UserProfile) -> bool:
    ...

def check_user_can_access_all_users(acting_user: Optional[UserProfile]) -> bool:
    ...

def check_can_access_user(target_user: UserProfile, user_profile: Optional[UserProfile] = None) -> bool:
    ...

def get_inaccessible_user_ids(target_user_ids: List[int], acting_user: UserProfile) -> Set[int]:
    ...

def get_user_ids_who_can_access_user(target_user: UserProfile) -> List[int]:
    ...

def get_subscribers_of_target_user_subscriptions(target_users: List[UserProfile], include_deactivated_users_for_dm_groups: bool = False) -> defaultdict[int, Set[int]]:
    ...

def get_users_involved_in_dms_with_target_users(target_users: List[UserProfile], realm: Realm, include_deactivated_users: bool = False) -> defaultdict[int, Set[int]]:
    ...

def user_profile_to_user_row(user_profile: UserProfile) -> RawUserDict:
    ...

@cache_with_key(get_cross_realm_dicts_key)
def get_cross_realm_dicts() -> List[Dict[str, Any]]:
    ...

def get_data_for_inaccessible_user(realm: Realm, user_id: int) -> Dict[str, Any]:
    ...

def get_accessible_user_ids(realm: Realm, user_profile: UserProfile, include_deactivated_users: bool = False) -> List[int]:
    ...

def get_user_dicts_in_realm(realm: Realm, user_profile: UserProfile) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ...

def get_custom_profile_field_values(custom_profile_field_values: QuerySet[CustomProfileFieldValue]) -> defaultdict[int, Dict[str, Dict[str, Any]]]:
    ...

def get_users_for_api(realm: Realm, acting_user: Optional[UserProfile], *, target_user: Optional[UserProfile] = None, client_gravatar: bool, user_avatar_url_field_optional: bool, include_custom_profile_fields: bool = True, user_list_incomplete: bool = False) -> Dict[int, Dict[str, Any]]:
    ...

def get_active_bots_owned_by_user(user_profile: UserProfile) -> QuerySet[UserProfile]:
    ...

def is_2fa_verified(user: UserProfile) -> bool:
    ...

def get_users_with_access_to_real_email(user_profile: UserProfile) -> List[int]:
    ...

def max_message_id_for_user(user_profile: Optional[UserProfile]) -> int:
    ...