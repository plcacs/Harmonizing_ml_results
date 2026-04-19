from typing import Any, Iterable, Mapping, Optional, Sequence, TypedDict
from typing_extensions import NotRequired
from django.db.models import QuerySet
from zerver.lib.types import ProfileDataElementUpdateDict, ProfileDataElementValue, RawUserDict
from zerver.models import (
    CustomProfileField,
    CustomProfileFieldValue,
    Message,
    Realm,
    Recipient,
    Service,
    Subscription,
    UserMessage,
    UserProfile,
)


class Account(TypedDict):
    realm_name: str
    realm_id: int
    full_name: str
    avatar: str


def check_full_name(
    full_name_raw: str, *, user_profile: Optional[UserProfile], realm: Realm
) -> str: ...


def check_bot_name_available(realm_id: int, full_name: str, *, is_activation: bool) -> None: ...


def check_short_name(short_name_raw: str) -> str: ...


def check_valid_bot_config(bot_type: int, service_name: str, config_data: Mapping[str, Any]) -> None: ...


def add_service(
    name: str, user_profile: UserProfile, base_url: str, interface: int, token: str
) -> None: ...


def check_can_create_bot(user_profile: UserProfile, bot_type: int) -> None: ...


def check_valid_bot_type(user_profile: UserProfile, bot_type: int) -> None: ...


def check_valid_interface_type(interface_type: int) -> None: ...


def is_administrator_role(role: int) -> bool: ...


def bulk_get_cross_realm_bots() -> dict[str, UserProfile]: ...


def user_ids_to_users(
    user_ids: Iterable[int], realm: Realm, *, allow_deactivated: bool
) -> list[UserProfile]: ...


def access_bot_by_id(user_profile: UserProfile, user_id: int) -> UserProfile: ...


def access_user_common(
    target: UserProfile,
    user_profile: Optional[UserProfile],
    allow_deactivated: bool,
    allow_bots: bool,
    for_admin: bool,
) -> UserProfile: ...


def access_user_by_id(
    user_profile: UserProfile,
    target_user_id: int,
    *,
    allow_deactivated: bool = False,
    allow_bots: bool = False,
    for_admin: bool,
) -> UserProfile: ...


def access_user_by_id_including_cross_realm(
    user_profile: UserProfile,
    target_user_id: int,
    *,
    allow_deactivated: bool = False,
    allow_bots: bool = False,
    for_admin: bool,
) -> UserProfile: ...


def access_user_by_email(
    user_profile: UserProfile,
    email: str,
    *,
    allow_deactivated: bool = False,
    allow_bots: bool = False,
    for_admin: bool,
) -> UserProfile: ...


def bulk_access_users_by_email(
    emails: Iterable[str],
    *,
    acting_user: UserProfile,
    allow_deactivated: bool = False,
    allow_bots: bool = False,
    for_admin: bool,
) -> set[UserProfile]: ...


def bulk_access_users_by_id(
    user_ids: Iterable[int],
    *,
    acting_user: UserProfile,
    allow_deactivated: bool = False,
    allow_bots: bool = False,
    for_admin: bool,
) -> set[UserProfile]: ...


def get_accounts_for_email(email: str) -> list[Account]: ...


def validate_user_custom_profile_field(
    realm_id: int, field: CustomProfileField, value: ProfileDataElementValue
) -> None: ...


def validate_user_custom_profile_data(
    realm_id: int, profile_data: Sequence[ProfileDataElementUpdateDict], acting_user: UserProfile
) -> None: ...


def can_access_delivery_email(
    user_profile: UserProfile, target_user_id: int, email_address_visibility: int
) -> bool: ...


class APIUserDict(TypedDict):
    email: str
    user_id: int
    avatar_version: int
    is_admin: bool
    is_owner: bool
    is_guest: bool
    role: int
    is_bot: bool
    full_name: str
    is_active: bool
    date_joined: str
    delivery_email: Optional[str]
    is_billing_admin: NotRequired[bool]
    timezone: NotRequired[str]
    avatar_url: NotRequired[str]
    bot_type: NotRequired[Optional[int]]
    is_system_bot: NotRequired[bool]
    bot_owner_id: NotRequired[Optional[int]]
    profile_data: NotRequired[dict[str, dict[str, str]]]


def format_user_row(
    realm_id: int,
    acting_user: Optional[UserProfile],
    row: RawUserDict,
    client_gravatar: bool,
    user_avatar_url_field_optional: bool,
    custom_profile_field_data: Optional[Mapping[str, Mapping[str, str]]] = ...,
) -> APIUserDict: ...


def user_access_restricted_in_realm(target_user: UserProfile) -> bool: ...


def check_user_can_access_all_users(acting_user: Optional[UserProfile]) -> bool: ...


def check_can_access_user(target_user: UserProfile, user_profile: Optional[UserProfile] = ...) -> bool: ...


def get_inaccessible_user_ids(
    target_user_ids: Iterable[int], acting_user: Optional[UserProfile]
) -> set[int]: ...


def get_user_ids_who_can_access_user(target_user: UserProfile) -> list[int]: ...


def get_subscribers_of_target_user_subscriptions(
    target_users: Iterable[UserProfile], include_deactivated_users_for_dm_groups: bool = ...
) -> dict[int, set[int]]: ...


def get_users_involved_in_dms_with_target_users(
    target_users: Iterable[UserProfile], realm: Realm, include_deactivated_users: bool = ...
) -> dict[int, set[int]]: ...


def user_profile_to_user_row(user_profile: UserProfile) -> RawUserDict: ...


def get_cross_realm_dicts() -> list[APIUserDict]: ...


def get_data_for_inaccessible_user(realm: Realm, user_id: int) -> APIUserDict: ...


def get_accessible_user_ids(
    realm: Realm, user_profile: UserProfile, include_deactivated_users: bool = ...
) -> list[int]: ...


def get_user_dicts_in_realm(
    realm: Realm, user_profile: Optional[UserProfile]
) -> tuple[list[RawUserDict], list[APIUserDict]]: ...


def get_custom_profile_field_values(
    custom_profile_field_values: Iterable[CustomProfileFieldValue],
) -> dict[int, dict[str, dict[str, str]]]: ...


def get_users_for_api(
    realm: Realm,
    acting_user: Optional[UserProfile],
    *,
    target_user: Optional[UserProfile] = ...,
    client_gravatar: bool,
    user_avatar_url_field_optional: bool,
    include_custom_profile_fields: bool = ...,
    user_list_incomplete: bool = ...,
) -> dict[int, APIUserDict]: ...


def get_active_bots_owned_by_user(user_profile: UserProfile) -> QuerySet[UserProfile]: ...


def is_2fa_verified(user: Any) -> bool: ...


def get_users_with_access_to_real_email(user_profile: UserProfile) -> list[int]: ...


def max_message_id_for_user(user_profile: Optional[UserProfile]) -> int: ...