from typing import TypedDict, Any, Mapping, Sequence, Dict, List, Set, DefaultDict
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import re
import unicodedata
from django.conf import settings
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
from zerver.models.users import active_non_guest_user_ids, active_user_ids, base_bulk_get_user_queryset, base_get_user_queryset, get_realm_user_dicts, get_user_by_id_in_realm_including_cross_realm, get_user_profile_by_id_in_realm
from zerver.models.users import access_bot_by_id, access_user_common, access_user_by_id, access_user_by_id_including_cross_realm, access_user_by_email, bulk_access_users_by_email, bulk_access_users_by_id
from zerver.models.users import check_can_create_bot, check_valid_bot_type, check_valid_interface_type, is_administrator_role, get_inaccessible_user_ids, get_user_ids_who_can_access_user, get_subscribers_of_target_user_subscriptions, get_users_involved_in_dms_with_target_users
from zerver.models.users import get_cross_realm_dicts, get_data_for_inaccessible_user, get_users_for_api, get_active_bots_owned_by_user, is_2fa_verified, get_users_with_access_to_real_email, max_message_id_for_user

class Account(TypedDict):
    pass

def get_accounts_for_email(email: str) -> List[Dict[str, Any]]:
    # ...

def format_user_row(realm_id: int, acting_user: UserProfile, row: Mapping[str, Any], client_gravatar: bool, user_avatar_url_field_optional: bool, custom_profile_field_data: Dict[str, Any]) -> Dict[str, Any]:
    # ...

def get_cross_realm_dicts() -> Dict[int, List[Dict[str, Any]]]:
    # ...

def get_users_for_api(realm: Realm, acting_user: UserProfile, target_user: UserProfile = None, client_gravatar: bool = False, user_avatar_url_field_optional: bool = False, include_custom_profile_fields: bool = True, user_list_incomplete: bool = False) -> Dict[int, Dict[str, Any]]:
    # ...

def get_active_bots_owned_by_user(user_profile: UserProfile) -> List[UserProfile]:
    # ...

def is_2fa_verified(user: UserProfile) -> bool:
    # ...

def get_users_with_access_to_real_email(user_profile: UserProfile) -> List[int]:
    # ...

def max_message_id_for_user(user_profile: UserProfile) -> int:
    # ...
