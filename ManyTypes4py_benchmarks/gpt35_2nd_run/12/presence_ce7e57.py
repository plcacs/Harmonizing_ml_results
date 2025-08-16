from typing import Any, Dict, List, Tuple
from django.conf import settings
from django.utils.timezone import now as timezone_now
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.users import check_user_can_access_all_users, get_accessible_user_ids
from zerver.models import Realm, UserPresence, UserProfile

def get_presence_dicts_for_rows(all_rows: List[Dict[str, Any]], slim_presence: bool) -> Dict[str, Dict[str, Any]]:
    ...

def user_presence_datetime_with_date_joined_default(dt: Any, date_joined: Any) -> Any:
    ...

def get_modern_user_presence_info(last_active_time: Any, last_connected_time: Any) -> Dict[str, Any]:
    ...

def get_legacy_user_presence_info(last_active_time: Any, last_connected_time: Any) -> Dict[str, Any]:
    ...

def format_legacy_presence_dict(last_active_time: Any, last_connected_time: Any) -> Dict[str, Any]:
    ...

def get_presence_for_user(user_profile_id: int, slim_presence: bool = False) -> Dict[str, Dict[str, Any]]:
    ...

def get_presence_dict_by_realm(realm: Realm, slim_presence: bool = False, last_update_id_fetched_by_client: int = None, history_limit_days: int = None, requesting_user_profile: UserProfile = None) -> Tuple[Dict[str, Dict[str, Any]], int]:
    ...

def get_presences_for_realm(realm: Realm, slim_presence: bool, last_update_id_fetched_by_client: int, history_limit_days: int, requesting_user_profile: UserProfile) -> Tuple[Dict[str, Dict[str, Any]], int]:
    ...

def get_presence_response(requesting_user_profile: UserProfile, slim_presence: bool, last_update_id_fetched_by_client: int = None, history_limit_days: int = None) -> Dict[str, Any]:
    ...
