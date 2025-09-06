from typing import Any, Dict, List, Tuple
from django.conf import settings
from django.utils.timezone import now as timezone_now
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.users import check_user_can_access_all_users, get_accessible_user_ids
from zerver.models import Realm, UserPresence, UserProfile

def func_0zr8n629(all_rows: List[Dict[str, Any]], slim_presence: bool) -> Dict[str, Dict[str, Any]]:
def func_nvjy4bri(dt: Any, date_joined: Any) -> Any:
def func_vryv3i17(last_active_time: datetime, last_connected_time: datetime) -> Dict[str, int]:
def func_tzp20bay(last_active_time: datetime, last_connected_time: datetime) -> Dict[str, Dict[str, Any]]:
def func_sjovzerc(last_active_time: datetime, last_connected_time: datetime) -> Dict[str, Any]:
def func_bjxdrq8q(user_profile_id: int, slim_presence: bool = False) -> Dict[str, Dict[str, Any]]:
def func_okz0pz7c(realm: Realm, slim_presence: bool = False, last_update_id_fetched_by_client: int = None, history_limit_days: int = None, requesting_user_profile: UserProfile = None) -> Tuple[Dict[str, Dict[str, Any]], int]:
def func_on1sau0r(realm: Realm, slim_presence: bool, last_update_id_fetched_by_client: int, history_limit_days: int, requesting_user_profile: UserProfile) -> Tuple[Dict[str, Dict[str, Any]], int]:
def func_l4k66qkt(requesting_user_profile: UserProfile, slim_presence: bool, last_update_id_fetched_by_client: int = None, history_limit_days: int = None) -> Dict[str, Any]:
