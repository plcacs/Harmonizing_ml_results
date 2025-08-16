from typing import List, Dict, Any

def create_user_group_in_database(name: str, members: List[UserProfile], realm: Realm, *, acting_user: UserProfile, description: str = '', group_settings_map: Dict[str, Any] = {}, is_system_group: bool = False) -> NamedUserGroup:
def update_users_in_full_members_system_group(realm: Realm, affected_user_ids: List[int] = [], *, acting_user: UserProfile) -> None:
def check_add_user_group(realm: Realm, name: str, initial_members: List[UserProfile], description: str = '', group_settings_map: Dict[str, Any] = {}, *, acting_user: UserProfile) -> NamedUserGroup:
def do_update_user_group_name(user_group: NamedUserGroup, name: str, *, acting_user: UserProfile) -> None:
def do_update_user_group_description(user_group: NamedUserGroup, description: str, *, acting_user: UserProfile) -> None:
def bulk_add_members_to_user_groups(user_groups: List[NamedUserGroup], user_profile_ids: List[int], *, acting_user: UserProfile) -> None:
def bulk_remove_members_from_user_groups(user_groups: List[NamedUserGroup], user_profile_ids: List[int], *, acting_user: UserProfile) -> None:
def add_subgroups_to_user_group(user_group: NamedUserGroup, subgroups: List[NamedUserGroup], *, acting_user: UserProfile) -> None:
def remove_subgroups_from_user_group(user_group: NamedUserGroup, subgroups: List[NamedUserGroup], *, acting_user: UserProfile) -> None:
def do_deactivate_user_group(user_group: NamedUserGroup, *, acting_user: UserProfile) -> None:
def do_change_user_group_permission_setting(user_group: NamedUserGroup, setting_name: str, setting_value_group: AnonymousSettingGroupDict, *, old_setting_api_value: AnonymousSettingGroupDict = None, acting_user: UserProfile) -> None:
