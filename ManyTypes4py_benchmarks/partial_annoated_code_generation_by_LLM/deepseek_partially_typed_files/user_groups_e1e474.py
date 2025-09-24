from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, TypedDict, Optional, Union, Dict, List, Set, ContextManager
from django.db import connection, transaction
from django.db.models import F, Q, QuerySet
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django_cte import With
from psycopg2.sql import SQL, Literal
from zerver.lib.exceptions import CannotDeactivateGroupInUseError, JsonableError, PreviousSettingValueMismatchedError, SystemGroupRequiredError
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.types import AnonymousSettingGroupDict, GroupPermissionSetting, ServerSupportedPermissionSettings
from zerver.models import GroupGroupMembership, NamedUserGroup, Realm, RealmAuditLog, Stream, UserGroup, UserGroupMembership, UserProfile
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType

@dataclass
class GroupSettingChangeRequest:
    new: Union[int, AnonymousSettingGroupDict]
    old: Optional[Union[int, AnonymousSettingGroupDict]] = None

class UserGroupDict(TypedDict):
    id: int
    name: str
    description: str
    members: List[int]
    direct_subgroup_ids: List[int]
    creator_id: Optional[int]
    date_created: Optional[int]
    is_system_group: bool
    can_add_members_group: Union[int, AnonymousSettingGroupDict]
    can_join_group: Union[int, AnonymousSettingGroupDict]
    can_leave_group: Union[int, AnonymousSettingGroupDict]
    can_manage_group: Union[int, AnonymousSettingGroupDict]
    can_mention_group: Union[int, AnonymousSettingGroupDict]
    can_remove_members_group: Union[int, AnonymousSettingGroupDict]
    deactivated: bool

@dataclass
class LockedUserGroupContext:
    """User groups in this dataclass are guaranteeed to be locked until the
    end of the current transaction.

    supergroup is the user group to have subgroups added or removed;
    direct_subgroups are user groups that are recursively queried for subgroups;
    recursive_subgroups include direct_subgroups and their descendants.
    """
    supergroup: NamedUserGroup
    direct_subgroups: List[NamedUserGroup]
    recursive_subgroups: List[NamedUserGroup]

def has_user_group_access_for_subgroup(user_group: NamedUserGroup, user_profile: UserProfile, *, allow_deactivated: bool = False) -> bool:
    """Minimal access control checks for whether the given group
    is visible to the given user for use as a subgroup.

    In the future, if groups whose existence is not visible to the
    entire organization are added, this may grow more complex.
    """
    if user_group.realm_id != user_profile.realm_id:
        return False
    if not allow_deactivated and user_group.deactivated:
        raise JsonableError(_('User group is deactivated.'))
    return True

def get_user_group_by_id_in_realm(user_group_id: int, realm: Realm, *, for_read: bool, for_setting: bool = False, allow_deactivated: bool = False) -> NamedUserGroup:
    """
    Internal function for accessing a single user group from client
    code. Locks the group if for_read is False.

    Notably does not do any access control checks, beyond only fetching
    groups from the provided realm.
    """
    try:
        if for_read:
            user_group: NamedUserGroup = NamedUserGroup.objects.get(id=user_group_id, realm=realm)
        else:
            user_group: NamedUserGroup = NamedUserGroup.objects.select_for_update().get(id=user_group_id, realm=realm)
        if not allow_deactivated and user_group.deactivated:
            raise JsonableError(_('User group is deactivated.'))
        return user_group
    except NamedUserGroup.DoesNotExist:
        raise JsonableError(_('Invalid user group'))

def access_user_group_to_read_membership(user_group_id: int, realm: Realm) -> NamedUserGroup:
    return get_user_group_by_id_in_realm(user_group_id, realm, for_read=True)

def access_user_group_for_update(user_group_id: int, user_profile: UserProfile, *, permission_setting: str, allow_deactivated: bool = False) -> NamedUserGroup:
    """
    Main entry point that views code should call when planning to modify
    a given user group on behalf of a given user.

    The permission_setting parameter indicates what permission to check;
    different features will be used when editing the membership vs.
    security-sensitive settings on a group.
    """
    user_group: NamedUserGroup = get_user_group_by_id_in_realm(user_group_id, user_profile.realm, for_read=False, allow_deactivated=allow_deactivated)
    if user_group.is_system_group:
        raise JsonableError(_('Insufficient permission'))
    if user_profile.can_manage_all_groups():
        return user_group
    user_has_permission: bool = user_has_permission_for_group_setting(user_group.can_manage_group, user_profile, NamedUserGroup.GROUP_PERMISSION_SETTINGS['can_manage_group'])
    if user_has_permission:
        return user_group
    if permission_setting != 'can_manage_group':
        assert permission_setting in NamedUserGroup.GROUP_PERMISSION_SETTINGS
        user_has_permission = user_has_permission_for_group_setting(getattr(user_group, permission_setting), user_profile, NamedUserGroup.GROUP_PERMISSION_SETTINGS[permission_setting])
        if user_has_permission:
            return user_group
    raise JsonableError(_('Insufficient permission'))

def access_user_group_for_deactivation(user_group_id: int, user_profile: UserProfile) -> NamedUserGroup:
    """
    Main security check / access function for whether the acting
    user has permission to deactivate a given user group.
    """
    user_group: NamedUserGroup = access_user_group_for_update(user_group_id, user_profile, permission_setting='can_manage_group')
    objections: List[Dict[str, Any]] = []
    supergroup_ids: QuerySet = user_group.direct_supergroups.exclude(named_user_group=None).filter(named_user_group__deactivated=False).values_list('id', flat=True)
    if supergroup_ids:
        objections.append(dict(type='subgroup', supergroup_ids=list(supergroup_ids)))
    anonymous_supergroup_ids: QuerySet = user_group.direct_supergroups.filter(named_user_group=None).values_list('id', flat=True)
    setting_group_ids_using_deactivating_user_group: Set[int] = {*set(anonymous_supergroup_ids), user_group.id}
    stream_setting_query: Q = Q()
    for setting_name in Stream.stream_permission_group_settings:
        stream_setting_query |= Q(**{f'{setting_name}__in': setting_group_ids_using_deactivating_user_group})
    for stream in Stream.objects.filter(realm_id=user_group.realm_id, deactivated=False).filter(stream_setting_query):
        objection_settings: List[str] = [setting_name for setting_name in Stream.stream_permission_group_settings if getattr(stream, setting_name + '_id') in setting_group_ids_using_deactivating_user_group]
        if len(objection_settings) > 0:
            objections.append(dict(type='channel', channel_id=stream.id, settings=objection_settings))
    group_setting_query: Q = Q()
    for setting_name in NamedUserGroup.GROUP_PERMISSION_SETTINGS:
        group_setting_query |= Q(**{f'{setting_name}__in': setting_group_ids_using_deactivating_user_group})
    for group in NamedUserGroup.objects.filter(realm_id=user_group.realm_id, deactivated=False).filter(group_setting_query):
        objection_settings = []
        for setting_name in NamedUserGroup.GROUP_PERMISSION_SETTINGS:
            if getattr(group, setting_name + '_id') in setting_group_ids_using_deactivating_user_group:
                objection_settings.append(setting_name)
        if len(objection_settings) > 0:
            objections.append(dict(type='user_group', group_id=group.id, settings=objection_settings))
    objection_settings = []
    realm: Realm = user_group.realm
    for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
        if getattr(realm, setting_name + '_id') in setting_group_ids_using_deactivating_user_group:
            objection_settings.append(setting_name)
    if objection_settings:
        objections.append(dict(type='realm', settings=objection_settings))
    if len(objections) > 0:
        raise CannotDeactivateGroupInUseError(objections)
    return user_group

@contextmanager
def lock_subgroups_with_respect_to_supergroup(potential_subgroup_ids: List[int], potential_supergroup_id: int, acting_user: UserProfile, *, permission_setting: Optional[str] = None, creating_group: bool = False) -> ContextManager[LockedUserGroupContext]:
    """This locks the user groups with the given potential_subgroup_ids, as well
    as their indirect subgroups, followed by the potential supergroup. It
    ensures that we lock the user groups in a consistent order topologically to
    avoid unnecessary deadlocks on non-conflicting queries.

    Regardless of whether the user groups returned are used, always call this
    helper before making changes to subgroup memberships. This avoids
    introducing cycles among user groups when there is a race condition in
    which one of these subgroups become an ancestor of the parent user group in
    another transaction.

    Note that it only does a permission check on the potential supergroup,
    not the potential subgroups or their recursive subgroups.
    """
    with transaction.atomic(savepoint=False):
        recursive_subgroups: List[NamedUserGroup] = list(get_recursive_subgroups_for_groups(potential_subgroup_ids, acting_user.realm).select_for_update(nowait=True))
        if creating_group:
            potential_supergroup: NamedUserGroup = get_user_group_by_id_in_realm(potential_supergroup_id, acting_user.realm, for_read=False)
        else:
            assert permission_setting is not None
            potential_supergroup: NamedUserGroup = access_user_group_for_update(potential_supergroup_id, acting_user, permission_setting=permission_setting)
        potential_subgroups: List[NamedUserGroup] = [user_group for user_group in recursive_subgroups if user_group.id in potential_subgroup_ids]
        group_ids_found: List[int] = [group.id for group in potential_subgroups]
        group_ids_not_found: List[int] = [group_id for group_id in potential_subgroup_ids if group_id not in group_ids_found]
        if group_ids_not_found:
            raise JsonableError(_('Invalid user group ID: {group_id}').format(group_id=group_ids_not_found[0]))
        for subgroup in potential_subgroups:
            if not has_user_group_access_for_subgroup(subgroup, acting_user):
                raise JsonableError(_('Insufficient permission'))
        yield LockedUserGroupContext(direct_subgroups=potential_subgroups, recursive_subgroups=recursive_subgroups, supergroup=potential_supergroup)

def check_setting_configuration_for_system_groups(setting_group: NamedUserGroup, setting_name: str, permission_configuration: GroupPermissionSetting) -> None:
    if permission_configuration.require_system_group and (not setting_group.is_system_group):
        raise SystemGroupRequiredError(setting_name)
    if not permission_configuration.allow_internet_group and setting_group.name == SystemGroups.EVERYONE_ON_INTERNET:
        raise JsonableError(_("'{setting_name}' setting cannot be set to 'role:internet' group.").format(setting_name=setting_name))
    if not permission_configuration.allow_nobody_group and setting_group.name == SystemGroups.NOBODY:
        raise JsonableError(_("'{setting_name}' setting cannot be set to 'role:nobody' group.").format(setting_name=setting_name))
    if not permission_configuration.allow_everyone_group and setting_group.name == SystemGroups.EVERYONE:
        raise JsonableError(_("'{setting_name}' setting cannot be set to 'role:everyone' group.").format(setting_name=setting_name))
    if permission_configuration.allowed_system_groups and setting_group.name not in permission_configuration.allowed_system_groups:
        raise JsonableError(_("'{setting_name}' setting cannot be set to '{group_name}' group.").format(setting_name=setting_name, group_name=setting_group.name))

def update_or_create_user_group_for_setting(user_profile: UserProfile, direct_members: List[int], direct_subgroups: List[int], current_setting_value: Optional[UserGroup]) -> UserGroup:
    realm: Realm = user_profile.realm
    if current_setting_value is not None and (not hasattr(current_setting_value, 'named_user_group')):
        user_group: UserGroup = current_setting_value
    else:
        user_group: UserGroup = UserGroup.objects.create(realm=realm)
    from zerver.lib.users import user_ids_to_users
    member_users: List[UserProfile] = user_ids_to_users(direct_members, realm, allow_deactivated=False)
    user_group.direct_members.set(member_users)
    potential_subgroups: QuerySet[NamedUserGroup] = NamedUserGroup.objects.select_for_update().filter(realm=realm, id__in=direct_subgroups)
    group_ids_found: List[int] = [group.id for group in potential_subgroups]
    group_ids_not_found: List[int] = [group_id for group_id in direct_subgroups if group_id not in group_ids_found]
    if group_ids_not_found:
        raise JsonableError(_('Invalid user group ID: {group_id}').format(group_id=group_ids_not_found[0]))
    for subgroup in potential_subgroups:
        if not has_user_group_access_for_subgroup(subgroup, user_profile):
            raise JsonableError(_('Insufficient permission'))
    user_group.direct_subgroups.set(group_ids_found)
    return user_group

def access_user_group_for_setting(setting_user_group: Union[int, AnonymousSettingGroupDict], user_profile: UserProfile, *, setting_name: str, permission_configuration: GroupPermissionSetting, current_setting_value: Optional[UserGroup] = None) -> UserGroup:
    """Given a permission setting and specification of what value it
    should have (setting_user_group), returns either a Named or
    anonymous `UserGroup` with the requested membership.
    """
    if isinstance(setting_user_group, int):
        named_user_group: NamedUserGroup = get_user_group_by_id_in_realm(setting_user_group, user_profile.realm, for_read=False, for_setting=True)
        check_setting_configuration_for_system_groups(named_user_group, setting_name, permission_configuration)
        return named_user_group.usergroup_ptr
    if permission_configuration.require_system_group:
        raise SystemGroupRequiredError(setting_name)
    user_group: UserGroup = update_or_create_user_group_for_setting(user_profile, setting_user_group.direct_members, setting_user_group.direct_subgroups, current_setting_value)
    return user_group

def check_user_group_name(group_name: str) -> str:
    if group_name.strip() == '':
        raise JsonableError(_("User group name can't be empty!"))
    if len(group_name) > NamedUserGroup.MAX_NAME_LENGTH:
        raise JsonableError(_('User group name cannot exceed {max_length} characters.').format(max_length=NamedUserGroup.MAX_NAME_LENGTH))
    for invalid_prefix in NamedUserGroup.INVALID_NAME_PREFIXES:
        if group_name.startswith(invalid_prefix):
            raise JsonableError(_("User group name cannot start with '{prefix}'.").format(prefix=invalid_prefix))
    return group_name

def get_group_setting_value_for_api(setting_value_group: UserGroup) -> Union[int, AnonymousSettingGroupDict]:
    if hasattr(setting_value_group, 'named_user_group'):
        return setting_value_group.id
    return AnonymousSettingGroupDict(direct_members=[member.id for member in setting_value_group.direct_members.filter(is_active=True)], direct_subgroups=[subgroup.id for subgroup in setting_value_group.direct_subgroups.all()])

def get_setting_value_for_user_group_object(setting_value_group: UserGroup, direct_members_dict: Dict[int, List[int]], direct_subgroups_dict: Dict[int, List[int]]) -> Union[int, AnonymousSettingGroupDict]:
    if hasattr(setting_value_group, 'named_user_group'):
        return setting_value_group.id
    direct_members: List[int] = []
    if setting_value_group.id in direct_members_dict:
        direct_members = direct_members_dict[setting_value_group.id]
    direct_subgroups: List[int] = []
    if setting_value_group.id in direct_subgroups_dict:
        direct_subgroups = direct_subgroups_dict[setting_value_group.id]
    return AnonymousSettingGroupDict(direct_members=direct_members, direct_subgroups=direct_subgroups)

def user_groups_in_realm_serialized(realm: Realm, *, include_deactivated_groups: bool) -> List[UserGroupDict]:
    """This function is used in do_events_register code path so this code
    should be performant.  We need to do 2 database queries because
    Django's ORM doesn't properly support the left join between
    UserGroup and UserGroupMembership that we need.
    """
    realm_groups: QuerySet[NamedUserGroup] = NamedUserGroup.objects.select_related('can_add_members_group', 'can_add_members_group__named_user_group', 'can_join_group', 'can_join_group__named_user_group', 'can_leave_group', 'can_leave_group__named_user_group', 'can_manage_group', 'can_manage_group__named_user_group', 'can_mention_group', 'can_mention_group__named_user_group', 'can_remove_members_group', 'can_remove_members_group__named_user_group').filter(realm=realm)
    if not include_deactivated_groups:
        realm_groups = realm_groups.filter(deactivated=False)
    membership: QuerySet = UserGroupMembership.objects.filter(user_group__realm=realm).exclude(user_profile__is_active=False).values_list('user_group_id', 'user_profile_id')
    group_membership: QuerySet = GroupGroupMembership.objects.filter(subgroup__realm=realm).values_list('subgroup_id', 'supergroup_id')
    group_m