import itertools
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from email.headerregistry import Address
from operator import itemgetter
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict, Union
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

class Account(TypedDict):
    realm_name: str
    realm_id: int
    full_name: str
    avatar: str

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
    avatar_url: NotRequired[str]
    bot_type: NotRequired[int]
    is_system_bot: NotRequired[bool]
    bot_owner_id: NotRequired[int]
    profile_data: NotRequired[Dict[str, Dict[str, Any]]]

def check_full_name(full_name_raw: str, *, user_profile: Optional[UserProfile], realm: Realm) -> str:
    full_name = full_name_raw.strip()
    if len(full_name) > UserProfile.MAX_NAME_LENGTH:
        raise JsonableError(_('Name too long!'))
    if len(full_name) < UserProfile.MIN_NAME_LENGTH:
        raise JsonableError(_('Name too short!'))
    if check_string_is_printable(full_name) is not None or any((character in full_name for character in UserProfile.NAME_INVALID_CHARS)):
        raise JsonableError(_('Invalid characters in name!'))
    if re.search('\\|\\d+$', full_name_raw):
        raise JsonableError(_('Invalid format!'))
    if require_unique_names(realm):
        normalized_user_full_name = unicodedata.normalize('NFKC', full_name).casefold()
        users_query = UserProfile.objects.filter(realm=realm)
        if user_profile is not None:
            existing_names = users_query.exclude(id=user_profile.id).values_list('full_name', flat=True)
        else:
            existing_names = users_query.values_list('full_name', flat=True)
        normalized_existing_names = [unicodedata.normalize('NFKC', full_name).casefold() for full_name in existing_names]
        if normalized_user_full_name in normalized_existing_names:
            raise JsonableError(_('Unique names required in this organization.'))
    return full_name

def check_bot_name_available(realm_id: int, full_name: str, *, is_activation: bool) -> None:
    dup_exists = UserProfile.objects.filter(realm_id=realm_id, full_name=full_name.strip(), is_active=True).exists()
    if dup_exists:
        if is_activation:
            raise JsonableError(f'There is already an active bot named "{full_name}" in this organization. To reactivate this bot, you must rename or deactivate the other one first.')
        else:
            raise JsonableError(_('Name is already in use!'))

def check_short_name(short_name_raw: str) -> str:
    short_name = short_name_raw.strip()
    if len(short_name) == 0:
        raise JsonableError(_('Bad name or username'))
    return short_name

def check_valid_bot_config(bot_type: int, service_name: str, config_data: Dict[str, Any]) -> None:
    if bot_type == UserProfile.INCOMING_WEBHOOK_BOT:
        from zerver.lib.integrations import WEBHOOK_INTEGRATIONS
        config_options = None
        for integration in WEBHOOK_INTEGRATIONS:
            if integration.name == service_name:
                config_options = {option.name: option.validator for option in integration.config_options}
                break
        if not config_options:
            raise JsonableError(_("Invalid integration '{integration_name}'.").format(integration_name=service_name))
        missing_keys = set(config_options.keys()) - set(config_data.keys())
        if missing_keys:
            raise JsonableError(_('Missing configuration parameters: {keys}').format(keys=missing_keys))
        for key, validator in config_options.items():
            value = config_data[key]
            error = validator(key, value)
            if error is not None:
                raise JsonableError(_('Invalid {key} value {value} ({error})').format(key=key, value=value, error=error))
    elif bot_type == UserProfile.EMBEDDED_BOT:
        try:
            from zerver.lib.bot_lib import get_bot_handler
            bot_handler = get_bot_handler(service_name)
            if hasattr(bot_handler, 'validate_config'):
                bot_handler.validate_config(config_data)
        except ConfigValidationError:
            raise JsonableError(_('Invalid configuration data!'))

def add_service(name: str, user_profile: UserProfile, base_url: str, interface: int, token: str) -> None:
    Service.objects.create(name=name, user_profile=user_profile, base_url=base_url, interface=interface, token=token)

def check_can_create_bot(user_profile: UserProfile, bot_type: int) -> None:
    if user_has_permission_for_group_setting(user_profile.realm.can_create_bots_group, user_profile, Realm.REALM_PERMISSION_GROUP_SETTINGS['can_create_bots_group']):
        return
    if bot_type == UserProfile.INCOMING_WEBHOOK_BOT and user_has_permission_for_group_setting(user_profile.realm.can_create_write_only_bots_group, user_profile, Realm.REALM_PERMISSION_GROUP_SETTINGS['can_create_write_only_bots_group']):
        return
    raise JsonableError(_('Insufficient permission'))

def check_valid_bot_type(user_profile: UserProfile, bot_type: int) -> None:
    if bot_type not in user_profile.allowed_bot_types:
        raise JsonableError(_('Invalid bot type'))

def check_valid_interface_type(interface_type: int) -> None:
    if interface_type not in Service.ALLOWED_INTERFACE_TYPES:
        raise JsonableError(_('Invalid interface type'))

def is_administrator_role(role: int) -> bool:
    return role in {UserProfile.ROLE_REALM_ADMINISTRATOR, UserProfile.ROLE_REALM_OWNER}

def bulk_get_cross_realm_bots() -> Dict[str, UserProfile]:
    emails = list(settings.CROSS_REALM_BOT_EMAILS)
    where_clause = 'upper(zerver_userprofile.email::text) IN (SELECT upper(email) FROM unnest(%s) AS email)'
    users = UserProfile.objects.filter(realm__string_id=settings.SYSTEM_BOT_REALM).extra(where=[where_clause], params=(emails,))
    return {user.email.lower(): user for user in users}

def user_ids_to_users(user_ids: List[int], realm: Realm, *, allow_deactivated: bool) -> List[UserProfile]:
    user_query = UserProfile.objects.filter(id__in=user_ids, realm=realm)
    if not allow_deactivated:
        user_query = user_query.filter(is_active=True)
    user_profiles = list(user_query.select_related('realm'))
    found_user_ids = {user_profile.id for user_profile in user_profiles}
    for user_id in user_ids:
        if user_id not in found_user_ids:
            raise JsonableError(_('Invalid user ID: {user_id}').format(user_id=user_id))
    return user_profiles

def access_bot_by_id(user_profile: UserProfile, user_id: int) -> UserProfile:
    try:
        target = get_user_profile_by_id_in_realm(user_id, user_profile.realm)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such bot'))
    if not target.is_bot:
        raise JsonableError(_('No such bot'))
    if not user_profile.can_admin_user(target):
        raise JsonableError(_('Insufficient permission'))
    if target.can_create_users and (not user_profile.is_realm_owner):
        raise OrganizationOwnerRequiredError
    return target

def access_user_common(target: UserProfile, user_profile: UserProfile, allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> UserProfile:
    if target.is_bot and (not allow_bots):
        raise JsonableError(_('No such user'))
    if not target.is_active and (not allow_deactivated):
        raise JsonableError(_('User is deactivated'))
    if not for_admin:
        if not check_can_access_user(target, user_profile):
            raise JsonableError(_('Insufficient permission'))
        return target
    if not user_profile.can_admin_user(target):
        raise JsonableError(_('Insufficient permission'))
    return target

def access_user_by_id(user_profile: UserProfile, target_user_id: int, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> UserProfile:
    try:
        target = get_user_profile_by_id_in_realm(target_user_id, user_profile.realm)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such user'))
    return access_user_common(target, user_profile, allow_deactivated, allow_bots, for_admin)

def access_user_by_id_including_cross_realm(user_profile: UserProfile, target_user_id: int, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> UserProfile:
    try:
        target = get_user_by_id_in_realm_including_cross_realm(target_user_id, user_profile.realm)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such user'))
    return access_user_common(target, user_profile, allow_deactivated, allow_bots, for_admin)

def access_user_by_email(user_profile: UserProfile, email: str, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> UserProfile:
    dummy_email_regex = get_dummy_email_address_for_display_regex(user_profile.realm)
    match = re.match(dummy_email_regex, email)
    if match:
        target_id = int(match.group(1))
        return access_user_by_id(user_profile, target_id, allow_deactivated=allow_deactivated, allow_bots=allow_bots, for_admin=for_admin)
    allowed_email_address_visibility_values = UserProfile.ROLE_TO_ACCESSIBLE_EMAIL_ADDRESS_VISIBILITY_IDS[user_profile.role]
    try:
        target = base_get_user_queryset().get(delivery_email__iexact=email.strip(), realm=user_profile.realm, email_address_visibility__in=allowed_email_address_visibility_values)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such user'))
    return access_user_common(target, user_profile, allow_deactivated, allow_bots, for_admin)

def bulk_access_users_by_email(emails: List[str], *, acting_user: UserProfile, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> Set[UserProfile]:
    target_emails_upper = [email.strip().upper() for email in emails]
    users = base_bulk_get_user_queryset().annotate(email_upper=Upper('email')).filter(email_upper__in=target_emails_upper, realm=acting_user.realm)
    valid_emails_upper = {user_profile.email_upper for user_profile in users}
    all_users_exist = all((email in valid_emails_upper for email in target_emails_upper))
    if not all_users_exist:
        raise JsonableError(_('No such user'))
    return {access_user_common(user_profile, acting_user, allow_deactivated, allow_bots, for_admin) for user_profile in users}

def bulk_access_users_by_id(user_ids: List[int], *, acting_user: UserProfile, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> Set[UserProfile]:
    users = base_bulk_get_user_queryset().filter(id__in=user_ids, realm=acting_user.realm)
    valid_user_ids = {user_profile.id for user_profile in users}
    all_users_exist = all((user_id in valid_user_ids for user_id in user_ids))
    if not all_users_exist:
        raise JsonableError(_('No such user'))
    return {access_user_common(user_profile, acting_user, allow_deactivated, allow_bots, for_admin) for user_profile in users}

def get_accounts_for_email(email: str) -> List[Account]:
    profiles = UserProfile.objects.select_related('realm').filter(delivery_email__iexact=email.strip(), is_active=True, realm__deactivated=False, is_bot=False).order_by('date_joined')
    return [dict(realm_name=profile.realm.name, realm_id=profile.realm.id, full_name=profile.full_name, avatar=avatar_url(profile, medium=True)) for profile in profiles]

def validate_user_custom_profile_field(realm_id: int, field: CustomProfileField, value: ProfileDataElementValue) -> Optional[str]:
    validators = CustomProfileField.FIELD_VALIDATORS
    field_type = field.field_type
    var_name = f'{field.name}'
    if field_type in validators:
        validator = validators[field_type]
        return validator(var_name, value)
    elif field_type == CustomProfileField.SELECT:
        choice_field_validator = CustomProfileField.SELECT_FIELD_VALIDATORS[field_type]
        field_data = field.field_data
        assert field_data is not None
        return choice_field_validator(var_name, field_data, value)
    elif field_type == CustomProfileField.USER:
        user_field_validator = CustomProfileField.USER_FIELD_VALIDATORS[field_type]
        return user_field_validator(realm_id, value, False)
    else:
        raise AssertionError('Invalid field type')

def validate_user_custom_profile_data(realm_id: int, profile_data: List[ProfileDataElementUpdateDict], acting_user: UserProfile) -> None:
    for item in profile_data:
        field_id = item['id']
        try:
            field = CustomProfileField.objects.get(id=field_id)
        except CustomProfileField.DoesNotExist:
            raise JsonableError(_('Field id {id} not found.').format(id=field_id))
        if not acting_user.is_realm_admin and (not field.editable_by_user):
            raise JsonableError(_('You are not allowed to change this field. Contact an administrator to update it.'))
        try:
            validate_user_custom_profile_field(realm_id, field, item['value'])
        except ValidationError as error:
            raise JsonableError(error.message)

def can_access_delivery_email(user_profile: UserProfile, target_user_id: int, email_address_visibility: int) -> bool:
    if target_user_id == user_profile.id:
        return True
    return email_address_visibility in UserProfile.ROLE_TO_ACCESSIBLE_EMAIL_ADDRESS_VISIBILITY_IDS[user_profile.role]

def format_user_row(realm_id: int, acting_user: Optional[UserProfile], row: RawUserDict, client_gravatar: bool, user_avatar_url_field_optional: bool, custom_profile_field_data: Optional[Dict[str, Dict[str, Any]]] = None) -> APIUserDict:
    is_admin = is_administrator_role(row['role'])
    is_owner = row['role'] == UserProfile.ROLE_REALM_OWNER
    is_guest = row['role'] == UserProfile.ROLE_GUEST
    is_bot = row['is_bot']
    delivery_email = None
    if acting_user is not None and can_access_delivery_email(acting_user, row['id'], row['email_address_visibility']):
        delivery_email = row['delivery_email']
    result = APIUserDict(email=row['email'], user_id=row['id'], avatar_version=row['avatar_version'], is_admin=is_admin, is_owner=is_owner, is_guest=is_guest, is_billing_admin=row['is_billing_admin'], role=row['role'], is_bot=is_bot, full_name=row['full_name'], timezone=canonicalize_timezone(row['timezone']), is_active=row['is_active'], date_joined=row['date_joined'].date().isoformat() if acting_user is None else row['date_joined'].isoformat(timespec='minutes'), delivery_email=delivery_email)
    if acting_user is None:
        del result['is_billing_admin']
        del result['timezone']
    include_avatar_url = not user_avatar_url_field_optional or not row['long_term_id