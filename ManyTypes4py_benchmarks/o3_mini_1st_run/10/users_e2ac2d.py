#!/usr/bin/env python3
from collections import defaultdict
import itertools
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from email.headerregistry import Address
from operator import itemgetter
from typing import Any, Dict, List, Optional, Set, Tuple, Mapping as TypingMapping, Sequence as TypingSequence
from typing_extensions import NotRequired, TypedDict

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import Q, QuerySet
from django.db.models.functions import Upper
from django.utils.translation import gettext as _
from django_otp.middleware import is_verified

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


def check_full_name(full_name_raw: str, *, user_profile: Optional[UserProfile], realm: Realm) -> str:
    full_name: str = full_name_raw.strip()
    if len(full_name) > UserProfile.MAX_NAME_LENGTH:
        raise JsonableError(_('Name too long!'))
    if len(full_name) < UserProfile.MIN_NAME_LENGTH:
        raise JsonableError(_('Name too short!'))
    if check_string_is_printable(full_name) is not None or any((character in full_name for character in UserProfile.NAME_INVALID_CHARS)):
        raise JsonableError(_('Invalid characters in name!'))
    if re.search('\\|\\d+$', full_name_raw):
        raise JsonableError(_('Invalid format!'))
    if require_unique_names(realm):
        normalized_user_full_name: str = unicodedata.normalize('NFKC', full_name).casefold()
        users_query = UserProfile.objects.filter(realm=realm)
        if user_profile is not None:
            existing_names = users_query.exclude(id=user_profile.id).values_list('full_name', flat=True)
        else:
            existing_names = users_query.values_list('full_name', flat=True)
        normalized_existing_names: List[str] = [unicodedata.normalize('NFKC', name).casefold() for name in existing_names]
        if normalized_user_full_name in normalized_existing_names:
            raise JsonableError(_('Unique names required in this organization.'))
    return full_name


def check_bot_name_available(realm_id: int, full_name: str, *, is_activation: bool) -> None:
    dup_exists: bool = UserProfile.objects.filter(realm_id=realm_id, full_name=full_name.strip(), is_active=True).exists()
    if dup_exists:
        if is_activation:
            raise JsonableError(f'There is already an active bot named "{full_name}" in this organization. To reactivate this bot, you must rename or deactivate the other one first.')
        else:
            raise JsonableError(_('Name is already in use!'))


def check_short_name(short_name_raw: str) -> str:
    short_name: str = short_name_raw.strip()
    if len(short_name) == 0:
        raise JsonableError(_('Bad name or username'))
    return short_name


def check_valid_bot_config(bot_type: int, service_name: str, config_data: TypingMapping[str, Any]) -> None:
    if bot_type == UserProfile.INCOMING_WEBHOOK_BOT:
        from zerver.lib.integrations import WEBHOOK_INTEGRATIONS
        config_options: Optional[Dict[str, Any]] = None
        for integration in WEBHOOK_INTEGRATIONS:
            if integration.name == service_name:
                config_options = {option.name: option.validator for option in integration.config_options}
                break
        if not config_options:
            raise JsonableError(_("Invalid integration '{integration_name}'.").format(integration_name=service_name))
        missing_keys: Set[str] = set(config_options.keys()) - set(config_data.keys())
        if missing_keys:
            raise JsonableError(_('Missing configuration parameters: {keys}').format(keys=missing_keys))
        for key, validator in config_options.items():
            value: Any = config_data[key]
            error: Optional[str] = validator(key, value)
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


def add_service(name: str, user_profile: UserProfile, base_url: str, interface: str, token: str) -> None:
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


def check_valid_interface_type(interface_type: str) -> None:
    if interface_type not in Service.ALLOWED_INTERFACE_TYPES:
        raise JsonableError(_('Invalid interface type'))


def is_administrator_role(role: int) -> bool:
    return role in {UserProfile.ROLE_REALM_ADMINISTRATOR, UserProfile.ROLE_REALM_OWNER}


@cache_with_key(get_cross_realm_dicts_key)
def bulk_get_cross_realm_bots() -> Dict[str, UserProfile]:
    emails: List[str] = list(settings.CROSS_REALM_BOT_EMAILS)
    where_clause: str = 'upper(zerver_userprofile.email::text) IN (SELECT upper(email) FROM unnest(%s) AS email)'
    users: QuerySet[UserProfile] = UserProfile.objects.filter(realm__string_id=settings.SYSTEM_BOT_REALM).extra(where=[where_clause], params=(emails,))
    return {user.email.lower(): user for user in users}


def user_ids_to_users(user_ids: TypingSequence[int], realm: Realm, *, allow_deactivated: bool) -> List[UserProfile]:
    user_query = UserProfile.objects.filter(id__in=user_ids, realm=realm)
    if not allow_deactivated:
        user_query = user_query.filter(is_active=True)
    user_profiles: List[UserProfile] = list(user_query.select_related('realm'))
    found_user_ids: Set[int] = {user_profile.id for user_profile in user_profiles}
    for user_id in user_ids:
        if user_id not in found_user_ids:
            raise JsonableError(_('Invalid user ID: {user_id}').format(user_id=user_id))
    return user_profiles


def access_bot_by_id(user_profile: UserProfile, user_id: int) -> UserProfile:
    try:
        target: UserProfile = get_user_profile_by_id_in_realm(user_id, user_profile.realm)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such bot'))
    if not target.is_bot:
        raise JsonableError(_('No such bot'))
    if not user_profile.can_admin_user(target):
        raise JsonableError(_('Insufficient permission'))
    if target.can_create_users and (not user_profile.is_realm_owner):
        raise OrganizationOwnerRequiredError
    return target


def access_user_common(target: UserProfile, user_profile: Optional[UserProfile], allow_deactivated: bool, allow_bots: bool, for_admin: bool) -> UserProfile:
    if target.is_bot and (not allow_bots):
        raise JsonableError(_('No such user'))
    if not target.is_active and (not allow_deactivated):
        raise JsonableError(_('User is deactivated'))
    if not for_admin:
        if not check_can_access_user(target, user_profile):
            raise JsonableError(_('Insufficient permission'))
        return target
    if not user_profile or not user_profile.can_admin_user(target):
        raise JsonableError(_('Insufficient permission'))
    return target


def access_user_by_id(user_profile: UserProfile, target_user_id: int, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> UserProfile:
    try:
        target: UserProfile = get_user_profile_by_id_in_realm(target_user_id, user_profile.realm)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such user'))
    return access_user_common(target, user_profile, allow_deactivated, allow_bots, for_admin)


def access_user_by_id_including_cross_realm(user_profile: UserProfile, target_user_id: int, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> UserProfile:
    try:
        target: UserProfile = get_user_by_id_in_realm_including_cross_realm(target_user_id, user_profile.realm)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such user'))
    return access_user_common(target, user_profile, allow_deactivated, allow_bots, for_admin)


def access_user_by_email(user_profile: UserProfile, email: str, *, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> UserProfile:
    dummy_email_regex: str = get_dummy_email_address_for_display_regex(user_profile.realm)
    match = re.match(dummy_email_regex, email)
    if match:
        target_id: int = int(match.group(1))
        return access_user_by_id(user_profile, target_id, allow_deactivated=allow_deactivated, allow_bots=allow_bots, for_admin=for_admin)
    allowed_email_address_visibility_values: Any = UserProfile.ROLE_TO_ACCESSIBLE_EMAIL_ADDRESS_VISIBILITY_IDS[user_profile.role]
    try:
        target: UserProfile = base_get_user_queryset().get(delivery_email__iexact=email.strip(), realm=user_profile.realm, email_address_visibility__in=allowed_email_address_visibility_values)
    except UserProfile.DoesNotExist:
        raise JsonableError(_('No such user'))
    return access_user_common(target, user_profile, allow_deactivated, allow_bots, for_admin)


def bulk_access_users_by_email(emails: List[str], *, acting_user: UserProfile, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> Set[UserProfile]:
    target_emails_upper: List[str] = [email.strip().upper() for email in emails]
    users: QuerySet[UserProfile] = base_bulk_get_user_queryset().annotate(email_upper=Upper('email')).filter(email_upper__in=target_emails_upper, realm=acting_user.realm)
    valid_emails_upper: Set[str] = {user_profile.email_upper for user_profile in users}
    all_users_exist: bool = all((email in valid_emails_upper for email in target_emails_upper))
    if not all_users_exist:
        raise JsonableError(_('No such user'))
    return {access_user_common(user_profile, acting_user, allow_deactivated, allow_bots, for_admin) for user_profile in users}


def bulk_access_users_by_id(user_ids: List[int], *, acting_user: UserProfile, allow_deactivated: bool = False, allow_bots: bool = False, for_admin: bool) -> Set[UserProfile]:
    users: QuerySet[UserProfile] = base_bulk_get_user_queryset().filter(id__in=user_ids, realm=acting_user.realm)
    valid_user_ids: Set[int] = {user_profile.id for user_profile in users}
    all_users_exist: bool = all((user_id in valid_user_ids for user_id in user_ids))
    if not all_users_exist:
        raise JsonableError(_('No such user'))
    return {access_user_common(user_profile, acting_user, allow_deactivated, allow_bots, for_admin) for user_profile in users}


class Account(TypedDict):
    pass


def get_accounts_for_email(email: str) -> List[Dict[str, Any]]:
    profiles = UserProfile.objects.select_related('realm').filter(delivery_email__iexact=email.strip(), is_active=True, realm__deactivated=False, is_bot=False).order_by('date_joined')
    return [dict(realm_name=profile.realm.name, realm_id=profile.realm.id, full_name=profile.full_name, avatar=avatar_url(profile, medium=True)) for profile in profiles]


def validate_user_custom_profile_field(realm_id: int, field: CustomProfileField, value: Any) -> Optional[str]:
    validators: Dict[int, Any] = CustomProfileField.FIELD_VALIDATORS
    field_type: int = field.field_type
    var_name: str = f'{field.name}'
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


def validate_user_custom_profile_data(realm_id: int, profile_data: List[Dict[str, Any]], acting_user: UserProfile) -> None:
    for item in profile_data:
        field_id: int = item['id']
        try:
            field: CustomProfileField = CustomProfileField.objects.get(id=field_id)
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


class APIUserDict(TypedDict):
    pass


def format_user_row(realm_id: int, acting_user: Optional[UserProfile], row: TypingMapping[str, Any], client_gravatar: bool, user_avatar_url_field_optional: bool, custom_profile_field_data: Optional[Dict[str, Any]] = None) -> APIUserDict:
    is_admin: bool = is_administrator_role(row['role'])
    is_owner: bool = row['role'] == UserProfile.ROLE_REALM_OWNER
    is_guest: bool = row['role'] == UserProfile.ROLE_GUEST
    is_bot: bool = row['is_bot']
    delivery_email: Optional[str] = None
    if acting_user is not None and can_access_delivery_email(acting_user, row['id'], row['email_address_visibility']):
        delivery_email = row['delivery_email']
    result: APIUserDict = APIUserDict(
        email=row['email'],
        user_id=row['id'],
        avatar_version=row['avatar_version'],
        is_admin=is_admin,
        is_owner=is_owner,
        is_guest=is_guest,
        is_billing_admin=row['is_billing_admin'],
        role=row['role'],
        is_bot=is_bot,
        full_name=row['full_name'],
        timezone=canonicalize_timezone(row['timezone']),
        is_active=row['is_active'],
        date_joined=(row['date_joined'].date().isoformat() if acting_user is None else row['date_joined'].isoformat(timespec='minutes')),
        delivery_email=delivery_email,
    )
    if acting_user is None:
        del result['is_billing_admin']
        del result['timezone']
    include_avatar_url: bool = not user_avatar_url_field_optional or not row['long_term_idle']
    if include_avatar_url:
        result['avatar_url'] = get_avatar_field(user_id=row['id'], realm_id=realm_id, email=row['delivery_email'], avatar_source=row['avatar_source'], avatar_version=row['avatar_version'], medium=False, client_gravatar=client_gravatar)
    if is_bot:
        result['bot_type'] = row['bot_type']
        if is_cross_realm_bot_email(row['email']):
            result['is_system_bot'] = True
        result['bot_owner_id'] = row['bot_owner_id']
    elif custom_profile_field_data is not None:
        result['profile_data'] = custom_profile_field_data
    return result


def user_access_restricted_in_realm(target_user: UserProfile) -> bool:
    if target_user.is_bot:
        return False
    realm: Realm = target_user.realm
    if realm.can_access_all_users_group.named_user_group.name == SystemGroups.EVERYONE:
        return False
    return True


def check_user_can_access_all_users(acting_user: Optional[UserProfile]) -> bool:
    if acting_user is None:
        return True
    if not acting_user.is_guest:
        return True
    realm: Realm = acting_user.realm
    if user_has_permission_for_group_setting(realm.can_access_all_users_group, acting_user, Realm.REALM_PERMISSION_GROUP_SETTINGS['can_access_all_users_group']):
        return True
    return False


def check_can_access_user(target_user: UserProfile, user_profile: Optional[UserProfile] = None) -> bool:
    if not user_access_restricted_in_realm(target_user):
        return True
    if check_user_can_access_all_users(user_profile):
        return True
    assert user_profile is not None
    if target_user.id == user_profile.id:
        return True
    subscribed_recipient_ids = Subscription.objects.filter(user_profile=user_profile, active=True, recipient__type__in=[Recipient.STREAM, Recipient.DIRECT_MESSAGE_GROUP]).values_list('recipient_id', flat=True)
    if Subscription.objects.filter(recipient_id__in=subscribed_recipient_ids, user_profile=target_user, active=True, is_user_active=True).exists():
        return True
    assert user_profile.recipient_id is not None
    assert target_user.recipient_id is not None
    direct_message_query = Message.objects.filter(recipient__type=Recipient.PERSONAL, realm=target_user.realm)
    if direct_message_query.filter(Q(sender_id=target_user.id, recipient_id=user_profile.recipient_id) | Q(recipient_id=target_user.recipient_id, sender_id=user_profile.id)).exists():
        return True
    return False


def get_inaccessible_user_ids(target_user_ids: TypingSequence[int], acting_user: UserProfile) -> Set[int]:
    if check_user_can_access_all_users(acting_user):
        return set()
    target_human_user_ids: List[int] = list(UserProfile.objects.filter(id__in=target_user_ids, is_bot=False).values_list('id', flat=True))
    if not target_human_user_ids:
        return set()
    subscribed_recipient_ids = Subscription.objects.filter(user_profile=acting_user, active=True, recipient__type__in=[Recipient.STREAM, Recipient.DIRECT_MESSAGE_GROUP]).values_list('recipient_id', flat=True)
    common_subscription_user_ids = Subscription.objects.filter(recipient_id__in=subscribed_recipient_ids, user_profile_id__in=target_human_user_ids, active=True, is_user_active=True).distinct('user_profile_id').values_list('user_profile_id', flat=True)
    possible_inaccessible_user_ids: Set[int] = set(target_human_user_ids) - set(common_subscription_user_ids)
    if not possible_inaccessible_user_ids:
        return set()
    target_user_recipient_ids = UserProfile.objects.filter(id__in=possible_inaccessible_user_ids).values_list('recipient_id', flat=True)
    direct_message_query = Message.objects.filter(recipient__type=Recipient.PERSONAL, realm=acting_user.realm)
    direct_messages_users = direct_message_query.filter(Q(sender_id__in=possible_inaccessible_user_ids, recipient_id=acting_user.recipient_id) | Q(recipient_id__in=target_user_recipient_ids, sender_id=acting_user.id)).values_list('sender_id', 'recipient__type_id')
    user_ids_involved_in_dms: Set[int] = set()
    for sender_id, recipient_user_id in direct_messages_users:
        if sender_id == acting_user.id:
            user_ids_involved_in_dms.add(recipient_user_id)
        else:
            user_ids_involved_in_dms.add(sender_id)
    inaccessible_user_ids: Set[int] = possible_inaccessible_user_ids - user_ids_involved_in_dms
    return inaccessible_user_ids


def get_user_ids_who_can_access_user(target_user: UserProfile) -> List[int]:
    realm: Realm = target_user.realm
    if not user_access_restricted_in_realm(target_user):
        return active_user_ids(realm.id)
    active_non_guest_user_ids_in_realm: List[int] = active_non_guest_user_ids(realm.id)
    users_sharing_any_subscription: Dict[int, Set[int]] = get_subscribers_of_target_user_subscriptions([target_user])
    users_involved_in_dms_dict: Dict[int, Set[int]] = get_users_involved_in_dms_with_target_users([target_user], realm)
    user_ids_who_can_access_target_user: Set[int] = {target_user.id} | set(active_non_guest_user_ids_in_realm) | users_sharing_any_subscription[target_user.id] | users_involved_in_dms_dict[target_user.id]
    return list(user_ids_who_can_access_target_user)


def get_subscribers_of_target_user_subscriptions(target_users: TypingSequence[UserProfile], include_deactivated_users_for_dm_groups: bool = False) -> Dict[int, Set[int]]:
    target_user_ids: List[int] = [user.id for user in target_users]
    target_user_subscriptions = Subscription.objects.filter(user_profile__in=target_user_ids, active=True, recipient__type__in=[Recipient.STREAM, Recipient.DIRECT_MESSAGE_GROUP]).order_by('user_profile_id').values('user_profile_id', 'recipient_id')
    target_users_subbed_recipient_ids: Set[int] = set()
    target_user_subscriptions_dict: Dict[int, Set[int]] = defaultdict(set)
    for user_profile_id, sub_rows in itertools.groupby(target_user_subscriptions, itemgetter('user_profile_id')):
        recipient_ids: Set[int] = {row['recipient_id'] for row in sub_rows}
        target_user_subscriptions_dict[user_profile_id] = recipient_ids
        target_users_subbed_recipient_ids |= recipient_ids
    subs_in_target_user_subscriptions_query = Subscription.objects.filter(recipient_id__in=list(target_users_subbed_recipient_ids), active=True)
    if include_deactivated_users_for_dm_groups:
        subs_in_target_user_subscriptions_query = subs_in_target_user_subscriptions_query.filter(Q(recipient__type=Recipient.STREAM, is_user_active=True) | Q(recipient__type=Recipient.DIRECT_MESSAGE_GROUP))
    else:
        subs_in_target_user_subscriptions_query = subs_in_target_user_subscriptions_query.filter(recipient__type__in=[Recipient.STREAM, Recipient.DIRECT_MESSAGE_GROUP], is_user_active=True)
    subs_in_target_user_subscriptions = subs_in_target_user_subscriptions_query.order_by('recipient_id').values('user_profile_id', 'recipient_id')
    subscribers_dict_by_recipient_ids: Dict[int, Set[int]] = defaultdict(set)
    for recipient_id, sub_rows in itertools.groupby(subs_in_target_user_subscriptions, itemgetter('recipient_id')):
        user_ids: Set[int] = {row['user_profile_id'] for row in sub_rows}
        subscribers_dict_by_recipient_ids[recipient_id] = user_ids
    users_subbed_to_target_user_subscriptions_dict: Dict[int, Set[int]] = defaultdict(set)
    for user_id in target_user_ids:
        target_user_subbed_recipients: Set[int] = target_user_subscriptions_dict[user_id]
        for recipient_id in target_user_subbed_recipients:
            users_subbed_to_target_user_subscriptions_dict[user_id] |= subscribers_dict_by_recipient_ids[recipient_id]
    return users_subbed_to_target_user_subscriptions_dict


def get_users_involved_in_dms_with_target_users(target_users: TypingSequence[UserProfile], realm: Realm, include_deactivated_users: bool = False) -> Dict[int, Set[int]]:
    target_user_ids: List[int] = [user.id for user in target_users]
    direct_messages_recipient_users = Message.objects.filter(sender_id__in=target_user_ids, realm=realm, recipient__type=Recipient.PERSONAL).order_by('sender_id').distinct('sender_id', 'recipient__type_id').values('sender_id', 'recipient__type_id')
    direct_messages_recipient_users_set: Set[int] = {obj['recipient__type_id'] for obj in direct_messages_recipient_users}
    active_direct_messages_recipient_user_ids = UserProfile.objects.filter(id__in=list(direct_messages_recipient_users_set), is_active=True).values_list('id', flat=True)
    direct_message_participants_dict: Dict[int, Set[int]] = defaultdict(set)
    for sender_id, message_rows in itertools.groupby(direct_messages_recipient_users, itemgetter('sender_id')):
        recipient_user_ids: Set[int] = {row['recipient__type_id'] for row in message_rows}
        if not include_deactivated_users:
            recipient_user_ids &= set(active_direct_messages_recipient_user_ids)
        direct_message_participants_dict[sender_id] = recipient_user_ids
    personal_recipient_ids_for_target_users: List[int] = [user.recipient_id for user in target_users]
    direct_message_senders_query = Message.objects.filter(realm=realm, recipient_id__in=personal_recipient_ids_for_target_users, recipient__type=Recipient.PERSONAL)
    if not include_deactivated_users:
        direct_message_senders_query = direct_message_senders_query.filter(sender__is_active=True)
    direct_messages_senders = direct_message_senders_query.order_by('recipient__type_id').distinct('sender_id', 'recipient__type_id').values('sender_id', 'recipient__type_id')
    for recipient_user_id, message_rows in itertools.groupby(direct_messages_senders, itemgetter('recipient__type_id')):
        sender_ids: Set[int] = {row['sender_id'] for row in message_rows}
        direct_message_participants_dict[recipient_user_id] |= sender_ids
    return direct_message_participants_dict


def user_profile_to_user_row(user_profile: UserProfile) -> RawUserDict:
    return RawUserDict(
        id=user_profile.id,
        full_name=user_profile.full_name,
        email=user_profile.email,
        avatar_source=user_profile.avatar_source,
        avatar_version=user_profile.avatar_version,
        is_active=user_profile.is_active,
        role=user_profile.role,
        is_billing_admin=user_profile.is_billing_admin,
        is_bot=user_profile.is_bot,
        timezone=user_profile.timezone,
        date_joined=user_profile.date_joined,
        bot_owner_id=user_profile.bot_owner_id,
        delivery_email=user_profile.delivery_email,
        bot_type=user_profile.bot_type,
        long_term_idle=user_profile.long_term_idle,
        email_address_visibility=user_profile.email_address_visibility,
    )


@cache_with_key(get_cross_realm_dicts_key)
def get_cross_realm_dicts() -> List[APIUserDict]:
    user_dict: Dict[str, UserProfile] = bulk_get_cross_realm_bots()
    users: List[UserProfile] = sorted(list(user_dict.values()), key=lambda user: user.full_name)
    result: List[APIUserDict] = []
    for user in users:
        user_row: RawUserDict = user_profile_to_user_row(user)
        user_row['bot_owner_id'] = None
        result.append(format_user_row(user.realm_id, acting_user=user, row=user_row, client_gravatar=False, user_avatar_url_field_optional=False, custom_profile_field_data=None))
    return result


def get_data_for_inaccessible_user(realm: Realm, user_id: int) -> APIUserDict:
    fake_email: str = Address(username=f'user{user_id}', domain=get_fake_email_domain(realm.host)).addr_spec
    user_date_joined = timestamp_to_datetime(0)
    user_dict: APIUserDict = APIUserDict(
        email=fake_email,
        user_id=user_id,
        avatar_version=1,
        is_admin=False,
        is_owner=False,
        is_guest=False,
        is_billing_admin=False,
        role=UserProfile.ROLE_MEMBER,
        is_bot=False,
        full_name=str(UserProfile.INACCESSIBLE_USER_NAME),
        timezone='',
        is_active=True,
        date_joined=user_date_joined.isoformat(),
        delivery_email=None,
        avatar_url=get_avatar_for_inaccessible_user(),
        profile_data={},
    )
    return user_dict


def get_accessible_user_ids(realm: Realm, user_profile: UserProfile, include_deactivated_users: bool = False) -> List[int]:
    subscribers_dict_of_target_user_subscriptions: Dict[int, Set[int]] = get_subscribers_of_target_user_subscriptions([user_profile], include_deactivated_users_for_dm_groups=include_deactivated_users)
    users_involved_in_dms_dict: Dict[int, Set[int]] = get_users_involved_in_dms_with_target_users([user_profile], realm, include_deactivated_users=include_deactivated_users)
    accessible_user_ids: Set[int] = {user_profile.id} | subscribers_dict_of_target_user_subscriptions[user_profile.id] | users_involved_in_dms_dict[user_profile.id]
    return list(accessible_user_ids)


def get_user_dicts_in_realm(realm: Realm, user_profile: UserProfile) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_allowed_to_access_all_users = realm.can_access_all_users_group
    assert group_allowed_to_access_all_users is not None
    all_user_dicts: List[Dict[str, Any]] = get_realm_user_dicts(realm.id)
    if check_user_can_access_all_users(user_profile):
        return (all_user_dicts, [])
    accessible_user_ids: List[int] = get_accessible_user_ids(realm, user_profile, include_deactivated_users=True)
    accessible_user_dicts: List[Dict[str, Any]] = []
    inaccessible_user_dicts: List[Dict[str, Any]] = []
    for user_dict in all_user_dicts:
        if user_dict['id'] in accessible_user_ids or user_dict['is_bot']:
            accessible_user_dicts.append(user_dict)
        else:
            inaccessible_user_dicts.append(get_data_for_inaccessible_user(realm, user_dict['id']))
    return (accessible_user_dicts, inaccessible_user_dicts)


def get_custom_profile_field_values(custom_profile_field_values: QuerySet[CustomProfileFieldValue]) -> Dict[int, Dict[str, Any]]:
    profiles_by_user_id: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for profile_field in custom_profile_field_values:
        user_id: int = profile_field.user_profile_id
        if profile_field.field.is_renderable():
            profiles_by_user_id[user_id][str(profile_field.field_id)] = {'value': profile_field.value, 'rendered_value': profile_field.rendered_value}
        else:
            profiles_by_user_id[user_id][str(profile_field.field_id)] = {'value': profile_field.value}
    return profiles_by_user_id


def get_users_for_api(realm: Realm, acting_user: UserProfile, *, target_user: Optional[UserProfile] = None, client_gravatar: bool, user_avatar_url_field_optional: bool, include_custom_profile_fields: bool = True, user_list_incomplete: bool = False) -> Dict[int, APIUserDict]:
    profiles_by_user_id: Optional[Dict[int, Dict[str, Any]]] = None
    custom_profile_field_data: Optional[Dict[str, Any]] = None
    accessible_user_dicts: List[Dict[str, Any]] = []
    inaccessible_user_dicts: List[Dict[str, Any]] = []
    if target_user is not None:
        accessible_user_dicts = [user_profile_to_user_row(target_user)]
    else:
        accessible_user_dicts, inaccessible_user_dicts = get_user_dicts_in_realm(realm, acting_user)
    if include_custom_profile_fields:
        base_query: QuerySet[CustomProfileFieldValue] = CustomProfileFieldValue.objects.select_related('field')
        if target_user is not None:
            custom_profile_field_values = base_query.filter(user_profile=target_user)
        else:
            custom_profile_field_values = base_query.filter(field__realm_id=realm.id)
        profiles_by_user_id = get_custom_profile_field_values(custom_profile_field_values)
    result: Dict[int, APIUserDict] = {}
    for row in accessible_user_dicts:
        if profiles_by_user_id is not None:
            custom_profile_field_data = profiles_by_user_id.get(row['id'], {})
        client_gravatar_for_user: bool = client_gravatar and row['email_address_visibility'] == UserProfile.EMAIL_ADDRESS_VISIBILITY_EVERYONE
        result[row['id']] = format_user_row(realm.id, acting_user=acting_user, row=row, client_gravatar=client_gravatar_for_user, user_avatar_url_field_optional=user_avatar_url_field_optional, custom_profile_field_data=custom_profile_field_data)
    if not user_list_incomplete:
        for inaccessible_user_row in inaccessible_user_dicts:
            user_id: int = inaccessible_user_row['user_id']
            result[user_id] = inaccessible_user_row
    return result


def get_active_bots_owned_by_user(user_profile: UserProfile) -> QuerySet[UserProfile]:
    return UserProfile.objects.filter(is_bot=True, is_active=True, bot_owner=user_profile)


def is_2fa_verified(user: UserProfile) -> bool:
    """
    It is generally unsafe to call is_verified directly on `request.user` since
    the attribute `otp_device` does not exist on an `AnonymousUser`, and `is_verified`
    does not make sense without 2FA being enabled.

    This wraps the checks for all these assumptions to make sure the call is safe.
    """
    assert settings.TWO_FACTOR_AUTHENTICATION_ENABLED
    return is_verified(user)


def get_users_with_access_to_real_email(user_profile: UserProfile) -> List[int]:
    if not user_access_restricted_in_realm(user_profile):
        active_users = user_profile.realm.get_active_users()
    else:
        user_ids_who_can_access_user = get_user_ids_who_can_access_user(user_profile)
        active_users = UserProfile.objects.filter(id__in=user_ids_who_can_access_user, is_active=True)
    return [user.id for user in active_users if can_access_delivery_email(user, user_profile.id, user_profile.email_address_visibility)]


def max_message_id_for_user(user_profile: Optional[UserProfile]) -> int:
    if user_profile is None:
        return -1
    max_message: Optional[UserMessage] = UserMessage.objects.filter(user_profile=user_profile).order_by('-message_id').only('message_id').first()
    if max_message:
        return max_message.message_id
    else:
        return -1
