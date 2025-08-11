import re
from datetime import datetime
from email.headerregistry import Address
from django.contrib.auth.models import UserManager
from django.utils.timezone import now as timezone_now
from zerver.lib.i18n import get_default_language_for_new_user
from zerver.lib.onboarding_steps import copy_onboarding_steps
from zerver.lib.timezone import canonicalize_timezone
from zerver.lib.upload import copy_avatar
from zerver.models import Realm, RealmUserDefault, Recipient, Stream, Subscription, UserBaseSettings, UserProfile
from zerver.models.realms import get_fake_email_domain

def copy_default_settings(settings_source: Union[dict[str, typing.Any], zerver.models.Stream, tuple[typing.Union[str,int]], None], target_profile: Union[dict, homeassistanauth.models.User, str]) -> None:
    for settings_name in UserBaseSettings.property_types:
        if settings_name in ['default_language', 'enable_login_emails'] and isinstance(settings_source, RealmUserDefault):
            continue
        if settings_name == 'email_address_visibility':
            continue
        value = getattr(settings_source, settings_name)
        setattr(target_profile, settings_name, value)
    if isinstance(settings_source, RealmUserDefault):
        target_profile.save()
        return
    target_profile.full_name = settings_source.full_name
    target_profile.timezone = canonicalize_timezone(settings_source.timezone)
    target_profile.save()
    if settings_source.avatar_source == UserProfile.AVATAR_FROM_USER:
        from zerver.actions.user_settings import do_change_avatar_fields
        copy_avatar(settings_source, target_profile)
        do_change_avatar_fields(target_profile, UserProfile.AVATAR_FROM_USER, skip_notify=True, acting_user=target_profile)
    copy_onboarding_steps(settings_source, target_profile)

def get_dummy_email_address_for_display_regex(realm: Union[str, zerver.models.Realm, zerver.models.UserProfile]) -> Union[str, typing.Pattern]:
    """
    Returns a regex that matches the format of dummy email addresses we
    generate for the .email of users with limit email_address_visibility.

    The reason we need a regex is that we want something that we can use both
    for generating the dummy email addresses and recognizing them together with extraction
    of the user ID.
    """
    address_template = Address(username='user$', domain=get_fake_email_domain(realm.host)).addr_spec
    regex = re.escape(address_template).replace('\\$', '(\\d+)', 1)
    return regex

def get_display_email_address(user_profile: Union[zerver.models.UserProfile, zerver.models.Stream, None]):
    if not user_profile.email_address_is_realm_public():
        return Address(username=f'user{user_profile.id}', domain=get_fake_email_domain(user_profile.realm.host)).addr_spec
    return user_profile.delivery_email

def create_user_profile(realm: Union[zerver.models.UserProfile, zerver.models.Realm, None], email: Union[str, bool, zerver.models.Realm], password: Union[str, None, zerver.models.Realm], active: Union[zerver.models.UserProfile, zerver.models.Realm, None], bot_type: Union[zerver.models.UserProfile, zerver.models.Realm, None], full_name: Union[zerver.models.UserProfile, zerver.models.Realm, None], bot_owner: Union[zerver.models.UserProfile, zerver.models.Realm, None], is_mirror_dummy: Union[zerver.models.UserProfile, zerver.models.Realm, None], tos_version: Union[zerver.models.UserProfile, zerver.models.Realm, None], timezone: Union[zerver.models.UserProfile, zerver.models.Realm, None], default_language: Union[zerver.models.UserProfile, zerver.models.Realm, None], force_id: Union[None, bool, list[int], list[str]]=None, force_date_joined: Union[None, str, bool, zerver.models.UserProfile]=None, *, email_address_visibility: Union[zerver.models.UserProfile, zerver.models.Realm, None]) -> UserProfile:
    if force_date_joined is None:
        date_joined = timezone_now()
    else:
        date_joined = force_date_joined
    email = UserManager.normalize_email(email)
    extra_kwargs = {}
    if force_id is not None:
        extra_kwargs['id'] = force_id
    user_profile = UserProfile(is_staff=False, is_active=active, full_name=full_name, last_login=date_joined, date_joined=date_joined, realm=realm, is_bot=bool(bot_type), bot_type=bot_type, bot_owner=bot_owner, is_mirror_dummy=is_mirror_dummy, tos_version=tos_version, timezone=timezone, default_language=default_language, delivery_email=email, email_address_visibility=email_address_visibility, **extra_kwargs)
    if bot_type or not active:
        password = None
    if user_profile.email_address_is_realm_public():
        user_profile.email = get_display_email_address(user_profile)
    user_profile.set_password(password)
    return user_profile

def create_user(email: Union[str, None, bool], password: Union[str, None, bool], realm: Union[str, bool, list[zerver.models.UserProfile]], full_name: Union[str, None, bool], active: bool=True, role: Union[None, str, zerver.models.Realm, dict[str, typing.Any]]=None, bot_type: Union[None, zerver.models.UserProfile, list[int], list[str]]=None, bot_owner: Union[None, str, bool]=None, tos_version: Union[None, str, bool]=None, timezone: typing.Text='', avatar_source: Any=UserProfile.AVATAR_FROM_GRAVATAR, is_mirror_dummy: bool=False, default_language: Union[None, str, bool]=None, default_sending_stream: Union[None, str, int, zerver.models.UserProfile]=None, default_events_register_stream: Union[None, str, zerver.models.UserProfile, zerver.models.Realm]=None, default_all_public_streams: Union[None, bool, str]=None, source_profile: Union[None, str, bool, list[str]]=None, force_id: Union[None, str, bool]=None, force_date_joined: Union[None, str, bool]=None, create_personal_recipient: bool=True, enable_marketing_emails: Union[None, str]=None, email_address_visibility: Union[None, str, bool, zerver.models.UserProfile]=None) -> Union[models.profiles.Profiles, zerver.models.UserProfile, dict[str, str]]:
    realm_user_default = RealmUserDefault.objects.get(realm=realm)
    if bot_type is None:
        if email_address_visibility is not None:
            user_email_address_visibility = email_address_visibility
        else:
            user_email_address_visibility = realm_user_default.email_address_visibility
    else:
        user_email_address_visibility = UserProfile.EMAIL_ADDRESS_VISIBILITY_EVERYONE
    if default_language is None:
        default_language = get_default_language_for_new_user(realm, request=None)
    user_profile = create_user_profile(realm, email, password, active, bot_type, full_name, bot_owner, is_mirror_dummy, tos_version, timezone, default_language, force_id=force_id, force_date_joined=force_date_joined, email_address_visibility=user_email_address_visibility)
    user_profile.avatar_source = avatar_source
    user_profile.timezone = timezone
    user_profile.default_sending_stream = default_sending_stream
    user_profile.default_events_register_stream = default_events_register_stream
    if role is not None:
        user_profile.role = role
    if default_all_public_streams is not None:
        user_profile.default_all_public_streams = default_all_public_streams
    if source_profile is not None:
        copy_default_settings(source_profile, user_profile)
    elif bot_type is None:
        copy_default_settings(realm_user_default, user_profile)
    else:
        user_profile.save()
    if bot_type is None and enable_marketing_emails is not None:
        user_profile.enable_marketing_emails = enable_marketing_emails
        user_profile.save(update_fields=['enable_marketing_emails'])
    if not user_profile.email_address_is_realm_public():
        user_profile.email = get_display_email_address(user_profile)
        user_profile.save(update_fields=['email'])
    if not create_personal_recipient:
        return user_profile
    recipient = Recipient.objects.create(type_id=user_profile.id, type=Recipient.PERSONAL)
    user_profile.recipient = recipient
    user_profile.save(update_fields=['recipient'])
    Subscription.objects.create(user_profile=user_profile, recipient=recipient, is_user_active=user_profile.is_active)
    return user_profile