from typing import Optional
from urllib.parse import urljoin
from django.conf import settings
from django.contrib.staticfiles.storage import staticfiles_storage
from zerver.lib.avatar_hash import gravatar_hash, user_avatar_base_path_from_ids, user_avatar_content_hash
from zerver.lib.thumbnail import MEDIUM_AVATAR_SIZE
from zerver.lib.upload import get_avatar_url
from zerver.lib.url_encoding import append_url_query_string
from zerver.models import UserProfile
from zerver.models.users import is_cross_realm_bot_email
STATIC_AVATARS_DIR: str = 'images/static_avatars/'
DEFAULT_AVATAR_FILE: str = 'images/default-avatar.png'

def avatar_url(user_profile, medium=False, client_gravatar=False):
    return get_avatar_field(user_id=user_profile.id, realm_id=user_profile.realm_id, email=user_profile.delivery_email, avatar_source=user_profile.avatar_source, avatar_version=user_profile.avatar_version, medium=medium, client_gravatar=client_gravatar)

def get_system_bots_avatar_file_name(email):
    system_bot_avatar_name_map: dict[str, str] = {settings.WELCOME_BOT: 'welcome-bot', settings.NOTIFICATION_BOT: 'notification-bot', settings.EMAIL_GATEWAY_BOT: 'emailgateway'}
    return urljoin(STATIC_AVATARS_DIR, system_bot_avatar_name_map.get(email, 'unknown'))

def get_static_avatar_url(email, medium):
    avatar_file_name: str = get_system_bots_avatar_file_name(email)
    avatar_file_name += '-medium.png' if medium else '.png'
    if settings.DEBUG:
        from django.contrib.staticfiles.finders import find
        if not find(avatar_file_name):
            raise AssertionError(f'Unknown avatar file for: {email}')
    elif settings.STATIC_ROOT and (not staticfiles_storage.exists(avatar_file_name)):
        return DEFAULT_AVATAR_FILE
    return staticfiles_storage.url(avatar_file_name)

def get_avatar_field(user_id, realm_id, email, avatar_source, avatar_version, medium, client_gravatar):
    """
    Most of the parameters to this function map to fields
    by the same name in UserProfile (avatar_source, realm_id,
    email, etc.).

    Then there are these:

        medium - This means we want a medium-sized avatar. This can
            affect the "s" parameter for gravatar avatars, or it
            can give us something like foo-medium.png for
            user-uploaded avatars.

        client_gravatar - If the client can compute their own
            gravatars, this will be set to True, and we'll avoid
            computing them on the server (mostly to save bandwidth).
    """
    if is_cross_realm_bot_email(email):
        return get_static_avatar_url(email, medium)
    '\n    If our client knows how to calculate gravatar hashes, we\n    will return None and let the client compute the gravatar\n    url.\n    '
    if client_gravatar and settings.ENABLE_GRAVATAR and (avatar_source == UserProfile.AVATAR_FROM_GRAVATAR):
        return None
    "\n    If we get this far, we'll compute an avatar URL that may be\n    either user-uploaded or a gravatar, and then we'll add version\n    info to try to avoid stale caches.\n    "
    if avatar_source == 'U':
        hash_key: str = user_avatar_base_path_from_ids(user_id, avatar_version, realm_id)
        return get_avatar_url(hash_key, medium=medium)
    return get_gravatar_url(email=email, avatar_version=avatar_version, realm_id=realm_id, medium=medium)

def get_gravatar_url(email, avatar_version, realm_id, medium=False):
    url: str = _get_unversioned_gravatar_url(email, medium, realm_id)
    return append_url_query_string(url, f'version={avatar_version:d}')

def _get_unversioned_gravatar_url(email, medium, realm_id):
    use_gravatar: bool = settings.ENABLE_GRAVATAR
    if realm_id in settings.GRAVATAR_REALM_OVERRIDE:
        use_gravatar = settings.GRAVATAR_REALM_OVERRIDE[realm_id]
    if use_gravatar:
        gravitar_query_suffix: str = f'&s={MEDIUM_AVATAR_SIZE}' if medium else ''
        hash_key: str = gravatar_hash(email)
        return f'https://secure.gravatar.com/avatar/{hash_key}?d=identicon{gravitar_query_suffix}'
    elif settings.DEFAULT_AVATAR_URI is not None:
        return settings.DEFAULT_AVATAR_URI
    else:
        return staticfiles_storage.url('images/default-avatar.png')

def absolute_avatar_url(user_profile):
    """
    Absolute URLs are used to simplify logic for applications that
    won't be served by browsers, such as rendering GCM notifications.
    """
    avatar: Optional[str] = avatar_url(user_profile)
    assert avatar is not None
    return urljoin(user_profile.realm.url, avatar)

def is_avatar_new(ldap_avatar, user_profile):
    new_avatar_hash: str = user_avatar_content_hash(ldap_avatar)
    if user_profile.avatar_hash and user_profile.avatar_hash == new_avatar_hash:
        return False
    return True

def get_avatar_for_inaccessible_user():
    return staticfiles_storage.url('images/unknown-user-avatar.png')