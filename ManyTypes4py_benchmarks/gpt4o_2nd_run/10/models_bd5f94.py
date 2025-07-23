__revision__ = '$Id: models.py 28 2009-10-22 15:03:02Z jarek.zgoda $'
import secrets
from base64 import b32encode
from collections.abc import Mapping
from datetime import timedelta
from typing import Optional, TypeAlias, Union, cast
from urllib.parse import urljoin
from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.db.models import CASCADE
from django.http import HttpRequest, HttpResponse
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils.timezone import now as timezone_now
from typing_extensions import override
from confirmation import settings as confirmation_settings
from zerver.lib.types import UNSET, Unset
from zerver.models import EmailChangeStatus, MultiuseInvite, PreregistrationRealm, PreregistrationUser, Realm, RealmReactivationStatus, UserProfile
if settings.ZILENCER_ENABLED:
    from zilencer.models import PreregistrationRemoteRealmBillingUser, PreregistrationRemoteServerBillingUser

class ConfirmationKeyError(Exception):
    WRONG_LENGTH = 1
    EXPIRED = 2
    DOES_NOT_EXIST = 3

    def __init__(self, error_type: int) -> None:
        super().__init__()
        self.error_type = error_type

def render_confirmation_key_error(request: HttpRequest, exception: ConfirmationKeyError) -> TemplateResponse:
    if exception.error_type == ConfirmationKeyError.WRONG_LENGTH:
        return TemplateResponse(request, 'confirmation/link_malformed.html', status=404)
    if exception.error_type == ConfirmationKeyError.EXPIRED:
        return TemplateResponse(request, 'confirmation/link_expired.html', status=404)
    return TemplateResponse(request, 'confirmation/link_does_not_exist.html', status=404)

def generate_key() -> str:
    return b32encode(secrets.token_bytes(15)).decode().lower()

NoZilencerConfirmationObjT: TypeAlias = MultiuseInvite | PreregistrationRealm | PreregistrationUser | EmailChangeStatus | UserProfile | RealmReactivationStatus
ZilencerConfirmationObjT: TypeAlias = Union[NoZilencerConfirmationObjT, 'PreregistrationRemoteServerBillingUser', 'PreregistrationRemoteRealmBillingUser']
ConfirmationObjT: TypeAlias = NoZilencerConfirmationObjT | ZilencerConfirmationObjT

def get_object_from_key(confirmation_key: str, confirmation_types: list[int], *, mark_object_used: bool, allow_used: bool = False) -> ConfirmationObjT:
    if len(confirmation_key) not in (24, 40):
        raise ConfirmationKeyError(ConfirmationKeyError.WRONG_LENGTH)
    try:
        confirmation = Confirmation.objects.get(confirmation_key=confirmation_key, type__in=confirmation_types)
    except Confirmation.DoesNotExist:
        raise ConfirmationKeyError(ConfirmationKeyError.DOES_NOT_EXIST)
    if confirmation.expiry_date is not None and timezone_now() > confirmation.expiry_date:
        raise ConfirmationKeyError(ConfirmationKeyError.EXPIRED)
    obj = confirmation.content_object
    assert obj is not None
    forbidden_statuses = {confirmation_settings.STATUS_REVOKED}
    if not allow_used:
        forbidden_statuses.add(confirmation_settings.STATUS_USED)
    if hasattr(obj, 'status') and obj.status in forbidden_statuses:
        raise ConfirmationKeyError(ConfirmationKeyError.EXPIRED)
    if mark_object_used:
        assert confirmation.type != Confirmation.MULTIUSE_INVITE
        assert hasattr(obj, 'status')
        obj.status = getattr(settings, 'STATUS_USED', 1)
        obj.save(update_fields=['status'])
    return obj

def create_confirmation_object(obj: ConfirmationObjT, confirmation_type: int, *, validity_in_minutes: Union[Unset, Optional[int]] = UNSET, no_associated_realm_object: bool = False) -> Confirmation:
    key = generate_key()
    if no_associated_realm_object:
        realm = None
    else:
        obj = cast(NoZilencerConfirmationObjT, obj)
        assert not isinstance(obj, PreregistrationRealm)
        realm = obj.realm
    current_time = timezone_now()
    expiry_date = None
    if not isinstance(validity_in_minutes, Unset):
        if validity_in_minutes is None:
            expiry_date = None
        else:
            assert validity_in_minutes is not None
            expiry_date = current_time + timedelta(minutes=validity_in_minutes)
    else:
        expiry_date = current_time + timedelta(days=_properties[confirmation_type].validity_in_days)
    return Confirmation.objects.create(content_object=obj, date_sent=current_time, confirmation_key=key, realm=realm, expiry_date=expiry_date, type=confirmation_type)

def create_confirmation_link(obj: ConfirmationObjT, confirmation_type: int, *, validity_in_minutes: Union[Unset, Optional[int]] = UNSET, url_args: dict = {}, no_associated_realm_object: bool = False) -> str:
    return confirmation_url_for(create_confirmation_object(obj, confirmation_type, validity_in_minutes=validity_in_minutes, no_associated_realm_object=no_associated_realm_object), url_args=url_args)

def confirmation_url_for(confirmation_obj: Confirmation, url_args: dict = {}) -> str:
    return confirmation_url(confirmation_obj.confirmation_key, confirmation_obj.realm, confirmation_obj.type, url_args)

def confirmation_url(confirmation_key: str, realm: Optional[Realm], confirmation_type: int, url_args: dict = {}) -> str:
    url_args = dict(url_args)
    url_args['confirmation_key'] = confirmation_key
    return urljoin(settings.ROOT_DOMAIN_URI if realm is None else realm.url, reverse(_properties[confirmation_type].url_name, kwargs=url_args))

class Confirmation(models.Model):
    content_type = models.ForeignKey(ContentType, on_delete=CASCADE)
    object_id = models.PositiveBigIntegerField(db_index=True)
    content_object = GenericForeignKey('content_type', 'object_id')
    date_sent = models.DateTimeField(db_index=True)
    confirmation_key = models.CharField(max_length=40, db_index=True)
    expiry_date = models.DateTimeField(db_index=True, null=True)
    realm = models.ForeignKey(Realm, null=True, on_delete=CASCADE)
    USER_REGISTRATION = 1
    INVITATION = 2
    EMAIL_CHANGE = 3
    UNSUBSCRIBE = 4
    SERVER_REGISTRATION = 5
    MULTIUSE_INVITE = 6
    REALM_CREATION = 7
    REALM_REACTIVATION = 8
    REMOTE_SERVER_BILLING_LEGACY_LOGIN = 9
    REMOTE_REALM_BILLING_LEGACY_LOGIN = 10
    type = models.PositiveSmallIntegerField()

    class Meta:
        unique_together = ('type', 'confirmation_key')
        indexes = [models.Index(fields=['content_type', 'object_id'])]

    @override
    def __str__(self) -> str:
        return f'{self.content_object!r}'

class ConfirmationType:
    def __init__(self, url_name: str, validity_in_days: int = settings.CONFIRMATION_LINK_DEFAULT_VALIDITY_DAYS) -> None:
        self.url_name = url_name
        self.validity_in_days = validity_in_days

_properties: dict[int, ConfirmationType] = {
    Confirmation.USER_REGISTRATION: ConfirmationType('get_prereg_key_and_redirect'),
    Confirmation.INVITATION: ConfirmationType('get_prereg_key_and_redirect', validity_in_days=settings.INVITATION_LINK_VALIDITY_DAYS),
    Confirmation.EMAIL_CHANGE: ConfirmationType('confirm_email_change'),
    Confirmation.UNSUBSCRIBE: ConfirmationType('unsubscribe', validity_in_days=1000000),
    Confirmation.MULTIUSE_INVITE: ConfirmationType('join', validity_in_days=settings.INVITATION_LINK_VALIDITY_DAYS),
    Confirmation.REALM_CREATION: ConfirmationType('get_prereg_key_and_redirect'),
    Confirmation.REALM_REACTIVATION: ConfirmationType('realm_reactivation')
}
if settings.ZILENCER_ENABLED:
    _properties[Confirmation.REMOTE_SERVER_BILLING_LEGACY_LOGIN] = ConfirmationType('remote_billing_legacy_server_from_login_confirmation_link')
    _properties[Confirmation.REMOTE_REALM_BILLING_LEGACY_LOGIN] = ConfirmationType('remote_realm_billing_from_login_confirmation_link')

def one_click_unsubscribe_link(user_profile: UserProfile, email_type: str) -> str:
    return create_confirmation_link(user_profile, Confirmation.UNSUBSCRIBE, url_args={'email_type': email_type})

def validate_key(creation_key: Optional[str]) -> Optional['RealmCreationKey']:
    if creation_key is None:
        return None
    try:
        key_record = RealmCreationKey.objects.get(creation_key=creation_key)
    except RealmCreationKey.DoesNotExist:
        raise RealmCreationKey.InvalidError
    time_elapsed = timezone_now() - key_record.date_created
    if time_elapsed.total_seconds() > settings.REALM_CREATION_LINK_VALIDITY_DAYS * 24 * 3600:
        raise RealmCreationKey.InvalidError
    return key_record

def generate_realm_creation_url(by_admin: bool = False) -> str:
    key = generate_key()
    RealmCreationKey.objects.create(creation_key=key, date_created=timezone_now(), presume_email_valid=by_admin)
    return urljoin(settings.ROOT_DOMAIN_URI, reverse('create_realm', kwargs={'creation_key': key}))

class RealmCreationKey(models.Model):
    creation_key = models.CharField('activation key', db_index=True, max_length=40)
    date_created = models.DateTimeField('created', default=timezone_now)
    presume_email_valid = models.BooleanField(default=False)

    class InvalidError(Exception):
        pass
