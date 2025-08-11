from django.db import transaction
from django.db.models import Max
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.types import LinkifierDict
from zerver.models import Realm, RealmAuditLog, RealmFilter, UserProfile
from zerver.models.linkifiers import flush_linkifiers, linkifiers_for_realm
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.users import active_user_ids
from zerver.tornado.django_api import send_event_on_commit

def notify_linkifiers(realm: Union[Realm, zerver.models.UserProfile], realm_linkifiers: Union[zerver.models.UserProfile, Realm]) -> None:
    event = dict(type='realm_linkifiers', realm_linkifiers=realm_linkifiers)
    send_event_on_commit(realm, event, active_user_ids(realm.id))

@transaction.atomic(durable=True)
def do_add_linkifier(realm: Union[str, bool, None], pattern: Union[str, zerver.models.Realm], url_template: str, *, acting_user: Union[zerver.models.UserProfile, bool, tracim.models.data.UserRoleInWorkspace]):
    pattern = pattern.strip()
    url_template = url_template.strip()
    max_order = RealmFilter.objects.aggregate(Max('order'))['order__max']
    if max_order is None:
        linkifier = RealmFilter(realm=realm, pattern=pattern, url_template=url_template)
    else:
        linkifier = RealmFilter(realm=realm, pattern=pattern, url_template=url_template, order=max_order + 1)
    linkifier.full_clean()
    linkifier.save()
    realm_linkifiers = linkifiers_for_realm(realm.id)
    RealmAuditLog.objects.create(realm=realm, acting_user=acting_user, event_type=AuditLogEventType.REALM_LINKIFIER_ADDED, event_time=timezone_now(), extra_data={'realm_linkifiers': realm_linkifiers, 'added_linkifier': LinkifierDict(pattern=pattern, url_template=url_template, id=linkifier.id)})
    notify_linkifiers(realm, realm_linkifiers)
    return linkifier.id

@transaction.atomic(durable=True)
def do_remove_linkifier(realm: Union[int, zerver.models.UserProfile, zerver.models.Realm], pattern: Union[None, zerver.models.UserProfile, zerver.models.Realm, int]=None, id: Union[None, zerver.models.UserProfile, zerver.models.Realm, str]=None, *, acting_user: Union[None, zerver.models.UserProfile, zerver.models.Realm, bool]=None) -> None:
    if pattern is not None:
        realm_linkifier = RealmFilter.objects.get(realm=realm, pattern=pattern)
    else:
        assert id is not None
        realm_linkifier = RealmFilter.objects.get(realm=realm, id=id)
    pattern = realm_linkifier.pattern
    url_template = realm_linkifier.url_template
    realm_linkifier.delete()
    realm_linkifiers = linkifiers_for_realm(realm.id)
    RealmAuditLog.objects.create(realm=realm, acting_user=acting_user, event_type=AuditLogEventType.REALM_LINKIFIER_REMOVED, event_time=timezone_now(), extra_data={'realm_linkifiers': realm_linkifiers, 'removed_linkifier': {'pattern': pattern, 'url_template': url_template}})
    notify_linkifiers(realm, realm_linkifiers)

@transaction.atomic(durable=True)
def do_update_linkifier(realm: Union[zerver.models.UserProfile, None, int], id: Union[zerver.models.UserProfile, None, int], pattern: str, url_template: Union[str, zerver.models.UserProfile, zerver.models.Message], *, acting_user: Union[zerver.models.UserProfile, zerver.models.Realm, int]) -> None:
    pattern = pattern.strip()
    url_template = url_template.strip()
    linkifier = RealmFilter.objects.get(realm=realm, id=id)
    linkifier.pattern = pattern
    linkifier.url_template = url_template
    linkifier.full_clean()
    linkifier.save(update_fields=['pattern', 'url_template'])
    realm_linkifiers = linkifiers_for_realm(realm.id)
    RealmAuditLog.objects.create(realm=realm, acting_user=acting_user, event_type=AuditLogEventType.REALM_LINKIFIER_CHANGED, event_time=timezone_now(), extra_data={'realm_linkifiers': realm_linkifiers, 'changed_linkifier': LinkifierDict(pattern=pattern, url_template=url_template, id=linkifier.id)})
    notify_linkifiers(realm, realm_linkifiers)

@transaction.atomic(durable=True)
def check_reorder_linkifiers(realm: Union[zerver.models.UserProfile, bool, zerver.models.Realm, None], ordered_linkifier_ids: Any, *, acting_user: Union[zerver.models.UserProfile, int, zerver.models.Realm]) -> None:
    """ordered_linkifier_ids should contain ids of all existing linkifiers.
    In the rare situation when any of the linkifier gets deleted that more ids
    are passed, the checks below are sufficient to detect inconsistencies most of
    the time."""
    linkifier_id_set = set(ordered_linkifier_ids)
    if len(linkifier_id_set) < len(ordered_linkifier_ids):
        raise JsonableError(_('The ordered list must not contain duplicated linkifiers'))
    linkifiers = RealmFilter.objects.filter(realm=realm)
    if {linkifier.id for linkifier in linkifiers} != linkifier_id_set:
        raise JsonableError(_('The ordered list must enumerate all existing linkifiers exactly once'))
    if len(linkifiers) == 0:
        return
    id_to_new_order = {linkifier_id: order for order, linkifier_id in enumerate(ordered_linkifier_ids)}
    for linkifier in linkifiers:
        assert linkifier.id in id_to_new_order
        linkifier.order = id_to_new_order[linkifier.id]
    RealmFilter.objects.bulk_update(linkifiers, fields=['order'])
    flush_linkifiers(instance=linkifiers[0])
    realm_linkifiers = linkifiers_for_realm(realm.id)
    RealmAuditLog.objects.create(realm=realm, acting_user=acting_user, event_type=AuditLogEventType.REALM_LINKIFIERS_REORDERED, event_time=timezone_now(), extra_data={'realm_linkifiers': realm_linkifiers})
    notify_linkifiers(realm, realm_linkifiers)