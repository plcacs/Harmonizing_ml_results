from django.db.models.query import QuerySet
from zerver.models import Realm, RealmFilter, UserProfile
from zerver.models.linkifiers import LinkifierDict
from zerver.models.realm_audit_logs import AuditLogEventType
from typing import List, Optional

def notify_linkifiers(realm: Realm, realm_linkifiers: List[RealmFilter]) -> None:
    event: dict = dict(type='realm_linkifiers', realm_linkifiers=realm_linkifiers)

def do_add_linkifier(realm: Realm, pattern: str, url_template: str, *, acting_user: UserProfile) -> int:
    max_order: Optional[int] = RealmFilter.objects.aggregate(Max('order'))['order__max']
    linkifier: RealmFilter
    realm_linkifiers: List[RealmFilter]

def do_remove_linkifier(realm: Realm, pattern: Optional[str] = None, id: Optional[int] = None, *, acting_user: Optional[UserProfile] = None) -> None:
    realm_linkifier: RealmFilter
    pattern: str
    url_template: str

def do_update_linkifier(realm: Realm, id: int, pattern: str, url_template: str, *, acting_user: UserProfile) -> None:
    linkifier: RealmFilter
    realm_linkifiers: List[RealmFilter]

def check_reorder_linkifiers(realm: Realm, ordered_linkifier_ids: List[int], *, acting_user: UserProfile) -> None:
    linkifier_id_set: set
    linkifiers: QuerySet[RealmFilter]
    id_to_new_order: dict
