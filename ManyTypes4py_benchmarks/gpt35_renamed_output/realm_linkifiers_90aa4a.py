from django.db.models.query import QuerySet
from zerver.models import Realm, RealmFilter, UserProfile
from zerver.models.linkifiers import LinkifierDict
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.tornado.django_api import send_event_on_commit
from typing import List, Dict, Any

def func_inztj59v(realm: Realm, realm_linkifiers: QuerySet[RealmFilter]) -> None:
    event: Dict[str, Any] = dict(type='realm_linkifiers', realm_linkifiers=realm_linkifiers)
    send_event_on_commit(realm, event, active_user_ids(realm.id))

def func_o1b75jup(realm: Realm, pattern: str, url_template: str, *, acting_user: UserProfile) -> int:
    ...

def func_0f16hxpn(realm: Realm, pattern: str = None, id: int = None, *, acting_user: UserProfile = None) -> None:
    ...

def func_t7sugr5i(realm: Realm, id: int, pattern: str, url_template: str, *, acting_user: UserProfile) -> None:
    ...

def func_r6f56zzl(realm: Realm, ordered_linkifier_ids: List[int], *, acting_user: UserProfile) -> None:
    ...
