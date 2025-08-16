from django.db.models.query import QuerySet
from zerver.models import Realm, RealmFilter, UserProfile
from zerver.models.linkifiers import LinkifierDict
from zerver.models.realm_audit_logs import AuditLogEventType
from typing import List, Optional

def notify_linkifiers(realm: Realm, realm_linkifiers: List[RealmFilter]) -> None:
    ...

def do_add_linkifier(realm: Realm, pattern: str, url_template: str, *, acting_user: UserProfile) -> int:
    ...

def do_remove_linkifier(realm: Realm, pattern: Optional[str] = None, id: Optional[int] = None, *, acting_user: Optional[UserProfile] = None) -> None:
    ...

def do_update_linkifier(realm: Realm, id: int, pattern: str, url_template: str, *, acting_user: UserProfile) -> None:
    ...

def check_reorder_linkifiers(realm: Realm, ordered_linkifier_ids: List[int], *, acting_user: UserProfile) -> None:
    ...
