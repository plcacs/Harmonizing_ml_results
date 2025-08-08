from typing import Optional, TypeAlias, Union
from zerver.models import EmailChangeStatus, MultiuseInvite, PreregistrationRealm, PreregistrationUser, Realm, RealmReactivationStatus, UserProfile

NoZilencerConfirmationObjT = Union[MultiuseInvite, PreregistrationRealm, PreregistrationUser, EmailChangeStatus, UserProfile, RealmReactivationStatus]
ZilencerConfirmationObjT = Union[NoZilencerConfirmationObjT, 'PreregistrationRemoteServerBillingUser', 'PreregistrationRemoteRealmBillingUser']
ConfirmationObjT = Union[NoZilencerConfirmationObjT, ZilencerConfirmationObjT]

def get_object_from_key(confirmation_key: str, confirmation_types: list, *, mark_object_used: bool, allow_used: bool = False) -> ConfirmationObjT:
    ...

def create_confirmation_object(obj: NoZilencerConfirmationObjT, confirmation_type: int, *, validity_in_minutes: Optional[int] = UNSET, no_associated_realm_object: bool = False) -> Confirmation:
    ...

def create_confirmation_link(obj: NoZilencerConfirmationObjT, confirmation_type: int, *, validity_in_minutes: Optional[int] = UNSET, url_args: dict = {}, no_associated_realm_object: bool = False) -> str:
    ...

def confirmation_url_for(confirmation_obj: Confirmation, url_args: dict = {}) -> str:
    ...

def confirmation_url(confirmation_key: str, realm: Optional[Realm], confirmation_type: int, url_args: dict = {}) -> str:
    ...

def one_click_unsubscribe_link(user_profile: UserProfile, email_type: str) -> str:
    ...

def validate_key(creation_key: Optional[str]) -> Optional[RealmCreationKey]:
    ...

def generate_realm_creation_url(by_admin: bool = False) -> str:
    ...
