from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from typing import TYPE_CHECKING
from zerver.models import UserProfile
from zerver.lib.exceptions import JsonableError
from zerver.lib.response import json_success

if TYPE_CHECKING:
    from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession

def billing_page(request: HttpRequest, *, success_message: str = '') -> HttpResponse:
    ...

def remote_realm_billing_page(request: HttpRequest, billing_session: 'RemoteRealmBillingSession', *, success_message: str = '') -> HttpResponse:
    ...

def remote_server_billing_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession', *, success_message: str = '') -> HttpResponse:
    ...

def update_plan(request: HttpRequest, user: UserProfile, *, status: str = None, licenses: int = None, licenses_at_next_renewal: int = None, schedule: str = None, toggle_license_management: bool = False) -> HttpResponse:
    ...

def update_plan_for_remote_realm(request: HttpRequest, billing_session: 'RemoteRealmBillingSession', *, status: str = None, licenses: int = None, licenses_at_next_renewal: int = None, schedule: str = None, toggle_license_management: bool = False) -> HttpResponse:
    ...

def update_plan_for_remote_server(request: HttpRequest, billing_session: 'RemoteServerBillingSession', *, status: str = None, licenses: int = None, licenses_at_next_renewal: int = None, schedule: str = None, toggle_license_management: bool = False) -> HttpResponse:
    ...

def remote_server_deactivate_page(request: HttpRequest, billing_session: 'RemoteServerBillingSession', *, confirmed: bool = None) -> HttpResponse:
    ...
