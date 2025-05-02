import logging
from typing import TYPE_CHECKING
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from corporate.lib.decorator import authenticated_remote_realm_management_endpoint, authenticated_remote_server_management_endpoint, self_hosting_management_endpoint
from zerver.decorator import require_organization_member, zulip_login_required
from zerver.lib.response import json_success
from zerver.lib.typed_endpoint import typed_endpoint
from zerver.models import UserProfile
if TYPE_CHECKING:
    from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession
billing_logger = logging.getLogger('corporate.stripe')

@require_organization_member
@typed_endpoint
def event_status(request, user, *, stripe_session_id=None, stripe_invoice_id=None):
    from corporate.lib.stripe import EventStatusRequest, RealmBillingSession
    event_status_request = EventStatusRequest(stripe_session_id=stripe_session_id, stripe_invoice_id=stripe_invoice_id)
    billing_session = RealmBillingSession(user)
    data = billing_session.get_event_status(event_status_request)
    return json_success(request, data)

@typed_endpoint
@authenticated_remote_realm_management_endpoint
def remote_realm_event_status(request, billing_session, *, stripe_session_id=None, stripe_invoice_id=None):
    from corporate.lib.stripe import EventStatusRequest
    event_status_request = EventStatusRequest(stripe_session_id=stripe_session_id, stripe_invoice_id=stripe_invoice_id)
    data = billing_session.get_event_status(event_status_request)
    return json_success(request, data)

@typed_endpoint
@authenticated_remote_server_management_endpoint
def remote_server_event_status(request, billing_session, *, stripe_session_id=None, stripe_invoice_id=None):
    from corporate.lib.stripe import EventStatusRequest
    event_status_request = EventStatusRequest(stripe_session_id=stripe_session_id, stripe_invoice_id=stripe_invoice_id)
    data = billing_session.get_event_status(event_status_request)
    return json_success(request, data)

@zulip_login_required
@typed_endpoint
def event_status_page(request, *, stripe_session_id='', stripe_invoice_id=''):
    context = {'stripe_session_id': stripe_session_id, 'stripe_invoice_id': stripe_invoice_id, 'billing_base_url': ''}
    return render(request, 'corporate/billing/event_status.html', context=context)

@self_hosting_management_endpoint
@typed_endpoint
def remote_realm_event_status_page(request, *, realm_uuid='', server_uuid='', stripe_session_id='', stripe_invoice_id=''):
    context = {'stripe_session_id': stripe_session_id, 'stripe_invoice_id': stripe_invoice_id, 'billing_base_url': f'/realm/{realm_uuid}'}
    return render(request, 'corporate/billing/event_status.html', context=context)

@self_hosting_management_endpoint
@typed_endpoint
def remote_server_event_status_page(request, *, realm_uuid='', server_uuid='', stripe_session_id='', stripe_invoice_id=''):
    context = {'stripe_session_id': stripe_session_id, 'stripe_invoice_id': stripe_invoice_id, 'billing_base_url': f'/server/{server_uuid}'}
    return render(request, 'corporate/billing/event_status.html', context=context)