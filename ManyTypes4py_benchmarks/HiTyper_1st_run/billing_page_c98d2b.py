import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal
from django.http import HttpRequest, HttpResponse, HttpResponseNotAllowed, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.utils.translation import gettext as _
from pydantic import AfterValidator, Json
from corporate.lib.decorator import authenticated_remote_realm_management_endpoint, authenticated_remote_server_management_endpoint
from corporate.models import CustomerPlan, get_current_plan_by_customer, get_customer_by_realm
from zerver.decorator import process_as_post, require_billing_access, zulip_login_required
from zerver.lib.exceptions import JsonableError
from zerver.lib.response import json_success
from zerver.lib.typed_endpoint import typed_endpoint
from zerver.lib.typed_endpoint_validators import check_int_in
from zerver.models import UserProfile
from zilencer.lib.remote_counts import MissingDataError
from zilencer.models import RemoteRealm, RemoteZulipServer
if TYPE_CHECKING:
    from corporate.lib.stripe import RemoteRealmBillingSession, RemoteServerBillingSession
billing_logger = logging.getLogger('corporate.stripe')
ALLOWED_PLANS_API_STATUS_VALUES = [CustomerPlan.ACTIVE, CustomerPlan.DOWNGRADE_AT_END_OF_CYCLE, CustomerPlan.SWITCH_TO_ANNUAL_AT_END_OF_CYCLE, CustomerPlan.SWITCH_TO_MONTHLY_AT_END_OF_CYCLE, CustomerPlan.FREE_TRIAL, CustomerPlan.DOWNGRADE_AT_END_OF_FREE_TRIAL, CustomerPlan.ENDED]

@zulip_login_required
@typed_endpoint
def billing_page(request: django.http.HttpRequest, *, success_message: typing.Text='') -> Union[str, django.http.response.HttpResponse, bytes, HttpResponseRedirect]:
    from corporate.lib.stripe import RealmBillingSession
    user = request.user
    assert user.is_authenticated
    billing_session = RealmBillingSession(user=user, realm=user.realm)
    context = {'admin_access': user.has_billing_access, 'has_active_plan': False, 'org_name': billing_session.org_name(), 'billing_base_url': ''}
    if not user.has_billing_access:
        return render(request, 'corporate/billing/billing.html', context=context)
    if user.realm.plan_type == user.realm.PLAN_TYPE_STANDARD_FREE:
        return HttpResponseRedirect(reverse('sponsorship_request'))
    customer = get_customer_by_realm(user.realm)
    if customer is not None and customer.sponsorship_pending:
        if not billing_session.on_paid_plan():
            return HttpResponseRedirect(reverse('sponsorship_request'))
        context['sponsorship_pending'] = True
    if user.realm.plan_type == user.realm.PLAN_TYPE_LIMITED:
        return HttpResponseRedirect(reverse('plans'))
    if customer is None or get_current_plan_by_customer(customer) is None:
        return HttpResponseRedirect(reverse('upgrade_page'))
    main_context = billing_session.get_billing_page_context()
    if main_context:
        if main_context.get('current_plan_downgraded') is True:
            return HttpResponseRedirect(reverse('plans'))
        context.update(main_context)
        context['success_message'] = success_message
    return render(request, 'corporate/billing/billing.html', context=context)

@typed_endpoint
@authenticated_remote_realm_management_endpoint
def remote_realm_billing_page(request: django.http.HttpRequest, billing_session: Union[str, django.http.HttpRequest], *, success_message: typing.Text='') -> Union[HttpResponseRedirect, django.http.HttpResponse, str, apistar.http.Response]:
    realm_uuid = billing_session.remote_realm.uuid
    context = {'admin_access': billing_session.has_billing_access(), 'has_active_plan': False, 'org_name': billing_session.org_name(), 'billing_base_url': billing_session.billing_base_url}
    if billing_session.remote_realm.plan_type == RemoteRealm.PLAN_TYPE_COMMUNITY:
        return HttpResponseRedirect(reverse('remote_realm_sponsorship_page', args=(realm_uuid,)))
    customer = billing_session.get_customer()
    if customer is not None and customer.sponsorship_pending:
        if not billing_session.on_paid_plan() and billing_session.get_complimentary_access_next_plan_name(customer) is None:
            return HttpResponseRedirect(reverse('remote_realm_sponsorship_page', args=(realm_uuid,)))
        context['sponsorship_pending'] = True
    if customer is None or get_current_plan_by_customer(customer) is None or (billing_session.get_complimentary_access_next_plan_name(customer) is None and billing_session.remote_realm.plan_type in [RemoteRealm.PLAN_TYPE_SELF_MANAGED, RemoteRealm.PLAN_TYPE_SELF_MANAGED_LEGACY]):
        return HttpResponseRedirect(reverse('remote_realm_plans_page', args=(realm_uuid,)))
    try:
        main_context = billing_session.get_billing_page_context()
    except MissingDataError:
        return billing_session.missing_data_error_page(request)
    if main_context:
        if main_context.get('current_plan_downgraded') is True:
            return HttpResponseRedirect(reverse('remote_realm_plans_page', args=(realm_uuid,)))
        context.update(main_context)
        context['success_message'] = success_message
    return render(request, 'corporate/billing/billing.html', context=context)

@typed_endpoint
@authenticated_remote_server_management_endpoint
def remote_server_billing_page(request: Union[django.http.HttpRequest, django.http.requesHttpRequest], billing_session: Union[str, django.db.models.fields.Field, utils.clienClient], *, success_message: typing.Text='') -> Union[HttpResponseRedirect, django.http.HttpResponse, str]:
    context = {'admin_access': billing_session.has_billing_access(), 'has_active_plan': False, 'org_name': billing_session.org_name(), 'billing_base_url': billing_session.billing_base_url}
    if billing_session.remote_server.plan_type == RemoteZulipServer.PLAN_TYPE_COMMUNITY:
        return HttpResponseRedirect(reverse('remote_server_sponsorship_page', kwargs={'server_uuid': billing_session.remote_server.uuid}))
    customer = billing_session.get_customer()
    if customer is not None and customer.sponsorship_pending:
        if not billing_session.on_paid_plan() and billing_session.get_complimentary_access_next_plan_name(customer) is None:
            return HttpResponseRedirect(reverse('remote_server_sponsorship_page', kwargs={'server_uuid': billing_session.remote_server.uuid}))
        context['sponsorship_pending'] = True
    if customer is None or get_current_plan_by_customer(customer) is None or (billing_session.get_complimentary_access_next_plan_name(customer) is None and billing_session.remote_server.plan_type in [RemoteZulipServer.PLAN_TYPE_SELF_MANAGED, RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY]):
        return HttpResponseRedirect(reverse('remote_server_upgrade_page', kwargs={'server_uuid': billing_session.remote_server.uuid}))
    try:
        main_context = billing_session.get_billing_page_context()
    except MissingDataError:
        return billing_session.missing_data_error_page(request)
    if main_context:
        if main_context.get('current_plan_downgraded') is True:
            return HttpResponseRedirect(reverse('remote_server_plans_page', kwargs={'server_uuid': billing_session.remote_server.uuid}))
        context.update(main_context)
        context['success_message'] = success_message
    return render(request, 'corporate/billing/billing.html', context=context)

@require_billing_access
@typed_endpoint
def update_plan(request: Union[django.http.HttpRequest, bool, scrapy.http.Request], user: Union[zerver.models.UserProfile, int, str, None], *, status: Union[None, django.http.HttpRequest, scrapy.http.Request, ajapaik.ajapaik_face_recognition.models.FaceRecognitionRectangle]=None, licenses: Union[None, django.http.HttpRequest, scrapy.http.Request, ajapaik.ajapaik_face_recognition.models.FaceRecognitionRectangle]=None, licenses_at_next_renewal: Union[None, django.http.HttpRequest, scrapy.http.Request, ajapaik.ajapaik_face_recognition.models.FaceRecognitionRectangle]=None, schedule: Union[None, django.http.HttpRequest, scrapy.http.Request, ajapaik.ajapaik_face_recognition.models.FaceRecognitionRectangle]=None, toggle_license_management: bool=False) -> Union[str, rotkehlchen.tests.utils.mock.MockResponse, dict]:
    from corporate.lib.stripe import RealmBillingSession, UpdatePlanRequest
    update_plan_request = UpdatePlanRequest(status=status, licenses=licenses, licenses_at_next_renewal=licenses_at_next_renewal, schedule=schedule, toggle_license_management=toggle_license_management)
    billing_session = RealmBillingSession(user=user)
    billing_session.do_update_plan(update_plan_request)
    return json_success(request)

@process_as_post
@typed_endpoint
@authenticated_remote_realm_management_endpoint
def update_plan_for_remote_realm(request: Union[django.http.HttpRequest, dict[str, str], scrapy.http.Request], billing_session: Union[bool, str, lunch_buddies.models.polls.Poll], *, status: Union[None, str, scrapy.http.Request]=None, licenses: Union[None, str, scrapy.http.Request]=None, licenses_at_next_renewal: Union[None, str, scrapy.http.Request]=None, schedule: Union[None, str, scrapy.http.Request]=None, toggle_license_management: bool=False) -> Union[str, bool, django.http.HttpResponse]:
    from corporate.lib.stripe import UpdatePlanRequest
    update_plan_request = UpdatePlanRequest(status=status, licenses=licenses, licenses_at_next_renewal=licenses_at_next_renewal, schedule=schedule, toggle_license_management=toggle_license_management)
    billing_session.do_update_plan(update_plan_request)
    return json_success(request)

@process_as_post
@typed_endpoint
@authenticated_remote_server_management_endpoint
def update_plan_for_remote_server(request: Union[dict, django.http.HttpRequest, dict[str, str]], billing_session: Union[bool, lunch_buddies.models.polls.Poll, str], *, status: Union[None, str]=None, licenses: Union[None, str]=None, licenses_at_next_renewal: Union[None, str]=None, schedule: Union[None, str]=None, toggle_license_management: bool=False) -> Union[bool, str, dict[str, typing.Any]]:
    from corporate.lib.stripe import UpdatePlanRequest
    update_plan_request = UpdatePlanRequest(status=status, licenses=licenses, licenses_at_next_renewal=licenses_at_next_renewal, schedule=schedule, toggle_license_management=toggle_license_management)
    billing_session.do_update_plan(update_plan_request)
    return json_success(request)

@typed_endpoint
@authenticated_remote_server_management_endpoint
def remote_server_deactivate_page(request: django.http.HttpRequest, billing_session: Union[str, dict, zerver.models.UserProfile, zilencer.models.RemoteZulipServer], *, confirmed: Union[None, dict[str, typing.Any], dict]=None) -> Union[HttpResponseNotAllowed, django.http.HttpResponse, str]:
    from corporate.lib.stripe import ServerDeactivateWithExistingPlanError, do_deactivate_remote_server
    if request.method not in ['GET', 'POST']:
        return HttpResponseNotAllowed(['GET', 'POST'])
    remote_server = billing_session.remote_server
    context = {'server_hostname': remote_server.hostname, 'action_url': reverse(remote_server_deactivate_page, args=[str(remote_server.uuid)])}
    if request.method == 'GET':
        return render(request, 'corporate/billing/remote_billing_server_deactivate.html', context=context)
    assert request.method == 'POST'
    if confirmed is None:
        raise JsonableError(_("Parameter 'confirmed' is required"))
    try:
        do_deactivate_remote_server(remote_server, billing_session)
    except ServerDeactivateWithExistingPlanError:
        context['show_existing_plan_error'] = 'true'
        return render(request, 'corporate/billing/remote_billing_server_deactivate.html', context=context)
    return render(request, 'corporate/billing/remote_billing_server_deactivated_success.html', context={'server_hostname': remote_server.hostname})