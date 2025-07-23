import uuid
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from operator import attrgetter
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union, cast
from urllib.parse import urlencode, urlsplit
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.db.models import Q
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.utils.timesince import timesince
from django.utils.timezone import now as timezone_now
from pydantic import AfterValidator, Json, NonNegativeInt
from confirmation.models import Confirmation, confirmation_url
from confirmation.settings import STATUS_USED
from corporate.lib.activity import format_optional_datetime, realm_support_link, remote_installation_stats_link
from corporate.lib.billing_types import BillingModality
from corporate.models import CustomerPlan
from zerver.actions.create_realm import do_change_realm_subdomain
from zerver.actions.realm_settings import do_change_realm_max_invites, do_change_realm_org_type, do_change_realm_plan_type, do_deactivate_realm, do_scrub_realm, do_send_realm_reactivation_email
from zerver.actions.users import do_delete_user_preserving_messages
from zerver.decorator import require_server_admin, zulip_login_required
from zerver.forms import check_subdomain_available
from zerver.lib.rate_limiter import rate_limit_request_by_ip
from zerver.lib.realm_icon import realm_icon_url
from zerver.lib.send_email import FromAddress, send_email
from zerver.lib.subdomains import get_subdomain_from_hostname
from zerver.lib.typed_endpoint import ApiParamConfig, typed_endpoint, typed_endpoint_without_parameters
from zerver.lib.validator import check_date
from zerver.models import MultiuseInvite, PreregistrationRealm, PreregistrationUser, Realm, RealmReactivationStatus, UserProfile
from zerver.models.realms import get_default_max_invites_for_realm_plan_type, get_org_type_display_name, get_realm
from zerver.models.users import get_user_profile_by_id
from zerver.views.invite import get_invitee_emails_set
from zilencer.lib.remote_counts import MissingDataError, compute_max_monthly_messages
from zilencer.models import RemoteRealm, RemoteRealmBillingUser, RemoteServerBillingUser, RemoteZulipServer

class SupportRequestForm(forms.Form):
    MAX_SUBJECT_LENGTH: int = 50
    request_subject = forms.CharField(max_length=MAX_SUBJECT_LENGTH)
    request_message = forms.CharField(widget=forms.Textarea)

class DemoRequestForm(forms.Form):
    MAX_INPUT_LENGTH: int = 50
    SORTED_ORG_TYPE_NAMES: List[str] = sorted([org_type['name'] for org_type in Realm.ORG_TYPES.values() if not org_type['hidden']])
    full_name = forms.CharField(max_length=MAX_INPUT_LENGTH)
    email = forms.EmailField()
    role = forms.CharField(max_length=MAX_INPUT_LENGTH)
    organization_name = forms.CharField(max_length=MAX_INPUT_LENGTH)
    organization_type = forms.CharField()
    organization_website = forms.URLField(required=True, assume_scheme='https')
    expected_user_count = forms.CharField(max_length=MAX_INPUT_LENGTH)
    message = forms.CharField(widget=forms.Textarea)

class SalesRequestForm(forms.Form):
    MAX_INPUT_LENGTH: int = 50
    organization_website = forms.URLField(required=True, assume_scheme='https')
    expected_user_count = forms.CharField(max_length=MAX_INPUT_LENGTH)
    message = forms.CharField(widget=forms.Textarea)

@zulip_login_required
@typed_endpoint_without_parameters
def support_request(request: HttpRequest) -> HttpResponse:
    from corporate.lib.stripe import build_support_url
    user = request.user
    assert user.is_authenticated
    context: Dict[str, Any] = {'email': user.delivery_email, 'realm_name': user.realm.name, 'MAX_SUBJECT_LENGTH': SupportRequestForm.MAX_SUBJECT_LENGTH}
    if request.POST:
        post_data = request.POST.copy()
        form = SupportRequestForm(post_data)
        if form.is_valid():
            email_context = {'requested_by': user.full_name, 'realm_string_id': user.realm.string_id, 'request_subject': form.cleaned_data['request_subject'], 'request_message': form.cleaned_data['request_message'], 'support_url': build_support_url('support', user.realm.string_id), 'user_role': user.get_role_name()}
            send_email('zerver/emails/support_request', to_emails=[FromAddress.SUPPORT], from_name='Zulip support request', from_address=FromAddress.tokenized_no_reply_address(), reply_to_email=user.delivery_email, context=email_context)
            response = render(request, 'corporate/support/support_request_thanks.html', context=context)
            return response
    response = render(request, 'corporate/support/support_request.html', context=context)
    return response

@typed_endpoint_without_parameters
def demo_request(request: HttpRequest) -> HttpResponse:
    from corporate.lib.stripe import BILLING_SUPPORT_EMAIL
    context: Dict[str, Any] = {'MAX_INPUT_LENGTH': DemoRequestForm.MAX_INPUT_LENGTH, 'SORTED_ORG_TYPE_NAMES': DemoRequestForm.SORTED_ORG_TYPE_NAMES}
    if request.POST:
        post_data = request.POST.copy()
        form = DemoRequestForm(post_data)
        if form.is_valid():
            rate_limit_request_by_ip(request, domain='sends_email_by_ip')
            email_context = {'full_name': form.cleaned_data['full_name'], 'email': form.cleaned_data['email'], 'role': form.cleaned_data['role'], 'organization_name': form.cleaned_data['organization_name'], 'organization_type': form.cleaned_data['organization_type'], 'organization_website': form.cleaned_data['organization_website'], 'expected_user_count': form.cleaned_data['expected_user_count'], 'message': form.cleaned_data['message']}
            send_email('zerver/emails/demo_request', to_emails=[BILLING_SUPPORT_EMAIL], from_name='Zulip demo request', from_address=FromAddress.tokenized_no_reply_address(), reply_to_email=email_context['email'], context=email_context)
            response = render(request, 'corporate/support/support_request_thanks.html', context=context)
            return response
    response = render(request, 'corporate/support/demo_request.html', context=context)
    return response

@zulip_login_required
@typed_endpoint_without_parameters
def sales_support_request(request: HttpRequest) -> HttpResponse:
    from corporate.lib.stripe import BILLING_SUPPORT_EMAIL
    assert request.user.is_authenticated
    if not request.user.is_realm_admin:
        return render(request, '404.html', status=404)
    context: Dict[str, Any] = {'MAX_INPUT_LENGTH': SalesRequestForm.MAX_INPUT_LENGTH, 'user_email': request.user.delivery_email, 'user_full_name': request.user.full_name}
    if request.POST:
        post_data = request.POST.copy()
        form = SalesRequestForm(post_data)
        if form.is_valid():
            rate_limit_request_by_ip(request, domain='sends_email_by_ip')
            email_context = {'full_name': request.user.full_name, 'email': request.user.delivery_email, 'role': UserProfile.ROLE_ID_TO_API_NAME[request.user.role], 'organization_name': request.user.realm.name, 'organization_type': get_org_type_display_name(request.user.realm.org_type), 'organization_website': form.cleaned_data['organization_website'], 'expected_user_count': form.cleaned_data['expected_user_count'], 'message': form.cleaned_data['message'], 'support_link': realm_support_link(request.user.realm.string_id)}
            send_email('zerver/emails/sales_support_request', to_emails=[BILLING_SUPPORT_EMAIL], from_name='Sales support request', from_address=FromAddress.tokenized_no_reply_address(), reply_to_email=email_context['email'], context=email_context)
            response = render(request, 'corporate/support/support_request_thanks.html', context=context)
            return response
    response = render(request, 'corporate/support/sales_support_request.html', context=context)
    return response

def get_plan_type_string(plan_type: int) -> str:
    return {Realm.PLAN_TYPE_SELF_HOSTED: 'Self-hosted', Realm.PLAN_TYPE_LIMITED: 'Limited', Realm.PLAN_TYPE_STANDARD: 'Standard', Realm.PLAN_TYPE_STANDARD_FREE: 'Standard free', Realm.PLAN_TYPE_PLUS: 'Plus', RemoteZulipServer.PLAN_TYPE_SELF_MANAGED: 'Free', RemoteZulipServer.PLAN_TYPE_SELF_MANAGED_LEGACY: CustomerPlan.name_from_tier(CustomerPlan.TIER_SELF_HOSTED_LEGACY), RemoteZulipServer.PLAN_TYPE_COMMUNITY: 'Community', RemoteZulipServer.PLAN_TYPE_BASIC: 'Basic', RemoteZulipServer.PLAN_TYPE_BUSINESS: 'Business', RemoteZulipServer.PLAN_TYPE_ENTERPRISE: 'Enterprise'}[plan_type]

def get_confirmations(types: List[int], object_ids: List[int], hostname: Optional[str] = None) -> List[Dict[str, Any]]:
    lowest_datetime = timezone_now() - timedelta(days=30)
    confirmations = Confirmation.objects.filter(type__in=types, object_id__in=object_ids, date_sent__gte=lowest_datetime)
    confirmation_dicts: List[Dict[str, Any]] = []
    for confirmation in confirmations:
        realm = confirmation.realm
        content_object = confirmation.content_object
        type = confirmation.type
        expiry_date = confirmation.expiry_date
        assert content_object is not None
        if hasattr(content_object, 'status'):
            if content_object.status == STATUS_USED:
                link_status = 'Link has been used'
            else:
                link_status = 'Link has not been used'
        else:
            link_status = ''
        now = timezone_now()
        if expiry_date is None:
            expires_in = 'Never'
        elif now < expiry_date:
            expires_in = timesince(now, expiry_date)
        else:
            expires_in = 'Expired'
        url = confirmation_url(confirmation.confirmation_key, realm, type)
        confirmation_dicts.append({'object': confirmation.content_object, 'url': url, 'type': type, 'link_status': link_status, 'expires_in': expires_in})
    return confirmation_dicts

@dataclass
class SupportSelectOption:
    name: str
    value: Any

def get_remote_plan_tier_options() -> List[SupportSelectOption]:
    remote_plan_tiers = [SupportSelectOption('None', 0), SupportSelectOption(CustomerPlan.name_from_tier(CustomerPlan.TIER_SELF_HOSTED_BASIC), CustomerPlan.TIER_SELF_HOSTED_BASIC), SupportSelectOption(CustomerPlan.name_from_tier(CustomerPlan.TIER_SELF_HOSTED_BUSINESS), CustomerPlan.TIER_SELF_HOSTED_BUSINESS)]
    return remote_plan_tiers

def get_realm_plan_type_options() -> List[SupportSelectOption]:
    plan_types = [SupportSelectOption(get_plan_type_string(Realm.PLAN_TYPE_SELF_HOSTED), Realm.PLAN_TYPE_SELF_HOSTED), SupportSelectOption(get_plan_type_string(Realm.PLAN_TYPE_LIMITED), Realm.PLAN_TYPE_LIMITED), SupportSelectOption(get_plan_type_string(Realm.PLAN_TYPE_STANDARD), Realm.PLAN_TYPE_STANDARD), SupportSelectOption(get_plan_type_string(Realm.PLAN_TYPE_STANDARD_FREE), Realm.PLAN_TYPE_STANDARD_FREE), SupportSelectOption(get_plan_type_string(Realm.PLAN_TYPE_PLUS), Realm.PLAN_TYPE_PLUS)]
    return plan_types

def get_realm_plan_type_options_for_discount() -> List[SupportSelectOption]:
    plan_types = [SupportSelectOption('None', 0), SupportSelectOption(CustomerPlan.name_from_tier(CustomerPlan.TIER_CLOUD_STANDARD), CustomerPlan.TIER_CLOUD_STANDARD), SupportSelectOption(CustomerPlan.name_from_tier(CustomerPlan.TIER_CLOUD_PLUS), CustomerPlan.TIER_CLOUD_PLUS)]
    return plan_types

def get_default_max_invites_for_plan_type(realm: Realm) -> int:
    default_max = get_default_max_invites_for_realm_plan_type(realm.plan_type)
    if default_max is None:
        return settings.INVITES_DEFAULT_REALM_DAILY_MAX
    return default_max

def check_update_max_invites(realm: Realm, new_max: int, default_max: int) -> bool:
    if new_max in [0, default_max]:
        return realm.max_invites != default_max
    return new_max > default_max

ModifyPlan = Literal['downgrade_at_billing_cycle_end', 'downgrade_now_without_additional_licenses', 'downgrade_now_void_open_invoices', 'upgrade_plan_tier']
RemoteServerStatus = Literal['active', 'deactivated']

def shared_support_context() -> Dict[str, Any]:
    from corporate.lib.stripe import cents_to_dollar_string
    return {'get_org_type_display_name': get_org_type_display_name, 'get_plan_type_name': get_plan_type_string, 'dollar_amount': cents_to_dollar_string}

@require_server_admin
@typed_endpoint
def support(
    request: HttpRequest,
    *,
    realm_id: Optional[int] = None,
    plan_type: Optional[int] = None,
    monthly_discounted_price: Optional[int] = None,
    annual_discounted_price: Optional[int] = None,
    minimum_licenses: Optional[int] = None,
    required_plan_tier: Optional[int] = None,
    new_subdomain: Optional[str] = None,
    status: Optional[str] = None,
    billing_modality: Optional[str] = None,
    sponsorship_pending: Optional[bool] = None,
    approve_sponsorship: bool = False,
    modify_plan: Optional[ModifyPlan] = None,
    scrub_realm: bool = False,
    delete_user_by_id: Optional[int] = None,
    query: Optional[str] = None,
    org_type: Optional[int] = None,
    max_invites: Optional[int] = None,
    plan_end_date: Optional[str] = None,
    fixed_price: Optional[int] = None,
    sent_invoice_id: Optional[str] = None,
    delete_fixed_price_next_plan: bool = False,
) -> HttpResponse:
    from corporate.lib.stripe import RealmBillingSession, SupportRequestError, SupportType, SupportViewRequest
    from corporate.lib.support import CloudSupportData, get_data_for_cloud_support_view
    context = shared_support_context()
    if 'success_message' in request.session:
        context['success_message'] = request.session['success_message']
        del request.session['success_message']
    acting_user = request.user
    assert isinstance(acting_user, UserProfile)
    if settings.BILLING_ENABLED and request.method == 'POST':
        keys = set(request.POST.keys())
        keys.discard('csrfmiddlewaretoken')
        assert realm_id is not None
        realm = Realm.objects.get(id=realm_id)
        support_view_request = None
        if approve_sponsorship:
            support_view_request = SupportViewRequest(support_type=SupportType.approve_sponsorship)
        elif sponsorship_pending is not None:
            support_view_request = SupportViewRequest(support_type=SupportType.update_sponsorship_status, sponsorship_status=sponsorship_pending)
        elif monthly_discounted_price is not None or annual_discounted_price is not None:
            support_view_request = SupportViewRequest(support_type=SupportType.attach_discount, monthly_discounted_price=monthly_discounted_price, annual_discounted_price=annual_discounted_price)
        elif minimum_licenses is not None:
            support_view_request = SupportViewRequest(support_type=SupportType.update_minimum_licenses, minimum_licenses=minimum_licenses)
        elif required_plan_tier is not None:
            support_view_request = SupportViewRequest(support_type=SupportType.update_required_plan_tier, required_plan_tier=required_plan_tier)
        elif billing_modality is not None:
            support_view_request = SupportViewRequest(support_type=SupportType.update_billing_modality, billing_modality=billing_modality)
        elif modify_plan is not None:
            support_view_request = SupportViewRequest(support_type=SupportType.modify_plan, plan_modification=modify_plan)
            if modify_plan == 'upgrade_plan_tier':
                support_view_request['new_plan_tier'] = CustomerPlan.TIER_CLOUD_PLUS
        elif plan_end_date is not None:
            support_view_request = SupportViewRequest(support_type=SupportType.update_plan_end_date, plan_end_date=plan_end_date)
        elif fixed_price is not None:
            if sent_invoice_id is not None and sent_invoice_id.strip() == '':
                sent_invoice_id = None
            support_view_request = SupportViewRequest(support_type=SupportType.configure_fixed_price_plan, fixed_price=fixed_price, sent_invoice_id=sent_invoice_id)
        elif delete_fixed_price_next_plan:
            support_view_request = SupportViewRequest(support_type=SupportType.delete_fixed_price_next_plan)
        elif plan_type is not None:
            current_plan_type = realm.plan_type
            do_change_realm_plan_type(realm, plan_type, acting_user=acting_user)
            msg = f'Plan type of {realm.string_id} changed from {get_plan_type_string(current_plan_type)} to {get_plan_type_string(plan_type)} '
            context['