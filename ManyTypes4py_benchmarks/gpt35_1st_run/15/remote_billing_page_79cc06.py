from typing import Any, Literal, TypeAlias
from django.http import HttpRequest, HttpResponse, HttpResponseNotAllowed, HttpResponseRedirect
from confirmation.models import Confirmation, ConfirmationKeyError
from corporate.lib.decorator import self_hosting_management_endpoint
from corporate.lib.remote_billing_util import LegacyServerIdentityDict, RemoteBillingIdentityDict, RemoteBillingUserDict, RemoteBillingIdentityExpiredError, RemoteBillingAuthenticationError
from corporate.models import CustomerPlan, RemoteRealm, RemoteRealmBillingUser, RemoteServerBillingUser, RemoteZulipServer
from zerver.lib.exceptions import JsonableError, MissingRemoteRealmError, RateLimitedError, RemoteRealmServerMismatchError
from zerver.lib.rate_limiter import rate_limit_request_by_ip, rate_limit_remote_server
from zerver.lib.response import json_success
from zerver.lib.send_email import FromAddress, send_email
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.typed_endpoint import typed_endpoint
from zilencer.models import PreregistrationRemoteRealmBillingUser, PreregistrationRemoteServerBillingUser
