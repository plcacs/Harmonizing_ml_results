import logging
import math
import os
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, IntEnum
from functools import wraps
from typing import Any, Literal, TypedDict, TypeVar
from urllib.parse import urlencode, urljoin

import stripe
from django import forms
from django.conf import settings
from django.core import signing
from django.core.signing import Signer
from django.db import transaction
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.utils.translation import override as override_language
from typing_extensions import ParamSpec, override

from corporate.lib.billing_types import BillingModality, BillingSchedule, LicenseManagement
from corporate.models import (
    Customer,
    CustomerPlan,
    CustomerPlanOffer,
    Invoice,
    LicenseLedger,
    Session,
    SponsoredPlanTypes,
    ZulipSponsorshipRequest,
    get_current_plan_by_customer,
    get_current_plan_by_realm,
    get_customer_by_realm,
    get_customer_by_remote_realm,
    get_customer_by_remote_server,
)
from zerver.lib.cache import cache_with_key, get_realm_seat_count_cache_key
from zerver.lib.exceptions import JsonableError
from zerver.lib.logging_util import log_to_file
from zerver.lib.send_email import (
    FromAddress,
    send_email,
    send_email_to_billing_admins_and_realm_owners,
)
from zerver.lib.timestamp import datetime_to_timestamp, timestamp_to_datetime
from zerver.lib.url_encoding import append_url_query_string
from zerver.lib.utils import assert_is_not_none
from zerver.models import Realm, RealmAuditLog, Stream, UserProfile
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.realms import get_org_type_display_name, get_realm
from zerver.models.streams import get_stream
from zerver.models.users import get_system_bot
from zilencer.lib.remote_counts import MissingDataError
from zilencer.models import (
    RemoteRealm,
    RemoteRealmAuditLog,
    RemoteRealmBillingUser,
    RemoteServerBillingUser,
    RemoteZulipServer,
    RemoteZulipServerAuditLog,
    get_remote_realm_guest_and_non_guest_count,
    get_remote_server_guest_and_non_guest_count,
    has_stale_audit_log,
)
from zproject.config import get_secret
