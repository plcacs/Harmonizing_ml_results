import logging
from collections.abc import Collection
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from django.db.models import Q, QuerySet, Sum
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from zxcvbn import zxcvbn
from analytics.lib.counts import COUNT_STATS, do_increment_logging_stat
from analytics.models import RealmCount
from confirmation import settings as confirmation_settings
from confirmation.models import (
    Confirmation,
    confirmation_url_for,
    create_confirmation_link,
    create_confirmation_object,
)
from zerver.context_processors import common_context
from zerver.lib.email_validation import (
    get_existing_user_errors,
    get_realm_email_validator,
    validate_email_is_valid,
)
from zerver.lib.exceptions import InvitationError
from zerver.lib.invites import notify_invites_changed
from zerver.lib.queue import queue_event_on_commit
from zerver.lib.send_email import (
    FromAddress,
    clear_scheduled_invitation_emails,
    send_future_email,
)
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.utils import assert_is_not_none
from zerver.models import (
    Message,
    MultiuseInvite,
    NamedUserGroup,
    PreregistrationUser,
    Realm,
    Stream,
    UserProfile,
)
from zerver.models.prereg_users import filter_to_valid_prereg_users


def estimate_recent_invites(realms: Collection[Realm], *, days: int) -> int:
    """An upper bound on the number of invites sent in the last `days` days"""
    recent_invites_data: Dict[str, Optional[int]] = RealmCount.objects.filter(
        realm__in=realms,
        property="invites_sent::day",
        subgroup=None,
        end_time__gte=timezone_now() - timedelta(days=days),
    ).aggregate(Sum("value"))
    recent_invites = recent_invites_data.get("value__sum")
    if recent_invites is None:
        return 0
    return recent_invites


def too_many_recent_realm_invites(realm: Realm, num_invitees: int) -> bool:
    recent_invites: int = estimate_recent_invites([realm], days=1)
    if num_invitees + recent_invites > realm.max_invites:
        return True
    if realm.plan_type != Realm.PLAN_TYPE_LIMITED:
        return False
    if realm.max_invites != settings.INVITES_DEFAULT_REALM_DAILY_MAX:
        return False
    warning_flags: List[str] = []
    if zxcvbn(realm.string_id)["score"] == 4:
        warning_flags.append("random-realm-name")
    if not realm.description:
        warning_flags.append("no-realm-description")
    if realm.icon_source == Realm.ICON_FROM_GRAVATAR:
        warning_flags.append("no-realm-icon")
    if realm.date_created >= timezone_now() - timedelta(hours=1):
        warning_flags.append("realm-created-in-last-hour")
    current_user_count: int = UserProfile.objects.filter(
        realm=realm, is_bot=False, is_active=True
    ).count()
    if current_user_count == 1:
        warning_flags.append("only-one-user")
    estimated_sent_data: Dict[str, Optional[int]] = RealmCount.objects.filter(
        realm=realm, property="messages_sent:message_type:day"
    ).aggregate(messages=Sum("value"))
    estimated_sent: Optional[int] = estimated_sent_data.get("messages")
    if not estimated_sent and not Message.objects.filter(
        realm=realm, sender__is_bot=False
    ).exists():
        warning_flags.append("no-messages-sent")
    if len(warning_flags) == 6:
        permitted_ratio: float = 2.0
    elif len(warning_flags) >= 3:
        permitted_ratio = 3.0
    else:
        permitted_ratio = 5.0
    ratio: float = (num_invitees + recent_invites) / current_user_count
    logging.log(
        logging.WARNING if ratio > permitted_ratio else logging.INFO,
        "%s (!: %s) inviting %d more, have %d recent, but only %d current users.  Ratio %.1f, %d allowed",
        realm.string_id,
        ",".join(warning_flags),
        num_invitees,
        recent_invites,
        current_user_count,
        ratio,
        permitted_ratio,
    )
    return ratio > permitted_ratio


def check_invite_limit(realm: Realm, num_invitees: int) -> None:
    """Discourage using invitation emails as a vector for carrying spam."""
    msg = _(
        "To protect users, Zulip limits the number of invitations you can send in one day. Because you have reached the limit, no invitations were sent."
    )
    if not settings.OPEN_REALM_CREATION:
        return
    if too_many_recent_realm_invites(realm, num_invitees):
        raise InvitationError(
            msg,
            [],
            sent_invitations=False,
            daily_limit_reached=True,
        )
    default_max: int = settings.INVITES_DEFAULT_REALM_DAILY_MAX
    newrealm_age: timedelta = timedelta(days=settings.INVITES_NEW_REALM_DAYS)
    if realm.date_created <= timezone_now() - newrealm_age:
        return
    if realm.max_invites > default_max:
        return
    new_realms: QuerySet[Realm] = Realm.objects.filter(
        date_created__gte=timezone_now() - newrealm_age,
        _max_invites__lte=default_max,
    ).all()
    for days, count in settings.INVITES_NEW_REALM_LIMIT_DAYS:
        recent_invites: int = estimate_recent_invites(new_realms, days=days)
        if num_invitees + recent_invites > count:
            raise InvitationError(
                msg,
                [],
                sent_invitations=False,
                daily_limit_reached=True,
            )


@transaction.atomic(durable=True)
def do_invite_users(
    user_profile: UserProfile,
    invitee_emails: Collection[str],
    streams: Collection[Stream],
    notify_referrer_on_join: bool = True,
    user_groups: Collection[NamedUserGroup] = [],
    *,
    invite_expires_in_minutes: int,
    include_realm_default_subscriptions: bool,
    invite_as: str = PreregistrationUser.INVITE_AS["MEMBER"],
) -> List[Tuple[str, str, bool]]:
    num_invites: int = len(invitee_emails)
    realm: Realm = Realm.objects.select_for_update().get(id=user_profile.realm_id)
    check_invite_limit(realm, num_invites)
    if settings.BILLING_ENABLED:
        from corporate.lib.registration import (
            check_spare_licenses_available_for_inviting_new_users,
        )
        if invite_as == PreregistrationUser.INVITE_AS["GUEST_USER"]:
            check_spare_licenses_available_for_inviting_new_users(
                realm, extra_guests_count=num_invites
            )
        else:
            check_spare_licenses_available_for_inviting_new_users(
                realm, extra_non_guests_count=num_invites
            )
    if not realm.invite_required:
        min_age: timedelta = timedelta(days=settings.INVITES_MIN_USER_AGE_DAYS)
        if (
            user_profile.date_joined > timezone_now() - min_age
            and not user_profile.is_realm_admin
        ):
            raise InvitationError(
                _(
                    "Your account is too new to send invites for this organization. Ask an organization admin, or a more experienced user."
                ),
                [],
                sent_invitations=False,
            )
    good_emails: Set[str] = set()
    errors: List[Tuple[str, str, bool]] = []
    validate_email_allowed_in_realm = get_realm_email_validator(realm)
    for email in invitee_emails:
        if email == "":
            continue
        email_error = validate_email_is_valid(email, validate_email_allowed_in_realm)
        if email_error:
            errors.append((email, email_error, False))
        else:
            good_emails.add(email)
    "\n    good_emails are emails that look ok so far,\n    but we still need to make sure they're not\n    gonna conflict with existing users\n    "
    error_dict: Dict[str, Tuple[str, bool]] = get_existing_user_errors(
        realm, good_emails
    )
    skipped: List[Tuple[str, str, bool]] = []
    for email in error_dict:
        msg, deactivated = error_dict[email]
        skipped.append((email, msg, deactivated))
        good_emails.remove(email)
    validated_emails: List[str] = list(good_emails)
    if errors:
        raise InvitationError(
            _("Some emails did not validate, so we didn't send any invitations."),
            errors + skipped,
            sent_invitations=False,
            daily_limit_reached=False,
        )
    if skipped and len(skipped) == len(invitee_emails):
        raise InvitationError(
            _("We weren't able to invite anyone."),
            skipped,
            sent_invitations=False,
            daily_limit_reached=False,
        )
    for email in validated_emails:
        prereg_user: PreregistrationUser = PreregistrationUser(
            email=email,
            referred_by=user_profile,
            invited_as=invite_as,
            realm=realm,
            include_realm_default_subscriptions=include_realm_default_subscriptions,
            notify_referrer_on_join=notify_referrer_on_join,
        )
        prereg_user.save()
        stream_ids: List[int] = [stream.id for stream in streams]
        prereg_user.streams.set(stream_ids)
        group_ids: List[int] = [user_group.id for user_group in user_groups]
        prereg_user.groups.set(group_ids)
        confirmation: Confirmation = create_confirmation_object(
            prereg_user,
            Confirmation.INVITATION,
            validity_in_minutes=invite_expires_in_minutes,
        )
        do_send_user_invite_email(
            prereg_user,
            confirmation=confirmation,
            invite_expires_in_minutes=invite_expires_in_minutes,
        )
    notify_invites_changed(realm, changed_invite_referrer=user_profile)
    return skipped


def get_invitation_expiry_date(confirmation_obj: Confirmation) -> Optional[int]:
    expiry_date: Optional[datetime] = confirmation_obj.expiry_date
    if expiry_date is None:
        return expiry_date
    return datetime_to_timestamp(expiry_date)


def do_get_invites_controlled_by_user(
    user_profile: UserProfile,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts representing invitations that can be controlled by user_profile.
    This isn't necessarily the same as all the invitations generated by the user, as administrators
    can control also invitations that they did not themselves create.
    """
    if user_profile.is_realm_admin:
        prereg_users: QuerySet[PreregistrationUser] = filter_to_valid_prereg_users(
            PreregistrationUser.objects.filter(
                realm=user_profile.realm, referred_by__isnull=False
            )
        )
    else:
        prereg_users = filter_to_valid_prereg_users(
            PreregistrationUser.objects.filter(referred_by=user_profile)
        )
    invites: List[Dict[str, Any]] = []
    for invitee in prereg_users:
        assert invitee.referred_by is not None
        invites.append(
            {
                "email": invitee.email,
                "invited_by_user_id": invitee.referred_by.id,
                "invited": datetime_to_timestamp(invitee.invited_at),
                "expiry_date": get_invitation_expiry_date(invitee.confirmation.get()),
                "id": invitee.id,
                "invited_as": invitee.invited_as,
                "is_multiuse": False,
                "notify_referrer_on_join": invitee.notify_referrer_on_join,
            }
        )
    if user_profile.is_realm_admin:
        multiuse_confirmation_objs: QuerySet[Confirmation] = Confirmation.objects.filter(
            realm=user_profile.realm,
            type=Confirmation.MULTIUSE_INVITE,
        ).filter(
            Q(expiry_date__gte=timezone_now()) | Q(expiry_date=None)
        )
    else:
        multiuse_invite_ids: List[int] = list(
            MultiuseInvite.objects.filter(referred_by=user_profile).values_list(
                "id", flat=True
            )
        )
        multiuse_confirmation_objs = Confirmation.objects.filter(
            type=Confirmation.MULTIUSE_INVITE,
            object_id__in=multiuse_invite_ids,
        ).filter(
            Q(expiry_date__gte=timezone_now()) | Q(expiry_date=None)
        )
    for confirmation_obj in multiuse_confirmation_objs:
        invite: Union[MultiuseInvite, PreregistrationUser] = confirmation_obj.content_object
        assert invite is not None
        assert invite.status != confirmation_settings.STATUS_REVOKED
        invites.append(
            {
                "invited_by_user_id": invite.referred_by.id,
                "invited": datetime_to_timestamp(confirmation_obj.date_sent),
                "expiry_date": get_invitation_expiry_date(confirmation_obj),
                "id": invite.id,
                "link_url": confirmation_url_for(confirmation_obj),
                "invited_as": invite.invited_as,
                "is_multiuse": True,
            }
        )
    return invites


@transaction.atomic(durable=True)
def do_create_multiuse_invite_link(
    referred_by: UserProfile,
    invited_as: str,
    invite_expires_in_minutes: int,
    include_realm_default_subscriptions: bool,
    streams: List[Stream] = [],
    user_groups: List[NamedUserGroup] = [],
) -> str:
    realm: Realm = referred_by.realm
    invite: MultiuseInvite = MultiuseInvite.objects.create(
        realm=realm,
        referred_by=referred_by,
        include_realm_default_subscriptions=include_realm_default_subscriptions,
    )
    if streams:
        invite.streams.set(streams)
    if user_groups:
        invite.groups.set(user_groups)
    invite.invited_as = invited_as
    invite.save()
    notify_invites_changed(referred_by.realm, changed_invite_referrer=referred_by)
    return create_confirmation_link(
        invite, Confirmation.MULTIUSE_INVITE, validity_in_minutes=invite_expires_in_minutes
    )


@transaction.atomic(durable=True)
def do_revoke_user_invite(prereg_user: PreregistrationUser) -> None:
    email: str = prereg_user.email
    realm: Realm = assert_is_not_none(prereg_user.realm)
    content_type: ContentType = ContentType.objects.get_for_model(PreregistrationUser)
    Confirmation.objects.filter(
        content_type=content_type, object_id=prereg_user.id
    ).delete()
    prereg_user.delete()
    clear_scheduled_invitation_emails(email)
    notify_invites_changed(realm, changed_invite_referrer=prereg_user.referred_by)


@transaction.atomic(durable=True)
def do_revoke_multi_use_invite(multiuse_invite: MultiuseInvite) -> None:
    realm: Realm = multiuse_invite.referred_by.realm
    content_type: ContentType = ContentType.objects.get_for_model(MultiuseInvite)
    Confirmation.objects.filter(
        content_type=content_type, object_id=multiuse_invite.id
    ).delete()
    multiuse_invite.status = confirmation_settings.STATUS_REVOKED
    multiuse_invite.save(update_fields=["status"])
    notify_invites_changed(realm, changed_invite_referrer=multiuse_invite.referred_by)


@transaction.atomic(savepoint=False)
def do_send_user_invite_email(
    prereg_user: PreregistrationUser,
    *,
    confirmation: Optional[Confirmation] = None,
    event_time: Optional[datetime] = None,
    invite_expires_in_minutes: Optional[float] = None,
) -> None:
    realm_id: int = assert_is_not_none(prereg_user.realm_id)
    realm: Realm = Realm.objects.select_for_update().get(id=realm_id)
    check_invite_limit(realm, 1)
    referrer: UserProfile = assert_is_not_none(prereg_user.referred_by)
    if event_time is None:
        event_time = prereg_user.invited_at
    do_increment_logging_stat(
        realm, COUNT_STATS["invites_sent::day"], None, event_time
    )
    if confirmation is None:
        confirmation = prereg_user.confirmation.get()
    event: Dict[str, Any] = {
        "template_prefix": "zerver/emails/invitation",
        "to_emails": [prereg_user.email],
        "from_address": FromAddress.tokenized_no_reply_address(),
        "language": realm.default_language,
        "context": {
            "referrer_full_name": referrer.full_name,
            "referrer_email": referrer.delivery_email,
            "activate_url": confirmation_url_for(confirmation),
            "referrer_realm_name": realm.name,
            "corporate_enabled": settings.CORPORATE_ENABLED,
        },
        "realm_id": realm.id,
    }
    queue_event_on_commit("email_senders", event)
    clear_scheduled_invitation_emails(prereg_user.email)
    if invite_expires_in_minutes is None and confirmation.expiry_date is not None:
        invite_expires_in_minutes = (confirmation.expiry_date - event_time).total_seconds() / 60
    if invite_expires_in_minutes is None or invite_expires_in_minutes < 4 * 24 * 60:
        return
    context: Dict[str, Any] = common_context(referrer)
    context.update(
        activate_url=confirmation_url_for(confirmation),
        referrer_name=referrer.full_name,
        referrer_email=referrer.delivery_email,
        referrer_realm_name=realm.name,
    )
    send_future_email(
        "zerver/emails/invitation_reminder",
        realm,
        to_emails=[prereg_user.email],
        from_address=FromAddress.tokenized_no_reply_placeholder,
        language=realm.default_language,
        context=context,
        delay=timedelta(minutes=invite_expires_in_minutes - 2 * 24 * 60),
    )
