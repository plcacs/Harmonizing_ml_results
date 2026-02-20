from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import Any, Literal

from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import F
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language

from confirmation import settings as confirmation_settings
from zerver.actions.message_send import (
    internal_send_group_direct_message,
    internal_send_private_message,
    internal_send_stream_message,
)
from zerver.actions.streams import bulk_add_subscriptions, send_peer_subscriber_events
from zerver.actions.user_groups import (
    bulk_add_members_to_user_groups,
    do_send_user_group_members_update_event,
)
from zerver.actions.users import (
    change_user_is_active,
    get_service_dicts_for_bot,
    send_update_events_for_anonymous_group_settings,
)
from zerver.lib.avatar import avatar_url
from zerver.lib.create_user import create_user
from zerver.lib.default_streams import get_slim_realm_default_streams
from zerver.lib.email_notifications import enqueue_welcome_emails, send_account_registered_email
from zerver.lib.exceptions import JsonableError
from zerver.lib.invites import notify_invites_changed
from zerver.lib.mention import silent_mention_syntax_for_user
from zerver.lib.remote_server import maybe_enqueue_audit_log_upload
from zerver.lib.send_email import clear_scheduled_invitation_emails
from zerver.lib.streams import can_access_stream_history
from zerver.lib.subscription_info import bulk_get_subscriber_peer_info
from zerver.lib.user_counts import realm_user_count, realm_user_count_by_role
from zerver.lib.user_groups import get_system_user_group_for_user
from zerver.lib.users import (
    can_access_delivery_email,
    format_user_row,
    get_data_for_inaccessible_user,
    get_user_ids_who_can_access_user,
    user_access_restricted_in_realm,
    user_profile_to_user_row,
)
from zerver.models import (
    DefaultStreamGroup,
    Message,
    NamedUserGroup,
    OnboardingStep,
    OnboardingUserMessage,
    PreregistrationRealm,
    PreregistrationUser,
    Realm,
    RealmAuditLog,
    Recipient,
    Stream,
    Subscription,
    UserGroupMembership,
    UserMessage,
    UserProfile,
)
from zerver.models.groups import SystemGroups
from zerver.models.realm_audit_logs import AuditLogEventType
from zerver.models.users import active_user_ids, bot_owner_user_ids, get_system_bot
from zerver.tornado.django_api import send_event_on_commit

MAX_NUM_RECENT_MESSAGES = 1000
MAX_NUM_RECENT_UNREAD_MESSAGES = 20


def send_message_to_signup_notification_stream(
    sender: UserProfile, realm: Realm, message: str
) -> None:
    signup_announcements_stream = realm.signup_announcements_stream
    if signup_announcements_stream is None:
        return

    with override_language(realm.default_language):
        topic_name = _("signups")

    internal_send_stream_message(sender, signup_announcements_stream, topic_name, message)


def send_group_direct_message_to_admins(sender: UserProfile, realm: Realm, content: str) -> None:
    administrators = list(realm.get_human_admin_users())
    internal_send_group_direct_message(
        realm,
        sender,
        content,
        recipient_users=administrators,
    )


def notify_new_user(user_profile: UserProfile) -> None:
    user_count = realm_user_count(user_profile.realm)
    sender_email = settings.NOTIFICATION_BOT
    sender = get_system_bot(sender_email, user_profile.realm_id)

    is_first_user = user_count == 1
    if not is_first_user:
        with override_language(user_profile.realm.default_language):
            message = _("{user} joined this organization.").format(
                user=silent_mention_syntax_for_user(user_profile), user_count=user_count
            )
            send_message_to_signup_notification_stream(sender, user_profile.realm, message)

        if settings.BILLING_ENABLED:
            from corporate.lib.registration import generate_licenses_low_warning_message_if_required

            licenses_low_warning_message = generate_licenses_low_warning_message_if_required(
                user_profile.realm
            )
            if licenses_low_warning_message is not None:
                message += "\n"
                message += licenses_low_warning_message
                send_group_direct_message_to_admins(sender, user_profile.realm, message)


def set_up_streams_and_groups_for_new_human_user(
    *,
    user_profile: UserProfile,
    prereg_user: PreregistrationUser | None = None,
    default_stream_groups: Sequence[DefaultStreamGroup] = [],
    add_initial_stream_subscriptions: bool = True,
    realm_creation: bool = False,
) -> None:
    realm = user_profile.realm

    if prereg_user is not None:
        streams: list[Stream] = list(prereg_user.streams.all())
        user_groups: list[NamedUserGroup] = list(prereg_user.groups.all())
        acting_user: UserProfile | None = prereg_user.referred_by

        # A PregistrationUser should not be used for another UserProfile
        assert prereg_user.created_user is None, "PregistrationUser should not be reused"
    else:
        streams = []
        user_groups = []
        acting_user = None

    if add_initial_stream_subscriptions:
        # If prereg_user.include_realm_default_subscriptions is true, we
        # add the default streams for the realm to the list of streams.
        # Note that we are fine with "slim" Stream objects for calling
        # bulk_add_subscriptions and add_new_user_history, which we verify
        # in StreamSetupTest tests that check query counts.
        if prereg_user is None or prereg_user.include_realm_default_subscriptions:
            default_streams = get_slim_realm_default_streams(realm.id)
            streams = list(set(streams) | default_streams)

        for default_stream_group in default_stream_groups:
            default_stream_group_streams = default_stream_group.streams.all()
            for stream in default_stream_group_streams:
                if stream not in streams:
                    streams.append(stream)
    else:
        streams = []

    bulk_add_subscriptions(
        realm,
        streams,
        [user_profile],
        from_user_creation=True,
        acting_user=acting_user,
    )

    bulk_add_members_to_user_groups(
        user_groups,
        [user_profile.id],
        acting_user=acting_user,
    )

    add_new_user_history(user_profile, streams, realm_creation=realm_creation)


def add_new_user_history(
    user_profile: UserProfile,
    streams: Iterable[Stream],
    *,
    realm_creation: bool = False,
) -> None:
    """
    Give the user some messages in their feed, so that they can learn
    how to use the home view in a realistic way.

    For realms having older onboarding messages, mark the very
    most recent messages as unread. Otherwise, ONLY mark the
    messages tracked in 'OnboardingUserMessage' as unread.
    """

    realm = user_profile.realm
    # Find recipient ids for the user's streams, limiting to just
    # those where we can access the streams' full history.
    #
    # TODO: This will do database queries in a loop if many private
    # streams are involved.
    recipient_ids = [
        stream.recipient_id for stream in streams if can_access_stream_history(user_profile, stream)
    ]

    # Start by finding recent messages matching those recipients.
    recent_message_ids = set(
        Message.objects.filter(
            # Uses index: zerver_message_realm_recipient_id
            realm_id=realm.id,
            recipient_id__in=recipient_ids,
        )
        .order_by("-id")
        .values_list("id", flat=True)[0:MAX_NUM_RECENT_MESSAGES]
    )

    tracked_onboarding_message_ids = set()
    message_id_to_onboarding_user_message = {}
    onboarding_user_messages_queryset = OnboardingUserMessage.objects.filter(realm_id=realm.id)
    for onboarding_user_message in onboarding_user_messages_queryset:
        tracked_onboarding_message_ids.add(onboarding_user_message.message_id)
        message_id_to_onboarding_user_message[onboarding_user_message.message_id] = (
            onboarding_user_message
        )
    tracked_onboarding_messages_exist = len(tracked_onboarding_message_ids) > 0

    message_history_ids = recent_message_ids.union(tracked_onboarding_message_ids)

    if len(message_history_ids) > 0:
        # Handle the race condition where a message arrives between
        # bulk_add_subscriptions above and the recent message query just above
        already_used_ids = set(
            UserMessage.objects.filter(
                message_id__in=recent_message_ids, user_profile=user_profile
            ).values_list("message_id", flat=True)
        )

        # Exclude the already-used ids and sort them.
        backfill_message_ids = sorted(message_history_ids - already_used_ids)

        # Find which message ids we should mark as read.
        # (We don't want too many unread messages.)
        older_message_ids = set()
        if not tracked_onboarding_messages_exist:
            older_message_ids = set(backfill_message_ids[:-MAX_NUM_RECENT_UNREAD_MESSAGES])

        # Create UserMessage rows for the backfill.
        ums_to_create = []
        for message_id in backfill_message_ids:
            um = UserMessage(user_profile=user_profile, message_id=message_id)
            # Only onboarding messages are available for realm creator.
            # They are not marked as historical.
            if not realm_creation:
                um.flags = UserMessage.flags.historical
            if tracked_onboarding_messages_exist:
                if message_id not in tracked_onboarding_message_ids:
                    um.flags |= UserMessage.flags.read
                elif message_id_to_onboarding_user_message[message_id].flags.starred.is_set:
                    um.flags |= UserMessage.flags.starred
            elif message_id in older_message_ids:
                um.flags |= UserMessage.flags.read
            ums_to_create.append(um)

        UserMessage.objects.bulk_create(ums_to_create)


# Does the processing for a new user account:
# * Subscribes to default/invitation streams
# * Fills in some recent historical messages
# * Notifies other users in realm and Zulip about the signup
# * Deactivates PreregistrationUser objects
# * Mark 'visibility_policy_banner' as read
def process_new_human_user(
    user_profile: UserProfile,
    prereg_user: PreregistrationUser | None = None,
    default_stream_groups: Sequence[DefaultStreamGroup] = [],
    realm_creation: bool = False,
    add_initial_stream_subscriptions: bool = True,
) -> None:
    # subscribe to default/invitation streams and
    # fill in some recent historical messages
    set_up_streams_and_groups_for_new_human_user(
        user_profile=user_profile,
        prereg_user=prereg_user,
        default_stream_groups=default_stream_groups,
        add_initial_stream_subscriptions=add_initial_stream_subscriptions,
        realm_creation=realm_creation,
    )

    realm = user_profile.realm
    mit_beta_user = realm.is_zephyr_mirror_realm

    # mit_beta_users don't have a referred_by field
    if (
        not mit_beta_user
        and prereg_user is not None
        and prereg_user.referred_by is not None
        and prereg_user.referred_by.is_active
        and prereg_user.notify_referrer_on_join
    ):
        # This is a cross-realm direct message.
        with override_language(prereg_user.referred_by.default_language):
            internal_send_private_message(
                get_system_bot(settings.NOTIFICATION_BOT, prereg_user.referred_by.realm_id),
                prereg_user.referred_by,
                _("{user} accepted your invitation to join Zulip!").format(
                    user=silent_mention_syntax_for_user(user_profile)
                ),
            )

    # For the sake of tracking the history of UserProfiles,
    # we want to tie the newly created user to the PreregistrationUser
    # it was created from.
    if prereg_user is not None:
        prereg_user.status = confirmation_settings.STATUS_USED
        prereg_user.created_user = user_profile
        prereg_user.save(update_fields=["status", "created_user"])

    # Mark any other PreregistrationUsers in the realm that are STATUS_USED as
    # inactive so we can keep track of the PreregistrationUser we
    # actually used for analytics.
    if prereg_user is not None:
        PreregistrationUser.objects.filter(
            email__iexact=user_profile.delivery_email, realm=user_profile.realm
        ).exclude(id=prereg_user.id).update(status=confirmation_settings.STATUS_REVOKED)
    else:
        PreregistrationUser.objects.filter(
            email__iexact=user_profile.delivery_email, realm=user_profile.realm
        ).update(status=confirmation_settings.STATUS_REVOKED)

    if prereg_user is not None and prereg_user.referred_by is not None:
        notify_invites_changed(user_profile.realm, changed_invite_referrer=prereg_user.referred_by)

    notify_new_user(user_profile)
    # Clear any scheduled invitation emails to prevent them
    # from being sent after the user is created.
    clear_scheduled_invitation_emails(user_profile.delivery_email)
    if realm.send_welcome_emails:
        enqueue_welcome_emails(user_profile, realm_creation)

    # Schedule an initial email with the user's
    # new account details and log-in information.
    send_account_registered_email(user_profile, realm_creation)

    # We have an import loop here; it's intentional, because we want
    # to keep all the onboarding code in zerver/lib/onboarding.py.
    from zerver.lib.onboarding import send_initial_direct_message

    message_id = send_initial_direct_message(user_profile)
    UserMessage.objects.filter(user_profile=user_profile, message_id=message_id).update(
        flags=F("flags").bitor(UserMessage.flags.starred)
    )

    # The 'visibility_policy_banner' is only displayed to existing users.
    # Mark it as read for a new user.
    #
    # If the new user opted to import settings from an existing account, and
    # 'visibility_policy_banner' is already marked as read for the existing account,
    # 'copy_onboarding_steps' function already did the needed copying.
    # Simply ignore the IntegrityError in that case.
    #
    # The extremely brief nature of this subtransaction makes a savepoint safe.
    # See https://postgres.ai/blog/20210831-postgresql-subtransactions-considered-harmful
    # for context on risks around savepoints.
    with suppress(IntegrityError), transaction.atomic(savepoint=True):
        OnboardingStep.objects.create(user=user_profile, onboarding_step="visibility_policy_banner")


def notify_created_user(user_profile: UserProfile, notify_user_ids: list[int]) -> None:
    user_row = user_profile_to_user_row(user_profile)

    format_user_row_kwargs: dict[str, Any] = {
        "realm_id": user_profile.realm_id,
        "row": user_row,
        # Since we don't know what the client
        # supports at this point in the code, we
        # just assume client_gravatar and
        # user_avatar_url_field_optional = False :(
        "client_gravatar": False,
        "user_avatar_url_field_optional": False,
        # We assume there's no custom profile
        # field data for a new user; initial
        # values are expected to be added in a
        # later event.
        "custom_profile_field_data": {},
    }

    user_ids_without_access_to_created_user: list[int] = []
    users_with_access_to_created_users: list[UserProfile] = []

    if notify_user_ids:
        # This is currently used to send creation event when a guest
        # gains access to a user, so we depend on the caller to make
        # sure that only accessible users receive the user data.
        users_with_access_to_created_users = list(
            user_profile.realm.get_active_users().filter(id__in=notify_user_ids)
        )
    else:
        active_realm_users = list(user_profile.realm.get_active_users())

        # This call to user_access_restricted_in_realm results in
        # one extra query in the user creation codepath to check
        # "realm.can_access_all_users_group.name" because we do
        # not prefetch realm and its related fields when fetching
        # PreregistrationUser object.
        if user_access_restricted_in_realm(user_profile):
            for user in active_realm_users:
                if user.is_guest:
                    # This logic assumes that can_access_all_users_group
                    # setting can only be set to EVERYONE or MEMBERS.
                    user_ids_without_access_to_created_user.append(user.id)
                else:
                    users_with_access_to_created_users.append(user)
        else:
            users_with_access_to_created_users = active_realm_users

    user_ids_with_real_email_access = []
    user_ids_without_real_email_access = []

    person_for_real_email_access_users = None
    person_for_without_real_email_access_users = None
    for recipient_user in users_with_access_to_created_users:
        if can_access_delivery_email(
            recipient_user, user_profile.id, user_row["email_address_visibility"]
        ):
            user_ids_with_real_email_access.append(recipient_user.id)
            if person_for_real_email_access_users is None:
                # This caller assumes that "format_user_row" only depends on
                # specific value of "acting_user" among users in a realm in
                # email_address_visibility.
                person_for_real_email_access_users = format_user_row(
                    **format_user_row_kwargs,
                    acting_user=recipient_user,
                )
        else:
            user_ids_without_real_email_access.append(recipient_user.id)
            if person