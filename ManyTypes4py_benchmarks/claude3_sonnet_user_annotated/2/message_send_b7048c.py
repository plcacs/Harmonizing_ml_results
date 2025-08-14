import logging
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from email.headerregistry import Address
from typing import Any, TypedDict

import orjson
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models import F, Q, QuerySet
from django.utils.html import escape
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language

from zerver.actions.uploads import do_claim_attachments
from zerver.actions.user_topics import (
    bulk_do_set_user_topic_visibility_policy,
    do_set_user_topic_visibility_policy,
)
from zerver.lib.addressee import Addressee
from zerver.lib.alert_words import get_alert_word_automaton
from zerver.lib.cache import cache_with_key, user_profile_delivery_email_cache_key
from zerver.lib.create_user import create_user
from zerver.lib.exceptions import (
    DirectMessageInitiationError,
    DirectMessagePermissionError,
    JsonableError,
    MarkdownRenderingError,
    StreamDoesNotExistError,
    StreamWildcardMentionNotAllowedError,
    StreamWithIDDoesNotExistError,
    TopicWildcardMentionNotAllowedError,
    ZephyrMessageAlreadySentError,
)
from zerver.lib.markdown import MessageRenderingResult, render_message_markdown
from zerver.lib.markdown import version as markdown_version
from zerver.lib.mention import MentionBackend, MentionData
from zerver.lib.message import (
    SendMessageRequest,
    check_user_group_mention_allowed,
    normalize_body,
    set_visibility_policy_possible,
    stream_wildcard_mention_allowed,
    topic_wildcard_mention_allowed,
    truncate_topic,
    visibility_policy_for_send_message,
)
from zerver.lib.message_cache import MessageDict
from zerver.lib.muted_users import get_muting_users
from zerver.lib.notification_data import (
    UserMessageNotificationsData,
    get_user_group_mentions_data,
    user_allows_notifications_in_StreamTopic,
)
from zerver.lib.query_helpers import query_for_ids
from zerver.lib.queue import queue_event_on_commit
from zerver.lib.recipient_users import recipient_for_user_profiles
from zerver.lib.stream_subscription import (
    get_subscriptions_for_send_message,
    num_subscribers_for_stream_id,
)
from zerver.lib.stream_topic import StreamTopicTarget
from zerver.lib.streams import (
    access_stream_for_send_message,
    ensure_stream,
    notify_stream_is_recently_active_update,
    subscribed_to_stream,
)
from zerver.lib.string_validation import check_stream_name
from zerver.lib.thumbnail import get_user_upload_previews, rewrite_thumbnailed_images
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.lib.topic import participants_for_topic
from zerver.lib.url_preview.types import UrlEmbedData
from zerver.lib.user_groups import is_any_user_in_group, is_user_in_group
from zerver.lib.user_message import UserMessageLite, bulk_insert_ums
from zerver.lib.users import (
    check_can_access_user,
    get_inaccessible_user_ids,
    get_subscribers_of_target_user_subscriptions,
    get_user_ids_who_can_access_user,
    get_users_involved_in_dms_with_target_users,
    user_access_restricted_in_realm,
)
from zerver.lib.validator import check_widget_content
from zerver.lib.widget import do_widget_post_save_actions
from zerver.models import (
    Client,
    Message,
    Realm,
    Recipient,
    Stream,
    UserMessage,
    UserPresence,
    UserProfile,
    UserTopic,
)
from zerver.models.clients import get_client
from zerver.models.groups import SystemGroups
from zerver.models.recipients import get_direct_message_group_user_ids
from zerver.models.scheduled_jobs import NotificationTriggers
from zerver.models.streams import (
    get_stream_by_id_for_sending_message,
    get_stream_by_name_for_sending_message,
)
from zerver.models.users import get_system_bot, get_user_by_delivery_email, is_cross_realm_bot_email
from zerver.tornado.django_api import send_event_on_commit
