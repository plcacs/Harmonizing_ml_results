#!/usr/bin/env python3
from __future__ import annotations
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Sequence as Seq, Set, Tuple, TypeVar
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.core.exceptions import ValidationError
from django.db import connection
from django.utils.translation import gettext as _
from pydantic import BaseModel, model_validator
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection, Row
from sqlalchemy.sql import ClauseElement, ColumnElement, Select, and_, column, false, func, join, literal, literal_column, not_, or_, select, table, union_all
from sqlalchemy.sql.selectable import SelectBase
from sqlalchemy.types import ARRAY, Boolean, Integer, Text
from typing_extensions import override
from zerver.lib.addressee import get_user_profiles, get_user_profiles_by_ids
from zerver.lib.exceptions import ErrorCode, JsonableError, MissingAuthenticationError
from zerver.lib.message import access_message, access_web_public_message, get_first_visible_message_id
from zerver.lib.narrow_predicate import channel_operators, channels_operators
from zerver.lib.recipient_users import recipient_for_user_profiles
from zerver.lib.sqlalchemy_utils import get_sqlalchemy_connection
from zerver.lib.streams import can_access_stream_history_by_id, can_access_stream_history_by_name, get_public_streams_queryset, get_stream_by_narrow_operand_access_unchecked, get_web_public_streams_queryset
from zerver.lib.topic import maybe_rename_general_chat_to_empty_topic
from zerver.lib.topic_sqlalchemy import get_followed_topic_condition_sa, get_resolved_topic_condition_sa, topic_column_sa, topic_match_sa
from zerver.lib.types import Validator
from zerver.lib.user_topics import exclude_stream_and_topic_mutes
from zerver.lib.validator import check_bool, check_required_string, check_string, check_string_or_int, check_string_or_int_list
from zerver.models import DirectMessageGroup, Message, Realm, Recipient, Stream, Subscription, UserMessage, UserProfile
from zerver.models.recipients import get_direct_message_group_user_ids
from zerver.models.streams import get_active_streams
from zerver.models.users import get_user_by_id_in_realm_including_cross_realm, get_user_including_cross_realm

class NarrowParameter(BaseModel):
    negated: bool = False
    operator: str  # will be set during validation
    operand: Any   # can be str, int, or list

    @model_validator(mode='before')
    @classmethod
    def convert_term(cls, elem: Any) -> dict:
        if isinstance(elem, list):
            if len(elem) != 2 or any((not isinstance(x, str) for x in elem)):
                raise ValueError('element is not a string pair')
            return dict(operator=elem[0], operand=elem[1])
        elif isinstance(elem, dict):
            if 'operand' not in elem or elem['operand'] is None:
                raise ValueError('operand is missing')
            if 'operator' not in elem or elem['operator'] is None:
                raise ValueError('operator is missing')
            return elem
        else:
            raise ValueError('dict or list required')

    @model_validator(mode='after')
    def validate_terms(self) -> NarrowParameter:
        operators_supporting_id = [*channel_operators, 'id', 'sender', 'group-pm-with', 'dm-including', 'with']
        operators_supporting_ids = ['pm-with', 'dm']
        operators_non_empty_operand = {'search'}
        operator = self.operator
        if operator in operators_supporting_id:
            operand_validator = check_string_or_int
        elif operator in operators_supporting_ids:
            operand_validator = check_string_or_int_list
        elif operator in operators_non_empty_operand:
            operand_validator = check_required_string
        else:
            operand_validator = check_string
        try:
            self.operand = operand_validator('operand', self.operand)
            self.operator = check_string('operator', self.operator)
            if self.negated is not None:
                self.negated = check_bool('negated', self.negated)
        except ValidationError as error:
            raise JsonableError(error.message)
        return self

def is_spectator_compatible(narrow: Iterable[NarrowParameter]) -> bool:
    supported_operators = [*channel_operators, *channels_operators, 'topic', 'sender', 'has', 'search', 'near', 'id', 'with']
    for element in narrow:
        operator = element.operator
        if operator not in supported_operators:
            return False
    return True

def is_web_public_narrow(narrow: Optional[Seq[NarrowParameter]]) -> bool:
    if narrow is None:
        return False
    return any((term.operator in channels_operators and term.operand == 'web-public' and (term.negated is False) for term in narrow))

LARGER_THAN_MAX_MESSAGE_ID: int = 10000000000000000

class BadNarrowOperatorError(JsonableError):
    code = ErrorCode.BAD_NARROW
    data_fields = ['desc']

    def __init__(self, desc: str) -> None:
        self.desc = desc

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Invalid narrow operator: {desc}')

class InvalidOperatorCombinationError(JsonableError):
    code = ErrorCode.BAD_NARROW
    data_fields = ['desc']

    def __init__(self, desc: str) -> None:
        self.desc = desc

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Invalid narrow operator combination: {desc}')

ConditionTransform = Callable[[ClauseElement], ClauseElement]
TS_START: str = '<ts-match>'
TS_STOP: str = '</ts-match>'

def ts_locs_array(config: Any, text: Any, tsquery: Any) -> ClauseElement:
    options: str = f'HighlightAll = TRUE, StartSel = {TS_START}, StopSel = {TS_STOP}'
    delimited = func.ts_headline(config, text, tsquery, options, type_=Text)
    part = func.unnest(func.string_to_array(delimited, TS_START, type_=ARRAY(Text)), type_=Text).column_valued()
    part_len = func.length(part, type_=Integer) - len(TS_STOP)
    match_pos = func.sum(part_len, type_=Integer).over(rows=(None, -1)) + len(TS_STOP)
    match_len = func.strpos(part, TS_STOP, type_=Integer) - 1
    return func.array(select(postgresql.array([match_pos, match_len])).offset(1).scalar_subquery(), type_=ARRAY(Integer))

class NarrowBuilder:
    """
    Build up a SQLAlchemy query to find messages matching a narrow.
    """
    def __init__(self, user_profile: Optional[UserProfile], msg_id_column: ColumnElement, realm: Realm, is_web_public_query: bool = False) -> None:
        self.user_profile: Optional[UserProfile] = user_profile
        self.msg_id_column: ColumnElement = msg_id_column
        self.realm: Realm = realm
        self.is_web_public_query: bool = is_web_public_query
        self.by_method_map: dict[str, Callable[[ClauseElement, Any, Callable[[ClauseElement], ClauseElement]], ClauseElement]] = {
            'has': self.by_has,
            'in': self.by_in,
            'is': self.by_is,
            'channel': self.by_channel,
            'stream': self.by_channel,
            'channels': self.by_channels,
            'streams': self.by_channels,
            'topic': self.by_topic,
            'sender': self.by_sender,
            'near': self.by_near,
            'id': self.by_id,
            'search': self.by_search,
            'dm': self.by_dm,
            'pm-with': self.by_dm,
            'dm-including': self.by_dm_including,
            'group-pm-with': self.by_group_pm_with,
            'pm_with': self.by_dm,
            'group_pm_with': self.by_group_pm_with,
        }
        self.is_channel_narrow: bool = False
        self.is_dm_narrow: bool = False

    def check_not_both_channel_and_dm_narrow(self, maybe_negate: Callable[[ClauseElement], ClauseElement],
                                               is_dm_narrow: bool = False, is_channel_narrow: bool = False) -> None:
        if maybe_negate is not_:
            return
        if is_dm_narrow:
            self.is_dm_narrow = True
        if is_channel_narrow:
            self.is_channel_narrow = True
        if self.is_channel_narrow and self.is_dm_narrow:
            raise BadNarrowOperatorError('No message can be both a channel message and direct message')

    def add_term(self, query: ClauseElement, term: NarrowParameter) -> ClauseElement:
        operator: str = term.operator
        operand: Any = term.operand
        negated: bool = term.negated
        if operator in self.by_method_map:
            method = self.by_method_map[operator]
        else:
            raise BadNarrowOperatorError('unknown operator ' + operator)
        if negated:
            maybe_negate: Callable[[ClauseElement], ClauseElement] = not_
        else:
            maybe_negate = lambda cond: cond
        return method(query, operand, maybe_negate)

    def by_has(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        if operand not in ['attachment', 'image', 'link', 'reaction']:
            raise BadNarrowOperatorError("unknown 'has' operand " + operand)
        if operand == 'reaction':
            if self.msg_id_column.name == 'message_id':
                check_col: ColumnElement = literal_column('zerver_usermessage.message_id', Integer)
            else:
                check_col = literal_column('zerver_message.id', Integer)
            exists_cond = select(1).select_from(table('zerver_reaction')).where(check_col == literal_column('zerver_reaction.message_id', Integer)).exists()
            return query.where(maybe_negate(exists_cond))
        col_name: str = 'has_' + operand
        cond = column(col_name, Boolean)
        return query.where(maybe_negate(cond))

    def by_in(self, query: ClauseElement, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        assert not self.is_web_public_query
        assert self.user_profile is not None
        if operand == 'home':
            conditions = exclude_muting_conditions(self.user_profile, [NarrowParameter(operator='in', operand='home')])
            if conditions:
                return query.where(maybe_negate(and_(*conditions)))
            return query
        elif operand == 'all':
            return query
        raise BadNarrowOperatorError("unknown 'in' operand " + operand)

    def by_is(self, query: ClauseElement, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        assert not self.is_web_public_query
        assert self.user_profile is not None
        if operand in ['dm', 'private']:
            if maybe_negate is not_:
                self.check_not_both_channel_and_dm_narrow(maybe_negate=lambda cond: cond, is_channel_narrow=True)
            else:
                self.check_not_both_channel_and_dm_narrow(maybe_negate, is_dm_narrow=True)
            cond = column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'starred':
            cond = column('flags', Integer).op('&')(UserMessage.flags.starred.mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'unread':
            cond = column('flags', Integer).op('&')(UserMessage.flags.read.mask) == 0
            return query.where(maybe_negate(cond))
        elif operand == 'mentioned':
            mention_flags_mask = UserMessage.flags.mentioned.mask | UserMessage.flags.stream_wildcard_mentioned.mask | UserMessage.flags.topic_wildcard_mentioned.mask | UserMessage.flags.group_mentioned.mask
            cond = column('flags', Integer).op('&')(mention_flags_mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'alerted':
            cond = column('flags', Integer).op('&')(UserMessage.flags.has_alert_word.mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'resolved':
            cond = get_resolved_topic_condition_sa()
            return query.where(maybe_negate(cond))
        elif operand == 'followed':
            cond = get_followed_topic_condition_sa(self.user_profile.id)
            return query.where(maybe_negate(cond))
        raise BadNarrowOperatorError("unknown 'is' operand " + operand)

    _alphanum = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    def _pg_re_escape(self, pattern: str) -> str:
        """
        Escape user input to place in a regex

        Python's re.escape escapes Unicode characters in a way which PostgreSQL
        fails on, 'λ' to '\\λ'. This function will correctly escape
        them for PostgreSQL, 'λ' to '\\u03bb'.
        """
        s = list(pattern)
        for i, c in enumerate(s):
            if c not in self._alphanum:
                if ord(c) >= 128:
                    s[i] = f'\\u{ord(c):0>4x}'
                else:
                    s[i] = '\\' + c
        return ''.join(s)

    def by_channel(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_channel_narrow=True)
        try:
            channel: Stream = get_stream_by_narrow_operand_access_unchecked(operand, self.realm)
            if self.is_web_public_query and (not channel.is_web_public):
                raise BadNarrowOperatorError('unknown web-public channel ' + str(operand))
        except Stream.DoesNotExist:
            raise BadNarrowOperatorError('unknown channel ' + str(operand))
        if self.realm.is_zephyr_mirror_realm:
            assert not channel.is_public()
            m = re.search('^(?:un)*(.+?)(?:\\.d)*$', channel.name, re.IGNORECASE)
            assert m is not None
            base_channel_name = m.group(1)
            matching_channels = get_active_streams(self.realm).filter(name__iregex=f'^(un)*{self._pg_re_escape(base_channel_name)}(\\.d)*$')
            recipient_ids: List[int] = [matching_channel.recipient_id for matching_channel in matching_channels]
            cond = column('recipient_id', Integer).in_(recipient_ids)
            return query.where(maybe_negate(cond))
        recipient_id = channel.recipient_id
        assert recipient_id is not None
        cond = column('recipient_id', Integer) == recipient_id
        return query.where(maybe_negate(cond))

    def by_channels(self, query: ClauseElement, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_channel_narrow=True)
        if operand == 'public':
            recipient_queryset = get_public_streams_queryset(self.realm)
        elif operand == 'web-public':
            recipient_queryset = get_web_public_streams_queryset(self.realm)
        else:
            raise BadNarrowOperatorError('unknown channels operand ' + operand)
        recipient_ids = recipient_queryset.values_list('recipient_id', flat=True).order_by('id')
        cond = column('recipient_id', Integer).in_(recipient_ids)
        return query.where(maybe_negate(cond))

    def by_topic(self, query: ClauseElement, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_channel_narrow=True)
        if self.realm.is_zephyr_mirror_realm:
            m = re.search('^(.*?)(?:\\.d)*$', operand, re.IGNORECASE)
            assert m is not None
            base_topic = m.group(1)
            if base_topic in ('', 'personal', '(instance "")'):
                cond = or_(topic_match_sa(''), topic_match_sa('.d'), topic_match_sa('.d.d'), topic_match_sa('.d.d.d'), topic_match_sa('.d.d.d.d'),
                           topic_match_sa('personal'), topic_match_sa('personal.d'), topic_match_sa('personal.d.d'), topic_match_sa('personal.d.d.d'),
                           topic_match_sa('personal.d.d.d.d'), topic_match_sa('(instance "")'), topic_match_sa('(instance "").d'),
                           topic_match_sa('(instance "").d.d'), topic_match_sa('(instance "").d.d.d'), topic_match_sa('(instance "").d.d.d.d'))
            else:
                cond = or_(topic_match_sa(base_topic), topic_match_sa(base_topic + '.d'), topic_match_sa(base_topic + '.d.d'),
                           topic_match_sa(base_topic + '.d.d.d'), topic_match_sa(base_topic + '.d.d.d.d'))
            return query.where(maybe_negate(cond))
        cond = topic_match_sa(operand)
        return query.where(maybe_negate(cond))

    def by_sender(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        try:
            if isinstance(operand, str):
                sender: UserProfile = get_user_including_cross_realm(operand, self.realm)
            else:
                sender = get_user_by_id_in_realm_including_cross_realm(operand, self.realm)
        except UserProfile.DoesNotExist:
            raise BadNarrowOperatorError('unknown user ' + str(operand))
        cond = column('sender_id', Integer) == literal(sender.id)
        return query.where(maybe_negate(cond))

    def by_near(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        return query

    def by_id(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        if not str(operand).isdigit() or int(operand) > Message.MAX_POSSIBLE_MESSAGE_ID:
            raise BadNarrowOperatorError('Invalid message ID')
        cond = self.msg_id_column == literal(operand)
        return query.where(maybe_negate(cond))

    def by_dm(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        assert not self.is_web_public_query
        assert self.user_profile is not None
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_dm_narrow=True)
        try:
            if isinstance(operand, str):
                email_list = operand.split(',')
                user_profiles: List[UserProfile] = get_user_profiles(emails=email_list, realm=self.realm)
            else:
                # This is where we handle passing a list of user IDs for the narrow,
                # which is the preferred/cleaner API.
                user_profiles = get_user_profiles_by_ids(user_ids=operand, realm=self.realm)
            if user_profiles == []:
                return query.where(maybe_negate(false()))
            recipient = recipient_for_user_profiles(user_profiles=user_profiles, forwarded_mirror_message=False, forwarder_user_profile=None, sender=self.user_profile, allow_deactivated=True, create=False)
        except (JsonableError, ValidationError):
            raise BadNarrowOperatorError('unknown user in ' + str(operand))
        except DirectMessageGroup.DoesNotExist:
            return query.where(maybe_negate(false()))
        if recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
            cond = column('recipient_id', Integer) == recipient.id
            return query.where(maybe_negate(cond))
        other_participant: Optional[UserProfile] = None
        for user in user_profiles:
            if user.id != self.user_profile.id:
                other_participant = user
        if other_participant:
            self_recipient_id = self.user_profile.recipient_id
            cond = and_(
                column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0,
                column('realm_id', Integer) == self.realm.id,
                or_(
                    and_(
                        column('sender_id', Integer) == other_participant.id,
                        column('recipient_id', Integer) == self_recipient_id
                    ),
                    and_(
                        column('sender_id', Integer) == self.user_profile.id,
                        column('recipient_id', Integer) == recipient.id
                    )
                )
            )
            return query.where(maybe_negate(cond))
        cond = and_(
            column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0,
            column('realm_id', Integer) == self.realm.id,
            column('sender_id', Integer) == self.user_profile.id,
            column('recipient_id', Integer) == recipient.id
        )
        return query.where(maybe_negate(cond))

    def _get_direct_message_group_recipients(self, other_user: UserProfile) -> Set[int]:
        self_recipient_ids: List[int] = [recipient_tuple['recipient_id'] for recipient_tuple in Subscription.objects.filter(user_profile=self.user_profile, recipient__type=Recipient.DIRECT_MESSAGE_GROUP).values('recipient_id')]
        narrow_recipient_ids: List[int] = [recipient_tuple['recipient_id'] for recipient_tuple in Subscription.objects.filter(user_profile=other_user, recipient__type=Recipient.DIRECT_MESSAGE_GROUP).values('recipient_id')]
        return set(self_recipient_ids) & set(narrow_recipient_ids)

    def by_dm_including(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        assert not self.is_web_public_query
        assert self.user_profile is not None
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_dm_narrow=True)
        try:
            if isinstance(operand, str):
                narrow_user_profile: UserProfile = get_user_including_cross_realm(operand, self.realm)
            else:
                narrow_user_profile = get_user_by_id_in_realm_including_cross_realm(operand, self.realm)
        except UserProfile.DoesNotExist:
            raise BadNarrowOperatorError('unknown user ' + str(operand))
        if narrow_user_profile.id == self.user_profile.id:
            cond = column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0
            return query.where(maybe_negate(cond))
        direct_message_group_recipient_ids = self._get_direct_message_group_recipients(narrow_user_profile)
        self_recipient_id = self.user_profile.recipient_id
        cond = and_(
            column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0,
            column('realm_id', Integer) == self.realm.id,
            or_(
                and_(
                    column('sender_id', Integer) == narrow_user_profile.id,
                    column('recipient_id', Integer) == self_recipient_id
                ),
                and_(
                    column('sender_id', Integer) == self.user_profile.id,
                    column('recipient_id', Integer) == narrow_user_profile.recipient_id
                ),
                and_(
                    column('recipient_id', Integer).in_(direct_message_group_recipient_ids)
                )
            )
        )
        return query.where(maybe_negate(cond))

    def by_group_pm_with(self, query: ClauseElement, operand: Any, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        assert not self.is_web_public_query
        assert self.user_profile is not None
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_dm_narrow=True)
        try:
            if isinstance(operand, str):
                narrow_profile: UserProfile = get_user_including_cross_realm(operand, self.realm)
            else:
                narrow_profile = get_user_by_id_in_realm_including_cross_realm(operand, self.realm)
        except UserProfile.DoesNotExist:
            raise BadNarrowOperatorError('unknown user ' + str(operand))
        recipient_ids = self._get_direct_message_group_recipients(narrow_profile)
        cond = and_(
            column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0,
            column('realm_id', Integer) == self.realm.id,
            column('recipient_id', Integer).in_(recipient_ids)
        )
        return query.where(maybe_negate(cond))

    def by_search(self, query: ClauseElement, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        if settings.USING_PGROONGA:
            return self._by_search_pgroonga(query, operand, maybe_negate)
        else:
            return self._by_search_tsearch(query, operand, maybe_negate)

    def _by_search_pgroonga(self, query: ClauseElement, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        match_positions_character = func.pgroonga_match_positions_character
        query_extract_keywords = func.pgroonga_query_extract_keywords
        operand_escaped = func.escape_html(operand, type_=Text)
        keywords = query_extract_keywords(operand_escaped)
        query = query.add_columns(match_positions_character(column('rendered_content', Text), keywords).label('content_matches'),
                                  match_positions_character(func.escape_html(topic_column_sa(), type_=Text), keywords).label('topic_matches'))
        condition = column('search_pgroonga', Text).op('&@~')(operand_escaped)
        return query.where(maybe_negate(condition))

    def _by_search_tsearch(self, query: ClauseElement, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> ClauseElement:
        tsquery = func.plainto_tsquery(literal('zulip.english_us_search'), literal(operand))
        query = query.add_columns(ts_locs_array(literal('zulip.english_us_search', Text), column('rendered_content', Text), tsquery).label('content_matches'),
                                  ts_locs_array(literal('zulip.english_us_search', Text), func.escape_html(topic_column_sa(), type_=Text), tsquery).label('topic_matches'))
        for term in re.findall('"[^"]+"|\\S+', operand):
            if term[0] == '"' and term[-1] == '"':
                term = term[1:-1]
                term = '%' + connection.ops.prep_for_like_query(term) + '%'
                cond = or_(column('content', Text).ilike(term), topic_column_sa().ilike(term))
                query = query.where(maybe_negate(cond))
        cond = column('search_tsvector', postgresql.TSVECTOR).op('@@')(tsquery)
        return query.where(maybe_negate(cond))

def ok_to_include_history(narrow: Optional[Seq[NarrowParameter]], user_profile: Optional[UserProfile], is_web_public_query: bool) -> bool:
    if is_web_public_query:
        assert user_profile is None
        return True
    assert user_profile is not None
    include_history: bool = False
    if narrow is not None:
        for term in narrow:
            if term.operator in channel_operators and (not term.negated):
                operand = term.operand
                if isinstance(operand, str):
                    include_history = can_access_stream_history_by_name(user_profile, operand)
                else:
                    include_history = can_access_stream_history_by_id(user_profile, operand)
            elif term.operator in channels_operators and term.operand == 'public' and (not term.negated) and user_profile.can_access_public_streams():
                include_history = True
        for term in narrow:
            if term.operator == 'is' and term.operand != 'resolved':
                include_history = False
    return include_history

def get_channel_from_narrow_access_unchecked(narrow: Optional[Seq[NarrowParameter]], realm: Realm) -> Optional[Stream]:
    if narrow is not None:
        for term in narrow:
            if term.operator in channel_operators:
                return get_stream_by_narrow_operand_access_unchecked(term.operand, realm)
    return None

def can_narrow_define_conversation(narrow: Seq[NarrowParameter]) -> bool:
    contains_channel_term: bool = False
    contains_topic_term: bool = False
    for term in narrow:
        if term.operator in ['dm', 'pm-with']:
            return True
        elif term.operator in ['stream', 'channel']:
            contains_channel_term = True
        elif term.operator == 'topic':
            contains_topic_term = True
        if contains_channel_term and contains_topic_term:
            return True
    return False

def update_narrow_terms_containing_empty_topic_fallback_name(narrow: Optional[Seq[NarrowParameter]]) -> Optional[Seq[NarrowParameter]]:
    if narrow is None:
        return narrow
    for term in narrow:
        if term.operator == 'topic':
            term.operand = maybe_rename_general_chat_to_empty_topic(term.operand)
            break
    return narrow

def update_narrow_terms_containing_with_operator(realm: Realm, maybe_user_profile: Optional[UserProfile],
                                                   narrow: Optional[List[NarrowParameter]]) -> Optional[List[NarrowParameter]]:
    if narrow is None:
        return narrow
    with_operator_terms: List[NarrowParameter] = list(filter(lambda term: term.operator == 'with', narrow))
    can_user_access_target_message: bool = True
    if len(with_operator_terms) > 1:
        raise InvalidOperatorCombinationError(_("Duplicate 'with' operators."))
    elif len(with_operator_terms) == 0:
        return narrow
    with_term = with_operator_terms[0]
    narrow.remove(with_term)
    try:
        message_id = int(with_term.operand)
    except ValueError:
        raise BadNarrowOperatorError(_("Invalid 'with' operator"))
    if maybe_user_profile is not None and maybe_user_profile.is_authenticated:
        try:
            message = access_message(maybe_user_profile, message_id)
        except JsonableError:
            can_user_access_target_message = False
    else:
        try:
            message = access_web_public_message(realm, message_id)
        except MissingAuthenticationError:
            can_user_access_target_message = False
    if not can_user_access_target_message:
        if can_narrow_define_conversation(narrow):
            return narrow
        else:
            raise BadNarrowOperatorError(_("Invalid 'with' operator"))
    filtered_terms: List[NarrowParameter] = [term for term in narrow if term.operator not in ['stream', 'channel', 'topic', 'dm', 'pm-with']]
    if message.recipient.type == Recipient.STREAM:
        channel_id = message.recipient.type_id
        topic = message.topic_name()
        channel_conversation_terms: List[NarrowParameter] = [NarrowParameter(operator='channel', operand=channel_id),
                                                              NarrowParameter(operator='topic', operand=topic)]
        return channel_conversation_terms + filtered_terms
    elif message.recipient.type == Recipient.PERSONAL:
        dm_conversation_terms: List[NarrowParameter] = [NarrowParameter(operator='dm', operand=[message.recipient.type_id])]
        return dm_conversation_terms + filtered_terms
    elif message.recipient.type == Recipient.DIRECT_MESSAGE_GROUP:
        huddle_user_ids = list(get_direct_message_group_user_ids(message.recipient))
        dm_conversation_terms = [NarrowParameter(operator='dm', operand=huddle_user_ids)]
        return dm_conversation_terms + filtered_terms
    raise AssertionError('Invalid recipient type')

def exclude_muting_conditions(user_profile: UserProfile, narrow: Seq[NarrowParameter]) -> List[Any]:
    conditions: List[Any] = []
    channel_id: Optional[int] = None
    try:
        channel = get_channel_from_narrow_access_unchecked(narrow, user_profile.realm)
        if channel is not None:
            channel_id = channel.id
    except Stream.DoesNotExist:
        pass
    conditions = exclude_stream_and_topic_mutes(conditions, user_profile, channel_id)
    return conditions

def get_base_query_for_search(realm_id: int, user_profile: Optional[UserProfile],
                              need_message: bool, need_user_message: bool) -> Tuple[ClauseElement, ColumnElement]:
    if not need_user_message:
        assert need_message
        query = select(column('id', Integer).label('message_id')).select_from(table('zerver_message')).where(column('realm_id', Integer) == literal(realm_id))
        inner_msg_id_col = literal_column('zerver_message.id', Integer)
        return (query, inner_msg_id_col)
    assert user_profile is not None
    if need_message:
        query = select(column('message_id', Integer), column('flags', Integer)).where(column('user_profile_id', Integer) == literal(user_profile.id)).select_from(
            join(table('zerver_usermessage'), table('zerver_message'), literal_column('zerver_usermessage.message_id', Integer) == literal_column('zerver_message.id', Integer)))
        inner_msg_id_col = column('message_id', Integer)
        return (query, inner_msg_id_col)
    query = select(column('message_id', Integer), column('flags', Integer)).where(column('user_profile_id', Integer) == literal(user_profile.id)).select_from(table('zerver_usermessage'))
    inner_msg_id_col = column('message_id', Integer)
    return (query, inner_msg_id_col)

def add_narrow_conditions(user_profile: Optional[UserProfile], inner_msg_id_col: ColumnElement, query: ClauseElement,
                          narrow: Optional[Seq[NarrowParameter]], is_web_public_query: bool, realm: Realm) -> Tuple[ClauseElement, bool]:
    is_search: bool = False
    if narrow is None:
        return (query, is_search)
    builder = NarrowBuilder(user_profile, inner_msg_id_col, realm, is_web_public_query)
    search_operands: List[str] = []
    for term in narrow:
        if term.operator == 'search':
            search_operands.append(term.operand)
        else:
            query = builder.add_term(query, term)
    if search_operands:
        is_search = True
        query = query.add_columns(topic_column_sa(), column('rendered_content', Text))
        search_term = NarrowParameter(operator='search', operand=' '.join(search_operands))
        query = builder.add_term(query, search_term)
    return (query, is_search)

def find_first_unread_anchor(sa_conn: Connection, user_profile: Optional[UserProfile],
                             narrow: Optional[Seq[NarrowParameter]]) -> int:
    if user_profile is None:
        return LARGER_THAN_MAX_MESSAGE_ID
    need_user_message: bool = True
    need_message: bool = True
    query, inner_msg_id_col = get_base_query_for_search(realm_id=user_profile.realm_id, user_profile=user_profile, need_message=need_message, need_user_message=need_user_message)
    query, is_search = add_narrow_conditions(user_profile=user_profile, inner_msg_id_col=inner_msg_id_col, query=query, narrow=narrow, is_web_public_query=False, realm=user_profile.realm)
    condition = column('flags', Integer).op('&')(UserMessage.flags.read.mask) == 0
    muting_conditions = exclude_muting_conditions(user_profile, narrow) if narrow is not None else []
    if muting_conditions:
        condition = and_(condition, *muting_conditions)
    first_unread_query = query.where(condition)
    first_unread_query = first_unread_query.order_by(inner_msg_id_col.asc()).limit(1)
    first_unread_result = list(sa_conn.execute(first_unread_query).fetchall())
    if len(first_unread_result) > 0:
        anchor = first_unread_result[0][0]
    else:
        anchor = LARGER_THAN_MAX_MESSAGE_ID
    return anchor

def parse_anchor_value(anchor_val: Optional[str], use_first_unread_anchor: bool) -> Optional[int]:
    """Given the anchor and use_first_unread_anchor parameters passed by
    the client, computes what anchor value the client requested,
    handling backwards-compatibility and the various string-valued
    fields.  We encode use_first_unread_anchor as anchor=None.
    """
    if use_first_unread_anchor:
        return None
    if anchor_val is None:
        raise JsonableError(_("Missing 'anchor' argument."))
    if anchor_val == 'oldest':
        return 0
    if anchor_val == 'newest':
        return LARGER_THAN_MAX_MESSAGE_ID
    if anchor_val == 'first_unread':
        return None
    try:
        anchor = int(anchor_val)
        if anchor < 0:
            return 0
        elif anchor > LARGER_THAN_MAX_MESSAGE_ID:
            return LARGER_THAN_MAX_MESSAGE_ID
        return anchor
    except ValueError:
        raise JsonableError(_('Invalid anchor'))

def limit_query_to_range(query: ClauseElement, num_before: int, num_after: int, anchor: int, include_anchor: bool,
                         anchored_to_left: bool, anchored_to_right: bool, id_col: ColumnElement,
                         first_visible_message_id: int) -> ClauseElement:
    need_before_query: bool = not anchored_to_left and num_before > 0
    need_after_query: bool = not anchored_to_right and num_after > 0
    need_both_sides: bool = need_before_query and need_after_query
    if need_both_sides:
        before_anchor = anchor - 1
        after_anchor = max(anchor, first_visible_message_id)
        before_limit = num_before
        after_limit = num_after + 1
    elif need_before_query:
        before_anchor = anchor - (not include_anchor)
        before_limit = num_before
        if not anchored_to_right:
            before_limit += include_anchor
    elif need_after_query:
        after_anchor = max(anchor + (not include_anchor), first_visible_message_id)
        after_limit = num_after + include_anchor
    if need_before_query:
        before_query = query
        if not anchored_to_right:
            before_query = before_query.where(id_col <= before_anchor)
        before_query = before_query.order_by(id_col.desc())
        before_query = before_query.limit(before_limit)
    if need_after_query:
        after_query = query
        if not anchored_to_left:
            after_query = after_query.where(id_col >= after_anchor)
        after_query = after_query.order_by(id_col.asc())
        after_query = after_query.limit(after_limit)
    if need_both_sides:
        return union_all(before_query.self_group(), after_query.self_group())
    elif need_before_query:
        return before_query
    elif need_after_query:
        return after_query
    else:
        return query.where(id_col == anchor)

MessageRowT = TypeVar('MessageRowT', bound=Sequence[Any])

@dataclass
class LimitedMessages(Generic[MessageRowT]):
    rows: List[MessageRowT]
    found_anchor: bool
    found_newest: bool
    found_oldest: bool
    history_limited: bool

def post_process_limited_query(rows: List[Row], num_before: int, num_after: int, anchor: int,
                               anchored_to_left: bool, anchored_to_right: bool,
                               first_visible_message_id: int) -> LimitedMessages[Row]:
    if first_visible_message_id > 0:
        visible_rows: List[Row] = [r for r in rows if r[0] >= first_visible_message_id]
    else:
        visible_rows = rows
    rows_limited: bool = len(visible_rows) != len(rows)
    if anchored_to_right:
        num_after = 0
        before_rows: List[Row] = visible_rows[:]
        anchor_rows: List[Row] = []
        after_rows: List[Row] = []
    else:
        before_rows = [r for r in visible_rows if r[0] < anchor]
        anchor_rows = [r for r in visible_rows if r[0] == anchor]
        after_rows = [r for r in visible_rows if r[0] > anchor]
    if num_before:
        before_rows = before_rows[-1 * num_before:]
    if num_after:
        after_rows = after_rows[:num_after]
    limited_rows: List[Row] = [*before_rows, *anchor_rows, *after_rows]
    found_anchor: bool = len(anchor_rows) == 1
    found_oldest: bool = anchored_to_left or len(before_rows) < num_before
    found_newest: bool = anchored_to_right or len(after_rows) < num_after
    history_limited: bool = rows_limited and found_oldest
    return LimitedMessages(rows=limited_rows, found_anchor=found_anchor, found_newest=found_newest, found_oldest=found_oldest, history_limited=history_limited)

def clean_narrow_for_message_fetch(narrow: Optional[Seq[NarrowParameter]], realm: Realm,
                                   maybe_user_profile: Optional[UserProfile]) -> Optional[Seq[NarrowParameter]]:
    narrow = update_narrow_terms_containing_empty_topic_fallback_name(narrow)
    narrow = update_narrow_terms_containing_with_operator(realm, maybe_user_profile, narrow)  # type: ignore
    return narrow

@dataclass
class FetchedMessages(LimitedMessages[Row]):
    anchor: Optional[int]
    include_history: bool
    is_search: bool

def fetch_messages(*, narrow: Optional[Seq[NarrowParameter]], user_profile: Optional[UserProfile], realm: Realm,
                   is_web_public_query: bool, anchor: Optional[int], include_anchor: bool, num_before: int,
                   num_after: int, client_requested_message_ids: Optional[Seq[int]] = None) -> FetchedMessages:
    include_history: bool = ok_to_include_history(narrow, user_profile, is_web_public_query)
    if include_history:
        need_message: bool = True
        need_user_message: bool = False
    elif narrow is None:
        need_message = False
        need_user_message = True
    else:
        need_message = True
        need_user_message = True
    query, inner_msg_id_col = get_base_query_for_search(realm_id=realm.id, user_profile=user_profile, need_message=need_message, need_user_message=need_user_message)
    query, is_search = add_narrow_conditions(user_profile=user_profile, inner_msg_id_col=inner_msg_id_col, query=query, narrow=narrow, realm=realm, is_web_public_query=is_web_public_query)
    anchored_to_left: bool = False
    anchored_to_right: bool = False
    first_visible_message_id: int = get_first_visible_message_id(realm)
    with get_sqlalchemy_connection() as sa_conn:
        if client_requested_message_ids is not None:
            query = query.filter(inner_msg_id_col.in_(client_requested_message_ids))
        else:
            if anchor is None:
                anchor = find_first_unread_anchor(sa_conn, user_profile, narrow)
            anchored_to_left = anchor == 0
            anchored_to_right = anchor >= LARGER_THAN_MAX_MESSAGE_ID
            if anchored_to_right:
                num_after = 0
            query = limit_query_to_range(query=query, num_before=num_before, num_after=num_after, anchor=anchor, include_anchor=include_anchor, anchored_to_left=anchored_to_left, anchored_to_right=anchored_to_right, id_col=inner_msg_id_col, first_visible_message_id=first_visible_message_id)
            main_query = query.subquery()
            query = select(*main_query.c).select_from(main_query).order_by(column('message_id', Integer).asc())
        query = query.prefix_with('/* get_messages */')
        rows: List[Row] = list(sa_conn.execute(query).fetchall())
    if client_requested_message_ids is not None:
        if first_visible_message_id > 0:
            visible_rows = [r for r in rows if r[0] >= first_visible_message_id]
        else:
            visible_rows = rows
        return FetchedMessages(rows=visible_rows, found_anchor=False, found_newest=False, found_oldest=False, history_limited=False, anchor=None, include_history=include_history, is_search=is_search)
    assert anchor is not None
    query_info = post_process_limited_query(rows=rows, num_before=num_before, num_after=num_after, anchor=anchor, anchored_to_left=anchored_to_left, anchored_to_right=anchored_to_right, first_visible_message_id=first_visible_message_id)
    return FetchedMessages(rows=query_info.rows, found_anchor=query_info.found_anchor, found_newest=query_info.found_newest, found_oldest=query_info.found_oldest, history_limited=query_info.history_limited, anchor=anchor, include_history=include_history, is_search=is_search)