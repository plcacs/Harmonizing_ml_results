import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeAlias, TypeVar, Union, List, Dict, cast
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

T = TypeVar('T')
ConditionTransform: TypeAlias = Callable[[ClauseElement], ClauseElement]
MessageRowT = TypeVar('MessageRowT', bound=Sequence[Any])
NarrowParameterValue = Union[str, int, List[Union[str, int]]]

class NarrowParameter(BaseModel):
    operator: str
    operand: NarrowParameterValue
    negated: bool = False

    @model_validator(mode='before')
    @classmethod
    def convert_term(cls, elem: Any) -> Dict[str, Any]:
        if isinstance(elem, list):
            if len(elem) != 2 or any((not isinstance(x, str) for x in elem)):
                raise ValueError('element is not a string pair')
            return {'operator': elem[0], 'operand': elem[1]}
        elif isinstance(elem, dict):
            if 'operand' not in elem or elem['operand'] is None:
                raise ValueError('operand is missing')
            if 'operator' not in elem or elem['operator'] is None:
                raise ValueError('operator is missing')
            return elem
        else:
            raise ValueError('dict or list required')

    @model_validator(mode='after')
    def validate_terms(self) -> 'NarrowParameter':
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

def is_spectator_compatible(narrow: Optional[List[NarrowParameter]]) -> bool:
    supported_operators = [*channel_operators, *channels_operators, 'topic', 'sender', 'has', 'search', 'near', 'id', 'with']
    if narrow is None:
        return False
    for element in narrow:
        operator = element.operator
        if operator not in supported_operators:
            return False
    return True

def is_web_public_narrow(narrow: Optional[List[NarrowParameter]]) -> bool:
    if narrow is None:
        return False
    return any((term.operator in channels_operators and term.operand == 'web-public' and (term.negated is False) for term in narrow))

LARGER_THAN_MAX_MESSAGE_ID: int = 10000000000000000

class BadNarrowOperatorError(JsonableError):
    code: ErrorCode = ErrorCode.BAD_NARROW
    data_fields: List[str] = ['desc']

    def __init__(self, desc: str) -> None:
        self.desc: str = desc

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Invalid narrow operator: {desc}')

class InvalidOperatorCombinationError(JsonableError):
    code: ErrorCode = ErrorCode.BAD_NARROW
    data_fields: List[str] = ['desc']

    def __init__(self, desc: str) -> None:
        self.desc: str = desc

    @staticmethod
    @override
    def msg_format() -> str:
        return _('Invalid narrow operator combination: {desc}')

TS_START: str = '<ts-match>'
TS_STOP: str = '</ts-match>'

def ts_locs_array(config: str, text: ColumnElement, tsquery: ColumnElement) -> ColumnElement:
    options = f'HighlightAll = TRUE, StartSel = {TS_START}, StopSel = {TS_STOP}'
    delimited = func.ts_headline(config, text, tsquery, options, type_=Text)
    part = func.unnest(func.string_to_array(delimited, TS_START, type_=ARRAY(Text)), type_=Text).column_valued()
    part_len = func.length(part, type_=Integer) - len(TS_STOP)
    match_pos = func.sum(part_len, type_=Integer).over(rows=(None, -1)) + len(TS_STOP)
    match_len = func.strpos(part, TS_STOP, type_=Integer) - 1
    return func.array(select(postgresql.array([match_pos, match_len])).offset(1).scalar_subquery(), type_=ARRAY(Integer))

class NarrowBuilder:
    def __init__(
        self,
        user_profile: Optional[UserProfile],
        msg_id_column: ColumnElement,
        realm: Realm,
        is_web_public_query: bool = False
    ) -> None:
        self.user_profile: Optional[UserProfile] = user_profile
        self.msg_id_column: ColumnElement = msg_id_column
        self.realm: Realm = realm
        self.is_web_public_query: bool = is_web_public_query
        self.by_method_map: Dict[str, Callable[..., Select]] = {
            'has': self.by_has, 'in': self.by_in, 'is': self.by_is, 
            'channel': self.by_channel, 'stream': self.by_channel,
            'channels': self.by_channels, 'streams': self.by_channels,
            'topic': self.by_topic, 'sender': self.by_sender,
            'near': self.by_near, 'id': self.by_id, 'search': self.by_search,
            'dm': self.by_dm, 'pm-with': self.by_dm,
            'dm-including': self.by_dm_including,
            'group-pm-with': self.by_group_pm_with,
            'pm_with': self.by_dm, 'group_pm_with': self.by_group_pm_with
        }
        self.is_channel_narrow: bool = False
        self.is_dm_narrow: bool = False

    def check_not_both_channel_and_dm_narrow(
        self,
        maybe_negate: Callable[[ClauseElement], ClauseElement],
        is_dm_narrow: bool = False,
        is_channel_narrow: bool = False
    ) -> None:
        if maybe_negate is not_:
            return
        if is_dm_narrow:
            self.is_dm_narrow = True
        if is_channel_narrow:
            self.is_channel_narrow = True
        if self.is_channel_narrow and self.is_dm_narrow:
            raise BadNarrowOperatorError('No message can be both a channel message and direct message')

    def add_term(self, query: Select, term: NarrowParameter) -> Select:
        operator = term.operator
        operand = term.operand
        negated = term.negated
        if operator in self.by_method_map:
            method = self.by_method_map[operator]
        else:
            raise BadNarrowOperatorError('unknown operator ' + operator)
        if negated:
            maybe_negate = not_
        else:
            maybe_negate = lambda cond: cond
        return method(query, operand, maybe_negate)

    def by_has(self, query: Select, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> Select:
        if operand not in ['attachment', 'image', 'link', 'reaction']:
            raise BadNarrowOperatorError("unknown 'has' operand " + operand)
        if operand == 'reaction':
            if self.msg_id_column.name == 'message_id':
                check_col = literal_column('zerver_usermessage.message_id', Integer)
            else:
                check_col = literal_column('zerver_message.id', Integer)
            exists_cond = select(1).select_from(table('zerver_reaction')).where(check_col == literal_column('zerver_reaction.message_id', Integer)).exists()
            return query.where(maybe_negate(exists_cond))
        col_name = 'has_' + operand
        cond = column(col_name, Boolean)
        return query.where(maybe_negate(cond))

    def by_in(self, query: Select, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> Select:
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

    def by_is(self, query: Select, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> Select:
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
        s = list(pattern)
        for i, c in enumerate(s):
            if c not in self._alphanum:
                if ord(c) >= 128:
                    s[i] = f'\\u{ord(c):0>4x}'
                else:
                    s[i] = '\\' + c
        return ''.join(s)

    def by_channel(self, query: Select, operand: Union[str, int], maybe_negate: Callable[[ClauseElement], ClauseElement]) -> Select:
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_channel_narrow=True)
        try:
            channel = get_stream_by_narrow_operand_access_unchecked(operand, self.realm)
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
            recipient_ids = [matching_channel.recipient_id for matching_channel in matching_channels]
            cond = column('recipient_id', Integer).in_(recipient_ids)
            return query.where(maybe_negate(cond))
        recipient_id = channel.recipient_id
        assert recipient_id is not None
        cond = column('recipient_id', Integer) == recipient_id
        return query.where(maybe_negate(cond))

    def by_channels(self, query: Select, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> Select:
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

    def by_topic(self, query: Select, operand: str, maybe_negate: Callable[[ClauseElement], ClauseElement]) -> Select:
        self.check_not_both_channel_and_dm_narrow(maybe_negate, is_channel_narrow=True)
        if self.realm.is_zephyr_mirror_realm:
            m = re.search('^(.*?)(?:\\.d)*$', operand, re.IGNORECASE)
            assert m is not None
            base_topic = m.group(1)
            if base_topic in ('', 'personal', '(instance "")'):
                cond = or_(topic_match_sa(''), topic_match_sa('.d'), topic_match_sa('.d.d'), topic_match_sa('.d.d.d'), topic_match_sa('.d.d.d.d'), topic_match_sa('personal'), topic_match_sa('personal.d'), topic_match_sa('personal.d.d'), topic_match_sa('personal.d.d.d'), topic_match_sa('personal.d.d.d.d'), topic_match_sa('(instance "")'), topic_match_sa('(instance "").d'), topic_match_sa('(instance "").d.d'), topic_match_sa('(instance "").d.d.d'), topic_match_sa('(instance "").d.d.d.d'))
            else:
                cond = or_(topic_match_sa(base_topic), topic_match_s