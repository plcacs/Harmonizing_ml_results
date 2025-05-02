import abc
import asyncio
import typing
from typing import Any, Mapping, Optional, Pattern, Sequence, Set, Union
from mode import Seconds
from mode.utils.queues import ThrowableQueue
from .channels import ChannelT
from .codecs import CodecArg
from .tuples import TP
if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .models import ModelArg as _ModelArg
    from .serializers import SchemaT as _SchemaT
else:

    class _AppT:
        ...

    class _ModelArg:
        ...

    class _SchemaT:
        ...
__all__ = ['TopicT']

class TopicT(ChannelT):
    has_prefix = False

    @abc.abstractmethod
    def __init__(self, app, *, topics=None, pattern=None, schema=None, key_type=None, value_type=None, is_iterator=False, partitions=None, retention=None, compacting=None, deleting=None, replicas=None, acks=True, internal=False, config=None, queue=None, key_serializer=None, value_serializer=None, maxsize=None, root=None, active_partitions=None, allow_empty=False, has_prefix=False, loop=None):
        ...

    @property
    @abc.abstractmethod
    def pattern(self):
        ...

    @pattern.setter
    def pattern(self, pattern):
        ...

    @property
    @abc.abstractmethod
    def partitions(self):
        ...

    @partitions.setter
    def partitions(self, partitions):
        ...

    @abc.abstractmethod
    def derive(self, **kwargs):
        ...

    @abc.abstractmethod
    def derive_topic(self, *, topics=None, schema=None, key_type=None, value_type=None, partitions=None, retention=None, compacting=None, deleting=None, internal=False, config=None, prefix='', suffix='', **kwargs):
        ...