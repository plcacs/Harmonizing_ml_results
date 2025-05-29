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
    topics: Sequence[str]
    retention: Optional[Seconds]
    compacting: Optional[bool]
    deleting: Optional[bool]
    replicas: Optional[int]
    config: Optional[Mapping[str, Any]]
    acks: bool
    internal: bool
    has_prefix: bool = False
    active_partitions: Optional[Set[TP]]

    @abc.abstractmethod
    def __init__(self, app, *, topics: Optional[Sequence[str]]=None,
        pattern: Optional[Union[str, Pattern]]=None, schema: Optional[
        _SchemaT]=None, key_type: Optional[_ModelArg]=None, value_type:
        Optional[_ModelArg]=None, is_iterator: bool=False, partitions:
        Optional[int]=None, retention: Optional[Seconds]=None, compacting:
        Optional[bool]=None, deleting: Optional[bool]=None, replicas:
        Optional[int]=None, acks: bool=True, internal: bool=False, config:
        Optional[Mapping[str, Any]]=None, queue: Optional[ThrowableQueue]=
        None, key_serializer: Optional[CodecArg]=None, value_serializer:
        Optional[CodecArg]=None, maxsize: Optional[int]=None, root:
        Optional[ChannelT]=None, active_partitions: Optional[Set[TP]]=None,
        allow_empty: bool=False, has_prefix: bool=False, loop: Optional[
        asyncio.AbstractEventLoop]=None):
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
    def derive(self, **kwargs: Any):
        ...

    @abc.abstractmethod
    def derive_topic(self, *, topics: Optional[Sequence[str]]=None, schema:
        Optional[_SchemaT]=None, key_type: Optional[_ModelArg]=None,
        value_type: Optional[_ModelArg]=None, partitions: Optional[int]=
        None, retention: Optional[Seconds]=None, compacting: Optional[bool]
        =None, deleting: Optional[bool]=None, internal: bool=False, config:
        Optional[Mapping[str, Any]]=None, prefix: str='', suffix: str='',
        **kwargs: Any):
        ...
