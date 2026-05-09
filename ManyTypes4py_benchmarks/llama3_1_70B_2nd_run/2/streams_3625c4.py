class Stream(StreamT[T_co], Service):
    """A stream: async iterator processing events in channels/topics."""
    logger: logging.Logger
    mundane_level: str
    events_total: int
    _anext_started: bool
    _passive: bool
    _finalized: bool

    def __init__(self, channel: ChannelT, *, app: AppT, processors: Optional[List[Processor]] = None, 
                 combined: Optional[List[JoinableT]] = None, on_start: Optional[Callable] = None, 
                 join_strategy: Optional[JoinT] = None, beacon: Optional[Service] = None, 
                 concurrency_index: Optional[int] = None, prev: Optional[StreamT] = None, 
                 active_partitions: Optional[List[int]] = None, enable_acks: bool = True, 
                 prefix: str = '', loop: Optional[asyncio.AbstractEventLoop] = None):
        Service.__init__(self, loop=loop, beacon=beacon)
        self.app: AppT = app
        self.channel: ChannelT = channel
        self.outbox: ThrowableQueue = self.app.FlowControlQueue(maxsize=self.app.conf.stream_buffer_maxsize, 
                                                                loop=self.loop, clear_on_resume=True)
        self._passive_started: asyncio.Event = asyncio.Event(loop=self.loop)
        self.join_strategy: Optional[JoinT] = join_strategy
        self.combined: List[JoinableT] = combined if combined is not None else []
        self.concurrency_index: Optional[int] = concurrency_index
        self._prev: Optional[StreamT] = prev
        self.active_partitions: Optional[List[int]] = active_partitions
        self.enable_acks: bool = enable_acks
        self.prefix: str = prefix
        self._processors: List[Processor] = list(processors) if processors else []
        self._on_start: Optional[Callable] = on_start
        task: Optional[asyncio.Task] = current_task(loop=self.loop)
        if task is not None:
            self.task_owner: asyncio.Task = task
        self._on_stream_event_in: Callable = self.app.sensors.on_stream_event_in
        self._on_stream_event_out: Callable = self.app.sensors.on_stream_event_out
        self._on_message_in: Callable = self.app.sensors.on_message_in
        self._on_message_out: Callable = self.app.sensors.on_message_out
        self._skipped_value: object = object()

    def get_active_stream(self) -> StreamT:
        """Return the currently active stream."""
        return list(self._iter_ll_forwards())[-1]

    def get_root_stream(self) -> StreamT:
        """Get the root stream that this stream was derived from."""
        return list(self._iter_ll_backwards())[-1]

    def _iter_ll_forwards(self) -> Iterator[StreamT]:
        return self._iter_ll(_LinkedListDirectionFwd)

    def _iter_ll_backwards(self) -> Iterator[StreamT]:
        return self._iter_ll(_LinkedListDirectionBwd)

    def _iter_ll(self, dir_: _LinkedListDirection) -> Iterator[StreamT]:
        node: StreamT = self
        seen: Set[StreamT] = set()
        while node:
            if node in seen:
                raise RuntimeError('Loop in Stream.{dir_.attr}: Call support!')
            seen.add(node)
            yield node
            node = dir_.getter(node)

    def add_processor(self, processor: Processor) -> None:
        """Add processor callback executed whenever a new event is received."""
        self._processors.append(processor)

    def info(self) -> Dict[str, Any]:
        """Return stream settings as a dictionary."""
        return {'app': self.app, 'channel': self.channel, 'processors': self._processors, 
                'on_start': self._on_start, 'loop': self.loop, 'combined': self.combined, 
                'beacon': self.beacon, 'concurrency_index': self.concurrency_index, 
                'prev': self._prev, 'active_partitions': self.active_partitions}

    def clone(self, **kwargs: Any) -> StreamT:
        """Create a clone of this stream."""
        return self.__class__(**{**self.info(), **kwargs})

    def _chain(self, **kwargs: Any) -> StreamT:
        assert not self._finalized
        self._next: StreamT = new_stream = self.clone(on_start=self.maybe_start, prev=self, 
                                                      processors=list(self._processors), **kwargs)
        self._processors.clear()
        return new_stream

    def noack(self) -> StreamT:
        """Create new stream where acks are manual."""
        self._next: StreamT = new_stream = self.clone(enable_acks=False)
        return new_stream

    async def items(self) -> AsyncIterator[Tuple[K, T_co]]:
        """Iterate over the stream as ``key, value`` pairs."""
        async for event in self.events():
            yield (event.key, cast(T_co, event.value))

    async def events(self) -> AsyncIterator[EventT]:
        """Iterate over the stream as events exclusively."""
        async for _ in self:
            if self.current_event is not None:
                yield self.current_event

    async def take(self, max_: int, within: Seconds) -> AsyncIterator[List[T_co]]:
        """Buffer n values at a time and yield a list of buffered values."""
        buffer: List[T_co] = []
        events: List[EventT] = []
        buffer_add: Callable[[T_co], None] = buffer.append
        event_add: Callable[[EventT], None] = events.append
        buffer_size: Callable[[], int] = buffer.__len__
        buffer_full: asyncio.Event = asyncio.Event(loop=self.loop)
        buffer_consumed: asyncio.Event = asyncio.Event(loop=self.loop)
        timeout: Optional[Seconds] = want_seconds(within) if within else None
        stream_enable_acks: bool = self.enable_acks
        buffer_consuming: Optional[asyncio.Future] = None
        channel_it: AsyncIterator = aiter(self.channel)

        async def add_to_buffer(value: T_co) -> T_co:
            try:
                nonlocal buffer_consuming
                if buffer_consuming is not None:
                    try:
                        await buffer_consuming
                    finally:
                        buffer_consuming = None
                buffer_add(cast(T_co, value))
                event: EventT = self.current_event
                if event is None:
                    raise RuntimeError('Take buffer found current_event is None')
                event_add(event)
                if buffer_size() >= max_:
                    buffer_full.set()
                    buffer_consumed.clear()
                    await self.wait(buffer_consumed)
            except CancelledError:
                raise
            except Exception as exc:
                self.log.exception('Error adding to take buffer: %r', exc)
                await self.crash(exc)
            return value
        self.enable_acks = False
        self.add_processor(add_to_buffer)
        self._enable_passive(cast(ChannelT, channel_it))
        try:
            while not self.should_stop:
                await self.wait_for_stopped(buffer_full, timeout=timeout)
                if buffer:
                    buffer_consuming = self.loop.create_future()
                    try:
                        yield list(buffer)
                    finally:
                        buffer.clear()
                        for event in events:
                            await self.ack(event)
                        events.clear()
                        notify(buffer_consuming)
                        buffer_full.clear()
                        buffer_consumed.set()
                else:
                    pass
            else:
                pass
        finally:
            self.enable_acks = stream_enable_acks
            self._processors.remove(add_to_buffer)

    def enumerate(self, start: int = 0) -> AsyncIterator[Tuple[int, T_co]]:
        """Enumerate values received on this stream."""
        return aenumerate(self, start)

    def through(self, channel: ChannelT) -> StreamT:
        """Forward values to in this stream to channel."""
        if self._finalized:
            return self
        if self.concurrency_index is not None:
            raise ImproperlyConfigured('Agent with concurrency>1 cannot use stream.through!')
        if isinstance(channel, str):
            channelchannel: ChannelT = cast(ChannelT, self.derive_topic(channel))
        else:
            channelchannel: ChannelT = channel
        channel_it: AsyncIterator = aiter(channelchannel)
        if self._next is not None:
            raise ImproperlyConfigured('Stream is already using group_by/through')
        through: StreamT = self._chain(channel=channel_it)

        async def forward(value: T_co) -> T_co:
            event: EventT = self.current_event
            return await maybe_forward(event, channelchannel)
        self.add_processor(forward)
        self._enable_passive(cast(ChannelT, channel_it), declare=True)
        return through

    def _enable_passive(self, channel: ChannelT, *, declare: bool = False) -> None:
        if not self._passive:
            self._passive = True
            self.add_future(self._passive_drainer(channel, declare))

    async def _passive_drainer(self, channel: ChannelT, declare: bool = False) -> None:
        try:
            if declare:
                await channel.maybe_declare()
            self._passive_started.set()
            try:
                async for item in self:
                    ...
            except BaseException as exc:
                await channel.throw(exc)
        finally:
            self._channel_stop_iteration(channel)
            self._passive = False

    def _channel_stop_iteration(self, channel: ChannelT) -> None:
        try:
            on_stop_iteration: Callable = channel.on_stop_iteration
        except AttributeError:
            pass
        else:
            on_stop_iteration()

    def echo(self, *channels: ChannelT) -> StreamT:
        """Forward values to one or more channels."""
        _channels: List[ChannelT] = [self.derive_topic(c) if isinstance(c, str) else c for c in channels]

        async def echoing(value: T_co) -> T_co:
            await asyncio.wait([maybe_forward(value, channel) for channel in _channels], 
                               loop=self.loop, return_when=asyncio.ALL_COMPLETED)
            return value
        self.add_processor(echoing)
        return self

    def group_by(self, key: Union[FieldDescriptorT, Callable[[T_co], K]], *, name: Optional[str] = None, 
                 topic: Optional[TopicT] = None, partitions: Optional[int] = None) -> StreamT:
        """Create new stream that repartitions the stream using a new key."""
        if self._finalized:
            return self
        if self.concurrency_index is not None:
            raise ImproperlyConfigured('Agent with concurrency>1 cannot use stream.group_by!')
        if not name:
            if isinstance(key, FieldDescriptorT):
                name = key.ident
            else:
                raise TypeError('group_by with callback must set name=topic_suffix')
        if topic is not None:
            channel: ChannelT = topic
        else:
            prefix: str = ''
            if self.prefix and (not cast(TopicT, self.channel).has_prefix):
                prefix = self.prefix + '-'
            suffix: str = f'-{name}-repartition'
            p: Optional[int] = partitions if partitions else self.app.conf.topic_partitions
            channel: ChannelT = cast(ChannelT, self.channel).derive(prefix=prefix, suffix=suffix, partitions=p, internal=True)
        format_key: Callable[[Union[FieldDescriptorT, Callable[[T_co], K]], T_co], K] = self._format_key
        channel_it: AsyncIterator = aiter(channel)
        if self._next is not None:
            raise ImproperlyConfigured('Stream already uses group_by/through')
        grouped: StreamT = self._chain(channel=channel_it)

        async def repartition(value: T_co) -> T_co:
            event: EventT = self.current_event
            if event is None:
                raise RuntimeError('Cannot repartition stream with non-topic channel')
            new_key: K = await format_key(key, value)
            await event.forward(channel, key=new_key)
            return value
        self.add_processor(repartition)
        self._enable_passive(cast(ChannelT, channel_it), declare=True)
        return grouped

    def filter(self, fun: Callable[[T_co], bool]) -> StreamT:
        """Filter values from stream using callback."""
        async def on_value(value: T_co) -> T_co:
            if not await maybe_async(fun(value)):
                raise Skip()
            else:
                return value
        self.add_processor(on_value)
        return self

    async def _format_key(self, key: Union[FieldDescriptorT, Callable[[T_co], K]], value: T_co) -> K:
        try:
            if isinstance(key, FieldDescriptorT):
                return key.getattr(cast(ModelT, value))
            return await maybe_async(cast(Callable, key)(value))
        except BaseException as exc:
            self.log.exception('Error in grouping key : %r', exc)
            raise Skip() from exc

    def derive_topic(self, name: str, *, schema: Optional[SchemaT] = None, key_type: Optional[ModelT] = None, 
                     value_type: Optional[ModelT] = None, prefix: str = '', suffix: str = '') -> TopicT:
        """Create Topic description derived from the K/V type of this stream."""
        if isinstance(self.channel, TopicT):
            return cast(TopicT, self.channel).derive_topic(topics=[name], schema=schema, key_type=key_type, 
                                                           value_type=value_type, prefix=prefix, suffix=suffix)
        raise ValueError('Cannot derive topic from non-topic channel.')

    async def throw(self, exc: Exception) -> None:
        """Send exception to stream iteration."""
        await cast(ChannelT, self.channel).throw(exc)

    def combine(self, *nodes: JoinableT, **kwargs: Any) -> StreamT:
        """Combine streams and tables into joined stream."""
        if self._finalized:
            return self
        stream: StreamT = self._chain(combined=self.combined + list(nodes))
        for node in stream.combined:
            node.contribute_to_stream(stream)
        return stream

    def contribute_to_stream(self, active: StreamT) -> None:
        """Add stream as node in joined stream."""
        self.outbox = active.outbox

    async def remove_from_stream(self, stream: StreamT) -> None:
        """Remove as node in a joined stream."""
        await self.stop()

    def join(self, *fields: str) -> StreamT:
        """Create stream where events are joined."""
        return self._join(joins.RightJoin(stream=self, fields=fields))

    def left_join(self, *fields: str) -> StreamT:
        """Create stream where events are joined by LEFT JOIN."""
        return self._join(joins.LeftJoin(stream=self, fields=fields))

    def inner_join(self, *fields: str) -> StreamT:
        """Create stream where events are joined by INNER JOIN."""
        return self._join(joins.InnerJoin(stream=self, fields=fields))

    def outer_join(self, *fields: str) -> StreamT:
        """Create stream where events are joined by OUTER JOIN."""
        return self._join(joins.OuterJoin(stream=self, fields=fields))

    def _join(self, join_strategy: JoinT) -> StreamT:
        return self.clone(join_strategy=join_strategy)

    async def on_merge(self, value: Optional[T_co] = None) -> Optional[T_co]:
        """Signal called when an event is to be joined."""
        join_strategy: Optional[JoinT] = self.join_strategy
        if join_strategy:
            value = await join_strategy.process(value)
        return value

    async def on_start(self) -> None:
        """Signal called when the stream starts."""
        if self._on_start:
            await self._on_start()
        if self._passive:
            await self._passive_started.wait()

    async def stop(self) -> None:
        """Stop this stream."""
        for s in cast(Stream, self.get_root_stream())._iter_ll_forwards():
            await Service.stop(cast(Service, s))

    async def on_stop(self) -> None:
        """Signal that the stream is stopping."""
        self._passive = False
        self._passive_started.clear()
        for table_or_stream in self.combined:
            await table_or_stream.remove_from_stream(self)

    def __iter__(self) -> Iterator:
        raise NotImplementedError('Streams are asynchronous: use `async for`')

    def __next__(self) -> NoReturn:
        raise NotImplementedError('Streams are asynchronous: use `async for`')

    def __aiter__(self) -> AsyncIterator[T_co]:
        if _CStreamIterator is not None:
            return self._c_aiter()
        else:
            return self._py_aiter()

    async def _c_aiter(self) -> AsyncIterator[T_co]:
        self.log.dev('Using Cython optimized __aiter__')
        skipped_value: object = self._skipped_value
        self._finalized = True
        started_by_aiter: bool = await self.maybe_start()
        it: _CStreamIterator = _CStreamIterator(self)
        try:
            while not self.should_stop:
                do_ack: bool = self.enable_acks
                value: T_co
                sensor_state: Any
                value, sensor_state = await it.next()
                try:
                    if value is not skipped_value:
                        self.events_total += 1
                        yield value
                finally:
                    event: EventT
                    self.current_event: EventT
                    event, self.current_event = (self.current_event, None)
                    it.after(event, do_ack, sensor_state)
        except StopAsyncIteration:
            return
        finally:
            self._channel_stop_iteration(self.channel)
            if started_by_aiter:
                await self.stop()
                self.service_reset()

    def _set_current_event(self, event: Optional[EventT] = None) -> None:
        if event is None:
            _current_event.set(None)
        else:
            _current_event.set(weakref.ref(event))
        self.current_event: EventT = event

    async def _py_aiter(self) -> AsyncIterator[T_co]:
        self._finalized = True
        loop: asyncio.AbstractEventLoop = self.loop
        started_by_aiter: bool = await self.maybe_start()
        on_merge: Callable[[Optional[T_co]], Optional[T_co]] = self.on_merge
        on_stream_event_out: Callable = self._on_stream_event_out
        on_message_out: Callable = self._on_message_out
        channel: ChannelT = self.channel
        if isinstance(channel, ChannelT):
            chan_is_channel: bool = True
            chan: ChannelT = cast(ChannelT, self.channel)
            chan_queue: ThrowableQueue = chan.queue
            chan_queue_empty: Callable[[], bool] = chan_queue.empty
            chan_errors: Deque = chan_queue._errors
            chan_quick_get: Callable[[T_co], T_co] = chan_queue.get_nowait
        else:
            chan_is_channel: bool = False
            chan_queue: ThrowableQueue = cast(ThrowableQueue, None)
            chan_queue_empty: Callable[[], bool] = cast(Callable, None)
            chan_errors: Deque = cast(Deque, None)
            chan_quick_get: Callable[[T_co], T_co] = cast(Callable, None)
        chan_slow_get: Callable[[T_co], T_co] = channel.__anext__
        processors: List[Processor] = self._processors
        on_stream_event_in: Callable = self._on_stream_event_in
        create_ref: Callable[[EventT], weakref.ref] = weakref.ref
        _maybe_async: Callable[[Callable[[T_co], T_co]], Callable[[T_co], T_co]] = maybe_async
        event_cls: Type[EventT] = EventT
        _current_event_contextvar: ContextVar = _current_event
        consumer: ConsumerT = self.app.consumer
        unacked: Set[Message] = consumer.unacked
        add_unacked: Callable[[Message], None] = unacked.add
        acking_topics: Set[TopicT] = self.app.topics.acking_topics
        on_message_in: Callable = self._on_message_in
        sleep: Callable[[float, Optional[asyncio.AbstractEventLoop]], Awaitable] = asyncio.sleep
        trace: Callable[[str], Any] = self.app.trace
        _shortlabel: Callable[[Any], str] = shortlabel
        sensor_state: Any = None
        skipped_value: object = self._skipped_value
        try:
            while not self.should_stop:
                event: Optional[EventT] = None
                do_ack: bool = self.enable_acks
                value: Optional[T_co] = None
                while value is None and event is None:
                    await sleep(0, loop=loop)
                    if chan_is_channel:
                        if chan_errors:
                            raise chan_errors.popleft()
                        if chan_queue_empty():
                            channel_value: T_co = await chan_slow_get()
                        else:
                            channel_value: T_co = chan_quick_get()
                    else:
                        channel_value: T_co = await chan_slow_get()
                    if isinstance(channel_value, event_cls):
                        event: EventT = channel_value
                        message: Message = event.message
                        tp: TP = message.tp
                        offset: int = message.offset
                        if message.topic in acking_topics and (not message.tracked):
                            message.tracked = True
                            add_unacked(message)
                            on_message_in(message.tp, message.offset, message)
                        sensor_state = on_stream_event_in(tp, offset, self, event)
                        _current_event_contextvar.set(create_ref(event))
                        self.current_event: EventT = event
                        value: T_co = event.value
                    else:
                        value: T_co = channel_value
                        self.current_event: EventT = None
                        sensor_state: Any = None
                    try:
                        for processor in processors:
                            with trace(f'processor-{_shortlabel(processor)}'):
                                value: T_co = await _maybe_async(processor(value))
                        value: T_co = await on_merge(value)
                    except Skip:
                        value: T_co = skipped_value
                if value is skipped_value:
                    continue
                self.events_total += 1
                try:
                    yield value
                finally:
                    self.current_event: EventT = None
                    if do_ack and event is not None:
                        last_stream_to_ack: bool = event.ack()
                        message: Message = event.message
                        tp: TP = event.message.tp
                        offset: int = event.message.offset
                        on_stream_event_out(tp, offset, self, event, sensor_state)
                        if last_stream_to_ack:
                            on_message_out(tp, offset, message)
        except StopAsyncIteration:
            return
        finally:
            self._channel_stop_iteration(channel)
            if started_by_aiter:
                await self.stop()
                self.service_reset()

    async def __anext__(self) -> NoReturn:
        ...

    async def ack(self, event: EventT) -> bool:
        """Ack event."""
        last_stream_to_ack: bool = event.ack()
        message: Message = event.message
        tp: TP = event.message.tp
        offset: int = event.message.offset
        self._on_stream_event_out(tp, offset, self, event)
        if last_stream_to_ack:
            self._on_message_out(tp, offset, message)
        return last_stream_to_ack

    def __and__(self, other: JoinableT) -> StreamT:
        return self.combine(self, other)

    def __copy__(self) -> StreamT:
        return self.clone()

    def _repr_info(self) -> str:
        if self.combined:
            return reprlib.repr(self.combined)
        return reprlib.repr(self.channel)

    @property
    def label(self) -> str:
        """Return description of stream, used in graphs and logs."""
        return f'{type(self).__name__}: {self._repr_channel()}'

    def _repr_channel(self) -> str:
        return reprlib.repr(self.channel)

    @property
    def shortlabel(self) -> str:
        """Return short description of stream."""
        return f'Stream: {self._human_channel()}'

    def _human_channel(self) -> str:
        if self.combined:
            return '&'.join((s._human_channel() for s in self.combined))
        return f'{type(self.channel).__name__}: {self.channel}'
