from typing import Any, Dict, List, Set, Tuple, Union

class MyTable(Collection):
    datas: Dict[Any, Any]

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.datas = {}

    def _has_key(self, key: Any) -> bool:
        return key in self.datas

    def _get_key(self, key: Any) -> Any:
        return self.datas.get(key)

    def _set_key(self, key: Any, value: Any) -> None:
        self.datas[key] = value

    def _del_key(self, key: Any) -> None:
        self.datas.pop(key, None)

    def hopping(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def tumbling(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def using_window(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def as_ansitable(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def persisted_offset(self, tp: TP) -> Any:
        data = self._data
        return data.persisted_offset()

    async def need_active_standby_for(self, tp: TP) -> Any:
        data = self._data
        return await data.need_active_standby_for(tp)

    def reset_state(self) -> None:
        data = self._data
        data.reset_state()

    def _send_changelog(self, event: Event, key: Any, value: Any, key_serializer: str = 'json', value_serializer: str = 'json') -> None:
        self.changelog_topic.send_soon(key=key, value=value, partition=event.message.partition, key_serializer=key_serializer, value_serializer=value_serializer, callback=self._on_changelog_sent, eager_partitioning=True)

    def _on_changelog_sent(self, fut: asyncio.Future) -> None:
        data = self._data
        data.set_persisted_offset(fut.result().topic_partition, fut.result().offset)

    async def _del_old_keys(self) -> None:
        ...

    async def _clean_data(self, table: Any) -> None:
        ...

    def join(self, field1: Any, field2: Any) -> Any:
        ...

    def left_join(self, field1: Any, field2: Any) -> Any:
        ...

    def inner_join(self, field1: Any, field2: Any) -> Any:
        ...

    def outer_join(self, field1: Any, field2: Any) -> Any:
        ...

    def _verify_source_topic_partitions(self, topic: str) -> None:
        ...

    async def on_rebalance(self, assigned: Set[TP], revoked: Set[TP], newly_assigned: Set[TP]) -> None:
        ...

    async def on_changelog_event(self, event: Event) -> None:
        ...

    def apply_changelog_batch(self, batch: List[Any]) -> None:
        ...

    def partition_for_key(self, key: Any) -> Any:
        ...
