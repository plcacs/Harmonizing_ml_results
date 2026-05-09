from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import pytest
from faust.exceptions import ImproperlyConfigured
from faust.types import TP
from yarl import URL

TP1 = TP('foo', 0)
TP2 = TP('foo', 1)
TP3 = TP('bar', 2)
TP4 = TP('baz', 3)

class MockIterator(Mock):
    @classmethod
    def from_values(cls, values: List[Any]) -> 'MockIterator':
        ...

class test_RocksDBOptions:
    def test_init(self, arg: str) -> None:
        ...

    def test_defaults(self) -> None:
        ...

    def test_open(self) -> None:
        ...

class test_Store:
    @pytest.fixture()
    def table(self) -> Mock:
        ...

    @pytest.yield_fixture()
    def rocks(self) -> Mock:
        ...

    @pytest.yield_fixture()
    def no_rocks(self) -> None:
        ...

    @pytest.fixture()
    def store(self, app: Any, rocks: Mock, table: Mock) -> 'Store':
        ...

    @pytest.fixture()
    def db_for_partition(self) -> Mock:
        ...

    def test_default_key_index_size(self, store: 'Store') -> None:
        ...

    def test_set_key_index_size(self, app: Any, rocks: Mock, table: Mock) -> None:
        ...

    def test_no_rocksdb(self, app: Any, table: Mock) -> None:
        ...

    def test_url_without_path_adds_table_name(self, store: 'Store') -> None:
        ...

    def test_url_having_path(self, app: Any, rocks: Mock, table: Mock) -> None:
        ...

    def test_init(self, store: 'Store', app: Any) -> None:
        ...

    def test_persisted_offset(self, store: 'Store', db_for_partition: Mock) -> Optional[int]:
        ...

    def test_set_persisted_offset(self, store: 'Store', db_for_partition: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_need_active_standby_for(self, store: 'Store', db_for_partition: Mock) -> bool:
        ...

    @pytest.mark.asyncio
    async def test_need_active_standby_for__raises(self, store: 'Store', db_for_partition: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_need_active_standby_for__active(self, store: 'Store') -> bool:
        ...

    def test_apply_changelog_batch(self, store: 'Store', rocks: Mock, db_for_partition: Mock) -> None:
        ...

    @pytest.yield_fixture()
    def current_event(self) -> Mock:
        ...

    def test__set(self, store: 'Store', db_for_partition: Mock, current_event: Mock) -> None:
        ...

    def test_db_for_partition(self, store: 'Store') -> Mock:
        ...

    def test_open_for_partition(self, store: 'Store') -> Mock:
        ...

    def test__get__missing(self, store: 'Store') -> None:
        ...

    def test__get(self, store: 'Store') -> Optional[bytes]:
        ...

    def test__get__dbvalue_is_None(self, store: 'Store') -> Optional[bytes]:
        ...

    def test_get_bucket_for_key__is_in_index(self, store: 'Store') -> Optional[Tuple[Mock, bytes]]:
        ...

    def test_get_bucket_for_key__no_dbs(self, store: 'Store') -> None:
        ...

    def test_get_bucket_for_key__not_in_index(self, store: 'Store') -> Tuple[Mock, bytes]:
        ...

    def test__del(self, store: 'Store') -> None:
        ...

    @pytest.mark.asyncio
    async def test_on_rebalance(self, store: 'Store', table: Mock) -> None:
        ...

    def test_revoke_partitions(self, store: 'Store', table: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_assign_partitions(self, store: 'Store', app: Any, table: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_assign_partitions__empty_assignment(self, store: 'Store', app: Any, table: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_open_db_for_partition(self, store: 'Store', db_for_partition: Mock) -> Mock:
        ...

    @pytest.mark.asyncio
    async def test_open_db_for_partition_max_retries(self, store: 'Store', db_for_partition: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_open_db_for_partition__raises_unexpected_error(self, store: 'Store', db_for_partition: Mock) -> None:
        ...

    @pytest.mark.asyncio
    async def test_open_db_for_partition_retries_recovers(self, store: 'Store', db_for_partition: Mock) -> None:
        ...

    def test__contains(self, store: 'Store') -> bool:
        ...

    def test__dbs_for_key(self, store: 'Store') -> List[Mock]:
        ...

    def test__dbs_for_actives(self, store: 'Store', table: Mock) -> List[Mock]:
        ...

    def test__size(self, store: 'Store') -> int:
        ...

    def test__iterkeys(self, store: 'Store') -> List[bytes]:
        ...

    def test__itervalues(self, store: 'Store') -> List[bytes]:
        ...

    def test__iteritems(self, store: 'Store') -> List[Tuple[bytes, bytes]]:
        ...

    def test_clear(self, store: 'Store') -> None:
        ...

    def test_reset_state(self, store: 'Store') -> None:
        ...