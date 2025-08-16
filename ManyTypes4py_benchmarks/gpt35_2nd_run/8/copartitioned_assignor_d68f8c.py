from itertools import cycle
from math import ceil
from typing import Iterable, Iterator, MutableMapping, Optional, Sequence, Set, Counter
from mode.utils.typing import Counter
from .client_assignment import CopartitionedAssignment

__all__: List[str] = ['CopartitionedAssignor']

class CopartitionedAssignor:
    def __init__(self, topics: Set[str], cluster_asgn: MutableMapping[str, CopartitionedAssignment], num_partitions: int, replicas: int, capacity: Optional[int] = None) -> None:
        self._num_clients: int = len(cluster_asgn)
        self.num_partitions: int = num_partitions
        self.replicas: int = min(replicas, self._num_clients - 1)
        self.capacity: int = int(ceil(float(self.num_partitions) / self._num_clients)) if capacity is None else capacity
        self.topics: Set[str] = set(topics)
        self._client_assignments: MutableMapping[str, CopartitionedAssignment] = cluster_asgn

    def get_assignment(self) -> MutableMapping[str, CopartitionedAssignment]:
        ...

    def _all_assigned(self, active: bool) -> bool:
        ...

    def _assign(self, active: bool) -> None:
        ...

    def _assigned_partition_counts(self, active: bool) -> Counter[int]:
        ...

    def _get_client_limit(self, active: bool) -> int:
        ...

    def _total_assigns_per_partition(self, active: bool) -> int:
        ...

    def _unassign_overassigned(self, active: bool) -> None:
        ...

    def _get_unassigned(self, active: bool) -> List[int]:
        ...

    def _can_assign(self, assignment: CopartitionedAssignment, partition: int, active: bool) -> bool:
        ...

    def _client_exhausted(self, assignemnt: CopartitionedAssignment, active: bool, client_limit: Optional[int] = None) -> bool:
        ...

    def _find_promotable_standby(self, partition: int, candidates: Iterator[CopartitionedAssignment]) -> Optional[CopartitionedAssignment]:
        ...

    def _find_round_robin_assignable(self, partition: int, candidates: Iterator[CopartitionedAssignment], active: bool) -> Optional[CopartitionedAssignment]:
        ...

    def _assign_round_robin(self, unassigned: List[int], active: bool) -> None:
        ...
