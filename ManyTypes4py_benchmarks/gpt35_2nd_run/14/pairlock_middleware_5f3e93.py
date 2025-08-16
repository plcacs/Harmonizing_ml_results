from datetime import datetime
from typing import List, Optional

class PairLocks:
    locks: List[PairLock]
    timeframe: str

    @staticmethod
    def reset_locks() -> None:
        ...

    @staticmethod
    def lock_pair(pair: str, until: datetime, reason: Optional[str] = None, *, now: Optional[datetime] = None, side: str = '*') -> PairLock:
        ...

    @staticmethod
    def get_pair_locks(pair: str, now: Optional[datetime] = None, side: Optional[str] = None) -> List[PairLock]:
        ...

    @staticmethod
    def get_pair_longest_lock(pair: str, now: Optional[datetime] = None, side: str = '*') -> Optional[PairLock]:
        ...

    @staticmethod
    def unlock_pair(pair: str, now: Optional[datetime] = None, side: str = '*') -> None:
        ...

    @staticmethod
    def unlock_reason(reason: str, now: Optional[datetime] = None) -> None:
        ...

    @staticmethod
    def is_global_lock(now: Optional[datetime] = None, side: str = '*') -> bool:
        ...

    @staticmethod
    def is_pair_locked(pair: str, now: Optional[datetime] = None, side: str = '*') -> bool:
        ...

    @staticmethod
    def get_all_locks() -> List[PairLock]:
        ...
