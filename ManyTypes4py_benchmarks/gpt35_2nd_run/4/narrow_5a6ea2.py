from typing import List, Optional, Tuple, TypeVar

MessageRowT = TypeVar('MessageRowT', bound=Sequence[Any])

class LimitedMessages(Generic[MessageRowT]):
    rows: List[MessageRowT]

class FetchedMessages(LimitedMessages[Row]):
    rows: List[Row]
    found_anchor: bool
    found_newest: bool
    found_oldest: bool
    history_limited: bool
