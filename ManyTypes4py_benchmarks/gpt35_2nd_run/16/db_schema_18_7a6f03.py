from typing import Any, Dict, List, Optional

def process_timestamp(ts: Optional[DateTime]) -> Optional[DateTime]:
    ...

def process_timestamp_to_utc_isoformat(ts: Optional[DateTime]) -> Optional[str]:
    ...

class LazyState(State):
    def __init__(self, row: States) -> None:
        ...

    @property
    def attributes(self) -> Dict[str, Any]:
        ...

    @attributes.setter
    def attributes(self, value: Dict[str, Any]) -> None:
        ...

    @property
    def context(self) -> Context:
        ...

    @context.setter
    def context(self, value: Context) -> None:
        ...

    @property
    def last_changed(self) -> DateTime:
        ...

    @last_changed.setter
    def last_changed(self, value: DateTime) -> None:
        ...

    @property
    def last_updated(self) -> DateTime:
        ...

    @last_updated.setter
    def last_updated(self, value: DateTime) -> None:
        ...

    def as_dict(self) -> Dict[str, Any]:
        ...

    def __eq__(self, other: Any) -> bool:
        ...
