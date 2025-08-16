from typing import Any, Optional, TypeVar

CountT = TypeVar('CountT', bound=BaseCount)

def table_filtered_to_id(table: Any, key_id: int) -> Any:
    ...

def client_label_map(name: str) -> str:
    ...

def rewrite_client_arrays(value_arrays: dict[str, list[int]]) -> dict[str, list[int]]:
    ...

def get_time_series_by_subgroup(stat: CountStat, table: Any, key_id: int, end_times: list[datetime], subgroup_to_label: dict[Any, str], include_empty_subgroups: bool) -> dict[str, list[int]]:
    ...

def sort_by_totals(value_arrays: dict[str, list[int]]) -> list[str]:
    ...

def sort_client_labels(data: dict[str, dict[str, list[int]]]) -> list[str]:
    ...
