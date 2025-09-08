from typing import List, Dict, Optional, Tuple, Set, Any

def process_data(data: List[Dict[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in data:
        key: str = item.get('name', 'unknown')
        value: int = item.get('count', 0)
        result[key] = result.get(key, 0) + value
    return result

def find_max_value(data: Dict[str, int]) -> Tuple[str, int]:
    if not data:
        return ('', 0)
    max_key: str = max(data, key=data.get)  # type: ignore
    return (max_key, data[max_key])

def filter_values(data: Dict[str, int], threshold: int) -> List[str]:
    return [key for key, value in data.items() if value > threshold]

def merge_results(results1: Dict[str, int], results2: Dict[str, int]) -> Dict[str, int]:
    merged: Dict[str, int] = results1.copy()
    for key, value in results2.items():
        merged[key] = merged.get(key, 0) + value
    return merged

def get_unique_keys(data_list: List[Dict[str, int]]) -> Set[str]:
    unique_keys: Set[str] = set()
    for data in data_list:
        unique_keys.update(data.keys())
    return unique_keys

def find_item_by_id(items: List[Dict[str, Any]], item_id: int) -> Optional[Dict[str, Any]]:
    for item in items:
        if item.get('id') == item_id:
            return item
    return None
