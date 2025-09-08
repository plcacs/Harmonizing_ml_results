from typing import List, Dict, Tuple, Optional, Set, Any

def process_data(data: List[Dict[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in data:
        key: str = item.get('name', 'unknown')
        value: int = item.get('count', 0)
        result[key] = result.get(key, 0) + value
    return result

def filter_values(values: List[int], threshold: int = 10) -> List[int]:
    return [x for x in values if x > threshold]

def get_stats(numbers: List[int]) -> Tuple[float, int, int]:
    if not numbers:
        return 0.0, 0, 0
    avg: float = sum(numbers) / len(numbers)
    min_val: int = min(numbers)
    max_val: int = max(numbers)
    return avg, min_val, max_val

def find_duplicates(items: List[str]) -> Set[str]:
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates

def safe_divide(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b
