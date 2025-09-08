from typing import List, Dict, Optional, Tuple, Set, Any

def process_data(data: List[Dict[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in data:
        key: str = item.get('name', 'unknown')
        value: int = item.get('count', 0)
        result[key] = result.get(key, 0) + value
    return result

def filter_values(values: List[int], threshold: int = 10) -> List[int]:
    return [x for x in values if x > threshold]

class DataProcessor:
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config or {}
        self.processed_items: int = 0
        
    def process(self, items: List[Any]) -> Tuple[int, Set[str]]:
        unique_ids: Set[str] = set()
        for item in items:
            if isinstance(item, dict) and 'id' in item:
                unique_ids.add(item['id'])
        self.processed_items += len(items)
        return self.processed_items, unique_ids
