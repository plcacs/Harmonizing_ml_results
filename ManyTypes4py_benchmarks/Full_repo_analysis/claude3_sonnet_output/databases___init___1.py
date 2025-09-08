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
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.cache: Dict[str, Any] = {}
    
    def process(self, items: List[Any]) -> Tuple[List[Any], int]:
        processed: List[Any] = []
        count: int = 0
        
        for item in items:
            if self._should_process(item):
                processed.append(self._transform(item))
                count += 1
                
        return processed, count
    
    def _should_process(self, item: Any) -> bool:
        return bool(item)
    
    def _transform(self, item: Any) -> Any:
        return item

def find_duplicates(items: List[Any]) -> Set[Any]:
    seen: Set[Any] = set()
    duplicates: Set[Any] = set()
    
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
            
    return duplicates
