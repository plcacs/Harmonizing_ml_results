from typing import List, Dict, Optional, Tuple, Set, Any

def process_data(data: List[Dict[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in data:
        key: str = item.get('name', 'unknown')
        value: int = item.get('value', 0)
        result[key] = result.get(key, 0) + value
    return result

def filter_values(values: List[int], threshold: int = 10) -> List[int]:
    return [x for x in values if x > threshold]

class DataProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.cache: Dict[str, Any] = {}
    
    def process(self, items: List[str]) -> Tuple[List[str], int]:
        processed: List[str] = []
        count: int = 0
        
        for item in items:
            if item in self.cache:
                processed.append(self.cache[item])
            else:
                processed_item: str = item.upper()
                self.cache[item] = processed_item
                processed.append(processed_item)
            count += 1
            
        return processed, count
    
    def get_unique_items(self, items: List[str]) -> Set[str]:
        return set(items)
