from typing import List, Dict, Tuple, Optional, Union, Any

def process_data(data: List[Dict[str, Any]]) -> Dict[str, Union[int, float]]:
    result = {}
    for item in data:
        key = item.get('id', '')
        value = item.get('value', 0)
        if isinstance(value, (int, float)):
            result[key] = value
    return result

def calculate_statistics(numbers: List[float]) -> Tuple[float, float, float]:
    if not numbers:
        return 0.0, 0.0, 0.0
    
    total = sum(numbers)
    average = total / len(numbers)
    variance = sum((x - average) ** 2 for x in numbers) / len(numbers)
    std_dev = variance ** 0.5
    
    return average, variance, std_dev

def find_element(data: List[Any], target: Any) -> Optional[int]:
    for i, item in enumerate(data):
        if item == target:
            return i
    return None

class DataProcessor:
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self.data: List[Any] = []
    
    def add_item(self, item: Any) -> None:
        self.data.append(item)
    
    def get_items(self) -> List[Any]:
        return self.data
    
    def clear(self) -> None:
        self.data = []
