from typing import List, Dict, Tuple, Optional, Union, Any

def process_data(data: List[Dict[str, Any]]) -> Dict[str, Union[int, float]]:
    result = {}
    for item in data:
        key = item.get('id', '')
        value = item.get('value', 0)
        if isinstance(value, (int, float)):
            result[key] = value
    return result

def calculate_statistics(numbers: List[Union[int, float]]) -> Tuple[float, float, float]:
    if not numbers:
        return (0.0, 0.0, 0.0)
    
    total = sum(numbers)
    average = total / len(numbers)
    variance = sum((x - average) ** 2 for x in numbers) / len(numbers)
    std_dev = variance ** 0.5
    
    return (average, variance, std_dev)

def find_max_value(data: Dict[str, Union[int, float]]) -> Optional[Tuple[str, Union[int, float]]]:
    if not data:
        return None
    
    max_key = max(data, key=data.get)  # type: ignore
    return (max_key, data[max_key])

def main() -> None:
    sample_data: List[Dict[str, Any]] = [
        {'id': 'A', 'value': 10},
        {'id': 'B', 'value': 20},
        {'id': 'C', 'value': 15},
        {'id': 'D', 'value': 'invalid'}
    ]
    
    processed = process_data(sample_data)
    print(f"Processed data: {processed}")
    
    values: List[Union[int, float]] = list(processed.values())
    stats = calculate_statistics(values)
    print(f"Statistics (avg, var, std): {stats}")
    
    max_item = find_max_value(processed)
    if max_item:
        print(f"Max value: {max_item[0]} = {max_item[1]}")
