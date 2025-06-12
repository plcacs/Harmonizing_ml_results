from typing import List, Dict, Tuple, Optional

def process_data(data: List[Dict[str, int]]) -> Tuple[Optional[int], Optional[float]]:
    total: int = 0
    count: int = 0
    
    for item in data:
        if 'value' in item:
            total += item['value']
            count += 1
    
    if count == 0:
        return (None, None)
    
    average: float = total / count
    return (total, average)

def main() -> None:
    sample_data: List[Dict[str, int]] = [
        {'value': 10},
        {'value': 20},
        {'value': 30},
        {'other': 40}
    ]
    
    result: Tuple[Optional[int], Optional[float]] = process_data(sample_data)
    print(result)

if __name__ == '__main__':
    main()
