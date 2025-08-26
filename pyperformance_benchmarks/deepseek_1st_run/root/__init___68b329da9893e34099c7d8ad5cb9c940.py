from typing import List, Dict, Union, Optional

def process_data(data: List[Dict[str, Union[str, int]]]) -> List[Dict[str, Union[str, int]]]:
    result: List[Dict[str, Union[str, int]]] = []
    for item in data:
        processed_item: Dict[str, Union[str, int]] = {}
        for key, value in item.items():
            if isinstance(value, str):
                processed_item[key] = value.upper()
            else:
                processed_item[key] = value * 2
        result.append(processed_item)
    return result

def get_max_value(items: List[Dict[str, int]], key: str) -> Optional[int]:
    if not items:
        return None
    values: List[int] = [item[key] for item in items if key in item]
    return max(values) if values else None

def main() -> None:
    sample_data: List[Dict[str, Union[str, int]]] = [
        {"name": "alice", "age": 25, "city": "new york"},
        {"name": "bob", "age": 30, "city": "london"},
        {"name": "charlie", "age": 35, "city": "paris"}
    ]
    
    processed: List[Dict[str, Union[str, int]]] = process_data(sample_data)
    print("Processed data:", processed)
    
    numeric_data: List[Dict[str, int]] = [{"age": 25}, {"age": 30}, {"age": 35}]
    max_age: Optional[int] = get_max_value(numeric_data, "age")
    print("Maximum age:", max_age)

if __name__ == "__main__":
    main()
