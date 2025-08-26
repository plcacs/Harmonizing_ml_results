from typing import List, Tuple, Dict, Union

def add_numbers(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello, {name}!"

def calculate_area(radius: float) -> float:
    pi: float = 3.14159
    return pi * radius * radius

def find_max(numbers: List[int]) -> int:
    return max(numbers)

def get_user_info(user_id: int) -> Dict[str, Union[str, int]]:
    return {"id": user_id, "name": "John Doe", "age": 30}

def process_data(data: List[Tuple[int, str]]) -> List[str]:
    return [f"ID: {item[0]}, Name: {item[1]}" for item in data]
