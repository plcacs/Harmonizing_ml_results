from typing import List, Tuple, Dict, Union

def add_numbers(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello, {name}!"

def find_max(numbers: List[int]) -> int:
    return max(numbers)

def process_data(data: Dict[str, Union[int, float]]) -> float:
    total = 0.0
    for value in data.values():
        total += value
    return total

def get_coordinates() -> Tuple[float, float]:
    return (40.7128, -74.0060)

def calculate_area(width: float, height: float) -> float:
    return width * height

def is_even(number: int) -> bool:
    return number % 2 == 0
