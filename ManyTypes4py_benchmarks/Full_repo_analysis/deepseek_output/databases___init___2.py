def greet(name: str) -> str:
    return f"Hello, {name}!"

def add_numbers(a: int, b: int) -> int:
    return a + b

def process_data(data: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for i, value in enumerate(data):
        result[f"item_{i}"] = value
    return result

def calculate_average(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age
    
    def get_info(self) -> str:
        return f"{self.name} is {self.age} years old"

def main() -> None:
    greeting: str = greet("Alice")
    print(greeting)
    
    total: int = add_numbers(5, 3)
    print(f"Sum: {total}")
    
    numbers: list[int] = [1, 2, 3, 4, 5]
    processed: dict[str, int] = process_data(numbers)
    print(processed)
    
    avg: float = calculate_average([1.5, 2.5, 3.5])
    print(f"Average: {avg}")
    
    person: Person = Person("Bob", 30)
    info: str = person.get_info()
    print(info)

if __name__ == "__main__":
    main()
