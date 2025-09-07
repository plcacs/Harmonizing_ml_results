def greet(name: str) -> str:
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    return a + b


def process_data(data: list[int]) -> dict[str, int]:
    return {
        "sum": sum(data),
        "length": len(data)
    }


class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age
    
    def get_info(self) -> str:
        return f"{self.name} is {self.age} years old"


def main() -> None:
    message: str = greet("Alice")
    print(message)
    
    result: int = add(5, 3)
    print(f"5 + 3 = {result}")
    
    numbers: list[int] = [1, 2, 3, 4, 5]
    processed: dict[str, int] = process_data(numbers)
    print(processed)
    
    person: Person = Person("Bob", 30)
    info: str = person.get_info()
    print(info)


if __name__ == "__main__":
    main()
