def greet(name: str) -> str:
    return f"Hello, {name}"


def add_numbers(a: int, b: int) -> int:
    return a + b


def process_data(data: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in data:
        result[item] = len(item)
    return result


class Calculator:
    def __init__(self, initial_value: int = 0) -> None:
        self.value: int = initial_value
    
    def add(self, x: int) -> None:
        self.value += x
    
    def subtract(self, x: int) -> None:
        self.value -= x
    
    def get_value(self) -> int:
        return self.value


def main() -> None:
    message: str = greet("Alice")
    print(message)
    
    total: int = add_numbers(5, 3)
    print(f"Total: {total}")
    
    words: list[str] = ["apple", "banana", "cherry"]
    processed: dict[str, int] = process_data(words)
    print(processed)
    
    calc: Calculator = Calculator(10)
    calc.add(5)
    calc.subtract(3)
    result: int = calc.get_value()
    print(f"Calculator result: {result}")


if __name__ == "__main__":
    main()
