def greet(name: str) -> str:
    return f"Hello, {name}"


def add_numbers(a: int, b: int) -> int:
    return a + b


def process_data(data: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for i, value in enumerate(data):
        result[f"item_{i}"] = value
    return result


class Calculator:
    def __init__(self, initial_value: int = 0) -> None:
        self.value: int = initial_value

    def add(self, x: int) -> None:
        self.value += x

    def get_value(self) -> int:
        return self.value


def main() -> None:
    message: str = greet("Alice")
    print(message)
    
    total: int = add_numbers(5, 3)
    print(f"Total: {total}")
    
    numbers: list[int] = [1, 2, 3, 4, 5]
    processed: dict[str, int] = process_data(numbers)
    print(f"Processed data: {processed}")
    
    calc: Calculator = Calculator(10)
    calc.add(5)
    print(f"Calculator value: {calc.get_value()}")


if __name__ == "__main__":
    main()
