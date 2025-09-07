def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Negative input not allowed")
    if n in (0, 1):
        return 1
    return n * factorial(n - 1)


def sum_of_list(numbers: list[int]) -> int:
    total: int = 0
    for num in numbers:
        total += num
    return total


def greet(name: str) -> str:
    return f"Hello, {name}!"


def main() -> None:
    numbers: list[int] = [1, 2, 3, 4, 5]
    result: int = sum_of_list(numbers)
    print("Sum:", result)
    print(greet("Alice"))
    print("Factorial of 5:", factorial(5))


if __name__ == '__main__':
    main()