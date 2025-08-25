def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers")
    result: int = 1
    for i in range(2, n + 1):
        result *= i
    return result

def main() -> None:
    number: int = 5
    result: int = factorial(number)
    print(f"The factorial of {number} is {result}")

if __name__ == "__main__":
    main()