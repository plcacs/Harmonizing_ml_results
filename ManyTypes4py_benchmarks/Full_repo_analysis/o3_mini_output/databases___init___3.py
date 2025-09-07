def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def main() -> None:
    user_name: str = "Alice"
    greeting: str = greet(user_name)
    sum_result: int = add(5, 7)
    print(greeting)
    print(f"Sum: {sum_result}")

if __name__ == "__main__":
    main()