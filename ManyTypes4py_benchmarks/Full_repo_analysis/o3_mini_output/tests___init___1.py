def greet(name: str) -> str:
    return f"Hello, {name}!"

def main() -> None:
    user_name: str = "World"
    greeting: str = greet(user_name)
    print(greeting)

if __name__ == "__main__":
    main()