'The pgen2 package.'

from typing import List, Tuple, Optional, Any

def parse_file(filename: str) -> List[str]:
    with open(filename, 'r') as file:
        return file.readlines()

def tokenize_line(line: str) -> List[str]:
    return line.split()

def process_tokens(tokens: List[str]) -> List[Tuple[str, int]]:
    return [(token, len(token)) for token in tokens]

def main(filename: str) -> None:
    lines = parse_file(filename)
    for line in lines:
        tokens = tokenize_line(line)
        processed_tokens = process_tokens(tokens)
        for token, length in processed_tokens:
            print(f"Token: {token}, Length: {length}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as an argument.")
