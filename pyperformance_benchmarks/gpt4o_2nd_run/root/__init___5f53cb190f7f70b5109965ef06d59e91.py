'The pgen2 package.'

from typing import List, Tuple, Optional, Union, Dict, Any

def parse_file(filename: str) -> List[str]:
    with open(filename, 'r') as file:
        return file.readlines()

def tokenize_line(line: str) -> List[str]:
    return line.split()

def process_tokens(tokens: List[str]) -> Dict[str, int]:
    token_count: Dict[str, int] = {}
    for token in tokens:
        if token in token_count:
            token_count[token] += 1
        else:
            token_count[token] = 1
    return token_count

def main(filename: str) -> None:
    lines: List[str] = parse_file(filename)
    for line in lines:
        tokens: List[str] = tokenize_line(line)
        token_count: Dict[str, int] = process_tokens(tokens)
        print(token_count)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as an argument.")
