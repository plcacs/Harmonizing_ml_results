"""
Scan 500_untyped_files for files that use abstract classes or methods.

Detects:
  - Classes inheriting from ABC or ABCMeta
  - Methods decorated with @abstractmethod / @abstractclassmethod / @abstractstaticmethod
  - Imports from abc module

Saves results to abstract_files.json.
"""

import ast
import json
from pathlib import Path

UNTYPED_DIR = Path(__file__).resolve().parent.parent / "500_untyped_files"
OUTPUT = Path(__file__).resolve().parent / "abstract_files.json"


def uses_abstract(filepath):
    """Return True if the file uses abstract classes/methods."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "abc":
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "abc":
                    return True
    return False


def main():
    files = sorted(UNTYPED_DIR.glob("*.py"))
    abstract_files = [f.name for f in files if uses_abstract(f)]

    result = {
        "total_files": len(files),
        "abstract_count": len(abstract_files),
        "files": abstract_files,
    }

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Total files: {len(files)}")
    print(f"Files using abstract: {len(abstract_files)}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
