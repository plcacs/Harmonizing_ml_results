from typing import Generator

def generate_filenames(tmp_path: str, num_files: int) -> Generator[str, None, None]:
    ...

def setup(num_files: int) -> str:
    ...

def bench_pathlib(loops: int, tmp_path: str) -> float:
    ...
