from typing import Generator

def generate_filenames(tmp_path: str, num_files: int) -> Generator[str, None, None]:
    i = 0
    while num_files:
        for ext in ['.py', '.txt', '.tar.gz', '']:
            i += 1
            yield os.path.join(tmp_path, (str(i) + ext))

def setup(num_files: int) -> str:
    tmp_path = tempfile.mkdtemp()
    for fn in generate_filenames(tmp_path, num_files):
        with open(fn, 'wb') as f:
            f.write(b'benchmark')
    return tmp_path

def bench_pathlib(loops: int, tmp_path: str) -> float:
    base_path = pathlib.Path(tmp_path)
    path_objects = list(base_path.iterdir())
    for p in path_objects:
        p.stat()
    assert (len(path_objects) == NUM_FILES), len(path_objects)
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        for p in base_path.iterdir():
            p.stat()
        for p in base_path.glob('*.py'):
            p.stat()
        for p in base_path.iterdir():
            p.stat()
        for p in base_path.glob('*.py'):
            p.stat()
    return (pyperf.perf_counter() - t0)
