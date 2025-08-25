import re
import pyperf
from typing import List, Tuple, Optional, Any

def capture_regexes() -> List[Tuple[str, int]]:
    regexes: List[Tuple[str, int]] = []
    real_compile = re.compile
    real_search = re.search
    real_sub = re.sub

    def capture_compile(regex: str, flags: int = 0) -> re.Pattern:
        regexes.append((regex, flags))
        return real_compile(regex, flags)

    def capture_search(regex: str, target: str, flags: int = 0) -> Optional[re.Match]:
        regexes.append((regex, flags))
        return real_search(regex, target, flags)

    def capture_sub(regex: str, *args: Any) -> Any:
        regexes.append((regex, 0))
        return real_sub(regex, *args)

    re.compile = capture_compile
    re.search = capture_search
    re.sub = capture_sub
    try:
        import bm_regex_effbot
        bm_regex_effbot.bench_regex_effbot(1)
        import bm_regex_v8
        bm_regex_v8.bench_regex_v8(1)
    finally:
        re.compile = real_compile
        re.search = real_search
        re.sub = real_sub
    return regexes

def bench_regex_compile(loops: int, regexes: List[Tuple[str, int]]) -> float:
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        for (regex, flags) in regexes:
            re.purge()
            re.compile(regex, flags)
    return pyperf.perf_counter() - t0

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Test regex compilation performance'
    regexes: List[Tuple[str, int]] = capture_regexes()
    runner.bench_time_func('regex_compile', bench_regex_compile, regexes)
