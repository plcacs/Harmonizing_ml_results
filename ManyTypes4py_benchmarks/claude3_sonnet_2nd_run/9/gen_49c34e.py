import io
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, cast
import multidict
ROOT = pathlib.Path.cwd()
while ROOT.parent != ROOT and (not (ROOT / 'pyproject.toml').exists()):
    ROOT = ROOT.parent

def calc_headers(root: pathlib.Path) -> List[multidict.istr]:
    hdrs_file = root / 'aiohttp/hdrs.py'
    code = compile(hdrs_file.read_text(), str(hdrs_file), 'exec')
    globs: Dict[str, Any] = {}
    exec(code, globs)
    headers = [val for val in globs.values() if isinstance(val, multidict.istr)]
    return sorted(headers)
headers = calc_headers(ROOT)

def factory() -> Dict[Any, Any]:
    return defaultdict(factory)
TERMINAL = object()

def build(headers: List[multidict.istr]) -> Dict[Any, Any]:
    dct = defaultdict(factory)
    for hdr in headers:
        d = dct
        for ch in hdr:
            d = d[ch]
        d[TERMINAL] = hdr
    return dct
dct = build(headers)
HEADER: str = '/*  The file is autogenerated from aiohttp/hdrs.py\nRun ./tools/gen.py to update it after the origin changing. */\n\n#include "_find_header.h"\n\n#define NEXT_CHAR() \\\n{ \\\n    count++; \\\n    if (count == size) { \\\n        /* end of search */ \\\n        return -1; \\\n    } \\\n    pchar++; \\\n    ch = *pchar; \\\n    last = (count == size -1); \\\n} while(0);\n\nint\nfind_header(const char *str, int size)\n{\n    char *pchar = str;\n    int last;\n    char ch;\n    int count = -1;\n    pchar--;\n'
BLOCK: str = '\n{label}\n    NEXT_CHAR();\n    switch (ch) {{\n{cases}\n        default:\n            return -1;\n    }}\n'
CASE: str = "        case '{char}':\n            if (last) {{\n                return {index};\n            }}\n            goto {next};"
FOOTER: str = '\n{missing}\nmissing:\n    /* nothing found */\n    return -1;\n}}\n'

def gen_prefix(prefix: str, k: str) -> str:
    if k == '-':
        return prefix + '_'
    else:
        return prefix + k.upper()

def gen_block(dct: Dict[Any, Any], prefix: str, used_blocks: Set[str], missing: Set[str], out: io.StringIO) -> None:
    cases: Dict[str, str] = {}
    for k, v in dct.items():
        if k is TERMINAL:
            continue
        next_prefix = gen_prefix(prefix, k)
        term = v.get(TERMINAL)
        if term is not None:
            index = headers.index(cast(multidict.istr, term))
        else:
            index = -1
        hi = k.upper()
        lo = k.lower()
        case = CASE.format(char=hi, index=index, next=next_prefix)
        cases[hi] = case
        if lo != hi:
            case = CASE.format(char=lo, index=index, next=next_prefix)
            cases[lo] = case
    label = prefix + ':' if prefix else ''
    if cases:
        block = BLOCK.format(label=label, cases='\n'.join(cases.values()))
        out.write(block)
    else:
        missing.add(label)
    for k, v in dct.items():
        if not isinstance(v, defaultdict):
            continue
        block_name = gen_prefix(prefix, k)
        if block_name in used_blocks:
            continue
        used_blocks.add(block_name)
        gen_block(v, block_name, used_blocks, missing, out)

def gen(dct: Dict[Any, Any]) -> io.StringIO:
    out = io.StringIO()
    out.write(HEADER)
    missing: Set[str] = set()
    gen_block(dct, '', set(), missing, out)
    missing_labels = '\n'.join(sorted(missing))
    out.write(FOOTER.format(missing=missing_labels))
    return out

def gen_headers(headers: List[multidict.istr]) -> io.StringIO:
    out = io.StringIO()
    out.write('# The file is autogenerated from aiohttp/hdrs.py\n')
    out.write('# Run ./tools/gen.py to update it after the origin changing.')
    out.write('\n\n')
    out.write('from . import hdrs\n')
    out.write('cdef tuple headers = (\n')
    for hdr in headers:
        out.write('    hdrs.{},\n'.format(hdr.upper().replace('-', '_')))
    out.write(')\n')
    return out
folder = ROOT / 'aiohttp'
with (folder / '_find_header.c').open('w') as f:
    f.write(gen(dct).getvalue())
with (folder / '_headers.pxi').open('w') as f:
    f.write(gen_headers(headers).getvalue())
