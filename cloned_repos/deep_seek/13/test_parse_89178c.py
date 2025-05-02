import pytest
from hypothesis import given
from hypothesis import strategies as st
from isort import parse
from isort.settings import Config
from typing import Tuple, List, Any, Optional, Union

TEST_CONTENTS: str = '\nimport xyz\nimport abc\nimport (\\ # one\n    one as \\ # two\n    three)\nimport \\\n    zebra as \\ # one\n    not_bacon\nfrom x import (\\ # one\n    one as \\ # two\n    three)\n\n\ndef function():\n    pass\n'

def test_file_contents() -> None:
    in_lines: List[str]
    out_lines: List[str]
    import_index: int
    change_count: int
    original_line_count: int
    in_lines, out_lines, import_index, _, _, _, _, _, change_count, original_line_count, _, _, _, _ = parse.file_contents(TEST_CONTENTS, config=Config(default_section=''))
    assert '\n'.join(in_lines) == TEST_CONTENTS
    assert 'import' not in '\n'.join(out_lines)
    assert import_index == 1
    assert change_count == -11
    assert original_line_count == len(in_lines)

@given(contents=st.text())
def test_fuzz__infer_line_separator(contents: str) -> None:
    parse._infer_line_separator(contents=contents)

@given(import_string=st.text())
def test_fuzz__strip_syntax(import_string: str) -> None:
    parse.strip_syntax(import_string=import_string)

@given(line=st.text(), config=st.builds(Config))
def test_fuzz_import_type(line: str, config: Config) -> None:
    parse.import_type(line=line, config=config)

@given(line=st.text(), in_quote=st.text(), index=st.integers(), section_comments=st.lists(st.text()), needs_import=st.booleans())
def test_fuzz_skip_line(line: str, in_quote: str, index: int, section_comments: List[str], needs_import: bool) -> None:
    parse.skip_line(line=line, in_quote=in_quote, index=index, section_comments=section_comments, needs_import=needs_import)

@pytest.mark.parametrize('raw_line, expected', (('from . cimport a', 'from . cimport a'), ('from.cimport a', 'from . cimport a'), ('from..cimport a', 'from .. cimport a'), ('from . import a', 'from . import a'), ('from.import a', 'from . import a'), ('from..import a', 'from .. import a'), ('import *', 'import *'), ('import*', 'import *'), ('from . import a', 'from . import a'), ('from .import a', 'from . import a'), ('from ..import a', 'from .. import a'), ('from . cimport a', 'from . cimport a'), ('from .cimport a', 'from . cimport a'), ('from ..cimport a', 'from .. cimport a'), ('from\t.\timport a', 'from . import a')))
def test_normalize_line(raw_line: str, expected: str) -> None:
    line: str
    returned_raw_line: str
    line, returned_raw_line = parse.normalize_line(raw_line)
    assert line == expected
    assert returned_raw_line == raw_line
