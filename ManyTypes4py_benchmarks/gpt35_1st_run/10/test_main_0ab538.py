import json
import os
import pathlib
import shutil
import subprocess
from datetime import datetime
import pytest
from hypothesis import given
from hypothesis import strategies as st
from isort import main
from isort._version import __version__
from isort.exceptions import InvalidSettingsPath
from isort.settings import DEFAULT_CONFIG, Config
from .utils import as_stream
from io import BytesIO, TextIOWrapper
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass
else:
    from isort.wrap_modes import WrapModes

@given(file_name: str = st.text(), config: Config = st.builds(Config), check: bool = st.booleans(), ask_to_apply: bool = st.booleans(), write_to_stdout: bool = st.booleans())
def test_fuzz_sort_imports(file_name, config, check, ask_to_apply, write_to_stdout):
    main.sort_imports(file_name=file_name, config=config, check=check, ask_to_apply=ask_to_apply, write_to_stdout=write_to_stdout)

def test_sort_imports(tmpdir):
    tmp_file = tmpdir.join('file.py')
    tmp_file.write('import os, sys\n')
    assert main.sort_imports(str(tmp_file), DEFAULT_CONFIG, check=True).incorrectly_sorted
    main.sort_imports(str(tmp_file), DEFAULT_CONFIG)
    assert not main.sort_imports(str(tmp_file), DEFAULT_CONFIG, check=True).incorrectly_sorted
    skip_config = Config(skip=['file.py'])
    assert main.sort_imports(str(tmp_file), config=skip_config, check=True, disregard_skip=False).skipped
    assert main.sort_imports(str(tmp_file), config=skip_config, disregard_skip=False).skipped

def test_sort_imports_error_handling(tmpdir, mocker, capsys):
    tmp_file = tmpdir.join('file.py')
    tmp_file.write('import os, sys\n')
    mocker.patch('isort.core.process').side_effect = IndexError('Example unhandled exception')
    with pytest.raises(IndexError):
        main.sort_imports(str(tmp_file), DEFAULT_CONFIG, check=True).incorrectly_sorted
    out, error = capsys.readouterr()
    assert 'Unrecoverable exception thrown when parsing' in error

def test_parse_args():
    assert main.parse_args([]) == {}
    assert main.parse_args(['--multi-line', '1']) == {'multi_line_output': WrapModes.VERTICAL}
    assert main.parse_args(['--multi-line', 'GRID']) == {'multi_line_output': WrapModes.GRID}
    assert main.parse_args(['--dont-order-by-type']) == {'order_by_type': False}
    assert main.parse_args(['--dt']) == {'order_by_type': False}
    assert main.parse_args(['--only-sections']) == {'only_sections': True}
    assert main.parse_args(['--os']) == {'only_sections': True}
    assert main.parse_args(['--om']) == {'only_modified': True}
    assert main.parse_args(['--only-modified']) == {'only_modified': True}
    assert main.parse_args(['--csi']) == {'combine_straight_imports': True}
    assert main.parse_args(['--combine-straight-imports']) == {'combine_straight_imports': True}
    assert main.parse_args(['--dont-follow-links']) == {'follow_links': False}
    assert main.parse_args(['--overwrite-in-place']) == {'overwrite_in_place': True}
    assert main.parse_args(['--from-first']) == {'from_first': True}
    assert main.parse_args(['--resolve-all-configs']) == {'resolve_all_configs': True}

def test_ascii_art(capsys):
    main.main(['--version'])
    out, error = capsys.readouterr()
    assert out == f"\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION {__version__}\n\n"
    assert error == ''

def test_preconvert():
    assert main._preconvert(frozenset([1, 1, 2])) == [1, 2]
    assert main._preconvert(WrapModes.GRID) == 'GRID'
    assert main._preconvert(main._preconvert) == '_preconvert'
    with pytest.raises(TypeError):
        main._preconvert(datetime.now())

def test_show_files(capsys, tmpdir):
    tmpdir.join('a.py').write('import a')
    tmpdir.join('b.py').write('import b')
    main.main([str(tmpdir), '--show-files'])
    out, error = capsys.readouterr()
    assert 'a.py' in out
    assert 'b.py' in out
    assert not error
    with pytest.raises(SystemExit):
        main.main(['-', '--show-files'])
    with pytest.raises(SystemExit):
        main.main([str(tmpdir), '--show-files', '--show-config'])

def test_missing_default_section(tmpdir):
    config_file = tmpdir.join('.isort.cfg')
    config_file.write('\n[settings]\nsections=MADEUP\n')
    python_file = tmpdir.join('file.py')
    python_file.write('import os')
    with pytest.raises(SystemExit):
        main.main([str(python_file)])

def test_ran_against_root():
    with pytest.raises(SystemExit):
        main.main(['/'])

def test_main(capsys, tmpdir):
    base_args = ['-sp', str(tmpdir), '--virtual-env', str(tmpdir), '--src-path', str(tmpdir)]
    tmpdir.mkdir('.git')
    main.main([])
    out, error = capsys.readouterr()
    assert main.QUICK_GUIDE in out
    assert not error
    with pytest.raises(SystemExit):
        main.main(base_args)
    out, error = capsys.readouterr()
    assert main.QUICK_GUIDE in out
    main.main(base_args + ['--show-config'])
    out, error = capsys.readouterr()
    returned_config = json.loads(out)
    assert returned_config
    assert returned_config['virtual_env'] == str(tmpdir)
    main.main(base_args[2:] + ['--show-config'])
    out, error = capsys.readouterr()
    assert json.loads(out)['virtual_env'] == str(tmpdir)
    with pytest.raises(InvalidSettingsPath):
        main.main(base_args[2:] + ['--show-config'] + ['--settings-path', '/random-root-folder-that-cant-exist-right?'])
    config_file = tmpdir.join('.isort.cfg')
    config_file.write('\n[settings]\nprofile=hug\nverbose=true\n')
    config_args = ['--settings-path', str(config_file)]
    main.main(config_args + ['--virtual-env', '/random-root-folder-that-cant-exist-right?'] + ['--show-config'])
    out, error = capsys.readouterr()
    assert json.loads(out)['profile'] == 'hug'
    input_content = TextIOWrapper(BytesIO(b'\nimport b\nimport a\n'))
    main.main(config_args + ['-'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == f'\nelse-type place_module for b returned {DEFAULT_CONFIG.default_section}\nelse-type place_module for a returned {DEFAULT_CONFIG.default_section}\nimport a\nimport b\n'
    input_content = TextIOWrapper(BytesIO(b'\nimport b\nimport a\n'))
    main.main(config_args + ['-', '--diff'], stdin=input_content)
    out, error = capsys.readouterr()
    assert not error
    assert '+' in out
    assert '-' in out
    assert 'import a' in out
    assert 'import b' in out
    input_content_check = TextIOWrapper(BytesIO(b'\nimport b\nimport a\n'))
    with pytest.raises(SystemExit):
        main.main(config_args + ['-', '--check-only'], stdin=input_content_check)
    out, error = capsys.readouterr()
    assert error == 'ERROR:  Imports are incorrectly sorted and/or formatted.\n'
    python_file = tmpdir.join('has_imports.py')
    python_file.write('\nimport b\nimport a\n')
    main.main([str(python_file), '--filter-files', '--verbose'])
    assert python_file.read().lstrip() == 'import a\nimport b\n'
    should_skip = tmpdir.join('should_skip.py')
    should_skip.write('import nothing')
    main.main([str(python_file), str(should_skip), '--filter-files', '--verbose', '--skip', str(should_skip)])
    python_file.write('\nimport b\nimport a\n')
    with pytest.raises(SystemExit):
        main.main([str(python_file), str(should_skip), '--filter-files', '--verbose', '--check-only', '--skip', str(should_skip)])
    with pytest.raises(SystemExit):
        main.main([str(tmpdir), '--filter-files', '--verbose', '--check-only', '--skip', str(should_skip)])
    nested_file = tmpdir.mkdir('nested_dir').join('skip.py')
    nested_file.write('import b;import a')
    python_file.write('\nimport a\nimport b\n')
    main.main([str(tmpdir), '--extend-skip', 'skip.py', '--check'])
    main.main([str(python_file), str(should_skip), '--verbose', '--atomic'])
    with pytest.raises(SystemExit):
        main.main(['not-exist', '--check-only'])
    main.main([str(python_file), 'not-exist', '--verbose', '--check-only'])
    out, error = capsys.readouterr()
    assert 'Broken' in out
    with pytest.warns(UserWarning):
        main.main([str(python_file), '--recursive', '-fss'])
    with pytest.warns(UserWarning):
        main.main(['-sp', str(config_file), '-'], stdin=input_content)

def test_isort_filename_overrides(tmpdir, capsys):
    """Tests isorts available approaches for overriding filename and extension based behavior"""
    input_text = '\nimport b\n\n\ndef function():\n    pass\n\n\nimport a\n'
    config_file = tmpdir.join('.isort.cfg')
    config_file.write_text('\n[settings]\nfloat_to_top=True\n', encoding='utf8')
    file = tmpdir.join('file.py')
    file.write_text(input_text, encoding='utf8')
    main.main([str(file), '--float-to-top'])
    out, error = capsys.readouterr()
    assert not error
    assert file.read_text(encoding='utf8') == '\nimport a\nimport b\n\n\ndef function():\n    pass\n'
    main.main([str(file), '--dont-float-to-top'])
    _, error = capsys.readouterr()
    assert not error
    assert file.read_text(encoding='utf8') == input_text
    with pytest.raises(SystemExit):
        main.main([str(file), '--float-to-top', '--dont-float-to-top'])

def test_isort_with_stdin(capsys):
    input_content = as_stream('\nimport b\nimport a\n')
    main.main(['-'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport a\nimport b\n'
    input_content_from = as_stream('\nimport c\nimport b\nfrom a import z, y, x\n')
    main.main(['-'], stdin=input_content_from)
    out, error = capsys.readouterr()
    assert out == '\nimport b\nimport c\nfrom a import x, y, z\n'
    input_content = as_stream('\nimport sys\nimport pandas\nfrom z import abc\nfrom a import xyz\n')
    main.main(['-', '--fas'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom a import xyz\nfrom z import abc\n\nimport pandas\nimport sys\n'
    input_content = as_stream('\nfrom a import Path, abc\n')
    main.main(['-', '--fass'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom a import abc, Path\n'
    input_content = as_stream('\nimport b\nfrom c import x\nfrom a import y\n')
    main.main(['-', '--ff'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom a import y\nfrom c import x\nimport b\n'
    input_content = as_stream('\nimport b\nfrom a import a\n')
    main.main(['-', '--fss'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom a import a\nimport b\n'
    input_content = as_stream('\nimport a\nfrom b import c\n')
    main.main(['-', '--fss'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport a\nfrom b import c\n'
    input_content = as_stream('\nimport sys\nimport pandas\nimport a\n')
    main.main(['-', '--ds'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport a\nimport pandas\nimport sys\n'
    input_content = as_stream('\nfrom a import b\nfrom a import *\n')
    main.main(['-', '--cs'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom a import *\n'
    input_content = as_stream('\nfrom a import x as X\nfrom a import y as Y\n')
    main.main(['-', '--ca'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom a import x as X, y as Y\n'
    input_content = as_stream('\nimport os\nimport a\nimport b\n')
    main.main(['-', '--check-only', '--ws'], stdin=input_content)
    out, error = capsys.readouterr()
    assert not error
    input_content = as_stream('\nimport b\nimport a\n')
    with pytest.raises(SystemExit):
        main.main(['-', '--check', '--diff'], stdin=input_content)
    out, error = capsys.readouterr()
    assert error
    assert 'underlying stream is not seekable' not in error
    assert 'underlying stream is not seekable' not in error
    input_content = as_stream('\nimport abcdef\nimport x\n')
    main.main(['-', '--ls'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport x\nimport abcdef\n'
    input_content = as_stream('\nfrom z import b, c, a\n')
    main.main(['-', '--nis'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom z import b, c, a\n'
    input_content = as_stream('\nfrom z import b, c, a\n')
    main.main(['-', '--sl'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nfrom z import a\nfrom z import b\nfrom z import c\n'
    input_content = as_stream('\nimport os\nimport sys\n')
    main.main(['-', '--top', 'sys'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport sys\nimport os\n'
    input_content = as_stream('\nimport sys\nimport os\nimport z\nfrom a import b, e, c\n')
    main.main(['-', '--os'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport sys\nimport os\n\nimport z\nfrom a import b, e, c\n'
    input_content = as_stream('\nimport sys\nimport os\n')
    with pytest.warns(UserWarning):
        main.main(['-', '-ns'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport os\nimport sys\n'
    input_content = as_stream('\nimport sys\nimport os\n')
    with pytest.warns(UserWarning):
        main.main(['-', '-k'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport os\nimport sys\n'
    input_content = as_stream('\nimport a\nimport b\n')
    main.main(['-', '--verbose', '--only-modified'], stdin=input_content)
    out, error = capsys.readouterr()
    assert 'else-type place_module for a returned THIRDPARTY' not in out
    assert 'else-type place_module for b returned THIRDPARTY' not in out
    input_content = as_stream('\nimport a\nimport b\n')
    main.main(['-', '--combine-straight-imports'], stdin=input_content)
    out, error = capsys.readouterr()
    assert out == '\nimport a, b\n'

def test_unsupported_encodings(tmpdir, capsys):
    tmp_file = tmpdir.join('file.py')
    tmp_file.write_text('\n# [syntax-error]# -*- coding: IBO-8859-1 -*-\n""" check correct unknown encoding declaration\n"""\n__revision__ = \'יייי\'\n', encoding='utf8')
    with pytest.raises(SystemExit):
        main.main([str(tmp_file)])
    out, error = capsys.readouterr()
    assert 'No valid encodings.' in error
    normal_file = tmpdir.join('file1.py')
    normal_file.write_text('import os\nimport sys')
    main.main([str(tmp_file), str(normal_file), '--verbose'])
    out, error = capsys.readouterr()

def test_stream_skip_file(tmpdir, capsys):
    input_with_skip = '\n# isort: skip_file\nimport b\nimport a\n'
    stream_with_skip = as_stream(input_with_skip)
    main.main(['-'], stdin=stream_with_skip)
    out, error = capsys.readouterr()
    assert out == input_with_skip
    input_without_skip = input_with_skip.replace('isort: skip_file', 'generic comment')
    stream_without_skip = as_stream(input_without_skip)
    main.main(['-'], stdin=stream_without_skip)
    out, error = capsys.readouterr()
    assert out == '\n# generic comment\nimport a\nimport b\n'
    atomic_input_without_skip = input_with_skip.replace('isort: skip_file', 'generic comment')
    stream_without_skip = as_stream(atomic_input_without_skip)
    main.main(['-', '--atomic'], stdin=stream_without_skip)
    out, error = capsys.readouterr()
    assert out == '\n# generic comment\nimport a\nimport b\n'

def test_only_modified_flag(tmpdir, capsys):
    file1 = tmpdir.join('file1.py')
    file1.write('\nimport a\nimport b\n')
    file2 = tmpdir.join('file2.py')
    file2