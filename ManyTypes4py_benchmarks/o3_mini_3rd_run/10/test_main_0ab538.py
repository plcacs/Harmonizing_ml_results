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
from typing import Any, TextIO, Tuple, List, Union, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.tmpdir import TempPathFactory
else:
    from isort.wrap_modes import WrapModes

@given(
    file_name=st.text(),
    config=st.builds(Config),
    check=st.booleans(),
    ask_to_apply=st.booleans(),
    write_to_stdout=st.booleans()
)
def test_fuzz_sort_imports(
    file_name: str, config: Config, check: bool, ask_to_apply: bool, write_to_stdout: bool
) -> None:
    main.sort_imports(
        file_name=file_name,
        config=config,
        check=check,
        ask_to_apply=ask_to_apply,
        write_to_stdout=write_to_stdout,
    )

def test_sort_imports(tmpdir: Any) -> None:
    tmp_file = tmpdir.join('file.py')
    tmp_file.write('import os, sys\n')
    assert main.sort_imports(str(tmp_file), DEFAULT_CONFIG, check=True).incorrectly_sorted
    main.sort_imports(str(tmp_file), DEFAULT_CONFIG)
    assert not main.sort_imports(str(tmp_file), DEFAULT_CONFIG, check=True).incorrectly_sorted
    skip_config = Config(skip=['file.py'])
    assert main.sort_imports(
        str(tmp_file), config=skip_config, check=True, disregard_skip=False
    ).skipped
    assert main.sort_imports(
        str(tmp_file), config=skip_config, disregard_skip=False
    ).skipped

def test_sort_imports_error_handling(tmpdir: Any, mocker: Any, capsys: Any) -> None:
    tmp_file = tmpdir.join('file.py')
    tmp_file.write('import os, sys\n')
    mocker.patch('isort.core.process').side_effect = IndexError('Example unhandled exception')
    with pytest.raises(IndexError):
        main.sort_imports(str(tmp_file), DEFAULT_CONFIG, check=True).incorrectly_sorted
    out, error = capsys.readouterr()
    assert 'Unrecoverable exception thrown when parsing' in error

def test_parse_args() -> None:
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

def test_ascii_art(capsys: Any) -> None:
    main.main(['--version'])
    out, error = capsys.readouterr()
    expected = (
        f"\n                 _                 _\n"
        f"                (_) ___  ___  _ __| |_\n"
        f"                | |/ _/ / _ \\/ '__  _/\n"
        f"                | |\\__ \\/\\_\\/| |  | |_\n"
        f"                |_|\\___/\\___/\\_/   \\_/\n\n"
        f"      isort your imports, so you don't have to.\n\n"
        f"                    VERSION {__version__}\n\n"
    )
    assert out == expected
    assert error == ''

def test_preconvert() -> None:
    assert main._preconvert(frozenset([1, 1, 2])) == [1, 2]
    assert main._preconvert(WrapModes.GRID) == 'GRID'
    assert main._preconvert(main._preconvert) == '_preconvert'
    with pytest.raises(TypeError):
        main._preconvert(datetime.now())

def test_show_files(capsys: Any, tmpdir: Any) -> None:
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

def test_missing_default_section(tmpdir: Any) -> None:
    config_file = tmpdir.join('.isort.cfg')
    config_file.write('\n[settings]\nsections=MADEUP\n')
    python_file = tmpdir.join('file.py')
    python_file.write('import os')
    with pytest.raises(SystemExit):
        main.main([str(python_file)])

def test_ran_against_root() -> None:
    with pytest.raises(SystemExit):
        main.main(['/'])

def test_main(capsys: Any, tmpdir: Any) -> None:
    base_args: List[str] = ['-sp', str(tmpdir), '--virtual-env', str(tmpdir), '--src-path', str(tmpdir)]
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
    config_args: List[str] = ['--settings-path', str(config_file)]
    main.main(config_args + ['--virtual-env', '/random-root-folder-that-cant-exist-right?'] + ['--show-config'])
    out, error = capsys.readouterr()
    assert json.loads(out)['profile'] == 'hug'
    input_content = TextIOWrapper(BytesIO(b'\nimport b\nimport a\n'))
    main.main(config_args + ['-'], stdin=input_content)
    out, error = capsys.readouterr()
    expected_output = (
        f'\nelse-type place_module for b returned {DEFAULT_CONFIG.default_section}\n'
        f'else-type place_module for a returned {DEFAULT_CONFIG.default_section}\n'
        'import a\nimport b\n'
    )
    assert out == expected_output
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

def test_isort_filename_overrides(tmpdir: Any, capsys: Any) -> None:
    """Tests isorts available approaches for overriding filename and extension based behavior"""
    input_text: str = '\nimport b\nimport a\n\ndef function():\n    pass\n'

    def build_input_content() -> TextIO:
        return as_stream(input_text)
    main.main(['-'], stdin=build_input_content())
    out, error = capsys.readouterr()
    assert not error
    assert out == '\nimport a\nimport b\n\n\ndef function():\n    pass\n'
    main.main(['-', '--filename', 'x.py', '--skip', 'x.py', '--filter-files'], stdin=build_input_content())
    out, error = capsys.readouterr()
    assert not error
    assert out == '\nimport b\nimport a\n\ndef function():\n    pass\n'
    main.main(['-', '--ext-format', 'pyi'], stdin=build_input_content())
    out, error = capsys.readouterr()
    assert not error
    assert out == '\nimport a\nimport b\n\ndef function():\n    pass\n'
    tmp_file = tmpdir.join('tmp.pyi')
    tmp_file.write_text(input_text, encoding='utf8')
    main.main(['-', '--filename', str(tmp_file)], stdin=build_input_content())
    out, error = capsys.readouterr()
    assert not error
    assert out == '\nimport a\nimport b\n\ndef function():\n    pass\n'
    with pytest.raises(SystemExit):
        main.main([str(tmp_file), '--filename', str(tmp_file)], stdin=build_input_content())

def test_isort_float_to_top_overrides(tmpdir: Any, capsys: Any) -> None:
    """Tests isort supports overriding float to top from CLI"""
    test_input: str = '\nimport b\n\n\ndef function():\n    pass\n\n\nimport a\n'
    config_file = tmpdir.join('.isort.cfg')
    config_file.write('\n[settings]\nfloat_to_top=True\n')
    python_file = tmpdir.join('file.py')
    python_file.write(test_input)
    main.main([str(python_file)])
    out, error = capsys.readouterr()
    assert not error
    assert 'Fixing' in out
    assert python_file.read_text(encoding='utf8') == '\nimport a\nimport b\n\n\ndef function():\n    pass\n'
    python_file.write(test_input)
    main.main([str(python_file), '--dont-float-to-top'])
    _, error = capsys.readouterr()
    assert not error
    assert python_file.read_text(encoding='utf8') == test_input
    with pytest.raises(SystemExit):
        main.main([str(python_file), '--float-to-top', '--dont-float-to-top'])

def test_isort_with_stdin(capsys: Any) -> None:
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

def test_unsupported_encodings(tmpdir: Any, capsys: Any) -> None:
    tmp_file = tmpdir.join('file.py')
    tmp_file.write_text(
        '\n# [syntax-error]# -*- coding: IBO-8859-1 -*-\n""" check correct unknown encoding declaration\n"""\n__revision__ = \'יייי\'\n',
        encoding='utf8'
    )
    with pytest.raises(SystemExit):
        main.main([str(tmp_file)])
    out, error = capsys.readouterr()
    assert 'No valid encodings.' in error
    normal_file = tmpdir.join('file1.py')
    normal_file.write('import os\nimport sys')
    main.main([str(tmp_file), str(normal_file), '--verbose'])
    out, error = capsys.readouterr()

def test_stream_skip_file(tmpdir: Any, capsys: Any) -> None:
    input_with_skip: str = '\n# isort: skip_file\nimport b\nimport a\n'
    stream_with_skip = as_stream(input_with_skip)
    main.main(['-'], stdin=stream_with_skip)
    out, error = capsys.readouterr()
    assert out == input_with_skip
    input_without_skip: str = input_with_skip.replace('isort: skip_file', 'generic comment')
    stream_without_skip = as_stream(input_without_skip)
    main.main(['-'], stdin=stream_without_skip)
    out, error = capsys.readouterr()
    assert out == '\n# generic comment\nimport a\nimport b\n'
    atomic_input_without_skip: str = input_with_skip.replace('isort: skip_file', 'generic comment')
    stream_without_skip = as_stream(atomic_input_without_skip)
    main.main(['-', '--atomic'], stdin=stream_without_skip)
    out, error = capsys.readouterr()
    assert out == '\n# generic comment\nimport a\nimport b\n'

def test_only_modified_flag(tmpdir: Any, capsys: Any) -> None:
    file1 = tmpdir.join('file1.py')
    file1.write('\nimport a\nimport b\n')
    file2 = tmpdir.join('file2.py')
    file2.write('\nimport math\n\nimport pandas as pd\n')
    main.main([str(file1), str(file2), '--verbose', '--only-modified'])
    out, error = capsys.readouterr()
    expected = (
        f"\n                 _                 _\n"
        f"                (_) ___  ___  _ __| |_\n"
        f"                | |/ _/ / _ \\/ '__  _/\n"
        f"                | |\\__ \\/\\_\\/| |  | |_\n"
        f"                |_|\\___/\\___/\\_/   \\_/\n\n"
        f"      isort your imports, so you don't have to.\n\n"
        f"                    VERSION {__version__}\n\n"
    )
    assert out == expected
    assert not error
    file3 = tmpdir.join('file3.py')
    file3.write('\nimport sys\nimport os\n')
    main.main([str(file1), str(file2), str(file3), '--verbose', '--only-modified'])
    out, error = capsys.readouterr()
    assert 'else-type place_module for sys returned STDLIB' in out
    assert 'else-type place_module for os returned STDLIB' in out
    assert 'else-type place_module for math returned STDLIB' not in out
    assert 'else-type place_module for pandas returned THIRDPARTY' not in out
    assert not error
    main.main([str(file1), str(file2), '--check-only', '--verbose', '--only-modified'])
    out, error = capsys.readouterr()
    assert out == expected
    assert not error
    file4 = tmpdir.join('file4.py')
    file4.write('\nimport sys\nimport os\n')
    with pytest.raises(SystemExit):
        main.main([str(file2), str(file4), '--check-only', '--verbose', '--only-modified'])
    out, error = capsys.readouterr()
    assert 'else-type place_module for sys returned STDLIB' in out
    assert 'else-type place_module for os returned STDLIB' in out
    assert 'else-type place_module for math returned STDLIB' not in out
    assert 'else-type place_module for pandas returned THIRDPARTY' not in out

def test_identify_imports_main(tmpdir: Any, capsys: Any) -> None:
    file_content: str = 'import mod2\nimport mod2\na = 1\nimport mod1\n'
    some_file = tmpdir.join('some_file.py')
    some_file.write(file_content)
    file_imports: str = f'{some_file}:1 import mod2\n{some_file}:4 import mod1\n'
    file_imports_with_dupes: str = f'{some_file}:1 import mod2\n{some_file}:2 import mod2\n{some_file}:4 import mod1\n'
    main.identify_imports_main([str(some_file), '--unique'])
    out, error = capsys.readouterr()
    assert out.replace('\r\n', '\n') == file_imports
    assert not error
    main.identify_imports_main([str(some_file)])
    out, error = capsys.readouterr()
    assert out.replace('\r\n', '\n') == file_imports_with_dupes
    assert not error
    main.identify_imports_main(['-', '--unique'], stdin=as_stream(file_content))
    out, error = capsys.readouterr()
    assert out.replace('\r\n', '\n') == file_imports.replace(str(some_file), '')
    main.identify_imports_main(['-'], stdin=as_stream(file_content))
    out, error = capsys.readouterr()
    assert out.replace('\r\n', '\n') == file_imports_with_dupes.replace(str(some_file), '')
    main.identify_imports_main([str(tmpdir)])
    main.identify_imports_main(['-', '--packages'], stdin=as_stream(file_content))
    out, error = capsys.readouterr()
    assert len(out.split('\n')) == 6
    main.identify_imports_main(['-', '--modules'], stdin=as_stream(file_content))
    out, error = capsys.readouterr()
    assert len(out.split('\n')) == 3
    main.identify_imports_main(['-', '--attributes'], stdin=as_stream(file_content))
    out, error = capsys.readouterr()
    assert len(out.split('\n')) == 3

def test_gitignore(capsys: Any, tmp_path: pathlib.Path) -> None:
    import_content: str = '\nimport b\nimport a\n'

    def main_check(args: List[str]) -> Tuple[str, str]:
        try:
            main.main(args)
        except SystemExit:
            pass
        return capsys.readouterr()

    subprocess.run(['git', 'init', str(tmp_path)])
    python_file = tmp_path / 'has_imports.py'
    python_file.write_text(import_content)
    (tmp_path / 'no_imports.py').write_text('...')
    out, error = main_check([str(python_file), '--skip-gitignore', '--filter-files', '--check'])
    assert 'has_imports.py' in error and 'no_imports.py' not in error
    (tmp_path / '.gitignore').write_text('has_imports.py')
    out, error = main_check([str(python_file), '--check'])
    assert 'has_imports.py' in error and 'no_imports.py' not in error
    out, error = main_check([str(python_file), '--skip-gitignore', '--filter-files', '--check'])
    assert 'Skipped' in out
    (tmp_path / 'nested_dir').mkdir()
    (tmp_path / '.gitignore').write_text('nested_dir/has_imports.py')
    subfolder_file = tmp_path / 'nested_dir/has_imports.py'
    subfolder_file.write_text(import_content)
    out, error = main_check([str(tmp_path), '--skip-gitignore', '--filter-files', '--check'])
    assert 'has_imports.py' in error and 'nested_dir/has_imports.py' not in error
    currentdir: str = os.getcwd()
    os.chdir(tmp_path)
    out, error = main_check(['.', '--skip-gitignore', '--filter-files', '--check'])
    assert 'has_imports.py' in error and 'nested_dir/has_imports.py' not in error
    (tmp_path / '.gitignore').write_text('\nnested_dir/has_imports.py\nhas_imports.py\n')
    out, error = main_check(['.', '--skip-gitignore', '--filter-files', '--check'])
    assert 'Skipped' in out
    os.chdir(currentdir)
    shutil.rmtree(tmp_path / '.git')
    (tmp_path / '.gitignore').unlink()
    git_project0: pathlib.Path = tmp_path / 'git_project0'
    git_project0.mkdir()
    subprocess.run(['git', 'init', str(git_project0)])
    (git_project0 / '.gitignore').write_text('has_imports_ignored.py')
    (git_project0 / 'has_imports_ignored.py').write_text(import_content)
    (git_project0 / 'has_imports.py').write_text(import_content)
    git_project1: pathlib.Path = tmp_path / 'git_project1'
    git_project1.mkdir()
    subprocess.run(['git', 'init', str(git_project1)])
    (git_project1 / '.gitignore').write_text('\nnested_dir/has_imports_ignored.py\nnested_dir_ignored\n')
    (git_project1 / 'has_imports.py').write_text(import_content)
    nested_dir = git_project1 / 'nested_dir'
    nested_dir.mkdir()
    (nested_dir / 'has_imports.py').write_text(import_content)
    (nested_dir / 'has_imports_ignored.py').write_text(import_content)
    nested_ignored_dir = git_project1 / 'nested_dir_ignored'
    nested_ignored_dir.mkdir()
    (nested_ignored_dir / 'has_imports.py').write_text(import_content)
    should_check: List[str] = [
        '/has_imports.py',
        '/nested_dir/has_imports.py',
        '/git_project0/has_imports.py',
        '/git_project1/has_imports.py',
        '/git_project1/nested_dir/has_imports.py'
    ]
    out, error = main_check([str(tmp_path), '--skip-gitignore', '--filter-files', '--check'])
    if os.name == 'nt':
        should_check = [sc.replace('/', '\\') for sc in should_check]
    assert all((f'{tmp_path}{file}' in error for file in should_check))
    out, error = main_check([str(tmp_path), '--skip-gitignore', '--filter-files'])
    assert all((f'{tmp_path}{file}' in out for file in should_check))
    if os.name != 'nt':
        (git_project0 / 'has_imports_ignored.py').write_text(import_content)
        (git_project0 / 'has_imports.py').write_text(import_content)
        (tmp_path / 'has_imports.py').write_text(import_content)
        (tmp_path / 'nested_dir' / 'has_imports.py').write_text(import_content)
        (git_project0 / 'ignore_link.py').symlink_to(tmp_path / 'has_imports.py')
        (git_project0 / 'ignore_link').symlink_to(tmp_path / 'nested_dir')
        gitignore = git_project0 / '.gitignore'
        gitignore.write_text(gitignore.read_text() + 'ignore_link.py\nignore_link')
        out, error = main_check([str(git_project0), '--skip-gitignore', '--filter-files', '--check'])
        should_check = ['/git_project0/has_imports.py']
        assert all((f'{tmp_path}{file}' in error for file in should_check))
        out, error = main_check([str(git_project0), '--skip-gitignore', '--filter-files'])
        assert all((f'{tmp_path}{file}' in out for file in should_check))

def test_multiple_configs(capsys: Any, tmpdir: Any) -> None:
    setup_cfg: str = '\n[isort]\nfrom_first=True\n'
    pyproject_toml: str = '\n[tool.isort]\nno_inline_sort = "True"\n'
    isort_cfg: str = '\n[settings]\nforce_single_line=True\n'
    broken_isort_cfg: str = '\n[iaort_confg]\nforce_single_line=True\n'
    dir1 = tmpdir / 'subdir1'
    dir2 = tmpdir / 'subdir2'
    dir3 = tmpdir / 'subdir3'
    dir4 = tmpdir / 'subdir4'
    dir1.mkdir()
    dir2.mkdir()
    dir3.mkdir()
    dir4.mkdir()
    setup_cfg_file = dir1 / 'setup.cfg'
    setup_cfg_file.write_text(setup_cfg, 'utf-8')
    pyproject_toml_file = dir2 / 'pyproject.toml'
    pyproject_toml_file.write_text(pyproject_toml, 'utf-8')
    isort_cfg_file = dir3 / '.isort.cfg'
    isort_cfg_file.write_text(isort_cfg, 'utf-8')
    broken_isort_cfg_file = dir4 / '.isort.cfg'
    broken_isort_cfg_file.write_text(broken_isort_cfg, 'utf-8')
    import_section: str = '\nfrom a import y, z, x\nimport b\n'
    file1 = dir1 / 'file1.py'
    file1.write_text(import_section, 'utf-8')
    file2 = dir2 / 'file2.py'
    file2.write_text(import_section, 'utf-8')
    file3 = dir3 / 'file3.py'
    file3.write_text(import_section, 'utf-8')
    file4 = dir4 / 'file4.py'
    file4.write_text(import_section, 'utf-8')
    file5 = tmpdir / 'file5.py'
    file5.write_text(import_section, 'utf-8')
    main.main([str(tmpdir), '--resolve-all-configs', '--cr', str(tmpdir), '--verbose'])
    out, _ = capsys.readouterr()
    assert f'{str(setup_cfg_file)} used for file {str(file1)}' in out
    assert f'{str(pyproject_toml_file)} used for file {str(file2)}' in out
    assert f'{str(isort_cfg_file)} used for file {str(file3)}' in out
    assert f'default used for file {str(file4)}' in out
    assert f'default used for file {str(file5)}' in out
    assert file1.read() == '\nfrom a import x, y, z\nimport b\n'
    assert file2.read() == '\nimport b\nfrom a import y, z, x\n'
    assert file3.read() == '\nimport b\nfrom a import x\nfrom a import y\nfrom a import z\n'
    assert file4.read() == '\nimport b\nfrom a import x, y, z\n'
    assert file5.read() == '\nimport b\nfrom a import x, y, z\n'
    file6 = dir1 / 'file6.py'
    file6.write('\nimport b\nfrom a import x, y, z\n    ')
    with pytest.raises(SystemExit):
        main.main([str(tmpdir), '--resolve-all-configs', '--cr', str(tmpdir), '--check'])
    _, err = capsys.readouterr()
    assert f'{str(file6)} Imports are incorrectly sorted and/or formatted' in err

def test_multiple_src_paths(tmpdir: pathlib.Path, capsys: Any) -> None:
    """
    Ensure that isort has consistent behavior with multiple source paths
    """
    tests_module: pathlib.Path = tmpdir / 'tests'
    app_module: pathlib.Path = tmpdir / 'app'
    tests_module.mkdir()
    app_module.mkdir()
    pyproject_toml = tmpdir / 'pyproject.toml'
    pyproject_toml.write_text(
        '\n[tool.isort]\nprofile = "black"\nsrc_paths = ["app", "tests"]\nauto_identify_namespace_packages = false\n',
        'utf-8'
    )
    file = tmpdir / 'file.py'
    file.write_text('\nfrom app.something import something\nfrom tests.something import something_else\n', 'utf-8')
    for _ in range(10):
        main.main([str(tmpdir), '--verbose'])
        out, _ = capsys.readouterr()
        assert file.read() == '\nfrom app.something import something\nfrom tests.something import something_else\n'
        assert 'from-type place_module for tests.something returned FIRSTPARTY' in out
