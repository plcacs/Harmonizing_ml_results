import itertools
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Pattern, Tuple, Union
from git import RemoteProgress, Repo
from scripts.enabled_test_modules import EXTERNAL_MODULES, IGNORED_ERRORS, IGNORED_MODULES, MOCK_OBJECTS

DJANGO_COMMIT_REFS: Dict[str, Tuple[str, str]] = {
    '2.2': ('stable/2.2.x', '996be04c3ceb456754d9d527d4d708f30727f07e'),
    '3.0': ('stable/3.0.x', 'd9f1792c7649e9f946f4a3a35a76bddf5a412b8b'),
}
PROJECT_DIRECTORY: Path = Path(__file__).parent.parent
DJANGO_SOURCE_DIRECTORY: Path = PROJECT_DIRECTORY / 'django-sources'


def get_unused_ignores(ignored_message_freq: Dict[str, Dict[Union[Pattern[str], str], int]]) -> List[str]:
    unused_ignores: List[str] = []
    for root_key, patterns in IGNORED_ERRORS.items():
        for pattern in patterns:
            if (
                ignored_message_freq[root_key][pattern] == 0
                and pattern not in itertools.chain(EXTERNAL_MODULES, MOCK_OBJECTS)
            ):
                unused_ignores.append(f'{root_key}: {pattern}')
    return unused_ignores


def is_pattern_fits(pattern: Union[Pattern[str], str], line: str) -> bool:
    if isinstance(pattern, Pattern):
        if pattern.search(line):
            return True
    elif pattern in line:
        return True
    return False


def is_ignored(
    line: str,
    test_folder_name: str,
    *,
    ignored_message_freqs: Dict[str, Dict[Union[Pattern[str], str], int]]
) -> bool:
    if 'runtests' in line:
        return True
    if test_folder_name in IGNORED_MODULES:
        return True
    for pattern in IGNORED_ERRORS.get(test_folder_name, []):
        if is_pattern_fits(pattern, line):
            ignored_message_freqs[test_folder_name][pattern] += 1
            return True
    for pattern in IGNORED_ERRORS['__common__']:
        if is_pattern_fits(pattern, line):
            ignored_message_freqs['__common__'][pattern] += 1
            return True
    return False


def replace_with_clickable_location(error: str, abs_test_folder: Path) -> str:
    raw_path, _, error_line = error.partition(': ')
    fname, _, line_number = raw_path.partition(':')
    try:
        path = abs_test_folder.joinpath(fname).relative_to(PROJECT_DIRECTORY)
    except ValueError:
        return error
    clickable_location = f'./{path}:{line_number or 1}'
    return error.replace(raw_path, clickable_location)


class ProgressPrinter(RemoteProgress):

    def line_dropped(self, line: str) -> None:
        print(line)

    def update(self, op_code: int, cur_count: int, max_count: Union[int, None] = None, message: str = '') -> None:
        print(self._cur_line)


def get_django_repo_object(branch: str) -> Repo:
    if not DJANGO_SOURCE_DIRECTORY.exists():
        DJANGO_SOURCE_DIRECTORY.mkdir(exist_ok=True, parents=False)
        return Repo.clone_from(
            'https://github.com/django/django.git',
            DJANGO_SOURCE_DIRECTORY,
            progress=ProgressPrinter(),
            branch=branch,
            depth=100
        )
    else:
        repo: Repo = Repo(DJANGO_SOURCE_DIRECTORY)
        return repo


if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--django_version', choices=['2.2', '3.0'], required=True)
    args = parser.parse_args()
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'Django=={args.django_version}.*'])
    branch, commit_sha = DJANGO_COMMIT_REFS[args.django_version]
    repo: Repo = get_django_repo_object(branch)
    if repo.head.commit.hexsha != commit_sha:
        repo.remote('origin').fetch(branch, progress=ProgressPrinter(), depth=100)
        repo.git.checkout(commit_sha)
    mypy_config_file: Path = (PROJECT_DIRECTORY / 'scripts' / 'mypy.ini').absolute()
    mypy_cache_dir: Path = Path(__file__).parent / '.mypy_cache'
    tests_root: Path = DJANGO_SOURCE_DIRECTORY / 'tests'
    global_rc: int = 0
    try:
        mypy_options: List[str] = [
            '--cache-dir', str(mypy_config_file.parent / '.mypy_cache'),
            '--config-file', str(mypy_config_file),
            '--show-traceback',
            '--no-error-summary',
            '--hide-error-context'
        ]
        mypy_options += [str(tests_root)]
        import distutils.spawn
        mypy_executable: Union[str, None] = distutils.spawn.find_executable('mypy')
        if mypy_executable is None:
            raise RuntimeError('mypy executable not found')
        mypy_argv: List[str] = [mypy_executable, *mypy_options]
        completed: subprocess.CompletedProcess = subprocess.run(
            mypy_argv,
            env={'PYTHONPATH': str(tests_root)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        output: str = completed.stdout.decode()
        ignored_message_freqs: Dict[str, Dict[Union[Pattern[str], str], int]] = defaultdict(lambda: defaultdict(int))
        sorted_lines: List[str] = sorted(output.splitlines())
        for line in sorted_lines:
            try:
                path_to_error: str = line.split(':')[0]
                test_folder_name: str = path_to_error.split('/')[2]
            except IndexError:
                test_folder_name = 'unknown'
            if not is_ignored(line, test_folder_name, ignored_message_freqs=ignored_message_freqs):
                global_rc = 1
                print(line)
        unused_ignores: List[str] = get_unused_ignores(ignored_message_freqs)
        if unused_ignores:
            print('UNUSED IGNORES ------------------------------------------------')
            print('\n'.join(unused_ignores))
        sys.exit(global_rc)
    except BaseException as exc:
        shutil.rmtree(mypy_cache_dir, ignore_errors=True)
        raise exc
