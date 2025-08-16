from typing import Dict, List, Pattern, Tuple, Union
from git import RemoteProgress, Repo

EXTERNAL_MODULES: List[str]
IGNORED_ERRORS: Dict[str, List[Union[str, Pattern]]]
IGNORED_MODULES: List[str]
MOCK_OBJECTS: List[str]

def get_unused_ignores(ignored_message_freq: Dict[str, Dict[Union[str, Pattern], int]]) -> List[str]:
    unused_ignores: List[str] = []
    for root_key, patterns in IGNORED_ERRORS.items():
        for pattern in patterns:
            if ignored_message_freq[root_key][pattern] == 0 and pattern not in itertools.chain(EXTERNAL_MODULES, MOCK_OBJECTS):
                unused_ignores.append(f'{root_key}: {pattern}')
    return unused_ignores

def is_pattern_fits(pattern: Union[str, Pattern], line: str) -> bool:
    if isinstance(pattern, Pattern):
        if pattern.search(line):
            return True
    elif pattern in line:
        return True
    return False

def is_ignored(line: str, test_folder_name: str, *, ignored_message_freqs: Dict[str, Dict[Union[str, Pattern], int]]) -> bool:
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

def get_django_repo_object(branch: str) -> Repo:
    if not DJANGO_SOURCE_DIRECTORY.exists():
        DJANGO_SOURCE_DIRECTORY.mkdir(exist_ok=True, parents=False)
        return Repo.clone_from('https://github.com/django/django.git', DJANGO_SOURCE_DIRECTORY, progress=ProgressPrinter(), branch=branch, depth=100)
    else:
        repo = Repo(DJANGO_SOURCE_DIRECTORY)
        return repo
