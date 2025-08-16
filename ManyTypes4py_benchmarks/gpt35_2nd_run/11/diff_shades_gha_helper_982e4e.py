from typing import Any, Final, Literal

COMMENT_FILE: Final = '.pr-comment.json'
DIFF_STEP_NAME: Final = 'Generate HTML diff report'
DOCS_URL: Final = 'https://black.readthedocs.io/en/latest/contributing/gauging_changes.html#diff-shades'
USER_AGENT: Final = f'psf/black diff-shades workflow via urllib3/{urllib3.__version__}'
SHA_LENGTH: Final = 10
GH_API_TOKEN: str = os.getenv('GITHUB_TOKEN')
REPO: str = os.getenv('GITHUB_REPOSITORY', default='psf/black')

def set_output(name: str, value: str) -> None:
def http_get(url: str, *, is_json: bool = True, **kwargs: Any) -> Any:
def get_main_revision() -> str:
def get_pr_revision(pr: int) -> str:
def get_pypi_version() -> Version:
