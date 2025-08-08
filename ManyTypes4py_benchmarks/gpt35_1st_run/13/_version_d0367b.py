def get_keywords() -> Dict[str, str]:
    ...

class VersioneerConfig:
    VCS: str
    style: str
    tag_prefix: str
    parentdir_prefix: str
    versionfile_source: str
    verbose: bool

def get_config() -> VersioneerConfig:
    ...

def register_vcs_handler(vcs: str, method: str) -> Callable[[Callable], Callable]:
    ...

def run_command(commands: List[str], args: List[str], cwd: Optional[str] = None, verbose: bool = False, hide_stderr: bool = False, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[int]:
    ...

def versions_from_parentdir(parentdir_prefix: str, root: str, verbose: bool) -> Dict[str, Any]:
    ...

@register_vcs_handler('git', 'get_keywords')
def git_get_keywords(versionfile_abs: str) -> Dict[str, str]:
    ...

@register_vcs_handler('git', 'keywords')
def git_versions_from_keywords(keywords: Dict[str, str], tag_prefix: str, verbose: bool) -> Dict[str, Any]:
    ...

@register_vcs_handler('git', 'pieces_from_vcs')
def git_pieces_from_vcs(tag_prefix: str, root: str, verbose: bool, runner: Callable) -> Dict[str, Any]:
    ...

def plus_or_dot(pieces: Dict[str, Any]) -> str:
    ...

def render_pep440(pieces: Dict[str, Any]) -> str:
    ...

def render_pep440_branch(pieces: Dict[str, Any]) -> str:
    ...

def pep440_split_post(ver: str) -> Tuple[str, Optional[int]]:
    ...

def render_pep440_pre(pieces: Dict[str, Any]) -> str:
    ...

def render_pep440_post(pieces: Dict[str, Any]) -> str:
    ...

def render_pep440_post_branch(pieces: Dict[str, Any]) -> str:
    ...

def render_pep440_old(pieces: Dict[str, Any]) -> str:
    ...

def render_git_describe(pieces: Dict[str, Any]) -> str:
    ...

def render_git_describe_long(pieces: Dict[str, Any]) -> str:
    ...

def render(pieces: Dict[str, Any], style: str) -> Dict[str, Any]:
    ...

def get_versions() -> Dict[str, Any]:
    ...
