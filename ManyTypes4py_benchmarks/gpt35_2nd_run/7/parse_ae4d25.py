from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Set, Tuple
from . import place
from .comments import parse as parse_comments
from .exceptions import MissingSection
from .settings import DEFAULT_CONFIG, Config

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

CommentsAboveDict = TypedDict('CommentsAboveDict', {'straight': Dict[str, Any], 'from': Dict[str, Any]})
CommentsDict = TypedDict('CommentsDict', {'from': Dict[str, Any], 'straight': Dict[str, Any], 'nested': Dict[str, Any], 'above': CommentsAboveDict})

def _infer_line_separator(contents: str) -> str:
    ...

def normalize_line(raw_line: str) -> Tuple[str, str]:
    ...

def import_type(line: str, config: Config = DEFAULT_CONFIG) -> Optional[str]:
    ...

def strip_syntax(import_string: str) -> str:
    ...

def skip_line(line: str, in_quote: str, index: int, section_comments: bool, needs_import: bool = True) -> Tuple[bool, str]:
    ...

class ParsedContent(NamedTuple):
    pass

def file_contents(contents: str, config: Config = DEFAULT_CONFIG) -> ParsedContent:
    ...
