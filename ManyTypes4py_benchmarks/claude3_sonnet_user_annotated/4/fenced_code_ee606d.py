import re
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from typing import Any

import lxml.html
from django.utils.html import escape
from markdown import Markdown
from markdown.extensions import Extension, codehilite
from markdown.extensions.codehilite import CodeHiliteExtension, parse_hl_lines
from markdown.preprocessors import Preprocessor
from pygments.lexers import find_lexer_class_by_name
from pygments.util import ClassNotFound
from typing_extensions import override

from zerver.lib.exceptions import MarkdownRenderingError
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITIES
from zerver.lib.tex import render_tex

# Global vars
FENCE_RE = re.compile(
    r"""
    # ~~~ or 