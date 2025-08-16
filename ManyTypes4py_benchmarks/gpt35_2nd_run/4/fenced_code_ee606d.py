from typing import Any
import re
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from typing_extensions import override

class FencedCodeExtension:
    def __init__(self, config: dict = {}):
        self.config: dict[str, Any] = {'run_content_validators': [config.get('run_content_validators', False), 'Boolean specifying whether to run content validation code in CodeHandler']}
        for key, value in config.items():
            self.setConfig(key, value)

    @override
    def extendMarkdown(self, md: Any) -> None:
        pass

class ZulipBaseHandler:
    def __init__(self, processor: Any, output: Any, fence: Any = None, process_contents: bool = False) -> None:
        pass

class FencedBlockPreprocessor:
    def __init__(self, md: Any, run_content_validators: bool = False) -> None:
        pass

    @override
    def run(self, lines: Sequence[str]) -> Sequence[str]:
        pass

    def format_code(self, lang: str, text: str) -> str:
        pass

    def format_quote(self, text: str) -> str:
        pass

    def format_spoiler(self, header: str, text: str) -> str:
        pass

    def format_tex(self, text: str) -> str:
        pass

    def placeholder(self, code: str) -> str:
        pass

    def _escape(self, txt: str) -> str:
        pass

def makeExtension(*args: Any, **kwargs: Any) -> FencedCodeExtension:
    pass
