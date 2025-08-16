from typing import Any
import re
import subprocess
import os
import glob
import itertools
from collections.abc import Collection, Iterator, Mapping
import orjson
from django.core.management.base import CommandParser
from django.core.management.commands import makemessages
from django.template.base import BLOCK_TAG_END, BLOCK_TAG_START
from django.utils.translation import template
from typing_extensions import override

strip_whitespace_right: re.Pattern = re.compile(f'({BLOCK_TAG_START}-?\\s*(trans|pluralize).*?-{BLOCK_TAG_END})\\s+')
strip_whitespace_left: re.Pattern = re.compile(f'\\s+({BLOCK_TAG_START}-\\s*(endtrans|pluralize).*?-?{BLOCK_TAG_END})')
regexes: list[str] = ['{{~?#tr}}([\\s\\S]*?)(?:~?{{/tr}}|{{~?#\\*inline )', '{{~?\\s*t "([\\s\\S]*?)"\\W*~?}}', "{{~?\\s*t '([\\s\\S]*?)'\\W*~?}}", '\\(t "([\\s\\S]*?)"\\)', '=\\(t "([\\s\\S]*?)"\\)(?=[^{]*}})', "=\\(t '([\\s\\S]*?)'\\)(?=[^{]*}}"]
tags: list[tuple[str, str]] = [('err_', 'error')]
frontend_compiled_regexes: list[re.Pattern] = [re.compile(regex) for regex in regexes]

class Command(makemessages.Command):
    xgettext_options: list[str] = makemessages.Command.xgettext_options

    def add_arguments(self, parser: CommandParser) -> None:
        ...

    def handle(self, *args: Any, **options: Any) -> None:
        ...

    def handle_frontend_locales(self, *, frontend_source: str, frontend_output: str, frontend_namespace: str, locale: Any, exclude: Any, all: Any, **options: Any) -> None:
        ...

    def handle_django_locales(self, *args: Any, **options: Any) -> None:
        ...

    def extract_strings(self, data: str) -> list[str]:
        ...

    def ignore_javascript_comments(self, data: str) -> str:
        ...

    def get_translation_strings(self) -> list[str]:
        ...

    def get_template_dir(self) -> str:
        ...

    def get_namespace(self) -> str:
        ...

    def get_locales(self) -> list[str]:
        ...

    def get_base_path(self) -> str:
        ...

    def get_output_paths(self) -> Iterator[str]:
        ...

    def get_new_strings(self, old_strings: Mapping[str, str], translation_strings: list[str], locale: str) -> Mapping[str, str]:
        ...

    def write_translation_strings(self, translation_strings: list[str]) -> None:
        ...
