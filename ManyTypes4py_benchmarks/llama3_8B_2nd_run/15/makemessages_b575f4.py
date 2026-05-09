import glob
import itertools
import os
import re
import subprocess
from collections.abc import Collection, Iterator, Mapping
from typing import Any, Collection, Iterator, Mapping
import orjson
from django.core.management.base import CommandParser
from django.core.management.commands import makemessages
from django.template.base import BLOCK_TAG_END, BLOCK_TAG_START
from django.utils.translation import template
from typing_extensions import override

class Command(makemessages.Command):
    @override
    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument('--frontend-source', default='web/templates', help='Name of the Handlebars template directory')
        parser.add_argument('--frontend-output', default='locale', help='Name of the frontend messages output directory')
        parser.add_argument('--frontend-namespace', default='translations.json', help='Namespace of the frontend locale file')

    @override
    def handle(self, *args: Any, **options: Mapping[str, Any]) -> None:
        self.handle_django_locales(*args, **options)
        self.handle_frontend_locales(**options)

    def handle_frontend_locales(self, *, frontend_source: str, frontend_output: str, frontend_namespace: str, locale: str, exclude: Collection[str], all: bool, **options: Mapping[str, Any]) -> None:
        self.frontend_source = frontend_source
        self.frontend_output = frontend_output
        self.frontend_namespace = frontend_namespace
        self.frontend_locale = locale
        self.frontend_exclude = exclude
        self.frontend_all = all
        translation_strings = self.get_translation_strings()
        self.write_translation_strings(translation_strings)

    def handle_django_locales(self, *args: Any, **options: Mapping[str, Any]) -> None:
        old_endblock_re = template.endblock_re
        old_block_re = template.block_re
        old_constant_re = template.constant_re
        old_templatize = template.templatize
        template.endblock_re = re.compile(template.endblock_re.pattern + '|' + '^-?\\s*endtrans\\s*-?$')
        template.block_re = re.compile(template.block_re.pattern + '|' + '^-?\\s*trans(?:\\s+(?!\'|")(?=.*?=.*?)|\\s*-?$)')
        template.plural_re = re.compile(template.plural_re.pattern + '|' + '^-?\\s*pluralize(?:\\s+.+|-?$)')
        template.constant_re = re.compile('_\\(((?:".*?")|(?:\'.*?\')).*\\)')

        def my_templatize(src: str, *args: Any, **kwargs: Mapping[str, Any]) -> str:
            new_src = strip_whitespaces(src)
            return old_templatize(new_src, *args, **kwargs)
        template.templatize = my_templatize
        try:
            ignore_patterns = options.get('ignore_patterns', [])
            ignore_patterns.append('docs/*')
            ignore_patterns.append('templates/zerver/emails/custom/*')
            ignore_patterns.append('var/*')
            options['ignore_patterns'] = ignore_patterns
            super().handle(*args, **options)
        finally:
            template.endblock_re = old_endblock_re
            template.block_re = old_block_re
            template.templatize = old_templatize
            template.constant_re = old_constant_re

    # ... rest of the code ...
