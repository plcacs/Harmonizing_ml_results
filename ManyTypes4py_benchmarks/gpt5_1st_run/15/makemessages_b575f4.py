import glob
import itertools
import os
import re
import subprocess
from collections.abc import Collection, Iterator, Mapping
from typing import Any, cast

import orjson
from django.core.management.base import CommandParser
from django.core.management.commands import makemessages
from django.template.base import BLOCK_TAG_END, BLOCK_TAG_START
from django.utils.translation import template
from typing_extensions import override

strip_whitespace_right: re.Pattern[str] = re.compile(f'({BLOCK_TAG_START}-?\\s*(trans|pluralize).*?-{BLOCK_TAG_END})\\s+')
strip_whitespace_left: re.Pattern[str] = re.compile(f'\\s+({BLOCK_TAG_START}-\\s*(endtrans|pluralize).*?-?{BLOCK_TAG_END})')
regexes: list[str] = [
    '{{~?#tr}}([\\s\\S]*?)(?:~?{{/tr}}|{{~?#\\*inline )',
    '{{~?\\s*t "([\\s\\S]*?)"\\W*~?}}',
    "{{~?\\s*t '([\\s\\S]*?)'\\W*~?}}",
    '\\(t "([\\s\\S]*?)"\\)',
    '=\\(t "([\\s\\S]*?)"\\)(?=[^{]*}})',
    "=\\(t '([\\s\\S]*?)'\\)(?=[^{]*}})",
]
tags: list[tuple[str, str]] = [('err_', 'error')]
frontend_compiled_regexes: list[re.Pattern[str]] = [re.compile(regex) for regex in regexes]
multiline_js_comment: re.Pattern[str] = re.compile('/\\*.*?\\*/', re.DOTALL)
singleline_js_comment: re.Pattern[str] = re.compile('//.*?\\n')


def strip_whitespaces(src: str) -> str:
    src = strip_whitespace_left.sub('\\1', src)
    src = strip_whitespace_right.sub('\\1', src)
    return src


class Command(makemessages.Command):
    xgettext_options: list[str] = makemessages.Command.xgettext_options
    for func, tag in tags:
        xgettext_options += [f'--keyword={func}:1,"{tag}"']

    # Instance attribute annotations
    frontend_source: str
    frontend_output: str
    frontend_namespace: str
    frontend_locale: Collection[str] | None
    frontend_exclude: Collection[str]
    frontend_all: bool

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument('--frontend-source', default='web/templates', help='Name of the Handlebars template directory')
        parser.add_argument('--frontend-output', default='locale', help='Name of the frontend messages output directory')
        parser.add_argument('--frontend-namespace', default='translations.json', help='Namespace of the frontend locale file')

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        self.handle_django_locales(*args, **options)
        self.handle_frontend_locales(**options)

    def handle_frontend_locales(
        self,
        *,
        frontend_source: str,
        frontend_output: str,
        frontend_namespace: str,
        locale: Collection[str] | None,
        exclude: Collection[str],
        all: bool,
        **options: Any,
    ) -> None:
        self.frontend_source = frontend_source
        self.frontend_output = frontend_output
        self.frontend_namespace = frontend_namespace
        self.frontend_locale = locale
        self.frontend_exclude = exclude
        self.frontend_all = all
        translation_strings = self.get_translation_strings()
        self.write_translation_strings(translation_strings)

    def handle_django_locales(self, *args: Any, **options: Any) -> None:
        old_endblock_re: re.Pattern[str] = template.endblock_re
        old_block_re: re.Pattern[str] = template.block_re
        old_constant_re: re.Pattern[str] = template.constant_re
        old_templatize: Any = template.templatize
        template.endblock_re = re.compile(template.endblock_re.pattern + '|' + '^-?\\s*endtrans\\s*-?$')
        template.block_re = re.compile(template.block_re.pattern + '|' + '^-?\\s*trans(?:\\s+(?!\'|")(?=.*?=.*?)|\\s*-?$)')
        template.plural_re = re.compile(template.plural_re.pattern + '|' + '^-?\\s*pluralize(?:\\s+.+|-?$)')
        template.constant_re = re.compile('_\\(((?:".*?")|(?:\'.*?\')).*\\)')

        def my_templatize(src: str, *args: Any, **kwargs: Any) -> str:
            new_src = strip_whitespaces(src)
            return cast(str, old_templatize(new_src, *args, **kwargs))

        template.templatize = my_templatize
        try:
            ignore_patterns: list[str] = options.get('ignore_patterns', [])
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

    def extract_strings(self, data: str) -> list[str]:
        translation_strings: list[str] = []
        for regex in frontend_compiled_regexes:
            for match in regex.findall(data):
                match = match.strip()
                match = ' '.join((line.strip() for line in match.splitlines()))
                translation_strings.append(match)
        return translation_strings

    def ignore_javascript_comments(self, data: str) -> str:
        data = multiline_js_comment.sub('', data)
        data = singleline_js_comment.sub('', data)
        return data

    def get_translation_strings(self) -> list[str]:
        translation_strings: list[str] = []
        dirname = self.get_template_dir()
        for dirpath, dirnames, filenames in os.walk(dirname):
            for filename in [f for f in filenames if f.endswith('.hbs')]:
                if filename.startswith('.'):
                    continue
                with open(os.path.join(dirpath, filename)) as reader:
                    data = reader.read()
                    translation_strings.extend(self.extract_strings(data))
        for dirpath, dirnames, filenames in itertools.chain(os.walk('web/src'), os.walk('web/shared/src')):
            for filename in [f for f in filenames if f.endswith(('.js', '.ts'))]:
                if filename.startswith('.'):
                    continue
                with open(os.path.join(dirpath, filename)) as reader:
                    data = reader.read()
                    data = self.ignore_javascript_comments(data)
                    translation_strings.extend(self.extract_strings(data))
        extracted: bytes = subprocess.check_output(
            [
                'node_modules/.bin/formatjs',
                'extract',
                '--additional-function-names=$t,$t_html',
                '--format=simple',
                '--ignore=**/*.d.ts',
                'web/src/**/*.js',
                'web/src/**/*.ts',
            ]
        )
        mapping: Mapping[str, str] = cast(Mapping[str, str], orjson.loads(extracted))
        translation_strings.extend(list(mapping.values()))
        return list(set(translation_strings))

    def get_template_dir(self) -> str:
        return self.frontend_source

    def get_namespace(self) -> str:
        return self.frontend_namespace

    def get_locales(self) -> Collection[str]:
        locale = self.frontend_locale
        exclude = self.frontend_exclude
        process_all = self.frontend_all
        default_locale_path = self.default_locale_path
        paths = glob.glob(f'{default_locale_path}/*')
        all_locales: list[str] = [os.path.basename(path) for path in paths if os.path.isdir(path)]
        if process_all:
            return all_locales
        else:
            locales: Collection[str] = locale or all_locales
            return set(locales) - set(exclude)

    def get_base_path(self) -> str:
        return self.frontend_output

    def get_output_paths(self) -> Iterator[str]:
        base_path = self.get_base_path()
        locales = self.get_locales()
        for path in [os.path.join(base_path, locale) for locale in locales]:
            if not os.path.exists(path):
                os.makedirs(path)
            yield os.path.join(path, self.get_namespace())

    def get_new_strings(
        self,
        old_strings: Mapping[str, str],
        translation_strings: Collection[str],
        locale: str,
    ) -> dict[str, str]:
        """
        Missing strings are removed, new strings are added and already
        translated strings are not touched.
        """
        new_strings: dict[str, str] = {}
        for k in translation_strings:
            if locale == 'en':
                new_strings[k] = old_strings.get(k, k)
            else:
                new_strings[k] = old_strings.get(k, '')
        return new_strings

    def write_translation_strings(self, translation_strings: Collection[str]) -> None:
        for locale, output_path in zip(self.get_locales(), self.get_output_paths(), strict=False):
            self.stdout.write(f'[frontend] processing locale {locale}')
            try:
                with open(output_path, 'rb') as reader:
                    old_strings = cast(dict[str, str], orjson.loads(reader.read()))
            except (OSError, ValueError):
                old_strings = {}
            new_strings = self.get_new_strings(old_strings, translation_strings, locale)
            with open(output_path, 'wb') as writer:
                writer.write(
                    orjson.dumps(
                        new_strings,
                        option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
                    )
                )