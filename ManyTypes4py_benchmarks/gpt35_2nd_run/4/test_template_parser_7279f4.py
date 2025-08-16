import sys
import unittest
from tools.lib.template_parser import TemplateParserError, is_django_block_tag, tokenize, validate

class ParserTest(unittest.TestCase):

    def _assert_validate_error(self, error: str, fn: str = None, text: str = None, template_format: str = None) -> None:
        with self.assertRaisesRegex(TemplateParserError, error):
            validate(fn=fn, text=text, template_format=template_format)

    def test_is_django_block_tag(self) -> None:
        self.assertTrue(is_django_block_tag('block'))
        self.assertFalse(is_django_block_tag('not a django tag'))

    def test_validate_vanilla_html(self) -> None:
        my_html: str = '\n            <table>\n                <tr>\n                <td>foo</td>\n                </tr>\n            </table>'
        validate(text=my_html)

    def test_validate_handlebars(self) -> None:
        my_html: str = '\n            {{#with stream}}\n                <p>{{stream}}</p>\n            {{/with}}\n            '
        validate(text=my_html, template_format='handlebars')

    # Remaining test methods with appropriate type annotations
