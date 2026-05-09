import sys
import unittest
try:
    from tools.lib.template_parser import TemplateParserError, is_django_block_tag, tokenize, validate
except ImportError:
    print('ERROR!!! You need to run this via tools/test-tools.')
    sys.exit(1)

class ParserTest(unittest.TestCase):
    def _assert_validate_error(self, error: str, fn: str = None, text: str = None, template_format: str = None) -> None:
        with self.assertRaisesRegex(TemplateParserError, error):
            validate(fn=fn, text=text, template_format=template_format)

    def test_is_django_block_tag(self) -> None:
        self.assertTrue(is_django_block_tag('block'))
        self.assertFalse(is_django_block_tag('not a django tag'))

    # ... rest of the code ...
