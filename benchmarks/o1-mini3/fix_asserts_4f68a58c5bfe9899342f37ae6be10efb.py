from typing import Dict, Any
from ..fixer_base import BaseFix
from ..fixer_util import Name

NAMES: Dict[str, str] = {
    'assert_': 'assertTrue',
    'assertEquals': 'assertEqual',
    'assertNotEquals': 'assertNotEqual',
    'assertAlmostEquals': 'assertAlmostEqual',
    'assertNotAlmostEqual': 'assertNotAlmostEqual',
    'assertRegexpMatches': 'assertRegex',
    'assertRaisesRegexp': 'assertRaisesRegex',
    'failUnlessEqual': 'assertEqual',
    'failIfEqual': 'assertNotEqual',
    'failUnlessAlmostEqual': 'assertAlmostEqual',
    'failIfAlmostEqual': 'assertNotAlmostEqual',
    'failUnless': 'assertTrue',
    'failUnlessRaises': 'assertRaises',
    'failIf': 'assertFalse',
}

class FixAsserts(BaseFix):
    PATTERN: str = (
        "\n              power< any+ trailer< '.' meth=(%s)> any* >\n              " % '|'.join(map(repr, NAMES))
    )

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        name = results['meth'][0]
        name.replace(Name(NAMES[str(name)], prefix=name.prefix))
