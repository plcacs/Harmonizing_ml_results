import sys
import unittest
from typing import Optional, Union, Any
try:
    from tools.lib.template_parser import TemplateParserError, is_django_block_tag, tokenize, validate
except ImportError:
    print('ERROR!!! You need to run this via tools/test-tools.')
    sys.exit(1)


class ParserTest(unittest.TestCase):

    def _assert_validate_error(self, error, fn=None, text=None,
        template_format=None):
        with self.assertRaisesRegex(TemplateParserError, error):
            validate(fn=fn, text=text, template_format=template_format)

    def test_is_django_block_tag(self):
        self.assertTrue(is_django_block_tag('block'))
        self.assertFalse(is_django_block_tag('not a django tag'))

    def test_validate_vanilla_html(self):
        """
        Verify that validate() does not raise errors for
        well-formed HTML.
        """
        my_html: str = """
            <table>
                <tr>
                <td>foo</td>
                </tr>
            </table>"""
        validate(text=my_html)

    def test_validate_handlebars(self):
        my_html: str = """
            {{#with stream}}
                <p>{{stream}}</p>
            {{/with}}
            """
        validate(text=my_html, template_format='handlebars')

    def test_validate_handlebars_partial_block(self):
        my_html: str = """
            {{#> generic_thing }}
                <p>hello!</p>
            {{/generic_thing}}
            """
        validate(text=my_html, template_format='handlebars')

    def test_validate_bad_handlebars_partial_block(self):
        my_html: str = """
            {{#> generic_thing }}
                <p>hello!</p>
            {{# generic_thing}}
            """
        self._assert_validate_error(
            'Missing end tag for the token at row 4 13!', text=my_html,
            template_format='handlebars')

    def test_validate_comment(self):
        my_html: str = """
            <!---
                <h1>foo</h1>
            -->"""
        validate(text=my_html)

    def test_validate_django(self):
        my_html: str = """
            {% include "some_other.html" %}
            {% if foo %}
                <p>bar</p>
            {% endif %}
            """
        validate(text=my_html, template_format='django')
        my_html = """
            {% block "content" %}
                {% with className="class" %}
                {% include 'foobar' %}
                {% endwith %}
            {% endblock %}
            """
        validate(text=my_html)

    def test_validate_no_start_tag(self):
        my_html: str = """
            foo</p>
        """
        self._assert_validate_error('No start tag', text=my_html)

    def test_validate_mismatched_tag(self):
        my_html: str = """
            <b>foo</i>
        """
        self._assert_validate_error('Mismatched tags: \\(b != i\\)', text=
            my_html)

    def test_validate_bad_indentation(self):
        my_html: str = """
            <p>
                foo
                </p>
        """
        self._assert_validate_error(
            'Indentation for start/end tags does not match.', text=my_html)

    def test_validate_state_depth(self):
        my_html: str = """
            <b>
        """
        self._assert_validate_error('Missing end tag', text=my_html)

    def test_validate_incomplete_handlebars_tag_1(self):
        my_html: str = """
            {{# foo
        """
        self._assert_validate_error(
            """Tag missing "}}" at line 2 col 13:"{{# foo
        \"""",
            text=my_html, template_format='handlebars')

    def test_validate_incomplete_handlebars_tag_2(self):
        my_html: str = """
            {{# foo }
        """
        self._assert_validate_error(
            'Tag missing "}}" at line 2 col 13:"{{# foo }\n"', text=my_html,
            template_format='handlebars')

    def test_validate_incomplete_django_tag_1(self):
        my_html: str = """
            {% foo
        """
        self._assert_validate_error(
            """Tag missing "%}" at line 2 col 13:"{% foo
        \"""",
            text=my_html, template_format='django')

    def test_validate_incomplete_django_tag_2(self):
        my_html: str = """
            {% foo %
        """
        self._assert_validate_error(
            'Tag missing "%}" at line 2 col 13:"{% foo %\n"', text=my_html,
            template_format='django')

    def test_validate_incomplete_html_tag_1(self):
        my_html: str = """
            <b
        """
        self._assert_validate_error(
            """Tag missing ">" at line 2 col 13:"<b
        \"""", text=my_html
            )

    def test_validate_incomplete_html_tag_2(self):
        my_html: str = """
            <a href="
        """
        my_html1: str = """
            <a href=""
        """
        self._assert_validate_error(
            """Tag missing ">" at line 2 col 13:"<a href=""
        \"""",
            text=my_html1)
        self._assert_validate_error(
            """Unbalanced quotes at line 2 col 13:"<a href="
        \"""",
            text=my_html)

    def test_validate_empty_html_tag(self):
        my_html: str = """
            < >
        """
        self._assert_validate_error('Tag name missing', text=my_html)

    def test_code_blocks(self):
        my_html: str = """
            <code>
                x = 5
                y = x + 1
            </code>"""
        validate(text=my_html)
        my_html = '<code>process_widgets()</code>'
        self.assertIsInstance(my_html, str)
        validate(text=my_html)
        my_html: str = """
            <code>x =
            5</code>
            """
        self._assert_validate_error('Code tag is split across two lines.',
            text=my_html)

    def test_anchor_blocks(self):
        my_html: str = """
            <a href="/some/url">
            Click here
            for more info.
            </a>"""
        validate(text=my_html)
        my_html: str = '<a href="/some/url">click here</a>'
        validate(text=my_html)
        my_html: str = """
            <a class="twitter-timeline" href="https://twitter.com/ZulipStatus"
                data-widget-id="443457763394334720"
                data-screen-name="ZulipStatus"
                >@ZulipStatus on Twitter</a>.
            """
        validate(text=my_html)

    def test_validate_jinja2_whitespace_markers_1(self):
        my_html: str = """
        {% if foo -%}
        this is foo
        {% endif %}
        """
        validate(text=my_html, template_format='django')

    def test_validate_jinja2_whitespace_markers_2(self):
        my_html: str = """
        {% if foo %}
        this is foo
        {%- endif %}
        """
        validate(text=my_html, template_format='django')

    def test_validate_jinja2_whitespace_markers_3(self):
        my_html: str = """
        {% if foo %}
        this is foo
        {% endif -%}
        """
        validate(text=my_html, template_format='django')

    def test_validate_jinja2_whitespace_markers_4(self):
        my_html: str = """
        {%- if foo %}
        this is foo
        {% endif %}
        """
        validate(text=my_html, template_format='django')

    def test_validate_mismatch_jinja2_whitespace_markers_1(self):
        my_html: str = """
        {% if foo %}
        this is foo
        {%- if bar %}
        """
        self._assert_validate_error('Missing end tag', text=my_html,
            template_format='django')

    def test_validate_jinja2_whitespace_type2_markers(self):
        my_html: str = """
        {%- if foo -%}
        this is foo
        {% endif %}
        """
        validate(text=my_html, template_format='django')

    def test_tokenize(self):
        tag: str = '<!DOCTYPE html>'
        token: Any = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_doctype')
        tag = '<a>bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_start')
        self.assertEqual(token.tag, 'a')
        tag = '<br />bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_singleton')
        self.assertEqual(token.tag, 'br')
        tag = '<input>bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_start')
        self.assertEqual(token.tag, 'input')
        tag = '<input />bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_singleton')
        self.assertEqual(token.tag, 'input')
        tag = '</a>bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_end')
        self.assertEqual(token.tag, 'a')
        tag = '{{#with foo}}bla'
        token = tokenize(tag, template_format='handlebars')[0]
        self.assertEqual(token.kind, 'handlebars_start')
        self.assertEqual(token.tag, 'with')
        tag = '{{/with}}bla'
        token = tokenize(tag, template_format='handlebars')[0]
        self.assertEqual(token.kind, 'handlebars_end')
        self.assertEqual(token.tag, 'with')
        tag = '{{#> compose_banner }}bla'
        token = tokenize(tag, template_format='handlebars')[0]
        self.assertEqual(token.kind, 'handlebars_partial_block')
        self.assertEqual(token.tag, 'compose_banner')
        tag = '{% if foo %}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'django_start')
        self.assertEqual(token.tag, 'if')
        tag = '{% endif %}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'django_end')
        self.assertEqual(token.tag, 'if')
        tag = '{% if foo -%}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'jinja2_whitespace_stripped_start')
        self.assertEqual(token.tag, 'if')
        tag = '{%- endif %}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'jinja2_whitespace_stripped_end')
        self.assertEqual(token.tag, 'if')
        tag = '{%- if foo -%}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'jinja2_whitespace_stripped_type2_start')
        self.assertEqual(token.tag, 'if')
