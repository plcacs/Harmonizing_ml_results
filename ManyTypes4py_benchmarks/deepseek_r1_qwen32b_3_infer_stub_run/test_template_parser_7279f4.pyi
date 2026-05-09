import unittest
from tools.lib.template_parser import (
    TemplateParserError,
    is_django_block_tag,
    tokenize,
    validate,
)

class ParserTest(unittest.TestCase):
    def _assert_validate_error(
        self,
        error: str,
        fn: str | None = ...,
        text: str | None = ...,
        template_format: str | None = ...,
    ) -> None:
        ...

    def test_is_django_block_tag(self, tag: str) -> bool:
        ...

    def test_validate_vanilla_html(self, text: str) -> None:
        ...

    def test_validate_handlebars(self, text: str) -> None:
        ...

    def test_validate_handlebars_partial_block(self, text: str) -> None:
        ...

    def test_validate_bad_handlebars_partial_block(
        self,
        text: str,
        template_format: str,
    ) -> None:
        ...

    def test_validate_comment(self, text: str) -> None:
        ...

    def test_validate_django(self, text: str) -> None:
        ...

    def test_validate_no_start_tag(self, text: str) -> None:
        ...

    def test_validate_mismatched_tag(self, text: str) -> None:
        ...

    def test_validate_bad_indentation(self, text: str) -> None:
        ...

    def test_validate_state_depth(self, text: str) -> None:
        ...

    def test_validate_incomplete_handlebars_tag_1(self, text: str) -> None:
        ...

    def test_validate_incomplete_handlebars_tag_2(self, text: str) -> None:
        ...

    def test_validate_incomplete_django_tag_1(self, text: str) -> None:
        ...

    def test_validate_incomplete_django_tag_2(self, text: str) -> None:
        ...

    def test_validate_incomplete_html_tag_1(self, text: str) -> None:
        ...

    def test_validate_incomplete_html_tag_2(
        self,
        my_html: str,
        my_html1: str,
    ) -> None:
        ...

    def test_validate_empty_html_tag(self, text: str) -> None:
        ...

    def test_code_blocks(
        self,
        my_html: str,
        my_html1: str,
        my_html2: str,
    ) -> None:
        ...

    def test_anchor_blocks(
        self,
        my_html: str,
        my_html1: str,
        my_html2: str,
    ) -> None:
        ...

    def test_validate_jinja2_whitespace_markers_1(self, text: str) -> None:
        ...

    def test_validate_jinja2_whitespace_markers_2(self, text: str) -> None:
        ...

    def test_validate_jinja2_whitespace_markers_3(self, text: str) -> None:
        ...

    def test_validate_jinja2_whitespace_markers_4(self, text: str) -> None:
        ...

    def test_validate_mismatch_jinja2_whitespace_markers_1(
        self,
        text: str,
        template_format: str,
    ) -> None:
        ...

    def test_validate_jinja2_whitespace_type2_markers(self, text: str) -> None:
        ...

    def test_tokenize(
        self,
        tag: str,
        template_format: str | None = ...,
    ) -> None:
        ...