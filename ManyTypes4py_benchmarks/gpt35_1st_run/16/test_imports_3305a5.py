from __future__ import annotations
import astroid
from pylint.checkers import BaseChecker
import pylint.testutils
from pylint.testutils.unittest_linter import UnittestLinter
import pytest
from . import assert_adds_messages, assert_no_messages

def test_good_import(linter: UnittestLinter, imports_checker: BaseChecker, module_name: str, import_from: str, import_what: str) -> None:
    import_node: astroid.node_classes.NodeNG = astroid.extract_node(f'from {import_from} import {import_what} #@', module_name)
    imports_checker.visit_module(import_node.parent)
    with assert_no_messages(linter):
        imports_checker.visit_importfrom(import_node)

def test_bad_import(linter: UnittestLinter, imports_checker: BaseChecker, module_name: str, import_from: str, import_what: str, error_code: str) -> None:
    import_node: astroid.node_classes.NodeNG = astroid.extract_node(f'from {import_from} import {import_what} #@', module_name)
    imports_checker.visit_module(import_node.parent)
    with assert_adds_messages(linter, pylint.testutils.MessageTest(msg_id=error_code, node=import_node, args=None, line=1, col_offset=0, end_line=1, end_col_offset=len(import_from) + len(import_what) + 13)):
        imports_checker.visit_importfrom(import_node)

def test_good_root_import(linter: UnittestLinter, imports_checker: BaseChecker, import_node: str, module_name: str) -> None:
    node: astroid.node_classes.NodeNG = astroid.extract_node(f'{import_node} #@', module_name)
    imports_checker.visit_module(node.parent)
    with assert_no_messages(linter):
        if import_node.startswith('import'):
            imports_checker.visit_import(node)
        if import_node.startswith('from'):
            imports_checker.visit_importfrom(node)

def test_bad_root_import(linter: UnittestLinter, imports_checker: BaseChecker, import_node: str, module_name: str) -> None:
    node: astroid.node_classes.NodeNG = astroid.extract_node(f'{import_node} #@', module_name)
    imports_checker.visit_module(node.parent)
    with assert_adds_messages(linter, pylint.testutils.MessageTest(msg_id='hass-component-root-import', node=node, args=None, line=1, col_offset=0, end_line=1, end_col_offset=len(import_node))):
        if import_node.startswith('import'):
            imports_checker.visit_import(node)
        if import_node.startswith('from'):
            imports_checker.visit_importfrom(node)

def test_bad_namespace_import(linter: UnittestLinter, imports_checker: BaseChecker, import_node: str, module_name: str, expected_args: tuple) -> None:
    node: astroid.node_classes.NodeNG = astroid.extract_node(f'{import_node} #@', module_name)
    imports_checker.visit_module(node.parent)
    with assert_adds_messages(linter, pylint.testutils.MessageTest(msg_id='hass-helper-namespace-import', node=node, args=expected_args, line=1, col_offset=0, end_line=1, end_col_offset=len(import_node))):
        imports_checker.visit_importfrom(node)

def test_domain_alias(linter: UnittestLinter, imports_checker: BaseChecker, module_name: str, import_string: str, end_col_offset: int) -> None:
    import_node: astroid.node_classes.NodeNG = astroid.extract_node(f'{import_string}  #@', module_name)
    imports_checker.visit_module(import_node.parent)
    expected_messages = []
    if end_col_offset > 0:
        expected_messages.append(pylint.testutils.MessageTest(msg_id='hass-import-constant-alias', node=import_node, args=('DOMAIN', 'DOMAIN', 'OTHER_DOMAIN'), line=1, col_offset=0, end_line=1, end_col_offset=end_col_offset))
    with assert_adds_messages(linter, *expected_messages):
        if import_string.startswith('import'):
            imports_checker.visit_import(import_node)
        else:
            imports_checker.visit_importfrom(import_node)
