#!/usr/bin/env python3
import re
from typing import Any, Dict, List, Tuple
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.utils import unescape
from sphinx import addnodes
from sphinx.domains.changeset import VersionChange, versionlabels, versionlabel_classes
from sphinx.domains.python import PyFunction, PyMethod, PyModule
from sphinx.locale import _ as sphinx_gettext
from sphinx.util.docutils import SphinxDirective
ISSUE_URI: str = 'https://bugs.python.org/issue?@action=redirect&bpo=%s'
GH_ISSUE_URI: str = 'https://github.com/python/cpython/issues/%s'
SOURCE_URI: str = 'https://github.com/python/cpython/tree/3.13/%s'
from docutils.parsers.rst.states import Body

Body.enum.converters['loweralpha'] = Body.enum.converters['upperalpha'] = Body.enum.converters['lowerroman'] = Body.enum.converters['upperroman'] = (lambda x: None)
from sphinx.domains import std

std.token_re = re.compile('`((~?[\\w-]*:)?\\w+)`')
PyModule.option_spec['no-index'] = directives.flag


def issue_role(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Any,
    options: Dict[Any, Any] = {},
    content: List[str] = []
) -> Tuple[List[nodes.Node], List[Any]]:
    issue: str = unescape(text)
    if 47261 < int(issue) < 400000:
        msg = inliner.reporter.error(
            f'The BPO ID {text!r} seems too high -- use :gh:`...` for GitHub IDs',
            line=lineno
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    text = 'bpo-' + issue
    refnode: nodes.reference = nodes.reference(text, text, refuri=(ISSUE_URI % issue))
    return ([refnode], [])


def gh_issue_role(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Any,
    options: Dict[Any, Any] = {},
    content: List[str] = []
) -> Tuple[List[nodes.Node], List[Any]]:
    issue: str = unescape(text)
    if int(issue) < 32426:
        msg = inliner.reporter.error(
            f'The GitHub ID {text!r} seems too low -- use :issue:`...` for BPO IDs',
            line=lineno
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    text = 'gh-' + issue
    refnode: nodes.reference = nodes.reference(text, text, refuri=(GH_ISSUE_URI % issue))
    return ([refnode], [])


class ImplementationDetail(SphinxDirective):
    has_content: bool = True
    final_argument_whitespace: bool = True
    label_text: str = sphinx_gettext('CPython implementation detail:')

    def run(self) -> List[nodes.Node]:
        self.assert_has_content()
        pnode: nodes.compound = nodes.compound(classes=['impl-detail'])
        content = self.content
        add_text: nodes.strong = nodes.strong(self.label_text, self.label_text)
        self.state.nested_parse(content, self.content_offset, pnode)
        inline_content: nodes.inline = nodes.inline(pnode[0].rawsource, translatable=True)
        inline_content.source = pnode[0].source
        inline_content.line = pnode[0].line
        inline_content += pnode[0].children
        pnode[0].replace_self(
            nodes.paragraph('', '', add_text, nodes.Text(' '), inline_content, translatable=False)
        )
        return [pnode]


class PyDecoratorMixin:
    def handle_signature(self, sig: str, signode: nodes.Element) -> Any:
        ret: Any = super(PyDecoratorMixin, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_addname('@', '@'))
        return ret

    def needs_arglist(self) -> bool:
        return False


class PyDecoratorFunction(PyDecoratorMixin, PyFunction):
    def run(self) -> List[nodes.Node]:
        self.name = 'py:function'
        return PyFunction.run(self)


class PyDecoratorMethod(PyDecoratorMixin, PyMethod):
    def run(self) -> List[nodes.Node]:
        self.name = 'py:method'
        return PyMethod.run(self)


class PyCoroutineMixin:
    def handle_signature(self, sig: str, signode: nodes.Element) -> Any:
        ret: Any = super(PyCoroutineMixin, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_annotation('coroutine ', 'coroutine '))
        return ret


class PyAwaitableMixin:
    def handle_signature(self, sig: str, signode: nodes.Element) -> Any:
        ret: Any = super(PyAwaitableMixin, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_annotation('awaitable ', 'awaitable '))
        return ret


class PyCoroutineFunction(PyCoroutineMixin, PyFunction):
    def run(self) -> List[nodes.Node]:
        self.name = 'py:function'
        return PyFunction.run(self)


class PyCoroutineMethod(PyCoroutineMixin, PyMethod):
    def run(self) -> List[nodes.Node]:
        self.name = 'py:method'
        return PyMethod.run(self)


class PyAwaitableFunction(PyAwaitableMixin, PyFunction):
    def run(self) -> List[nodes.Node]:
        self.name = 'py:function'
        return PyFunction.run(self)


class PyAwaitableMethod(PyAwaitableMixin, PyMethod):
    def run(self) -> List[nodes.Node]:
        self.name = 'py:method'
        return PyMethod.run(self)


class PyAbstractMethod(PyMethod):
    def handle_signature(self, sig: str, signode: nodes.Element) -> Any:
        ret: Any = super(PyAbstractMethod, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_annotation('abstractmethod ', 'abstractmethod '))
        return ret

    def run(self) -> List[nodes.Node]:
        self.name = 'py:method'
        return PyMethod.run(self)


class DeprecatedRemoved(VersionChange):
    required_arguments: int = 2
    _deprecated_label: str = sphinx_gettext('Deprecated since version %s, will be removed in version %s')
    _removed_label: str = sphinx_gettext('Deprecated since version %s, removed in version %s')

    def run(self) -> List[nodes.Node]:
        version_deprecated: str = self.arguments[0]
        version_removed: str = self.arguments.pop(1)
        self.arguments[0] = (version_deprecated, version_removed)
        current_version: Tuple[int, ...] = tuple(map(int, self.config.version.split('.')))
        removed_version: Tuple[int, ...] = tuple(map(int, version_removed.split('.')))
        if current_version < removed_version:
            versionlabels[self.name] = self._deprecated_label
            versionlabel_classes[self.name] = 'deprecated'
        else:
            versionlabels[self.name] = self._removed_label
            versionlabel_classes[self.name] = 'removed'
        try:
            return super(DeprecatedRemoved, self).run()
        finally:
            versionlabels[self.name] = ''
            versionlabel_classes[self.name] = ''


def setup(app: Any) -> Dict[str, Any]:
    app.add_role('issue', issue_role)
    app.add_role('gh', gh_issue_role)
    app.add_directive('impl-detail', ImplementationDetail)
    app.add_directive('deprecated-removed', DeprecatedRemoved)
    app.add_directive_to_domain('py', 'decorator', PyDecoratorFunction)
    app.add_directive_to_domain('py', 'decoratormethod', PyDecoratorMethod)
    app.add_directive_to_domain('py', 'coroutinefunction', PyCoroutineFunction)
    app.add_directive_to_domain('py', 'coroutinemethod', PyCoroutineMethod)
    app.add_directive_to_domain('py', 'awaitablefunction', PyAwaitableFunction)
    app.add_directive_to_domain('py', 'awaitablemethod', PyAwaitableMethod)
    app.add_directive_to_domain('py', 'abstractmethod', PyAbstractMethod)
    return {'version': '1.0', 'parallel_read_safe': True}