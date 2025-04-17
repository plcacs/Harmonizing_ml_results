
import re
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.utils import unescape
from sphinx import addnodes
from sphinx.domains.changeset import VersionChange, versionlabels, versionlabel_classes
from sphinx.domains.python import PyFunction, PyMethod, PyModule
from sphinx.locale import _ as sphinx_gettext
from sphinx.util.docutils import SphinxDirective
ISSUE_URI = 'https://bugs.python.org/issue?@action=redirect&bpo=%s'
GH_ISSUE_URI = 'https://github.com/python/cpython/issues/%s'
SOURCE_URI = 'https://github.com/python/cpython/tree/3.13/%s'
from docutils.parsers.rst.states import Body
Body.enum.converters['loweralpha'] = Body.enum.converters['upperalpha'] = Body.enum.converters['lowerroman'] = Body.enum.converters['upperroman'] = (lambda x: None)
from sphinx.domains import std
std.token_re = re.compile('`((~?[\\w-]*:)?\\w+)`')
PyModule.option_spec['no-index'] = directives.flag

def issue_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    issue = unescape(text)
    if (47261 < int(issue) < 400000):
        msg = inliner.reporter.error(f'The BPO ID {text!r} seems too high -- use :gh:`...` for GitHub IDs', line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    text = ('bpo-' + issue)
    refnode = nodes.reference(text, text, refuri=(ISSUE_URI % issue))
    return ([refnode], [])

def gh_issue_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    issue = unescape(text)
    if (int(issue) < 32426):
        msg = inliner.reporter.error(f'The GitHub ID {text!r} seems too low -- use :issue:`...` for BPO IDs', line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    text = ('gh-' + issue)
    refnode = nodes.reference(text, text, refuri=(GH_ISSUE_URI % issue))
    return ([refnode], [])

class ImplementationDetail(SphinxDirective):
    has_content = True
    final_argument_whitespace = True
    label_text = sphinx_gettext('CPython implementation detail:')

    def run(self):
        self.assert_has_content()
        pnode = nodes.compound(classes=['impl-detail'])
        content = self.content
        add_text = nodes.strong(self.label_text, self.label_text)
        self.state.nested_parse(content, self.content_offset, pnode)
        content = nodes.inline(pnode[0].rawsource, translatable=True)
        content.source = pnode[0].source
        content.line = pnode[0].line
        content += pnode[0].children
        pnode[0].replace_self(nodes.paragraph('', '', add_text, nodes.Text(' '), content, translatable=False))
        return [pnode]

class PyDecoratorMixin(object):

    def handle_signature(self, sig, signode):
        ret = super(PyDecoratorMixin, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_addname('@', '@'))
        return ret

    def needs_arglist(self):
        return False

class PyDecoratorFunction(PyDecoratorMixin, PyFunction):

    def run(self):
        self.name = 'py:function'
        return PyFunction.run(self)

class PyDecoratorMethod(PyDecoratorMixin, PyMethod):

    def run(self):
        self.name = 'py:method'
        return PyMethod.run(self)

class PyCoroutineMixin(object):

    def handle_signature(self, sig, signode):
        ret = super(PyCoroutineMixin, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_annotation('coroutine ', 'coroutine '))
        return ret

class PyAwaitableMixin(object):

    def handle_signature(self, sig, signode):
        ret = super(PyAwaitableMixin, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_annotation('awaitable ', 'awaitable '))
        return ret

class PyCoroutineFunction(PyCoroutineMixin, PyFunction):

    def run(self):
        self.name = 'py:function'
        return PyFunction.run(self)

class PyCoroutineMethod(PyCoroutineMixin, PyMethod):

    def run(self):
        self.name = 'py:method'
        return PyMethod.run(self)

class PyAwaitableFunction(PyAwaitableMixin, PyFunction):

    def run(self):
        self.name = 'py:function'
        return PyFunction.run(self)

class PyAwaitableMethod(PyAwaitableMixin, PyMethod):

    def run(self):
        self.name = 'py:method'
        return PyMethod.run(self)

class PyAbstractMethod(PyMethod):

    def handle_signature(self, sig, signode):
        ret = super(PyAbstractMethod, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_annotation('abstractmethod ', 'abstractmethod '))
        return ret

    def run(self):
        self.name = 'py:method'
        return PyMethod.run(self)

class DeprecatedRemoved(VersionChange):
    required_arguments = 2
    _deprecated_label = sphinx_gettext('Deprecated since version %s, will be removed in version %s')
    _removed_label = sphinx_gettext('Deprecated since version %s, removed in version %s')

    def run(self):
        version_deprecated = self.arguments[0]
        version_removed = self.arguments.pop(1)
        self.arguments[0] = (version_deprecated, version_removed)
        current_version = tuple(map(int, self.config.version.split('.')))
        removed_version = tuple(map(int, version_removed.split('.')))
        if (current_version < removed_version):
            versionlabels[self.name] = self._deprecated_label
            versionlabel_classes[self.name] = 'deprecated'
        else:
            versionlabels[self.name] = self._removed_label
            versionlabel_classes[self.name] = 'removed'
        try:
            return super().run()
        finally:
            versionlabels[self.name] = ''
            versionlabel_classes[self.name] = ''

def setup(app):
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
