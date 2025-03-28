import typing
from docutils import nodes
from typing import Dict, Set, Tuple, Optional, Union, Any

try:
    from sphinx.errors import NoUri
except ImportError:
    from sphinx.environment import NoUri  # noqa

APPATTRS: Dict[str, str] = {
    'Stream': 'faust.Stream',
    'TableManager': 'faust.tables.TableManager',
    'Serializers': 'faust.serializers.Registry',
    'sensors': 'faust.sensors.SensorDelegate',
    'serializers': 'faust.serializers.Registry',
    'sources': 'faust.topics.TopicManager',
    'tables': 'faust.tables.TableManager',
    'monitor': 'faust.sensors.Monitor',
    'consumer': 'faust.transport.base.Consumer',
    'transport': 'faust.transport.base.Transport',
    'producer': 'faust.transport.base.Producer',
}

APPDIRECT: Set[str] = {
    'client_only',
    'agents',
    'main',
    'topic',
    'agent',
    'task',
    'timer',
    'crontab',
    'stream',
    'Table',
    'Set',
    'start_client',
    'maybe_start_client',
    'send',
    'maybe_start_producer',
    'discover',
    'service',
    'page',
    'command',
}

APPATTRS.update({x: 'faust.App.{0}'.format(x) for x in APPDIRECT})

ABBRS: Dict[str, str] = {
    'App': 'faust.App',
}

ABBR_EMPTY: Dict[str, str] = {
    'exc': 'faust.exceptions',
}
DEFAULT_EMPTY: str = 'faust.App'


def typeify(S: str, type: str) -> str:
    if type in ('meth', 'func'):
        return S + '()'
    return S


def shorten(S: str, newtarget: str, src_dict: Dict[str, str]) -> str:
    if S.startswith('@-'):
        return S[2:]
    elif S.startswith('@'):
        if src_dict is APPATTRS:
            return '.'.join(['app', S[1:]])
        return S[1:]
    return S


def get_abbr(pre: str, rest: str, type: str, orig: Optional[str] = None) -> Tuple[str, str, Dict[str, str]]:
    if pre:
        for d in [APPATTRS, ABBRS]:
            try:
                return d[pre], rest, d
            except KeyError:
                pass
        raise KeyError('Unknown abbreviation: {0} ({1})'.format(
            '.'.join([pre, rest]) if orig is None else orig, type,
        ))
    else:
        for d in [APPATTRS, ABBRS]:
            try:
                return d[rest], '', d
            except KeyError:
                pass
    return ABBR_EMPTY.get(type, DEFAULT_EMPTY), rest, ABBR_EMPTY


def resolve(S: str, type: str) -> Tuple[str, Optional[Dict[str, str]]]:
    if '.' not in S:
        try:
            getattr(typing, S)
        except AttributeError:
            pass
        else:
            return 'typing.{0}'.format(S), None
    orig = S
    if S.startswith('@'):
        S = S.lstrip('@-')
        try:
            pre, rest = S.split('.', 1)
        except ValueError:
            pre, rest = '', S

        target, rest, src = get_abbr(pre, rest, type, orig)
        return '.'.join([target, rest]) if rest else target, src
    return S, None


def pkg_of(module_fqdn: str) -> str:
    return module_fqdn.split('.', 1)[0]


def basename(module_fqdn: str) -> str:
    return module_fqdn.lstrip('@').rsplit('.', -1)[-1]


def modify_textnode(T: str, newtarget: str, node: nodes.Node, src_dict: Dict[str, str], type: str) -> nodes.Text:
    src = node.children[0].rawsource
    return nodes.Text(
        (typeify(basename(T), type) if '~' in src
        else typeify(shorten(T, newtarget, src_dict), type),
        src,
    )


def maybe_resolve_abbreviations(app: Any, env: Any, node: nodes.Node, contnode: nodes.Node) -> Optional[Any]:
    domainname = node.get('refdomain')
    target = node['reftarget']
    typ = node['reftype']
    if target.startswith('@'):
        newtarget, src_dict = resolve(target, typ)
        node['reftarget'] = newtarget
        # shorten text if '~' is not enabled.
        if len(contnode) and isinstance(contnode[0], nodes.Text):
            contnode[0] = modify_textnode(target, newtarget, node,
                                          src_dict, typ)
        if domainname:
            try:
                domain = env.domains[node.get('refdomain')]
            except KeyError:
                raise NoUri
            return domain.resolve_xref(env, node['refdoc'], app.builder,
                                       typ, newtarget,
                                       node, contnode)


def setup(app: Any) -> None:
    app.connect(
        'missing-reference',
        maybe_resolve_abbreviations,
    )
    app.add_crossref_type(
        directivename='sig',
        rolename='sig',
        indextemplate='pair: %s; sig',
    )
