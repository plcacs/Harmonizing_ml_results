from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
import json
import hashlib
import logging
import collections
import requests
import urlparse
import yaml
import validate
import pprint
import StringIO
from aslist import aslist
import rdflib
from rdflib.namespace import RDF, RDFS, OWL

_logger = logging.getLogger('salad')


class NormDict(dict):
    def __init__(self, normalize: Callable[[str], str] = unicode) -> None:
        super(NormDict, self).__init__()
        self.normalize: Callable[[str], str] = normalize

    def __getitem__(self, key: str) -> Any:
        return super(NormDict, self).__getitem__(self.normalize(key))

    def __setitem__(self, key: str, value: Any) -> None:
        super(NormDict, self).__setitem__(self.normalize(key), value)

    def __delitem__(self, key: str) -> None:
        super(NormDict, self).__delitem__(self.normalize(key))

    def __contains__(self, key: str) -> bool:
        return super(NormDict, self).__contains__(self.normalize(key))


def merge_properties(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    c: Dict[str, Any] = {}
    for i in a:
        if i not in b:
            c[i] = a[i]
    for i in b:
        if i not in a:
            c[i] = b[i]
    for i in a:
        if i in b:
            c[i] = aslist(a[i]) + aslist(b[i])
    return c


def SubLoader(loader: "Loader") -> "Loader":
    return Loader(loader.ctx,
                  schemagraph=loader.graph,
                  foreign_properties=loader.foreign_properties,
                  idx=loader.idx,
                  cache=loader.cache)


class Loader(object):
    def __init__(self,
                 ctx: Dict[str, Any],
                 schemagraph: Optional[rdflib.Graph] = None,
                 foreign_properties: Optional[Set[str]] = None,
                 idx: Optional[NormDict] = None,
                 cache: Optional[Dict[str, Any]] = None) -> None:
        normalize: Callable[[str], str] = lambda url: urlparse.urlsplit(url).geturl()
        if idx is not None:
            self.idx: NormDict = idx
        else:
            self.idx = NormDict(normalize)
        self.ctx: Dict[str, Any] = {}
        if schemagraph is not None:
            self.graph: rdflib.Graph = schemagraph
        else:
            self.graph = rdflib.Graph()
        if foreign_properties is not None:
            self.foreign_properties: Set[str] = foreign_properties
        else:
            self.foreign_properties = set()
        if cache is not None:
            self.cache: Dict[str, Any] = cache
        else:
            self.cache = {}
        self.url_fields: Set[str] = set()
        self.vocab_fields: Set[str] = set()
        self.identifiers: Set[str] = set()
        self.identity_links: Set[str] = set()
        self.standalone: Set[str] = set()
        self.nolinkcheck: Set[str] = set()
        self.vocab: Dict[str, str] = {}
        self.rvocab: Dict[str, str] = {}
        self.add_context(ctx)

    def expand_url(self, url: str, base_url: str, scoped: bool = False, vocab_term: bool = False) -> str:
        if url in ('@id', '@type'):
            return url
        if vocab_term and url in self.vocab:
            return url
        if self.vocab and ':' in url:
            prefix = url.split(':')[0]
            if prefix in self.vocab:
                url = self.vocab[prefix] + url[len(prefix) + 1:]
        split = urlparse.urlsplit(url)
        if split.scheme or url.startswith('$(') or url.startswith('${'):
            pass
        elif scoped and (not split.fragment):
            splitbase = urlparse.urlsplit(base_url)
            frg = ''
            if splitbase.fragment:
                frg = splitbase.fragment + '/' + split.path
            else:
                frg = split.path
            url = urlparse.urlunsplit((splitbase.scheme, splitbase.netloc, splitbase.path, splitbase.query, frg))
        else:
            url = urlparse.urljoin(base_url, url)
        if vocab_term and url in self.rvocab:
            return self.rvocab[url]
        else:
            return url

    def _add_properties(self, s: rdflib.term.Identifier) -> None:
        for _, _, rng in self.graph.triples((s, RDFS.range, None)):
            literal = str(rng).startswith('http://www.w3.org/2001/XMLSchema#') and (not str(rng) == 'http://www.w3.org/2001/XMLSchema#anyURI') or str(rng) == 'http://www.w3.org/2000/01/rdf-schema#Literal'
            if not literal:
                self.url_fields.add(str(s))
        self.foreign_properties.add(str(s))

    def add_namespaces(self, ns: Dict[str, str]) -> None:
        self.vocab.update(ns)

    def add_schemas(self, ns: Union[List[str], str], base_url: str) -> None:
        for sch in aslist(ns):
            self.graph.parse(urlparse.urljoin(base_url, sch))
        for s, _, _ in self.graph.triples((None, RDF.type, RDF.Property)):
            self._add_properties(s)
        for s, _, o in self.graph.triples((None, RDFS.subPropertyOf, None)):
            self._add_properties(s)
            self._add_properties(o)
        for s, _, _ in self.graph.triples((None, RDFS.range, None)):
            self._add_properties(s)
        for s, _, _ in self.graph.triples((None, RDF.type, OWL.ObjectProperty)):
            self._add_properties(s)
        for s, _, _ in self.graph.triples((None, None, None)):
            self.idx[str(s)] = True

    def add_context(self, newcontext: Dict[str, Any], baseuri: str = '') -> None:
        if self.vocab:
            raise validate.ValidationException('Refreshing context that already has stuff in it')
        self.url_fields = set()
        self.vocab_fields = set()
        self.identifiers = set()
        self.identity_links = set()
        self.standalone = set()
        self.nolinkcheck = set()
        self.vocab = {}
        self.rvocab = {}
        self.ctx.update({k: v for k, v in newcontext.iteritems() if k != '@context'})
        _logger.debug('ctx is %s', self.ctx)
        for c in self.ctx:
            if self.ctx[c] == '@id':
                self.identifiers.add(c)
                self.identity_links.add(c)
            elif isinstance(self.ctx[c], dict) and self.ctx[c].get('@type') == '@id':
                self.url_fields.add(c)
                if self.ctx[c].get('identity', False):
                    self.identity_links.add(c)
            elif isinstance(self.ctx[c], dict) and self.ctx[c].get('@type') == '@vocab':
                self.url_fields.add(c)
                self.vocab_fields.add(c)
            if isinstance(self.ctx[c], dict) and self.ctx[c].get('noLinkCheck'):
                self.nolinkcheck.add(c)
            if isinstance(self.ctx[c], dict) and '@id' in self.ctx[c]:
                self.vocab[c] = self.ctx[c]['@id']
            elif isinstance(self.ctx[c], basestring):
                self.vocab[c] = self.ctx[c]
        for k, v in self.vocab.items():
            self.rvocab[self.expand_url(v, '', scoped=False)] = k
        _logger.debug('identifiers is %s', self.identifiers)
        _logger.debug('identity_links is %s', self.identity_links)
        _logger.debug('url_fields is %s', self.url_fields)
        _logger.debug('vocab_fields is %s', self.vocab_fields)
        _logger.debug('vocab is %s', self.vocab)

    def resolve_ref(self, ref: Any, base_url: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        base_url = base_url or 'file://%s/' % os.path.abspath('.')
        obj: Optional[Any] = None
        inc: bool = False
        merge: Optional[Any] = None
        if isinstance(ref, dict):
            obj = ref
            if '$import' in ref:
                if len(obj) == 1:
                    ref = obj['$import']
                    obj = None
                else:
                    raise ValueError("'$import' must be the only field in %s" % str(obj))
            elif '$include' in obj:
                if len(obj) == 1:
                    ref = obj['$include']
                    inc = True
                    obj = None
                else:
                    raise ValueError("'$include' must be the only field in %s" % str(obj))
            else:
                ref = None
                for identifier in self.identifiers:
                    if identifier in obj:
                        ref = obj[identifier]
                        break
                if not ref:
                    raise ValueError('Object `%s` does not have identifier field in %s' % (obj, self.identifiers))
        if not isinstance(ref, basestring):
            raise ValueError('Must be string: `%s`' % str(ref))
        url: str = self.expand_url(ref, base_url, scoped=obj is not None)
        if url in self.idx:
            if merge:
                obj = self.idx[url].copy()
            else:
                return (self.idx[url], {})
        if inc:
            return (self.fetch_text(url), {})
        if obj:
            for identifier in self.identifiers:
                obj[identifier] = url
            doc_url = url
        else:
            doc_url, frg = urlparse.urldefrag(url)
            if doc_url in self.idx:
                raise validate.ValidationException('Reference `#%s` not found in file `%s`.' % (frg, doc_url))
            obj = self.fetch(doc_url)
        obj, metadata = self.resolve_all(obj, doc_url)
        if url is not None:
            if url in self.idx:
                obj = self.idx[url]
            else:
                raise RuntimeError('Reference `%s` is not in the index.  Index contains:\n  %s' % (url, '\n  '.join(self.idx)))
        if '$graph' in obj:
            metadata = {k: v for k, v in obj.items() if k != '$graph'}
            obj = obj['$graph']
            return (obj, metadata)
        else:
            return (obj, metadata)

    def resolve_all(self, document: Any, base_url: str, file_base: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        loader: Loader = self
        metadata: Dict[str, Any] = {}
        if file_base is None:
            file_base = base_url
        if isinstance(document, dict):
            if '$import' in document or '$include' in document:
                return self.resolve_ref(document, file_base)
        elif isinstance(document, list):
            pass
        else:
            return (document, metadata)
        newctx: Optional[Loader] = None
        if isinstance(document, dict):
            if '$base' in document:
                base_url = document['$base']
            if '$profile' in document:
                if not newctx:
                    newctx = SubLoader(self)
                prof = self.fetch(document['$profile'])
                newctx.add_namespaces(document.get('$namespaces', {}))
                newctx.add_schemas(document.get('$schemas', []), document['$profile'])
            if '$namespaces' in document:
                if not newctx:
                    newctx = SubLoader(self)
                newctx.add_namespaces(document['$namespaces'])
            if '$schemas' in document:
                if not newctx:
                    newctx = SubLoader(self)
                newctx.add_schemas(document['$schemas'], file_base)
            if newctx:
                loader = newctx
            if '$graph' in document:
                metadata = {k: v for k, v in document.items() if k != '$graph'}
                document = document['$graph']
                metadata, _ = loader.resolve_all(metadata, base_url, file_base)
        if isinstance(document, dict):
            for identifer in loader.identity_links:
                if identifer in document:
                    if isinstance(document[identifer], basestring):
                        document[identifer] = loader.expand_url(document[identifer], base_url, scoped=True)
                        if document[identifer] not in loader.idx or isinstance(loader.idx[document[identifer]], basestring):
                            loader.idx[document[identifer]] = document
                        base_url = document[identifer]
                    elif isinstance(document[identifer], list):
                        for n, v in enumerate(document[identifer]):
                            document[identifer][n] = loader.expand_url(document[identifer][n], base_url, scoped=True)
                            if document[identifer][n] not in loader.idx:
                                loader.idx[document[identifer][n]] = document[identifer][n]
            for d in list(document.keys()):
                d2 = loader.expand_url(d, '', scoped=False, vocab_term=True)
                if d != d2:
                    document[d2] = document[d]
                    del document[d]
            for d in loader.url_fields:
                if d in document:
                    if isinstance(document[d], basestring):
                        document[d] = loader.expand_url(document[d], base_url, scoped=False, vocab_term=d in loader.vocab_fields)
                    elif isinstance(document[d], list):
                        document[d] = [loader.expand_url(url, base_url, scoped=False, vocab_term=d in loader.vocab_fields)
                                       if isinstance(url, basestring) else url for url in document[d]]
            try:
                for key, val in document.items():
                    document[key], _ = loader.resolve_all(val, base_url, file_base)
            except validate.ValidationException as v:
                _logger.debug('loader is %s', id(loader))
                raise validate.ValidationException('(%s) (%s) Validation error in field %s:\n%s' %
                                                     (id(loader), file_base, key, validate.indent(str(v))))
        elif isinstance(document, list):
            i = 0
            try:
                while i < len(document):
                    val = document[i]
                    if isinstance(val, dict) and '$import' in val:
                        l, _ = loader.resolve_ref(val, file_base)
                        if isinstance(l, list):
                            del document[i]
                            for item in aslist(l):
                                document.insert(i, item)
                                i += 1
                        else:
                            document[i] = l
                            i += 1
                    else:
                        document[i], _ = loader.resolve_all(val, base_url, file_base)
                        i += 1
            except validate.ValidationException as v:
                raise validate.ValidationException('(%s) (%s) Validation error in position %i:\n%s' %
                                                     (id(loader), file_base, i, validate.indent(str(v))))
            for identifer in loader.identity_links:
                if identifer in metadata:
                    if isinstance(metadata[identifer], basestring):
                        metadata[identifer] = loader.expand_url(metadata[identifer], base_url, scoped=True)
                        loader.idx[metadata[identifer]] = document
        return (document, metadata)

    def fetch_text(self, url: str) -> str:
        if url in self.cache:
            return self.cache[url]
        split = urlparse.urlsplit(url)
        scheme, path = (split.scheme, split.path)
        if scheme in ['http', 'https'] and requests:
            try:
                resp = requests.get(url)
                resp.raise_for_status()
            except Exception as e:
                raise RuntimeError(url, e)
            return resp.text
        elif scheme == 'file':
            try:
                with open(path) as fp:
                    return fp.read().decode('utf-8')
            except (OSError, IOError) as e:
                raise RuntimeError('Error reading %s %s' % (url, e))
        else:
            raise ValueError('Unsupported scheme in url: %s' % url)

    def fetch(self, url: str) -> Any:
        if url in self.idx:
            return self.idx[url]
        try:
            text_io = StringIO.StringIO(self.fetch_text(url))
            text_io.name = url
            result = yaml.load(text_io)
        except yaml.parser.ParserError as e:
            raise validate.ValidationException('Syntax error %s' % e)
        if isinstance(result, dict) and self.identifiers:
            for identifier in self.identifiers:
                if identifier not in result:
                    result[identifier] = url
                self.idx[self.expand_url(result[identifier], url)] = result
        else:
            self.idx[url] = result
        return result

    def check_file(self, fn: str) -> bool:
        if fn.startswith('file://'):
            u = urlparse.urlsplit(fn)
            return os.path.exists(u.path)
        else:
            return False

    def validate_link(self, field: str, link: Any) -> bool:
        if field in self.nolinkcheck:
            return True
        if isinstance(link, basestring):
            if field in self.vocab_fields:
                if link not in self.vocab and link not in self.idx and (link not in self.rvocab):
                    if not self.check_file(link):
                        raise validate.ValidationException('Field `%s` contains undefined reference to `%s`' %
                                                             (field, link))
            elif link not in self.idx and link not in self.rvocab:
                if not self.check_file(link):
                    raise validate.ValidationException('Field `%s` contains undefined reference to `%s`' %
                                                         (field, link))
        elif isinstance(link, list):
            errors: List[Exception] = []
            for i in link:
                try:
                    self.validate_link(field, i)
                except validate.ValidationException as v:
                    errors.append(v)
            if errors:
                raise validate.ValidationException('\n'.join([str(e) for e in errors]))
        elif isinstance(link, dict):
            self.validate_links(link)
        return True

    def getid(self, d: Any) -> Optional[str]:
        if isinstance(d, dict):
            for i in self.identifiers:
                if i in d:
                    if isinstance(d[i], basestring):
                        return d[i]
        return None

    def validate_links(self, document: Any) -> None:
        docid = self.getid(document)
        if docid is None:
            docid = ''
        errors: List[validate.ValidationException] = []
        if isinstance(document, list):
            iterator = enumerate(document)
        elif isinstance(document, dict):
            try:
                for d in self.url_fields:
                    if d not in self.identity_links and d in document:
                        self.validate_link(d, document[d])
            except validate.ValidationException as v:
                errors.append(v)
            iterator = document.iteritems()
        else:
            return
        for key, val in iterator:
            try:
                self.validate_links(val)
            except validate.ValidationException as v:
                if key not in self.nolinkcheck:
                    docid_inner = self.getid(val)
                    if docid_inner:
                        errors.append(validate.ValidationException('While checking object `%s`\n%s' %
                                                                     (docid_inner, validate.indent(str(v)))))
                    elif isinstance(key, basestring):
                        errors.append(validate.ValidationException('While checking field `%s`\n%s' %
                                                                     (key, validate.indent(str(v)))))
                    else:
                        errors.append(validate.ValidationException('While checking position %s\n%s' %
                                                                     (key, validate.indent(str(v)))))
        if errors:
            if len(errors) > 1:
                raise validate.ValidationException('\n'.join([str(e) for e in errors]))
            else:
                raise errors[0]
        return
