import os
import json
import hashlib
import logging
import collections
import requests
import urllib.parse
import yaml
import validate
import pprint
import io
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from aslist import aslist
import rdflib
from rdflib.namespace import RDF, RDFS, OWL

_logger = logging.getLogger('salad')


class NormDict(dict):
    def __init__(self, normalize: Callable[[str], str] = str) -> None:
        super(NormDict, self).__init__()
        self.normalize = normalize

    def __getitem__(self, key: str) -> Any:
        return super(NormDict, self).__getitem__(self.normalize(key))

    def __setitem__(self, key: str, value: Any) -> None:
        return super(NormDict, self).__setitem__(self.normalize(key), value)

    def __delitem__(self, key: str) -> None:
        return super(NormDict, self).__delitem__(self.normalize(key))

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


def SubLoader(loader: 'Loader') -> 'Loader':
    return Loader(
        loader.ctx,
        schemagraph=loader.graph,
        foreign_properties=loader.foreign_properties,
        idx=loader.idx,
        cache=loader.cache
    )


class Loader:
    def __init__(
        self,
        ctx: Dict[str, Any],
        schemagraph: Optional[rdflib.Graph] = None,
        foreign_properties: Optional[Set[str]] = None,
        idx: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None
    ) -> None:
        normalize: Callable[[str], str] = lambda url: urllib.parse.urlsplit(url).geturl()
        if idx is not None:
            self.idx: NormDict = idx  # type: ignore
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

    def expand_url(
        self,
        url: str,
        base_url: str,
        scoped: bool = False,
        vocab_term: bool = False
    ) -> str:
        if url in ('@id', '@type'):
            return url
        if vocab_term and url in self.vocab:
            return url
        if self.vocab and ':' in url:
            prefix = url.split(':')[0]
            if prefix in self.vocab:
                url = self.vocab[prefix] + url[len(prefix) + 1:]
        split = urllib.parse.urlsplit(url)
        if split.scheme or url.startswith('$(') or url.startswith('${'):
            pass
        elif scoped and not split.fragment:
            splitbase = urllib.parse.urlsplit(base_url)
            frg = ''
            if splitbase.fragment:
                frg = splitbase.fragment + '/' + split.path
            else:
                frg = split.path
            url = urllib.parse.urlunsplit((
                splitbase.scheme,
                splitbase.netloc,
                splitbase.path,
                splitbase.query,
                frg
            ))
        else:
            url = urllib.parse.urljoin(base_url, url)
        if vocab_term and url in self.rvocab:
            return self.rvocab[url]
        else:
            return url

    def _add_properties(self, s: rdflib.URIRef) -> None:
        for _, _, rng in self.graph.triples((s, RDFS.range, None)):
            rng_str = str(rng)
            literal = (rng_str.startswith('http://www.w3.org/2001/XMLSchema#') and
                       rng_str != 'http://www.w3.org/2001/XMLSchema#anyURI') or \
                      rng_str == 'http://www.w3.org/2000/01/rdf-schema#Literal'
            if not literal:
                self.url_fields.add(str(s))
        self.foreign_properties.add(str(s))

    def add_namespaces(self, ns: Dict[str, str]) -> None:
        self.vocab.update(ns)

    def add_schemas(self, ns: List[str], base_url: str) -> None:
        for sch in aslist(ns):
            self.graph.parse(urllib.parse.urljoin(base_url, sch))
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
        self.ctx.update({k: v for k, v in newcontext.items() if k != '@context'})
        _logger.debug('ctx is %s', self.ctx)
        for c, v in self.ctx.items():
            if v == '@id':
                self.identifiers.add(c)
                self.identity_links.add(c)
            elif isinstance(v, dict) and v.get('@type') == '@id':
                self.url_fields.add(c)
                if v.get('identity', False):
                    self.identity_links.add(c)
            elif isinstance(v, dict) and v.get('@type') == '@vocab':
                self.url_fields.add(c)
                self.vocab_fields.add(c)
            if isinstance(v, dict) and v.get('noLinkCheck'):
                self.nolinkcheck.add(c)
            if isinstance(v, dict) and '@id' in v:
                self.vocab[c] = v['@id']
            elif isinstance(v, str):
                self.vocab[c] = v
        for k, v in self.vocab.items():
            expanded = self.expand_url(v, '', scoped=False)
            self.rvocab[expanded] = k
        _logger.debug('identifiers is %s', self.identifiers)
        _logger.debug('identity_links is %s', self.identity_links)
        _logger.debug('url_fields is %s', self.url_fields)
        _logger.debug('vocab_fields is %s', self.vocab_fields)
        _logger.debug('vocab is %s', self.vocab)

    def resolve_ref(
        self,
        ref: Union[Dict[str, Any], str],
        base_url: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        base_url = base_url or f'file://{os.path.abspath(".")}/'
        obj: Optional[Dict[str, Any]] = None
        inc: bool = False
        merge: Optional[Any] = None
        if isinstance(ref, dict):
            obj = ref
            if '$import' in ref:
                if len(obj) == 1:
                    ref = obj['$import']
                    obj = None
                else:
                    raise ValueError(f"'$import' must be the only field in {str(obj)}")
            elif '$include' in obj:
                if len(obj) == 1:
                    ref = obj['$include']
                    inc = True
                    obj = None
                else:
                    raise ValueError(f"'$include' must be the only field in {str(obj)}")
            else:
                ref = None
                for identifier in self.identifiers:
                    if identifier in obj:
                        ref = obj[identifier]
                        break
                if not ref:
                    raise ValueError(f'Object `{obj}` does not have identifier field in {self.identifiers}')
        if not isinstance(ref, str):
            raise ValueError(f'Must be string: `{str(ref)}`')
        url = self.expand_url(ref, base_url, scoped=obj is not None)
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
            doc_url, frg = urllib.parse.urldefrag(url)
            if doc_url in self.idx:
                raise validate.ValidationException(f'Reference `#{frg}` not found in file `{doc_url}`.')
            obj = self.fetch(doc_url)
        obj, metadata = self.resolve_all(obj, doc_url)
        if url is not None:
            if url in self.idx:
                obj = self.idx[url]
            else:
                raise RuntimeError(f'Reference `{url}` is not in the index.  Index contains:\n  {"\n  ".join(self.idx)}')
        if '$graph' in obj:
            metadata = {k: v for k, v in obj.items() if k != '$graph'}
            obj = obj['$graph']
            return (obj, metadata)
        else:
            return (obj, metadata)

    def resolve_all(
        self,
        document: Any,
        base_url: str,
        file_base: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
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
            for identifier in loader.identity_links:
                if identifier in document:
                    if isinstance(document[identifier], str):
                        document[identifier] = loader.expand_url(document[identifier], base_url, scoped=True)
                        if (document[identifier] not in loader.idx or
                                isinstance(loader.idx[document[identifier]], str)):
                            loader.idx[document[identifier]] = document
                        base_url = document[identifier]
                    elif isinstance(document[identifier], list):
                        for n, v in enumerate(document[identifier]):
                            if isinstance(v, str):
                                document[identifier][n] = loader.expand_url(v, base_url, scoped=True)
                                if document[identifier][n] not in loader.idx:
                                    loader.idx[document[identifier][n]] = document[identifier][n]
            for d in list(document.keys()):
                d2 = loader.expand_url(d, '', scoped=False, vocab_term=True)
                if d != d2:
                    document[d2] = document[d]
                    del document[d]
            for d in loader.url_fields:
                if d in document:
                    if isinstance(document[d], str):
                        document[d] = loader.expand_url(
                            document[d],
                            base_url,
                            scoped=False,
                            vocab_term=d in loader.vocab_fields
                        )
                    elif isinstance(document[d], list):
                        document[d] = [
                            loader.expand_url(url, base_url, scoped=False, vocab_term=d in loader.vocab_fields)
                            if isinstance(url, str) else url
                            for url in document[d]
                        ]
            try:
                for key, val in document.items():
                    document[key], _ = loader.resolve_all(val, base_url, file_base)
            except validate.ValidationException as v:
                _logger.debug('loader is %s', id(loader))
                raise validate.ValidationException(
                    f'({id(loader)}) ({file_base}) Validation error in field {key}:\n{validate.indent(str(v))}'
                )
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
                raise validate.ValidationException(
                    f'({id(loader)}) ({file_base}) Validation error in position {i}:\n{validate.indent(str(v))}'
                )
            for identifier in loader.identity_links:
                if identifier in metadata:
                    if isinstance(metadata[identifier], str):
                        metadata[identifier] = loader.expand_url(metadata[identifier], base_url, scoped=True)
                        loader.idx[metadata[identifier]] = document
        return (document, metadata)

    def fetch_text(self, url: str) -> str:
        if url in self.cache:
            return self.cache[url]
        split = urllib.parse.urlsplit(url)
        scheme, path = split.scheme, split.path
        if scheme in ['http', 'https'] and requests:
            try:
                resp = requests.get(url)
                resp.raise_for_status()
            except Exception as e:
                raise RuntimeError(url, e)
            return resp.text
        elif scheme == 'file':
            try:
                with open(path, encoding='utf-8') as fp:
                    return fp.read()
            except (OSError, IOError) as e:
                raise RuntimeError(f'Error reading {url} {e}')
        else:
            raise ValueError(f'Unsupported scheme in url: {url}')

    def fetch(self, url: str) -> Any:
        if url in self.idx:
            return self.idx[url]
        try:
            text = io.StringIO(self.fetch_text(url))
            text.name = url
            result = yaml.safe_load(text)
        except yaml.parser.ParserError as e:
            raise validate.ValidationException(f'Syntax error {e}')
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
            u = urllib.parse.urlsplit(fn)
            return os.path.exists(u.path)
        else:
            return False

    def validate_link(self, field: str, link: Any) -> bool:
        if field in self.nolinkcheck:
            return True
        if isinstance(link, str):
            if field in self.vocab_fields:
                if (link not in self.vocab and
                        link not in self.idx and
                        link not in self.rvocab):
                    if not self.check_file(link):
                        raise validate.ValidationException(
                            f'Field `{field}` contains undefined reference to `{link}`'
                        )
            elif link not in self.idx and link not in self.rvocab:
                if not self.check_file(link):
                    raise validate.ValidationException(
                        f'Field `{field}` contains undefined reference to `{link}`'
                    )
        elif isinstance(link, list):
            errors: List[validate.ValidationException] = []
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
                    if isinstance(d[i], str):
                        return d[i]
        return None

    def validate_links(self, document: Any) -> None:
        docid = self.getid(document)
        if docid is None:
            docid = ''
        errors: List[validate.ValidationException] = []
        if isinstance(document, list):
            iterator: Any = enumerate(document)
        elif isinstance(document, dict):
            try:
                for d in self.url_fields:
                    if d not in self.identity_links and d in document:
                        self.validate_link(d, document[d])
            except validate.ValidationException as v:
                errors.append(v)
            iterator = document.items()
        else:
            return
        for key, val in iterator:
            try:
                self.validate_links(val)
            except validate.ValidationException as v:
                if key not in self.nolinkcheck:
                    docid_val = self.getid(val)
                    if docid_val:
                        errors.append(validate.ValidationException(
                            f'While checking object `{docid_val}`\n{validate.indent(str(v))}'
                        ))
                    elif isinstance(key, str):
                        errors.append(validate.ValidationException(
                            f'While checking field `{key}`\n{validate.indent(str(v))}'
                        ))
                    else:
                        errors.append(validate.ValidationException(
                            f'While checking position {key}\n{validate.indent(str(v))}'
                        ))
        if errors:
            if len(errors) > 1:
                raise validate.ValidationException('\n'.join([str(e) for e in errors]))
            else:
                raise errors[0]
        return
