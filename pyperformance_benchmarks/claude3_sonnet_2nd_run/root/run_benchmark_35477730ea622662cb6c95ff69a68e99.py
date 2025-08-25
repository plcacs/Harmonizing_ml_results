"Benchmark script for testing the performance of ElementTree.\n\nThis is intended to support Unladen Swallow's pyperf.py.\n\nThis will have ElementTree, cElementTree and lxml (if available)\nparse a generated XML file, search it, create new XML trees from\nit and serialise the result.\n"
import io
import os
import sys
import tempfile
from collections import defaultdict
import pyperf
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast
from types import ModuleType
import argparse

__author__ = 'stefan_ml@behnel.de (Stefan Behnel)'
FALLBACK_ETMODULE = 'xml.etree.ElementTree'

def build_xml_tree(etree: ModuleType) -> Any:
    SubElement = etree.SubElement
    root = etree.Element('root')
    for c in range(50):
        child = SubElement(root, ('child-%d' % c), tag_type='child')
        for i in range(100):
            SubElement(child, 'subchild').text = ('LEAF-%d-%d' % (c, i))
    deep = SubElement(root, 'deepchildren', tag_type='deepchild')
    for i in range(50):
        deep = SubElement(deep, 'deepchild')
    SubElement(deep, 'deepleaf', tag_type='leaf').text = 'LEAF'
    nb_elems = sum((1 for elem in root.iter()))
    root.set('nb-elems', str(nb_elems))
    return root

def process(etree: ModuleType, xml_root: Optional[Any] = None) -> bytes:
    SubElement = etree.SubElement
    if (xml_root is not None):
        root = xml_root
    else:
        root = build_xml_tree(etree)
    found = sum(((child.find('.//deepleaf') is not None) for child in root))
    if (found != 1):
        raise RuntimeError('find() failed')
    text = 'LEAF-5-99'
    found = any((1 for child in root for el in child.iterfind('.//subchild') if (el.text == text)))
    if (not found):
        raise RuntimeError('iterfind() failed')
    found = sum(((el.text == 'LEAF') for el in root.findall('.//deepchild/deepleaf')))
    if (found != 1):
        raise RuntimeError('findall() failed')
    dest = etree.Element('root2')
    target = SubElement(dest, 'result-1')
    for child in root:
        SubElement(target, child.tag).text = str(len(child))
    if (len(target) != len(root)):
        raise RuntimeError('transform #1 failed')
    target = SubElement(dest, 'result-2')
    for child in root.iterfind('.//subchild'):
        SubElement(target, child.tag, attr=child.text).text = 'found'
    if ((len(target) < len(root)) or (not all(((el.text == 'found') for el in target.iterfind('subchild'))))):
        raise RuntimeError('transform #2 failed')
    orig_len = len(root[0])
    new_root = root.makeelement('parent', {})
    new_root[:] = root[0]
    el = root[0]
    del el[:]
    for child in new_root:
        if (child is not None):
            el.append(child)
    if (len(el) != orig_len):
        raise RuntimeError('child moving failed')
    d: Dict[str, List[Any]] = defaultdict(list)
    for child in root:
        tags = d[child.get('tag_type')]
        for sub in child.iter():
            tags.append(sub)
    check_dict = dict(((n, iter(ch)) for (n, ch) in d.items()))
    target = SubElement(dest, 'transform-2')
    for child in root:
        tags = check_dict[child.get('tag_type')]
        for sub in child.iter():
            if (sub is not next(tags)):
                raise RuntimeError('tree iteration consistency check failed')
            SubElement(target, sub.tag).text = 'worked'
    orig = etree.tostring(root, encoding='utf8')
    result = etree.tostring(dest, encoding='utf8')
    if ((len(result) < len(orig)) or (b'worked' not in result) or (b'>LEAF<' not in orig)):
        raise RuntimeError('serialisation probability check failed')
    return result

def bench_iterparse(etree: ModuleType, xml_file: str, xml_data: bytes, xml_root: Any) -> None:
    for _ in range(10):
        it = etree.iterparse(xml_file, ('start', 'end'))
        events1 = [(event, elem.tag) for (event, elem) in it]
        it = etree.iterparse(io.BytesIO(xml_data), ('start', 'end'))
        events2 = [(event, elem.tag) for (event, elem) in it]
    nb_elems = int(xml_root.get('nb-elems'))
    if ((len(events1) != (2 * nb_elems)) or (events1 != events2)):
        raise RuntimeError(('parsing check failed:\n%r\n%r\n' % (len(events1), events2[:10])))

def bench_parse(etree: ModuleType, xml_file: str, xml_data: bytes, xml_root: Any) -> None:
    for _ in range(30):
        root1 = etree.parse(xml_file).getroot()
        root2 = etree.fromstring(xml_data)
    result1 = etree.tostring(root1)
    result2 = etree.tostring(root2)
    if (result1 != result2):
        raise RuntimeError('serialisation check failed')

def bench_process(etree: ModuleType, xml_file: str, xml_data: bytes, xml_root: Any) -> None:
    result1 = process(etree, xml_root=xml_root)
    result2 = process(etree, xml_root=xml_root)
    if ((result1 != result2) or (b'>found<' not in result2)):
        raise RuntimeError('serialisation check failed')

def bench_generate(etree: ModuleType, xml_file: str, xml_data: bytes, xml_root: Any) -> None:
    output: List[bytes] = []
    for _ in range(10):
        root = build_xml_tree(etree)
        output.append(etree.tostring(root))
    length = None
    for xml in output:
        if (length is None):
            length = len(xml)
        elif (length != len(xml)):
            raise RuntimeError('inconsistent output detected')
        if (b'>LEAF<' not in xml):
            raise RuntimeError('unexpected output detected')

def bench_etree(iterations: int, etree: ModuleType, bench_func: Callable[[ModuleType, str, bytes, Any], None]) -> float:
    xml_root = build_xml_tree(etree)
    xml_data = etree.tostring(xml_root)
    (tf, file_path) = tempfile.mkstemp()
    try:
        etree.ElementTree(xml_root).write(file_path)
        t0 = pyperf.perf_counter()
        for _ in range(iterations):
            bench_func(etree, file_path, xml_data, xml_root)
        dt = (pyperf.perf_counter() - t0)
    finally:
        try:
            os.close(tf)
        except EnvironmentError:
            pass
        try:
            os.unlink(file_path)
        except EnvironmentError:
            pass
    return dt
BENCHMARKS = 'parse iterparse generate process'.split()

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(('--etree-module', args.etree_module))
    if args.no_accelerator:
        cmd.append('--no-accelerator')
    if args.benchmark:
        cmd.append(args.benchmark)
if (__name__ == '__main__'):
    default_etmodule = 'xml.etree.ElementTree'
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Test the performance of ElementTree XML processing.'
    parser = runner.argparser
    parser.add_argument('--etree-module', default=None, metavar='FQMN', help=("Select an ElementTree module to use (fully qualified module name). Default is '%s'" % default_etmodule))
    parser.add_argument('--no-accelerator', action='store_true', default=False, help="Disable the '_elementree' accelerator module for ElementTree.")
    parser.add_argument('benchmark', nargs='?', choices=BENCHMARKS)
    options = runner.parse_args()
    if (not options.etree_module):
        if options.no_accelerator:
            options.etree_module = FALLBACK_ETMODULE
        else:
            options.etree_module = default_etmodule
    if options.no_accelerator:
        sys.modules['_elementtree'] = None
        import xml.etree.ElementTree as et
        if (et.SubElement.__module__ != 'xml.etree.ElementTree'):
            raise RuntimeError('Unexpected C accelerator for ElementTree')
    try:
        from importlib import import_module
    except ImportError:

        def import_module(module_name: str) -> ModuleType:
            __import__(module_name)
            return sys.modules[module_name]
    try:
        etree_module = import_module(options.etree_module)
    except ImportError:
        if (options.etree_module != default_etmodule):
            raise
        etree_module = import_module(FALLBACK_ETMODULE)
    module = etree_module.__name__
    if hasattr(etree_module, '_Element_Py'):
        accelerator = (etree_module.Element is not etree_module._Element_Py)
    elif options.no_accelerator:
        accelerator = False
    else:
        accelerator = True
    if accelerator:
        module += ' (with C accelerator)'
    else:
        module += ' (pure Python)'
    runner.metadata['elementtree_module'] = module
    if options.benchmark:
        benchmarks = (options.benchmark,)
    else:
        benchmarks = BENCHMARKS
    for bench in benchmarks:
        if accelerator:
            name = ('xml_etree_%s' % bench)
        else:
            name = ('xml_etree_pure_python_%s' % bench)
        bench_func = globals()[('bench_%s' % bench)]
        runner.bench_time_func(name, bench_etree, etree_module, bench_func)
