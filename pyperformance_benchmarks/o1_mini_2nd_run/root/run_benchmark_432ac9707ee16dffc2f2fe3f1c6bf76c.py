import datetime
import random
import sys
import pyperf
from typing import Any, Callable, Dict, List, Tuple, Union

IS_PYPY: bool = (pyperf.python_implementation() == 'pypy')
__author__: str = 'collinwinter@google.com (Collin Winter)'

DICT: Dict[str, Any] = {
    'ads_flags': 0,
    'age': 18,
    'birthday': datetime.date(1980, 5, 7),
    'bulletin_count': 0,
    'comment_count': 0,
    'country': 'BR',
    'encrypted_id': 'G9urXXAJwjE',
    'favorite_count': 9,
    'first_name': '',
    'flags': 412317970704,
    'friend_count': 0,
    'gender': 'm',
    'gender_for_display': 'Male',
    'id': 302935349,
    'is_custom_profile_icon': 0,
    'last_name': '',
    'locale_preference': 'pt_BR',
    'member': 0,
    'tags': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
    'profile_foo_id': 827119638,
    'secure_encrypted_id': 'Z_xxx2dYx3t4YAdnmfgyKw',
    'session_number': 2,
    'signup_id': '201-19225-223',
    'status': 'A',
    'theme': 1,
    'time_created': 1225237014,
    'time_updated': 1233134493,
    'unread_message_count': 0,
    'user_group': '0',
    'username': 'collinwinter',
    'play_count': 9,
    'view_count': 7,
    'zip': ''
}

TUPLE: Tuple[List[int], int] = (
    [265867233, 265868503, 265252341, 265243910, 265879514, 266219766, 266021701, 265843726,
     265592821, 265246784, 265853180, 45526486, 265463699, 265848143, 265863062, 265392591,
     265877490, 265823665, 265828884, 265753032],
    60
)

def mutate_dict(orig_dict: Dict[Union[int, bytes, str], Any], random_source: random.Random) -> Dict[Union[int, bytes, str], Any]:
    new_dict: Dict[Union[int, bytes, str], Any] = dict(orig_dict)
    for key, value in new_dict.items():
        rand_val: float = random_source.random() * sys.maxsize
        if isinstance(key, (int, bytes, str)):
            new_dict[key] = type(key)(rand_val)
    return new_dict

random_source: random.Random = random.Random(5)
DICT_GROUP: List[Dict[Union[int, bytes, str], Any]] = [mutate_dict(DICT, random_source) for _ in range(3)]

def bench_pickle(loops: int, pickle_module: Any, options: Any) -> float:
    range_it = range(loops)
    dumps = pickle_module.dumps
    objs: Tuple[Any, ...] = (DICT, TUPLE, DICT_GROUP)
    protocol: Union[int, None] = options.protocol
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        for obj in objs:
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
            dumps(obj, protocol)
    return pyperf.perf_counter() - t0

def bench_unpickle(loops: int, pickle_module: Any, options: Any) -> float:
    pickled_dict: bytes = pickle_module.dumps(DICT, options.protocol)
    pickled_tuple: bytes = pickle_module.dumps(TUPLE, options.protocol)
    pickled_dict_group: bytes = pickle_module.dumps(DICT_GROUP, options.protocol)
    range_it = range(loops)
    loads = pickle_module.loads
    objs: Tuple[bytes, ...] = (pickled_dict, pickled_tuple, pickled_dict_group)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        for obj in objs:
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
            loads(obj)
    return pyperf.perf_counter() - t0

LIST: List[List[int]] = [[list(range(10)), list(range(10))] for _ in range(10)]

def bench_pickle_list(loops: int, pickle_module: Any, options: Any) -> float:
    range_it = range(loops)
    dumps = pickle_module.dumps
    obj: List[List[int]] = LIST
    protocol: Union[int, None] = options.protocol
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
        dumps(obj, protocol)
    return pyperf.perf_counter() - t0

def bench_unpickle_list(loops: int, pickle_module: Any, options: Any) -> float:
    pickled_list: bytes = pickle_module.dumps(LIST, options.protocol)
    range_it = range(loops)
    loads = pickle_module.loads
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
        loads(pickled_list)
    return pyperf.perf_counter() - t0

MICRO_DICT: Dict[int, Dict[int, None]] = dict(((key, dict.fromkeys(range(10))) for key in range(100)))

def bench_pickle_dict(loops: int, pickle_module: Any, options: Any) -> float:
    range_it = range(loops)
    protocol: Union[int, None] = options.protocol
    obj: Dict[int, Dict[int, None]] = MICRO_DICT
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        pickle_module.dumps(obj, protocol)
        pickle_module.dumps(obj, protocol)
        pickle_module.dumps(obj, protocol)
        pickle_module.dumps(obj, protocol)
        pickle_module.dumps(obj, protocol)
    return pyperf.perf_counter() - t0

BENCHMARKS: Dict[str, Tuple[Callable[[int, Any, Any], float], int]] = {
    'pickle': (bench_pickle, 20),
    'unpickle': (bench_unpickle, 20),
    'pickle_list': (bench_pickle_list, 10),
    'unpickle_list': (bench_unpickle_list, 10),
    'pickle_dict': (bench_pickle_dict, 5)
}

def is_accelerated_module(module: Any) -> bool:
    return getattr(pickle.Pickler, '__module__', '<jython>') != 'pickle'

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    if args.pure_python:
        cmd.append('--pure-python')
    cmd.extend(('--protocol', str(args.protocol)))
    cmd.append(args.benchmark)

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Test the performance of pickling.'
    parser = runner.argparser
    parser.add_argument('--pure-python', action='store_true', help='Use the C version of pickle.')
    parser.add_argument('--protocol', action='store', default=None, type=int, help='Which protocol to use (default: highest protocol).')
    benchmarks: List[str] = sorted(BENCHMARKS)
    parser.add_argument('benchmark', choices=benchmarks)
    options = runner.parse_args()
    benchmark, inner_loops = BENCHMARKS[options.benchmark]
    name: str = options.benchmark
    if options.pure_python:
        name += '_pure_python'
    if not (options.pure_python or IS_PYPY):
        import pickle
        if not is_accelerated_module(pickle):
            raise RuntimeError('Missing C accelerators for pickle')
    else:
        sys.modules['_pickle'] = None
        import pickle
        if is_accelerated_module(pickle):
            raise RuntimeError('Unexpected C accelerators for pickle')
    if options.protocol is None:
        options.protocol = pickle.HIGHEST_PROTOCOL
    runner.metadata['pickle_protocol'] = str(options.protocol)
    runner.metadata['pickle_module'] = pickle.__name__
    runner.bench_time_func(name, benchmark, pickle, options, inner_loops=inner_loops)
