import json
import random
import sys
import pyperf
from typing import Any, Dict, List, Tuple

DICT: Dict[str, Any] = {
    'ads_flags': 0,
    'age': 18,
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
    [265867233, 265868503, 265252341, 265243910, 265879514, 266219766, 266021701, 265843726, 265592821, 265246784, 265853180, 45526486, 265463699, 265848143, 265863062, 265392591, 265877490, 265823665, 265828884, 265753032],
    60
)

def mutate_dict(orig_dict: Dict[str, Any], random_source: random.Random) -> Dict[str, Any]:
    new_dict = dict(orig_dict)
    for (key, value) in new_dict.items():
        rand_val = (random_source.random() * sys.maxsize)
        if isinstance(key, (int, bytes, str)):
            new_dict[key] = type(key)(rand_val)
    return new_dict

random_source = random.Random(5)
DICT_GROUP: List[Dict[str, Any]] = [mutate_dict(DICT, random_source) for _ in range(3)]

def bench_json_loads(objs: Tuple[str, str, str]) -> None:
    for obj in objs:
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)
        json.loads(obj)

if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark json.loads()'
    json_dict: str = json.dumps(DICT)
    json_tuple: str = json.dumps(TUPLE)
    json_dict_group: str = json.dumps(DICT_GROUP)
    objs: Tuple[str, str, str] = (json_dict, json_tuple, json_dict_group)
    runner.bench_func('json_loads', bench_json_loads, objs, inner_loops=20)
