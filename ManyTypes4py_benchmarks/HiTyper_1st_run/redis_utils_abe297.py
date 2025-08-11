import os
import re
import secrets
from collections.abc import Mapping
from typing import Any
import orjson
import redis
from django.conf import settings
MAX_KEY_LENGTH = 1024

class ZulipRedisError(Exception):
    pass

class ZulipRedisKeyTooLongError(ZulipRedisError):
    pass

class ZulipRedisKeyOfWrongFormatError(ZulipRedisError):
    pass

def get_redis_client():
    return redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD, db=0, decode_responses=False)

def put_dict_in_redis(redis_client: Union[redis.StrictRedis, str, int], key_format: str, data_to_store: Union[typing.Mapping, dict[str, typing.Any], dict], expiration_seconds: Union[int, str, None, float], token_length: int=64, token: Union[None, int, bytes]=None) -> str:
    key_length = len(key_format) - len('{token}') + token_length
    if key_length > MAX_KEY_LENGTH:
        raise ZulipRedisKeyTooLongError(f'Requested key too long in put_dict_in_redis. Key format: {key_format}, token length: {token_length}')
    if token is None:
        token = secrets.token_hex(token_length // 2)
    key = key_format.format(token=token)
    with redis_client.pipeline() as pipeline:
        pipeline.set(key, orjson.dumps(data_to_store))
        pipeline.expire(key, expiration_seconds)
        pipeline.execute()
    return key

def get_dict_from_redis(redis_client: redis.StrictRedis, key_format: Union[str, bytes, None], key: Union[str, bytes]) -> None:
    if len(key) > MAX_KEY_LENGTH:
        raise ZulipRedisKeyTooLongError(f'Requested key too long in get_dict_from_redis: {key}')
    validate_key_fits_format(key, key_format)
    data = redis_client.get(key)
    if data is None:
        return None
    return orjson.loads(data)

def validate_key_fits_format(key: Union[str, int], key_format: str) -> None:
    assert '{token}' in key_format
    regex = key_format.format(token='[a-zA-Z0-9]+')
    if not re.fullmatch(regex, key):
        raise ZulipRedisKeyOfWrongFormatError(f'{key} does not match format {key_format}')
REDIS_KEY_PREFIX = ''

def bounce_redis_key_prefix_for_testing(test_name: str) -> None:
    global REDIS_KEY_PREFIX
    REDIS_KEY_PREFIX = test_name + ':' + str(os.getpid()) + ':'