#!/usr/bin/env python3
"""
捕获类似curses键盘输入流,生成指令流
"""
import curses
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
from .config import Config

ERASE_SPEED: int = 5
__all__ = ['cmd_parser', 'parse_keylist', 'coroutine', 'erase_coroutine']
KEY_MAP: dict[str, str] = Config().get('keymap')

def coroutine(func: Callable[..., Generator[Any, Any, Any]]) -> Callable[..., Generator[Any, Any, Any]]:
    @wraps(func)
    def primer(*args: Any, **kwargs: Any) -> Generator[Any, Any, Any]:
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return primer

def _cmd_parser() -> Generator[int, int, List[int]]:
    """
    A generator receive key value typed by user return constant keylist.
    输入键盘输入流,输出指令流,以curses默认-1为信号终止.
    """
    pre_key: int = -1
    keylist: List[int] = []
    while True:
        key: int = (yield)
        if key > 0 and pre_key == -1:
            keylist.append(key)
        elif key > 0 and pre_key > 0:
            keylist.append(key)
        elif curses.keyname(key).decode('utf-8') in KEY_MAP.values() and pre_key > 0:
            keylist.append(key)
            return keylist
        pre_key = key

def cmd_parser(results: List[int]) -> Generator[List[int], int, None]:
    """
    A generator manager which can catch StopIteration and start a new Generator.
    生成器管理对象,可以优雅地屏蔽生成器的终止信号,并重启生成器
    """
    while True:
        results.clear()
        results += (yield from _cmd_parser())
        yield results

def _erase_coroutine() -> Generator[int, int, List[int]]:
    keylist: List[int] = []
    while True:
        key: int = (yield)
        keylist.append(key)
        if len(set(keylist)) > 1:
            return keylist
        elif len(keylist) >= ERASE_SPEED * 2:
            return keylist

def erase_coroutine(erase_cmd_list: List[int]) -> Generator[List[int], int, None]:
    while True:
        erase_cmd_list.clear()
        erase_cmd_list += (yield from _erase_coroutine())
        yield erase_cmd_list

def parse_keylist(keylist: List[int]) -> Optional[Union[int, Tuple[int, int]]]:
    """
    '2' '3' '4' 'j'  ----> 234 j
    supoort keys  [  ]   j  k  <KEY_UP> <KEY_DOWN>
    """
    keylist = deepcopy(keylist)
    if keylist == []:
        return None
    if set(keylist) | {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])} == {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])}:
        delta_key: int = keylist.count(ord(KEY_MAP['nextSong'])) - keylist.count(ord(KEY_MAP['prevSong']))
        if delta_key < 0:
            return (-delta_key, ord(KEY_MAP['prevSong']))
        return (delta_key, ord(KEY_MAP['nextSong']))
    tail_cmd: int = keylist.pop()
    if tail_cmd in range(48, 58) and set(keylist) | set(range(48, 58)) == set(range(48, 58)):
        return int(''.join([chr(i) for i in keylist] + [chr(tail_cmd)]))
    if len(keylist) == 0:
        return (0, tail_cmd)
    if tail_cmd in (ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong']), ord(KEY_MAP['down']), ord(KEY_MAP['up'])) and max(keylist) <= 57 and (min(keylist) >= 48):
        return (int(''.join([chr(i) for i in keylist])), tail_cmd)
    return None

def main(data: List[int]) -> None:
    """
    tset code
    测试代码
    """
    results: List[int] = []
    group: Generator[List[int], int, None] = cmd_parser(results)
    next(group)
    for i in data:
        group.send(i)
    group.send(-1)
    print(results)
    next(group)
    for i in data:
        group.send(i)
    group.send(-1)
    print(results)
    x: Generator[int, int, List[int]] = _cmd_parser()
    print('-----------')
    print(x.send(None))
    print(x.send(1))
    print(x.send(2))
    print(x.send(3))
    print(x.send(3))
    print(x.send(3))
    try:
        print(x.send(-1))
    except Exception as e:
        print(e.value)

if __name__ == '__main__':
    main(list(range(1, 12)[::-1]))
