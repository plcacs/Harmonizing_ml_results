from time import time
from typing import Generator, List, Any

class scrollstring(object):

    def __init__(self, content: str, START: float) -> None:
        self.content: str = content
        self.display: str = content
        self.START: float = START // 1
        self.update()

    def update(self) -> None:
        self.display = self.content
        curTime: float = time() // 1
        offset: int = max(int((curTime - self.START) % len(self.content)) - 1, 0)
        while offset > 0:
            if self.display[0] > chr(127):
                offset -= 1
                self.display = self.display[1:] + self.display[:1]
            else:
                offset -= 1
                self.display = self.display[2:] + self.display[:2]

    def __repr__(self) -> str:
        return self.display

def truelen(string: str) -> int:
    """
    It appears one Asian character takes two spots, but __len__
    counts it as three, so this function counts the dispalyed
    length of the string.

    >>> truelen('abc')
    3
    >>> truelen('你好')
    4
    >>> truelen('1二3')
    4
    >>> truelen('')
    0
    """
    return len(string) + sum((1 for c in string if c > chr(127)))

def truelen_cut(string: str, length: int) -> str:
    current_length: int = 0
    current_pos: int = 0
    for c in string:
        current_length += 2 if c > chr(127) else 1
        if current_length > length:
            return string[:current_pos]
        current_pos += 1
    return string
