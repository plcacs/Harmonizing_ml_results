'\nArtificial, floating point-heavy benchmark originally used by Factor.\n'
import pyperf
from math import sin, cos, sqrt
from typing import List

POINTS: int = 100000

class Point(object):
    __slots__ = ('x', 'y', 'z')

    def __init__(self, i: int) -> None:
        self.x: float = x = sin(i)
        self.y: float = (cos(i) * 3)
        self.z: float = ((x * x) / 2)

    def __repr__(self) -> str:
        return ('<Point: x=%s, y=%s, z=%s>' % (self.x, self.y, self.z))

    def normalize(self) -> None:
        x = self.x
        y = self.y
        z = self.z
        norm = sqrt((((x * x) + (y * y)) + (z * z)))
        self.x /= norm
        self.y /= norm
        self.z /= norm

    def maximize(self, other: 'Point') -> 'Point':
        self.x = (self.x if (self.x > other.x) else other.x)
        self.y = (self.y if (self.y > other.y) else other.y)
        self.z = (self.z if (self.z > other.z) else other.z)
        return self

def maximize(points: List[Point]) -> Point:
    next = points[0]
    for p in points[1:]:
        next = next.maximize(p)
    return next

def benchmark(n: int) -> Point:
    points: List[Point] = ([None] * n)
    for i in range(n):
        points[i] = Point(i)
    for p in points:
        p.normalize()
    return maximize(points)

if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Float benchmark'
    points = POINTS
    runner.bench_func('float', benchmark, points)
