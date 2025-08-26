'\nArtificial, floating point-heavy benchmark originally used by Factor.\n'
import pyperf
from math import sin, cos, sqrt
from typing import List

POINTS: int = 100000

class Point:
    __slots__ = ('x', 'y', 'z')

    x: float
    y: float
    z: float

    def __init__(self, i: int) -> None:
        self.x = sin(i)
        self.y = cos(i) * 3
        self.z = (self.x * self.x) / 2

    def __repr__(self) -> str:
        return f'<Point: x={self.x}, y={self.y}, z={self.z}>'

    def normalize(self) -> None:
        norm: float = sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))
        self.x /= norm
        self.y /= norm
        self.z /= norm

    def maximize(self, other: 'Point') -> 'Point':
        self.x = self.x if self.x > other.x else other.x
        self.y = self.y if self.y > other.y else other.y
        self.z = self.z if self.z > other.z else other.z
        return self

def maximize(points: List[Point]) -> Point:
    next_point: Point = points[0]
    for p in points[1:]:
        next_point = next_point.maximize(p)
    return next_point

def benchmark(n: int) -> Point:
    points: List[Point] = [None] * n  # type: ignore
    for i in range(n):
        points[i] = Point(i)
    for p in points:
        p.normalize()
    return maximize(points)

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Float benchmark'
    points: int = POINTS
    runner.bench_func('float', benchmark, points)
