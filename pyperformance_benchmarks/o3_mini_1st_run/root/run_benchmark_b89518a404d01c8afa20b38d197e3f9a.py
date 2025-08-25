#!/usr/bin/env python3
"""
This file contains definitions for a simple raytracer.
Copyright Callum and Tony Garnock-Jones, 2008.

This file may be freely redistributed under the MIT license,
http://www.opensource.org/licenses/mit-license.php

From http://www.lshift.net/blog/2008/10/29/toy-raytracer-in-python
"""

import array
import math
import pyperf
from typing import Optional, List, Tuple, Union, Any
from argparse import Namespace

DEFAULT_WIDTH: int = 100
DEFAULT_HEIGHT: int = 100
EPSILON: float = 1e-05


class Vector:
    def __init__(self, initx: float, inity: float, initz: float) -> None:
        self.x: float = initx
        self.y: float = inity
        self.z: float = initz

    def __str__(self) -> str:
        return '(%s,%s,%s)' % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return 'Vector(%s,%s,%s)' % (self.x, self.y, self.z)

    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    def __add__(self, other: Union['Vector', 'Point']) -> Union['Vector', 'Point']:
        if other.isPoint():
            return Point(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector') -> 'Vector':
        other.mustBeVector()
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> 'Vector':
        return Vector(factor * self.x, factor * self.y, factor * self.z)

    def dot(self, other: 'Vector') -> float:
        other.mustBeVector()
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def cross(self, other: 'Vector') -> 'Vector':
        other.mustBeVector()
        return Vector((self.y * other.z) - (self.z * other.y),
                      (self.z * other.x) - (self.x * other.z),
                      (self.x * other.y) - (self.y * other.x))

    def normalized(self) -> 'Vector':
        return self.scale(1.0 / self.magnitude())

    def negated(self) -> 'Vector':
        return self.scale(-1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

    def isVector(self) -> bool:
        return True

    def isPoint(self) -> bool:
        return False

    def mustBeVector(self) -> 'Vector':
        return self

    def mustBePoint(self) -> None:
        raise Exception('Vectors are not points!')

    def reflectThrough(self, normal: 'Vector') -> 'Vector':
        d: Vector = normal.scale(self.dot(normal))
        return self - d.scale(2)


Vector.ZERO: Vector = Vector(0, 0, 0)
Vector.RIGHT: Vector = Vector(1, 0, 0)
Vector.UP: Vector = Vector(0, 1, 0)
Vector.OUT: Vector = Vector(0, 0, 1)


class Point:
    def __init__(self, initx: float, inity: float, initz: float) -> None:
        self.x: float = initx
        self.y: float = inity
        self.z: float = initz

    def __str__(self) -> str:
        return '(%s,%s,%s)' % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return 'Point(%s,%s,%s)' % (self.x, self.y, self.z)

    def __add__(self, other: 'Vector') -> 'Point':
        other.mustBeVector()
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Union['Point', 'Vector']) -> Union['Vector', 'Point']:
        if other.isPoint():
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def isVector(self) -> bool:
        return False

    def isPoint(self) -> bool:
        return True

    def mustBeVector(self) -> None:
        raise Exception('Points are not vectors!')

    def mustBePoint(self) -> 'Point':
        return self


class Sphere:
    def __init__(self, centre: Point, radius: float) -> None:
        centre.mustBePoint()
        self.centre: Point = centre
        self.radius: float = radius

    def __repr__(self) -> str:
        return 'Sphere(%s,%s)' % (repr(self.centre), self.radius)

    def intersectionTime(self, ray: 'Ray') -> Optional[float]:
        cp: Vector = self.centre - ray.point  # returns Vector
        v: float = cp.dot(ray.vector)
        discriminant: float = (self.radius * self.radius) - (cp.dot(cp) - (v * v))
        if discriminant < 0:
            return None
        else:
            return v - math.sqrt(discriminant)

    def normalAt(self, p: Point) -> Vector:
        return (p - self.centre).normalized()


class Halfspace:
    def __init__(self, point: Point, normal: Vector) -> None:
        self.point: Point = point
        self.normal: Vector = normal.normalized()

    def __repr__(self) -> str:
        return 'Halfspace(%s,%s)' % (repr(self.point), repr(self.normal))

    def intersectionTime(self, ray: 'Ray') -> Optional[float]:
        v: float = ray.vector.dot(self.normal)
        if v:
            return 1 / (-v)
        else:
            return None

    def normalAt(self, p: Point) -> Vector:
        return self.normal


class Ray:
    def __init__(self, point: Point, vector: Vector) -> None:
        self.point: Point = point
        self.vector: Vector = vector.normalized()

    def __repr__(self) -> str:
        return 'Ray(%s,%s)' % (repr(self.point), repr(self.vector))

    def pointAtTime(self, t: float) -> Point:
        return self.point + self.vector.scale(t)


Point.ZERO: Point = Point(0, 0, 0)


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.bytes: array.array = array.array('B', [0] * ((width * height) * 3))
        for i in range(width * height):
            self.bytes[(i * 3) + 2] = 255
        self.width: int = width
        self.height: int = height

    def plot(self, x: int, y: int, r: float, g: float, b: float) -> None:
        i: int = (((self.height - y) - 1) * self.width + x) * 3
        self.bytes[i] = max(0, min(255, int(r * 255)))
        self.bytes[i + 1] = max(0, min(255, int(g * 255)))
        self.bytes[i + 2] = max(0, min(255, int(b * 255)))

    def write_ppm(self, filename: str) -> None:
        header: str = 'P6 %d %d 255\n' % (self.width, self.height)
        with open(filename, 'wb') as fp:
            fp.write(header.encode('ascii'))
            fp.write(self.bytes.tobytes())


def firstIntersection(intersections: List[Tuple[Any, Optional[float], Any]]) -> Optional[Tuple[Any, float, Any]]:
    result: Optional[Tuple[Any, float, Any]] = None
    for i in intersections:
        candidateT: Optional[float] = i[1]
        if candidateT is not None and candidateT > -EPSILON:
            if result is None or candidateT < result[1]:
                result = i  # type: ignore
    return result


class Scene:
    def __init__(self) -> None:
        self.objects: List[Tuple[Any, Any]] = []
        self.lightPoints: List[Point] = []
        self.position: Point = Point(0, 1.8, 10)
        self.lookingAt: Point = Point.ZERO
        self.fieldOfView: float = 45
        self.recursionDepth: int = 0

    def moveTo(self, p: Point) -> None:
        self.position = p

    def lookAt(self, p: Point) -> None:
        self.lookingAt = p

    def addObject(self, obj: Any, surface: 'SimpleSurface') -> None:
        self.objects.append((obj, surface))

    def addLight(self, p: Point) -> None:
        self.lightPoints.append(p)

    def render(self, canvas: Canvas) -> None:
        fovRadians: float = (math.pi * (self.fieldOfView / 2.0)) / 180.0
        halfWidth: float = math.tan(fovRadians)
        halfHeight: float = 0.75 * halfWidth
        width: float = halfWidth * 2
        height: float = halfHeight * 2
        pixelWidth: float = width / (canvas.width - 1)
        pixelHeight: float = height / (canvas.height - 1)
        eye: Ray = Ray(self.position, self.lookingAt - self.position)
        vpRight: Vector = eye.vector.cross(Vector.UP).normalized()
        vpUp: Vector = vpRight.cross(eye.vector).normalized()
        for y in range(canvas.height):
            for x in range(canvas.width):
                xcomp: Vector = vpRight.scale(x * pixelWidth - halfWidth)
                ycomp: Vector = vpUp.scale(y * pixelHeight - halfHeight)
                ray: Ray = Ray(eye.point, (eye.vector + xcomp) + ycomp)
                colour: Tuple[float, float, float] = self.rayColour(ray)
                canvas.plot(x, y, *colour)

    def rayColour(self, ray: Ray) -> Tuple[float, float, float]:
        if self.recursionDepth > 3:
            return (0, 0, 0)
        try:
            self.recursionDepth += 1
            intersections: List[Tuple[Any, Optional[float], Any]] = [
                (o, o.intersectionTime(ray), s) for (o, s) in self.objects
            ]
            i: Optional[Tuple[Any, float, Any]] = firstIntersection(intersections)
            if i is None:
                return (0, 0, 0)
            else:
                o, t, s = i
                p: Point = ray.pointAtTime(t)
                return s.colourAt(self, ray, p, o.normalAt(p))
        finally:
            self.recursionDepth -= 1

    def _lightIsVisible(self, l: Point, p: Point) -> bool:
        for (o, s) in self.objects:
            t: Optional[float] = o.intersectionTime(Ray(p, l - p))
            if t is not None and t > EPSILON:
                return False
        return True

    def visibleLights(self, p: Point) -> List[Point]:
        result: List[Point] = []
        for l in self.lightPoints:
            if self._lightIsVisible(l, p):
                result.append(l)
        return result


def addColours(a: Tuple[float, float, float], scale: float,
               b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


class SimpleSurface:
    def __init__(self, **kwargs: Any) -> None:
        self.baseColour: Tuple[float, float, float] = kwargs.get('baseColour', (1, 1, 1))
        self.specularCoefficient: float = kwargs.get('specularCoefficient', 0.2)
        self.lambertCoefficient: float = kwargs.get('lambertCoefficient', 0.6)
        self.ambientCoefficient: float = 1.0 - self.specularCoefficient - self.lambertCoefficient

    def baseColourAt(self, p: Point) -> Tuple[float, float, float]:
        return self.baseColour

    def colourAt(self, scene: Scene, ray: Ray, p: Point, normal: Vector) -> Tuple[float, float, float]:
        b: Tuple[float, float, float] = self.baseColourAt(p)
        c: Tuple[float, float, float] = (0, 0, 0)
        if self.specularCoefficient > 0:
            reflectedRay: Ray = Ray(p, ray.vector.reflectThrough(normal))
            reflectedColour: Tuple[float, float, float] = scene.rayColour(reflectedRay)
            c = addColours(c, self.specularCoefficient, reflectedColour)
        if self.lambertCoefficient > 0:
            lambertAmount: float = 0
            for lightPoint in scene.visibleLights(p):
                contribution: float = (lightPoint - p).normalized().dot(normal)
                if contribution > 0:
                    lambertAmount += contribution
            lambertAmount = min(1, lambertAmount)
            c = addColours(c, self.lambertCoefficient * lambertAmount, b)
        if self.ambientCoefficient > 0:
            c = addColours(c, self.ambientCoefficient, b)
        return c


class CheckerboardSurface(SimpleSurface):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.otherColour: Tuple[float, float, float] = kwargs.get('otherColour', (0, 0, 0))
        self.checkSize: float = kwargs.get('checkSize', 1)

    def baseColourAt(self, p: Point) -> Tuple[float, float, float]:
        v: Vector = p - Point.ZERO  # yields a Vector
        # Normalize the scale relative to checkSize.
        scaled: Vector = v.scale(1.0 / self.checkSize)
        if ((int(abs(scaled.x) + 0.5) + int(abs(scaled.y) + 0.5) + int(abs(scaled.z) + 0.5)) % 2):
            return self.otherColour
        else:
            return self.baseColour


def bench_raytrace(loops: int, width: int, height: int, filename: Optional[str]) -> float:
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    for i in range_it:
        canvas: Canvas = Canvas(width, height)
        s: Scene = Scene()
        s.addLight(Point(30, 30, 10))
        s.addLight(Point(-10, 100, 30))
        s.lookAt(Point(0, 3, 0))
        s.addObject(Sphere(Point(1, 3, -10), 2), SimpleSurface(baseColour=(1, 1, 0)))
        for y in range(6):
            s.addObject(Sphere(Point(-3 - (y * 0.4), 2.3, -5), 0.4),
                        SimpleSurface(baseColour=(y / 6.0, 1 - (y / 6.0), 0.5)))
        s.addObject(Halfspace(Point(0, 0, 0), Vector.UP), CheckerboardSurface())
        s.render(canvas)
    dt: float = pyperf.perf_counter() - t0
    if filename:
        canvas.write_ppm(filename)
    return dt


def add_cmdline_args(cmd: List[str], args: Namespace) -> None:
    cmd.append('--width=%s' % args.width)
    cmd.append('--height=%s' % args.height)
    if args.filename:
        cmd.extend(('--filename', args.filename))


if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    cmd = runner.argparser  # type: Any
    cmd.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                     help='Image width (default: %s)' % DEFAULT_WIDTH)
    cmd.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                     help='Image height (default: %s)' % DEFAULT_HEIGHT)
    cmd.add_argument('--filename', metavar='FILENAME.PPM',
                     help='Output filename of the PPM picture')
    args = runner.parse_args()
    runner.metadata['description'] = 'Simple raytracer'
    runner.metadata['raytrace_width'] = args.width
    runner.metadata['raytrace_height'] = args.height
    runner.bench_time_func('raytrace', bench_raytrace, args.width, args.height, args.filename)