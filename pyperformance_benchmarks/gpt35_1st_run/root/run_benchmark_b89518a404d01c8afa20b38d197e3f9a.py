from typing import Tuple

class Vector:
    x: float
    y: float
    z: float

    def __init__(self, initx: float, inity: float, initz: float) -> None:
        self.x = initx
        self.y = inity
        self.z = initz

    def __str__(self) -> str:
        return ('(%s,%s,%s)' % (self.x, self.y, self.z))

    def __repr__(self) -> str:
        return ('Vector(%s,%s,%s)' % (self.x, self.y, self.z)

    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    def __add__(self, other: 'Vector') -> 'Point':
        if other.isPoint():
            return Point((self.x + other.x), (self.y + other.y), (self.z + other.z))
        else:
            return Vector((self.x + other.x), (self.y + other.y), (self.z + other.z))

    def __sub__(self, other: 'Vector') -> 'Vector':
        other.mustBeVector()
        return Vector((self.x - other.x), (self.y - other.y), (self.z - other.z))

    def scale(self, factor: float) -> 'Vector':
        return Vector((factor * self.x), (factor * self.y), (factor * self.z))

    def dot(self, other: 'Vector') -> float:
        other.mustBeVector()
        return (((self.x * other.x) + (self.y * other.y)) + (self.z * other.z))

    def cross(self, other: 'Vector') -> 'Vector':
        other.mustBeVector()
        return Vector(((self.y * other.z) - (self.z * other.y)), ((self.z * other.x) - (self.x * other.z)), ((self.x * other.y) - (self.y * other.x))

    def normalized(self) -> 'Vector':
        return self.scale((1.0 / self.magnitude()))

    def negated(self) -> 'Vector':
        return self.scale((- 1))

    def __eq__(self, other: 'Vector') -> bool:
        return ((self.x == other.x) and (self.y == other.y) and (self.z == other.z))

    def isVector(self) -> bool:
        return True

    def isPoint(self) -> bool:
        return False

    def mustBeVector(self) -> 'Vector':
        return self

    def mustBePoint(self) -> None:
        raise 'Vectors are not points!'

    def reflectThrough(self, normal: 'Vector') -> 'Vector':
        d = normal.scale(self.dot(normal))
        return (self - d.scale(2))

class Point:
    x: float
    y: float
    z: float

    def __init__(self, initx: float, inity: float, initz: float) -> None:
        self.x = initx
        self.y = inity
        self.z = initz

    def __str__(self) -> str:
        return ('(%s,%s,%s)' % (self.x, self.y, self.z))

    def __repr__(self) -> str:
        return ('Point(%s,%s,%s)' % (self.x, self.y, self.z))

    def __add__(self, other: 'Vector') -> 'Point':
        other.mustBeVector()
        return Point((self.x + other.x), (self.y + other.y), (self.z + other.z))

    def __sub__(self, other: 'Point') -> 'Vector':
        if other.isPoint():
            return Vector((self.x - other.x), (self.y - other.y), (self.z - other.z))
        else:
            return Point((self.x - other.x), (self.y - other.y), (self.z - other.z))

    def isVector(self) -> bool:
        return False

    def isPoint(self) -> bool:
        return True

    def mustBeVector(self) -> None:
        raise 'Points are not vectors!'

    def mustBePoint(self) -> 'Point':
        return self

class Sphere:
    centre: Point
    radius: float

    def __init__(self, centre: Point, radius: float) -> None:
        centre.mustBePoint()
        self.centre = centre
        self.radius = radius

    def __repr__(self) -> str:
        return ('Sphere(%s,%s)' % (repr(self.centre), self.radius)

    def intersectionTime(self, ray: 'Ray') -> float:
        cp = (self.centre - ray.point)
        v = cp.dot(ray.vector)
        discriminant = ((self.radius * self.radius) - (cp.dot(cp) - (v * v)))
        if (discriminant < 0):
            return None
        else:
            return (v - math.sqrt(discriminant))

    def normalAt(self, p: Point) -> Vector:
        return (p - self.centre).normalized()

class Halfspace:
    point: Point
    normal: Vector

    def __init__(self, point: Point, normal: Vector) -> None:
        self.point = point
        self.normal = normal.normalized()

    def __repr__(self) -> str:
        return ('Halfspace(%s,%s)' % (repr(self.point), repr(self.normal)))

    def intersectionTime(self, ray: 'Ray') -> float:
        v = ray.vector.dot(self.normal)
        if v:
            return (1 / (- v))
        else:
            return None

    def normalAt(self, p: Point) -> Vector:
        return self.normal

class Ray:
    point: Point
    vector: Vector

    def __init__(self, point: Point, vector: Vector) -> None:
        self.point = point
        self.vector = vector.normalized()

    def __repr__(self) -> str:
        return ('Ray(%s,%s)' % (repr(self.point), repr(self.vector)))

    def pointAtTime(self, t: float) -> Point:
        return (self.point + self.vector.scale(t))

class Canvas:
    bytes: array.array
    width: int
    height: int

    def __init__(self, width: int, height: int) -> None:
        self.bytes = array.array('B', ([0] * ((width * height) * 3)))
        for i in range((width * height)):
            self.bytes[((i * 3) + 2)] = 255
        self.width = width
        self.height = height

    def plot(self, x: int, y: int, r: float, g: float, b: float) -> None:
        i = (((((self.height - y) - 1) * self.width) + x) * 3)
        self.bytes[i] = max(0, min(255, int((r * 255))))
        self.bytes[(i + 1)] = max(0, min(255, int((g * 255))))
        self.bytes[(i + 2)] = max(0, min(255, int((b * 255)))

    def write_ppm(self, filename: str) -> None:
        header = ('P6 %d %d 255\n' % (self.width, self.height))
        with open(filename, 'wb') as fp:
            fp.write(header.encode('ascii'))
            fp.write(self.bytes.tobytes())

class Scene:
    objects: list
    lightPoints: list
    position: Point
    lookingAt: Point
    fieldOfView: float
    recursionDepth: int

    def __init__(self) -> None:
        self.objects = []
        self.lightPoints = []
        self.position = Point(0, 1.8, 10)
        self.lookingAt = Point.ZERO
        self.fieldOfView = 45
        self.recursionDepth = 0

    def moveTo(self, p: Point) -> None:
        self.position = p

    def lookAt(self, p: Point) -> None:
        self.lookingAt = p

    def addObject(self, object, surface) -> None:
        self.objects.append((object, surface))

    def addLight(self, p: Point) -> None:
        self.lightPoints.append(p)

    def render(self, canvas: Canvas) -> None:
        ...

    def rayColour(self, ray: Ray) -> Tuple[float, float, float]:
        ...

    def _lightIsVisible(self, l: Point, p: Point) -> bool:
        ...

    def visibleLights(self, p: Point) -> list:
        ...

class SimpleSurface:
    baseColour: Tuple[float, float, float]
    specularCoefficient: float
    lambertCoefficient: float
    ambientCoefficient: float

    def __init__(self, **kwargs) -> None:
        ...

    def baseColourAt(self, p: Point) -> Tuple[float, float, float]:
        ...

    def colourAt(self, scene: Scene, ray: Ray, p: Point, normal: Vector) -> Tuple[float, float, float]:
        ...

class CheckerboardSurface(SimpleSurface):
    otherColour: Tuple[float, float, float]
    checkSize: int

    def __init__(self, **kwargs) -> None:
        ...

    def baseColourAt(self, p: Point) -> Tuple[float, float, float]:
        ...

def bench_raytrace(loops: int, width: int, height: int, filename: str) -> float:
    ...

def addColours(a: Tuple[float, float, float], scale: float, b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    ...

def add_cmdline_args(cmd, args) -> None:
    ...

if (__name__ == '__main__'):
    ...
