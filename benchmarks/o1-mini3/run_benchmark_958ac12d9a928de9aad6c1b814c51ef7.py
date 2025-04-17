'create chaosgame-like fractals\n\nCopyright (C) 2005 Carl Friedrich Bolz\n'
import math
import random
import pyperf
from typing import List, Optional, Tuple

DEFAULT_THICKNESS: float = 0.25
DEFAULT_WIDTH: int = 256
DEFAULT_HEIGHT: int = 256
DEFAULT_ITERATIONS: int = 5000
DEFAULT_RNG_SEED: int = 1234


class GVector:
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def Mag(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def dist(self, other: 'GVector') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def __add__(self, other: 'GVector') -> 'GVector':
        if not isinstance(other, GVector):
            raise ValueError(f"Can't add GVector to {type(other)}")
        v = GVector(self.x + other.x, self.y + other.y, self.z + other.z)
        return v

    def __sub__(self, other: 'GVector') -> 'GVector':
        return self + (other * -1)

    def __mul__(self, other: float) -> 'GVector':
        v = GVector(self.x * other, self.y * other, self.z * other)
        return v

    __rmul__ = __mul__

    def linear_combination(self, other: 'GVector', l1: float, l2: Optional[float] = None) -> 'GVector':
        if l2 is None:
            l2 = 1 - l1
        v = GVector(
            self.x * l1 + other.x * l2,
            self.y * l1 + other.y * l2,
            self.z * l1 + other.z * l2
        )
        return v

    def __str__(self) -> str:
        return f'<{self.x}, {self.y}, {self.z}>'

    def __repr__(self) -> str:
        return f'GVector({self.x}, {self.y}, {self.z})'


class Spline:
    'Class for representing B-Splines and NURBS of arbitrary degree'

    points: List[GVector]
    degree: int
    knots: List[float]

    def __init__(self, points: List[GVector], degree: int, knots: List[float]) -> None:
        'Creates a Spline.\n\n        points is a list of GVector, degree is the degree of the Spline.\n        '
        if len(points) > (len(knots) - degree + 1):
            raise ValueError('too many control points')
        elif len(points) < (len(knots) - degree + 1):
            raise ValueError('not enough control points')
        last = knots[0]
        for cur in knots[1:]:
            if cur < last:
                raise ValueError('knots not strictly increasing')
            last = cur
        self.knots = knots
        self.points = points
        self.degree = degree

    def GetDomain(self) -> Tuple[float, float]:
        'Returns the domain of the B-Spline'
        return (self.knots[self.degree - 1], self.knots[len(self.knots) - self.degree])

    def __call__(self, u: float) -> GVector:
        'Calculates a point of the B-Spline using de Boors Algorithm'
        dom = self.GetDomain()
        if u < dom[0] or u > dom[1]:
            raise ValueError('Function value not in domain')
        if u == dom[0]:
            return self.points[0]
        if u == dom[1]:
            return self.points[-1]
        I = self.GetIndex(u)
        d: List[GVector] = [self.points[(I - self.degree + 1) + ii] for ii in range(self.degree + 1)]
        U = self.knots
        for ik in range(1, self.degree + 1):
            for ii in range(I - self.degree + ik + 1, I + 2):
                ua = U[ii + self.degree - ik]
                ub = U[ii - 1]
                co1 = (ua - u) / (ua - ub)
                co2 = (u - ub) / (ua - ub)
                index = (ii - I + self.degree - ik - 1)
                d[index] = d[index].linear_combination(d[index + 1], co1, co2)
        return d[0]

    def GetIndex(self, u: float) -> int:
        dom = self.GetDomain()
        for ii in range(self.degree - 1, len(self.knots) - self.degree):
            if self.knots[ii] <= u < self.knots[ii + 1]:
                return ii
        return int(dom[1] - 1)

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f'Spline({repr(self.points)}, {repr(self.degree)}, {repr(self.knots)})'


def write_ppm(im: List[List[int]], filename: str) -> None:
    magic = 'P6\n'
    maxval = 255
    w = len(im)
    h = len(im[0])
    with open(filename, 'w', encoding='latin1', newline='') as fp:
        fp.write(magic)
        fp.write(f'{w} {h}\n{maxval}\n')
        for j in range(h):
            for i in range(w):
                val = im[i][j]
                c = int(val * 255)
                fp.write(chr(c) + chr(c) + chr(c))


class Chaosgame:
    splines: List[Spline]
    thickness: float
    minx: float
    miny: float
    maxx: float
    maxy: float
    height: float
    width: float
    num_trafos: List[int]
    num_total: int

    def __init__(self, splines: List[Spline], thickness: float = 0.1) -> None:
        self.splines = splines
        self.thickness = thickness
        self.minx = min(p.x for spl in splines for p in spl.points)
        self.miny = min(p.y for spl in splines for p in spl.points)
        self.maxx = max(p.x for spl in splines for p in spl.points)
        self.maxy = max(p.y for spl in splines for p in spl.points)
        self.height = self.maxy - self.miny
        self.width = self.maxx - self.minx
        self.num_trafos = []
        maxlength = (thickness * self.width) / self.height
        for spl in splines:
            length = 0.0
            curr = spl(0.0)
            for i in range(1, 1000):
                last = curr
                t = (1 / 999) * i
                curr = spl(t)
                length += curr.dist(last)
            self.num_trafos.append(max(1, int((length / maxlength) * 1.5)))
        self.num_total = sum(self.num_trafos)

    def get_random_trafo(self) -> Tuple[int, int]:
        r = random.randrange(int(self.num_total) + 1)
        l = 0
        for i, num in enumerate(self.num_trafos):
            if l <= r < l + num:
                return (i, random.randrange(num))
            l += num
        return (len(self.num_trafos) - 1, random.randrange(self.num_trafos[-1]))

    def transform_point(self, point: GVector, trafo: Optional[Tuple[int, int]] = None) -> GVector:
        x = (point.x - self.minx) / self.width
        y = (point.y - self.miny) / self.height
        if trafo is None:
            trafo = self.get_random_trafo()
        start, end = self.splines[trafo[0]].GetDomain()
        length = end - start
        seg_length = length / self.num_trafos[trafo[0]]
        t = start + seg_length * trafo[1] + seg_length * x
        basepoint = self.splines[trafo[0]](t)
        if t + (1 / 50000) > end:
            neighbour = self.splines[trafo[0]](t - (1 / 50000))
            derivative = neighbour - basepoint
        else:
            neighbour = self.splines[trafo[0]](t + (1 / 50000))
            derivative = basepoint - neighbour
        if derivative.Mag() != 0:
            basepoint.x += (derivative.y / derivative.Mag()) * (y - 0.5) * self.thickness
            basepoint.y += (-derivative.x / derivative.Mag()) * (y - 0.5) * self.thickness
        else:
            print('r', end='')
        self.truncate(basepoint)
        return basepoint

    def truncate(self, point: GVector) -> None:
        if point.x >= self.maxx:
            point.x = self.maxx
        if point.y >= self.maxy:
            point.y = self.maxy
        if point.x < self.minx:
            point.x = self.minx
        if point.y < self.miny:
            point.y = self.miny

    def create_image_chaos(
        self, w: int, h: int, iterations: int, filename: Optional[str], rng_seed: int
    ) -> None:
        random.seed(rng_seed)
        im: List[List[int]] = [[1 for _ in range(h)] for _ in range(w)]
        point = GVector((self.maxx + self.minx) / 2, (self.maxy + self.miny) / 2, 0)
        for _ in range(iterations):
            point = self.transform_point(point)
            x = int((point.x - self.minx) / self.width * w)
            y = int((point.y - self.miny) / self.height * h)
            if x == w:
                x -= 1
            if y == h:
                y -= 1
            im[x][h - y - 1] = 0
        if filename:
            write_ppm(im, filename)


def main(runner: pyperf.Runner, args: Any) -> None:
    splines: List[Spline] = [
        Spline(
            [
                GVector(1.59735, 3.30446, 0.0),
                GVector(1.57581, 4.12326, 0.0),
                GVector(1.31321, 5.28835, 0.0),
                GVector(1.6189, 5.32991, 0.0),
                GVector(2.88994, 5.5027, 0.0),
                GVector(2.37306, 4.38183, 0.0),
                GVector(1.662, 4.36028, 0.0),
            ],
            3,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ),
        Spline(
            [
                GVector(2.8045, 4.01735, 0.0),
                GVector(2.5505, 3.52523, 0.0),
                GVector(1.97901, 2.62036, 0.0),
                GVector(1.97901, 2.62036, 0.0),
            ],
            3,
            [0, 0, 0, 1, 1, 1],
        ),
        Spline(
            [
                GVector(2.00167, 4.01132, 0.0),
                GVector(2.33504, 3.31283, 0.0),
                GVector(2.3668, 3.23346, 0.0),
                GVector(2.3668, 3.23346, 0.0),
            ],
            3,
            [0, 0, 0, 1, 1, 1],
        ),
    ]
    runner.metadata['chaos_thickness'] = args.thickness
    runner.metadata['chaos_width'] = args.width
    runner.metadata['chaos_height'] = args.height
    runner.metadata['chaos_iterations'] = args.iterations
    runner.metadata['chaos_rng_seed'] = args.rng_seed
    chaos = Chaosgame(splines, args.thickness)
    runner.bench_func(
        'chaos',
        chaos.create_image_chaos,
        args.width,
        args.height,
        args.iterations,
        args.filename,
        args.rng_seed,
    )


def add_cmdline_args(cmd: List[str], args: Any) -> None:
    cmd.append(f'--width={args.width}')
    cmd.append(f'--height={args.height}')
    cmd.append(f'--thickness={args.thickness}')
    cmd.append(f'--rng-seed={args.rng_seed}')
    if args.filename:
        cmd.extend(['--filename', args.filename])


if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'Create chaosgame-like fractals'
    cmd = runner.argparser
    cmd.add_argument(
        '--thickness',
        type=float,
        default=DEFAULT_THICKNESS,
        help=f'Thickness (default: {DEFAULT_THICKNESS})',
    )
    cmd.add_argument(
        '--width',
        type=int,
        default=DEFAULT_WIDTH,
        help=f'Image width (default: {DEFAULT_WIDTH})',
    )
    cmd.add_argument(
        '--height',
        type=int,
        default=DEFAULT_HEIGHT,
        help=f'Image height (default: {DEFAULT_HEIGHT})',
    )
    cmd.add_argument(
        '--iterations',
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f'Number of iterations (default: {DEFAULT_ITERATIONS})',
    )
    cmd.add_argument(
        '--filename',
        metavar='FILENAME.PPM',
        type=str,
        default=None,
        help='Output filename of the PPM picture',
    )
    cmd.add_argument(
        '--rng-seed',
        type=int,
        default=DEFAULT_RNG_SEED,
        help=f'Random number generator seed (default: {DEFAULT_RNG_SEED})',
    )
    args = runner.parse_args()
    main(runner, args)
