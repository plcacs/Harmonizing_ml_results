import copy
import numpy as np
from scipy.spatial import ConvexHull
from nevergrad.common.typing import ArrayLike
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import transforms as trans
from . import sequences
from . import base
from .base import IntOrParameter
from . import utils

def convex_limit(struct_points: ArrayLike) -> int:
    """Given points in order from best to worst,
    Returns the length of the maximum initial segment of points such that quasiconvexity is verified."""
    points = []
    d = len(struct_points[0])
    if len(struct_points) < 2 * d + 2:
        return len(struct_points) // 2
    for i in range(0, min(2 * d + 2, len(struct_points)), 2):
        points += [struct_points[i]]
    hull = ConvexHull(points[:d + 1], incremental=True)
    num_points = len(hull.vertices)
    k = len(points) - 1
    for i in range(num_points, len(points)):
        hull.add_points(points[i:i + 1])
        num_points += 1
        if len(hull.vertices) != num_points:
            return num_points - 1
        for j in range(i + 1, len(points)):
            hull_copy = copy.deepcopy(hull)
            hull_copy.add_points(points[j:j + 1])
            if len(hull_copy.vertices) != num_points + 1:
                return num_points - 1
    return k

def hull_center(points: ArrayLike, k: int) -> np.ndarray:
    """Center of the cuboid enclosing the hull."""
    hull = ConvexHull(points[:k])
    maxi = np.asarray(hull.vertices[0])
    mini = np.asarray(hull.vertices[0])
    for v in hull.vertices:
        maxi = np.maximum(np.asarray(v), maxi)
        mini = np.minimum(np.asarray(v), mini)
    return 0.5 * (maxi + mini)

def avg_of_k_best(archive, method: str = 'dimfourth') -> np.ndarray:
    """Operators inspired by the work of Yann Chevaleyre, Laurent Meunier, Clement Royer, Olivier Teytaud, Fabien Teytaud.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
        Provides a random recommendation instead of the best point so far (for baseline)
    method: str
        If dimfourth, we use the Fteytaud heuristic, i.e. k = min(len(archive) // 4, dimension)
        If exp, we use the Lmeunier method, i.e. k=max(1, len(archiv) // (2**dimension))
        If hull, we use the maximum k <= dimfourth-value, such that the function looks quasiconvex on the k best points.
    """
    items = list(archive.items_as_arrays())
    dimension = len(items[0][0])
    if method == 'dimfourth':
        k = min(len(archive) // 4, dimension)
    elif method == 'exp':
        k = max(1, int(len(archive) // 1.1 ** dimension))
    elif method == 'hull':
        k = convex_limit(np.concatenate(sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation('pessimistic')), axis=0))
        k = min(len(archive) // 4, min(k, int(len(archive) / 1.1 ** dimension))
    else:
        raise ValueError(f'{method} not implemented as a method for choosing k in avg_of_k_best.')
    k = 1 if k < 1 else int(k)
    first_k_individuals = sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation('pessimistic'))[:k]
    assert len(first_k_individuals) == k
    return np.array(sum((p[0] for p in first_k_individuals)) / k)
