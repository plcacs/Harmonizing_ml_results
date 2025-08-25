import collections
from collections import defaultdict
from fractions import Fraction
import pyperf
from typing import Dict, List, Tuple, Set, Callable, TypeVar, DefaultDict, Iterator, Union, Optional, Any, NamedTuple

T = TypeVar('T')

def topoSort(roots: List[T], getParents: Callable[[T], List[T]]) -> List[T]:
    'Return a topological sorting of nodes in a graph.\n\n    roots - list of root nodes to search from\n    getParents - function which returns the parents of a given node\n    '
    results: List[T] = []
    visited: Set[T] = set()
    stack: List[Tuple[T, int]] = [(node, 0) for node in roots]
    while stack:
        (current, state) = stack.pop()
        if (state == 0):
            if (current not in visited):
                visited.add(current)
                stack.append((current, 1))
                stack.extend(((parent, 0) for parent in getParents(current)))
        else:
            assert (current in visited)
            results.append(current)
    return results

def getDamages(L: int, A: int, D: int, B: int, stab: bool, te: float) -> List[int]:
    x: int = ((2 * L) // 5)
    x = (((((x + 2) * A) * B) // (D * 50)) + 2)
    if stab:
        x += (x // 2)
    x = int((x * te))
    return [((x * z) // 255) for z in range(217, 256)]

def getCritDist(L: int, p: Fraction, A1: int, A2: int, D1: int, D2: int, B: int, stab: bool, te: float) -> DefaultDict[int, Fraction]:
    p = min(p, Fraction(1))
    norm: List[int] = getDamages(L, A1, D1, B, stab, te)
    crit: List[int] = getDamages((L * 2), A2, D2, B, stab, te)
    dist: DefaultDict[int, Fraction] = defaultdict(Fraction)
    for (mult, vals) in zip([(1 - p), p], [norm, crit]):
        mult /= len(vals)
        for x in vals:
            dist[x] += mult
    return dist

def plus12(x: int) -> int:
    return (x + (x // 8))

class stats_t(NamedTuple):
    atk: int
    df: int
    speed: int
    spec: int

NOMODS: stats_t = stats_t(0, 0, 0, 0)

class fixeddata_t(NamedTuple):
    maxhp: int
    stats: stats_t
    lvl: int
    badges: stats_t
    basespeed: int

class halfstate_t(NamedTuple):
    fixed: fixeddata_t
    hp: int
    status: int
    statmods: stats_t
    stats: stats_t

def applyHPChange(hstate: halfstate_t, change: int) -> halfstate_t:
    hp = min(hstate.fixed.maxhp, max(0, (hstate.hp + change)))
    return hstate._replace(hp=hp)

def applyBadgeBoosts(badges: Tuple[int, int, int, int], stats: stats_t) -> stats_t:
    return stats_t(*[(plus12(x) if b else x) for (x, b) in zip(stats, badges)])

class attack_stats_t(NamedTuple):
    power: int
    isspec: bool
    stab: bool
    te: float
    crit: bool

attack_data: Dict[str, attack_stats_t] = {'Ember': attack_stats_t(40, True, True, 0.5, False), 'Dig': attack_stats_t(100, False, False, 1, False), 'Slash': attack_stats_t(70, False, False, 1, True), 'Water Gun': attack_stats_t(40, True, True, 2, False), 'Bubblebeam': attack_stats_t(65, True, True, 2, False)}

StateType = Tuple[halfstate_t, halfstate_t, int]

def _applyActionSide1(state: StateType, act: str) -> Dict[StateType, Fraction]:
    (me, them, extra) = state
    if (act == 'Super Potion'):
        me = applyHPChange(me, 50)
        return {(me, them, extra): Fraction(1)}
    mdata = attack_data[act]
    aind = (3 if mdata.isspec else 0)
    dind = (3 if mdata.isspec else 1)
    pdiv = (64 if mdata.crit else 512)
    dmg_dist = getCritDist(me.fixed.lvl, Fraction(me.fixed.basespeed, pdiv), me.stats[aind], me.fixed.stats[aind], them.stats[dind], them.fixed.stats[dind], mdata.power, mdata.stab, mdata.te)
    dist: DefaultDict[StateType, Fraction] = defaultdict(Fraction)
    for (dmg, p) in dmg_dist.items():
        them2 = applyHPChange(them, (- dmg))
        dist[(me, them2, extra)] += p
    return dist

def _applyAction(state: StateType, side: int, act: str) -> Dict[StateType, Fraction]:
    if (side == 0):
        return _applyActionSide1(state, act)
    else:
        (me, them, extra) = state
        dist = _applyActionSide1((them, me, extra), act)
        return {(k[1], k[0], k[2]): v for (k, v) in dist.items()}

StateP = Union[Tuple[int, StateType], Tuple[int, StateType, int, str], Tuple[int, bool]]

class Battle(object):

    def __init__(self) -> None:
        self.successors: Dict[StateP, List[Union[Tuple[StateP, float], StateP]]] = {}
        self.min: DefaultDict[StateP, float] = defaultdict(float)
        self.max: DefaultDict[StateP, float] = defaultdict((lambda : 1.0))
        self.frozen: Set[StateP] = set()
        self.win: StateP = (4, True)
        self.loss: StateP = (4, False)
        self.max[self.loss] = 0.0
        self.min[self.win] = 1.0
        self.frozen.update([self.win, self.loss])

    def _getSuccessorsA(self, statep: Tuple[int, StateType]) -> Iterator[Tuple[int, StateType, str]]:
        (st, state) = statep
        for action in ['Dig', 'Super Potion']:
            (yield (1, state, action))

    def _applyActionPair(self, state: StateType, side1: int, act1: str, side2: int, act2: str, dist: DefaultDict[StateP, Fraction], pmult: Fraction) -> None:
        for (newstate, p) in _applyAction(state, side1, act1).items():
            if (newstate[0].hp == 0):
                newstatep: StateP = self.loss
            elif (newstate[1].hp == 0):
                newstatep = self.win
            else:
                newstatep = (2, newstate, side2, act2)
            dist[newstatep] += (p * pmult)

    def _getSuccessorsB(self, statep: Tuple[int, StateType, str]) -> Dict[StateP, float]:
        (st, state, action) = statep
        dist: DefaultDict[StateP, Fraction] = defaultdict(Fraction)
        for (eact, p) in [('Water Gun', Fraction(64, 130)), ('Bubblebeam', Fraction(66, 130))]:
            priority1 = (state[0].stats.speed + (10000 * (action == 'Super Potion')))
            priority2 = (state[1].stats.speed + (10000 * (action == 'X Defend')))
            if (priority1 > priority2):
                self._applyActionPair(state, 0, action, 1, eact, dist, p)
            elif (priority1 < priority2):
                self._applyActionPair(state, 1, eact, 0, action, dist, p)
            else:
                self._applyActionPair(state, 0, action, 1, eact, dist, (p / 2))
                self._applyActionPair(state, 1, eact, 0, action, dist, (p / 2))
        return {k: float(p) for (k, p) in dist.items() if (p > 0)}

    def _getSuccessorsC(self, statep: Tuple[int, StateType, int, str]) -> Dict[StateP, float]:
        (st, state, side, action) = statep
        dist: DefaultDict[StateP, Fraction] = defaultdict(Fraction)
        for (newstate, p) in _applyAction(state, side, action).items():
            if (newstate[0].hp == 0):
                newstatep: StateP = self.loss
            elif (newstate[1].hp == 0):
                newstatep = self.win
            else:
                newstatep = (0, newstate)
            dist[newstatep] += p
        return {k: float(p) for (k, p) in dist.items() if (p > 0)}

    def getSuccessors(self, statep: StateP) -> List[Union[Tuple[StateP, float], StateP]]:
        try:
            return self.successors[statep]
        except KeyError:
            st = statep[0]
        if (st == 0):
            result = list(self._getSuccessorsA(statep))  # type: ignore
        else:
            if (st == 1):
                dist = self._getSuccessorsB(statep)  # type: ignore
            elif (st == 2):
                dist = self._getSuccessorsC(statep)  # type: ignore
            result = sorted(dist.items(), key=(lambda t: ((- t[1]), t[0])))
        self.successors[statep] = result
        return result

    def getSuccessorsList(self, statep: StateP) -> List[StateP]:
        if (statep[0] == 4):
            return []
        temp = self.getSuccessors(statep)
        if (statep[0] != 0):
            temp = (list(zip(*temp))[0] if temp else [])  # type: ignore
        return temp

    def evaluate(self, tolerance: float = 0.15) -> float:
        badges = (1, 0, 0, 0)
        starfixed = fixeddata_t(59, stats_t(40, 44, 56, 50), 11, NOMODS, 115)
        starhalf = halfstate_t(starfixed, 59, 0, NOMODS, stats_t(40, 44, 56, 50))
        charfixed = fixeddata_t(63, stats_t(39, 34, 46, 38), 26, badges, 65)
        charhalf = halfstate_t(charfixed, 63, 0, NOMODS, applyBadgeBoosts(badges, stats_t(39, 34, 46, 38)))
        initial_state: StateType = (charhalf, starhalf, 0)
        initial_statep: StateP = (0, initial_state)
        (dmin, dmax, frozen) = (self.min, self.max, self.frozen)
        stateps = topoSort([initial_statep], self.getSuccessorsList)
        itercount = 0
        while ((dmax[initial_statep] - dmin[initial_statep]) > tolerance):
            itercount += 1
            for sp in stateps:
                if (sp in frozen):
                    continue
                if (sp[0] == 0):
                    dmin[sp] = max((dmin[sp2] for sp2 in self.getSuccessors(sp)))  # type: ignore
                    dmax[sp] = max((dmax[sp2] for sp2 in self.getSuccessors(sp)))  # type: ignore
                else:
                    dmin[sp] = sum(((dmin[sp2] * p) for (sp2, p) in self.getSuccessors(sp)))  # type: ignore
                    dmax[sp] = sum(((dmax[sp2] * p) for (sp2, p) in self.getSuccessors(sp)))  # type: ignore
                if (dmin[sp] >= dmax[sp]):
                    dmax[sp] = dmin[sp] = ((dmin[sp] + dmax[sp]) / 2)
                    frozen.add(sp)
        return ((dmax[initial_statep] + dmin[initial_statep]) / 2)

def bench_mdp(loops: int) -> float:
    expected = 0.89873589887
    max_diff = 1e-06
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        result = Battle().evaluate(0.192)
    dt = (pyperf.perf_counter() - t0)
    if (abs((result - expected)) > max_diff):
        raise Exception(('invalid result: got %s, expected %s (diff: %s, max diff: %s)' % (result, expected, (result - expected), max_diff)))
    return dt

if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'MDP benchmark'
    runner.bench_time_func('mdp', bench_mdp)
