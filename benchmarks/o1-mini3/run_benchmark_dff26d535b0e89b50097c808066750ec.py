from bisect import bisect
import pyperf
from typing import List, Dict, Callable, Set, FrozenSet, Tuple

SOLVE_ARG: int = 60
WIDTH: int
HEIGHT: int
WIDTH, HEIGHT = (5, 10)
DIR_NO: int = 6
S: int
E: int
S, E = (WIDTH * HEIGHT, 2)
SE: float = S + (E / 2)
SW: float = SE - E
W: int = -E
NW: float = -SE
NE: float = -SW
SOLUTIONS: List[str] = [
    '00001222012661126155865558633348893448934747977799', 
    '00001222012771127148774485464855968596835966399333', 
    '00001222012771127148774485494855998596835966366333', 
    '00001222012771127148774485994855948596835966366333', 
    '00001222012771127166773863384653846538445584959999', 
    '00001222012771127183778834833348555446554666969999', 
    '00001222012771127183778834833348555446554966699996', 
    '00001223012331123166423564455647456775887888979999', 
    '00001555015541144177484726877268222683336689399993', 
    '00001555015541144177484728677286222863338669399993', 
    '00001599015591159148594482224827748276837766366333', 
    '00001777017871184155845558449984669662932629322333', 
    '00004222042664426774996879687759811598315583153331', 
    '00004222042774427384773183331866118615586555969999', 
    '00004223042334423784523785771855718566186611969999', 
    '00004227042874428774528735833155831566316691169999', 
    '00004333045534455384175781777812228116286662969999', 
    '00004333045934459384559185991866118612286727267772', 
    '00004333047734427384277182221866118615586555969999', 
    '00004555045514411224172621768277368736683339899998', 
    '00004555045514411224172721777299998966836688368333', 
    '00004555045534413334132221177266172677886888969999', 
    '00004555045534473334739967792617926192661882211888', 
    '00004555045564466694699992288828811233317273177731', 
    '00004555045564466694699997773172731233312881122888', 
    '00004555045564496664999962288828811233317273177731', 
    '00004555045564496664999967773172731233312881122888', 
    '00004555045584411884171187722866792679236992369333', 
    '00004555045584411884191189999832226333267372677766', 
    '00004555045584411884191189999866222623336727367773', 
    '13335138551389511895778697869947762446624022240000', 
    '13777137271333211882888226999946669446554055540000', 
    '13777137271333211882888229999649666446554055540000', 
    '27776272768221681166819958195548395443954033340000', 
    '33322392623926696648994485554855148117871077710000', 
    '33366366773867284772842228449584195119551099510000', 
    '33366366953869584955849958447784172117721022210000', 
    '33366366953869589955849458447784172117721022210000', 
    '33386388663866989999277712727142211441554055540000', 
    '33396329963297629766822778117148811448554055540000', 
    '33399366953869586955846458447784172117721022210000', 
    '37776372763332622266899998119148811448554055540000', 
    '39999396683336822268277682748477144114551055510000', 
    '39999398663338622286277862748477144114551055510000', 
    '66777627376233362223899998119148811448554055540000', 
    '69999666945564455584333843887738172117721022210000', 
    '88811228816629162971629776993743337443554055540000', 
    '88822118821333213727137776999946669446554055540000', 
    '88822118821333213727137779999649666446554055540000', 
    '89999893338663786377286712627142211441554055540000', 
    '99777974743984439884333685556855162116621022210000', 
    '99995948554483564835648336837766172117721022210000', 
    '99996119661366513855133853782547782447824072240000', 
    '99996911668166581755817758732548732443324032240000', 
    '99996926668261182221877718757148355443554033340000', 
    '99996955568551681166812228177248372443774033340000', 
    '99996955568551681166813338137748372447724022240000', 
    '99996966645564455584333843887738172117721022210000', 
    '99996988868877627166277112223143331443554055540000', 
    '99997988878857765474655446532466132113321032210000'
]

E: int
NE: float
NW: float
SE: float
SW: float
W: int
E, NE, NW, W, SW, SE = E, NE, NW, W, SW, SE

def rotate(ido: List[int], rd: Dict[int, float] = {E: NE, NE: NW, NW: W, W: SW, SW: SE, SE: E}) -> List[float]:
    return [rd[o] for o in ido]

def flip(ido: List[int], fd: Dict[int, float] = {E: E, NE: SE, NW: SW, W: W, SW: NW, SE: NE}) -> List[float]:
    return [fd[o] for o in ido]

def permute(ido: List[int], r_ido: List[int], rotate_func: Callable[[List[int]], List[float]] = rotate, flip_func: Callable[[List[int]], List[float]] = flip) -> List[List[float]]:
    ps: List[List[float]] = [ido]
    for r in range(DIR_NO - 1):
        ps.append(rotate_func(ps[-1]))
        if ido == r_ido:
            ps = ps[:DIR_NO // 2]
    for pp in ps[:]:
        ps.append(flip_func(pp))
    return ps

def convert(ido: List[int]) -> List[int]:
    'incremental direction offsets -> "coordinate offsets" '
    out: List[int] = [0]
    for o in ido:
        out.append(out[-1] + o)
    return list(set(out))

def get_footprints(board: List[int], cti: Dict[int, int], pieces: List[List[int]]) -> List[List[FrozenSet[int]]]:
    fps: List[List[FrozenSet[int]]] = [[[] for _ in range(len(pieces))] for _ in range(len(board))]
    for c in board:
        for pi, p in enumerate(pieces):
            for pp in p:
                fp: FrozenSet[int] = frozenset([cti[c + o] for o in pp if (c + o) in cti])
                if len(fp) == 5:
                    fps[min(fp)][pi].append(fp)
    return fps

def get_senh(board: List[int], cti: Dict[int, int]) -> List[FrozenSet[int]]:
    '-> south-east neighborhood'
    se_nh: List[FrozenSet[int]] = []
    nh_offsets: List[int] = [E, SW, SE]
    for c in board:
        neighbors: Set[int] = set()
        for o in nh_offsets:
            if (c + o) in cti:
                neighbors.add(cti[c + o])
        se_nh.append(frozenset(neighbors))
    return se_nh

def get_puzzle(width: int, height: int) -> Tuple[List[int], Dict[int, int], List[List[int]]]:
    board: List[int] = [((E * x) + (S * y) + (y % 2)) for y in range(height) for x in range(width)]
    cti: Dict[int, int] = {board[i]: i for i in range(len(board))}
    idos: List[List[int]] = [
        [E, E, E, SE], 
        [SE, SW, W, SW], 
        [W, W, SW, SE], 
        [E, E, SW, SE], 
        [NW, W, NW, SE, SW], 
        [E, E, NE, W], 
        [NW, NE, NE, W], 
        [NE, SE, E, NE], 
        [SE, SE, E, SE], 
        [E, NW, NW, NW]
    ]
    perms_generator = (permute(p, idos[3]) for p in idos)
    pieces: List[List[int]] = [[convert(pp) for pp in p] for p in perms_generator]
    return board, cti, pieces

def solve(
    n: int, 
    i_min: int, 
    free: FrozenSet[int], 
    curr_board: List[int], 
    pieces_left: List[int], 
    solutions: List[str], 
    fps: List[List[FrozenSet[int]]], 
    se_nh: List[FrozenSet[int]], 
    bisect_func: Callable[[List[str], str], int] = bisect
) -> None:
    fp_i_cands: List[FrozenSet[int]] = fps[i_min]
    for p in pieces_left:
        fp_cands: List[FrozenSet[int]] = fp_i_cands[p]
        for fp in fp_cands:
            if fp.issubset(free):
                n_curr_board: List[int] = curr_board[:]
                for ci in fp:
                    n_curr_board[ci] = p
                if len(pieces_left) > 1:
                    n_free: FrozenSet[int] = free - fp
                    n_i_min: int = min(n_free)
                    if len(n_free & se_nh[n_i_min]) > 0:
                        n_pieces_left: List[int] = pieces_left[:]
                        n_pieces_left.remove(p)
                        solve(n, n_i_min, n_free, n_curr_board, n_pieces_left, solutions, fps, se_nh)
                else:
                    s: str = ''.join(map(str, n_curr_board))
                    solutions.insert(bisect_func(solutions, s), s)
                    rs: str = s[::-1]
                    solutions.insert(bisect_func(solutions, rs), rs)
                    if len(solutions) >= n:
                        return
        if len(solutions) >= n:
            return

def bench_meteor_contest(
    loops: int, 
    board: List[int], 
    pieces: List[List[int]], 
    solve_arg: int, 
    fps: List[List[FrozenSet[int]]], 
    se_nh: List[FrozenSet[int]], 
    bisect_func: Callable[[List[str], str], int] = bisect
) -> float:
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    solutions_reference: List[str] = []
    for _ in range_it:
        free: FrozenSet[int] = frozenset(range(len(board)))
        curr_board: List[int] = [-1] * len(board)
        pieces_left: List[int] = list(range(len(pieces)))
        solutions: List[str] = []
        solve(solve_arg, 0, free, curr_board, pieces_left, solutions, fps, se_nh)
    dt: float = pyperf.perf_counter() - t0
    if solutions != SOLUTIONS:
        raise ValueError('unexpected solutions')
    return dt

def main() -> None:
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Solver for Meteor Puzzle board'
    board, cti, pieces = get_puzzle(WIDTH, HEIGHT)
    fps = get_footprints(board, cti, pieces)
    se_nh = get_senh(board, cti)
    solve_arg = SOLVE_ARG
    runner.bench_time_func(
        'meteor_contest', 
        bench_meteor_contest, 
        board, 
        pieces, 
        solve_arg, 
        fps, 
        se_nh
    )

if __name__ == '__main__':
    main()
