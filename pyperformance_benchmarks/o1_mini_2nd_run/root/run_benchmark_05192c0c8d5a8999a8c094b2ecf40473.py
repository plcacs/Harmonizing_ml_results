from __future__ import annotations
import math
import random
import pyperf
from typing import List, Set, Optional, Tuple

SIZE: int = 9
GAMES: int = 200
KOMI: float = 7.5
EMPTY: int = 0
WHITE: int = 1
BLACK: int = 2
SHOW: dict[int, str] = {EMPTY: '.', WHITE: 'o', BLACK: 'x'}
PASS: int = -1
MAXMOVES: int = (SIZE * SIZE) * 3
TIMESTAMP: int = 0
MOVES: int = 0

def to_pos(x: int, y: int) -> int:
    return (y * SIZE) + x

def to_xy(pos: int) -> Tuple[int, int]:
    y, x = divmod(pos, SIZE)
    return (x, y)

class Square:
    board: Board
    pos: int
    timestamp: int
    removestamp: int
    zobrist_strings: List[int]
    neighbours: List[Square]
    color: int
    reference: Square
    ledges: int
    used: bool
    temp_ledges: int

    def __init__(self, board: Board, pos: int) -> None:
        self.board = board
        self.pos = pos
        self.timestamp = TIMESTAMP
        self.removestamp = TIMESTAMP
        self.zobrist_strings = [random.randrange(9223372036854775807) for _ in range(3)]
        self.neighbours = []
        self.color = EMPTY
        self.reference = self
        self.ledges = 0
        self.used = False

    def set_neighbours(self) -> None:
        x, y = self.pos % SIZE, self.pos // SIZE
        self.neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            newx, newy = x + dx, y + dy
            if 0 <= newx < SIZE and 0 <= newy < SIZE:
                self.neighbours.append(self.board.squares[to_pos(newx, newy)])

    def move(self, color: int) -> None:
        global TIMESTAMP, MOVES
        TIMESTAMP += 1
        MOVES += 1
        self.board.zobrist.update(self, color)
        self.color = color
        self.reference = self
        self.ledges = 0
        self.used = True
        for neighbour in self.neighbours:
            neighcolor = neighbour.color
            if neighcolor == EMPTY:
                self.ledges += 1
            else:
                neighbour_ref = neighbour.find(update=True)
                if neighcolor == color:
                    if neighbour_ref.reference.pos != self.pos:
                        self.ledges += neighbour_ref.ledges
                        neighbour_ref.reference = self
                    self.ledges -= 1
                else:
                    neighbour_ref.ledges -= 1
                    if neighbour_ref.ledges == 0:
                        neighbour.remove(neighbour_ref)
        self.board.zobrist.add()

    def remove(self, reference: Square, update: bool = True) -> None:
        self.board.zobrist.update(self, EMPTY)
        self.removestamp = TIMESTAMP
        if update:
            self.color = EMPTY
            self.board.emptyset.add(self.pos)
        for neighbour in self.neighbours:
            if neighbour.color != EMPTY and neighbour.removestamp != TIMESTAMP:
                neighbour_ref = neighbour.find(update)
                if neighbour_ref.pos == reference.pos:
                    neighbour.remove(reference, update)
                elif update:
                    neighbour_ref.ledges += 1

    def find(self, update: bool = False) -> Square:
        reference = self.reference
        if reference.pos != self.pos:
            reference = reference.find(update)
            if update:
                self.reference = reference
        return reference

    def __repr__(self) -> str:
        return repr(to_xy(self.pos))

class EmptySet:
    board: Board
    empties: List[int]
    empty_pos: List[int]

    def __init__(self, board: Board) -> None:
        self.board = board
        self.empties = list(range(SIZE * SIZE))
        self.empty_pos = list(range(SIZE * SIZE))

    def random_choice(self) -> int:
        choices = len(self.empties)
        while choices:
            i = int(random.random() * choices)
            pos = self.empties[i]
            if self.board.useful(pos):
                return pos
            choices -= 1
            self.set(i, self.empties[choices])
            self.set(choices, pos)
        return PASS

    def add(self, pos: int) -> None:
        self.empty_pos[pos] = len(self.empties)
        self.empties.append(pos)

    def remove(self, pos: int) -> None:
        self.set(self.empty_pos[pos], self.empties[-1])
        self.empties.pop()

    def set(self, i: int, pos: int) -> None:
        self.empties[i] = pos
        self.empty_pos[pos] = i

class ZobristHash:
    board: Board
    hash_set: Set[int]
    hash: int

    def __init__(self, board: Board) -> None:
        self.board = board
        self.hash_set = set()
        self.hash = 0
        for square in self.board.squares:
            self.hash ^= square.zobrist_strings[EMPTY]
        self.hash_set.clear()
        self.hash_set.add(self.hash)

    def update(self, square: Square, color: int) -> None:
        self.hash ^= square.zobrist_strings[square.color]
        self.hash ^= square.zobrist_strings[color]

    def add(self) -> None:
        self.hash_set.add(self.hash)

    def dupe(self) -> bool:
        return self.hash in self.hash_set

class Board:
    squares: List[Square]
    emptyset: EmptySet
    zobrist: ZobristHash
    color: int
    finished: bool
    lastmove: int
    history: List[int]
    white_dead: int
    black_dead: int

    def __init__(self) -> None:
        self.squares = [Square(self, pos) for pos in range(SIZE * SIZE)]
        for square in self.squares:
            square.set_neighbours()
        self.reset()

    def reset(self) -> None:
        for square in self.squares:
            square.color = EMPTY
            square.used = False
        self.emptyset = EmptySet(self)
        self.zobrist = ZobristHash(self)
        self.color = BLACK
        self.finished = False
        self.lastmove = -2
        self.history = []
        self.white_dead = 0
        self.black_dead = 0

    def move(self, pos: int) -> None:
        square = self.squares[pos]
        if pos != PASS:
            square.move(self.color)
            self.emptyset.remove(square.pos)
        elif self.lastmove == PASS:
            self.finished = True
        self.color = WHITE if self.color == BLACK else BLACK
        self.lastmove = pos
        self.history.append(pos)

    def random_move(self) -> int:
        return self.emptyset.random_choice()

    def useful_fast(self, square: Square) -> bool:
        if not square.used:
            for neighbour in square.neighbours:
                if neighbour.color == EMPTY:
                    return True
        return False

    def useful(self, pos: int) -> bool:
        global TIMESTAMP
        TIMESTAMP += 1
        square = self.squares[pos]
        if self.useful_fast(square):
            return True
        old_hash = self.zobrist.hash
        self.zobrist.update(square, self.color)
        empties = opps = weak_opps = neighs = weak_neighs = 0
        for neighbour in square.neighbours:
            neighcolor = neighbour.color
            if neighcolor == EMPTY:
                empties += 1
                continue
            neighbour_ref = neighbour.find()
            if neighbour_ref.timestamp != TIMESTAMP:
                if neighcolor == self.color:
                    neighs += 1
                else:
                    opps += 1
                neighbour_ref.timestamp = TIMESTAMP
                neighbour_ref.temp_ledges = neighbour_ref.ledges
            neighbour_ref.temp_ledges -= 1
            if neighbour_ref.temp_ledges == 0:
                if neighcolor == self.color:
                    weak_neighs += 1
                else:
                    weak_opps += 1
                    neighbour_ref.remove(neighbour_ref, update=False)
        dupe = self.zobrist.dupe()
        self.zobrist.hash = old_hash
        strong_neighs = neighs - weak_neighs
        strong_opps = opps - weak_opps
        return (not dupe) and (bool(empties) or bool(weak_opps) or (strong_neighs and (strong_opps or weak_neighs)))

    def useful_moves(self) -> List[int]:
        return [pos for pos in self.emptyset.empties if self.useful(pos)]

    def replay(self, history: List[int]) -> None:
        for pos in history:
            self.move(pos)

    def score(self, color: int) -> float:
        if color == WHITE:
            count = KOMI + self.black_dead
        else:
            count = self.white_dead
        for square in self.squares:
            squarecolor = square.color
            if squarecolor == color:
                count += 1
            elif squarecolor == EMPTY:
                surround = sum(1 for neighbour in square.neighbours if neighbour.color == color)
                if surround == len(square.neighbours):
                    count += 1
        return count

    def check(self) -> None:
        for square in self.squares:
            if square.color == EMPTY:
                continue
            members1: Set[Square] = {square}
            changed: bool = True
            while changed:
                changed = False
                for member in list(members1):
                    for neighbour in member.neighbours:
                        if neighbour.color == square.color and neighbour not in members1:
                            changed = True
                            members1.add(neighbour)
            ledges1: int = 0
            for member in members1:
                for neighbour in member.neighbours:
                    if neighbour.color == EMPTY:
                        ledges1 += 1
            root = square.find()
            members2: Set[Square] = {square2 for square2 in self.squares if square2.color != EMPTY and square2.find() == root}
            ledges2: int = root.ledges
            assert members1 == members2
            assert ledges1 == ledges2, f'ledges differ at {square!r}: {ledges1} {ledges2}'
            set(self.emptyset.empties)
            empties2: Set[int] = {sq.pos for sq in self.squares if sq.color == EMPTY}

    def __repr__(self) -> str:
        result: List[str] = []
        for y in range(SIZE):
            start = to_pos(0, y)
            line = ''.join([SHOW[square.color] + ' ' for square in self.squares[start:start + SIZE]])
            result.append(line)
        return '\n'.join(result)

class UCTNode:
    bestchild: Optional[UCTNode]
    pos: int
    wins: int
    losses: int
    pos_child: List[Optional[UCTNode]]
    parent: Optional[UCTNode]
    unexplored: List[int]

    def __init__(self) -> None:
        self.bestchild = None
        self.pos = -1
        self.wins = 0
        self.losses = 0
        self.pos_child = [None for _ in range(SIZE * SIZE)]
        self.parent = None
        self.unexplored = []

    def play(self, board: Board) -> None:
        """uct tree search"""
        color = board.color
        node: UCTNode = self
        path: List[UCTNode] = [node]
        while True:
            pos = node.select(board)
            if pos == PASS:
                break
            board.move(pos)
            child = node.pos_child[pos]
            if not child:
                child = UCTNode()
                node.pos_child[pos] = child
                child.unexplored = board.useful_moves()
                child.pos = pos
                child.parent = node
                path.append(child)
                break
            path.append(child)
            node = child
        self.random_playout(board)
        self.update_path(board, color, path)

    def select(self, board: Board) -> int:
        """select move; unexplored children first, then according to uct value"""
        if self.unexplored:
            i = random.randrange(len(self.unexplored))
            pos = self.unexplored[i]
            self.unexplored[i] = self.unexplored[-1]
            self.unexplored.pop()
            return pos
        elif self.bestchild:
            return self.bestchild.pos
        else:
            return PASS

    def random_playout(self, board: Board) -> None:
        """random play until both players pass"""
        for _ in range(MAXMOVES):
            if board.finished:
                break
            board.move(board.random_move())

    def update_path(self, board: Board, color: int, path: List[UCTNode]) -> None:
        """update win/loss count along path"""
        wins = board.score(BLACK) >= board.score(WHITE)
        for node in path:
            color = WHITE if color == BLACK else BLACK
            if wins == (color == BLACK):
                node.wins += 1
            else:
                node.losses += 1
            if node.parent:
                node.parent.bestchild = node.parent.best_child()

    def score(self) -> float:
        winrate = self.wins / float(self.wins + self.losses) if (self.wins + self.losses) > 0 else 0.0
        parentvisits = self.parent.wins + self.parent.losses if self.parent else 0
        if not parentvisits:
            return winrate
        nodevisits = self.wins + self.losses
        if nodevisits == 0:
            return winrate
        return winrate + math.sqrt(math.log(parentvisits) / (5 * nodevisits))

    def best_child(self) -> Optional[UCTNode]:
        maxscore: float = -1.0
        maxchild: Optional[UCTNode] = None
        for child in self.pos_child:
            if child and child.score() > maxscore:
                maxchild = child
                maxscore = child.score()
        return maxchild

    def best_visited(self) -> Optional[UCTNode]:
        maxvisits: int = -1
        maxchild: Optional[UCTNode] = None
        for child in self.pos_child:
            if child and (child.wins + child.losses) > maxvisits:
                maxvisits = child.wins + child.losses
                maxchild = child
        return maxchild

def computer_move(board: Board) -> int:
    pos = board.random_move()
    if pos == PASS:
        return PASS
    tree = UCTNode()
    tree.unexplored = board.useful_moves()
    nboard = Board()
    for _ in range(GAMES):
        node = tree
        nboard.reset()
        nboard.replay(board.history)
        node.play(nboard)
    best = tree.best_visited()
    return best.pos if best else PASS

def versus_cpu() -> int:
    random.seed(1)
    board = Board()
    return computer_move(board)

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Test the performance of the Go benchmark'
    runner.bench_func('go', versus_cpu)
