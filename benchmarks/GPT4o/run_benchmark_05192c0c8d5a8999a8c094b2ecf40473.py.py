'\nGo board game\n'
import math
import random
import pyperf
from typing import List, Tuple, Optional

SIZE: int = 9
GAMES: int = 200
KOMI: float = 7.5
(EMPTY, WHITE, BLACK) = (0, 1, 2)
SHOW = {EMPTY: '.', WHITE: 'o', BLACK: 'x'}
PASS: int = -1
MAXMOVES: int = (SIZE * SIZE) * 3
TIMESTAMP: int = 0
MOVES: int = 0

def to_pos(x: int, y: int) -> int:
    return (y * SIZE) + x

def to_xy(pos: int) -> Tuple[int, int]:
    (y, x) = divmod(pos, SIZE)
    return (x, y)

class Square:
    def __init__(self, board: 'Board', pos: int) -> None:
        self.board = board
        self.pos = pos
        self.timestamp = TIMESTAMP
        self.removestamp = TIMESTAMP
        self.zobrist_strings = [random.randrange(9223372036854775807) for i in range(3)]
        self.color: int = EMPTY
        self.reference: 'Square' = self
        self.ledges: int = 0
        self.used: bool = False
        self.neighbours: List['Square'] = []

    def set_neighbours(self) -> None:
        (x, y) = (self.pos % SIZE, self.pos // SIZE)
        self.neighbours = []
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            (newx, newy) = (x + dx, y + dy)
            if (0 <= newx < SIZE) and (0 <= newy < SIZE):
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

    def remove(self, reference: 'Square', update: bool = True) -> None:
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

    def find(self, update: bool = False) -> 'Square':
        reference = self.reference
        if reference.pos != self.pos:
            reference = reference.find(update)
            if update:
                self.reference = reference
        return reference

    def __repr__(self) -> str:
        return repr(to_xy(self.pos))

class EmptySet:
    def __init__(self, board: 'Board') -> None:
        self.board = board
        self.empties: List[int] = list(range(SIZE * SIZE))
        self.empty_pos: List[int] = list(range(SIZE * SIZE))

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
        self.set(self.empty_pos[pos], self.empties[len(self.empties) - 1])
        self.empties.pop()

    def set(self, i: int, pos: int) -> None:
        self.empties[i] = pos
        self.empty_pos[pos] = i

class ZobristHash:
    def __init__(self, board: 'Board') -> None:
        self.board = board
        self.hash_set: set = set()
        self.hash: int = 0
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
    def __init__(self) -> None:
        self.squares: List[Square] = [Square(self, pos) for pos in range(SIZE * SIZE)]
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
        self.history: List[int] = []
        self.white_dead = 0
        self.black_dead = 0

    def move(self, pos: int) -> None:
        square = self.squares[pos]
        if pos != PASS:
            square.move(self.color)
            self.emptyset.remove(square.pos)
        elif self.lastmove == PASS:
            self.finished = True
        if self.color == BLACK:
            self.color = WHITE
        else:
            self.color = BLACK
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
        return (not dupe) and (empties or weak_opps or (strong_neighs and (strong_opps or weak_neighs)))

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
                surround = 0
                for neighbour in square.neighbours:
                    if neighbour.color == color:
                        surround += 1
                if surround == len(square.neighbours):
                    count += 1
        return count

    def check(self) -> None:
        for square in self.squares:
            if square.color == EMPTY:
                continue
            members1 = set([square])
            changed = True
            while changed:
                changed = False
                for member in members1.copy():
                    for neighbour in member.neighbours:
                        if neighbour.color == square.color and neighbour not in members1:
                            changed = True
                            members1.add(neighbour)
            ledges1 = 0
            for member in members1:
                for neighbour in member.neighbours:
                    if neighbour.color == EMPTY:
                        ledges1 += 1
            root = square.find()
            members2 = set()
            for square2 in self.squares:
                if square2.color != EMPTY and square2.find() == root:
                    members2.add(square2)
            ledges2 = root.ledges
            assert members1 == members2
            assert ledges1 == ledges2, ('ledges differ at %r: %d %d' % (square, ledges1, ledges2))
            set(self.emptyset.empties)
            empties2 = set()
            for square in self.squares:
                if square.color == EMPTY:
                    empties2.add(square.pos)

    def __repr__(self) -> str:
        result = []
        for y in range(SIZE):
            start = to_pos(0, y)
            result.append(''.join([SHOW[square.color] + ' ' for square in self.squares[start:(start + SIZE)]]))
        return '\n'.join(result)

class UCTNode:
    def __init__(self) -> None:
        self.bestchild: Optional[UCTNode] = None
        self.pos: int = -1
        self.wins: int = 0
        self.losses: int = 0
        self.pos_child: List[Optional[UCTNode]] = [None for x in range(SIZE * SIZE)]
        self.parent: Optional[UCTNode] = None
        self.unexplored: List[int] = []

    def play(self, board: Board) -> None:
        ' uct tree search '
        color = board.color
        node = self
        path = [node]
        while True:
            pos = node.select(board)
            if pos == PASS:
                break
            board.move(pos)
            child = node.pos_child[pos]
            if not child:
                child = node.pos_child[pos] = UCTNode()
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
        ' select move; unexplored children first, then according to uct value '
        if self.unexplored:
            i = random.randrange(len(self.unexplored))
            pos = self.unexplored[i]
            self.unexplored[i] = self.unexplored[len(self.unexplored) - 1]
            self.unexplored.pop()
            return pos
        elif self.bestchild:
            return self.bestchild.pos
        else:
            return PASS

    def random_playout(self, board: Board) -> None:
        ' random play until both players pass '
        for x in range(MAXMOVES):
            if board.finished:
                break
            board.move(board.random_move())

    def update_path(self, board: Board, color: int, path: List['UCTNode']) -> None:
        ' update win/loss count along path '
        wins = board.score(BLACK) >= board.score(WHITE)
        for node in path:
            if color == BLACK:
                color = WHITE
            else:
                color = BLACK
            if wins == (color == BLACK):
                node.wins += 1
            else:
                node.losses += 1
            if node.parent:
                node.parent.bestchild = node.parent.best_child()

    def score(self) -> float:
        winrate = self.wins / float(self.wins + self.losses)
        parentvisits = self.parent.wins + self.parent.losses
        if not parentvisits:
            return winrate
        nodevisits = self.wins + self.losses
        return winrate + math.sqrt(math.log(parentvisits) / (5 * nodevisits))

    def best_child(self) -> Optional['UCTNode']:
        maxscore = -1.0
        maxchild = None
        for child in self.pos_child:
            if child and child.score() > maxscore:
                maxchild = child
                maxscore = child.score()
        return maxchild

    def best_visited(self) -> Optional['UCTNode']:
        maxvisits = -1
        maxchild = None
        for child in self.pos_child:
            if child and (child.wins + child.losses) > maxvisits:
                maxvisits, maxchild = (child.wins + child.losses), child
        return maxchild

def computer_move(board: Board) -> int:
    pos = board.random_move()
    if pos == PASS:
        return PASS
    tree = UCTNode()
    tree.unexplored = board.useful_moves()
    nboard = Board()
    for game in range(GAMES):
        node = tree
        nboard.reset()
        nboard.replay(board.history)
        node.play(nboard)
    return tree.best_visited().pos

def versus_cpu() -> int:
    random.seed(1)
    board = Board()
    return computer_move(board)

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Test the performance of the Go benchmark'
    runner.bench_func('go', versus_cpu)
