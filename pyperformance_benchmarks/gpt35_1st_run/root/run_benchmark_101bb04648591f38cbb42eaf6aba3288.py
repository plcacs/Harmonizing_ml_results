from typing import List, Tuple

class Dir:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class Done:
    MIN_CHOICE_STRATEGY: int = 0
    MAX_CHOICE_STRATEGY: int = 1
    HIGHEST_VALUE_STRATEGY: int = 2
    FIRST_STRATEGY: int = 3
    MAX_NEIGHBORS_STRATEGY: int = 4
    MIN_NEIGHBORS_STRATEGY: int = 5

    def __init__(self, count: int, empty: bool = False):
        self.count = count
        self.cells = (None if empty else [[0, 1, 2, 3, 4, 5, 6, EMPTY] for i in range(count)])

    def clone(self) -> 'Done':
        ret = Done(self.count, True)
        ret.cells = [self.cells[i][:] for i in range(self.count)]
        return ret

    def __getitem__(self, i: int) -> List[int]:
        return self.cells[i]

    def set_done(self, i: int, v: int):
        self.cells[i] = [v]

    def already_done(self, i: int) -> bool:
        return (len(self.cells[i]) == 1)

    def remove(self, i: int, v: int) -> bool:
        if (v in self.cells[i]):
            self.cells[i].remove(v)
            return True
        else:
            return False

    def remove_all(self, v: int):
        for i in range(self.count):
            self.remove(i, v)

    def remove_unfixed(self, v: int) -> bool:
        changed = False
        for i in range(self.count):
            if (not self.already_done(i)):
                if self.remove(i, v):
                    changed = True
        return changed

    def filter_tiles(self, tiles: List[int]):
        for v in range(8):
            if (tiles[v] == 0):
                self.remove_all(v)

    def next_cell_min_choice(self) -> int:
        minlen = 10
        mini = -1
        for i in range(self.count):
            if (1 < len(self.cells[i]) < minlen):
                minlen = len(self.cells[i])
                mini = i
        return mini

    def next_cell_max_choice(self) -> int:
        maxlen = 1
        maxi = -1
        for i in range(self.count):
            if (maxlen < len(self.cells[i])):
                maxlen = len(self.cells[i])
                maxi = i
        return maxi

    def next_cell_highest_value(self) -> int:
        maxval = -1
        maxi = -1
        for i in range(self.count):
            if (not self.already_done(i)):
                maxvali = max((k for k in self.cells[i] if (k != EMPTY)))
                if (maxval < maxvali):
                    maxval = maxvali
                    maxi = i
        return maxi

    def next_cell_first(self) -> int:
        for i in range(self.count):
            if (not self.already_done(i)):
                return i
        return -1

    def next_cell_max_neighbors(self, pos) -> int:
        maxn = -1
        maxi = -1
        for i in range(self.count):
            if (not self.already_done(i)):
                cells_around = pos.hex.get_by_id(i).links
                n = sum(((1 if (self.already_done(nid) and (self[nid][0] != EMPTY)) else 0) for nid in cells_around))
                if (n > maxn):
                    maxn = n
                    maxi = i
        return maxi

    def next_cell_min_neighbors(self, pos) -> int:
        minn = 7
        mini = -1
        for i in range(self.count):
            if (not self.already_done(i)):
                cells_around = pos.hex.get_by_id(i).links
                n = sum(((1 if (self.already_done(nid) and (self[nid][0] != EMPTY)) else 0) for nid in cells_around))
                if (n < minn):
                    minn = n
                    mini = i
        return mini

    def next_cell(self, pos, strategy: int = HIGHEST_VALUE_STRATEGY) -> int:
        if (strategy == Done.HIGHEST_VALUE_STRATEGY):
            return self.next_cell_highest_value()
        elif (strategy == Done.MIN_CHOICE_STRATEGY):
            return self.next_cell_min_choice()
        elif (strategy == Done.MAX_CHOICE_STRATEGY):
            return self.next_cell_max_choice()
        elif (strategy == Done.FIRST_STRATEGY):
            return self.next_cell_first()
        elif (strategy == Done.MAX_NEIGHBORS_STRATEGY):
            return self.next_cell_max_neighbors(pos)
        elif (strategy == Done.MIN_NEIGHBORS_STRATEGY):
            return self.next_cell_min_neighbors(pos)
        else:
            raise Exception(('Wrong strategy: %d' % strategy))

class Node:
    def __init__(self, pos, id, links):
        self.pos = pos
        self.id = id
        self.links = links

class Hex:
    def __init__(self, size: int):
        self.size = size
        self.count = (((3 * size) * (size - 1)) + 1)
        self.nodes_by_id = (self.count * [None])
        self.nodes_by_pos = {}
        id = 0
        for y in range(size):
            for x in range((size + y)):
                pos = (x, y)
                node = Node(pos, id, [])
                self.nodes_by_pos[pos] = node
                self.nodes_by_id[node.id] = node
                id += 1
        for y in range(1, size):
            for x in range(y, ((size * 2) - 1)):
                ry = ((size + y) - 1)
                pos = (x, ry)
                node = Node(pos, id, [])
                self.nodes_by_pos[pos] = node
                self.nodes_by_id[node.id] = node
                id += 1

    def link_nodes(self):
        for node in self.nodes_by_id:
            (x, y) = node.pos
            for dir in DIRS:
                nx = (x + dir.x)
                ny = (y + dir.y)
                if self.contains_pos((nx, ny)):
                    node.links.append(self.nodes_by_pos[(nx, ny)].id)

    def contains_pos(self, pos: Tuple[int, int]) -> bool:
        return (pos in self.nodes_by_pos)

    def get_by_pos(self, pos: Tuple[int, int]) -> Node:
        return self.nodes_by_pos[pos]

    def get_by_id(self, id: int) -> Node:
        return self.nodes_by_id[id]

class Pos:
    def __init__(self, hex: Hex, tiles: List[int], done: Done = None):
        self.hex = hex
        self.tiles = tiles
        self.done = (Done(hex.count) if (done is None) else done)

    def clone(self) -> 'Pos':
        return Pos(self.hex, self.tiles, self.done.clone())

def constraint_pass(pos: Pos, last_move: int = None) -> bool:
    changed = False
    left = pos.tiles[:]
    done = pos.done
    free_cells = (range(done.count) if (last_move is None) else pos.hex.get_by_id(last_move).links)
    for i in free_cells:
        if (not done.already_done(i)):
            vmax = 0
            vmin = 0
            cells_around = pos.hex.get_by_id(i).links
            for nid in cells_around:
                if done.already_done(nid):
                    if (done[nid][0] != EMPTY):
                        vmin += 1
                        vmax += 1
                else:
                    vmax += 1
            for num in range(7):
                if ((num < vmin) or (num > vmax)):
                    if done.remove(i, num):
                        changed = True
    for cell in done.cells:
        if (len(cell) == 1):
            left[cell[0]] -= 1
    for v in range(8):
        if ((pos.tiles[v] > 0) and (left[v] == 0)):
            if done.remove_unfixed(v):
                changed = True
        else:
            possible = sum(((1 if (v in cell) else 0) for cell in done.cells))
            if (pos.tiles[v] == possible):
                for i in range(done.count):
                    cell = done.cells[i]
                    if ((not done.already_done(i)) and (v in cell)):
                        done.set_done(i, v)
                        changed = True
    filled_cells = (range(done.count) if (last_move is None) else [last_move])
    for i in filled_cells:
        if done.already_done(i):
            num = done[i][0]
            empties = 0
            filled = 0
            unknown = []
            cells_around = pos.hex.get_by_id(i).links
            for nid in cells_around:
                if done.already_done(nid):
                    if (done[nid][0] == EMPTY):
                        empties += 1
                    else:
                        filled += 1
                else:
                    unknown.append(nid)
            if (len(unknown) > 0):
                if (num == filled):
                    for u in unknown:
                        if (EMPTY in done[u]):
                            done.set_done(u, EMPTY)
                            changed = True
                elif (num == (filled + len(unknown))):
                    for u in unknown:
                        if done.remove(u, EMPTY):
                            changed = True
    return changed

ASCENDING: int = 1
DESCENDING: int = -1

def find_moves(pos: Pos, strategy: int, order: int) -> List[Tuple[int, int]]:
    done = pos.done
    cell_id = done.next_cell(pos, strategy)
    if (cell_id < 0):
        return []
    if (order == ASCENDING):
        return [(cell_id, v) for v in done[cell_id]]
    else:
        moves = list(reversed([(cell_id, v) for v in done[cell_id] if (v != EMPTY)]))
        if (EMPTY in done[cell_id]):
            moves.append((cell_id, EMPTY))
        return moves

def play_move(pos: Pos, move: Tuple[int, int]):
    (cell_id, i) = move
    pos.done.set_done(cell_id, i)

def print_pos(pos: Pos, output):
    hex = pos.hex
    done = pos.done
    size = hex.size
    for y in range(size):
        print((' ' * ((size - y) - 1)), end='', file=output)
        for x in range((size + y)):
            pos2 = (x, y)
            id = hex.get_by_pos(pos2).id
            if done.already_done(id):
                c = (str(done[id][0]) if (done[id][0] != EMPTY) else '.')
            else:
                c = '?'
            print(('%s ' % c), end='', file=output)
        print(end='\n', file=output)
    for y in range(1, size):
        print((' ' * y), end='', file=output)
        for x in range(y, ((size * 2) - 1)):
            ry = ((size + y) - 1)
            pos2 = (x, ry)
            id = hex.get_by_pos(pos2).id
            if done.already_done(id):
                c = (str(done[id][0]) if (done[id][0] != EMPTY) else '.')
            else:
                c = '?'
            print(('%s ' % c), end='', file=output)
        print(end='\n', file=output)

OPEN: int = 0
SOLVED: int = 1
IMPOSSIBLE: int = -1

def solved(pos: Pos, output, verbose: bool = False) -> int:
    hex = pos.hex
    tiles = pos.tiles[:]
    done = pos.done
    exact = True
    all_done = True
    for i in range(hex.count):
        if (len(done[i]) == 0):
            return IMPOSSIBLE
        elif done.already_done(i):
            num = done[i][0]
            tiles[num] -= 1
            if (tiles[num] < 0):
                return IMPOSSIBLE
            vmax = 0
            vmin = 0
            if (num != EMPTY):
                cells_around = hex.get_by_id(i).links
                for nid in cells_around:
                    if done.already_done(nid):
                        if (done[nid][0] != EMPTY):
                            vmin += 1
                            vmax += 1
                    else:
                        vmax += 1
                if ((num < vmin) or (num > vmax)):
                    return IMPOSSIBLE
                if (num != vmin):
                    exact = False
        else:
            all_done = False
    if ((not all_done) or (not exact)):
        return OPEN
    print_pos(pos, output)
    return SOLVED

def solve_step(prev: Pos, strategy: int, order: int, output, first: bool = False) -> int:
    if first:
        pos = prev.clone()
        while constraint_pass(pos):
            pass
    else:
        pos = prev
    moves = find_moves(pos, strategy, order)
    if (len(moves) == 0):
        return solved(pos, output)
    else:
        for move in moves:
            ret = OPEN
            new_pos = pos.clone()
            play_move(new_pos, move)
            while constraint_pass(new_pos, move[0]):
                pass
            cur_status = solved(new_pos, output)
            if (cur_status != OPEN):
                ret = cur_status
            else:
                ret = solve_step(new_pos, strategy, order, output)
            if (ret == SOLVED):
                return SOLVED
    return IMPOSSIBLE

def check_valid(pos: Pos):
    hex = pos.hex
    tiles = pos.tiles
    tot = 0
    for i in range(8):
        if (tiles[i] > 0):
            tot += tiles[i]
        else:
            tiles[i] = 0
    if (tot != hex.count):
        raise Exception(('Invalid input. Expected %d tiles, got %d.' % (hex.count, tot)))

def solve(pos: Pos, strategy: int, order: int, output) -> int:
    check_valid(pos)
    return solve_step(pos, strategy, order, output, first=True)

def read_file(file: str) -> Pos:
    lines = [line.strip('\r\n') for line in file.splitlines()]
    size = int(lines[0])
    hex = Hex(size)
    linei = 1
    tiles = [0, 0, 0, 0, 0, 0, 0, 0]
    done = Done(hex.count)
    for y in range(size):
        line = lines[linei][((size - y) - 1):]
        p = 0
        for x in range((size + y)):
            tile = line[p:(p + 2)]
            p += 2
            if (tile[1] == '.'):
                inctile = EMPTY
            else:
                inctile = int(tile)
            tiles[inctile] += 1
            if (tile[0] == '+'):
                done.set_done(hex.get_by_pos((x, y)).id, inctile)
        linei += 1
    for y in range(1, size):
        ry = ((size - 1) + y)
        line = lines[linei][y:]
        p = 0
        for x in range(y, ((size * 2) - 1)):
            tile = line[p:(p + 2)]
            p += 2
            if (tile[1] == '.'):
                inctile = EMPTY
            else:
                inctile = int(tile)
            tiles[inctile] += 1
            if (tile[0] == '+'):
                done.set_done(hex.get_by_pos((x, ry)).id, inctile)
        linei += 1
    hex.link_nodes()
    done.filter_tiles(tiles)
    return Pos(hex, tiles, done)

def solve_file(file: str, strategy: int, order: int, output):
    pos = read_file(file)
    solve(pos, strategy, order, output)

LEVELS: dict = {}
LEVELS[2] = ('\n2\n  . 1\n . 1 1\n  1 .\n', ' 1 1\n. . .\n 1 1\n')
LEVELS[10] = ('\n3\n  +.+. .\n +. 0 . 2\n . 1+2 1 .\n  2 . 0+.\n   .+.+.\n', '  . . 1\n . 1 . 2\n0 . 2 2 .\n . . . .\n  0 . .\n')
LEVELS[20] = ('\n3\n   . 5 4\n  . 2+.+1\n . 3+2 3 .\n +2+. 5 .\n   . 3 .\n', '  3 3 2\n 4 5 . 1\n3 5 2 . .\n 2 . . .\n  . . .\n')
LEVELS[25] = ('\n3\n   4 . .\n  . . 2 .\n 4 3 2 . 4\n  2 2 3 .\n   4 2 4\n', '  3 4 2\n 2 4 4 .\n. . . 4 2\n . 2 4 3\n  . 2 .\n')
LEVELS[30] = ('\n4\n    5 5 . .\n   3 . 2+2 6\n  3 . 2 . 5 .\n . 3 3+4 4 . 3\n  4 5 4 . 5 4\n   5+2 . . 3\n    4 . . .\n', '   3 4 3 .\n  4 6 5 2 .\n 2 5 5 . . 2\n. . 5 4 . 4 3\n . 3 5 4 5 4\n  . 2 . 3 3\n   . . . .\n')
LEVELS[36] = ('\n4\n    2 1 1 2\n   3