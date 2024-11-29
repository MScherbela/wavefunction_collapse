import numpy as np
import dataclasses
import functools


def add_undecided_auxilliary_tile(constraints):
    """Add an auxilliary tile to the constraints that is undecided and always allowed.

    Changes the shape of constraings from (n_tiles, 4, n_tiles) to (n_tiles + 1, 4, n_tiles + 1).
    """
    n_tiles = constraints.shape[0]
    padded = np.ones((n_tiles + 1, 4, n_tiles + 1), dtype=bool)
    padded[:n_tiles, :, :n_tiles] = constraints
    return padded


def get_next_tile(tile_map, possible):
    # possible: [width, height, n_tiles+1]
    n_tiles = possible.shape[2] - 1
    undecided = tile_map == n_tiles
    n_possible = possible.sum(axis=2) - 1  # -1 to exclude the undecided tile
    idx = np.unravel_index(np.argmin(n_possible + n_tiles * (1 - undecided)), n_possible.shape)
    return idx


def get_all_possible_tiles(tile_map, constraints):
    # tile_map: [width, height]
    # constraints: [n_tiles, 4, n_tiles]

    # Pad the constraints, so we can also index with the undecided tile,
    # which will allow all neighbours in all directions)
    n_tiles = constraints.shape[0]
    padding = np.ones([1, 4, n_tiles], dtype=bool)
    padded_constraints = np.concatenate([constraints, padding], axis=0)

    possible = np.ones(tile_map.shape + (n_tiles,), dtype=bool)
    possible[:-1, :] &= padded_constraints[tile_map[1:, :], 0]
    possible[1:, :] &= padded_constraints[tile_map[:-1, :], 2]
    possible[:, 1:] &= padded_constraints[tile_map[:, :-1], 1]
    possible[:, :-1] &= padded_constraints[tile_map[:, 1:], 3]
    return possible


def get_possible_tiles(tile_map, constraints, r, c):
    n_tiles = constraints.shape[0]
    rows, cols = tile_map.shape
    possible = np.ones((n_tiles,), dtype=bool)
    if (r > 0) and tile_map[r - 1, c] != n_tiles:
        possible &= constraints[tile_map[r - 1, c], 2]
    if (r < rows - 1) and tile_map[r + 1, c] != n_tiles:
        possible &= constraints[tile_map[r + 1, c], 0]
    if (c > 0) and tile_map[r, c - 1] != n_tiles:
        possible &= constraints[tile_map[r, c - 1], 1]
    if (c < cols - 1) and tile_map[r, c + 1] != n_tiles:
        possible &= constraints[tile_map[r, c + 1], 3]
    return possible


# @functools.total_ordering
# @dataclasses.dataclass
# class Tile:
#     r: int
#     c: int
#     n_possible: np.ndarray

#     def _priority(self):
#         return (self.n_possible[self.r, self.c], self.r, self.c)

#     def __lt__(self, other):
#         return self._priority() < other._priority()

#     def __eq__(self, other):
#         return self._priority() == other._priority()


class Frontier:
    def __init__(self, n_possible, tile_map, n_tiles):
        self.rows, self.cols = n_possible.shape
        self.n_tiles = n_tiles
        self.size = 0

        self.frontier = np.ones([self.rows * self.cols, 2], dtype=int) * 1000
        self.n_possible = n_possible
        self.is_in_frontier = np.zeros([self.rows, self.cols], dtype=bool)
        self.tile_map = tile_map

    def push(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return
        if self.is_in_frontier[r, c]:
            # Already in the frontier
            return
        if self.tile_map[r, c] != self.n_tiles:
            # Already decided
            return
        self.is_in_frontier[r, c] = True
        self.frontier[self.size] = r, c
        self.size += 1

    def get_idx_min(self):
        n_possible = self.n_possible[self.frontier[: self.size, 0], self.frontier[: self.size, 1]]
        priority = (
            n_possible * self.rows * self.cols
            + self.frontier[: self.size, 0] * self.cols
            + self.frontier[: self.size, 1]
        )
        return np.argmin(priority)

    def pop(self):
        if self.size == 0:
            raise ValueError("Frontier is empty")
        idx_min = self.get_idx_min()
        r, c = self.frontier[idx_min]
        self.is_in_frontier[r, c] = False
        self.frontier[idx_min : self.size - 1] = self.frontier[idx_min + 1 : self.size]
        self.size -= 1
        return r, c

    def __len__(self):
        return self.size

    def __bool__(self):
        return self.size > 0

    def _assert_consistent(self):
        assert self.size <= len(self.frontier)
        for r, c in self.frontier[: self.size]:
            assert self.is_in_frontier[r, c]
            assert self.tile_map[r, c] == self.n_tiles


class PossibleTiles:
    def __init__(self, possible):
        self.possible = possible
        self.n_possible = possible.sum(axis=2)
        self.allowed = np.ones_like(possible, dtype=bool)
        self.rows, self.cols = possible.shape[:2]

    def block(self, r, c, tile):
        self.allowed[r, c, tile] = False
        self.possible[r, c, tile] = False
        self.n_possible[r, c] = self.possible[r, c].sum()

    def set(self, r, c, possible):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return
        self.possible[r, c] = possible & self.allowed[r, c]
        self.n_possible[r, c] = self.possible[r, c].sum()

    def constrain(self, r, c, possible):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return
        self.set(r, c, self.possible[r, c] & possible)

    def constrain_neighbours(self, r, c, tile, constraints):
        self.constrain(r - 1, c, constraints[tile, 0])
        self.constrain(r + 1, c, constraints[tile, 2])
        self.constrain(r, c + 1, constraints[tile, 1])
        self.constrain(r, c - 1, constraints[tile, 3])

    def _assert_consistent(self):
        assert (self.possible.sum(axis=2) == self.n_possible).all()


def generate(tile_map, constraints, tile_weights, rows=None, cols=None):
    n_tiles = constraints.shape[0]

    if tile_map is None:
        if rows is None or cols is None:
            raise ValueError("Either tile_map or both n_rows and n_cols must be provided")
        tile_map = np.ones([rows, cols], dtype=int) * n_tiles
    else:
        rows, cols = tile_map.shape

    possible_tiles = PossibleTiles(get_all_possible_tiles(tile_map, constraints))
    frontier = Frontier(possible_tiles.n_possible, tile_map, n_tiles)
    for r in range(rows):
        for c in range(cols):
            if possible_tiles.n_possible[r, c] < n_tiles:
                frontier.push(r, c)
    if len(frontier) == 0:
        frontier.push(0, 0)

    generated_tiles = []
    while frontier:
        row, col = frontier.pop()
        n_pos = possible_tiles.n_possible[row, col]
        while n_pos == 0:
            # No tile possible to place. Re-adding currently unsolvable tile to the frontier
            frontier.push(row, col)

            # Backtrack, by undoing the last decision
            row, col, prev_tile = generated_tiles.pop()

            # Mark tile as undecided and add it to the frontier
            tile_map[row, col] = n_tiles
            frontier.push(row, col)

            # Prohibit from making the same choice again
            possible_tiles.block(row, col, prev_tile)

            # Relax neighbour constraints (given that the previous tile is no longer placed)
            for r, c in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                if (0 <= r < rows) and (0 <= c < cols):
                    possible_tiles.set(r, c, get_possible_tiles(tile_map, constraints, r, c))

            # Get the next tile to decide
            row, col = frontier.pop()
            n_pos = possible_tiles.n_possible[row, col]

        # Randomly select a tile from the possible tiles
        possible_new_tiles = np.where(possible_tiles.possible[row, col])[0]
        possible_weights = tile_weights[possible_new_tiles]
        possible_weights /= possible_weights.sum()
        new_tile = np.random.choice(possible_new_tiles, p=possible_weights)

        # Update neighbour constraints and add undecided neighbours to the frontier
        generated_tiles.append((row, col, new_tile))
        tile_map[row, col] = new_tile
        possible_tiles.constrain_neighbours(row, col, new_tile, constraints)
        if row > 0:
            frontier.push(row - 1, col)
        if row < rows - 1:
            frontier.push(row + 1, col)
        if col > 0:
            frontier.push(row, col - 1)
        if col < cols - 1:
            frontier.push(row, col + 1)

    return tile_map
