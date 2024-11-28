import numpy as np


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
    idx = np.unravel_index(
        np.argmin(n_possible + n_tiles * (1 - undecided)), n_possible.shape
    )
    return idx


def get_possible_tiles(tile_map, padded_constraints):
    # tile_map: [width, height]
    # padded_constraints: [n_tiles+1, 4, n_tiles+1]
    n_tiles = padded_constraints.shape[0] - 1
    possible = np.ones(tile_map.shape + (n_tiles + 1,), dtype=bool)
    possible[:-1, :] &= padded_constraints[tile_map[1:, :], 0]
    possible[1:, :] &= padded_constraints[tile_map[:-1, :], 2]
    possible[:, 1:] &= padded_constraints[tile_map[:, :-1], 1]
    possible[:, :-1] &= padded_constraints[tile_map[:, 1:], 3]
    return possible


def generate(tile_map, constraints, tile_weights):
    map_shape = tile_map.shape
    rows, cols = map_shape
    n_tiles = constraints.shape[0]

    initial_tile_map = tile_map.copy()
    constraints = add_undecided_auxilliary_tile(constraints)
    possible = get_possible_tiles(tile_map, constraints)

    generated_tiles = []
    backtrack_constraints = np.ones(map_shape + (n_tiles + 1,), dtype=bool)

    while True:
        # TODO: user better data structure to keep track of undecided tiles
        # Maybe a heap with the number of possible tiles as the key?
        row, col = get_next_tile(tile_map, possible)
        if tile_map[row, col] != n_tiles:
            # No undecided tiles left: Done
            break
        while possible[row, col].sum() <= 1:
            # Only undecided tile possible. Backtrack the last move
            # print(f"Backtracking after {len(generated_tiles)} tiles.")
            row, col, prev_tile = generated_tiles.pop()
            tile_map[row, col] = n_tiles
            backtrack_constraints[row, col, prev_tile] = False
            # TODO: avoid fully recomputing possible
            possible = get_possible_tiles(tile_map, constraints) & backtrack_constraints

        # Randomly select a tile from the possible tiles
        possible_new_tiles = np.where(possible[row, col, :-1])[0]
        possible_weights = tile_weights[possible_new_tiles]
        possible_weights /= possible_weights.sum()
        new_tile = np.random.choice(possible_new_tiles, p=possible_weights)

        # Update neighbour constraints
        generated_tiles.append((row, col, new_tile))
        tile_map[row, col] = new_tile
        if row > 0:
            possible[row - 1, col, :] &= constraints[new_tile, 0]
        if row < rows - 1:
            possible[row + 1, col, :] &= constraints[new_tile, 2]
        if col > 0:
            possible[row, col - 1, :] &= constraints[new_tile, 3]
        if col < cols - 1:
            possible[row, col + 1, :] &= constraints[new_tile, 1]

    return tile_map
