# %%
from wavefunction_collapse.tiles import load_tiles, render_tiles
from wavefunction_collapse.generator import generate
import numpy as np

tiles, constraints, tile_weights = load_tiles("../../tiles/road")
n_tiles = tiles.shape[0]

map_shape = [20, 40]
tile_map = np.ones(map_shape, dtype=int) * n_tiles

# Initial features: a circle of water in the center
circle_radius = 5
row = np.arange(map_shape[0])[:, None]
col = np.arange(map_shape[1])[None, :]
is_in_circle = (row - map_shape[0] / 2) ** 2 + (
    col - map_shape[1] / 2
) ** 2 < circle_radius**2
tile_map[is_in_circle] = 15

# and a triangle of grass in the top left
tile_map[row + col < 10] = 0

# Generate rest of the map
tile_map = generate(tile_map, constraints, tile_weights)
image = render_tiles(tile_map, tiles)
image.show()
image.save("output.png")
