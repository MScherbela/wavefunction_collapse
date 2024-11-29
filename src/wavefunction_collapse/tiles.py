import pathlib
import numpy as np
from PIL import Image
import yaml


def load_tiles(tile_directory):
    tile_directory = pathlib.Path(tile_directory)
    tile_fnames = tile_directory.glob("*.png")
    tiles = {}
    for fname in tile_fnames:
        tiles[int(fname.stem)] = np.array(Image.open(fname))
    if len(tiles) == 0:
        raise ValueError(f"No tiles found in {tile_directory}")

    n_tiles = max(tiles.keys()) + 1
    for i in range(n_tiles):
        if i not in tiles:
            raise ValueError(f"Missing tile {i}")

    with open(tile_directory / "constraints.yaml") as f:
        tile_data = yaml.safe_load(f)

    tile_weights = np.zeros(n_tiles)
    tile_borders = {}
    for i, data in tile_data.items():
        i = int(i)
        tile_weights[i] = data.get("weight", 1.0)
        tile_borders[i] = data["borders"]

    possible_neighbours = np.zeros([n_tiles, 4, n_tiles], dtype=bool)
    for i in range(n_tiles):
        for n in range(n_tiles):
            for direction in range(4):
                possible_neighbours[i, direction, n] = (
                    tile_borders[i][direction] == tile_borders[n][(direction + 2) % 4]
                )

    tiles = np.stack([tiles[i] for i in range(n_tiles)])
    return tiles, possible_neighbours, tile_weights


def render_tiles(tile_map, tiles, as_numpy=False):
    tile_shape = tiles[0].shape
    blank_tile = np.ones(tile_shape, dtype=np.uint8) * 255
    tiles = np.concatenate([tiles, [blank_tile]], axis=0)
    image = tiles[tile_map]  # [width, height, tile_width, tile_height, channels]
    image = np.swapaxes(image, 1, 2)  # [width, tile_width, height, tile_height, channels]
    image = np.reshape(
        image,
        [
            image.shape[0] * image.shape[1],
            image.shape[2] * image.shape[3],
            image.shape[4],
        ],
    )  # [width * tile_width, height * tile_height, channels]
    if not as_numpy:
        image = Image.fromarray(image)
    return image
