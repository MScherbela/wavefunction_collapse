import argparse
from wavefunction_collapse.generator import generate
from wavefunction_collapse.tiles import load_tiles, render_tiles
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a tile map")
    parser.add_argument("--tiles", type=str, help="Directory containing tiles", default="")
    parser.add_argument("--output", "-o", type=str, help="Output file", default="output.png")
    parser.add_argument("--width", type=int, help="Width of the map (in tiles)", default=40)
    parser.add_argument("--height", type=int, help="Height of the map (in tiles)", default=30)
    args = parser.parse_args()
    if args.tiles == "":
        args.tiles = Path(__file__).parent.parent.parent / "tiles" / "example"
    return args


def main():
    args = parse_args()
    tiles, constraints, tile_weights = load_tiles(args.tiles)
    tile_map = generate(None, constraints, tile_weights, rows=args.height, cols=args.width)
    image = render_tiles(tile_map, tiles)
    image.save(args.output)
