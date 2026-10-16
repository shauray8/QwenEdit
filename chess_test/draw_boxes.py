"""Draw move-marker boxes (or a separate region mask) on a top-down chess board image.

Usage:
  python draw_boxes.py IN OUT magenta=e2 cyan=e4 [yellow=h1 blue=f1] [--mask] [--crop X0,Y0,SIZE]

Squares are algebraic (a1 bottom-left, White at the bottom). The board is assumed to
fill the image (after the optional crop) as an 8x8 grid.
"""

import argparse

from PIL import Image, ImageDraw

COLORS = {
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "yellow": (255, 230, 0),
    "blue": (0, 90, 255),
}


def square_box(square, size):
    file, rank = "abcdefgh".index(square[0]), int(square[1])
    cell = size / 8
    x0, y0 = file * cell, (8 - rank) * cell
    return x0, y0, x0 + cell, y0 + cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inp")
    parser.add_argument("out")
    parser.add_argument("marks", nargs="+", help="color=square, e.g. magenta=e2")
    parser.add_argument("--mask", action="store_true", help="solid fills on black instead of outlines")
    parser.add_argument("--crop", help="X0,Y0,SIZE square board crop applied first")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--inset", type=int, default=4)
    args = parser.parse_args()

    image = Image.open(args.inp).convert("RGB")
    if args.crop:
        x0, y0, size = map(int, args.crop.split(","))
        image = image.crop((x0, y0, x0 + size, y0 + size))
    size = min(image.size)
    canvas = Image.new("RGB", image.size, (0, 0, 0)) if args.mask else image.copy()
    draw = ImageDraw.Draw(canvas)
    for mark in args.marks:
        color, square = mark.split("=")
        x0, y0, x1, y1 = square_box(square, size)
        if args.mask:
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=COLORS[color])
        else:
            i = args.inset
            draw.rectangle((x0 + i, y0 + i, x1 - 1 - i, y1 - 1 - i), outline=COLORS[color], width=args.width)
    canvas.save(args.out)


if __name__ == "__main__":
    main()
