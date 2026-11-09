#!/usr/bin/env python3
"""Extract a contact sheet from completed review videos."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('videos', nargs='+', type=Path)
    a = p.parse_args()
    canvas = Image.new('RGB', (480 * 3, 320 * len(a.videos)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, path in enumerate(a.videos):
        with imageio.get_reader(str(path)) as reader:
            metadata = reader.get_meta_data()
            duration = metadata['duration']
            for col, fraction in enumerate((.2, .5, .9)):
                time = duration * fraction
                frame = reader.get_data(int(time * metadata['fps']))
                canvas.paste(Image.fromarray(frame).resize((480, 288)), (col*480, row*320+32))
                draw.text((col*480+5, row*320+5), f'{path.parent.name} | {time:.2f}s', fill='black')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.output)


if __name__ == '__main__':
    main()
