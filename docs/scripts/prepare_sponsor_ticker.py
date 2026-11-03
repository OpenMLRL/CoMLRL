#!/usr/bin/env python3
"""Build the README sponsor ticker from the existing sponsor artwork.

Requires Pillow >= 10. Run from any directory:
    python3 docs/scripts/prepare_sponsor_ticker.py

The original collage stays unchanged. Combine its six logos with the supplied
Amazon Science and CloudBank artwork, normalize their optical sizes, and compose
a restrained monochrome marquee. All artwork is local; no fonts, downloads, or
runtime dependencies are needed to view the README.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps


DOCS = Path(__file__).resolve().parents[1]
SOURCE = DOCS / "assets" / "sponsors.png"
OUTPUT = DOCS / "assets" / "sponsors"
AMAZON_SOURCE = OUTPUT / "sources" / "amazon-science.png"
CLOUDBANK_SOURCE = OUTPUT / "sources" / "cloudbank.png"
WIDTH, HEIGHT, SCALE = 960, 96, 2
GAP = 88
STEP = 4  # Physical pixels per frame; constant speed across the loop boundary.
FRAME_MS = 50
COLORS = 32

# Source bounds separate the original logos; None uses the entire source image.
# Each size is an optical bounding box in CSS pixels, not a forced aspect ratio.
SPONSORS = (
    (
        "Northeastern University Research Computing",
        SOURCE,
        (0, 0, 1400, 330),
        (240, 50),
    ),
    ("Amazon Science", AMAZON_SOURCE, None, (200, 40)),
    ("Amazon Web Services", SOURCE, (1600, 0, 2270, 390), (76, 46)),
    ("CloudBank", CLOUDBANK_SOURCE, None, (200, 42)),
    ("NCSA", SOURCE, (2530, 0, 3890, 330), (160, 40)),
    ("Lambda", SOURCE, (60, 520, 1260, 806), (144, 34)),
    ("Vast.ai", SOURCE, (1410, 520, 2470, 806), (128, 38)),
    ("MGHPCC", SOURCE, (2610, 520, 3950, 806), (164, 36)),
)
THEMES = {
    "light": {"paper": (255, 255, 255), "ink": (91, 96, 110)},
    "dark": {"paper": (13, 17, 23), "ink": (181, 188, 203)},
}


def logo_masks():
    """Extract antialiased silhouettes, preserving each original aspect ratio."""
    masks = []
    for name, path, bounds, size in SPONSORS:
        with Image.open(path) as source:
            artwork = source.crop(bounds) if bounds is not None else source
            artwork = artwork.convert("RGBA")
            # Transparent source backgrounds must not become part of the logo.
            paper = Image.new("RGBA", artwork.size, "white")
            red, green, blue = (
                Image.alpha_composite(paper, artwork).convert("RGB").split()
            )
        # Saturated brand colors remain as legible as the black wordmarks.
        mask = ImageOps.invert(ImageChops.darker(ImageChops.darker(red, green), blue))
        bbox = mask.point(lambda value: 255 if value > 16 else 0).getbbox()
        if bbox is None:
            raise ValueError(f"Missing artwork for {name}")
        mask = mask.crop(bbox)
        target = tuple(value * SCALE for value in size)
        # Small supplied images also need the intended optical size at 2x.
        if mask.width < target[0] and mask.height < target[1]:
            mask = ImageOps.contain(mask, target, Image.Resampling.LANCZOS)
        else:
            mask.thumbnail(target, Image.Resampling.LANCZOS)
        masks.append(mask)
    return masks


def make_track(masks):
    period = sum(size[0] + GAP for _, _, _, size in SPONSORS) * SCALE
    if period % STEP:
        raise ValueError("The track must divide evenly into constant-speed steps")
    track = Image.new("L", (period, HEIGHT * SCALE))
    draw = ImageDraw.Draw(track)
    cursor = 0
    for mask, (_, _, _, size) in zip(masks, SPONSORS):
        slot = (size[0] + GAP) * SCALE
        track.paste(
            mask,
            (cursor + (slot - mask.width) // 2, (track.height - mask.height) // 2),
        )
        square = 3 * SCALE
        x, y = cursor + slot - square // 2, track.height // 2 - square // 2
        draw.rectangle((x, y, x + square - 1, y + square - 1), fill=74)
        if x + square > period:
            draw.rectangle((0, y, x + square - period - 1, y + square - 1), fill=74)
        cursor += slot
    repeated = Image.new("L", (period * 2, track.height))
    repeated.paste(track, (0, 0))
    repeated.paste(track, (period, 0))
    return repeated, period


def edge_fade():
    width, height = WIDTH * SCALE, HEIGHT * SCALE
    fade = 36 * SCALE
    row = Image.new("L", (width, 1))
    row.putdata(
        [round(255 * min(1, x / fade, (width - 1 - x) / fade)) for x in range(width)]
    )
    return row.resize((width, height))


def palette(theme):
    # One shared color ramp prevents dithering noise and palette flicker.
    colors = []
    for value in range(256):
        strength = min(value, COLORS - 1) / (COLORS - 1)
        colors.extend(
            round(bg + (fg - bg) * strength)
            for bg, fg in zip(theme["paper"], theme["ink"])
        )
    return colors


def indexed(mask, colors):
    frame = mask.point(lambda value: round(value * (COLORS - 1) / 255))
    # L values are already palette indices; convert('P') must not quantize them.
    frame = Image.frombytes("P", frame.size, frame.tobytes())
    frame.putpalette(colors)
    return frame


def static_mask(masks):
    """Show all sponsors at once for readers who request reduced motion."""
    result = Image.new("L", (WIDTH * SCALE, HEIGHT * SCALE))
    margin = 20 * SCALE
    available = result.width - 2 * margin
    gap = 24 * SCALE
    factor = (available - gap * (len(masks) - 1)) / sum(mask.width for mask in masks)
    cursor = margin
    for mask in masks:
        resized = mask.resize(
            (round(mask.width * factor), round(mask.height * factor)),
            Image.Resampling.LANCZOS,
        )
        result.paste(resized, (cursor, (result.height - resized.height) // 2))
        cursor += resized.width + gap
    return result


def add_rules(mask):
    draw = ImageDraw.Draw(mask)
    for y in (0, mask.height - SCALE):
        draw.rectangle((0, y, mask.width, y + SCALE - 1), fill=32)
    return mask


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    masks = logo_masks()
    track, period = make_track(masks)
    fade = edge_fade()
    still = add_rules(static_mask(masks))
    for name, theme in THEMES.items():
        colors = palette(theme)
        indexed(still, colors).save(output / f"ticker-{name}-static.png", optimize=True)
        frames = []
        for offset in range(0, period, STEP):
            viewport = track.crop((offset, 0, offset + WIDTH * SCALE, HEIGHT * SCALE))
            viewport = add_rules(ImageChops.multiply(viewport, fade))
            viewport = viewport.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            frames.append(indexed(viewport, colors))
        destination = output / f"ticker-{name}.gif"
        frames[0].save(
            destination,
            save_all=True,
            append_images=frames[1:],
            duration=FRAME_MS,
            loop=0,
            disposal=1,
            optimize=False,
            palette=bytes(colors[: COLORS * 3]),
        )
        print(
            f"{destination}: {len(frames)} frames, "
            f"{len(frames) * FRAME_MS / 1000:.2f}s, "
            f"{destination.stat().st_size:,} bytes",
            flush=True,
        )
        # Check actual encoded metadata, including all frame intervals.
        with Image.open(destination) as encoded:
            assert encoded.size == (WIDTH, HEIGHT)
            assert encoded.n_frames == len(frames)
            assert encoded.info.get("loop") == 0
            for frame in range(encoded.n_frames):
                encoded.seek(frame)
                assert encoded.info["duration"] == FRAME_MS
        del frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    build(parser.parse_args().output)
