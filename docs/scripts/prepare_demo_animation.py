#!/usr/bin/env python3
"""Create the overview's sharper 2x animation without redrawing its content.

Requires Pillow==12.3.0 with WebP support. Run from any directory:
    python docs/scripts/prepare_demo_animation.py

The original GIF remains the source of truth. Each composited frame is resized
with Lanczos and gently sharpened; no text, math, colors, or timings are inferred.
This improves raster presentation, but cannot recover missing source detail.
Lossless WebP keeps the resulting antialiased pixels without a GIF palette.

Precomputing append_images for Pillow's public writer would retain several
gigabytes of processed frames. Its version-pinned streaming encoder keeps only
the current frame and compressed animation data in memory. The finished file is
decoded and checked against all original frame intervals before installation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from PIL import Image, ImageFilter, __version__ as PILLOW_VERSION, _webp, features


DOCS = Path(__file__).resolve().parents[1]
SOURCE = DOCS / "static" / "img" / "demo.gif"
DEFAULT_OUTPUT = DOCS / "static" / "img" / "demo-hd.webp"
SOURCE_SHA256 = "1b53f9188cba46b5908ca2aed36bbcbeb2b34091e62e6cd669ae5a36a7065de0"
PILLOW_REQUIRED = "12.3.0"
SCALE = 2
SHARPEN = {"radius": 1, "percent": 140, "threshold": 2}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare_frame(image):
    """Process the decoded pixels identically for every frame, including fades."""
    return (
        image.convert("RGB")
        .resize((image.width * SCALE, image.height * SCALE), Image.Resampling.LANCZOS)
        .filter(ImageFilter.UnsharpMask(**SHARPEN))
    )


def validate(path, expected, dimensions, duration, loop):
    source_index = 0
    output_frames = 0
    output_time = 0
    with Image.open(path) as animation:
        assert animation.size == dimensions, "Output dimensions changed"
        assert animation.info.get("loop") == loop, "Loop behavior changed"
        for output_index in range(animation.n_frames):
            animation.seek(output_index)
            animation.load()  # WebP timestamp/duration are available after decode.
            start = animation.info["timestamp"]
            end = start + animation.info["duration"]
            assert start == output_time, "Gap or overlap in output timeline"
            frame_hash = digest(animation.convert("RGB").tobytes())
            while source_index < len(expected) and expected[source_index][0] < end:
                expected_start, expected_end, expected_hash = expected[source_index]
                assert (
                    start <= expected_start < expected_end <= end
                ), f"Timing changed at source frame {source_index}"
                assert (
                    frame_hash == expected_hash
                ), f"Pixels changed at source frame {source_index}"
                source_index += 1
            output_time = end
            output_frames += 1
        assert output_time == duration, "Total duration changed"
        assert source_index == len(expected), "Some source frames were omitted"
    return output_frames


def build(output, report_path=None):
    if PILLOW_VERSION != PILLOW_REQUIRED or not features.check("webp"):
        raise RuntimeError(
            f"Install Pillow=={PILLOW_REQUIRED} with WebP support; "
            f"found Pillow {PILLOW_VERSION}. The streaming encoder uses its "
            "version-specific API."
        )
    original_hash = digest(SOURCE.read_bytes())
    assert (
        original_hash == SOURCE_SHA256
    ), "Source GIF differs from the reviewed original"
    assert output.resolve() != SOURCE.resolve(), "Never overwrite the source GIF"
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = []
    durations = []
    timestamp = 0
    with Image.open(SOURCE) as source:
        source_dimensions = source.size
        dimensions = (source.width * SCALE, source.height * SCALE)
        loop = source.info.get("loop", 0)
        # Arguments match Pillow 12.3.0's WebPImagePlugin._save_all. minimize_size
        # optimizes deltas and may merge identical frames while retaining time.
        encoder = _webp.WebPAnimEncoder(
            dimensions, 0xFFFFFFFF, loop, True, 9, 17, False, False
        )
        for index in range(source.n_frames):
            source.seek(index)
            duration = source.info["duration"]
            assert duration > 0, "Every source frame needs an explicit duration"
            frame = prepare_frame(source)
            expected.append((timestamp, timestamp + duration, digest(frame.tobytes())))
            durations.append(duration)
            # lossless=True; quality=100 controls compression effort rather than
            # lossiness. RGBX is the encoder's opaque, four-channel input format.
            encoder.add(frame.convert("RGBX").getim(), timestamp, True, 100, 100, 4)
            timestamp += duration
            if (index + 1) % 50 == 0 or index + 1 == source.n_frames:
                print(f"Encoded {index + 1}/{source.n_frames} frames", flush=True)
        encoder.add(None, timestamp, True, 100, 100, 0)
        encoded = encoder.assemble("", "", "")
        if encoded is None:
            raise RuntimeError("WebP encoder returned no animation")

    # Only replace the destination after checking every decoded pixel and timing.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".demo-hd-", suffix=".webp", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        output_frames = validate(temporary, expected, dimensions, timestamp, loop)
        assert digest(SOURCE.read_bytes()) == original_hash, "Source GIF was modified"
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    report = {
        "source": str(SOURCE),
        "source_sha256": original_hash,
        "source_unchanged": True,
        "source_dimensions": list(source_dimensions),
        "source_frames": len(expected),
        "source_durations_ms": sorted(set(durations)),
        "output": str(output),
        "output_sha256": digest(encoded),
        "output_dimensions": list(dimensions),
        "output_frames": output_frames,
        "output_bytes": len(encoded),
        "duration_ms": timestamp,
        "loop": loop,
        "scale": SCALE,
        "resampling": "Lanczos",
        "unsharp_mask": SHARPEN,
        "lossless": True,
        "all_source_intervals_pixel_exact": True,
        "pillow_version": PILLOW_VERSION,
        "libwebp_version": features.version("webp"),
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if report_path:
        report_path.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    build(args.output, args.report)
