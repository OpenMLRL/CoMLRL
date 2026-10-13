#!/usr/bin/env python3
"""Add animation stages without reserializing or redrawing the original SVGs.

Run from any directory with Python 3. The generated assets retain every original
byte, including the glyph paths, use positions, definitions, and draw order; only
two data attributes are inserted on each top-level drawing element.
"""

from bisect import bisect_right
from collections import Counter
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from xml.parsers import expat


DOCS = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
ANNOTATIONS = re.compile(
    rb' data-rollout-stage="\d+" data-rollout-kind="(?:node|edge)"'
)
# Boundaries fall between complete rows/formulae, never through a formula.
# The chain has five visible nodes; its trailing ellipsis shares the final stage.
LAYOUTS = {
    "joint-tree": (1, (26, 60, 94)),
    "align-tree": (1, (22, 60, 92)),
    "cross-tree": (1, (30, 74, 118)),
    "ac": (0, (42, 95, 144, 197)),
}


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def path_points(path):
    """Read the source's M/L/C coordinates and apply its existing matrix."""
    data = path.attrib["d"]
    commands = set(re.findall(r"[A-Za-z]", data))
    assert commands <= {"M", "L", "C", "Z"}, commands
    numbers = [float(value) for value in NUMBER.findall(data)]
    matrix = path.get("transform", "matrix(1, 0, 0, 1, 0, 0)")
    assert matrix.startswith("matrix(")
    a, b, c, d, tx, ty = [float(value) for value in NUMBER.findall(matrix)]
    return [
        (a * x + c * y + tx, b * x + d * y + ty)
        for x, y in zip(numbers[::2], numbers[1::2])
    ]


def annotation(element, axis, boundaries):
    uses = [child for child in element.iter() if tag(child) == "use"]
    paths = [child for child in element.iter() if tag(child) == "path"]
    # Closed rectangles/circles are nodes. Only straight connecting paths are
    # drawn on; arrowheads and pre-existing dashed paths fade in unmodified.
    connecting = bool(paths) and all(
        set(re.findall(r"[A-Za-z]", path.attrib["d"])) <= {"M", "L"}
        and "stroke-dasharray" not in path.attrib
        for path in paths
    )
    if uses:
        coordinate = sum(float(use.attrib["xy"[axis]]) for use in uses) / len(uses)
    elif connecting:
        coordinate = max(path_points(path)[-1][axis] for path in paths)
    else:
        points = [point[axis] for path in paths for point in path_points(path)]
        assert points, "Every drawing element must have a known position"
        coordinate = (min(points) + max(points)) / 2
    stage = bisect_right(boundaries, coordinate)
    return stage, "edge" if connecting else "node"


def top_level_offsets(source):
    """Locate start tags in the original bytes, without XML normalization."""
    parser = expat.ParserCreate()
    depth = 0
    offsets = []

    def start(name, attributes):
        nonlocal depth
        if depth == 1:
            offsets.append((parser.CurrentByteIndex, name))
        depth += 1

    def end(name):
        nonlocal depth
        depth -= 1

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.Parse(source, True)
    return offsets


def prepare(name, axis, boundaries):
    source = (DOCS / "static" / "img" / f"{name}.svg").read_bytes()
    root = ET.fromstring(source)
    offsets = top_level_offsets(source)
    assert len(root) == len(offsets)
    insertions = []
    counts = Counter()
    for element, (offset, name_in_source) in zip(root, offsets):
        if tag(element) == "defs":
            continue
        assert tag(element) in {"g", "path"}
        stage, kind = annotation(element, axis, boundaries)
        attributes = (
            f' data-rollout-stage="{stage}" data-rollout-kind="{kind}"'.encode()
        )
        insertions.append((offset + 1 + len(name_in_source), attributes))
        counts[stage, kind] += 1
    output = source
    for offset, attributes in reversed(insertions):
        output = output[:offset] + attributes + output[offset:]
    # This stronger-than-geometry check also protects definitions, colors,
    # formula glyph references/positions, precision, and painter's order.
    assert ANNOTATIONS.sub(b"", output) == source
    assert {stage for stage, kind in counts} == set(range(len(boundaries) + 1))
    destination = DOCS / "assets" / "rollouts" / f"{name}.svg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output)
    print(
        f"{name}: {len(boundaries) + 1} stages; {dict(sorted(counts.items()))}; original bytes verified"
    )


if __name__ == "__main__":
    for name, (axis, boundaries) in LAYOUTS.items():
        prepare(name, axis, boundaries)
