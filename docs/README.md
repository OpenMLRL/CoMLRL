# CoMLRL Docs (Hugo Book)

This folder contains the Hugo site for CoMLRL’s documentation using the Hugo Book theme.

## Local preview

1. Install Hugo Extended: https://gohugo.io/installation/
2. From the repo root, run:

```bash
hugo server -s docs -D
```

Hugo will fetch the Book theme via Hugo Modules on first run (requires internet access). Open the local URL printed by Hugo (usually http://localhost:1313).

## Research publications

Research publications are maintained in the OpenMLRL website repository at `content/research/` for `https://openmlrl.github.io/research/`. This documentation site shows only Documentation and Examples in its area navigation. The former `content/docs/research/_index.md` is retained as a draft for reference and excluded from search.

## Appearance

The header groups three theme buttons: sun for the light palette, moon for the blue-tinted dark palette, and eye for the warm ivory reading palette. Light mode uses the original purple logo. Dark mode uses gold, coral, rose, and violet accents; reading mode uses ochre, copper, rose-brown, and muted olive accents. The first visit follows the system preference; an explicit choice is saved locally. `assets/theme.js` applies it before the first paint, and `assets/_theme.scss` defines the shared color roles and logo backgrounds. The dark and reading header logos use the original `static/img/comlrl-logo.png` alpha as a CSS mask, retaining the exact lettering and emblem. Both themed wordmarks use vertical gradients across the letters' actual height so the color change remains visible at header size.

The `static/img/comlrl-logo-dark-colors.svg` and `static/img/comlrl-logo-reading-colors.svg` color maps preserve the daylight emblem's four alternating gradient groups: two outer groups and two inner groups, each repeated on three petals. Every other petal in a ring shares the same palette, normalized over that petal's local vertical height; the central C stays solid. The group assignments come from the original PNG's colors, which follow local vertical gradients rather than gradients rotated around the flower. Both maps share the original color regions and keep neighboring petals separate, while the PNG supplies the outlines.

Keep syntax highlighting class-based (`markup.highlight.noClasses = false`) so `assets/_syntax.scss` can style both themes. `assets/_diagram-theme.scss` adjusts SVG paint for dark backgrounds without changing mathematical glyphs, geometry, or animation stages. Printing uses the light palette.

The GitHub link includes the repository's star count. `data/github.json` holds the verified fallback snapshot; `assets/github-stars.js` refreshes it from GitHub's public API and caches successful responses locally for one hour. API errors retain the last available count.

## Overview demo

The homepage displays `static/img/demo-hd.webp`, a 2612 × 1460 derivative of the original Manim export. The source `static/img/demo.gif` remains unchanged: 1306 × 730, 404 frames at 70 ms each, looping every 28.28 seconds. Each composited source frame is resized with Lanczos and receives an unsharp mask (radius 1, percent 140, threshold 2), then is encoded as lossless animated WebP. This improves edge definition without OCR, generated symbols, or re-typesetting; it cannot recover detail absent from the original export. The source frame order and timing are preserved, including fades. WebP may merge identical consecutive frames while retaining their complete duration.

Regenerate the derivative from the repository root using the Pillow version specified in the script:

```bash
python3 docs/scripts/prepare_demo_animation.py
```

The generator verifies every decoded output interval against its processed source frame, the total duration, and the original file's checksum. Hugo serves the checked-in derivative; Python is not required to build the site.

The `.docs-demo` frame uses the theme's paper surface, a fine outer border, rounded corners, and vertical spacing without reducing the displayed image width. A modest CSS saturation reduction softens colored titles and arrows. Light and reading modes blend the baked-in white background into the surface. Dark mode reverses luminance with a hue correction, then blends the black background into the surface. Theme changes only affect CSS; they never replace the image or restart its timeline. Avoid stacking another sharpening filter over the processed asset. Printing uses the light artwork without CSS filters.

`assets/visual-viewer.js` adds an expand icon to the demo and rollout diagrams. The raster animation retains its original declared zoom size of 1306 × 730 CSS pixels, backed by twice as many pixels in each direction for Retina displays. SVG diagrams remain resolution independent. The viewer keeps each original element in place, so opening, zooming, and closing do not replace it or restart playback. Escape or the close icon restores keyboard focus and scroll position.

## Rollout diagrams

The Multi-Turn Interaction animations reuse the original SVG artwork in `static/img/`. To regenerate their staged copies after updating an original, run:

```bash
python3 docs/scripts/prepare_rollout_svgs.py
```

The script only inserts animation attributes and verifies that removing them reproduces each source file byte for byte. If the diagram layout changes, update its stage boundaries in the script. The `rollout-diagram` shortcode namespaces SVG glyph IDs; `assets/rollout-diagrams.js` controls playback. Without JavaScript, or with reduced motion enabled, the complete diagrams remain visible.

## Deploy (GitHub Pages – native actions)

Publishing is currently manual. Run `.github/workflows/docs.yml` with the workflow-dispatch input `publish=true`; a push to `main` does not deploy this site. The workflow:

- Configures Pages and computes the correct base URL for a project site.
- Builds the Hugo site from `docs/`.
- Uploads the static site as a Pages artifact and deploys it.

One-time repo setting: in GitHub → Settings → Pages → Build and deployment, set Source to “GitHub Actions”.

> Note: GitHub Pages sites are publicly accessible even if the repository is private. For private/internal docs, preview locally or host behind authentication (e.g., GitHub Enterprise Pages with SSO or another internal host).
