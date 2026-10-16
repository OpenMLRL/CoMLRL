# CoMLRL Docs (Hugo Book)

This folder contains the Hugo site for CoMLRL’s documentation using the Hugo Book theme.

## Local preview

1. Install Hugo Extended: https://gohugo.io/installation/
2. From the repo root, run:

```bash
python3 docs/scripts/preview.py
```

Hugo will fetch the Book theme via Hugo Modules on first run (requires internet access). The preview command performs a fresh production build, checks the generated HTML for corrupted text and unexpanded shortcodes, and serves it at http://localhost:1313. Draft pages remain excluded.

After editing, stop the preview with Ctrl+C, rerun the command, and refresh the browser. This uses complete static builds because incremental previews have produced malformed shortcode output during documentation edits. The server disables browser caching.

Existing build output can also be checked before publishing:

```bash
python3 docs/scripts/check_html.py docs/public
```

## Multilingual preview

English remains the only maintained content in `content/`. The multilingual preview includes English, Simplified Chinese, Japanese, Korean, Spanish, French, and German:

```bash
python3 docs/scripts/preview.py --multilingual
```

The header's language menu switches to the same page and retains the current section anchor. Generated translations are labelled and link to the English original. Code blocks, code comments, model identifiers, formulas, and link destinations are preserved. The language menu lists only languages included in the current build.

Translations are cached independently in the Git-ignored `.cache/docs-translations/<language>/` directories at the repository root. A source hash determines whether each page needs translation. Unchanged pages reuse their cache; changed or missing pages are translated by a local Ollama service at `http://127.0.0.1:11434`. The default model is `qwen2.5-coder:3b`; `--model` selects another installed model. Local translation requires Ollama to be running and the chosen model to be installed. It does not require Crowdin or a paid translation API, but uses the local device's compute resources. Translation quality still benefits from review.

The first build generates any missing translations and can take longer; subsequent builds reuse them. Cached pages are validated before reuse; invalid entries are regenerated when Ollama is available. `--cache-only` builds without contacting Ollama and fails if any selected language has a missing, stale, or invalid translation. Caches are local: a fresh checkout needs to generate them, or restore them from a build cache. Generated translated Markdown exists only in a temporary build directory; generated HTML is never added to Git by these scripts.

To build a subset, use `--languages zh-cn ja es` (English is always included). The existing `--bilingual` option still builds only English and Simplified Chinese. Language codes, names, and terminology are configured in `data/languages.json`.

To produce a deployment artifact in an **empty** output directory:

```bash
python3 docs/scripts/build.py --multilingual \
  --destination /tmp/comlrl-multilingual-site \
  --base-url https://openmlrl.github.io/CoMLRL/
```

The build checks protected Markdown, preserves English heading and API-card anchors, and validates rendered code, formulas, links, assets, language switches, and redirects before writing the output. A failed translation or validation stops the build. The output is a deployment artifact and should not be copied into the OpenMLRL repository and committed. The production website workflow restores translation caches and runs its own Ollama service on CPU for missing or changed pages. It does not depend on the Ollama service or caches on a developer's computer. The first build can take substantially longer than later builds because it generates translations for all six target languages.

The translation guards and cache behavior can be checked without a model or network access:

```bash
python3 -B -m unittest discover -s docs/scripts -p 'test_translation*.py'
```

## Research publications

Research publications are maintained in the OpenMLRL website repository at `content/research/` for `https://openmlrl.github.io/research/`. This documentation site shows only Documentation and Examples in its area navigation. The former `content/docs/research/_index.md` is retained as a draft for reference and excluded from search.

## Appearance

The header groups three theme buttons: sun for the light palette, moon for the blue-tinted dark palette, and eye for the warm ivory reading palette. Light mode uses the original purple logo. Dark mode uses gold, coral, rose, and violet accents; reading mode uses ochre, copper, rose-brown, and muted olive accents. The first visit follows the system preference; an explicit choice is saved locally. `assets/theme.js` applies it before the first paint, and `assets/_theme.scss` defines the shared color roles and logo backgrounds. The dark and reading header logos use the original `static/img/comlrl-logo.png` alpha as a CSS mask, retaining the exact lettering and emblem. Both themed wordmarks use vertical gradients across the letters' actual height so the color change remains visible at header size.

The `static/img/comlrl-logo-dark-colors.svg` and `static/img/comlrl-logo-reading-colors.svg` color maps preserve the daylight emblem's four alternating gradient groups: two outer groups and two inner groups, each repeated on three petals. Every other petal in a ring shares the same palette, normalized over that petal's local vertical height; the central C stays solid. The group assignments come from the original PNG's colors, which follow local vertical gradients rather than gradients rotated around the flower. Both maps share the original color regions and keep neighboring petals separate, while the PNG supplies the outlines.

Keep syntax highlighting class-based (`markup.highlight.noClasses = false`) so `assets/_syntax.scss` can style both themes. `assets/_diagram-theme.scss` adjusts SVG paint for dark backgrounds without changing mathematical glyphs, geometry, or animation stages. Printing uses the light palette.

The header groups the OpenMLRL link, GitHub link, and theme controls with compact spacing. The GitHub link does not display or fetch a star count.

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

## Deploy (GitHub Pages)

The `Pages` workflow in `OpenMLRL/OpenMLRL.github.io` is the single publisher for the website and CoMLRL documentation. It builds the documentation in English, Simplified Chinese, Japanese, Korean, Spanish, French, and German, then bundles the output under `/CoMLRL/` in the website's Pages artifact. Generated translated Markdown, translation caches, and HTML are not committed to either repository by this pipeline.

On documentation changes pushed to CoMLRL's `main` branch, `.github/workflows/docs-build.yml` builds and validates the English documentation and checks translation protection. It then sends a `comlrl-docs-updated` event, including the source commit, to the website repository. Pull requests run validation only. This replaces the previous workflow that committed English HTML to `OpenMLRL.github.io/static/CoMLRL`.

The notification uses the existing `OPENMLRL_TOKEN` secret in CoMLRL. It must allow repository dispatches to `OpenMLRL/OpenMLRL.github.io` (Contents write permission for a fine-grained token). The website's Pages source must be set to **GitHub Actions**. The website workflow must include the `comlrl-docs-updated` trigger before CoMLRL can request deployments through it.

For a manual deployment, users can run `.github/workflows/docs.yml` with `publish=true`. This validates the documentation and requests a deployment from the same website publisher. With `publish=false`, it only runs validation. This workflow no longer publishes a separate English-only Pages site.

The website workflow restores a separate cache for each translation language and uses Ollama on its GitHub-hosted CPU runner to translate missing or changed pages. This requires no translation API key. The first build is slower; subsequent builds reuse unchanged translations. English remains the only maintained content. After all six translation builds succeed, the website is built with the validated caches and uploaded directly as a Pages artifact. A failed translation or validation prevents publication, leaving the previous site available.

> Note: GitHub Pages sites are publicly accessible even if the repository is private. For private/internal docs, preview locally or host behind authentication (e.g., GitHub Enterprise Pages with SSO or another internal host).
