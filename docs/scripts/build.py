#!/usr/bin/env python3
"""Build documentation; optionally generate translations using local Ollama."""

import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from check_html import check_site
from check_translations import check_site as check_translated_site
from translations import FRONT, LANGUAGES, heading_lines, load_translation


DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parent


class Headings(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if re.fullmatch(r"h[1-6]", tag) and attrs.get("id"):
            self.items.append((int(tag[1]), attrs["id"]))


def hugo(destination, base_url, overlay=None):
    command = [
        "hugo",
        "--source",
        str(DOCS),
        "--destination",
        str(destination),
        "--baseURL",
        base_url,
        "--environment",
        "production",
        "--minify",
    ]
    if overlay:
        command.extend(["--config", f"{DOCS / 'hugo.toml'},{overlay}"])
    subprocess.run(
        command, check=True, env={**os.environ, "HUGO_RELATIVEURLS": "false"}
    )


def preserve_anchors(source, translated, html):
    parsed = Headings()
    parsed.feed(html.read_text(encoding="utf-8"))
    headings = heading_lines(source)
    translated_headings = heading_lines(translated)
    if [h[1] for h in headings] != [h[0] for h in parsed.items]:
        raise ValueError(f"Cannot match source headings to rendered anchors: {html}")
    lines = translated.splitlines(keepends=True)
    for (line, level, text), (_, anchor) in zip(translated_headings, parsed.items):
        existing = re.search(r"\{#([^}]+)\}\s*$", text)
        if existing:
            if existing[1] != anchor:
                raise ValueError(f"Changed explicit heading anchor: {html}#{anchor}")
        else:
            lines[line] = lines[line].rstrip("\r\n") + " {#" + anchor + "}\n"
    return "".join(lines)


def preserve_api_identifiers(source, translated):
    """Keep API-card detection and IDs independent of translated display text."""
    hints = re.compile(
        r"(\{\{%\s+hint\s+info)(\s+%\}\})(.*?)(\{\{%\s+/hint\s+%\}\})", re.S
    )
    originals = iter(hints.finditer(source))

    def mark(match):
        original = next(originals, None)
        if original is None:
            raise ValueError("Translation changed info hint structure")
        title = original[3].strip().split("\n")[0]
        return (
            match[1]
            + " "
            + json.dumps(title, ensure_ascii=False)
            + match[2]
            + match[3]
            + match[4]
        )

    result = hints.sub(mark, translated)
    if next(originals, None) is not None:
        raise ValueError("Translation removed an info hint")
    return result


def build_site(
    destination,
    *,
    base_url,
    bilingual=False,
    languages=None,
    cache_only=False,
    endpoint="http://127.0.0.1:11434",
    model="translategemma:12b",
):
    selected = list(
        dict.fromkeys(
            languages if languages is not None else (["zh-cn"] if bilingual else [])
        )
    )
    unknown = set(selected) - LANGUAGES.keys()
    if unknown:
        raise ValueError(f"Unsupported languages: {', '.join(sorted(unknown))}")
    selected = sorted(
        (lang for lang in selected if lang != "en"),
        key=lambda lang: LANGUAGES[lang]["weight"],
    )
    destination = Path(destination).resolve()
    # Never write generated pages inside version-controlled source directories.
    if destination == REPO or destination in REPO.parents or destination == DOCS:
        raise ValueError("Choose a dedicated build output directory")
    if (
        destination.is_relative_to(REPO)
        and not subprocess.run(
            ["git", "check-ignore", "-q", str(destination / "index.html")],
            cwd=REPO,
        ).returncode
        == 0
    ):
        raise ValueError("Build output inside this repository must be Git-ignored")
    # Build into a fresh staging directory, validate, then replace the output.
    with TemporaryDirectory(prefix="comlrl-build-") as temporary:
        staging = Path(temporary)
        english = staging / "english"
        hugo(english, base_url)
        check_site(english)
        output = english
        if selected:
            configs = [
                '[languages.en]\nlanguageName = "English"\nlanguageCode = "en-US"\nweight = 1\n'
                f'contentDir = {json.dumps(str(DOCS / "content"))}\n'
            ]
            count = 0
            for language in selected:
                translated_content = staging / f"content-{language}"
                for path in sorted((DOCS / "content").rglob("*.md")):
                    relative = path.relative_to(DOCS / "content")
                    source = path.read_text(encoding="utf-8")
                    front = FRONT.match(source)
                    if front and re.search(r"(?m)^draft:\s*true\s*$", front[1]):
                        continue
                    rendered = english / relative.parent
                    if relative.stem != "_index":
                        rendered /= relative.stem
                    rendered /= "index.html"
                    if not rendered.exists():
                        raise ValueError(f"Missing English page: {relative}")
                    translated = load_translation(
                        source,
                        relative,
                        cache_root=REPO / ".cache/docs-translations" / language,
                        cache_only=cache_only,
                        endpoint=endpoint,
                        model=model,
                        language=language,
                    )
                    translated = preserve_anchors(source, translated, rendered)
                    translated = preserve_api_identifiers(source, translated)
                    # Mark generated pages without storing translated source in Git.
                    match = FRONT.match(translated)
                    frontmatter = match[1]
                    # Absolute aliases otherwise overwrite the English redirects.
                    frontmatter = re.sub(
                        r"(?m)^(\s+-\s+)([\"']?)(/docs/[^\n]+)$",
                        lambda m: m[1] + m[2] + f"/{language}" + m[3],
                        frontmatter,
                    )
                    translated = (
                        "---\n"
                        + frontmatter
                        + '\ntranslationSource: "en"\n'
                        + f"translationModel: {json.dumps(model)}\n---\n"
                        + translated[match.end() :]
                    )
                    target = translated_content / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(translated, encoding="utf-8")
                    count += 1
                settings = LANGUAGES[language]
                configs.append(
                    f"[languages.{language}]\n"
                    f'languageName = {json.dumps(settings["name"], ensure_ascii=False)}\n'
                    f'languageCode = {json.dumps(settings["languageCode"])}\n'
                    f'weight = {settings["weight"]}\n'
                    f'hasCJKLanguage = {str(settings["cjk"]).lower()}\n'
                    f"contentDir = {json.dumps(str(translated_content))}\n"
                )
            overlay = staging / "languages.toml"
            overlay.write_text("\n".join(configs), encoding="utf-8")
            output = staging / "multilingual"
            hugo(output, base_url, overlay)
            print(
                f"Included {count} translations in {len(selected)} languages; English sources unchanged.",
                flush=True,
            )
        count = check_site(output)
        if selected:
            integrity = check_translated_site(
                output, base_url=base_url, selected=["en", *selected]
            )
            if integrity.errors:
                raise ValueError(
                    "Translation integrity checks failed:\n"
                    + "\n".join(integrity.errors)
                )
            print(f"Validated {integrity.summary()}.", flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        # Avoid deleting unrelated files: caller must supply an empty destination.
        if any(destination.iterdir()):
            raise ValueError(f"Build destination must be empty: {destination}")
        shutil.copytree(output, destination, dirs_exist_ok=True)
        print(f"Validated {count} HTML files: {destination}", flush=True)
        return count


def add_translation_arguments(parser):
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--bilingual", action="store_true", help="Build English and Simplified Chinese"
    )
    selection.add_argument(
        "--multilingual", action="store_true", help="Build all supported languages"
    )
    selection.add_argument(
        "--languages",
        nargs="+",
        choices=list(LANGUAGES),
        help="Translation languages to build (English is always included)",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Require current cached translations; do not call Ollama",
    )
    parser.add_argument(
        "--model", default="translategemma:12b", help="Local Ollama translation model"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Local Ollama service URL",
    )


def selected_languages(args):
    if args.multilingual:
        return list(LANGUAGES)
    if args.bilingual:
        return ["zh-cn"]
    return args.languages or []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Empty directory for generated deployment artifacts",
    )
    parser.add_argument(
        "--base-url", default="/", help="Final site URL, including any project path"
    )
    add_translation_arguments(parser)
    args = parser.parse_args()
    build_site(
        args.destination,
        base_url=args.base_url,
        languages=selected_languages(args),
        cache_only=args.cache_only,
        endpoint=args.ollama_url,
        model=args.model,
    )


if __name__ == "__main__":
    main()
