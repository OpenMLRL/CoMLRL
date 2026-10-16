#!/usr/bin/env python3
"""Validate translated Hugo artifacts without fetching external resources.

English lives at the output root; other languages use their catalog code as a
directory. --base-url may include a project prefix such as /CoMLRL/. The default
language catalog is docs/data/languages.json. Without it, language homepages are
inferred from their HTML lang attributes. --languages limits validation to an
explicit subset, useful for a preview containing fewer languages than production.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit


VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I)


@dataclass
class Reference:
    url: str
    location: str
    article: bool = False
    language_link: bool = False
    asset: bool = False


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.lang = ""
        self.canonical = ""
        self.og_url = ""
        self.refresh = ""
        self.article = False
        self.ids = set()
        self.article_ids = Counter()
        self.headings = []
        self.blocks = []
        self.inline_code = []
        self.math = []
        self.references = []
        self.switches = []
        self.stack = []
        self.captures = []

    def within(self, key):
        return any(frame.get(key) for frame in self.stack)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        frame = {
            "tag": tag,
            "article": tag == "article" and "book-article" in classes,
            "switcher": "docs-language-switcher" in classes,
            "svg": tag == "svg",
        }
        article = frame["article"] or self.within("article")
        in_svg = frame["svg"] or self.within("svg")
        language_link = "data-language-link" in attrs
        if tag == "html":
            self.lang = (attrs.get("lang") or "").lower()
        if frame["article"]:
            self.article = True
        if attrs.get("id"):
            self.ids.add(attrs["id"])
            if article and not in_svg:
                self.article_ids[attrs["id"]] += 1
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        if article and not in_svg and re.fullmatch(r"h[1-6]", tag):
            self.headings.append((tag, attrs.get("id")))
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href") or ""
        if tag == "meta":
            if attrs.get("property") == "og:url":
                self.og_url = attrs.get("content") or ""
            if (attrs.get("http-equiv") or "").lower() == "refresh":
                match = re.search(
                    r"\burl\s*=\s*(['\"]?)(.*?)\1\s*$", attrs.get("content") or "", re.I
                )
                if match:
                    self.refresh = match[2]
                    self.references.append(Reference(self.refresh, "meta refresh"))
        if tag == "a" and self.within("switcher"):
            self.switches.append(
                (
                    (attrs.get("hreflang") or attrs.get("lang") or "").lower(),
                    attrs.get("href") or "",
                )
            )
        for attribute in ("href", "src", "poster", "data", "xlink:href"):
            value = attrs.get(attribute)
            if value and (attribute != "data" or tag == "object"):
                self.references.append(
                    Reference(
                        value,
                        f"{tag}[{attribute}]",
                        article,
                        language_link,
                        tag not in {"a", "link"}
                        or (tag == "link" and attrs.get("rel") != "canonical"),
                    )
                )
        if attrs.get("srcset") and not attrs["srcset"].lstrip().startswith("data:"):
            for candidate in attrs["srcset"].split(","):
                if candidate.strip():
                    self.references.append(
                        Reference(candidate.split()[0], f"{tag}[srcset]", asset=True)
                    )
        for match in CSS_URL.finditer(attrs.get("style") or ""):
            self.references.append(Reference(match[2], f"{tag}[style]", asset=True))
        if article and tag == "pre":
            frame["pre"] = True
            self.captures.append([len(self.stack), "blocks", []])
        elif article and tag == "code" and not self.within("pre"):
            self.captures.append([len(self.stack), "inline_code", []])
        if article and "book-katex" in classes:
            self.captures.append([len(self.stack), "math", []])
        if tag not in VOID:
            self.stack.append(frame)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                while self.captures and self.captures[-1][0] >= index:
                    _, kind, text = self.captures.pop()
                    getattr(self, kind).append("".join(text))
                del self.stack[index:]
                return

    def handle_data(self, data):
        for _, _, text in self.captures:
            text.append(data)


def read_page(path):
    page = Page(path)
    source = path.read_text(encoding="utf-8")
    page.script_redirect = "location.replace(" in source
    page.feed(source)
    page.close()
    return page


def language_catalog(directory, pages, selected=None):
    catalog_path = Path(__file__).resolve().parents[1] / "data/languages.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(catalog, dict) or "en" not in catalog:
            raise ValueError(
                f"Expected a language-code dictionary including en: {catalog_path}"
            )
        languages = {
            code: (values.get("languageCode") or code).lower()
            for code, values in catalog.items()
        }
    else:
        languages = {"en": pages["index.html"].lang or "en"}
        for relative, page in pages.items():
            parts = Path(relative).parts
            if len(parts) == 2 and parts[1] == "index.html" and page.article:
                code = parts[0].lower()
                if (
                    page.lang
                    and page.lang != languages["en"]
                    and (page.lang == code or page.lang.startswith(code + "-"))
                ):
                    languages[code] = page.lang
    if selected:
        unknown = set(selected) - languages.keys()
        if unknown:
            raise ValueError(f"Unknown languages: {', '.join(sorted(unknown))}")
        languages = {
            code: language
            for code, language in languages.items()
            if code in selected or code == "en"
        }
    return languages


@dataclass
class Result:
    languages: int = 0
    html: int = 0
    translations: int = 0
    code_blocks: int = 0
    math: int = 0
    references: int = 0
    redirects: int = 0
    errors: list = field(default_factory=list)

    def summary(self):
        return (
            f"{self.languages} languages, {self.html} HTML files, {self.translations} translated pages, "
            f"{self.code_blocks} code blocks, {self.math} formulas, "
            f"{self.references} internal references, {self.redirects} redirects"
        )


def check_site(directory, *, base_url=None, selected=None):
    directory = Path(directory).resolve()
    pages = {}
    for path in sorted(directory.rglob("*.html")):
        try:
            pages[path.relative_to(directory).as_posix()] = read_page(path)
        except UnicodeDecodeError as error:
            raise ValueError(f"Invalid UTF-8 in {path}: {error}") from error
    if "index.html" not in pages:
        raise ValueError(f"Missing site homepage: {directory / 'index.html'}")
    known_languages = language_catalog(directory, pages)
    languages = language_catalog(directory, pages, selected)
    result = Result(languages=len(languages), html=len(pages))
    base_url = (
        base_url or pages["index.html"].og_url or pages["index.html"].canonical or "/"
    )
    if not urlsplit(base_url).netloc:
        homepage_url = pages["index.html"].og_url or pages["index.html"].canonical
        origin = (
            homepage_url if urlsplit(homepage_url).netloc else "https://docs.invalid/"
        )
        base_url = urljoin(origin, base_url)
    base_url = base_url.rstrip("/") + "/"
    parsed_base = urlsplit(base_url)
    base_path = unquote(parsed_base.path)

    def issue(relative, message):
        result.errors.append(f"{relative}: {message}")

    def page_url(relative):
        suffix = (
            relative.removesuffix("index.html")
            if relative.endswith("index.html")
            else relative
        )
        return urljoin(base_url, suffix)

    def resolve(relative, reference):
        url = urlsplit(urljoin(page_url(relative), reference))
        if url.scheme not in {"http", "https"} or url.netloc != parsed_base.netloc:
            return None
        path = unquote(url.path) or "/"
        if not path.startswith(base_path):
            # Links to the containing organization's homepage are outside this
            # site. A known local asset/page missing its project prefix is an error.
            candidate = directory / path.lstrip("/")
            if path != "/" and (
                candidate.is_file() or (candidate / "index.html").is_file()
            ):
                raise ValueError(
                    f"URL {reference!r} omits the site prefix {base_path!r}"
                )
            return None
        target = directory / path[len(base_path) :]
        if target.is_dir() or path.endswith("/"):
            target /= "index.html"
        elif not target.exists() and not target.suffix:
            target /= "index.html"
        if not target.resolve().is_relative_to(directory):
            raise ValueError(f"URL {reference!r} escapes the output directory")
        return target.relative_to(directory).as_posix(), unquote(url.fragment)

    def identity(relative):
        prefix, separator, rest = relative.partition("/")
        if separator and prefix in known_languages and prefix != "en":
            return prefix, rest
        return "en", relative

    def translated_path(relative, language):
        return relative if language == "en" else language + "/" + relative

    def resolve_checked(relative, reference):
        try:
            return resolve(relative, reference)
        except ValueError as error:
            issue(relative, str(error))
            return None

    def anchor_exists(relative, fragment, visited=None):
        page = pages.get(relative)
        if not page or fragment in page.ids:
            return bool(page)
        visited = set() if visited is None else visited
        if relative in visited or not page.refresh:
            return False
        visited.add(relative)
        # Legacy aliases route some bookmarks to different pages in JavaScript.
        # Their conditional routing is outside a static HTML validator's scope.
        if page.script_redirect:
            return True
        target = resolve_checked(relative, page.refresh)
        return bool(target and anchor_exists(target[0], target[1] or fragment, visited))

    def mismatch(original, actual):
        if isinstance(original, Counter):
            missing, extra = original - actual, actual - original
            return f"missing {list(missing.items())[:2]!r}; extra {list(extra.items())[:2]!r}"[
                :400
            ]
        for index, (old, new) in enumerate(zip(original, actual), 1):
            if old != new:
                if isinstance(old, str) and isinstance(new, str):
                    offset = next(
                        (i for i, (a, b) in enumerate(zip(old, new)) if a != b),
                        min(len(old), len(new)),
                    )
                    return f"entry {index}, character {offset + 1}: expected {old[offset:offset + 80]!r}, got {new[offset:offset + 80]!r}"
                return f"entry {index}: expected {old!r}, got {new!r}"
        return f"expected {len(original)} entries, got {len(actual)}"

    for code in languages:
        root = translated_path("index.html", code)
        if root not in pages or not pages[root].article:
            issue(root, "missing language homepage (or overwritten by an alias)")

    for relative, page in pages.items():
        language, logical = identity(relative)
        if language not in languages:
            continue
        expected = languages[language]
        if page.lang not in {language, expected}:
            issue(
                relative,
                f"html lang={page.lang!r}, expected {expected!r}; possible alias clobber",
            )
        if page.refresh:
            result.redirects += 1
            target = resolve_checked(relative, page.refresh)
            if target and identity(target[0])[0] != language:
                issue(relative, f"redirect crosses languages: {page.refresh!r}")
        if page.article:
            if language != "en" and logical not in pages:
                issue(relative, f"missing English source page {logical}")
            found_switches = set()
            for code, href in page.switches:
                code = next(
                    (key for key, value in languages.items() if code in {key, value}),
                    code,
                )
                if code not in languages:
                    if selected:
                        continue
                    issue(relative, f"language switch has unknown hreflang={code!r}")
                    continue
                found_switches.add(code)
                target = resolve_checked(relative, href)
                wanted = translated_path(logical, code)
                if target is None or target[0] != wanted:
                    issue(
                        relative,
                        f"{code} switch must target the same page {wanted}, got {href!r}",
                    )
            if len(languages) > 1:
                missing = languages.keys() - found_switches
                if missing:
                    issue(
                        relative,
                        f"missing language switch: {', '.join(sorted(missing))}",
                    )
        for ref in page.references:
            target = resolve_checked(relative, ref.url)
            if target is None:
                continue
            result.references += 1
            target_relative, fragment = target
            target_path = directory / target_relative
            if not target_path.is_file():
                issue(
                    relative, f"{ref.location} {ref.url!r} -> missing {target_relative}"
                )
                continue
            target_page = pages.get(target_relative)
            if fragment and not fragment.startswith(":~:text="):
                if target_page:
                    if not anchor_exists(target_relative, fragment):
                        issue(
                            relative,
                            f"{ref.location} {ref.url!r} -> missing anchor #{fragment} in {target_relative}",
                        )
                elif target_path.suffix.lower() == ".svg":
                    svg = read_page(target_path)
                    if fragment not in svg.ids:
                        issue(
                            relative,
                            f"{ref.location} {ref.url!r} -> missing SVG fragment #{fragment}",
                        )
            if (
                ref.article
                and not ref.language_link
                and target_page
                and language != "en"
            ):
                target_language, target_logical = identity(target_relative)
                local_version = translated_path(target_logical, language)
                if target_language != language and local_version in pages:
                    issue(
                        relative,
                        f"content link {ref.url!r} leaves {language}; corresponding page is {local_version}",
                    )

    english_pages = {
        relative: page
        for relative, page in pages.items()
        if identity(relative)[0] == "en" and not relative.startswith("en/")
    }
    for language in languages.keys() - {"en"}:
        if language + "/index.html" not in pages:
            continue
        for relative, english in english_pages.items():
            translated = language + "/" + relative
            page = pages.get(translated)
            if page is None:
                issue(translated, f"missing counterpart of {relative}")
                continue
            if page.article != english.article:
                issue(
                    translated,
                    f"article/alias differs from {relative}; possible alias clobber",
                )
                continue
            if english.article:
                result.translations += 1
                result.code_blocks += len(page.blocks)
                result.math += len(page.math)
                for name, original, actual in [
                    ("code blocks", english.blocks, page.blocks),
                    (
                        "inline code",
                        Counter(english.inline_code),
                        Counter(page.inline_code),
                    ),
                    ("KaTeX formulas", Counter(english.math), Counter(page.math)),
                    ("heading levels/anchors", english.headings, page.headings),
                    ("article anchors", english.article_ids, page.article_ids),
                ]:
                    if original != actual:
                        issue(
                            translated,
                            f"{name} differ from {relative}: {mismatch(original, actual)}",
                        )
            if english.refresh:
                source_target = resolve_checked(relative, english.refresh)
                actual_target = (
                    resolve_checked(translated, page.refresh) if page.refresh else None
                )
                if source_target:
                    expected = translated_path(identity(source_target[0])[1], language)
                    if actual_target is None or actual_target[0] != expected:
                        issue(
                            translated,
                            f"alias must preserve target {expected}; got {page.refresh!r}",
                        )
                elif page.refresh != english.refresh:
                    issue(
                        translated,
                        f"external redirect changed from {english.refresh!r} to {page.refresh!r}",
                    )

    # CSS fonts and images do not appear as HTML src/href attributes.
    for path in sorted(directory.rglob("*.css")):
        relative = path.relative_to(directory).as_posix()
        for match in CSS_URL.finditer(path.read_text(encoding="utf-8")):
            if match[2].startswith("#"):
                continue
            target = resolve_checked(relative, match[2])
            if target:
                result.references += 1
                if not (directory / target[0]).is_file():
                    issue(relative, f"CSS url({match[2]!r}) -> missing {target[0]}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Generated Hugo site directory")
    parser.add_argument("--base-url", help="Site URL including any deployment prefix")
    parser.add_argument(
        "--languages", help="Comma-separated language subset, e.g. en,zh-cn"
    )
    args = parser.parse_args()
    selected = (
        [item.strip() for item in args.languages.split(",") if item.strip()]
        if args.languages
        else None
    )
    try:
        result = check_site(args.directory, base_url=args.base_url, selected=selected)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Translation validation failed: {error}\n")
    if result.errors:
        parser.exit(
            1,
            f"Translation validation failed ({result.summary()}):\n"
            + "\n".join(result.errors)
            + "\n",
        )
    print(f"Validated {result.summary()}.")


if __name__ == "__main__":
    main()
