#!/usr/bin/env python3
"""Reject corrupted text and unexpanded Hugo shortcodes in generated HTML."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re


SHORTCODE = re.compile(r"\{\{\s*[<%]")


class RenderedText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ignored = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag in {"pre", "code", "script", "style"}:
            self.ignored.append(tag)
            self.text.append("\n")
        if tag == "img":
            self.check(dict(attrs).get("alt") or "")

    def handle_endtag(self, tag):
        if self.ignored and self.ignored[-1] == tag:
            self.ignored.pop()

    def handle_data(self, data):
        self.check(data)

    def check(self, text):
        if not self.ignored:
            self.text.append(text)

    @property
    def issues(self):
        # The parser can split a literal '<' across multiple text callbacks.
        text = "".join(self.text)
        issues = []
        if "\ufffd" in text:
            issues.append("Unicode replacement character in rendered text")
        if SHORTCODE.search(text):
            issues.append("unexpanded Hugo shortcode in rendered text")
        return issues


def check_site(directory):
    files = sorted(directory.rglob("*.html"))
    if not files:
        raise ValueError(f"No HTML files found in {directory}")
    issues = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"{path.relative_to(directory)}: invalid UTF-8")
            continue
        parser = RenderedText()
        parser.feed(source)
        parser.close()
        issues.extend(
            f"{path.relative_to(directory)}: {issue}" for issue in parser.issues
        )
    if issues:
        raise ValueError("\n".join(issues))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Hugo output directory")
    args = parser.parse_args()
    try:
        count = check_site(args.directory)
    except ValueError as error:
        parser.exit(1, f"HTML validation failed:\n{error}\n")
    print(f"Validated {count} HTML files: no corrupted text or unexpanded shortcodes.")


if __name__ == "__main__":
    main()
