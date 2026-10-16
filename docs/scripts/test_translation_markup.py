"""HTML translation protection, including Hugo shortcodes in attributes."""

from pathlib import Path
import re
import unittest
from unittest.mock import patch

import translations


FRONT = "---\ntitle: Example\nweight: 1\n---\n"
REPO = Path(__file__).resolve().parents[2]


class HTMLProtectionTests(unittest.TestCase):
    def assert_tag_protected(self, tag):
        self.assertEqual(
            [match[0] for match in translations.PROTECTED.finditer(tag)], [tag]
        )

    def test_overview_image_is_one_complete_protected_span(self):
        source = (REPO / "docs/content/_index.md").read_text(encoding="utf-8")
        image = next(line.strip() for line in source.splitlines() if "<img " in line)
        self.assert_tag_protected(image)
        self.assertTrue(image.endswith("/>"))
        self.assertIn('width="1306"', image)

    def test_complete_html_attributes_are_hidden_from_translation(self):
        source = FRONT + (
            '<img src="{{< relurl "img/demo-hd.webp" >}}" '
            'width="1306" height="730" alt="Example image" />\n'
            "Users can view the image above.\n"
        )
        seen = []

        def translate(fields, **kwargs):
            seen.extend(fields.values())
            return {
                key: value.replace('width="1306"', 'width="10"').replace(
                    "Users can view the image above.", "Benutzer sehen das Bild oben."
                )
                for key, value in fields.items()
            }

        with patch.object(translations, "translate_batch", side_effect=translate):
            result = translations.translate(
                source, endpoint="unused", model="unused", language="de"
            )
        self.assertIn('width="1306"', result)
        self.assertIn("Benutzer sehen das Bild oben.", result)
        for protected in ("relurl", "img/demo-hd.webp", "width=", "height=", "alt="):
            self.assertNotIn(protected, " ".join(seen))

    def test_changes_after_shortcode_are_rejected(self):
        source = FRONT + (
            '<img src="{{< relurl "img/demo-hd.webp" >}}" '
            'width="1306" height="730" decoding="async" />\n'
        )
        for before, after in (
            ('width="1306"', 'width="10"'),
            ('height="730"', 'height="1"'),
            ('decoding="async"', 'decoding="sync"'),
            ("img/demo-hd.webp", "img/other.webp"),
        ):
            with self.subTest(attribute=before):
                with self.assertRaisesRegex(ValueError, "changed protected"):
                    translations.validate_translation(
                        source, source.replace(before, after)
                    )

    def test_quoted_greater_than_characters_and_multiline_attributes(self):
        tags = (
            '<span title="a > b" class="example">',
            "<span title='a > b' class='example'>",
            '<img\n src="image.webp"\n data-caption="a > b"\n width="100" />',
            "<custom-element data-value='a > b'>",
        )
        for tag in tags:
            with self.subTest(tag=tag):
                self.assert_tag_protected(tag)

    def test_shortcode_quotes_do_not_close_the_surrounding_attribute(self):
        tags = (
            '<a href="{{< relref "docs/guide" >}}" class="example">',
            "<a href='{{< relref \"docs/guide\" >}}' class='example'>",
            '<a href="{{% relref "docs/guide" %}}" class="example">',
            '<img src="{{< relurl "img/example >.webp" >}}" width="100" />',
            '<img src="{{< relurl "img/example >}}.webp" >}}" width="100" />',
            '<img src="{{< relurl `img/example >}}.webp` >}}" width="100" />',
            '<img src="{{< relurl "img/\\"example.webp" >}}" width="100" />',
        )
        for tag in tags:
            with self.subTest(tag=tag):
                self.assert_tag_protected(tag)

    def test_inline_html_shortcodes_and_link_destinations_remain_separate(self):
        source = (
            '<strong>Yes.</strong> {{< relref "docs/guide" >}} '
            "[Guide](<../path with spaces>) <br />"
        )
        self.assertEqual(
            [match[0] for match in translations.PROTECTED.finditer(source)],
            [
                "<strong>",
                "</strong>",
                '{{< relref "docs/guide" >}}',
                "<../path with spaces>",
                "<br />",
            ],
        )

    def test_unclosed_tag_does_not_swallow_remaining_document(self):
        source = '<span title="unfinished\n\nUse `num_agents=3`.\n'
        self.assertEqual(
            [match[0] for match in translations.PROTECTED.finditer(source)],
            ["`num_agents=3`"],
        )

    def test_all_active_documents_round_trip_without_markup_changes(self):
        count = 0
        with patch.object(
            translations, "translate_batch", side_effect=lambda fields, **kwargs: fields
        ):
            for path in sorted((REPO / "docs/content").rglob("*.md")):
                source = path.read_text(encoding="utf-8")
                front = translations.FRONT.match(source)
                if front and re.search(r"(?m)^draft:\s*true\s*$", front[1]):
                    continue
                with self.subTest(path=path.relative_to(REPO)):
                    result = translations.translate(
                        source, endpoint="unused", model="unused", language="de"
                    )
                    translations.validate_translation(source, result)
                    self.assertEqual(
                        translations.FRONT.sub("", source),
                        translations.FRONT.sub("", result),
                    )
                    count += 1
        self.assertGreaterEqual(count, 31)


if __name__ == "__main__":
    unittest.main()
