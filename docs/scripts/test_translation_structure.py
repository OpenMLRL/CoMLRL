"""Recovery when translations retain tokens but alter their Markdown context."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import translations


FRONT = "---\ntitle: Example\nweight: 1\n---\n"
REPO = Path(__file__).resolve().parents[2]


class StructureRecoveryTests(unittest.TestCase):
    def translate(self, source):
        return translations.translate(
            source, endpoint="unused", model="unused", language="de"
        )

    def test_added_backticks_are_recovered_without_sending_tokens_again(self):
        source = FRONT + "Use `num_agents=3`.\n"

        def initial(fields, **kwargs):
            return {
                key: translations.TOKEN.sub(lambda m: f"`{m[0]}`", value)
                for key, value in fields.items()
            }

        def recover(fields, **kwargs):
            self.assertEqual(list(fields.values()), ["Use"])
            return {key: "Verwenden" for key in fields}

        with patch.object(translations, "translate_batch", side_effect=initial):
            with patch.object(
                translations, "_request_batch", side_effect=recover
            ) as api:
                result = self.translate(source)
        self.assertIn("Verwenden `num_agents=3`.\n", result)
        api.assert_called_once()
        translations.validate_translation(source, result)

    def test_localized_parentheses_restore_link_label_and_delimiters(self):
        for destination in (
            "https://example.test/guide_(one)",
            '{{< relref "docs/guide" >}}',
        ):
            with self.subTest(destination=destination):
                source = FRONT + f"[**Guide**]({destination}).\n"

                def initial(fields, **kwargs):
                    return {
                        key: value.replace("](", "]（").replace(")", "）")
                        for key, value in fields.items()
                    }

                def recover(fields, **kwargs):
                    self.assertEqual(list(fields.values()), ["Guide"])
                    return {key: "Anleitung" for key in fields}

                with patch.object(translations, "translate_batch", side_effect=initial):
                    with patch.object(
                        translations, "_request_batch", side_effect=recover
                    ):
                        result = self.translate(source)
                self.assertIn(f"[**Anleitung**]({destination}).", result)
                translations.validate_translation(source, result)

    def test_lost_shortcode_link_is_rejected_even_when_shortcode_survives(self):
        shortcode = '{{< relref "docs/guide" >}}'
        source = FRONT + f"[Guide]({shortcode})\n"
        damaged = FRONT + f"Anleitung {shortcode}\n"
        self.assertEqual(
            translations._protected_parts(source),
            translations._protected_parts(damaged),
        )
        with self.assertRaisesRegex(ValueError, "changed protected"):
            translations.validate_translation(source, damaged)

    def test_sibling_fields_are_not_retranslated(self):
        source = FRONT + "## Stable heading\n\nRead the [Guide](../guide/).\n"

        def initial(fields, **kwargs):
            return {
                key: value.replace("Stable heading", "Stabile Überschrift")
                .replace("](", "]（")
                .replace(")", "）")
                for key, value in fields.items()
            }

        def recover(fields, **kwargs):
            self.assertEqual(list(fields.values()), ["Read the", "Guide"])
            return {
                key: {"Read the": "Öffnen", "Guide": "Anleitung"}[value]
                for key, value in fields.items()
            }

        with patch.object(translations, "translate_batch", side_effect=initial):
            with patch.object(
                translations, "_request_batch", side_effect=recover
            ) as api:
                result = self.translate(source)
        self.assertIn("## Stabile Überschrift", result)
        self.assertIn("Öffnen [Anleitung](../guide/).", result)
        api.assert_called_once()

    def test_html_wrapper_backticks_are_recovered(self):
        source = FRONT + "<strong>Welcome</strong>\n"

        def initial(fields, **kwargs):
            return {
                key: translations.TOKEN.sub(lambda m: f"`{m[0]}`", value)
                for key, value in fields.items()
            }

        def recover(fields, **kwargs):
            self.assertEqual(list(fields.values()), ["Welcome"])
            return {key: "Willkommen" for key in fields}

        with patch.object(translations, "translate_batch", side_effect=initial):
            with patch.object(translations, "_request_batch", side_effect=recover):
                result = self.translate(source)
        self.assertIn("<strong>Willkommen</strong>", result)

    def test_original_german_failure_with_incomplete_image_wrapper_is_rejected(self):
        tag = (
            '<img src="{{< relurl "img/demo-hd.webp" >}}" '
            'width="1306" height="730" alt="Example image" />'
        )
        source = FRONT + tag + "\n"
        damaged = FRONT + '<img src="' + tag + "\n"
        # The inner tag survives, so a protected-token count alone misses this.
        self.assertEqual(
            translations._protected_parts(source),
            translations._protected_parts(damaged),
        )
        with self.assertRaisesRegex(ValueError, "changed protected"):
            translations.validate_translation(source, damaged)

    def test_incomplete_added_html_wrapper_recovers_only_prose(self):
        source = FRONT + "<strong>Welcome</strong>\n"

        def initial(fields, **kwargs):
            return {
                key: '<img src="' + value if "Welcome" in value else value
                for key, value in fields.items()
            }

        def recover(fields, **kwargs):
            self.assertEqual(list(fields.values()), ["Welcome"])
            return {key: "Willkommen" for key in fields}

        with patch.object(translations, "translate_batch", side_effect=initial):
            with patch.object(translations, "_request_batch", side_effect=recover):
                result = self.translate(source)
        self.assertIn("<strong>Willkommen</strong>", result)
        self.assertNotIn("<img", result)

    def test_existing_literal_incomplete_tag_remains_permitted(self):
        source = FRONT + 'An incomplete tag: <span title="\n'
        translated = FRONT + 'Ein unvollständiges Tag: <span title="\n'
        translations.validate_translation(source, translated)

    def test_failing_recovery_is_bounded_and_does_not_write_cache(self):
        source = FRONT + "Use `num_agents=3`.\n"

        def initial(fields, **kwargs):
            return {
                key: translations.TOKEN.sub(lambda m: f"`{m[0]}`", value)
                for key, value in fields.items()
            }

        def broken(fields, **kwargs):
            return {key: "`Verwenden`" for key in fields}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(translations, "translate_batch", side_effect=initial):
                with patch.object(
                    translations, "_request_batch", side_effect=broken
                ) as api:
                    with self.assertRaisesRegex(ValueError, "structure recovery"):
                        translations.load_translation(
                            source,
                            Path("docs/example.md"),
                            cache_root=root,
                            cache_only=False,
                            endpoint="unused",
                            model="unused",
                            language="de",
                        )
            self.assertEqual(api.call_count, 2)
            self.assertEqual(list(root.rglob("*.json")), [])
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_real_overview_recovers_changed_code_context(self):
        source = (REPO / "docs/content/_index.md").read_text(encoding="utf-8")

        def initial(fields, **kwargs):
            return {
                key: (
                    translations.TOKEN.sub(lambda m: f"`{m[0]}`", value)
                    if value.startswith("Joint responses from aligned")
                    else value
                )
                for key, value in fields.items()
            }

        def recover(fields, **kwargs):
            self.assertEqual(
                list(fields.values()),
                ["Joint responses from aligned individual responses with"],
            )
            return {
                key: "Gemeinsame Antworten aus abgestimmten Einzelantworten mit"
                for key in fields
            }

        with patch.object(translations, "translate_batch", side_effect=initial):
            with patch.object(
                translations, "_request_batch", side_effect=recover
            ) as api:
                result = self.translate(source)
        api.assert_called_once()
        self.assertIn("mit `joint_mode='aligned'`.", result)
        translations.validate_translation(source, result)

    def test_initial_batches_are_small_and_bounded(self):
        for paragraph in ("Example paragraph.", "This paragraph contains text. " * 20):
            with self.subTest(paragraph_length=len(paragraph)):
                source = FRONT + "\n".join([paragraph] * 30) + "\n"
                with patch.object(
                    translations,
                    "translate_batch",
                    side_effect=lambda fields, **kwargs: fields,
                ) as api:
                    self.translate(source)
                self.assertGreater(api.call_count, 1)
                for call in api.call_args_list:
                    fields = call.args[0]
                    self.assertLessEqual(len(fields), 12)
                    self.assertLessEqual(sum(map(len, fields.values())), 1800)

    def test_short_fragment_echoes_are_rejected_after_reassembly(self):
        source = (
            "Users can [read this guide](ZXQ00000QXZ) "
            "before they initialize their language model agents."
        )
        with patch.object(
            translations, "_request_batch", side_effect=lambda fields, **kwargs: fields
        ) as api:
            with self.assertRaisesRegex(ValueError, "untranslated English echo"):
                translations._translate_structured_fragments(
                    source, endpoint="unused", model="unused", language="de"
                )
        api.assert_called_once()


if __name__ == "__main__":
    unittest.main()
