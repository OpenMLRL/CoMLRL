"""Translation service and cache recovery checks without a running model."""

from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import translations


class ServiceRecoveryTests(unittest.TestCase):
    @staticmethod
    def response(fields):
        return StringIO(json.dumps({"message": {"content": json.dumps(fields)}}))

    @staticmethod
    def requested_fields(call):
        payload = json.loads(call.args[0].data)
        return json.loads(payload["messages"][-1]["content"])

    def translate(self, fields):
        return translations.translate_batch(
            fields, endpoint="http://local-test", model="test", language="fr"
        )

    def test_single_field_timeout_retries_once(self):
        with patch.object(
            translations,
            "urlopen",
            side_effect=[TimeoutError("timed out"), self.response({"s0": "Titre"})],
        ) as api:
            self.assertEqual(self.translate({"s0": "Heading"}), {"s0": "Titre"})
        self.assertEqual(api.call_count, 2)
        self.assertTrue(
            all(call.kwargs["timeout"] == 300 for call in api.call_args_list)
        )

    def test_persistent_network_errors_have_bounded_retries(self):
        for error in (
            TimeoutError("timed out"),
            URLError("connection refused"),
            ConnectionResetError("connection reset"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch.object(translations, "urlopen", side_effect=error) as api:
                    with self.assertRaisesRegex(
                        translations.TranslationServiceError, "failed fields: s0"
                    ):
                        self.translate({"s0": "Heading"})
                self.assertEqual(api.call_count, 2)

    def test_failed_batch_recovers_in_two_smaller_requests(self):
        fields = {f"s{i}": f"Heading {i}" for i in range(5)}
        first = {key: "Titre" for key in list(fields)[:3]}
        second = {key: "Titre" for key in list(fields)[3:]}
        with patch.object(
            translations,
            "urlopen",
            side_effect=[
                TimeoutError("timed out"),
                self.response(first),
                self.response(second),
            ],
        ) as api:
            self.assertEqual(self.translate(fields), {**first, **second})
        self.assertEqual(api.call_count, 3)
        self.assertEqual(self.requested_fields(api.call_args_list[0]), fields)
        self.assertEqual(
            list(self.requested_fields(api.call_args_list[1])), list(first)
        )
        self.assertEqual(
            list(self.requested_fields(api.call_args_list[2])), list(second)
        )

    def test_failed_split_does_not_split_recursively(self):
        with patch.object(translations, "urlopen", side_effect=TimeoutError()) as api:
            with self.assertRaises(translations.TranslationServiceError):
                self.translate({f"s{i}": "Heading" for i in range(12)})
        self.assertEqual(api.call_count, 2)
        self.assertEqual(len(self.requested_fields(api.call_args_list[1])), 6)

    def test_transient_http_statuses_can_recover(self):
        for status in (429, 500, 503):
            with self.subTest(status=status):
                with patch.object(
                    translations,
                    "urlopen",
                    side_effect=[
                        HTTPError(
                            "http://local-test", status, "service unavailable", {}, None
                        ),
                        self.response({"s0": "Titre"}),
                    ],
                ) as api:
                    self.assertEqual(self.translate({"s0": "Heading"}), {"s0": "Titre"})
                self.assertEqual(api.call_count, 2)

    def test_permanent_http_statuses_fail_without_retry(self):
        for status in (400, 401, 404):
            with self.subTest(status=status):
                with patch.object(
                    translations,
                    "urlopen",
                    side_effect=HTTPError(
                        "http://local-test", status, "bad request", {}, None
                    ),
                ) as api:
                    with self.assertRaisesRegex(
                        RuntimeError, f"HTTP {status}"
                    ) as caught:
                        self.translate({"s0": "Heading"})
                self.assertNotIsInstance(
                    caught.exception, translations.TranslationServiceError
                )
                self.assertEqual(api.call_count, 1)

    def test_recovery_retains_strict_response_validation(self):
        for invalid in ({}, {"unexpected": "Titre"}, ["Titre"]):
            with self.subTest(invalid=invalid):
                with patch.object(
                    translations,
                    "urlopen",
                    side_effect=[TimeoutError(), self.response(invalid)],
                ) as api:
                    with self.assertRaises(ValueError):
                        self.translate({"s0": "Heading"})
                self.assertEqual(api.call_count, 2)

    def test_recovery_does_not_accept_english_echo(self):
        fields = {
            "s0": "Users can initialize the agents with the model identifier shown below."
        }
        with patch.object(
            translations,
            "urlopen",
            side_effect=[TimeoutError(), self.response(fields)],
        ) as api:
            with self.assertRaisesRegex(ValueError, "untranslated English echo"):
                self.translate(fields)
        self.assertEqual(api.call_count, 2)

    def test_request_schema_remains_strict(self):
        with patch.object(
            translations, "urlopen", return_value=self.response({"s0": "Titre"})
        ) as api:
            self.translate({"s0": "Heading"})
        payload = json.loads(api.call_args.args[0].data)
        self.assertEqual(payload["format"]["required"], ["s0"])
        self.assertFalse(payload["format"]["additionalProperties"])
        self.assertEqual(payload["format"]["properties"], {"s0": {"type": "string"}})


class FragmentRecoveryTests(unittest.TestCase):
    def test_shifted_madpo_fragment_batch_is_fully_retranslated(self):
        # The fresh German MADPO run merged p10 into p8, shifted p12/p14,
        # and left p14 empty. Retrying just p14 would retain wrong meanings.
        source = (
            "For each task item in ZXQ00011QXZ, every agent samples ZXQ00012QXZ "
            "responses from its formatted prompt. Responses with the same candidate "
            "index form a joint response and share one reward from ZXQ00013QXZ "
            "(after the optional ZXQ00014QXZ). For two joint responses with different "
            "rewards, the response with the higher reward is the winner ZXQ00015QXZ "
            "and the other the loser ZXQ00016QXZ; ties are discarded. Each pair "
            "ZXQ00017QXZ contains the two joint histories for one task."
        )
        valid = {
            "p0": "对于每个任务条目",
            "p2": "每个智能体采样",
            "p4": "来自格式化提示的响应。同一候选索引的响应构成联合响应，并共享一个奖励，来自",
            "p6": "在可选步骤之后",
            "p8": "对于奖励不同的两个联合响应，奖励较高的响应是胜者",
            "p10": "另一个是败者",
            "p12": "平局会被丢弃。每一对",
            "p14": "包含一个任务的两段联合历史",
        }
        invalid = {
            **valid,
            "p8": valid["p8"] + valid["p10"],
            "p10": valid["p12"],
            "p12": valid["p14"],
            "p14": "",
        }

        def respond(fields, **kwargs):
            if len(fields) > 1:
                return invalid
            return {key: valid[key] for key in fields}

        with patch.object(translations, "_request_batch", side_effect=respond) as api:
            result = translations._translate_fragments(
                source, endpoint="unused", model="test", language="zh-cn"
            )
        fields = api.call_args_list[0].args[0]
        expected = source
        for key, original in fields.items():
            expected = expected.replace(original, valid[key])
        self.assertEqual(result, expected)
        self.assertEqual(api.call_count, 9)
        self.assertEqual(
            [call.args[0] for call in api.call_args_list[1:]],
            [{key: value} for key, value in fields.items()],
        )
        self.assertEqual(
            translations.TOKEN.findall(result), translations.TOKEN.findall(source)
        )

    def test_persistent_invalid_individual_fragment_stops_after_two_calls(self):
        fields = {"p0": "Users can initialize the agents with a saved model."}
        for invalid in (
            {},
            ["译文"],
            {"unexpected": "译文"},
            {"p0": ""},
            {"p0": "两行\n译文"},
            {"p0": "ZXQ00000QXZ 译文"},
            {"p0": fields["p0"]},
            {"p0": "`增加的代码标记`"},
        ):
            with self.subTest(invalid=invalid):
                with patch.object(
                    translations, "_request_batch", return_value=invalid
                ) as api:
                    with self.assertRaises(ValueError):
                        translations._translate_prose_fragments(
                            fields, endpoint="unused", model="test", language="zh-cn"
                        )
                self.assertEqual(api.call_count, 2)

    def test_malformed_fragment_batch_recovers_individually(self):
        fields = {"p0": "First part", "p2": "Second part"}
        with patch.object(
            translations,
            "_request_batch",
            side_effect=[
                ValueError("Translator returned malformed JSON"),
                {"p0": "第一部分"},
                {"p2": "第二部分"},
            ],
        ) as api:
            result = translations._translate_prose_fragments(
                fields, endpoint="unused", model="test", language="zh-cn"
            )
        self.assertEqual(result, {"p0": "第一部分", "p2": "第二部分"})
        self.assertEqual(api.call_count, 3)

    def test_valid_fragment_batch_needs_one_request(self):
        expected = {"p0": "第一部分", "p2": "第二部分"}
        with patch.object(translations, "_request_batch", return_value=expected) as api:
            self.assertEqual(
                translations._translate_prose_fragments(
                    {"p0": "First part", "p2": "Second part"},
                    endpoint="unused",
                    model="test",
                    language="zh-cn",
                ),
                expected,
            )
        api.assert_called_once()

    def test_fragment_service_recovery_is_bounded(self):
        fields = {"p0": "First part", "p2": "Second part"}
        with patch.object(
            translations,
            "_request_batch",
            side_effect=[
                translations.TranslationServiceError("timed out"),
                {"p0": "第一部分"},
                {"p2": "第二部分"},
            ],
        ) as api:
            result = translations._translate_prose_fragments(
                fields, endpoint="unused", model="test", language="zh-cn"
            )
        self.assertEqual(result, {"p0": "第一部分", "p2": "第二部分"})
        self.assertEqual(api.call_count, 3)
        with patch.object(
            translations,
            "_request_batch",
            side_effect=translations.TranslationServiceError("timed out"),
        ) as api:
            with self.assertRaises(translations.TranslationServiceError):
                translations._translate_prose_fragments(
                    fields, endpoint="unused", model="test", language="zh-cn"
                )
        self.assertEqual(api.call_count, 2)
        with patch.object(
            translations, "_request_batch", side_effect=RuntimeError("HTTP 404")
        ) as api:
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                translations._translate_prose_fragments(
                    fields, endpoint="unused", model="test", language="zh-cn"
                )
        api.assert_called_once()

    def test_structure_fallback_also_recovers_empty_fragment_batch(self):
        source = "Read [Guide](ZXQ00000QXZ)."
        with patch.object(
            translations,
            "_request_batch",
            side_effect=[
                {"p0": "阅读指南", "p2": ""},
                {"p0": "阅读"},
                {"p2": "指南"},
            ],
        ) as api:
            result = translations._translate_structured_fragments(
                source, endpoint="unused", model="test", language="zh-cn"
            )
        self.assertEqual(result, "阅读 [指南](ZXQ00000QXZ).")
        self.assertEqual(api.call_count, 3)


class CacheRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="comlrl-runtime-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = Path("docs/example.md")
        self.cache = self.root / "docs/example.md.json"
        self.source = (
            '---\ntitle: "Example"\nweight: 1\n---\n# Example\n\nOriginal prose.\n'
        )
        self.translated = self.source.replace("Example", "Présentation").replace(
            "Original prose.", "Texte original."
        )
        self.record = {
            "language": "fr",
            "source_sha256": translations.digest(self.source),
            "translator": "ollama:old",
            "translation": self.translated,
        }

    def write_cache(self, content):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text(content, encoding="utf-8")

    def load(self, *, cache_only=False):
        return translations.load_translation(
            self.source,
            self.path,
            cache_root=self.root,
            cache_only=cache_only,
            endpoint="http://local-test",
            model="test",
            language="fr",
        )

    def test_invalid_current_cache_regenerates_in_live_mode(self):
        self.record["translation"] = self.translated.replace(
            "# Présentation", "## Présentation"
        )
        self.write_cache(json.dumps(self.record))
        with patch.object(
            translations, "translate", return_value=self.translated
        ) as model:
            self.assertEqual(self.load(), self.translated)
        model.assert_called_once()
        record = json.loads(self.cache.read_text())
        self.assertEqual(record["translation"], self.translated)
        self.assertEqual(record["translator"], "ollama:test")
        self.assertFalse(self.cache.with_suffix(".tmp").exists())

    def test_corrupt_cache_regenerates_in_live_mode(self):
        for invalid in (
            "{broken",
            "[]",
            json.dumps({**self.record, "translation": None}),
        ):
            with self.subTest(invalid=invalid):
                self.write_cache(invalid)
                with patch.object(
                    translations, "translate", return_value=self.translated
                ) as model:
                    self.assertEqual(self.load(), self.translated)
                model.assert_called_once()
                self.assertEqual(
                    json.loads(self.cache.read_text())["translation"], self.translated
                )

    def test_invalid_cache_only_fails_with_path_and_language(self):
        self.record["translation"] = self.translated.replace(
            "# Présentation", "## Présentation"
        )
        self.write_cache(json.dumps(self.record))
        with patch.object(translations, "translate") as model:
            with self.assertRaisesRegex(
                ValueError,
                "Invalid French translation cache for docs/example.md:.*heading hierarchy",
            ):
                self.load(cache_only=True)
        model.assert_not_called()

    def test_valid_cache_from_older_translator_is_reused(self):
        self.write_cache(json.dumps(self.record))
        with patch.object(translations, "translate") as model:
            self.assertEqual(self.load(), self.translated)
        model.assert_not_called()

    def test_failed_regeneration_preserves_existing_cache(self):
        original = "{broken"
        self.write_cache(original)
        with patch.object(
            translations,
            "translate",
            side_effect=ValueError("changed protected content"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "French translation failed for docs/example.md: changed protected content",
            ):
                self.load()
        self.assertEqual(self.cache.read_text(), original)
        self.assertFalse(self.cache.with_suffix(".tmp").exists())

    def test_service_failure_has_page_context_and_creates_no_cache(self):
        with patch.object(
            translations,
            "translate",
            side_effect=translations.TranslationServiceError("timed out"),
        ):
            with self.assertRaisesRegex(
                translations.TranslationServiceError,
                "French translation failed for docs/example.md: timed out",
            ):
                self.load()
        self.assertEqual(list(self.root.rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
