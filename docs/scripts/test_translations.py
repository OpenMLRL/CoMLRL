"""Regression checks for translation protection; no model or network required."""

from io import StringIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import translations


FRONT = "---\ntitle: Example\nweight: 1\n---\n"
REPO = Path(__file__).resolve().parents[2]


class ProtectionTests(unittest.TestCase):
    def assert_rejected(self, source, translated):
        with self.assertRaises(ValueError):
            translations.validate_translation(FRONT + source, FRONT + translated)

    def test_longer_closing_fences_protect_the_entire_block(self):
        for opening, closing in (("```", "````"), ("~~~", "~~~~~")):
            with self.subTest(opening=opening):
                block = (
                    f'{opening}python\n# Internal heading\nprint("hello")\n{closing}\n'
                )
                self.assertEqual(
                    [m[0] for m in translations.PROTECTED.finditer(block)], [block]
                )
                self.assert_rejected(block, block.replace('"hello"', '"changed"'))
                self.assertEqual(
                    translations.heading_lines(FRONT + block + "# Real\n"),
                    [(8, 1, "Real")],
                )

    def test_inline_code_with_internal_backticks_and_newlines(self):
        for code in ("``a`b``", "`a``b`", "``a\nb``"):
            with self.subTest(code=code):
                self.assertEqual(
                    [m[0] for m in translations.PROTECTED.finditer(code)], [code]
                )
                self.assert_rejected("Use " + code, "Use " + code.replace("b", "c"))

    def test_link_destinations_with_nested_parentheses(self):
        for target in (
            "https://example.test/a_(b)/old",
            "../a_(b_(c))/old",
            r"../a_\(b\)/old",
            "<../some page.md>",
        ):
            with self.subTest(target=target):
                source = f"[Help]({target})\n"
                self.assertEqual(
                    [m[0] for m in translations.PROTECTED.finditer(source)], [target]
                )
                self.assert_rejected(
                    source, source.replace("old", "new").replace("page", "other")
                )

    def test_reference_links_preserve_definition_and_reference_id(self):
        source = '[Help][guide]\n\n[guide]: ../target/ "Optional title"\n'
        self.assert_rejected(source, source.replace("../target/", "../other/"))
        self.assert_rejected(source, source.replace("[Help][guide]", "[Help][other]"))
        translations.validate_translation(
            FRONT + source, FRONT + source.replace("[Help]", "[帮助]")
        )

    def test_model_identifiers_and_ascii_counts(self):
        source = "Load 3x _Qwen/Qwen2.5-1.5B_ agents.\n"
        self.assertEqual(
            [m[0] for m in translations.PROTECTED.finditer(source)],
            ["3x", "Qwen/Qwen2.5-1.5B"],
        )
        self.assert_rejected(source, source.replace("3x", "3 个"))
        self.assert_rejected(source, source.replace("Qwen2.5", "Qwen2.6"))
        # A prose length ratio remains translatable; x here denotes agent count.
        translations.validate_translation(
            FRONT + "Use 2–3× as many characters.\n", FRONT + "使用 2–3 倍的字符数。\n"
        )

    def test_structure_damage_is_rejected(self):
        source = "# Heading\n\n## Child\n"
        self.assert_rejected(source, source.replace("## Child", "### Child"))
        self.assert_rejected(source, source.replace("## Child\n", ""))
        with self.assertRaises(ValueError):
            translations.validate_translation(
                FRONT + source, (FRONT + source).replace("weight: 1", "weight: 2")
            )
        self.assert_rejected(
            '```python\nprint("hello")\n```\n', '```python\nprint("hello")\n'
        )

    def test_comments_and_multiline_code_do_not_create_headings(self):
        source = (
            FRONT + "<!--\n# Not a heading\n-->\n``code\n# Still code\n``\n# Heading\n"
        )
        self.assertEqual(
            [h[2] for h in translations.heading_lines(source)], ["Heading"]
        )

    def test_math_and_shortcode_parameters_are_protected(self):
        source = '{{< katex >}}x+y{{< /katex >}}\n{{< rollout-diagram diagram="mapl" title="Example" >}}\n'
        self.assert_rejected(source, source.replace("x+y", "x-y"))
        self.assert_rejected(
            source, source.replace('diagram="mapl"', 'diagram="other"')
        )
        translations.validate_translation(
            FRONT + source, FRONT + source.replace('title="Example"', 'title="示例"')
        )

    def test_automatic_masking_hides_protected_content(self):
        source = FRONT + (
            "# Heading\n\nLoad 3x _Qwen/Qwen2.5-1.5B_ agents with ``a`b``.\n"
            "[Help](https://example.test/a_(b)/old)\n\n"
            '```python\nprint("protected content")\n````\n'
        )
        seen = []

        def identity(fields, **kwargs):
            seen.extend(fields.values())
            return fields

        with patch.object(translations, "translate_batch", side_effect=identity):
            result = translations.translate(source, endpoint="unused", model="unused")
        fields = " ".join(seen)
        for protected in (
            "3x",
            "Qwen/Qwen2.5-1.5B",
            "a`b",
            "https://",
            "protected content",
        ):
            self.assertNotIn(protected, fields)
        translations.validate_translation(source, result)
        self.assertEqual(
            translations.FRONT.sub("", source), translations.FRONT.sub("", result)
        )


class BatchTranslationTests(unittest.TestCase):
    @staticmethod
    def response(fields):
        return StringIO(
            json.dumps({"message": {"content": json.dumps(fields, ensure_ascii=False)}})
        )

    @staticmethod
    def requested_fields(call):
        request = json.loads(call.args[0].data)
        return json.loads(request["messages"][-1]["content"])

    def test_only_failed_field_is_retried_and_restored(self):
        fields = {
            "s0": "Keep ZXQ00000QXZ unchanged.",
            "s1": "Use ZXQ00001QXZ with ZXQ00002QXZ.",
        }
        first = {"s0": "保持 ZXQ00000QXZ 不变。", "s1": "使用 ZXQ00001QXZ。"}
        recovered = {"s1": "将 ZXQ00001QXZ 与 ZXQ00002QXZ 配合使用。"}
        with patch.object(
            translations,
            "urlopen",
            side_effect=[self.response(first), self.response(recovered)],
        ) as api:
            result = translations.translate_batch(
                fields, endpoint="http://unused", model="test"
            )
        self.assertEqual(result, {"s0": first["s0"], **recovered})
        self.assertEqual(api.call_count, 2)
        self.assertEqual(self.requested_fields(api.call_args_list[0]), fields)
        self.assertEqual(
            self.requested_fields(api.call_args_list[1]), {"s1": fields["s1"]}
        )

    def test_persistent_invalid_response_fails_without_writing_cache(self):
        first = {"s0": "示例", "s1": "使用智能体。"}
        second = {"s1": "使用智能体。"}
        with TemporaryDirectory(prefix="comlrl-failed-translation-test-") as directory:
            root = Path(directory)
            with patch.object(
                translations,
                "urlopen",
                side_effect=[
                    self.response(first),
                    self.response(second),
                    self.response({}),
                    self.response({}),
                ],
            ) as api:
                with self.assertRaisesRegex(ValueError, "invalid fragment fields"):
                    translations.load_translation(
                        FRONT + "Use 3x agents.\n",
                        Path("docs/example.md"),
                        cache_root=root,
                        cache_only=False,
                        endpoint="http://unused",
                        model="test",
                    )
            self.assertEqual(api.call_count, 4)
            self.assertEqual(
                self.requested_fields(api.call_args_list[1]),
                {"s1": "Use ZXQ00000QXZ agents."},
            )
            self.assertEqual(list(root.rglob("*")), [])

    def test_fragment_fallback_preserves_tokens_outside_generation(self):
        fields = {"s0": "Use ZXQ00001QXZ agents with ZXQ00002QXZ."}
        bad = {"s0": "使用智能体。"}
        fragments = {"p0": "使用", "p2": "智能体，配置为"}
        with patch.object(
            translations,
            "urlopen",
            side_effect=[
                self.response(bad),
                self.response(bad),
                self.response(fragments),
            ],
        ) as api:
            result = translations.translate_batch(
                fields, endpoint="http://unused", model="test"
            )
        self.assertEqual(result["s0"], "使用 ZXQ00001QXZ 智能体，配置为 ZXQ00002QXZ.")
        self.assertEqual(
            self.requested_fields(api.call_args_list[2]),
            {"p0": "Use", "p2": "agents with"},
        )

    def test_english_echo_is_rejected(self):
        fields = {
            "s0": "Users can initialize the agents with the model identifier shown below."
        }
        with patch.object(
            translations,
            "urlopen",
            side_effect=[self.response(fields), self.response(fields)],
        ) as api:
            with self.assertRaisesRegex(ValueError, "no Simplified Chinese text"):
                translations.translate_batch(
                    fields, endpoint="http://unused", model="test"
                )
        self.assertEqual(api.call_count, 2)

    def test_missing_field_retries_only_that_field(self):
        fields = {"s0": "First heading", "s1": "Second heading"}
        with patch.object(
            translations,
            "urlopen",
            side_effect=[
                self.response({"s0": "Premier titre"}),
                self.response({"s1": "Deuxième titre"}),
            ],
        ) as api:
            result = translations.translate_batch(
                fields, endpoint="http://unused", model="test", language="fr"
            )
        self.assertEqual(result, {"s0": "Premier titre", "s1": "Deuxième titre"})
        self.assertEqual(api.call_count, 2)
        self.assertEqual(
            self.requested_fields(api.call_args_list[1]), {"s1": fields["s1"]}
        )

    def test_malformed_batch_splits_into_single_field_retries(self):
        fields = {"s0": "First heading", "s1": "Second heading"}
        malformed = StringIO(json.dumps({"message": {"content": '{"s0":'}}))
        with patch.object(
            translations,
            "urlopen",
            side_effect=[
                malformed,
                self.response({"s0": "Erste Überschrift"}),
                self.response({"s1": "Zweite Überschrift"}),
            ],
        ) as api:
            result = translations.translate_batch(
                fields, endpoint="http://unused", model="test", language="de"
            )
        self.assertEqual(
            result, {"s0": "Erste Überschrift", "s1": "Zweite Überschrift"}
        )
        self.assertEqual(api.call_count, 3)
        self.assertEqual(
            self.requested_fields(api.call_args_list[1]), {"s0": fields["s0"]}
        )
        self.assertEqual(
            self.requested_fields(api.call_args_list[2]), {"s1": fields["s1"]}
        )

    def test_persistently_malformed_json_has_a_bounded_retry(self):
        responses = [
            StringIO(json.dumps({"message": {"content": "not JSON"}})) for _ in range(2)
        ]
        with patch.object(translations, "urlopen", side_effect=responses) as api:
            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                translations.translate_batch(
                    {"s0": "Heading"},
                    endpoint="http://unused",
                    model="test",
                    language="es",
                )
        self.assertEqual(api.call_count, 2)

    def test_persistent_missing_or_unexpected_fields_fail_closed(self):
        for wrong in ({}, {"unexpected": "Texto"}, ["Texto"]):
            with self.subTest(response=wrong):
                with patch.object(
                    translations,
                    "urlopen",
                    side_effect=[self.response(wrong), self.response(wrong)],
                ) as api:
                    with self.assertRaises(ValueError):
                        translations.translate_batch(
                            {"s0": "Heading"},
                            endpoint="http://unused",
                            model="test",
                            language="es",
                        )
                self.assertEqual(api.call_count, 2)


class LanguageTests(unittest.TestCase):
    response = staticmethod(BatchTranslationTests.response)

    def test_catalog_contains_shared_metadata_and_target_glossaries(self):
        self.assertEqual(
            list(translations.LANGUAGES), ["en", "zh-cn", "ja", "ko", "es", "fr", "de"]
        )
        for weight, (language, info) in enumerate(translations.LANGUAGES.items(), 1):
            with self.subTest(language=language):
                self.assertEqual(info["weight"], weight)
                self.assertEqual(info["cjk"], language in {"zh-cn", "ja", "ko"})
                for field in (
                    "name",
                    "languageCode",
                    "target",
                    "instruction",
                    "example",
                    "glossary",
                ):
                    self.assertTrue(info[field])
                self.assertEqual(
                    translations.TOKEN.findall(info["example"]),
                    ["ZXQ00000QXZ", "ZXQ00001QXZ"],
                )

    def test_target_prompt_uses_catalog_and_accepts_each_language(self):
        source = (
            "Users can initialize the agents with the model identifier shown below."
        )
        results = {
            "zh-cn": "用户可以使用下方所示的模型标识符初始化智能体。",
            "ja": "ユーザーは以下のモデル識別子でエージェントを初期化できます。",
            "ko": "사용자는 아래의 모델 식별자로 에이전트를 초기화할 수 있습니다.",
            "es": "Los usuarios pueden inicializar los agentes con el identificador del modelo que aparece a continuación.",
            "fr": "Les utilisateurs peuvent initialiser les agents avec l’identifiant du modèle indiqué ci-dessous.",
            "de": "Benutzer können die Agenten mit der unten angegebenen Modellkennung initialisieren.",
        }
        for language, text in results.items():
            with self.subTest(language=language):
                with patch.object(
                    translations, "urlopen", return_value=self.response({"s0": text})
                ) as api:
                    result = translations.translate_batch(
                        {"s0": source},
                        endpoint="http://unused",
                        model="test",
                        language=language,
                    )
                self.assertEqual(result, {"s0": text})
                api.assert_called_once()
                prompt = json.loads(api.call_args.args[0].data)["messages"][0][
                    "content"
                ]
                info = translations.LANGUAGES[language]
                for expected in (
                    info["target"],
                    info["name"],
                    info["instruction"],
                    info["example"],
                ):
                    self.assertIn(expected, prompt)
                for term in info["glossary"].values():
                    self.assertIn(term, prompt)

    def test_language_scripts_distinguish_hangul_kana_and_han(self):
        source = (
            "Users can initialize the agents with the model identifier shown below."
        )
        allowed = {"zh-cn": "模型初始化", "ja": "モデル", "ko": "모델 초기화"}
        wrong = {"zh-cn": "모델 초기화", "ja": "모델 초기화", "ko": "模型初始化"}
        for language in allowed:
            with self.subTest(language=language):
                self.assertIsNone(
                    translations._field_error(source, allowed[language], language)
                )
                self.assertIsNotNone(
                    translations._field_error(source, wrong[language], language)
                )
        self.assertIsNone(translations._field_error(source, "模型初期化", "ja"))

    def test_non_cjk_english_echo_is_rejected_after_one_retry(self):
        fields = {
            "s0": "Users can initialize the agents with the model identifier shown below."
        }
        for language in ("es", "fr", "de"):
            with self.subTest(language=language):
                with patch.object(
                    translations,
                    "urlopen",
                    side_effect=[self.response(fields), self.response(fields)],
                ) as api:
                    with self.assertRaisesRegex(
                        ValueError, "untranslated English echo"
                    ):
                        translations.translate_batch(
                            fields,
                            endpoint="http://unused",
                            model="test",
                            language=language,
                        )
                self.assertEqual(api.call_count, 2)

    def test_unknown_language_fails_before_request(self):
        with patch.object(translations, "urlopen") as api:
            with self.assertRaisesRegex(ValueError, "Unsupported translation language"):
                translations.translate_batch(
                    {"s0": "Heading"},
                    endpoint="http://unused",
                    model="test",
                    language="unknown",
                )
        api.assert_not_called()

    def test_translate_forwards_language_without_touching_protected_text(self):
        source = FRONT + "# Heading\n\nUse `num_agents=3`.\n"
        with patch.object(
            translations, "translate_batch", side_effect=lambda fields, **kwargs: fields
        ) as batch:
            result = translations.translate(
                source, endpoint="unused", model="test", language="fr"
            )
        self.assertEqual(batch.call_args.kwargs["language"], "fr")
        translations.validate_translation(source, result)

    def test_translate_keeps_rl_terms_in_english(self):
        source = (
            FRONT + "# Loading Critics\n\n"
            "Each Actor-Critic actor and its critic share rollouts; "
            "`critic_model` stays code.\n"
        )
        with patch.object(
            translations, "translate_batch", side_effect=lambda fields, **kwargs: fields
        ) as batch:
            result = translations.translate(
                source, endpoint="unused", model="test", language="zh-cn"
            )
        sent = [
            value for call in batch.call_args_list for value in call.args[0].values()
        ]
        for value in sent:
            self.assertNotRegex(value, r"(?i)actor|critic|rollout")
        self.assertTrue(result.endswith(source[len(FRONT) :]))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="comlrl-translation-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = Path("docs/example.md")
        self.cache = self.root / "docs/example.md.json"
        self.cache.parent.mkdir(parents=True)
        self.source = FRONT + "# Example\n\nOriginal prose.\n"
        self.translated = FRONT.replace("Example", "示例") + "# 示例\n\n原始正文。\n"
        self.cache.write_text(
            json.dumps(
                {
                    "language": "zh-cn",
                    "source_sha256": translations.digest(self.source),
                    "translator": "test",
                    "translation": self.translated,
                }
            ),
            encoding="utf-8",
        )

    def load(self, source, *, cache_only):
        return translations.load_translation(
            source,
            self.path,
            cache_root=self.root,
            cache_only=cache_only,
            endpoint="unused",
            model="test",
        )

    def test_current_cache_does_not_call_model(self):
        with patch.object(translations, "translate") as model:
            self.assertEqual(self.load(self.source, cache_only=True), self.translated)
        model.assert_not_called()

    def test_stale_cache_only_fails_without_model_call(self):
        with patch.object(translations, "translate") as model:
            with self.assertRaisesRegex(ValueError, "Missing or stale"):
                self.load(self.source.replace("Original", "Changed"), cache_only=True)
        model.assert_not_called()

    def test_source_change_regenerates_cache_atomically(self):
        source = self.source.replace("Original", "Changed")
        translated = self.translated.replace("原始", "更新后")
        with patch.object(translations, "translate", return_value=translated) as model:
            self.assertEqual(self.load(source, cache_only=False), translated)
        model.assert_called_once_with(
            source, endpoint="unused", model="test", language="zh-cn"
        )
        record = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertEqual(record["source_sha256"], translations.digest(source))
        self.assertEqual(record["translation"], translated)
        self.assertEqual(record["translator"], "ollama:test")
        self.assertFalse(self.cache.with_suffix(".tmp").exists())

    def test_invalid_current_cache_is_rejected(self):
        record = json.loads(self.cache.read_text(encoding="utf-8"))
        record["translation"] = self.translated.replace("# 示例", "## 示例")
        self.cache.write_text(json.dumps(record), encoding="utf-8")
        with patch.object(translations, "translate") as model:
            with self.assertRaisesRegex(ValueError, "heading hierarchy"):
                self.load(self.source, cache_only=True)
        model.assert_not_called()

    def test_cache_record_for_another_language_is_not_reused(self):
        original = self.cache.read_bytes()
        with patch.object(translations, "translate") as model:
            with self.assertRaisesRegex(
                ValueError, "Missing or stale French translation"
            ):
                translations.load_translation(
                    self.source,
                    self.path,
                    cache_root=self.root,
                    cache_only=True,
                    endpoint="unused",
                    model="test",
                    language="fr",
                )
        model.assert_not_called()
        self.assertEqual(self.cache.read_bytes(), original)

    def test_language_specific_cache_roots_remain_isolated(self):
        outputs = {
            "fr": FRONT + "# Example\n\nTexte original.\n",
            "de": FRONT + "# Beispiel\n\nUrsprünglicher Text.\n",
        }
        original_zh = self.cache.read_bytes()
        for language, translated in outputs.items():
            with self.subTest(language=language):
                root = self.root / language
                with patch.object(
                    translations, "translate", return_value=translated
                ) as model:
                    result = translations.load_translation(
                        self.source,
                        self.path,
                        cache_root=root,
                        cache_only=False,
                        endpoint="unused",
                        model="test",
                        language=language,
                    )
                model.assert_called_once_with(
                    self.source, endpoint="unused", model="test", language=language
                )
                record = json.loads(
                    (root / "docs/example.md.json").read_text(encoding="utf-8")
                )
                self.assertEqual(record["language"], language)
                self.assertEqual(record["translation"], result)
                with patch.object(translations, "translate") as cached_model:
                    self.assertEqual(
                        translations.load_translation(
                            self.source,
                            self.path,
                            cache_root=root,
                            cache_only=True,
                            endpoint="unused",
                            model="test",
                            language=language,
                        ),
                        translated,
                    )
                cached_model.assert_not_called()
        self.assertEqual(self.cache.read_bytes(), original_zh)


class RepositoryTests(unittest.TestCase):
    def sources(self):
        for path in sorted((REPO / "docs/content").rglob("*.md")):
            source = path.read_text(encoding="utf-8")
            front = translations.FRONT.match(source)
            if front and re.search(r"(?m)^draft:\s*true\s*$", front[1]):
                continue
            yield path, source

    def test_all_source_documents_round_trip(self):
        with patch.object(
            translations, "translate_batch", side_effect=lambda fields, **kwargs: fields
        ):
            for path, source in self.sources():
                with self.subTest(path=path.name):
                    translated = translations.translate(
                        source, endpoint="unused", model="unused"
                    )
                    translations.validate_translation(source, translated)
                    self.assertEqual(
                        translations.FRONT.sub("", source),
                        translations.FRONT.sub("", translated),
                    )

    def test_available_seed_caches_validate(self):
        root = REPO / ".cache/docs-translations/zh-cn"
        if not root.exists():
            self.skipTest(
                "Ignored translation seed caches are not present in this checkout"
            )
        for path, source in self.sources():
            relative = path.relative_to(REPO / "docs/content")
            cache = root / (relative.as_posix() + ".json")
            if cache.exists():
                with self.subTest(path=relative):
                    record = json.loads(cache.read_text(encoding="utf-8"))
                    translations.validate_translation(source, record["translation"])


if __name__ == "__main__":
    unittest.main()
