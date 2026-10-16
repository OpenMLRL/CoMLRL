"""Ignored translation cache and local Ollama translation for English Markdown."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import Request, urlopen


LANGUAGES = json.loads(
    (Path(__file__).resolve().parents[1] / "data/languages.json").read_text(
        encoding="utf-8"
    )
)
LANGUAGE_SCRIPTS = {
    "zh-cn": r"[\u3400-\u4dbf\u4e00-\u9fff]",
    "ja": r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9d]",
    "ko": r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]",
}
FRONT = re.compile(r"\A---\n(.*?)\n---(?:\n|$)", re.S)
DISPLAY = r"title|linkTitle|description"
FENCE = (
    r"(?m:^[ \t]*(?P<backtick_fence>`{3,})(?!`)[^`\n]*\n"
    r"[\s\S]*?^[ \t]*(?P=backtick_fence)`*[ \t]*(?:\n|$)"
    r"|^[ \t]*(?P<tilde_fence>~{3,})(?!~)[^\n]*\n"
    r"[\s\S]*?^[ \t]*(?P=tilde_fence)~*[ \t]*(?:\n|$))"
)
MATH = r"\{\{[<%]\s*katex\b[\s\S]*?\{\{[<%]\s*/katex\s*[>%]\}\}"
SHORTCODE = r"\{\{[<%][\s\S]*?[>%]\}\}"
COMMENT = r"<!--[\s\S]*?-->"
INLINE_CODE = r"(?<!`)(?P<inline_ticks>`+)(?!`)[\s\S]*?(?<!`)(?P=inline_ticks)(?!`)"
# Model repository IDs contain a numeric model/version component. Underscores
# around an emphasized ID are not part of the identifier.
MODEL_ID = (
    r"(?<![A-Za-z0-9/])[A-Za-z][A-Za-z0-9_.-]*/"
    r"(?=[A-Za-z0-9_.-]*\d)[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?"
)
COUNT = r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?x(?![A-Za-z0-9])"
_PROTECTED = re.compile(
    "|".join(
        (
            FENCE,
            MATH,
            COMMENT,
            INLINE_CODE,
            r"\$\$[\s\S]*?\$\$|(?<!\$)\$[^$\n]+\$",
            SHORTCODE,
            r"(?<=\]\()<[^>\n]*>|</?[A-Za-z][^>]*>",
            # Reference definitions include the ID and optional link title.
            r"(?m:^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*[^\n]+)",
            r"(?<=\]\[)[^\]\n]+(?=\])",
            r"(?P<destination>https?://|(?<=\]\()[^\s)<>\"])",
            r"\{#[^}]+\}",
            MODEL_ID,
            COUNT,
        )
    )
)


class _ProtectedMatch:
    """The small Match interface used by masking and validation."""

    def __init__(self, text, start, end):
        self.text, self._start, self._end = text, start, end

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.text[self._start : self._end]

    def start(self):
        return self._start

    def end(self):
        return self._end


class _ProtectedSpans:
    """Regex spans, with a balanced scanner for Markdown link destinations."""

    def finditer(self, text):
        position = 0
        while match := _PROTECTED.search(text, position):
            end = match.end()
            if match.group("destination") is not None:
                end, depth = match.start(), 0
                while end < len(text):
                    char = text[end]
                    if char == "\\" and end + 1 < len(text):
                        end += 2
                        continue
                    if char.isspace() or char in '<>"':
                        break
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        if not depth:
                            break
                        depth -= 1
                    end += 1
            yield _ProtectedMatch(text, match.start(), end)
            position = end

    def sub(self, replacement, text):
        parts, position = [], 0
        for match in self.finditer(text):
            parts.extend((text[position : match.start()], replacement(match)))
            position = match.end()
        return "".join(parts) + text[position:]


PROTECTED = _ProtectedSpans()
TOKEN = re.compile(r"ZXQ\d{5}QXZ")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def heading_lines(markdown):
    """ATX headings outside fenced code, with their source line numbers."""
    front = FRONT.match(markdown)
    start = front.end() if front else 0
    protected = [
        (m.start(), m.end())
        for m in re.finditer("|".join((FENCE, MATH, COMMENT, INLINE_CODE)), markdown)
    ]
    result = []
    offset = 0
    for index, line in enumerate(markdown.splitlines(keepends=True)):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match and offset >= start and not any(a <= offset < b for a, b in protected):
            result.append((index, len(match[1]), match[2]))
        offset += len(line)
    return result


def validate_translation(source, translated):
    """Reject damaged code, math, links, shortcodes, or document structure."""
    if "\ufffd" in translated or TOKEN.search(translated):
        raise ValueError("Invalid Unicode or unrestored translation placeholder")
    src_front, dst_front = FRONT.match(source), FRONT.match(translated)
    if not src_front or not dst_front:
        raise ValueError("Missing YAML frontmatter")
    strip_titles = lambda text: re.sub(rf"(?m)^(?:{DISPLAY}):.*\n?", "", text)
    if strip_titles(src_front[1]) != strip_titles(dst_front[1]):
        raise ValueError("Translation changed frontmatter configuration")
    if [h[1] for h in heading_lines(source)] != [
        h[1] for h in heading_lines(translated)
    ]:
        raise ValueError("Translation changed the heading hierarchy")

    def protected_parts(text):
        # Display-only shortcode arguments and HTML alternative text may differ.
        def neutralize(match):
            value = match[0]
            if value.startswith("{{") and not re.match(r"\{\{[<%]\s*katex\b", value):
                value = re.sub(
                    r'\b(title|label|description|caption)="[^"]*"', r'\1=""', value
                )
                value = re.sub(r'(\btab\s+)"[^"]*"', r'\1""', value)
            if value.startswith("<") and not value.startswith("<!--"):
                value = re.sub(r'\b(alt|title|aria-label)="[^"]*"', r'\1=""', value)
            return value

        return Counter(neutralize(m) for m in PROTECTED.finditer(text))

    if protected_parts(source[src_front.end() :]) != protected_parts(
        translated[dst_front.end() :]
    ):
        raise ValueError("Translation changed protected code, math, links, or markup")


def language_info(language):
    try:
        return LANGUAGES[language]
    except KeyError as exc:
        raise ValueError(f"Unsupported translation language: {language}") from exc


def translate(source, *, endpoint, model, language="zh-cn"):
    """Send prose only to a local model; restore protected spans ourselves."""
    language_info(language)
    fields = {}
    restorations = {}

    def field(value):
        key = f"s{len(fields)}"
        fields[key] = value
        return key

    def protect(match):
        value = match[0]
        if value.startswith("{{") and not re.match(r"\{\{[<%]\s*katex\b", value):
            value = re.sub(
                r'(\b(?:title|label|description|caption)=)"([^"]*)"',
                lambda m: m[1] + '"@FIELD:' + field(m[2]) + '@"',
                value,
            )
            value = re.sub(
                r'(\btab\s+)"([^"]*)"',
                lambda m: m[1] + '"@FIELD:' + field(m[2]) + '@"',
                value,
            )
        token = f"ZXQ{len(restorations):05d}QXZ"
        restorations[token] = value
        return token

    front = FRONT.match(source)
    if not front:
        raise ValueError("Expected YAML frontmatter")
    front_text = re.sub(
        rf"(?m)^({DISPLAY}):\s*(.*?)$",
        lambda m: m[1] + ": @YAML:" + field(m[2].strip("\"'")) + "@",
        front[0],
    )
    masked = PROTECTED.sub(protect, source[front.end() :])
    lines = []
    for line in masked.splitlines(keepends=True):
        match = re.match(r"^(\s*(?:(?:#{1,6}|[-*+]|\d+\.|>)\s+)?)(.*?)(\n?)$", line)
        prefix, text, newline = match.groups()
        if re.search(r"[A-Za-z]", TOKEN.sub("", text)):
            lines.append(prefix + "@FIELD:" + field(text) + "@" + newline)
        else:
            lines.append(line)
    translated = {}
    batch = {}
    size = 0
    for key, value in fields.items():
        if batch and size + len(value) > 4500:
            translated.update(
                translate_batch(
                    batch, endpoint=endpoint, model=model, language=language
                )
            )
            batch, size = {}, 0
        batch[key] = value
        size += len(value)
    if batch:
        translated.update(
            translate_batch(batch, endpoint=endpoint, model=model, language=language)
        )

    def substitute(text):
        return re.sub(
            r"@FIELD:(s\d+)@",
            lambda m: (
                translated[m[1]].replace('"', "&quot;")
                if text.startswith("{{")
                else translated[m[1]]
            ),
            text,
        )

    result = substitute("".join(lines))
    result = TOKEN.sub(lambda m: substitute(restorations[m[0]]), result)
    front_text = re.sub(
        r"@YAML:(s\d+)@",
        lambda m: json.dumps(translated[m[1]], ensure_ascii=False),
        front_text,
    )
    result = front_text + result
    validate_translation(source, result)
    return result


def _request_batch(fields, *, endpoint, model, language):
    info = language_info(language)
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in fields},
        "required": list(fields),
        "additionalProperties": False,
    }
    prompt = (
        info["instruction"] + "\n"
        f"Target language: {info['target']} ({info['name']}). Translate English documentation into this language only. "
        "Preserve technical meaning and Markdown punctuation. Do not add explanations. "
        "Keep CoMLRL, model identifiers, API names, algorithm names, actor and critic in English. "
        "Use this terminology: "
        + json.dumps(info["glossary"], ensure_ascii=False)
        + ". "
        "Every ZXQ00000QXZ-style placeholder must remain unchanged in its original field, with the same occurrence count. "
        "Never remove, add, or renumber placeholders. Return one string for every input key, without newlines. "
        'Example input: {"s0":"Users can use ZXQ00000QXZ with ZXQ00001QXZ."} '
        "Example output: "
        + json.dumps({"s0": info["example"]}, ensure_ascii=False)
        + ". "
        f"Return only a JSON object containing {info['target']} translations."
    )
    request = Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(
            {
                "model": model,
                "stream": False,
                "think": False,
                "format": schema,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(fields, ensure_ascii=False)},
                ],
                "options": {"temperature": 0, "num_ctx": 8192},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            result = json.loads(json.load(response)["message"]["content"])
    except URLError as exc:
        raise RuntimeError(
            f"Local Ollama translation failed ({model} at {endpoint}): {exc}"
        ) from exc
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Translator returned malformed JSON") from exc
    return result


def _field_error(source, translated, language):
    if (
        not isinstance(translated, str)
        or not translated.strip()
        or "\n" in translated
        or "\r" in translated
    ):
        return "invalid text"
    if "\ufffd" in translated or Counter(TOKEN.findall(translated)) != Counter(
        TOKEN.findall(source)
    ):
        return "changed protected placeholders"
    prose = TOKEN.sub("", source)
    # Short labels may legitimately consist only of names such as MARLHF.
    if len(re.findall(r"[A-Za-z]+", prose)) >= 8:
        script = LANGUAGE_SCRIPTS.get(language)
        if script and not re.search(script, translated):
            return f"no {language_info(language)['target']} text"
        if language != "en" and not script and translated.strip() == source.strip():
            return "an untranslated English echo"
    return None


def translate_batch(fields, *, endpoint, model, language="zh-cn", _retry=False):
    """Translate a batch, retrying each damaged or missing field at most once."""
    language_info(language)
    if not fields:
        return {}
    try:
        result = _request_batch(
            fields, endpoint=endpoint, model=model, language=language
        )
        if not isinstance(result, dict) or result.keys() - fields.keys():
            raise ValueError(
                "Translator returned unexpected fields or a non-object response"
            )
    except ValueError:
        if _retry:
            raise
        # A malformed batch cannot be reused. Smaller, independent requests
        # reduce competing context without allowing an unbounded retry loop.
        return {
            key: translate_batch(
                {key: value},
                endpoint=endpoint,
                model=model,
                language=language,
                _retry=True,
            )[key]
            for key, value in fields.items()
        }
    for key, source in fields.items():
        error = _field_error(source, result.get(key), language)
        if error:
            if _retry:
                if error == "changed protected placeholders":
                    result[key] = _translate_fragments(
                        source, endpoint=endpoint, model=model, language=language
                    )
                    continue
                raise ValueError(f"Translator returned {error} in {key}")
            result[key] = translate_batch(
                {key: source},
                endpoint=endpoint,
                model=model,
                language=language,
                _retry=True,
            )[key]
    return result


def _translate_fragments(source, *, endpoint, model, language):
    """Keep stubborn code/formula placeholders entirely outside generation."""
    parts = re.split(f"({TOKEN.pattern})", source)
    fields, wrappers = {}, {}
    for index, part in enumerate(parts):
        if TOKEN.fullmatch(part) or not re.search(r"[A-Za-z]", part):
            continue
        # Preserve surrounding Markdown, punctuation, and spacing verbatim.
        match = re.fullmatch(r"([^A-Za-z]*)(.*?[A-Za-z0-9])([^A-Za-z0-9]*)", part, re.S)
        if not match:
            raise ValueError("Cannot safely split translation around protected content")
        key = f"p{index}"
        fields[key] = match[2]
        wrappers[key] = (index, match[1], match[3])
    translated = _request_batch(
        fields, endpoint=endpoint, model=model, language=language
    )
    if not isinstance(translated, dict) or translated.keys() != fields.keys():
        raise ValueError("Translator returned invalid fragment fields")
    for key, text in translated.items():
        error = _field_error(fields[key], text, language)
        if error:
            raise ValueError(f"Translator returned {error} in fragment {key}")
        index, prefix, suffix = wrappers[key]
        parts[index] = prefix + text + suffix
    result = "".join(parts)
    if _field_error(source, result, language):
        raise ValueError("Fragment translation changed protected content")
    return result


def load_translation(
    source, path, *, cache_root, cache_only, endpoint, model, language="zh-cn"
):
    """Read or generate one cache entry in the caller's language-specific root."""
    info = language_info(language)
    cache = cache_root / (path.as_posix() + ".json")
    if cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        if (
            record.get("source_sha256") == digest(source)
            and record.get("language") == language
        ):
            validate_translation(source, record["translation"])
            return record["translation"]
    if cache_only:
        raise ValueError(
            f"Missing or stale {info['target']} translation: {path}. Run without --cache-only with Ollama running."
        )
    print(f"Translating {path} into {info['target']} with {model}…", flush=True)
    result = translate(source, endpoint=endpoint, model=model, language=language)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "language": language,
                "source_sha256": digest(source),
                "translator": f"ollama:{model}",
                "translation": result,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache)
    return result
