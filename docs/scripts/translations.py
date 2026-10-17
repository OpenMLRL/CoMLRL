"""Ignored translation cache and local Ollama translation for English Markdown."""

from collections import Counter
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
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
# A fence ends before its newline, so the next line keeps its own list marker.
FENCE = (
    r"(?m:^[ \t]*(?P<backtick_fence>`{3,})(?!`)[^`\n]*\n"
    r"[\s\S]*?^[ \t]*(?P=backtick_fence)`*[ \t]*$"
    r"|^[ \t]*(?P<tilde_fence>~{3,})(?!~)[^\n]*\n"
    r"[\s\S]*?^[ \t]*(?P=tilde_fence)~*[ \t]*$)"
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
# Block elements whose tags open and close one line stay outside the translated text.
BLOCK_TAG = r"<(?:h[1-6]|p|div|li|summary|figcaption)\b"
BLOCK_MARKER = re.compile(
    rf"[ \t]*(?:(?:#{{1,6}}|[-*+]|\d+\.|>)(?=[ \t])|{BLOCK_TAG})|"
)
# Bold markers and word-boundary underscores; an unpaired one renders literally.
EMPHASIS = (
    re.compile(r"\*\*"),
    re.compile(r"(?<![A-Za-z0-9_])_(?=\S)|(?<=\S)_(?![A-Za-z0-9_])"),
)
# Terms every language keeps in English: masked before translation, then validated.
# The project's full name (also with bold initials), RL jargon, product names, and
# words with two or more capitals (algorithm names, acronyms, class and dataset
# names). Inside compounds such as "generated-token", only the term is kept.
# Boundaries are ASCII letters and digits, so CJK particles ("actors를") and
# Markdown emphasis ("_Minecraft_") do not hide a term.
KEEP_ENGLISH = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:\*\*)?Co(?:\*\*)?operative (?:\*\*)?M(?:\*\*)?ulti-"
    r"(?:\*\*)?L(?:\*\*)?LM (?:\*\*)?R(?:\*\*)?einforcement (?:\*\*)?L(?:\*\*)?earning"
    r"|[Aa]ctor-[Cc]ritic|[Aa]ctors?|[Cc]ritics?|[Rr]ollouts?"
    r"|[Ee]pisodes?|[Ee]pochs?|[Tt]okens?|[Tt]okenizers?|[Tt]rainers?|Top-[kp]"
    r"|(?i:value[ -]heads?)|[Bb]ackbones?|[Mm]argins?"
    r"|Weights & Biases|Hugging Face|Minecraft|LLaMA-Factory|Dec-POMDP"
    r"|(?!ZXQ\d{5}QXZ)(?=[A-Za-z0-9]*[A-Z][A-Za-z0-9]*[A-Z])[A-Za-z][A-Za-z0-9]*)"
    r"(?![A-Za-z0-9])"
)
_PROTECTED = re.compile(
    "|".join(
        (
            FENCE,
            MATH,
            COMMENT,
            INLINE_CODE,
            r"\$\$[\s\S]*?\$\$|(?<!\$)\$[^$\n]+\$",
            SHORTCODE,
            r"(?<=\]\()<[^>\n]*>",
            r"(?P<html_tag></?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>]))",
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
    """Regex spans with scanners for link destinations and quoted HTML tags."""

    @staticmethod
    def _html_tag_end(text, position):
        """Skip quoted attributes and Hugo shortcodes before closing a tag."""
        quote = None
        while position < len(text):
            if text.startswith(("{{<", "{{%"), position):
                closing = ">}}" if text[position + 2] == "<" else "%}}"
                position += 3
                shortcode_quote = None
                while position < len(text):
                    char = text[position]
                    if shortcode_quote:
                        if char == "\\" and shortcode_quote != "`":
                            position += 2
                            continue
                        if char == shortcode_quote:
                            shortcode_quote = None
                    elif text.startswith(closing, position):
                        position += len(closing)
                        break
                    elif char in "\"'`":
                        shortcode_quote = char
                    position += 1
                else:
                    return None
                continue
            char = text[position]
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == ">":
                return position + 1
            position += 1
        return None

    def finditer(self, text):
        position = 0
        while match := _PROTECTED.search(text, position):
            end = match.end()
            if match.group("html_tag") is not None:
                end = self._html_tag_end(text, end)
                if end is None:
                    position = match.end()
                    continue
            elif match.group("destination") is not None:
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


def _protected_parts(text):
    """Protected content, excluding intentionally translatable display labels."""

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


def _link_structure(text):
    """Markdown link/image targets and reference IDs, independent of label prose."""
    protected = {}

    def hide(match):
        token = f"\x00{len(protected)}\x00"
        protected[token] = match[0]
        return token

    masked = PROTECTED.sub(hide, text)
    result = Counter()
    position = 0
    while position < len(masked):
        if masked[position] == "\\":
            position += 2
            continue
        if masked[position] != "[":
            position += 1
            continue
        start, end, depth = position, position + 1, 1
        while end < len(masked) and depth:
            if masked[end] == "\\":
                end += 2
                continue
            if masked[end] == "[":
                depth += 1
            elif masked[end] == "]":
                depth -= 1
            end += 1
        if depth or end >= len(masked) or masked[end] not in "([":
            position += 1
            continue
        opening = masked[end]
        closing = ")" if opening == "(" else "]"
        target_start, cursor, depth = end + 1, end + 1, 1
        quote = None
        while cursor < len(masked) and depth:
            char = masked[cursor]
            if char == "\\":
                cursor += 2
                continue
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'" and opening == "(":
                quote = char
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
            cursor += 1
        if depth:
            position += 1
            continue
        target = masked[target_start : cursor - 1]
        target = re.sub(r"\x00\d+\x00", lambda m: protected[m[0]], target)
        # Display text in shortcode destinations can be translated separately.
        identity = tuple(sorted(_protected_parts(target).items()))
        residual = PROTECTED.sub(lambda m: "", target).strip()
        result[
            (start > 0 and masked[start - 1] == "!", opening, identity, residual)
        ] += 1
        position = cursor
    return result


def _block_markers(markdown):
    """Indentation and list, quote, heading, or block-tag marker starting each line."""
    return [BLOCK_MARKER.match(line)[0] for line in markdown.splitlines()]


def _unpaired_emphasis(source, translated):
    """Whether a translated line has an odd marker count where English has pairs."""

    def parities(text):
        prose = PROTECTED.sub(lambda match: "x", text)
        return [
            [len(marker.findall(line)) % 2 for marker in EMPHASIS]
            for line in prose.splitlines()
        ]

    return any(
        dst > src
        for src_line, dst_line in zip(parities(source), parities(translated))
        for src, dst in zip(src_line, dst_line)
    )


def _same_structure(source, translated):
    def unmatched_html_prefixes(text):
        # An unterminated outer tag may otherwise be skipped while its nested
        # intact tag is counted, hiding a malformed duplicate such as <img src=".
        residual = PROTECTED.sub(lambda match: "", text)
        return Counter(re.findall(r"</?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>]|$)", residual))

    return (
        _protected_parts(source) == _protected_parts(translated)
        and _link_structure(source) == _link_structure(translated)
        and unmatched_html_prefixes(source) == unmatched_html_prefixes(translated)
    )


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
    if any(re.search(r"\{#[^}]*\}(?!$)", h[2]) for h in heading_lines(translated)):
        raise ValueError("Translation moved a heading anchor")
    if _block_markers(source) != _block_markers(translated):
        raise ValueError("Translation changed list, quote, or heading markers")
    if _unpaired_emphasis(source, translated):
        raise ValueError("Translation left a bold or italic marker unpaired")

    if not _same_structure(source[src_front.end() :], translated[dst_front.end() :]):
        raise ValueError("Translation changed protected code, math, links, or markup")
    # Count from each term's start: a model may attach a plural "s" or a particle,
    # but an identifier such as MAGRPOTrainer must not stand in for "Trainer".
    missing = [
        term
        for term, count in Counter(KEEP_ENGLISH.findall(source)).items()
        if len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(term)}", translated)) < count
    ]
    if missing:
        raise ValueError(
            "Translation changed terms that stay in English: "
            + ", ".join(sorted(missing))
        )


def language_info(language):
    try:
        return LANGUAGES[language]
    except KeyError as exc:
        raise ValueError(f"Unsupported translation language: {language}") from exc


def _translate_structured_fragments(source, *, endpoint, model, language):
    """Recover one damaged field without asking the model to recreate markup."""
    scaffold = re.compile(
        rf"({TOKEN.pattern}|\\.|&(?:\#[xX]?[0-9A-Fa-f]+|[A-Za-z]+);|[\[\]()!*_`<>|{{}}]+)"
    )
    parts = scaffold.split(source)
    fields, wrappers = {}, {}
    for index, part in enumerate(parts):
        if scaffold.fullmatch(part) or not re.search(r"[A-Za-z]", part):
            continue
        match = re.fullmatch(r"([^A-Za-z]*)(.*?[A-Za-z0-9])([^A-Za-z0-9]*)", part, re.S)
        if not match:
            raise ValueError("Cannot safely separate translation prose from markup")
        key = f"p{index}"
        fields[key] = match[2]
        wrappers[key] = (index, match[1], match[3])
    if not fields:
        return source
    translated = _translate_prose_fragments(
        fields,
        endpoint=endpoint,
        model=model,
        language=language,
    )
    for key, text in translated.items():
        index, prefix, suffix = wrappers[key]
        parts[index] = prefix + text + suffix
    result = "".join(parts)
    error = _field_error(source, result, language)
    if error:
        raise ValueError(f"Structure recovery returned {error}")
    return result


def translate(source, *, endpoint, model, language="zh-cn"):
    """Send prose only to a local model; restore protected spans ourselves."""
    language_info(language)
    fields = {}
    restorations = {}
    terms = set()

    def field(value):
        # NULs keep a term beside a placeholder, such as an <a> tag, its own word.
        value = TOKEN.sub(lambda m: f"\0{m[0]}\0", value)
        key = f"s{len(fields)}"
        fields[key] = KEEP_ENGLISH.sub(keep_english, value).replace("\0", "")
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

    def keep_english(match):
        token = protect(match)
        terms.add(token)
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
        # Keep a heading's explicit {#anchor} last; Hugo ignores one anywhere else.
        anchor = re.search(rf"\s+({TOKEN.pattern})$", text)
        if anchor and restorations[anchor[1]].startswith("{#"):
            text, newline = text[: anchor.start()], anchor[0] + newline
        # Keep a block element that wraps the line, such as <h1>…</h1>, around it.
        wrapped = re.fullmatch(rf"({TOKEN.pattern})(.*)({TOKEN.pattern})", text)
        if (
            wrapped
            and re.match(BLOCK_TAG, restorations[wrapped[1]])
            and restorations[wrapped[3]].startswith("</")
        ):
            prefix, text = prefix + wrapped[1], wrapped[2]
            newline = wrapped[3] + newline
        if re.search(r"[A-Za-z]", TOKEN.sub("", text)):
            lines.append(prefix + "@FIELD:" + field(text) + "@" + newline)
        else:
            lines.append(line)
    translated = {}
    batch = {}
    size = 0
    for key, value in fields.items():
        if batch and (size + len(value) > 1800 or len(batch) >= 12):
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

    def expand(text):
        return TOKEN.sub(lambda m: restorations[m[0]], text)

    for key, value in fields.items():
        if not _same_structure(expand(value), expand(translated[key])):
            translated[key] = _translate_structured_fragments(
                value, endpoint=endpoint, model=model, language=language
            )
            if not _same_structure(expand(value), expand(translated[key])):
                raise ValueError(
                    f"Fragment translation changed protected structure in {key}"
                )
        # A model may drop one marker of a pair; losing the emphasis reads better.
        for marker in EMPHASIS:
            if (
                len(marker.findall(translated[key])) % 2
                > len(marker.findall(value)) % 2
            ):
                translated[key] = marker.sub("", translated[key])
    # Titles and shortcode labels are inserted after restoration; restore terms first.
    for key, value in translated.items():
        translated[key] = TOKEN.sub(
            lambda m: restorations[m[0]] if m[0] in terms else m[0], value
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


class TranslationServiceError(RuntimeError):
    """A temporary connection or server failure eligible for bounded recovery."""


class RepetitionError(ValueError):
    """Ollama aborted a generation that kept repeating; retrying it repeats too."""


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
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", "")
        except (OSError, ValueError, AttributeError):
            detail = ""
        # Greedy decoding can loop on the same input every time (HTTP 500).
        if "repeat limit" in detail:
            raise RepetitionError(f"Translator aborted a repeating output: {detail}")
        error = (
            TranslationServiceError
            if exc.code == 429 or 500 <= exc.code < 600
            else RuntimeError
        )
        raise error(
            f"Local Ollama translation failed ({model} at {endpoint}): "
            f"HTTP {exc.code} {detail}".rstrip()
        ) from exc
    except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
        raise TranslationServiceError(
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
    except TranslationServiceError as exc:
        if _retry:
            raise TranslationServiceError(
                f"{exc}; failed fields: {', '.join(fields)}"
            ) from exc
        # One recovery layer only: smaller requests may finish within the CPU
        # timeout, while a persistent outage must not trigger an expanding tree.
        items = list(fields.items())
        middle = (len(items) + 1) // 2
        recovered = {}
        for part in (items[:middle], items[middle:]):
            if part:
                recovered.update(
                    translate_batch(
                        dict(part),
                        endpoint=endpoint,
                        model=model,
                        language=language,
                        _retry=True,
                    )
                )
        return recovered
    except RepetitionError:
        # Retry fields alone; one that still loops is translated between its
        # placeholders, which keeps them out of generation.
        if len(fields) > 1:
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
        return {
            key: _translate_fragments(
                value, endpoint=endpoint, model=model, language=language
            )
            for key, value in fields.items()
        }
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


def _translate_prose_fragments(fields, *, endpoint, model, language):
    """Recover a damaged fragment batch with one independent call per fragment."""
    if not fields:
        return {}

    def validate(expected, result):
        if not isinstance(result, dict) or result.keys() != expected.keys():
            raise ValueError("Translator returned invalid fragment fields")
        for key, source in expected.items():
            error = _field_error(source, result[key], language)
            if error:
                raise ValueError(f"Translator returned {error} in fragment {key}")
            if not _same_structure(source, result[key]):
                raise ValueError(
                    f"Translator damaged plain text in structure recovery: {key}"
                )
        return result

    try:
        return validate(
            fields,
            _request_batch(fields, endpoint=endpoint, model=model, language=language),
        )
    except (ValueError, TranslationServiceError):
        # An empty fragment can indicate that the model merged neighboring
        # meanings and shifted their field assignments. Discard the whole batch
        # instead of retaining apparently valid but misaligned translations.
        translated = {}
        for key, source in fields.items():
            fragment = {key: source}
            translated.update(
                validate(
                    fragment,
                    _request_batch(
                        fragment, endpoint=endpoint, model=model, language=language
                    ),
                )
            )
        return translated


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
    translated = _translate_prose_fragments(
        fields, endpoint=endpoint, model=model, language=language
    )
    for key, text in translated.items():
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
        try:
            record = json.loads(cache.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("Cache record is not a JSON object")
            if (
                record.get("source_sha256") == digest(source)
                and record.get("language") == language
            ):
                translated = record.get("translation")
                if not isinstance(translated, str):
                    raise ValueError("Cache record has no translation text")
                validate_translation(source, translated)
                return translated
        except ValueError as exc:
            if cache_only:
                raise ValueError(
                    f"Invalid {info['target']} translation cache for {path}: {exc}"
                ) from exc
            print(
                f"Regenerating invalid {info['target']} translation cache for {path}: {exc}",
                flush=True,
            )
    if cache_only:
        raise ValueError(
            f"Missing or stale {info['target']} translation: {path}. Run without --cache-only with Ollama running."
        )
    print(f"Translating {path} into {info['target']} with {model}…", flush=True)
    try:
        result = translate(source, endpoint=endpoint, model=model, language=language)
    except (ValueError, RuntimeError) as exc:
        raise type(exc)(
            f"{info['target']} translation failed for {path}: {exc}"
        ) from exc
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
