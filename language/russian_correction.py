import re
from functools import lru_cache
from typing import List

try:
    import pymorphy3
except Exception:
    pymorphy3 = None


LAT_TO_CYR = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м",
    "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
    "I": "І", "i": "і",
}

CYR_TO_LAT = {
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "к": "k", "м": "m",
    "о": "o", "р": "p", "т": "t", "х": "x", "у": "y",
}

CYR_RE = re.compile(r"[А-Яа-яЁё]")
LAT_RE = re.compile(r"[A-Za-z]")


@lru_cache(maxsize=1)
def _get_morph():
    if pymorphy3 is None:
        return None
    try:
        return pymorphy3.MorphAnalyzer()
    except Exception:
        return None


def _is_known_russian(word: str) -> bool:
    morph = _get_morph()
    if morph is None:
        return False
    try:
        return morph.word_is_known(word.lower())
    except Exception:
        return False


def _script_counts(word: str) -> tuple[int, int]:
    cyr = len(CYR_RE.findall(word))
    lat = len(LAT_RE.findall(word))
    return cyr, lat


def _translate_lookalikes(word: str, target: str) -> str:
    if target == "cyr":
        return "".join(LAT_TO_CYR.get(ch, ch) for ch in word)
    if target == "lat":
        return "".join(CYR_TO_LAT.get(ch, ch) for ch in word)
    return word


def _preserve_case(source: str, corrected: str) -> str:
    if source.isupper():
        return corrected.upper()
    if source.istitle():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def _split_punctuation(token: str) -> tuple[str, str, str]:
    m = re.match(r"^([^A-Za-zА-Яа-яЁё0-9]*)(.*?)([^A-Za-zА-Яа-яЁё0-9]*)$", token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def _correct_core_word(core: str) -> str:
    if not core:
        return core

    if len(core) <= 1 or core.isdigit():
        return core

    if "-" in core:
        parts = core.split("-")
        return "-".join(_correct_core_word(part) for part in parts)

    cyr_count, lat_count = _script_counts(core)

    candidates: List[str] = [core]

    if cyr_count >= lat_count and lat_count > 0:
        candidates.append(_translate_lookalikes(core, "cyr"))
    elif lat_count > cyr_count and cyr_count > 0:
        candidates.append(_translate_lookalikes(core, "lat"))
    else:
        candidates.append(_translate_lookalikes(core, "cyr"))

    candidates.extend([c.lower() for c in candidates if c])

    for cand in candidates:
        if cand and _is_known_russian(cand):
            return _preserve_case(core, cand)

    if cyr_count >= lat_count and lat_count > 0:
        chosen = candidates[1]
    elif lat_count > cyr_count and cyr_count > 0:
        chosen = candidates[1]
    else:
        chosen = core

    return _preserve_case(core, chosen)


def _correct_token(token: str) -> str:
    prefix, core, suffix = _split_punctuation(token)
    if not core:
        return token
    fixed = _correct_core_word(core)
    return prefix + fixed + suffix


def correct_russian_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    tokens = re.findall(r"\S+|\n", text, flags=re.UNICODE)
    corrected = []
    for tok in tokens:
        if tok == "\n":
            corrected.append(tok)
        else:
            corrected.append(_correct_token(tok))

    out = " ".join(part for part in corrected if part != "\n")
    out = out.replace(" \n ", "\n")
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,.;:!?])([^\s\n])", r"\1 \2", out)

    return out.strip()


correct_text = correct_russian_text