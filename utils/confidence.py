import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


CYR_RE = re.compile(r"[А-Яа-яЁё]")
LAT_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class OCRCandidate:
    text: str
    score: float
    engine: str = ""
    source: str = ""


def _count_scripts(text: str) -> Tuple[int, int, int]:
    cyr = len(CYR_RE.findall(text))
    lat = len(LAT_RE.findall(text))
    dig = len(DIGIT_RE.findall(text))
    return cyr, lat, dig


def text_quality_score(text: str, base_score: float = 0.0) -> float:
    if not text or not text.strip():
        return 0.0

    stripped = text.strip()
    words = WORD_RE.findall(stripped)
    word_count = len(words)
    char_count = len(stripped)
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    line_count = len(lines)

    cyr, lat, dig = _count_scripts(stripped)

    structure = 0.0
    structure += min(char_count / 4000.0, 0.35)
    structure += min(word_count / 150.0, 0.25)
    structure += min(line_count / 40.0, 0.10)

    script_score = 0.0
    total_alpha = cyr + lat
    if total_alpha > 0:
        cyr_ratio = cyr / total_alpha
        lat_ratio = lat / total_alpha

        if cyr_ratio >= 0.60:
            script_score += 0.20
        elif cyr_ratio >= 0.40:
            script_score += 0.10
        else:
            script_score -= 0.05

        if lat_ratio > 0.60 and cyr_ratio > 0.15:
            script_score -= 0.05

    letters = cyr + lat
    if letters > 0:
        symbol_ratio = max(0, char_count - letters - dig) / max(char_count, 1)
        if symbol_ratio > 0.35:
            script_score -= 0.10

    punctuation_bonus = 0.0
    if any(p in stripped for p in [".", ",", ":", ";"]):
        punctuation_bonus += 0.03
    if "\n" in stripped:
        punctuation_bonus += 0.03

    final = base_score + structure + script_score + punctuation_bonus
    return max(0.0, min(final, 1.5))


def line_quality_map(text: str) -> List[Tuple[str, float]]:
    if not text:
        return []

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        score = text_quality_score(line)
        lines.append((line, score))
    return lines


def choose_best_result(candidates: List[OCRCandidate]) -> OCRCandidate:
    if not candidates:
        return OCRCandidate(text="", score=0.0, engine="None", source="")

    best = candidates[0]
    best_total = text_quality_score(best.text, best.score)

    for cand in candidates[1:]:
        total = text_quality_score(cand.text, cand.score)
        if total > best_total:
            best = cand
            best_total = total

    return best


def is_low_confidence(text: str, score: float, threshold: float = 0.45) -> bool:
    quality = text_quality_score(text, score)
    return quality < threshold


def format_review_report(text: str, score: float, engine: str = "") -> Dict[str, object]:
    lines = line_quality_map(text)
    weak_lines = [ln for ln, sc in lines if sc < 0.35]

    return {
        "engine": engine,
        "score": round(score, 3),
        "quality": round(text_quality_score(text, score), 3),
        "line_count": len(lines),
        "weak_line_count": len(weak_lines),
        "weak_lines": weak_lines,
    }