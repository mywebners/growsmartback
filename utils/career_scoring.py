"""
Hybrid career scoring: align P1–P8 with training (sorted by marks),
then re-rank ML probabilities using stream + skill-domain heuristics.
"""
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def convert_marks_to_category(mark: float) -> int:
    if mark >= 80:
        return 2
    if mark >= 60:
        return 1
    return 0


def marks_to_pslots_sorted(
    matric_marks: Optional[Mapping[str, Any]],
    inter_marks: Optional[Mapping[str, Any]],
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Match train_model / preprocess: P1 = strongest performance, P8 = weakest among 8 slots.
    Concatenate matric + intermediate numeric marks, sort descending, map to 0/1/2.
    """
    vals: List[float] = []
    for d in (matric_marks or {}), (inter_marks or {}):
        for v in d.values():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
    vals.sort(reverse=True)
    while len(vals) < 8:
        vals.append(0.0)
    vals = vals[:8]
    return tuple(convert_marks_to_category(m) for m in vals)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# Keyword groups (substring match on career label)
_TECH = (
    "software",
    "developer",
    "programmer",
    "computer",
    "data scien",
    "data analyst",
    "it ",
    " i.t",
    "network",
    "cyber",
    "web ",
    "database",
    "system analyst",
    "technolog",
    "devops",
    "machine learning",
    "ai ",
    " artificial",
    "cloud",
    "blockchain",
)
_ENGINEERING = (
    "engineer",
    "engineering",
    "civil",
    "mechanical",
    "electrical",
    "aerospace",
    "chemical engineer",
    "architect",
)
_MEDICAL = (
    "doctor",
    "physician",
    "medical",
    "surgeon",
    "nurse",
    "dentist",
    "pharma",
    "radiolog",
    "patholog",
    "cardio",
    "dermatolog",
    "psychiatr",
    "gynecolog",
    "pediatr",
    "ortho",
    "mbbs",
)
_PSY_SOCIAL = (
    "psycholog",
    "psychiatr",
    "counselor",
    "therapist",
    "social work",
    "sociolog",
    "human resource",
    "hr ",
)
_LEGAL_BUSINESS = (
    "law",
    "lawyer",
    "legal",
    "business",
    "marketing",
    "account",
    "finance",
    "commerce",
    "economist",
    "bank",
)
_EDU_LANG = (
    "teacher",
    "lecturer",
    "professor",
    "education",
    "linguist",
    "journalist",
    "writer",
    "translator",
)


def _has_any(c: str, needles: Sequence[str]) -> bool:
    return any(n in c for n in needles)


def stream_domain_multiplier(
    career_name: str,
    matric_stream: Optional[str],
    intermediate_stream: Optional[str],
) -> float:
    """
    Penalize careers that clash with the student's academic track (e.g. ICS → heavy tech expectation).
    Intermediate stream is weighted higher than matric.
    """
    c = _norm(career_name)
    if not c:
        return 1.0

    tech = _has_any(c, _TECH)
    eng = _has_any(c, _ENGINEERING)
    med = _has_any(c, _MEDICAL)
    psy = _has_any(c, _PSY_SOCIAL)
    law_bus = _has_any(c, _LEGAL_BUSINESS)
    edu = _has_any(c, _EDU_LANG)

    def apply_inter(stream: Optional[str]) -> float:
        if not stream:
            return 1.0
        s = stream.strip().lower()
        mult = 1.0

        if s == "ics":
            if tech:
                mult *= 1.55
            if eng and not tech:
                mult *= 1.15
            if med:
                mult *= 0.35
            if psy and not tech and not edu:
                mult *= 0.3
            if law_bus and not tech:
                mult *= 0.55

        elif s == "pre-eng":
            if eng or tech:
                mult *= 1.5
            if med or psy:
                mult *= 0.45

        elif s == "pre-med":
            if med or _has_any(c, ("biotech", "microbio", "bioinform", "research scientist")):
                mult *= 1.55
            if tech and not med:
                mult *= 0.65
            if psy and "psychiatr" not in c and "psycholog" in c:
                mult *= 0.75

        elif s == "arts":
            if psy or law_bus or edu:
                mult *= 1.45
            if tech and not edu:
                mult *= 0.5
            if med:
                mult *= 0.4

        return mult

    def apply_matric(stream: Optional[str]) -> float:
        if not stream:
            return 1.0
        s = stream.strip().lower()
        mult = 1.0
        if s == "cs":
            if tech or eng:
                mult *= 1.2
            if med and not tech:
                mult *= 0.75
        elif s == "bio":
            if med:
                mult *= 1.2
            if tech and not med:
                mult *= 0.85
        elif s == "arts":
            if psy or law_bus or edu:
                mult *= 1.15
            if tech:
                mult *= 0.8
        return mult

    m = apply_inter(intermediate_stream) * apply_matric(matric_stream)
    return max(0.15, min(m, 2.8))


def skill_alignment_multiplier(
    career_name: str,
    logical: int,
    spatial: int,
    linguistic: int,
    interpersonal: int,
    intrapersonal: int,
    musical: int,
    bodily: int,
    naturalist: int,
) -> float:
    """
    Light nudge so dominant intelligences favour plausible career families.
    """
    c = _norm(career_name)
    if not c:
        return 1.0

    mult = 1.0
    L, S = logical, spatial
    Li, Ie = linguistic, interpersonal
    Ip = intrapersonal

    if _has_any(c, _TECH + ("analyst", "program")):
        top = (L + S) / 2
        if top >= 2:
            mult *= 1.0 + 0.08 * (top - 1)
        else:
            mult *= 0.88

    if _has_any(c, _PSY_SOCIAL) and "psychiatr" not in c:
        if (Ie + Ip + Li) / 3 >= 2:
            mult *= 1.0 + 0.06 * ((Ie + Ip + Li) / 3 - 1)
        else:
            mult *= 0.9

    if _has_any(c, _MEDICAL):
        if (L + naturalist + interpersonal) / 3 >= 2:
            mult *= 1.05

    if _has_any(c, _EDU_LANG) and Li >= 2:
        mult *= 1.06

    if musical >= 3 and _has_any(c, ("music", "composer", "audio", "sound")):
        mult *= 1.12

    if bodily >= 3 and _has_any(c, ("sport", "fitness", "physio", "athlete")):
        mult *= 1.1

    return max(0.5, min(mult, 1.35))


def blend_career_probabilities(
    probs: "Any",
    class_indices: Sequence[int],
    career_labels: Sequence[str],
    matric_stream: Optional[str],
    intermediate_stream: Optional[str],
    skills: Mapping[str, int],
) -> List[Tuple[int, float]]:
    """
    Returns list of (class_index, adjusted_score) sorted by adjusted_score descending.
    """
    logical = int(skills.get("Logical", skills.get("Logical - Mathematical", 2)))
    spatial = int(skills.get("Spatial", skills.get("Spatial-Visualization", 2)))
    linguistic = int(skills.get("Linguistic", 2))
    interpersonal = int(skills.get("Interpersonal", 2))
    intrapersonal = int(skills.get("Intrapersonal", 2))
    musical = int(skills.get("Musical", 2))
    bodily = int(skills.get("Bodily", 2))
    naturalist = int(skills.get("Naturalist", 2))

    scored: List[Tuple[int, float]] = []
    for idx, label in zip(class_indices, career_labels):
        base = float(probs[idx])
        sm = stream_domain_multiplier(label, matric_stream, intermediate_stream)
        km = skill_alignment_multiplier(
            label, logical, spatial, linguistic, interpersonal,
            intrapersonal, musical, bodily, naturalist,
        )
        scored.append((idx, base * sm * km))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
