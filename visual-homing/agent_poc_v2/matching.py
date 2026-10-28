"""
How the path finder decides whether two descriptions are about the same thing.

This is deliberately simple: it compares the important words in two descriptions.
It is the spot where a smarter method plugs in later, for example comparing
BLIP-2 embeddings, or asking a small language model "are these the same object?".
"""
import re

# Small words that say little about which object it is
STOP_WORDS = {"a", "an", "the", "on", "in", "by", "of", "to", "with", "and", "is",
              "near", "next", "beside", "under", "at", "there", "this", "that", "made"}


def key_words(description: str) -> set[str]:
    words = re.findall(r"[a-z]+", description.lower())
    return {w for w in words if w not in STOP_WORDS}


def similarity(desc_a: str, desc_b: str) -> float:
    """0 = nothing in common, 1 = the same key words."""
    a, b = key_words(desc_a), key_words(desc_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def object_similarity(obj_a: dict, obj_b: dict) -> float:
    """Compare two reported objects.

    Real robots report fields (color, material, what is next to it): the score is the share
    of fields both robots answered that agree. Objects without fields (the made-up scene)
    fall back to comparing their descriptions word by word.
    """
    fields_a, fields_b = obj_a.get("attributes"), obj_b.get("attributes")
    if fields_a and fields_b:
        compared = agreed = 0
        for name in fields_a:
            if fields_a.get(name) and fields_b.get(name):
                compared += 1
                agreed += bool(key_words(fields_a[name]) & key_words(fields_b[name]))
        if compared:
            return agreed / compared
    return similarity(obj_a["description"], obj_b["description"])