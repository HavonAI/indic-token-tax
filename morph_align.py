"""Morpheme-boundary alignment of tokenizers on Hindi and Marathi.

Gold segmentations: MorphTok (Brahma et al., arXiv:2504.10335), HF dataset karthika95/morphTok at a
pinned revision, CC-BY-4.0. Each entry is a word and its '+'-delimited morpheme segmentation.

Only entries whose segments concatenate exactly to the word (NFC) are used, so that every gold
boundary is a character offset in the surface form; sandhi entries with character changes are
dropped. Words listed more than once with different segmentations are dropped as ambiguous.

Each word is encoded alone with a leading space (as it would appear mid-sentence), with no special
tokens. A predicted boundary is a character offset strictly inside the word where one token ends and
the next begins; splits inside a single character (byte fallback) are not boundaries. We report
micro-averaged boundary precision, recall and F1 over all words, the share of words whose token
boundaries match the gold boundaries exactly, and mean tokens per word. A character-level
reference row (a boundary after every character) calibrates precision.

Usage: python morph_align.py [--tokenizer NAME=PATH/tokenizer.json ...]
Outputs: results/morph_alignment.csv, results/morph_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

from measure import DATA, HF_TOKENIZERS, RESULTS, TIKTOKEN, sha256

MORPHTOK = ("karthika95/morphTok", "25d11d41b0428bbe4c9b2a7d4f9de92351709364")
FILES = {"hin": "hindi.csv", "mar": "marathi.csv"}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


def load_gold() -> tuple[dict, dict]:
    from huggingface_hub import hf_hub_download

    gold, meta = {}, {}
    for lang, fname in FILES.items():
        path = hf_hub_download(MORPHTOK[0], fname, repo_type="dataset", revision=MORPHTOK[1],
                               cache_dir=str(DATA / "hf"))
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        seen: dict[str, tuple] = {}
        conflict: set[str] = set()
        lossy = 0
        for r in rows:
            word = nfc(r["word"])
            segs = [nfc(x) for x in r["segmentation"].split("+") if nfc(x)]
            if not word or " " in word or "".join(segs) != word:
                lossy += 1
                continue
            cuts, pos = [], 0
            for s in segs[:-1]:
                pos += len(s)
                cuts.append(pos)
            key = tuple(cuts)
            if word in seen and seen[word] != key:
                conflict.add(word)
            seen[word] = key
        words = {w: set(c) for w, c in seen.items() if w not in conflict}
        gold[lang] = words
        meta[lang] = {"file": f"hf://datasets/{MORPHTOK[0]}@{MORPHTOK[1]}/{fname}",
                      "sha256": sha256(Path(path)), "rows": len(rows),
                      "dropped_non_concatenative": lossy, "dropped_conflicting": len(conflict),
                      "words_used": len(words),
                      "words_with_boundary": sum(1 for c in words.values() if c),
                      "gold_boundaries": sum(len(c) for c in words.values())}
    return gold, meta


def tiktoken_cuts(enc):
    def cuts(word: str) -> tuple[set, int]:
        text = " " + word
        ids = enc.encode_ordinary(text)
        byte_cuts, b = set(), 0
        for i in ids[:-1]:
            b += len(enc.decode_single_token_bytes(i))
            byte_cuts.add(b)
        out, b = set(), 0
        for p, ch in enumerate(text):  # character offset p (in text) -> byte offset b
            if 1 < p and b in byte_cuts:
                out.add(p - 1)
            b += len(ch.encode("utf-8"))
        return out, len(ids)
    return cuts


def hf_cuts(tk):
    def cuts(word: str) -> tuple[set, int]:
        e = tk.encode(" " + word, add_special_tokens=False)
        offs = e.offsets
        out = set()
        for (s0, e0), (s1, _) in zip(offs, offs[1:]):
            if s1 >= e0 and 0 < e0 - 1 < len(word):  # token ends and next starts at a char boundary
                out.add(e0 - 1)
        return out, len(e.ids)
    return cuts


def char_cuts(word: str) -> tuple[set, int]:
    return set(range(1, len(word))), len(word)


def load(extra: list[str]) -> dict:
    import tiktoken
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    toks = {name: tiktoken_cuts(tiktoken.get_encoding(enc)) for name, enc in TIKTOKEN.items()}
    for name, (repo, rev) in HF_TOKENIZERS.items():
        p = hf_hub_download(repo, "tokenizer.json", revision=rev, cache_dir=str(DATA / "hf"))
        toks[name] = hf_cuts(Tokenizer.from_file(p))
    for spec in extra:
        name, path = spec.split("=", 1)
        toks[name] = hf_cuts(Tokenizer.from_file(path))
    toks["Character-level reference"] = char_cuts
    return toks


def score(fn, words: dict) -> dict:
    tp = fp = fn_ = exact = ntok = 0
    for w, g in words.items():
        pred, n = fn(w)
        ntok += n
        tp += len(pred & g)
        fp += len(pred - g)
        fn_ += len(g - pred)
        exact += pred == g
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn_) if tp + fn_ else 0.0
    return {"words": len(words), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "exact_match": exact / len(words), "tokens_per_word": ntok / len(words)}


def main(args) -> None:
    gold, meta = load_gold()
    toks = load(args.tokenizer)
    rows = []
    for name, fn in toks.items():
        for lang, words in gold.items():
            rows.append({"tokenizer": name, "lang": lang, **score(fn, words)})
        print("done", name)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "morph_alignment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    (RESULTS / "morph_summary.json").write_text(json.dumps(
        {"gold": meta, "license": "CC-BY-4.0", "citation": "Brahma et al., arXiv:2504.10335"},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", action="append", default=[],
                    help="extra local tokenizer as NAME=path/to/tokenizer.json (repeatable)")
    main(ap.parse_args())
